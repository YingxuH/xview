import os
import json
from datetime import date
from tqdm.auto import tqdm
from typing import List, Dict, Union
from pathlib import Path
from collections import defaultdict
from functools import partial

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from src.hierarchy_tree import HierarchyTree
from src.utils import (
    get_polygons,
    box_distance,
    box_distance_with_type,
    box_distance_array,
    get_distance_matrix,
    get_clusters_ids,
    detect_line_shape,
    detect_orientation,
    describe_relations,
    random_test_sub_blocks
)
from src.prompts import (
    initial_prompt,
    identify_types_response_template,
    identify_shape_response_template,
    identify_relations_response_template
)

_file_path = Path(__file__)
_hierarchy_path = os.path.join(_file_path.parents[0], "hierarchy.json")

_system_message_template = "today is 13-09-2023, you are a helpful assistant who is proficient at understanding satellite images."

_prompt_template = """
You will be provided with description of a satellite image. Please provide several captions as strings in a python list.
The image has multiple objects, which have been clustered into groups based on their types and locations. 
Here are some examples:

# example image description 1:
## objects/object groups information
group 0: 1 damaged building
group 1: a line of 5 building
group 2: a line of 3 building

## significant geographical relations
group 2 is next to group 0

## captions
["There are two lines of buildings in the image.", "A damaged building is next to a line of buildings in the image"]

# example image description 2:
## objects/object groups information
group 0: 1 building
group 1: 1 building
group 2: 1 truck
group 3: 2 building, including 1 building, 1 shed
group 4: 2 building, including 1 building, 1 damaged building

## significant geographical relations
group 3 is next to group 1

## captions
["There are several buildings and one truck in the image.", "A building and one shed sit side by side", "There is a building and a damaged building standing together.", "Some buildings are very close to each other while some others are not"]

# example image description 3:
## objects/object groups information
group 0: 1 building
group 1: a line of 3 building

## significant geographical relations
None

## captions
["There is a solitary building in the image.", "There is a line of three buildings in the image."]

# real image description:
## objects/object groups information
{objects_information}

## significant geographical relations
{objects_relations}

## captions
"""

_short_prompt_template = """
## objects/object groups information
{objects_information}

## significant geographical relations
{objects_relations}
"""


def _decode_polygons(polygons, tree):
    original_types = [poly['object'] for poly in polygons["polygons"]]
    types = [tree.find_significant_parent(poly['object']) for poly in polygons["polygons"]]
    coordinates = [poly['rectangle_coordinates'] for poly in polygons["polygons"]]

    types_encodings = np.unique(types, return_inverse=True)[1]
    combined_encodings = np.concatenate([coordinates, np.expand_dims(types_encodings, axis=1)], axis=1)

    return original_types, types, combined_encodings


def _minimum_spanning_tree(encodings: Union[List, np.array], distance_func=box_distance_with_type):
    """
    :param types:
    :param encodings: numpy array: nx2
    :param distance_func:
    :return:
    """
    dist_matrix = csr_matrix(get_distance_matrix(encodings, func=distance_func))
    weights_matrix = minimum_spanning_tree(dist_matrix).toarray()

    edges = np.stack(np.triu_indices(n=weights_matrix.shape[0], k=1)).T
    weights = weights_matrix[np.triu_indices(n=weights_matrix.shape[0], k=1)]
    return edges, weights


class GeographicalAPIManager:
    def __init__(self, blocks_info):
        self.blocks_info = blocks_info

        self.tree = HierarchyTree(_hierarchy_path)
        self.decoded_blocks_info = dict()
        for block_id in tqdm(self.blocks_info, desc="Decoding json"):
            polygons = get_polygons(self.blocks_info[block_id], image_size=256)
            polygons["polygons"] = [poly for poly in polygons["polygons"] if poly["object"] != "construction site"]
            if not polygons["polygons"]:
                continue

            original_types, types, encodings = _decode_polygons(polygons, self.tree)
            self.decoded_blocks_info[block_id] = {
                "original_types": original_types,
                "types": types,
                "encodings": encodings
            }

        self.hetero_multiplier = 1000
        self.cluster_distance_threshold_percentile = None
        self.hetero_distances = None
        self.normal_distance_lower_percentile = None
        self.normal_distance_upper_percentile = None
        self.train()

    def train(self):
        homo_distances_freq = defaultdict(lambda: [])
        hetero_distances = []

        for block_id in tqdm(self.decoded_blocks_info, desc="Training"):
            types = self.decoded_blocks_info[block_id]["types"]
            encodings = self.decoded_blocks_info[block_id]["encodings"]

            box_distance_custom = partial(box_distance_with_type, multiplier=self.hetero_multiplier)
            edges, weights= _minimum_spanning_tree(encodings, distance_func=box_distance_custom)
            edges_types = np.array(types)[edges]

            homo_edge_conditions = (weights > 0) & (edges_types[:, 0] == edges_types[:, 1])
            hetero_edge_conditions = (weights > 0) & (edges_types[:, 0] != edges_types[:, 1])

            homo_edge_indices = np.where(homo_edge_conditions)[0]
            homo_edge_weights = weights[homo_edge_indices]
            homo_edge_types = edges_types[homo_edge_indices]

            hetero_edge_indices = np.where(hetero_edge_conditions)[0]
            hetero_edge_weights = weights[hetero_edge_indices]

            for i, (start_type, _) in enumerate(homo_edge_types):
                homo_distances_freq[start_type].append(homo_edge_weights[i])

            hetero_distances.append(hetero_edge_weights)

        hetero_distances_array = np.sqrt(
            np.square(np.concatenate(hetero_distances)) - np.square(self.hetero_multiplier))
        self.cluster_distance_threshold_percentile = {
            key: np.percentile(item, 75) for key, item in homo_distances_freq.items()
        }

        self.hetero_distances = hetero_distances_array
        self.normal_distance_lower_percentile = np.percentile(hetero_distances_array, 75)
        self.normal_distance_upper_percentile = np.percentile(hetero_distances_array, 90)

    def infer(self, types, encodings):
        # polygons = get_polygons(self.unique_blocks_info[block_id], image_size=256)
        # polygons["polygons"] = [poly for poly in polygons["polygons"] if poly["object"] != "construction site"]
        # original_types = [poly['object'] for poly in polygons["polygons"]]

        edges, weights = _minimum_spanning_tree(encodings)
        edges_types = np.array(types)[edges]

        upper_threshold = np.array([self.cluster_distance_threshold_percentile[t] for t in edges_types[:, 0]])

        valid_edges_condition = edges_types[:, 0] == edges_types[:, 1]
        none_empty_weights_condition = weights > 0
        valid_weights_condition = weights < upper_threshold

        valid_indices = np.where(valid_weights_condition & none_empty_weights_condition & valid_weights_condition)[0]
        valid_edges = edges[valid_indices]
        clusters = get_clusters_ids(encodings.shape[0], valid_edges)

        # TODO: check this
        cluster_lst = []
        for cid in np.unique(clusters):
            cluster_lst.append(encodings[clusters == cid])

        cluster_edges, cluster_weights = _minimum_spanning_tree(cluster_lst, distance_func=box_distance_array)
        none_empty_weights_condition = cluster_weights > 0
        valid_cluster_edges = cluster_edges[none_empty_weights_condition]
        clusters_connectivity = np.concatenate([valid_cluster_edges, np.flip(valid_cluster_edges, axis=1)])

        return clusters, clusters_connectivity

    def get_api(self, block_id):
        block_info = self.decoded_blocks_info[block_id]

        tree = HierarchyTree(_hierarchy_path)
        clusters, connectivity = self.infer(block_info["types"], block_info["encodings"])

        return GeographicalAPI(
            block_info,
            tree,
            clusters,
            connectivity,
            self.normal_distance_lower_percentile,
            self.normal_distance_upper_percentile
        )


class GeographicalAPI:
    def __init__(self, block_info, tree, clusters, connectivity, lower_threshold, upper_threshold):
        self.original_types = block_info["original_types"]
        self.encodings = block_info["encodings"]
        self.tree = tree
        self.clusters = clusters
        self.connectivity = connectivity
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

    def _get_common_object(self, cluster_id):
        types = np.array(self.original_types)[self.clusters == cluster_id]
        return self.tree.find_common_parent(np.unique(types))

    def _get_detailed_objects_of_group(self, cluster_id):
        types = np.array(self.original_types)[self.clusters == cluster_id]
        return [f"{cnt} {t}" for t, cnt in zip(*np.unique(types, return_counts=True))]

    def _describe_objects(self, cluster_id):
        encodings = self.encodings[self.clusters == cluster_id]
        objects_common_parent = self._get_common_object(cluster_id)

        shape_prefix = encodings.shape[0]
        if detect_line_shape(encodings, ae_threshold=0.7):
            shape_prefix = f"a line of {shape_prefix}"
        return f"{shape_prefix} {objects_common_parent}"

    def _get_relations(self):
        cluster_ids, counts = np.unique(self.clusters, return_counts=True)

        queue = [cluster_ids[np.argmax(counts)]]
        visited_edges = set()
        relations = []

        while queue:
            source = queue.pop(0)
            neighbours = self.connectivity[self.connectivity[:, 0] == source, 1].tolist()
            valid_neighbours = []

            for neigh in neighbours:
                current_edge = tuple(sorted([source, neigh]))
                if current_edge not in visited_edges:
                    queue.append(neigh)
                    valid_neighbours.append(neigh)
                    visited_edges.add(current_edge)

            if valid_neighbours:
                relations.append([source, valid_neighbours])
        return relations

    def get_list_attributes(self):
        cluster_ids, counts = np.unique(self.clusters, return_counts=True)
        group_key = "group {cluster_id}"

        objects_information = []
        for cid in cluster_ids:
            current_key = group_key.format(cluster_id=cid)
            current_desc = self._describe_objects(cid)
            current_details = self._get_detailed_objects_of_group(cid)
            info = f"{current_key}: {current_desc}"
            if len(current_details) > 1:
                info = f"{info}, including {', '.join(current_details)}"
            objects_information.append(info)

        # relations = self._get_relations()

        objects_relations = []
        visited_edges = set()
        # print(relations)
        for source, target in self.connectivity:
            edge_key = tuple(sorted([source, target]))
            if edge_key in visited_edges:
                continue
            source_encodings = self.encodings[self.clusters == source]
            target_encodings = self.encodings[self.clusters == target]

            source_key = group_key.format(cluster_id=source)
            target_key = group_key.format(cluster_id=target)

            is_pos_inside, is_pos_outside, is_pos_mixture = detect_orientation(
                source_encodings[:, :-1],
                target_encodings[:, :-1]
            )

            is_neg_inside, is_neg_outside, is_neg_mixture = detect_orientation(
                target_encodings[:, :-1],
                source_encodings[:, :-1]
            )
            if is_pos_inside:
                objects_relations.append(
                    f"{source_key} is surrounded by {target_key}"
                )
                visited_edges.add(edge_key)
                continue

            if is_neg_inside:
                objects_relations.append(
                    f"{target_key} is surrounded by {source_key}"
                )
                visited_edges.add(edge_key)
                continue

            if is_pos_mixture or is_neg_mixture:
                objects_relations.append(
                    f"{source_key} is adjacent to group {target_key}"
                )
                visited_edges.add(edge_key)
                continue

            distances = box_distance_array(source_encodings, target_encodings)
            source_edge_count = self.connectivity[self.connectivity[:, 0] == source].shape[0]

            if distances.min() < self.lower_threshold:
                objects_relations.append(
                    f"{source_key} is close to {target_key}"
                )
                visited_edges.add(edge_key)
                continue

            # TODO: replace to self.connectivity[self.connectivity[0]==source].shape[0] == 1
            if distances.min() > self.upper_threshold and source_edge_count == 1:
                objects_relations.append(
                    f"{source_key} is far from other objects"
                )
                visited_edges.add(edge_key)
                continue

        return objects_information, objects_relations, visited_edges

    def get_image_description(self):
        objects_information, objects_relations, _ = self.get_list_attributes()
        information_str = "\n".join(objects_information) if objects_information else "None"
        relations_str = "\n".join(objects_relations) if objects_relations else "None"

        current_prompt = _prompt_template.format(
            objects_information=information_str,
            objects_relations=relations_str
        )

        current_short_prompt = _short_prompt_template.format(
            objects_information=information_str,
            objects_relations=relations_str
        )
        return _system_message_template, current_prompt, current_short_prompt

    def identify_types_of_objects(self, cluster_id: int) -> str:
        """ provide the objects' type for the specified cluster. """
        c_types = [t for i, t in enumerate(self.original_types) if self.clusters[i] == cluster_id]
        objects_common_parents = self.tree.find_common_parent(np.unique(c_types))
        return identify_types_response_template.format(
            cluster_id=cluster_id,
            num_objects=len(c_types),
            objects_type=objects_common_parents
        )

    def identify_shape_of_objects(self, cid: int) -> str:
        """ provide the visual shape of the cluster of objects in the image. """
        c_types = [t for i, t in enumerate(self.original_types) if self.clusters[i] == cid]
        objects_common_parents = self.tree.find_common_parent(np.unique(c_types))

        shape_description = "are scattered around"
        if len(c_types) == 1:
            shape_description = "stands alone"
        elif detect_line_shape(self.encodings[self.clusters == cid], ae_threshold=0.7):
            shape_description = "form a line shape"

        return identify_shape_response_template.format(
            objects_type=objects_common_parents,
            cluster_id=cid,
            shape_description=shape_description
        )

    def identify_relations_of_clusters(self, cluster_id_a: int, cluster_id_b: int) -> str:
        """ provide the geographical relations between two clusters of objects in the image. """
        is_inside, is_outside, is_mixture, _ = detect_orientation(
            self.encodings[self.clusters == cluster_id_a, :-1],
            self.encodings[self.clusters == cluster_id_b, :-1]
        )

        relation_description = ""
        if is_inside:
            relation_description = "is surrounded by"
        elif is_outside:
            relation_description = "is in close distance to"
        elif is_mixture:
            relation_description = "is adjacent to"

        return identify_relations_response_template.format(
            cluster_id_a=cluster_id_a,
            cluster_id_b=cluster_id_b,
            relation_description=relation_description
        )


if __name__ == "__main__":
    with open("../results/unique_blocks_info.json", "r") as f:
        unique_blocks_info = json.load(f)

    api_manager = GeographicalAPIManager(unique_blocks_info)
    block_id = "1044.tif_1"

    api = api_manager.get_api(block_id)

    polygons = get_polygons(api_manager.blocks_info[block_id], image_size=256)
    # print(api.get_default_descriptions())
    random_test_sub_blocks(polygons, block_id, api.clusters, r"D:\xview\train_blocks\train_blocks")
