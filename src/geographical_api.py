import os
import json
from datetime import date
from tqdm.auto import tqdm
from typing import List, Dict
from pathlib import Path
from collections import defaultdict

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


def _decode_polygons(polygons, tree):
    original_types = [poly['object'] for poly in polygons["polygons"]]
    types = [tree.find_significant_parent(poly['object']) for poly in polygons["polygons"]]
    coordinates = [poly['rectangle_coordinates'] for poly in polygons["polygons"]]

    types_encodings = np.unique(types, return_inverse=True)[1]
    combined_encodings = np.concatenate([coordinates, np.expand_dims(types_encodings, axis=1)], axis=1)

    return original_types, types, combined_encodings


def _minimum_spanning_tree(types, encodings, distance_func=box_distance_with_type):
    dist_matrix = csr_matrix(get_distance_matrix(encodings, func=distance_func))
    weights_matrix = minimum_spanning_tree(dist_matrix).toarray()

    edges = np.stack(np.triu_indices(n=weights_matrix.shape[0], k=1)).T
    weights = weights_matrix[np.triu_indices(n=weights_matrix.shape[0], k=1)]
    edges_types = np.array(types)[edges]
    return edges, weights, edges_types


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

        self.cluster_distance_threshold_percentile = None
        self.normal_distance_lower_percentile = None
        self.normal_distance_upper_percentile = None
        self.train()

    def train(self):
        typed_distances_freq = defaultdict(lambda: [])
        normal_distances = []

        for block_id in tqdm(self.decoded_blocks_info, desc="Training"):
            types = self.decoded_blocks_info[block_id]["types"]
            encodings = self.decoded_blocks_info[block_id]["encodings"]
            _, weights, edges_types = _minimum_spanning_tree(types, encodings)

            valid_edges_condition = edges_types[:, 0] == edges_types[:, 1]
            valid_weights_condition = weights > 0

            valid_indices = np.where(valid_weights_condition & valid_edges_condition)[0]
            valid_weights = weights[valid_indices]
            valid_edges_types = edges_types[valid_indices]

            for i, (start_type, _) in enumerate(valid_edges_types):
                typed_distances_freq[start_type].append(valid_weights[i])

            _, normal_weights, _ = _minimum_spanning_tree(types, encodings, distance_func=box_distance)
            valid_indices = np.where(normal_weights > 0)[0]
            valid_normal_weights = normal_weights[valid_indices]
            normal_distances.append(valid_normal_weights)

        all_distances_array = np.concatenate(normal_distances)
        self.cluster_distance_threshold_percentile = {
            key: np.percentile(item, 75) for key, item in typed_distances_freq.items()
        }
        self.normal_distance_lower_percentile = np.percentile(all_distances_array, 25)
        self.normal_distance_upper_percentile = np.percentile(all_distances_array, 75)

    def infer(self, types, encodings):
        # polygons = get_polygons(self.unique_blocks_info[block_id], image_size=256)
        # polygons["polygons"] = [poly for poly in polygons["polygons"] if poly["object"] != "construction site"]
        # original_types = [poly['object'] for poly in polygons["polygons"]]

        edges, weights, edges_types = _minimum_spanning_tree(types, encodings)

        upper_threshold = np.array([self.cluster_distance_threshold_percentile[t] for t in edges_types[:, 0]])

        valid_edges_condition = edges_types[:, 0] == edges_types[:, 1]
        none_empty_weights_condition = weights > 0
        valid_weights_condition = weights < upper_threshold

        valid_indices = np.where(valid_weights_condition & none_empty_weights_condition & valid_weights_condition)[0]
        valid_edges = edges[valid_indices]
        clusters = get_clusters_ids(encodings.shape[0], valid_edges)

        invalid_indices = np.where(none_empty_weights_condition & ~(valid_weights_condition & valid_edges_condition))[0]
        invalid_edges = edges[invalid_indices]
        clusters_connectivity = np.concatenate([clusters[invalid_edges], np.flip(clusters[invalid_edges], axis=1)])

        return clusters, clusters_connectivity

    def get_api(self, block_id):
        original_types = self.decoded_blocks_info[block_id]["original_types"]
        types = self.decoded_blocks_info[block_id]["types"]
        encodings = self.decoded_blocks_info[block_id]["encodings"]

        tree = HierarchyTree(_hierarchy_path)
        clusters, connectivity = self.infer(types, encodings)

        return GeographicalAPI(
            original_types,
            encodings,
            tree,
            clusters,
            connectivity,
            self.normal_distance_lower_percentile,
            self.normal_distance_upper_percentile
        )


class GeographicalAPI:
    def __init__(self, original_types, encodings, tree, clusters, connectivity, lower_threshold, upper_threshold):
        self.original_types = original_types
        self.encodings = encodings
        self.tree = tree
        self.clusters = clusters
        self.connectivity = connectivity
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

    def _describe_objects(self, encodings, types):
        objects_common_parents = self.tree.find_common_parent(np.unique(types))
        shape_prefix = "some"
        if encodings.shape[0] == 1:
            shape_prefix = "a"
        if detect_line_shape(encodings, ae_threshold=0.7):
            shape_prefix = "a line of"

        return f"{shape_prefix} {objects_common_parents}"

    def fill_prompt(self):
        if self.connectivity.shape[0] == 0:
            return []

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

        objects_relations = []
        visited = set()
        for source, targets in relations:
            for target in targets:
                source_encodings = self.encodings[self.clusters == source]
                target_encodings = self.encodings[self.clusters == target]

                source_types = np.array(self.original_types)[self.clusters == source]
                target_types = np.array(self.original_types)[self.clusters == target]

                source_description = self._describe_objects(source_encodings, source_types)
                target_description = self._describe_objects(target_encodings, target_types)

                is_pos_inside, is_pos_outside, is_pos_mixture = detect_orientation(
                    source_encodings[:, :-1],
                    target_encodings[:, :-1]
                )

                is_neg_inside, is_neg_outside, is_neg_mixture = detect_orientation(
                    target_encodings[:, :-1],
                    source_encodings[:, :-1]
                )

                if is_pos_inside:
                    objects_relations.append(f"{source_description} is surrounded by {target_description}")
                    visited.add([source, target])
                    continue

                if is_neg_inside:
                    objects_relations.append(f"{target_description} is surrounded by {source_description}")
                    visited.add([source, target])
                    continue

                if is_pos_mixture or is_neg_mixture:
                    objects_relations.append(f"{source_description} is adjacent to group {target_description}")
                    visited.add([source, target])
                    continue

                distances = box_distance_array(source_encodings, target_encodings)

                if distances.min() < self.lower_threshold:
                    objects_relations.append(f"{source_description} is close to {target_description}")
                    visited.add([source, target])
                    continue

                if distances.min() > self.upper_threshold:
                    objects_relations.append(f"{source_description} is far from other objects")
                    visited.update([source])
                    continue

            objects_existences = []
            for cid in cluster_ids:
                if cid not in visited:
                    current_encodings = self.encodings[self.clusters == cid]
                    current_types = np.array(self.original_types)[self.clusters == cid]
                    current_description = self._describe_objects(current_encodings, current_types)

                    objects_existences.append(f"There is/are {current_description} in the image.")

            return objects_relations, objects_existences

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
    print(api.fill_prompt())
    random_test_sub_blocks(polygons, block_id, api.clusters, r"D:\xview\train_blocks\train_blocks")
