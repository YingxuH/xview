import os
from datetime import date
from tqdm.auto import tqdm
from typing import List, Dict
from pathlib import Path

import numpy as np

from src.hierarchy_tree import HierarchyTree
from src.clusters_identifier import ClustersIdentifier
from src.utils import get_polygons, detect_line_shape, detect_orientation, describe_relations
from src.prompts import (
    initial_prompt,
    identify_types_response_template,
    identify_shape_response_template,
    identify_relations_response_template
)


_file_path = Path(__file__)
_hierarchy_path = os.path.join(_file_path.parents[0], "hierarchy.json")


def decode_polygons(polygons, tree):
    original_types = [poly['object'] for poly in polygons["polygons"]]
    types = [tree.find_significant_parent(poly['object']) for poly in polygons["polygons"]]
    coordinates = [poly['rectangle_coordinates'] for poly in polygons["polygons"]]

    types_encodings = np.unique(types, return_inverse=True)[1]
    combined_encodings = np.concatenate([coordinates, np.expand_dims(types_encodings, axis=1)], axis=1)

    return original_types, types, combined_encodings


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

            original_types, types, encodings = decode_polygons(polygons, self.tree)
            self.decoded_blocks_info[block_id] = {
                "original_types": original_types,
                "types": types,
                "encodings": encodings
            }

        self.cluster_identifier = ClustersIdentifier(self.decoded_blocks_info)
        self.cluster_identifier.train()

    def get_api(self, block_id):
        return GeographicalAPI(self.decoded_blocks_info[block_id], self.cluster_identifier)


class GeographicalAPI:
    def __init__(self, block_info, cluster_identifier):
        self.original_types = block_info["original_types"]
        self.types = block_info["types"]
        self.encodings = block_info["encodings"]

        self.tree = HierarchyTree(_hierarchy_path)
        self.cluster_identifier = cluster_identifier
        self.clusters, self.connectivity = self.cluster_identifier.infer(self.types, self.encodings)

    def fill_prompt(self):
        # TODO: identify close to, in reasonable distance to, or far away from
        # TODO: split the describe_relations function into function identifying the closest neighbour.
        # TODO: For the spatial relation part, only report surrounded by, next to something/far from everyone.
        # TODO: only ask LLM to summarize. detect group name, increase temperature. 
        clusters_ids = np.unique(self.clusters)
        cluster_names = [f"cluster {cluster_id}" for cluster_id in clusters_ids]

        for i, cid in enumerate(np.unique(self.clusters)):
            c_types = [t for i, t in enumerate(self.original_types) if self.clusters[i] == cid]
            objects_common_parents = self.tree.find_common_parent(np.unique(c_types))
            if len(c_types) == 1:
                print(f"A {objects_common_parents} stands alone.")
                continue

            if detect_line_shape(self.encodings[self.clusters == cid], ae_threshold=0.7):
                print(f"group {cid} contains a line of {objects_common_parents}")
            else:
                print(f"group {cid} contains some {objects_common_parents}")
        print(describe_relations(self.clusters, self.connectivity, self.encodings))
        # return initial_prompt.format(
        #     today_date = date.today(),
        #     num_objects=self.encodings.shape[0],
        #     num_clusters=len(clusters_ids),
        #     cluster_names=", ".join(cluster_names)
        # ) +

    def identify_types_of_objects(self, clusterId: int) -> str:
        """ provide the objects' type for the specified cluster. """
        c_types = [t for i, t in enumerate(self.original_types) if self.clusters[i] == clusterId]
        objects_common_parents = self.tree.find_common_parent(np.unique(c_types))
        return identify_types_response_template.format(
            cluster_id=clusterId,
            num_objects=len(c_types),
            objects_type=objects_common_parents
        )

    def identify_shape_of_objects(self, cid: int) -> str:
        """ provide the visual shape of the cluster of objects in the image. """
        c_types = [t for i, t in enumerate(self.original_types) if self.clusters[i] == cid]
        objects_common_parents = self.tree.find_common_parent(np.unique(c_types))

        shape_description = "are scattered around"
        if detect_line_shape(self.encodings[self.clusters == cid], ae_threshold=0.7):
            shape_description = "form a line shape"

        return identify_shape_response_template.format(
            objects_type=objects_common_parents,
            cluster_id=cid,
            shape_description=shape_description
        )

    def identify_relations_of_clusters(self, clusterIdA: int, clusterIdB: int) -> str:
        """ provide the geographical relations between two clusters of objects in the image. """
        is_inside, is_outside, is_mixture, _ = detect_orientation(
            self.encodings[self.clusters == clusterIdA, :-1],
            self.encodings[self.clusters == clusterIdB, :-1]
        )

        relation_description = ""
        if is_inside:
            relation_description = "is surrounded by"
        elif is_outside:
            relation_description = "is in close distance to"
        elif is_mixture:
            relation_description = "is adjacent to"

        return identify_relations_response_template.format(
            cluster_id_a=clusterIdA,
            cluster_id_b=clusterIdB,
            relation_description=relation_description
        )
