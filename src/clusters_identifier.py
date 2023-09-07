import os
from tqdm.auto import tqdm
from pathlib import Path
from typing import Dict
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from src.hierarchy_tree import HierarchyTree
from src.utils import (
    get_polygons,
    get_distance_matrix,
    get_clusters_ids,
    detect_line_shape,
    describe_relations
)

_file_path = Path(__file__)
_hierarchy_path = os.path.join(_file_path, "hierarchy.json")


class ClustersIdentifier:
    def __init__(self, unique_blocks_info):
        self.tree = HierarchyTree(_hierarchy_path)
        self.unique_blocks_info = unique_blocks_info
        self.distance_threshold_percentile = None

    def _minimum_spanning_tree(self, polygons: Dict):
        types = [self.tree.find_significant_parent(poly['object']) for poly in polygons["polygons"]]
        coordinates = [poly['rectangle_coordinates'] for poly in polygons["polygons"]]

        types_encodings = np.unique(types, return_inverse=True)[1]
        combined_encodings = np.concatenate([coordinates, np.expand_dims(types_encodings, axis=1)], axis=1)

        dist_matrix = csr_matrix(get_distance_matrix(combined_encodings))
        weights_matrix = minimum_spanning_tree(dist_matrix).toarray()

        edges = np.stack(np.triu_indices(n=weights_matrix.shape[0], k=1)).T
        weights = weights_matrix[np.triu_indices(n=weights_matrix.shape[0], k=1)]
        edges_types = np.array(types)[edges]
        return combined_encodings, edges, weights, edges_types

    def train(self):
        distance_freq = defaultdict(lambda: [])

        for block_id in tqdm(self.unique_blocks_info):

            polygons = get_polygons(self.unique_blocks_info[block_id], image_size=256)
            _, _, weights, edges_types = self._minimum_spanning_tree(polygons)

            valid_edges_condition = edges_types[:, 0] == edges_types[:, 1]
            valid_weights_condition = weights > 0

            valid_indices = np.where(valid_weights_condition & valid_edges_condition)[0]
            valid_weights = weights[valid_indices]
            valid_edges_types = edges_types[valid_indices]

            for i, (start_type, _) in enumerate(valid_edges_types):
                distance_freq[start_type].append(valid_weights[i])

        self.distance_threshold_percentile = {key: np.percentile(item, 75) for key, item in distance_freq.items()}

    def infer(self, polygons: Dict):
        # polygons = get_polygons(self.unique_blocks_info[block_id], image_size=256)
        polygons["polygons"] = [poly for poly in polygons["polygons"] if poly["object"] != "construction site"]
        original_types = [poly['object'] for poly in polygons["polygons"]]

        x, edges, weights, edges_types = self._minimum_spanning_tree(polygons)

        upper_threshold = np.array([self.distance_threshold_percentile[t] for t in edges_types[:, 0]])

        valid_edges_condition = edges_types[:, 0] == edges_types[:, 1]
        none_empty_weights_condition = weights > 0
        valid_weights_condition = weights < upper_threshold

        valid_indices = np.where(valid_weights_condition & none_empty_weights_condition & valid_weights_condition)[0]
        valid_edges = edges[valid_indices]
        clusters = get_clusters_ids(x.shape[0], valid_edges)

        invalid_indices = np.where(none_empty_weights_condition & ~(valid_weights_condition & valid_edges_condition))[0]
        invalid_edges = edges[invalid_indices]
        clusters_connectivity = np.concatenate([clusters[invalid_edges], np.flip(clusters[invalid_edges], axis=1)])

        for i, cid in enumerate(np.unique(clusters)):
            c_types = [original_types[i] for i, t in enumerate(original_types) if clusters[i] == cid]
            objects_common_parents = self.tree.find_common_parent(np.unique(c_types))
            if detect_line_shape(x[clusters == cid], ae_threshold=0.7):
                print(f"group {cid} contains a line of {objects_common_parents}")
            else:
                print(f"group {cid} contains some {objects_common_parents}")

        describe_relations(clusters, clusters_connectivity, x)


