import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict

from src.hierarchy_tree import HierarchyTree
from src.utils import *


class ClusterExtractor:
    def __init__(self, unique_blocks_info):
        self.tree = HierarchyTree(hierarchy_path)
        self.unique_blocks_info = unique_blocks_info
        self.distance_threshold_elbow = None

    def train(self):
        distance_freq = defaultdict(lambda: [])

        for block_id in tqdm(self.unique_blocks_info):

            polygons = get_polygons(self.unique_blocks_info[block_id], image_size=256)
            x_coordinates = [poly['rectangle_coordinates'] for poly in polygons["polygons"]]
            types = [self.tree.find_significant_parent(poly['object']) for poly in polygons["polygons"]]

            types_indices = np.unique(types, return_inverse=True)[1]
            x = np.concatenate([x_coordinates, np.expand_dims(types_indices, axis=1)], axis=1)

            dist_matrix = csr_matrix(get_distance_matrix(x))

            weights_matrix = minimum_spanning_tree(dist_matrix).toarray()
            edges = np.stack(np.triu_indices(n=weights_matrix.shape[0], k=1)).T
            weights = weights_matrix[np.triu_indices(n=weights_matrix.shape[0], k=1)]
            edges_types = np.array(types)[edges]

            valid_edges_condition = edges_types[:, 0] == edges_types[:, 1]
            valid_weights_condition = weights > 0

            valid_indices = np.where(valid_weights_condition & valid_edges_condition)[0]
            valid_edges = edges[valid_indices]
            valid_weights = weights[valid_indices]
            valid_edges_types = edges_types[valid_indices]

            for i, (start, _) in enumerate(valid_edges_types):
                distance_freq[start].append(valid_weights[i])

        self.distance_threshold_elbow = {key: elbow_cut_off(item) for key, item in distance_freq.items()}

    def infer(self, block_id):
        polygons = get_polygons(self.unique_blocks_info[block_id], image_size=256)
        polygons["polygons"] = [poly for poly in polygons["polygons"] if poly["object"] != "construction site"]
        x_coordinates = [poly['rectangle_coordinates'] for poly in polygons["polygons"]]
        original_types = [poly['object'] for poly in polygons["polygons"]]
        types = [self.tree.find_significant_parent(poly['object']) for poly in polygons["polygons"]]

        types_indices = np.unique(types, return_inverse=True)[1]
        x = np.concatenate([x_coordinates, np.expand_dims(types_indices, axis=1)], axis=1)

        dist_matrix = csr_matrix(get_distance_matrix(x))

        weights_matrix = minimum_spanning_tree(dist_matrix).toarray()
        edges = np.stack(np.triu_indices(n=weights_matrix.shape[0], k=1)).T
        weights = weights_matrix[np.triu_indices(n=weights_matrix.shape[0], k=1)]
        edges_types = np.array(types)[edges]
        upper_thres = np.array([distance_thres_percentile[t] for t in edges_types[:, 0]])
