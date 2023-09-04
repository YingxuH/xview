"""
with open("../results/unique_blocks_info.json", "r") as f:
    unique_blocks_info = json.load(f)

tree = HierarchyTree(hierarchy_path)


distance_freq = defaultdict(lambda: [])

for block_id in tqdm(unique_blocks_info):

    polygons = get_polygons(unique_blocks_info[block_id], image_size=256)
    X_coordinates = [poly['rectangle_coordinates'] for poly in polygons["polygons"]]
    types = [tree.find_significant_parent(poly['object']) for poly in polygons["polygons"]]

    types_indices = np.unique(types, return_inverse=True)[1]
    X = np.concatenate([X_coordinates, np.expand_dims(types_indices, axis=1)], axis=1)

    dist_matrix = csr_matrix(get_distance_matrix(X))

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


distance_thres_elbow = {key: elbow_cut_off(item) for key, item in distance_freq.items()}
distance_thres_percentile = {key: np.percentile(item, 75) for key, item in distance_freq.items()}

count = 0
for block_id in tqdm(unique_blocks_info):
    if 60 <= count <= 69:
        polygons = get_polygons(unique_blocks_info[block_id], image_size=256)
        polygons["polygons"] = [poly for poly in polygons["polygons"] if poly["object"] != "construction site"]
        X_coordinates = [poly['rectangle_coordinates'] for poly in polygons["polygons"]]
        original_types = [poly['object'] for poly in polygons["polygons"]]
        types = [tree.find_significant_parent(poly['object']) for poly in polygons["polygons"]]

        types_indices = np.unique(types, return_inverse=True)[1]
        X = np.concatenate([X_coordinates, np.expand_dims(types_indices, axis=1)], axis=1)

        dist_matrix = csr_matrix(get_distance_matrix(X))

        weights_matrix = minimum_spanning_tree(dist_matrix).toarray()
        edges = np.stack(np.triu_indices(n=weights_matrix.shape[0], k=1)).T
        weights = weights_matrix[np.triu_indices(n=weights_matrix.shape[0], k=1)]
        edges_types = np.array(types)[edges]
        upper_thres = np.array([distance_thres_percentile[t] for t in edges_types[:, 0]])

        valid_edges_condition = edges_types[:, 0] == edges_types[:, 1]
        none_empty_weights_condition = weights > 0
        valid_weights_condition = weights < upper_thres

        valid_indices = np.where(valid_weights_condition & none_empty_weights_condition & valid_weights_condition)[0]
        valid_edges = edges[valid_indices]
        valid_weights = weights[valid_indices]
        valid_edges_types = edges_types[valid_indices]

        clusters = get_clusters(X.shape[0], valid_edges)

        invalid_indices = np.where(none_empty_weights_condition & ~(valid_weights_condition & valid_edges_condition))[0]
        invalid_edges = edges[invalid_indices]
        connect_array = np.concatenate([clusters[invalid_edges], np.flip(clusters[invalid_edges], axis=1)])

        for i, cid in enumerate(np.unique(clusters)):
            c_types = [original_types[i] for i, t in enumerate(original_types) if clusters[i] == cid]
            objects_common_parents = tree.find_common_parent(np.unique(c_types))
            if detect_line_shape(X[clusters == cid], ae_thres=0.7):
                print(f"group {cid} contains a line of {objects_common_parents}")
            else:
                print(f"group {cid} contains some {objects_common_parents}")

        describe_relations(connect_array, X, clusters)

        random_test_sub_blocks(polygons, block_id, clusters, train_blocks_path)

    count += 1

def describe_relations(connectivity, X, clusters):
    if connectivity.shape[0] == 0:
        return []

    cluster_ids, counts = np.unique(clusters, return_counts=True)

    queue = [cluster_ids[np.argmax(counts)]]
    visited_edges = set()
    relations = []

    while queue:
        source = queue.pop(0)
        neighbours = connectivity[connectivity[:, 0] == source, 1].tolist()
        valid_neighbours = []

        for neigh in neighbours:
            current_edge = tuple(sorted([source, neigh]))
            if current_edge not in visited_edges:
                queue.append(neigh)
                valid_neighbours.append(neigh)
                visited_edges.add(current_edge)
        relations.append([source, valid_neighbours])

    for source, targets in relations:
        for target in targets:
            is_inside, is_outside, is_mixture, orientation = detect_orientation(X[clusters == source, :-1], X[clusters == target, :-1])
            if is_inside or is_mixture:
                print(f"group {source} is between group {target}")
            elif is_outside:
                print(f"group {source} is on the {orientation} side of group {target}")

    return relations


"""