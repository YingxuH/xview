import os
import re
import time
import random
from typing import Dict, List, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.linear_model import LinearRegression


cmap = plt.get_cmap('viridis')


class LLMValidationException(Exception):
    def __init__(self, message):
        super(LLMValidationException, self).__init__(message)


class MaxRetryException(Exception):
    def __init__(self, exceptions: List[Exception]):
        message = f"Maximal number of retries exceeded ({len(exceptions)}):\n"
        for i, exception in enumerate(exceptions):
            message += f" {i+1}. {exception.__class__.__name__}: {str(exception)}\n"

        super(MaxRetryException, self).__init__(message)


def retry_upon_exceptions(*exceptions, retry=3, wait_time=0):
    def decorator(function):
        def wrapper(*args, **kwargs):

            result = None
            current_retry = retry

            past_exceptions = []
            while current_retry > 0:
                if current_retry < retry:
                    time.sleep(wait_time)

                try:
                    result = function(*args, **kwargs)
                    break
                except exceptions as last_exception:
                    past_exceptions.append(last_exception)
                    current_retry -= 1

            if current_retry <= 0:
                raise MaxRetryException(past_exceptions)

            return result
        return wrapper
    return decorator


def random_test(blocks_info, image_path, key=None):
    if key is None:
        key = random.choice(list(blocks_info.keys()))
        while len(blocks_info[key]["boxes"]) == 0:
            key = random.choice(list(blocks_info.keys()))

    item = blocks_info[key]
    file = key.split("_")[0]

    image_file_path = os.path.join(image_path, file)

    image = plt.imread(image_file_path)
    lb_x, lb_y, ru_x, ru_y = [int(value) for value in item["chip_coords"].split(",")]

    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    rect = patches.Rectangle((lb_x, lb_y), ru_x - lb_x, ru_y - lb_y, linewidth=1, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    plt.gca().add_patch(rect)

    for box in item["boxes"]:
        coord = box['properties']['bounds_imcoords'].split(",")
        coord = [int(value) for value in coord]
        left_bottom = coord[:2]
        height, width = coord[2] - coord[0], coord[3] - coord[1]
        rect = patches.Rectangle(left_bottom, height, width, linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        plt.gca().add_patch(rect)

    plt.xlim(lb_x, ru_x)
    plt.ylim(ru_y, lb_y)

    plt.show()
    plt.close()

    return key


def random_test_sub_blocks(polygons, key, labels, texts, image_path, save_path):
    unique_labels, unique_indices = np.unique(labels, return_index=True)
    colors = cmap(np.linspace(0, 1, unique_labels.shape[0]))
    colors_map = dict(zip(unique_labels, colors))

    image_file_path = os.path.join(image_path, key + ".png")

    image = plt.imread(image_file_path)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    rectangles = []
    for i, box in enumerate(polygons["polygons"]):
        color_encoding = colors_map[labels[i]]

        coord = box['rectangle_coordinates']

        left_bottom = coord[:2]
        height, width = coord[2] - coord[0], coord[3] - coord[1]
        rect = patches.Rectangle(
            left_bottom, height, width,
            linewidth=1,
            edgecolor=color_encoding,
            facecolor='none'
        )

        # Add the patch to the Axes
        plt.gca().add_patch(rect)
        rectangles.append(rect)

        rx, ry = rect.get_xy()
        cx = rx + rect.get_width() / 2.0
        cy = ry + rect.get_height() / 2.0

        plt.annotate(texts[i], (cx, cy), color="w", weight="bold", fontsize=8, ha='center', va='center')

    unique_rects = [rectangles[i] for i in unique_indices]
    plt.legend(unique_rects, unique_labels)
    file_name = f"{key}.png"
    plt.savefig(os.path.join(save_path, file_name))
    # plt.show()
    plt.close()

    return key


def coord_to_location(coords, image_size):
    if coords[0] <= (image_size * 0.4):
        hori = "left"
    elif coords[0] >= (image_size * 0.6):
        hori = "right"
    else:
        hori = "middle"

    if coords[1] <= (image_size * 0.4):
        vert = "top"
    elif coords[1] >= (image_size * 0.6):
        vert = "bottom"
    else:
        vert = "middle"

    if (hori == "middle") and (vert == "middle"):
        position = "center"
    elif hori == "middle":
        position = f"center towards the {vert}"
    else:
        position = f"{vert} {hori}"

    return position


def polygons_to_nl(polygons):
    sentence = "There is a 256 by 256 aerial image."
    for poly in polygons:
        sentence += f" One {poly['object']} is located at x = {poly['x']} and y = {poly['y']}, in the {poly['position']} of the image, with a size = {poly['size']} pixels."
    return sentence


def get_polygons(block, image_size=256) -> Dict:
    lb_x, lb_y, ru_x, ru_y = [int(value) for value in block["chip_coords"].split(",")]
    lb_coords = np.array([lb_x, lb_y])

    lat_long_middles = []
    rectangle_middles = []
    for box in block["boxes"]:
        lat_long_coords = box["geometry"]["coordinates"][0]
        lat_long_middle = np.unique(np.array(lat_long_coords), axis=0).mean(axis=0)

        rectangle_box_coords = box["properties"]["bounds_imcoords"].split(",")
        rectangle_box_coords = [int(value) for value in rectangle_box_coords]
        pixel_middle = np.array(
            [
                (rectangle_box_coords[0] + rectangle_box_coords[2])/2,
                (rectangle_box_coords[1] + rectangle_box_coords[3])/2
            ]
        )

        lat_long_middles.append(lat_long_middle)
        rectangle_middles.append(pixel_middle)

    lat_long_middles = np.stack(lat_long_middles, axis=0)
    pixel_middles = np.stack(rectangle_middles, axis=0)

    lat_reg = LinearRegression()
    lat_reg.fit(lat_long_middles[:, [0]], pixel_middles[:, 0])

    long_reg = LinearRegression()
    long_reg.fit(lat_long_middles[:, [1]], pixel_middles[:, 1])

    polygons = []
    for box in block["boxes"]:
        lat_long_coords = np.array(box["geometry"]["coordinates"][0])
        x_coords = np.round(lat_reg.predict(lat_long_coords[:, [0]]), 0).astype(int)
        y_coords = np.round(long_reg.predict(lat_long_coords[:, [1]]), 0).astype(int)
        pixel_coords = np.unique(np.stack([x_coords, y_coords], 1), axis=0) - lb_coords
        pixel_middle = pixel_coords.mean(axis=0).tolist()
        width, height = np.ptp(pixel_coords, axis=0)
        position = coord_to_location(pixel_middle, image_size)

        obj = re.findall('[A-Z][^A-Z]*', box["properties"]["type_id"])
        obj = " ".join(obj).lower()
        polygons.append({
            "object": obj,
            "bounding_box_coordinates": pixel_coords.tolist(),
            "rectangle_coordinates": [*pixel_coords.tolist()[0], *pixel_coords.tolist()[-1]],
            "position": position,
            "x": str(int(pixel_middle[0])),
            "y": str(int(pixel_middle[1])),
            "size": str(height * width),
        })

    polygons = sorted(polygons, key=lambda x: x["object"])
    sentence = polygons_to_nl(polygons)

    return {"polygons": polygons, "sentence": sentence}


def box_distance(array_a, array_b):
    """
    calcualte the distance between any two boxes. Do not consider the type difference
    :param array_a:
    :param array_b:
    :return:
    """
    if (not isinstance(array_a, np.ndarray)) and (not isinstance(array_b, np.ndarray)):
        array_a = np.array(array_a)
        array_b = np.array(array_b)

    x_a, y_a = array_a[[0, 2]], array_a[[1, 3]]
    x_b, y_b = array_b[[0, 2]], array_b[[1, 3]]

    x_sign, y_sign = np.sign(x_a - x_b), np.sign(y_a - y_b)
    x_sign_valid = np.clip(x_sign * np.flip(x_sign), a_min=0, a_max=None)
    y_sign_valid = np.clip(y_sign * np.flip(y_sign), a_min=0, a_max=None)
    x_sign = x_sign * x_sign_valid
    y_sign = y_sign * y_sign_valid

    x_dist, y_dist = x_a - np.flip(x_b), y_a - np.flip(y_b)
    x_dist = np.clip(x_sign * x_dist, a_min=0, a_max=None)
    y_dist = np.clip(y_sign * y_dist, a_min=0, a_max=None)

    x_dist_min, y_dist_min = x_dist.min(), y_dist.min()

    return np.sqrt(np.square(x_dist_min) + np.square(y_dist_min)) + 0.01


def box_distance_array(array_a: np.ndarray, array_b: np.ndarray):
    n_a, n_b = array_a.shape[0], array_b.shape[0]
    array_a = np.tile(np.expand_dims(array_a, axis=0), (n_b, 1, 1))
    array_b = np.tile(np.expand_dims(array_b, axis=1), (1, n_a, 1))

    x_a, y_a = array_a[:, :, [0, 2]], array_a[:, :, [1, 3]]
    x_b, y_b = array_b[:, :, [0, 2]], array_b[:, :, [1, 3]]

    x_sign, y_sign = np.sign(x_a - x_b), np.sign(y_a - y_b)
    x_sign_valid = np.clip(x_sign * np.flip(x_sign, axis=-1), a_min=0, a_max=None)
    y_sign_valid = np.clip(y_sign * np.flip(y_sign, axis=-1), a_min=0, a_max=None)
    x_sign = x_sign * x_sign_valid
    y_sign = y_sign * y_sign_valid

    x_dist, y_dist = x_a - np.flip(x_b, axis=-1), y_a - np.flip(y_b, axis=-1)
    x_dist = np.clip(x_sign * x_dist, a_min=0, a_max=None)
    y_dist = np.clip(y_sign * y_dist, a_min=0, a_max=None)

    x_dist_min, y_dist_min = x_dist.min(axis=-1), y_dist.min(axis=-1)
    distances = np.sqrt(np.square(x_dist_min) + np.square(y_dist_min)) + 0.01

    return distances.min()


def box_distance_with_type(array_a, array_b, multiplier=1000):
    """
    calculate the distance between boxes considering the difference between object types.
    :param array_a:
    :param array_b:
    :param multiplier:
    :return:
    """
    if (not isinstance(array_a, np.ndarray)) and (not isinstance(array_b, np.ndarray)):
        array_a = np.array(array_a)
        array_b = np.array(array_b)

    coordinates_a, coordinates_b = array_a[:-1], array_b[:-1]
    type_a, type_b = array_a[-1], array_b[-1]

    dist = box_distance(coordinates_a, coordinates_b)
    type_dist = 0 if type_a == type_b else multiplier

    return np.sqrt(np.square(dist) + np.square(type_dist))


def get_outlier_threshold(array):
    median = np.median(array)
    per_75 = np.percentile(array, 75)
    per_25 = np.percentile(array, 25)
    print(median, per_25, per_75)
    return median + (per_75 - per_25) * 1.5


def get_distance_matrix(encodings: Union[np.ndarray, List], func):
    if isinstance(encodings, np.ndarray):
        n = encodings.shape[0]
    else:
        n = len(encodings)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if j <= i:
                dist_matrix[i][j] = 0
            else:
                dist_matrix[i][j] = func(encodings[i], encodings[j])

    return dist_matrix


def elbow_cut_off(array, bin_size=2.0, window_size=6, increase_thres=4):
    if not isinstance(array, np.ndarray):
        array = np.array(array)

    array_min = array.min()
    array_max = array.max()
    bins = np.arange(np.floor(array_min / bin_size), np.ceil(array_max / bin_size) + 1) * bin_size
    counts, bins = np.histogram(array, bins)
    starts = bins[1:]

    perc_counts = (counts - counts.min()) / (counts.max() - counts.min())
    perc_starts = (starts - starts.min()) / (starts.max() - starts.min())
    weights = perc_counts + perc_starts

    indices = np.tile(np.expand_dims(np.arange(len(counts) - window_size + 1), axis=1), (1, window_size)) + np.arange(window_size)
    increase_trends = (weights[indices][:, :-1] < weights[indices][:, 1:]).sum(axis=1)
    thres_index = np.where(increase_trends >= increase_thres)[0][0]

    return starts[thres_index]


def get_clusters_ids(n, edges) -> np.ndarray:
    ids = np.arange(n)
    next_id = n

    for edge in edges:
        start_id, end_id = ids[edge]
        ids[(ids == start_id) | (ids == end_id)] = next_id
        next_id += 1

    return np.unique(ids, return_inverse=True)[1]


def root_mean_squared_error(prediction, reference):
    preds_array = np.array(prediction)
    refers_array = np.array(reference)

    return np.sqrt(np.square(preds_array - refers_array).sum()) / preds_array.shape[0]


def detect_line_shape(encodings, ae_threshold=1, angles_threshold=0.25):
    """
    detect whether a cluster of objects form a line shape.
    :param encodings:
    :param ae_threshold:
    :param angles_threshold:
    :return:
    """
    if encodings.shape[0] < 3:
        return False

    center_x = encodings[:, [0, 2]].mean(axis=1)
    center_y = encodings[:, [1, 3]].mean(axis=1)

    tl_x, tl_y = encodings[:, 0], encodings[:, 1]
    br_x, br_y = encodings[:, 2], encodings[:, 3]

    diameters = (np.abs(br_x - tl_x) + np.abs(br_y - tl_y)) / 2

    center_lr = LinearRegression()
    tl_lr = LinearRegression()
    br_lr = LinearRegression()

    center_lr.fit(np.expand_dims(center_x, 1), center_y)
    tl_lr.fit(np.expand_dims(tl_x, 1), tl_y)
    br_lr.fit(np.expand_dims(br_x, 1), br_y)

    center_y_hat = center_lr.predict(np.expand_dims(center_x, 1))
    tl_y_hat = tl_lr.predict(np.expand_dims(tl_x, 1))
    br_y_hat = br_lr.predict(np.expand_dims(br_x, 1))

    center_lr_ae = np.abs(center_y_hat - center_y) / diameters
    tl_lr_ae = np.abs(tl_y_hat - tl_y) / diameters
    br_lr_ae = np.abs(br_y_hat - br_y) / diameters

    center_lr_tan = center_lr.coef_[0]
    tl_lr_tan = tl_lr.coef_[0]
    br_lr_tan = br_lr.coef_[0]

    all_aes = np.stack([center_lr_ae, tl_lr_ae, br_lr_ae])
    all_degrees = np.arctan([center_lr_tan, tl_lr_tan, br_lr_tan]) / np.pi

    valid_aes = all(all_aes.max(axis=1) < ae_threshold)
    valid_angles = np.ptp(all_degrees) < angles_threshold

    return valid_aes and valid_angles


def unit_vector(vector):
    """ Returns the unit vector of the vector. """
    return vector / np.linalg.norm(vector, axis=-1, keepdims=True)


def detect_orientation(encodings_a: np.ndarray, encodings_b: np.ndarray):
    """
    whether object a is surrounded by object b.
    :param encodings_a:
    :param encodings_b:
    :return:
    """

    n_a = encodings_a.shape[0]
    n_b = encodings_b.shape[0]

    vectors = np.tile(np.expand_dims(encodings_a, axis=1), (1, n_b, 1)) \
              - np.tile(np.expand_dims(encodings_b, axis=0), (n_a, 1, 1))
    flatten_vectors = np.concatenate([vectors[:, :, :2], vectors[:, :, 2:]], axis=1)
    phi_vectors = np.arctan2(flatten_vectors[:, :, 1], flatten_vectors[:, :, 0]) * 180 / np.pi

    objects_surround = np.full(phi_vectors.shape[0], False)
    objects_between = np.full(phi_vectors.shape[0], False)

    for i in range(phi_vectors.shape[0]):
        current_vector = np.sort(phi_vectors[i] + 180)
        diffs = np.diff(current_vector, append=current_vector[0] + 360)
        # origin_last_diff = diffs[-1]
        # if diffs[-1] > 180:
        #     diffs[-1] = 360 - diffs[-1]
        # print(diffs)
        is_between = np.sum((diffs >= 135) & (diffs <= 225)) >= 2
        is_surround = np.all(diffs < 190)

        objects_between[i] = is_between
        objects_surround[i] = is_surround

    print(objects_between)
    print(objects_surround)

    # n_vectors = flatten_vectors.shape[1]
    #
    # vectors_a = np.tile(np.expand_dims(flatten_vectors, axis=2), (1, 1, n_vectors, 1))
    # vectors_b = np.tile(np.expand_dims(flatten_vectors, axis=1), (1, n_vectors, 1, 1))
    # unit_vectors_a, unit_vectors_b = unit_vector(vectors_a), unit_vector(vectors_b)
    #
    # products = np.sum(unit_vectors_a * unit_vectors_b, axis=-1)  # a x 2b x 2b
    # angles = np.arccos(np.clip(products, -1.0, 1.0)) / np.pi  # a x 2b x 2b
    # vectors_opposite = np.any(angles > (135 / 180), axis=-1)  # a x 2b
    # vectors_surround = np.any((angles > (45 / 180)) & (angles <= (135 / 180)), axis=-1)  # a x 2b
    # vectors_surround = vectors_surround & vectors_opposite
    #
    # objects_surround = np.any(vectors_surround, axis=-1)  # a
    # objects_between = np.any(vectors_opposite, axis=-1)
    is_surround = np.all(objects_surround)
    is_between = np.all(objects_between)

    is_outside = np.all(~(objects_surround | objects_between))

    is_mixture = (not is_surround) and (not is_outside)

    # unit_flatten_vectors = unit_vector(flatten_vectors)
    # orientation_objects = unit_vector(np.sum(unit_flatten_vectors, axis=-2))
    # orientation = unit_vector(np.sum(orientation_objects, axis=-2))
    # horizontal_string = ("right" if orientation[0] > 0 else "left") if np.abs(orientation[0]) > 0.6 else ""
    # vertical_string = ("bottom" if orientation[1] > 0 else "top") if np.abs(orientation[1]) > 0.6 else ""
    # orientation_string = f"{vertical_string} {horizontal_string}".strip()
    return is_surround, is_between, is_outside, is_mixture#, orientation_string


def describe_relations(clusters, connectivity, encodings, lower_threshold, upper_threshold):
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

    ans = "\n"
    for source, targets in relations:
        for target in targets:
            is_pos_inside, is_pos_outside, is_pos_mixture = detect_orientation(
                encodings[clusters == source, :-1],
                encodings[clusters == target, :-1]
            )

            is_neg_inside, is_neg_outside, is_neg_mixture = detect_orientation(
                encodings[clusters == target, :-1],
                encodings[clusters == source, :-1]
            )

            if is_pos_inside:
                ans += f"group {source} is surrounded by group {target}\n"
                continue

            if is_neg_inside:
                ans += f"group {target} is surrounded by group {source}\n"
                continue

            if is_pos_mixture or is_neg_mixture:
                ans += f"group {source} is adjacent to group {target}\n"
                continue

            distances = box_distance_array(encodings[clusters == source], encodings[clusters == target])

            if distances.min() < lower_threshold:
                ans += f"group {source} is close to group {target}\n"
                continue

            if distances.min() > upper_threshold:
                ans += f"group {source} is far from other objects\n"

    return ans
