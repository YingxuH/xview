import os
import re
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.linear_model import LinearRegression

cmap = plt.get_cmap('viridis')


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


def random_test_sub_blocks(polygons, key, labels, image_path):
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

    unique_rects = [rectangles[i] for i in unique_indices]
    plt.legend(unique_rects, unique_labels)
    plt.show()
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


def get_polygons(block, image_size=256):
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
    if (not isinstance(array_a, np.ndarray)) and (not isinstance(array_b, np.ndarray)):
        array_a = np.array(array_a)
        array_b = np.array(array_b)

    x_a, y_a = array_a[[0, 2]], array_a[[1, 3]]
    x_b, y_b = array_b[[0, 2]], array_b[[1, 3]]

    x_sign, y_sign = np.sign(x_a - x_b), np.sign(y_a - y_b)
    x_dist, y_dist = x_a - np.flip(x_b), y_a - np.flip(y_b)

    x_dist = np.clip(x_sign * x_dist, a_min=0, a_max=None)
    y_dist = np.clip(y_sign * y_dist, a_min=0, a_max=None)

    x_dist_min, y_dist_min = x_dist.min(), y_dist.min()

    return np.sqrt(np.square(x_dist_min) + np.square(y_dist_min)) + 0.01


def box_distance_with_type(array_a, array_b, multiplier=1000):
    if (not isinstance(array_a, np.ndarray)) and (not isinstance(array_b, np.ndarray)):
        array_a = np.array(array_a)
        array_b = np.array(array_b)

    dist = box_distance(array_a[:-1], array_b[:-1])
    type_a, type_b = array_a[-1], array_b[-1]
    type_dist = 0 if type_a == type_b else multiplier

    return np.sqrt(np.square(dist) + np.square(type_dist))


def get_outlier_threshold(array):
    median = np.median(array)
    per_75 = np.percentile(array, 75)
    per_25 = np.percentile(array, 25)
    print(median, per_25, per_75)
    return median + (per_75 - per_25) * 1.5


def get_distance_matrix(encodings: np.ndarray, func=box_distance_with_type):
    n = encodings.shape[0]
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
