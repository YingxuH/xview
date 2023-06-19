import os
import re
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.linear_model import LinearRegression


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
