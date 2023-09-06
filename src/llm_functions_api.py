import typing

import numpy as np


def identify_clusters(coordinates: np.ndarray) -> typing.List:
    """
    identify the clusters from bounding boxes coordinates.
    :param coordinates:
    :return:
    """
    pass


def identify_shape(coordinates: np.ndarray) -> str:
    """
    return the shape of any particular cluster/group of objects.
    :param coordinates:
    :return:
    """
    pass


def identify_relationship(coordinates_a: np.ndarray, coordinates_b: np.ndarray) -> str:
    """
    describe the relationships between objects.
    :param coordinates_a:
    :param coordinates_b:
    :return:
    """
    pass


prompt = """
today is xxx. you are an experienced xxx. you are provided with an annotation about an image. 

Please help me analyse the geographical features of the objects in the image and generate image captions.  
"""