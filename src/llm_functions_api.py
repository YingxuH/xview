import functools
from typing import List, Dict

import numpy as np

from src.clusters_identifier import ClustersIdentifier
from src.utils import detect_shape, detect_orientation


def get_identify_clusters_function(blocks_info: Dict, block_id: str):
    # dentify the clusters from bounding boxes coordinates.

    identifier = ClustersIdentifier(blocks_info)
    identifier.train()
    return functools.partial(identifier.infer, block_id)


def get_identify_shape_function(encoding: np.ndarray):
    # return the shape of any particular cluster/group of objects.
    return functools.partial(detect_shape, encoding=encoding)


def get_identify_relationship_function(encodings_a: np.ndarray, encodings_b: np.ndarray):
    # describe the relationships between objects.
    return functools.partial(detect_orientation, encodings_a=encodings_a, encodings_b=encodings_b)


prompt = """
today is xxx. you are an experienced xxx. you are provided with an annotation about an image. 

Please help me analyse the geographical features of the objects in the image and generate image captions.  
"""