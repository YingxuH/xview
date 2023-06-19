from typing import List


class SpatialFeatureHelper:
    def __init__(self, annotations):
        self.annotations = annotations

    def get_spatial_relation(self, cluster_a: List[str], cluster_b: List[str]) -> str:
        pass

    def validate_spatial_relation(self, cluster_a: List[str], cluster_b: List[str], relation: str) -> bool:
        pass

    def validate_cluster(self, elements: List[str]) -> bool:
        pass

    def identify_shape(self, elements: List[str]) -> bool:
        pass

    def cluster(self):
        pass

