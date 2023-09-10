initial_prompt = """
Today is {today_date}. You are an experienced editor of an aerial images magazine.
You are now provided with a remote sensing image, please provide caption for the image.

Your caption should include patterns in objects' locations, such as the clustering of objects, and significant shapes formed by objects.
You should also include spatial relations between objects/groups. 

There are roughly {num_objects} objects, which formed {num_clusters} clusters. They are {cluster_names}.
"""

identify_types_response_template = \
    "cluster {cluster_id} contains {num_objects} {objects_type}."

identify_shape_response_template = \
    "the {objects_type} in cluster {cluster_id} {shape_description}."

identify_relations_response_template = \
    "cluster {cluster_id_a} {relation_description} cluster {cluster_id_b}"
