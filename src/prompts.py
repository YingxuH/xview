initial_prompt = """
Today is {today_date}. You are an experienced editor of an aerial images magazine.
You are provided with an aerial image, please provide caption for the image.
Your caption should include possible patterns in objects' locations, such as clustering, geographical concentration, etc.
You should include the shape of clusters and relations between clusters. 

There are roughly {num_objects} objects, which can be divided into {num_clusters} clusters. They are {cluster_names}
"""

identify_types_response_template = \
    "cluster {cluster_id} contains {num_objects} {objects_type}."

identify_shape_response_template = \
    "the {objects_type} in cluster {cluster_id} {shape_description}."

identify_relations_response_template = \
    "cluster {cluster_id_a} {relation_description} cluster {cluster_id_b}"
