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


examples = [
"""
## objects/object groups information
group 0: 1 damaged building
group 1: a line of 5 building
group 2: a line of 3 building

## significant geographical relations
group 2 is close to group 0

## captions
["There are two lines of buildings in the image.", "A damaged building is close to a line of buildings in the image"]
""",
"""
## objects/object groups information
group 0: 1 building
group 1: 1 building
group 2: 1 building
group 3: 1 damaged building
group 4: 1 truck
group 5: 2 building, including 1 building, 1 shed

## significant geographical relations
group 5 is close to group 1
group 2 is close to group 3

## captions
["There are several buildings and one truck in the image.", ""]

"""

]


test = """
today is 13-09-2023, you are a helpful assistant who is proficient at understanding satellite images. 

You are provided with description of a satellite image. Please provide several captions as strings in a python list.
The image has multiple objects, which have been clustered into groups based on their types and locations. 

## objects/object groups information
group 0: 1 damaged building
group 1: 1 excavator
group 2: 2 engineering vehicle, including 1 excavator, 1 mobile crane
group 3: 2 vehicle, including 1 passenger vehicle, 1 truck with flatbed

## significant geographical relations
group 2 is far from other objects
group 3 is close to group 0

# captions:
"""
