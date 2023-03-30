# xview

sample input: 

This is an annotation for polygons in a satellite image. With that, can you caption the image with a brief description on the number of each type of polygons 
and their spatial relations? Focus on their locations in the image. Please act like you are looking at the image without using any word like "polygon" or "annotation". 
You can start with the caption right away.  

[{'polygon_coordinates': [[[100.764096, 13.656486],
    [100.764096, 13.656279],
    [100.763869, 13.656279],
    [100.763869, 13.656486],
    [100.764096, 13.656486]]],
  'polygon_type': 'Building'},
 {'polygon_coordinates': [[[100.764337, 13.656157],
    [100.764337, 13.656107],
    [100.764298, 13.656107],
    [100.764298, 13.656157],
    [100.764337, 13.656157]]],
  'polygon_type': 'SmallCar'},
 {'polygon_coordinates': [[[100.764105, 13.656166],
    [100.764105, 13.656129],
    [100.764161, 13.656129],
    [100.764161, 13.656166],
    [100.764105, 13.656166]]],
  'polygon_type': 'SmallCar'},
 {'polygon_coordinates': [[[100.763769, 13.656257],
    [100.763769, 13.656131],
    [100.763917, 13.656131],
    [100.763917, 13.656257],
    [100.763769, 13.656257]]],
  'polygon_type': 'Building'},
 {'polygon_coordinates': [[[100.764317, 13.655808],
    [100.764317, 13.655717],
    [100.764207, 13.655717],
    [100.764207, 13.655808],
    [100.764317, 13.655808]]],
  'polygon_type': 'Building'},
 {'polygon_coordinates': [[[100.764065, 13.655895],
    [100.764065, 13.655758],
    [100.764249, 13.655758],
    [100.764249, 13.655895],
    [100.764065, 13.655895]]],
  'polygon_type': 'Building'},
 {'polygon_coordinates': [[[100.764004, 13.655852],
    [100.764004, 13.655771],
    [100.764076, 13.655771],
    [100.764076, 13.655852],
    [100.764004, 13.655852]]],
  'polygon_type': 'Building'},
 {'polygon_coordinates': [[[100.764053, 13.65566],
    [100.764053, 13.655551],
    [100.764174, 13.655551],
    [100.764174, 13.65566],
    [100.764053, 13.65566]]],
  'polygon_type': 'Building'},
 {'polygon_coordinates': [[[100.764046, 13.655631],
    [100.764046, 13.655517],
    [100.763925, 13.655517],
    [100.763925, 13.655631],
    [100.764046, 13.655631]]],
  'polygon_type': 'Building'},
 {'polygon_coordinates': [[[100.763783, 13.655643],
    [100.763783, 13.655561],
    [100.763877, 13.655561],
    [100.763877, 13.655643],
    [100.763783, 13.655643]]],
  'polygon_type': 'Building'}]
 """
 
 output: This image shows various structures and vehicles in a given area. 
 At the center is the largest structure with rectangular shape, while nearby are four more smaller structures with triangular shape. 
 There are also two small cars parked next to each other. 
 The structures and vehicles are arranged in a loose grouping, with the largest rectangular structure serving as the focal point.
