# xview

Assume you are Joshua, an experienced data analyst. I will give you a detailed description of objects in an image and will ask you to complete a few tasks:

Description: "There is a 256 by 256 aerial image. One damaged building is located at x = 192 and y = 44, in the center towards the bottom of the image, with a size = 1485 pixels. One small car is located at x = 23 and y = 25, in the center towards the bottom of the image, with a size = 204 pixels. One small car is located at x = 25 and y = 168, in the center towards the bottom of the image, with a size = 165 pixels. One small car is located at x = 41 and y = 185, in the center towards the bottom of the image, with a size = 240 pixels. One small car is located at x = 74 and y = 220, in the center towards the bottom of the image, with a size = 90 pixels. One small car is located at x = 69 and y = 186, in the center towards the bottom of the image, with a size = 182 pixels. One small car is located at x = 123 and y = 253, in the center towards the bottom of the image, with a size = 120 pixels. One small car is located at x = 91 and y = 242, in the center towards the bottom of the image, with a size = 196 pixels. One small car is located at x = 112 and y = 241, on the center towards the bottom of the image, with a size = 110 pixels."

Task 1: Determine spatial relations between objects. Task 2: Extract high-level spatial patterns between the objects, such as direction, clustering, dispersion, encirclement, interposition, etc. Task 3: Generate one or more independent image captions with high-level significant spatial patterns. Each starts with "CAP", with no longer than 32 tokens.


Assume you are Joshua, an experienced data analyst. I will give you a detailed description of objects in an image and will ask you to complete a few tasks:

Description: "There is a 256 by 256 aerial image. One building is located at x = 44 and y = 5, on the top right of the image, with a size = 5346 pixels. One building is located at x = 204 and y = 203, on the top right of the image, with a size = 7865 pixels. One building is located at x = 60 and y = 31, on the top right of the image, with a size = 3010 pixels. One building is located at x = 209 and y = 5, on the top right of the image, with a size = 3705 pixels. One building is located at x = 74 and y = 75, on the top right of the image, with a size = 6150 pixels. One building is located at x = 160 and y = 22, on the top right of the image, with a size = 2484 pixels. One building is located at x = 16 and y = 96, on the top right of the image, with a size = 5120 pixels. One building is located at x = 146 and y = 218, on the top right of the image, with a size = 513 pixels. One building is located at x = 244 and y = 5, on the top right of the image, with a size = 3055 pixels. One cargo truck is located at x = 153 and y = 87, on the top right of the image, with a size = 702 pixels. One truck is located at x = 233 and y = 44, on the top right of the image, with a size = 950 pixels."

Tasks:
Task 1: Determine spatial relationships between objects, in terms of directions, distances, etc. Task 2: Identify high-level spatial patterns from the objects, such as direction, clustering, dispersion, encirclement, interposition, etc. Task 3: Generate one or more independent image captions with high-level significant spatial patterns. Each starts with "CAP", with no longer than 32 tokens.


TODO 1: Ask LLM to rephrase the prompt into multiple versions. 
TODO 2: sample multiple times and do random forest. 
TODO 3: evaluation: between image and text (CLIP distance?). text to text (clustering?). inter-text (duplication?). 
