# xview

Assume you are Joshua, an experienced data analyst. I will give you a detailed description of objects in an image and will ask you to complete a few tasks:

Description: "There is a 256 by 256 aerial image. One building is located at x = 67 and y = 8, in the top left of the image, with a size = 5775 pixels. One building is located at x = 114 and y = 59, in the center towards the top of the image, with a size = 3185 pixels. One building is located at x = 188 and y = 83, in the top right of the image, with a size = 2565 pixels. One building is located at x = 196 and y = 239, in the bottom right of the image, with a size = 2337 pixels. One building is located at x = 188 and y = 162, in the bottom right of the image, with a size = 6160 pixels. One building is located at x = 145 and y = 207, in the center towards the bottom of the image, with a size = 8374 pixels. One building is located at x = 153 and y = 41, in the center towards the top of the image, with a size = 3120 pixels. One building is located at x = 199 and y = 2, in the top right of the image, with a size = 7884 pixels. One building is located at x = 222 and y = 60, in the top right of the image, with a size = 2300 pixels. One small car is located at x = 30 and y = 34, in the top left of the image, with a size = 182 pixels. One small car is located at x = 38 and y = 50, in the top left of the image, with a size = 168 pixels. One small car is located at x = 56 and y = 70, in the top left of the image, with a size = 240 pixels. One small car is located at x = 86 and y = 83, in the top left of the image, with a size = 288 pixels."

Task 1: Determine spatial relations between objects. Task 2: Extract high-level spatial patterns between the objects, such as direction, clustering, dispersion, encirclement, interposition, etc. Task 3: Generate one or more independent image captions with high-level significant spatial patterns. Each starts with "CAP", with no longer than 32 tokens.


Assume you are Joshua, an experienced data analyst. I will give you a detailed description of objects in an image and will ask you to complete a few tasks:

Description: "There is a 256 by 256 aerial image. One building is located at x = 31 and y = 83, in the top left of the image, with a size = 3658 pixels. One building is located at x = 35 and y = 153, in the middle left of the image, with a size = 5360 pixels. One building is located at x = 125 and y = 251, in the center towards the bottom of the image, with a size = 3416 pixels. One building is located at x = 133 and y = 69, in the center towards the top of the image, with a size = 24650 pixels. One building is located at x = 152 and y = 224, in the center towards the bottom of the image, with a size = 2025 pixels. One building is located at x = 172 and y = 165, in the bottom right of the image, with a size = 2106 pixels. One building is located at x = 194 and y = 195, in the bottom right of the image, with a size = 2000 pixels. One building is located at x = 196 and y = 113, in the middle right of the image, with a size = 2622 pixels. One building is located at x = 222 and y = 221, in the bottom right of the image, with a size = 3100 pixels. One building is located at x = 230 and y = 155, in the bottom right of the image, with a size = 910 pixels. One building is located at x = 248 and y = 194, in the bottom right of the image, with a size = 3445 pixels."

Tasks:
Task 1: Determine spatial relationships between objects, in terms of directions, distances, etc. Task 2: Identify high-level spatial patterns from the objects, such as direction, clustering, dispersion, encirclement, interposition, etc. Task 3: Generate one or more independent image captions with high-level significant spatial patterns. Each starts with "CAP", with no longer than 32 tokens.


TODO 1: Ask LLM to rephrase the prompt into multiple versions. 
TODO 2: sample multiple times and do random forest. 
TODO 3: evaluation: between image and text (CLIP distance?). text to text (clustering?). inter-text (duplication?). 

