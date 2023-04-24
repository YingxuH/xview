# xview

Assume you are Joshua, an experienced data analyst. I will give you a detailed description of objects in an image and will ask you to complete a few tasks:

Description: "There is a 256 by 256 aerial image. One damaged building is located at x = 192 and y = 44, in the center towards the bottom of the image, with a size = 1485 pixels. One small car is located at x = 23 and y = 25, in the center towards the bottom of the image, with a size = 204 pixels. One small car is located at x = 25 and y = 168, in the center towards the bottom of the image, with a size = 165 pixels. One small car is located at x = 41 and y = 185, in the center towards the bottom of the image, with a size = 240 pixels. One small car is located at x = 74 and y = 220, in the center towards the bottom of the image, with a size = 90 pixels. One small car is located at x = 69 and y = 186, in the center towards the bottom of the image, with a size = 182 pixels. One small car is located at x = 123 and y = 253, in the center towards the bottom of the image, with a size = 120 pixels. One small car is located at x = 91 and y = 242, in the center towards the bottom of the image, with a size = 196 pixels. One small car is located at x = 112 and y = 241, on the center towards the bottom of the image, with a size = 110 pixels."

Task 1: Determine spatial relations between objects. Task 2: Extract high-level spatial patterns between the objects, such as direction, clustering, dispersion, encirclement, interposition, etc. Task 3: Generate one or more independent image captions with high-level significant spatial patterns. Each starts with "CAP", with no longer than 32 tokens.



TODO 1: Ask LLM to rephrase the prompt into multiple versions. 
TODO 2: sample multiple times and do random forest. 
TODO 3: evaluation: between image and text (CLIP distance?). text to text (clustering?). inter-text (duplication?). 
