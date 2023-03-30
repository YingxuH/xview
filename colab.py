import os
import json
import openai

openai.organization = "org-g1Oot6wcrGN5vljW3gu8b4je"
openai.api_key = "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"

annotations_path = "xview/annotations/"
annotations = {}
for file in os.listdir(annotations_path):
  if file.endswith(".json"):
    with open(os.path.join(annotations_path, file), "r") as f:
      annotations[file.replace(".tif.json", "")] = json.load(f)
      
text_prompt = "This is an annotation for polygons in a satellite image. With that, can you caption the image with a brief description on the number of each type of polygons and their spatial relations? You could focus on their quantity, shapes, relative locations in the image, etc. Focus on details, be accurate on the polygon types, but do not throw numbers overwhelmingly. Please act like you are looking at the image without using any words like 'polygon' or 'annotation' or mentioning this set of polygon markings. You can start with the caption right away."

anno_block = annotations['2131']
records = {}
for i in range(0,len(anno_block),10):
  polygons = anno_block[i:10+i]
  text_input = f"{text_prompt}: {polygons}"
  result = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": text_input}
        ])

  response = result['choices'][0]['message']['content']
  records[i] = {"poly": polygons, "res":response}
