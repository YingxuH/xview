INPUT_PROMPT = "Act as an image captioner to provide an accurate caption in 2-5 sentences for a 256 x 256 satellite image based on a set of object annotations. I want you to describe only the sizes and positional relationships of these objects in plain English. I want you to describe features using English instead of throwing the exact numbers overwhelmingly. Here is the annotation"

DO_NOT_THROW_NUMBER_PROMPT = "Please describe in English rather than throwing out numbers overwhelmingly. Try one more time."
DO_NOT_MAKE_UP_PROMPT = "I want you to only describe the content of the annotation. Do not make anything up. Try one more time."
DO_NOT_MAKE_UP_KEYWORDS = ["colored", "colour", "color", "blue", "red", "yellow", "orange", "brown", "color", "window", "windows", "roof", "flat", "fence", "dark", "darker"]
DO_NOT_MENTION_ANNOTATION_KEYWORDS = ["annotation", "annotations", "marked", "mark", "marks"]


PROMPT_PREFIX = ["Assume you are Joshua, an experienced data analyst. I will give you a detailed description of objects in an image and will ask you to complete a few tasks:"]

PROMPT_SUFFIX = [
    ["Task 1: Determine spatial relationships between objects, in terms of directions, distances, etc.", 
     "Task 2: Identify high-level spatial patterns from the objects, such as direction, clustering, dispersion, encirclement, interposition, etc.", 
     "Task 3: Generate one or more independent image captions with high-level significant spatial patterns. Each starts with 'CAP', with no longer than 32 tokens."],
    ["Task 1: Determine spatial relations between objects.",
     "Task 2: Extract high-level spatial patterns between the objects, such as direction, clustering, dispersion, encirclement, interposition, etc.",
     "Task 3: Generate one or more independent image captions with high-level significant spatial patterns. Each starts with "CAP", with no longer than 32 tokens."]
]