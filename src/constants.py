INPUT_PROMPT = "Act as an image captioner to provide an accurate caption in 2-5 sentences for a 256 x 256 satellite image based on a set of object annotations. I want you to describe only the sizes and positional relationships of these objects in plain English. I want you to describe features using English instead of throwing the exact numbers overwhelmingly. Here is the annotation"

DO_NOT_THROW_NUMBER_PROMPT = "Please describe in English rather than throwing out numbers overwhelmingly. Try one more time."
DO_NOT_MAKE_UP_PROMPT = "I want you to only describe the content of the annotation. Do not make anything up. Try one more time."
DO_NOT_MAKE_UP_KEYWORDS = ["colored", "colour", "color", "blue", "red", "yellow", "orange", "brown", "color", "window", "windows", "roof", "flat", "fence", "dark", "darker"]
DO_NOT_MENTION_ANNOTATION_KEYWORDS = ["annotation", "annotations", "marked", "mark", "marks"]