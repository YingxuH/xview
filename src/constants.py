FREE_GPT_URL = "https://gptai.cloud"
CHAT_COMPLETION_MODELS = ["gpt-3.5-turbo"]
TEXT_COMPLETION_MODELS = ["text-davinci-003"]

INPUT_PROMPT = "Act as an image captioner to provide an accurate caption in 2-5 sentences for a 256 x 256 satellite image based on a set of object annotations. I want you to describe only the sizes and positional relationships of these objects in plain English. I want you to describe features using English instead of throwing the exact numbers overwhelmingly. Here is the annotation"

DO_NOT_THROW_NUMBER_PROMPT = "Please describe in English rather than throwing out numbers overwhelmingly. Try one more time."
DO_NOT_MAKE_UP_PROMPT = "I want you to only describe the content of the annotation. Do not make anything up. Try one more time."
DO_NOT_MAKE_UP_KEYWORDS = ["colored", "colour", "color", "blue", "red", "yellow", "orange", "brown", "color", "window", "windows", "roof", "flat", "fence", "dark", "darker"]
DO_NOT_MENTION_ANNOTATION_KEYWORDS = ["annotation", "annotations", "marked", "mark", "marks"]
FOLLOW_TASK_FORMAT_PROMPT = "Please make sure the answer task 3 starts with 'Task 3'. Please try again."
FOLLOW_CAPTION_FORMAT_PROMPT = "Please start each caption with 'CAP'. Please try one more time."
FOLLOW_CAPTION_LENGTH_PROMPT = "Please shorten the captions with each no longer than 32 tokens while keeping the original meaning. Let's do it!"

PROMPT_PREFIX = [
    "Assume you are Joshua, an experienced data analyst. I will give you a detailed description of objects in an image and will ask you to complete a few tasks"
]

PROMPT_SUFFIX = [
    ["Task 1: Determine spatial relationships between objects, in terms of directions, distances, etc.", 
     "Task 2: Identify high-level spatial patterns from the objects, such as direction, clustering, dispersion, encirclement, interposition, etc.", 
     "Task 3: Generate one or more independent image captions with high-level significant spatial patterns. Each starts with 'CAP', with no longer than 32 tokens."],
    ["Task 1: Determine spatial relations between objects.",
     "Task 2: Extract high-level spatial patterns between the objects, such as direction, clustering, dispersion, encirclement, interposition, etc.",
     "Task 3: Generate one or more independent image captions with high-level significant spatial patterns. Each starts with 'CAP', with no longer than 32 tokens."],
    ["Task 1: Analyze the input description to identify the objects present in the image and their spatial coordinates and sizes.",
     "Task 2: Determine the spatial relationships between the objects, such as their relative positions, directions, clustering, dispersion, encirclement, interposition, etc.",
     "Task 3: Use the identified spatial relationships to generate one or more independent image captions that accurately and vividly describe the high-level significant spatial patterns present in the image. Each caption should start with ‘CAP’ and be no longer than 32 tokens."],
    ["Task 1: Determine spatial relations between objects.",
     "Task 2: Extract high-level spatial patterns between the objects, such as adjacency (objects very close to other objects), clustering (objects concentrated in the same area), formation (objects form a line/arc/circle shape, etc.), dispersion (objects scattered around), encirclement (objects surrounded by others), interposition (objects between others), etc.",
     "Task 3: Generate one or more independent image captions with high-level significant spatial patterns. Each starts with 'CAP', with no longer than 32 tokens."],
    ["Task 1: Identify and extract the coordinates, size, and type of each object in the input description.",
     "Task 2: Determine the spatial relationships between objects, such as their relative positions (e.g., top left, bottom right), distances, and formations (e.g., line, arc, circle)."
     "Task 3: Generate one or more independent image captions that accurately describe the significant spatial patterns and relationships between the objects. Each caption should start with ‘Caption’ and be no longer than 32 tokens."],
    ["Task 1: Identify the objects in the image and their locations and sizes. Use the format: ‘There is a/an [object] at [x,y] with a size of [pixels].’ Separate each object with a period.",
     "Task 2: Compare the locations and sizes of the objects and describe how they relate to each other in terms of distance and direction. Use the format: ‘[Object A] is [relation] to [object B].’ Separate each relation with a period.",
     "Task 3: Summarize the main spatial patterns of the objects in one or more sentences. Start each sentence with ‘CAP’. Use descriptive words and phrases to make the captions vivid and accurate."],
    ["Task 1: Identify the objects in the image and their locations and sizes.",
     "Task 2: Determine the spatial relationships between the objects, such as their relative positions and distances from each other.",
     "Task 3: Identify any patterns or formations in the arrangement of the objects, such as lines, arcs, clusters, or dispersion.",
     "Task 4: Generate one or more independent image captions that accurately describe the significant spatial patterns and relationships between the objects. Each caption should start with ‘CAP’ and be no longer than 32 tokens."],
    ["Task 1: Identify the objects in the image and their locations and sizes. Use the format: ‘There is a/an [object] at [x,y] with a size of [pixels].’ Separate each object with a period.",
     "Task 2: Compare the locations and sizes of the objects and describe how they relate to each other in terms of distance and direction. Use the format: ‘[Object A] is [relation] to [object B].’ Use words like 'close to', 'on the top', 'on the bottom' , 'on the left', 'on the right', etc. Separate each relation with a period.",
     "Task 3: Identify any patterns or formations in the arrangement of the objects, such as lines, arcs, clusters, or dispersion.",
     "Task 4: Generate one or more independent image captions that accurately describe the significant spatial patterns and relationships between the objects. Each caption should start with ‘CAP’ and be no longer than 32 tokens."],
    ["Task 1: Identify the objects in the image and their locations and sizes. Use the format: ‘There is a/an [object] at [x,y] with a size of [pixels].’ Separate each object with a period.",
     "Task 2: Compare the locations and sizes of the objects and describe how they relate to each other in terms of distance and direction. Use the format: ‘[Object A] is [relation] to [object B].’ Separate each relation with a period.",
     "Task 3: Identify any patterns or formations in the arrangement of the objects, such as lines, arcs, clusters, or dispersion.",
     "Task 4: Generate one or more independent image captions that accurately describe the significant spatial patterns and relationships between the objects. Each caption should start with ‘CAP’ and be no longer than 32 tokens."]

]
