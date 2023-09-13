import re
import os
import time
import json
from pathlib import Path
import concurrent.futures as futures

import openai
from openai.error import APIError, Timeout, APIConnectionError, ServiceUnavailableError

from src.geographical_api import *
from src.utils import *

openai.api_key = "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"


@retry_upon_exceptions(
    APIError, Timeout, APIConnectionError, ServiceUnavailableError,
    retry=3, wait_time=30
)
def get_raw_result(model_name, system_message, prompt):

    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
    )

    return completion.choices[0].message.content


@retry_upon_exceptions(
    json.JSONDecodeError, LLMValidationException,
    retry=5, wait_time=1
)
def fetch_result(model_name, system_message, prompt):
    invalid_regex = r"group[-_:\s]*\d+"

    response = get_raw_result(model_name, system_message, prompt)
    response_json = json.loads(response)

    if type(response_json) is not list:
        raise LLMValidationException(f"expected python list, got {type(response_json)}")

    if not response_json:
        raise LLMValidationException(f"invalid python list, got {response_json}")

    for caption in response_json:
        if re.search(invalid_regex, caption.lower()):
            raise LLMValidationException(f"invalid caption, got {caption}")

    return response_json


if __name__ == "__main__":
    with open("../results/unique_blocks_info.json", "r") as f:
        unique_blocks_info = json.load(f)

    api_manager = GeographicalAPIManager(unique_blocks_info)
    block_id = "1044.tif_1"
    api = api_manager.get_api(block_id)

    # polygons = get_polygons(api_manager.blocks_info[block_id], image_size=256)
    # print(api.get_default_descriptions())

    system_message, prompt, _ = api.get_image_description()
    print(fetch_result("gpt-3.5-turbo", system_message, prompt))

