import re
import os
import time
import json
import asyncio
from pathlib import Path
import concurrent.futures as futures
from aiohttp import ClientSession
from litellm import acompletion

import openai
from openai.error import APIError, Timeout, APIConnectionError, ServiceUnavailableError

from src.geographical_api import *
from src.utils import *


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
    APIError, Timeout, APIConnectionError, ServiceUnavailableError,
    retry=3, wait_time=30
)
async def async_get_raw_result(model_name, system_message, prompt):

    completion = await acompletion(
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


@retry_upon_exceptions(
    json.JSONDecodeError, LLMValidationException,
    retry=5, wait_time=1
)
async def async_fetch_result(model_name, system_message, prompt, global_base: List):
    invalid_regex = r"group[-_:\s]*\d+"

    response = await async_get_raw_result(model_name, system_message, prompt)
    response_json = json.loads(response)

    if type(response_json) is not list:
        raise LLMValidationException(f"expected python list, got {type(response_json)}")

    if not response_json:
        raise LLMValidationException(f"invalid python list, got {response_json}")

    for caption in response_json:
        if re.search(invalid_regex, caption.lower()):
            raise LLMValidationException(f"invalid caption, got {caption}")

    global_base.append(response_json)
    return response_json


def execute_tasks(tasks_params: List):
    results = defaultdict(lambda: [])

    loop = asyncio.get_event_loop()
    tasks = [
        loop.create_task(async_fetch_result(
            param["model"],
            param["system_message"],
            param["prompt"],
            results[param["block_id"]]
        ))
        for param in tasks_params
    ]

    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()

    return results


if __name__ == "__main__":
    print("pass")
    # with open("../results/unique_blocks_info.json", "r") as f:
    #     unique_blocks_info = json.load(f)
    #
    # api_manager = GeographicalAPIManager(unique_blocks_info)
    # block_id = "1044.tif_1"
    # api = api_manager.get_api(block_id)
    #
    # # polygons = get_polygons(api_manager.blocks_info[block_id], image_size=256)
    # # print(api.get_default_descriptions())
    #
    # system_message, prompt, _ = api.get_image_description()
    # print(fetch_result("gpt-3.5-turbo", system_message, prompt))
