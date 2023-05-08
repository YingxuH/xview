import re
import os
import time
import json
from pathlib import Path
import concurrent.futures as futures

import openai
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from gensim.utils import tokenize

from src import constants


class FreeChatGptCaller:
    def __init__(self, prefix_index=0, suffix_index=0, save=True):
        self.prefix_prompt = constants.PROMPT_PREFIX[prefix_index]
        self.suffix_prompt = " ".join(constants.PROMPT_SUFFIX[suffix_index])
        self.caption_task_regex = "Task\s*3[:=-](.*)$"
        self.caption_head_regex = "CAP\s*\d*\s*[:=-]"
        self.save = save

        self.dummy_response = ''
        self.save_directory = "log/free/"
        self.time_limit = 120
        self.do_not_throw_number_prompt = constants.DO_NOT_THROW_NUMBER_PROMPT
        self.do_not_make_up_prompt = constants.DO_NOT_MAKE_UP_PROMPT
        self.do_not_make_up_keywords = constants.DO_NOT_MAKE_UP_KEYWORDS
        self.follow_task_format_prompt = constants.FOLLOW_TASK_FORMAT_PROMPT
        self.follow_caption_format_prompt = constants.FOLLOW_CAPTION_FORMAT_PROMPT
        self.follow_caption_length_prompt = constants.FOLLOW_CAPTION_LENGTH_PROMPT

        self.history = []
        self.session = []

        self.target_url = constants.FREE_GPT_URL
        self.driver = None
        self.init_webdriver()

    def start_session(self):
        self.session = []

    def clear_session(self):
        self.history.append(self.session)
        self.session = []
        clear_button = self.get_buttons()[0]
        clear_button.click()

    def clear_history(self):
        if self.save:
            Path(self.save_directory).mkdir(parents=True, exist_ok=True)
            file_name = f"{time.strftime('%y%m%d%H%M%S')}.json"
            with open(os.path.join(self.save_directory, file_name), "w") as f:
                json.dump(self.history, f, indent=4)

        self.history = []

    def init_webdriver(self, retry=3):
        if retry == 0:
            return

        try:
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--disable-gpu')
            self.driver = webdriver.Chrome(chrome_options=options)
            self.driver.get(self.target_url)
        except Exception:
            self.init_webdriver(retry=retry-1)

    def quit_webdriver(self):
        self.driver.quit()
        self.driver = None

    def get_text_area(self):
        return self.driver.find_element(By.XPATH, "//textarea")

    def get_buttons(self):
        buttons = self.driver.find_elements(By.XPATH, f"//button")
        clear_button, send_button = None, None
        for button in buttons:
            if button.get_attribute("title") == "Clear":
                clear_button = button
            else:
                send_button = button
        return clear_button, send_button

    def has_xpath_element(self, path):
        elements = self.driver.find_elements(By.XPATH, path)
        return len(elements) > 0

    def _get_response(self):
        while not self.has_xpath_element("//textarea"):
            time.sleep(1)

        responses = self.driver.find_elements(By.XPATH, "//div[@class='flex gap-3 rounded-lg']")

        return responses[-1].text

    def get_response(self):
        # Main
        with futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._get_response)
            resp = future.result(self.time_limit)
            executor._threads.clear()
            futures.thread._threads_queues.clear()
        return resp

    def call(self, input_message):
        input_ele = self.get_text_area()
        input_ele.clear()
        input_ele.send_keys(input_message)
        _, send_button = self.get_buttons()
        send_button.click()
        response = self.get_response()

        self.session.append({"role": "user", "content": input_message})
        self.session.append({"role": "assistant", "content": response})
        return response

    def caption_validation(self, response):
        task_group = re.search(self.caption_task_regex, response.replace("\n", ""), re.IGNORECASE)
        if task_group is None:
            return self.follow_task_format_prompt

        captions_heads = re.findall(self.caption_head_regex, task_group.group(1))
        if len(captions_heads) == 0:
            return self.follow_caption_format_prompt

        captions = re.split(self.caption_head_regex, task_group.group(1))[1:]
        captions_lens = [len(list(tokenize(cap.strip()))) > 25 for cap in captions]
        if any(captions_lens):
            return self.follow_caption_length_prompt

        return ""

    def _query_image(self, polygons):
        input_message = f"{self.prefix_prompt}: Description: '{polygons}' Tasks: {self.suffix_prompt}"

        response = self.call(input_message)

        follow_up_prompt = self.caption_validation(response)
        if follow_up_prompt != "":
            response = self.call(follow_up_prompt)

        return response

    def query_image(self, polygons, retry=3):
        if retry == 0:
            return self.dummy_response

        self.start_session()
        try:
            response = self._query_image(polygons)
        except Exception as e:
            print(f"retry {retry}", e)
            self.quit_webdriver()
            self.init_webdriver()
            response = self.query_image(polygons, retry=retry-1)

        self.clear_session()
        return response


class APIGptCaller:
    def __init__(self, model_name, system_message, limit=2500, save=True):
        self.model_name = model_name
        self.system_message = system_message

        self.total_use = 0
        self.limit = limit * 1000
        self.dummy_response = ''
        self.save = save
        self.save_directory = "log/"

        self.history = []
        self.session = []

    def start_session(self):
        self.session = [{"role": "system", "content": self.system_message}]

    def clear_session(self):
        self.history.append(self.session)
        self.session = []

    def clear_history(self):
        if self.save:
            Path(self.save_directory).mkdir(parents=True, exist_ok=True)
            file_name = f"{time.strftime('%y%m%d%H%M%S')}.json"
            with open(os.path.join(self.save_directory, file_name), "w") as f:
                json.dump(self.history, f, indent=4)

        self.history = []

    def _call(self, input_message):
        return 0, self.dummy_response

    def call(self, input_message, retry=3):
        if retry == 0:
            return self.dummy_response

        try:
            use, response = self._call(
                input_message=input_message
            )
        except Exception as e:
            print(f"retry {retry}: {e}")
            return self.call(input_message, retry=retry-1)

        self.session.append({"role": "user", "content": input_message})
        self.session.append({"role": "assistant", "content": response})
        self.total_use += use
        return response

    def query_image(self, input_message, retry=3):
        self.start_session()
        response = self.call(input_message, retry=retry)
        self.clear_session()
        return response


class ChatGptCaller(APIGptCaller):
    def __init__(self, model_name, system_message, limit=2500, save=True):
        super().__init__(model_name, system_message, limit, save)

    def _call(self, input_message):
        messages = self.session + [{"role": "user", "content": input_message}]

        result = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                max_tokens=512,
                temperature=0.7
            )

        use = result["usage"]["total_tokens"]
        response = result['choices'][0]['message']['content']
        return use, response


class CompletionGptCaller(APIGptCaller):
    def __init__(self, model_name, system_message, limit=2500, save=True):
        super().__init__(model_name, system_message, limit, save)

    def _call(self, input_message):
        result = openai.Completion.create(
            model=self.model_name,
            prompt=input_message,
            max_tokens=512,
            temperature=0.7
        )

        use = result["usage"]["total_tokens"]
        response = result['choices'][0]["text"]
        return use, response

class _ChatGptCaller:
    def __init__(self, model_name, system_message, mode="chat", limit=2500, save=True):
        self.model_name = model_name
        self.system_message = system_message
        if mode == "text":
            self.wrapper = openai.Completion 
        elif mode == "chat":
            self.wrapper = openai.ChatCompletion
        
        self.total_use = 0
        self.limit = limit * 1000
        self.dummy_response = ''
        self.save = save
        self.save_directory = "log/"
        self.do_not_throw_number_prompt = constants.DO_NOT_THROW_NUMBER_PROMPT
        self.do_not_make_up_prompt = constants.DO_NOT_MAKE_UP_PROMPT
        self.do_not_make_up_keywords = constants.DO_NOT_MAKE_UP_KEYWORDS
        
        self.history = []
        self.session = []
        
    def start_session(self):
        self.session = [{"role": "system", "content": self.system_message}]
        
    def clear_session(self):
        self.history.append(self.session)
        self.session = []
        
    def clear_history(self):
        if self.save:
            Path(self.save_directory).mkdir(parents=True, exist_ok=True)
            file_name = f"{time.strftime('%y%m%d%H%M%S')}.json"
            with open(os.path.join(self.save_directory, file_name), "w") as f:
                json.dump(self.history, f, indent=4)
                
        self.history = []
        
    
    def call(self, input_message, retry=3):
        if retry == 0:
            return self.dummy_response
        if self.model_name not in constants.TEXT_COMPLETION_MODELS + constants.CHAT_COMPLETION_MODELS:
            raise Exception(f"unknown model name: {self.model}")
        
        if self.model_name in constants.TEXT_COMPLETION_MODELS:
            messages = input_message
        elif self.model_name in constants.CHAT_COMPLETION_MODELS:
            messages = self.session + [{"role": "user", "content": input_message}]
            
        try:
            result = _unified_gpt_create(
                model=self.model_name,
                message=messages,
                max_tokens=256,
                temperature=0.7
            )
        except Exception as e:
            raise e
#             print(f"retry {retry}: {e}")
#             return self.call(messages, retry=retry-1)
#         print(result)
        
        if self.model_name in constants.TEXT_COMPLETION_MODELS:
            response = result['choices'][0]["text"]
        elif self.model_name in constants.CHAT_COMPLETION_MODELS:
            response = result['choices'][0]['message']['content']
        use = result["usage"]["total_tokens"]
        
        self.session.append({"role": "user", "content": input_message})
        self.session.append({"role": "assistant", "content": response})
        self.total_use += use
        return response
    
    def do_not_throw_number_validation(self, polygons, response):
        image_size_mentions = 2
        other_mentions = min(len(polygons), 2)
        
        return len(re.findall("\d+", response)) <= (image_size_mentions + other_mentions)
    
    def do_not_make_up_validation(self, polygons, response):
        counts = [token for token in tokenize(response) if token in self.do_not_make_up_keywords]
        
        return len(counts) < 2
        
    def query_image(self, polygons, retry=3):
        self.start_session()
#         input_message = f"{self.input_prompt}: {polygons}".replace("\n", "").replace("\'", "")
        input_message = polygons
        response = self.call(input_message)
        
#         if not self.do_not_throw_number_validation(polygons, response):
#             print(f"follow up: {response}")
#             response = self.call(self.do_not_throw_number_prompt)
        
#         if not self.do_not_make_up_validation(polygons, response):
#             print(f"follow up: {response}")
#             response = self.call(self.do_not_make_up_prompt)
            
        self.clear_session()
        return response
