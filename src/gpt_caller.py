import re
import time
import concurrent.futures as futures

import openai
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from gensim.utils import tokenize

from src import constants


class FreeChatGptCaller:
    def __init__(self, input_prompt, save=True):
        self.input_prompt = input_prompt 
        self.save = save
        
        self.dummy_response = ''
        self.save_directory = "log/free/"
        self.time_limit = 60
        self.do_not_throw_number_prompt = constants.DO_NOT_THROW_NUMBER_PROMPT
        self.do_not_make_up_prompt = constants.DO_NOT_MAKE_UP_PROMPT
        self.do_not_make_up_keywords = constants.DO_NOT_MAKE_UP_KEYWORDS
        
        self.history = []
        self.session = []
        
        self.target_url = "https://gptai.cloud"
        self.driver = webdriver.Chrome()
        self.driver.get(self.target_url)
        
    def get_text_area(self):
        return self.driver.find_element(By.XPATH, "//textarea")
        
    def get_buttons(self):
        buttons = self.driver.find_elements(By.XPATH, f"//button")
        for button in buttons:
            if button.get_attribute("title") == "Clear":
                clear_button = button
            else:
                send_button = button
        return clear_button, send_button
    
    def _get_response(self):
        prev_txt = None
        txt = ""
        equal_count = 0
        
        while (prev_txt is None) or (prev_txt != txt) or equal_count <= 5:
            time.sleep(1)
            responses = self.driver.find_elements(By.XPATH, "//div[@class='flex gap-3 rounded-lg']")            
            if not responses:
                continue
            if txt == responses[-1].text:
                equal_count += 1
            prev_txt = txt
            txt = responses[-1].text
            
        return txt
    
    def get_response(self):
        # Main
        with futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._get_response)
            resp = future.result(self.time_limit)
            executor._threads.clear()
            futures.thread._threads_queues.clear()
        return resp
        
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
        
    
    def call(self, input_message, retry=3):
        if retry == 0:
            return self.dummy_response
        
        try:
            input_ele = self.get_text_area()
            input_ele.clear()
            input_ele.send_keys(input_message)
            send_button = self.get_buttons()[1]
            send_button.click()
            response = self.get_response()
        except Exception as e:
            print(f"retry {retry}: {e}")
            return self.call(input_message, retry=retry-1)
                        
        self.session.append({"role": "user", "content": input_message})
        self.session.append({"role": "assistant", "content": response})
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
        input_message = f"{self.input_prompt}: {polygons}".replace("\n", "").replace("\'", "")

        response = self.call(input_message)
        
        if not self.do_not_throw_number_validation(polygons, response):
            print(f"follow up: {response}")
            response = self.call(self.do_not_throw_number_prompt)
        
        if not self.do_not_make_up_validation(polygons, response):
            print(f"follow up: {response}")
            response = self.call(self.do_not_make_up_prompt)
            
        self.clear_session()
        return response
    
    
class ChatGptCaller:
    def __init__(self, model_name, system_message, input_prompt, limit=2500, save=True):
        self.input_prompt = input_prompt
        self.model_name = model_name
        self.system_message = system_message
        
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
        
        messages = self.session + [{"role": "user", "content": input_message}]

        try:
            result = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages)
        except Exception as e:
            raise e
#             print(f"retry {retry}: {e}")
#             return self.call(messages, retry=retry-1)
            
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
        input_message = f"{self.input_prompt}: {polygons}".replace("\n", "").replace("\'", "")

        response = self.call(input_message)
        
        if not self.do_not_throw_number_validation(polygons, response):
            print(f"follow up: {response}")
            response = self.call(self.do_not_throw_number_prompt)
        
        if not self.do_not_make_up_validation(polygons, response):
            print(f"follow up: {response}")
            response = self.call(self.do_not_make_up_prompt)
            
        self.clear_session()
        return response