from abc import abstractmethod
import requests
import json
import time
import html
from typing import Generator
from loguru import logger
from more_itertools import peekable
from typing import Generator, Optional, Tuple, Union
from dotenv import load_dotenv
import os

load_dotenv()  


class BaseLLM:
    @abstractmethod
    def chat(self,
             query: str,
             history: list,
             system_prompt: str = "",
             trace_id: str = "",
             cache: bool = False,
             stream: bool = False,
             **kwargs) -> Union[str, Generator]:
        pass

    @abstractmethod
    def set_para(self, **kwargs):
        pass


class OpenAILLM(BaseLLM):
    server_source = 'OpenAI'

    def __init__(self,
                 model_name='gpt-4o',
                 temperature=0.3,
                 top_p=1.0,
                 max_token=8192,
                 max_msg_num=20,
                 timeout=60):
        self.set_para(model_name, temperature, top_p, max_token, max_msg_num)
        self.timeout = timeout

    def set_para(self,
                 model_name,
                 temperature,
                 top_p,
                 max_token,
                 max_msg_num):
        self.max_msg_num = max_msg_num
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_token = max_token

        self.token = os.getenv("OpenAILLM_KEY") 
        self.url = "https://api.openai.com/v1/chat/completions"

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }

    def _format_request_data_(self, messages, stream=False, cache=False, **kwargs):
        temperature = self.temperature if not cache else 0

        model_name = kwargs.get("model_name", self.model_name)

        return {
            "model": model_name,
            "temperature": temperature,
            "top_p": 1,
            "max_tokens": self.max_token,
            "messages": messages,
            "stream": stream,
            "use_cache": cache
        }

    def send(self, messages, trace_id='', cache=False, retry=3, **kwargs):
        for _ in range(retry):
            try:
                time0 = time.time()

                data = self._format_request_data_(
                    messages, stream=False, trace_id=trace_id, cache=cache, **kwargs)

                response = requests.post(self.url,
                                         headers=self.headers,
                                         data=json.dumps(data),
                                         verify=True,
                                         timeout=self.timeout)
                
                if response.status_code == 200:
                    resp_json = response.json()
                else:
                    raise Exception(f'status_code={response.status_code}, {response.text}')

                if 'error' in resp_json and 'type' in resp_json['error'] and 'message' in resp_json['error']:
                    msg = resp_json['error']['message']
                    logger.error(f"type={resp_json['error']['type']}, message={msg}")
                    raise Exception(msg)

                time1 = time.time()
                latency = (time1 - time0) * 1_000

                usage = resp_json.get("usage", {})
                answer = resp_json["choices"][0]['message']['content']
                if answer:
                    answer = html.unescape(answer)
                    answer = answer.strip()

                return answer

            except Exception as e:
                logger.error(f"[{self.server_source}] API request error: {e}, retrying...")
                time.sleep(0.5)

        raise TimeoutError(f"# {trace_id} # [{self.server_source}] {kwargs.get('model_name', self.model_name)} request error: Max retry exceeded")

    def send_stream(self, messages, trace_id='', cache=False, retry=3, **kwargs) -> Generator:
        for _ in range(retry):
            try:
                response_stream = self.remote_call_stream(
                        messages, trace_id=trace_id, cache=cache, **kwargs)
                peekable_stream = peekable(response_stream)
                response = self.decode_stream(peekable_stream)

                return response

            except Exception as e:
                logger.error(f"[{self.server_source}] stream request error: {e}, retrying...")
                time.sleep(0.5)

        raise TimeoutError(f"# {trace_id} # [{self.server_source}] {kwargs.get('model_name', self.model_name)} stream request error: Max retry exceeded")

    def remote_call_stream(self, messages, trace_id='', cache=False, **kwargs):

        data = self._format_request_data_(
            messages,
            stream=True,
            trace_id=trace_id,
            cache=cache,
            **kwargs)

        time0 = time.time()
        try:
            response = requests.post(
                    self.url,
                    headers=self.headers,
                    json=data,
                    stream=True)
        except Exception as e:
            logger.error(f"[{self.server_source}] {kwargs.get('model_name', self.model_name)} API request error: {e}")
            raise e
        
        return self._yield_stream(response, trace_id, time0, **kwargs)
    
    def _yield_stream(self, response: Generator, trace_id: str, start_time: float, **kwargs):
        first_token = False
        for line in response.iter_lines():
            if line == b'':
                continue
            if not first_token:
                first_token = True
                latency = (time.time() - start_time) * 1_000
                
            yield line + bytes("\n", encoding="utf-8")

    def decode_stream(self, peekable_stream) -> Generator:
        # handle error
        try:
            line = peekable_stream.peek()
            line_str = line.decode(encoding="utf-8")
        except Exception as e:
            line_str = ""
        if line_str.startswith("{"):
            err_lines = []
            for line in peekable_stream:
                err_lines += [str(line, encoding="utf-8").rstrip("\n")]
            content = "".join(err_lines)
            try:
                resp_json = json.loads(content)
                err = resp_json.get("error", {})
                raise Exception(message=err.get("code", "Unparsable error"), param="param")
            except Exception as e:
                logger.error(f"[{self.server_source}] API error: {e}")
                raise e

        # assume normal message stream with assistant response only
        stream_prefix = "data: "
        prefix_len = len(stream_prefix)
        for line in peekable_stream:
            line_str = line.decode(encoding="utf-8").rstrip("\n")
            if line_str == "":
                continue
            if line_str.startswith(stream_prefix) and line_str != "{}[DONE]".format(stream_prefix):
                try:
                    body = json.loads(line_str[prefix_len:].rstrip("\n"))
                    delta = body.get("choices", [{}])[0].get("delta", {})
                    if delta.get("content", "") != "":
                        yield str(delta["content"])
                except GeneratorExit:
                    return
                except BaseException:
                    logger.error(f"[{self.server_source}] API error: {e}")
                    continue

    def create_messages(self, query, history, instruct=None):
        messages = []
        if instruct:
            messages.append({"role": "system", "content": instruct})

        min_len = min(self.max_msg_num, len(history))
        history = history[-min_len:]
        if history:
            for msg in history:
                if msg.get("role") == "assistant" or msg.get("role") == "user":
                    messages.append(msg)
                else:
                    messages.append({"role": "assistant", "content": msg.get("content")})

        messages.append({"role": "user", "content": query})
        return messages

    def chat(
            self,
            query: str,
            history: list = [],
            system_prompt: str = None,
            trace_id: str = '',
            cache: bool = False,
            stream: bool = False,
            **kwargs) -> Union[str, Generator]:
        messages = self.create_messages(query, history, system_prompt)

        if stream:
            return self.send_stream(messages, trace_id=trace_id, cache=cache, **kwargs)
        else:
            return self.send(messages, trace_id=trace_id, cache=cache, **kwargs)


class AzureLLM(OpenAILLM):
    server_source = 'Azure'

    model_map = {
        "gpt-4o": "gpt-4o",
        "gpt-35-turbo": "gpt-3.5-turbo-0125",
        "gpt-4": "iegg-gpt-4-turbo",
        "gpt-4.1": "gpt-4.1"
    }
    
    def model_mapping(self, model_name):
        return self.model_map.get(model_name, model_name)

    def set_para(self,
                 model_name,
                 temperature,
                 top_p,
                 max_token,
                 max_msg_num,
                 streaming=False):
        self.max_msg_num = max_msg_num
        self.model_name = model_name
        self.engine = self.model_mapping(model_name)
        self.temperature = temperature
        self.top_p = top_p
        self.max_token = max_token
        self.streaming = streaming
        self.version = "2024-02-01"
        self.type = "azure"

        # self.version = '2023-12-01-preview'
        # self.base_url = 'https://iegg-gpt.openai.azure.com/'
        # self.token = os.getenv("AzureLLM_KEY_iegg-gpt")  

        # self.base_url = 'https://iegg-gpt3.openai.azure.com/'
        # self.token = os.getenv("AzureLLM_KEY_iegg-gpt3") 

        # self.version = '2024-03-01-preview'
        # self.base_url = 'https://iegg-gpt-166.openai.azure.com/'
        # self.token = os.getenv("AzureLLM_KEY_iegg-iegg-gpt-166")   

        self.base_url = 'https://iegg-gpt-300.openai.azure.com'
        self.token = os.getenv("AzureLLM_KEY_iegg-gpt-300")    

        self.url = f'{self.base_url}/openai/deployments/{self.engine}/chat/completions?api-version={self.version}'
        self.headers = {
            'Content-Type': 'application/json',
            'api-key': self.token,
        }
    
    def _format_request_data_(self, messages, stream=False, **kwargs):
        model_name = kwargs.get("model_name", self.model_name)

        data = {
            "model": model_name,
            "temperature": self.temperature,
            "top_p": 1,
            "max_tokens": self.max_token,
            "messages": messages,
            "stream": stream,
        }

        return data


class GPTProxyLLM(OpenAILLM):
    server_source = 'GPT Proxy'
    valid_model_names = [
        "gpt-3.5-turbo-0125",
        "gpt-4o",
        "gpt-4o-mini",
        "deepseek-reasoner",
        "deepseek-chat",
        # "gemini-2.5-pro",
        "gemini-2.0-flash",
        "google/gemini-2.0-flash-001",
        # "gemini-2.5-flash",
        "google/gemini-2.5-pro",
        "google/gemini-2.5-flash"
    ]

    def __init__(self,
                 model_name='gpt-4o',
                 temperature=0.3,
                 top_p=1.0,
                 max_token=8192,
                 max_msg_num=20,
                 timeout=120,
                 component_name: str = ""):
        self.set_para(model_name, temperature, top_p, max_token, max_msg_num)
        self.timeout = timeout
        self.component_name = component_name

    def set_para(self,
                 model_name,
                 temperature,
                 top_p,
                 max_token,
                 max_msg_num):
        self.max_msg_num = max_msg_num
        self.temperature = temperature
        self.top_p = top_p
        self.max_token = max_token

        self.url = "http://gptproxy.ai-dev.levelinfinite.com/chat/completions"
        # self.token = os.getenv("GPTProxyLLM_KEY")      
        self.token = "c434373f9a44bc1cf5446177c6a31d9b@pubgm_bot"
        self.model_name = model_name
        if self.model_name not in self.valid_model_names:
            raise ValueError(f"Invalid model name: {self.model_name}")

        self.headers = {
            'Content-Type': 'application/json',
            "Authorization": f"Bearer {self.token}",
        }

    def _format_request_data_(self, messages, stream=False, trace_id='', **kwargs):
        model_name = kwargs.get("model_name", self.model_name)
        if model_name not in self.valid_model_names:
            logger.error(
                f"Invalid model name: {model_name}, using default model: {self.model_name}")
            model_name = self.model_name

        data = {
            "model": model_name,
            "temperature": self.temperature,
            "top_p": 1,
            "max_tokens": self.max_token,
            "request_id": trace_id,
            "messages": messages,
            "stream": stream,
        }

        return data


class DeepSeekLLM(OpenAILLM):
    server_source = 'DeepSeek'

    def __init__(self,
                 temperature=0.3,
                 top_p=1.0,
                 max_token=8192,
                 model_name='deepseek-reasoner', # deepseek-reasoner, deepseek-chat
                 max_msg_num=20):
        self.set_para(temperature, top_p, max_token, model_name, max_msg_num)

    def set_para(self,
                 temperature,
                 top_p,
                 max_token,
                 model_name,
                 max_msg_num):
        self.max_msg_num = max_msg_num
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_token = max_token

        # self.base_url = "http://gptproxy.ai.levelinfinite.com"
        # self.token =  os.getenv("DeepSeekLLMM_KEY-gptproxy")       

        self.base_url = 'https://api.deepseek.com'
        self.token = os.getenv("DeepSeekLLMM_KEY-official") 

        self.url = f'{self.base_url}/chat/completions'
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }

    def __format_request_data_(self, messages, trace_id='', cache=False):
        return {
            "model": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            # "max_tokens": self.max_token,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "request_id": trace_id,
            "use_cache": cache
        }

    def send(self, messages, trace_id='', cache=False, retry=3):
        for _ in range(retry):
            try:
                time0 = time.time()

                data = self.__format_request_data_(
                    messages, trace_id=trace_id, cache=cache)

                response = requests.post(
                    self.url,
                    headers=self.headers,
                    data=json.dumps(data),
                    verify=True,
                    timeout=10000)

                resp_json = response.json()
                if not resp_json:
                    raise Exception("Failed to get valid response from server")

                # logger.info(
                # f'{trace_id}  # {self.model_name} [{self.server_source}] BaseLLM
                # response: {resp_json}')

                if 'error' in resp_json and 'type' in resp_json['error'] and 'message' in resp_json['error']:
                    msg = resp_json['error']['message']
                    logger.error(f"type={resp_json['error']['type']}, message={msg}")
                    raise Exception(msg)

                time1 = time.time()
                latency = (time1 - time0) * 1_000
                logger.info(
                    f'{trace_id}  # {self.model_name} [{self.server_source}] BaseLLM time elapsed: {latency}')

                usage = resp_json.get("usage", {})
                print(f"usage: {usage}")
                reasoning_content = ""
                if self.model_name == "deepseek-reasoner":
                    reasoning_content = resp_json["choices"][0]['message']['reasoning_content']
                answer = resp_json["choices"][0]['message']['content']

                result = ""
                if reasoning_content:
                    reasoning_content = html.unescape(reasoning_content)
                    reasoning_content = reasoning_content.strip()
                    result += reasoning_content + "\n<thinking>\n"

                if answer:
                    answer = html.unescape(answer)
                    result += answer.strip()

                return result

            except Exception as e:
                logger.error(f"[{self.server_source}] API request error: {e}, retrying...")
                time.sleep(1)

        raise TimeoutError("Max retry exceeded")

    def create_messages(self, query, history, instruct=None):
        messages = []
        if instruct:
            messages.append({"role": "system", "content": instruct})

        min_len = min(self.max_msg_num, len(history))
        history = history[-min_len:]
        if history:
            for msg in history:
                if msg.get("role") == "assistant" or msg.get("role") == "user":
                    messages.append(msg)
                else:
                    messages.append({"role": "assistant", "content": msg.get("content")})

        messages.append({"role": "user", "content": query})
        return messages

    def chat(self, query, history=[], system_prompt=None, trace_id='', cache=False):
        messages = self.create_messages(query, history, system_prompt)

        # logger.info(f'OpenaiLLM messages: {messages}')

        return self.send(messages, trace_id=trace_id, cache=cache)


class DeepSeekLLM_11(OpenAILLM):
    server_source = 'DeepSeek'

    def __init__(self,
                 temperature=0.3,
                 top_p=1.0,
                 max_token=8192,
                 model_name='deepseek-r1', # deepseek-r1, deepseek-chat
                 max_msg_num=20):
        self.set_para(temperature, top_p, max_token, model_name, max_msg_num)

    def set_para(self,
                 temperature,
                 top_p,
                 max_token,
                 model_name,
                 max_msg_num):
        self.max_msg_num = max_msg_num
        self.model_name = 'deepseek-r1'
        self.temperature = temperature
        self.top_p = top_p
        self.max_token = max_token

        self.url = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
        self.token = os.getenv("DeepSeekLLMM_KEY-ali") 

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }

    def __format_request_data_(self, messages, trace_id='', cache=False):
        return {
            "model": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            # "max_tokens": self.max_token,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "request_id": trace_id,
            "use_cache": cache
        }

    def send(self, messages, trace_id='', cache=False, retry=3):
        for _ in range(retry):
            try:
                time0 = time.time()

                data = self.__format_request_data_(
                    messages, trace_id=trace_id, cache=cache)

                response = requests.post(
                    self.url,
                    headers=self.headers,
                    data=json.dumps(data),
                    verify=True,
                    timeout=120)

                resp_json = response.json()
                if not resp_json:
                    raise Exception("Failed to get valid response from server")

                # logger.info(
                # f'{trace_id}  # {self.model_name} [{self.server_source}] BaseLLM
                # response: {resp_json}')

                if 'error' in resp_json and 'type' in resp_json['error'] and 'message' in resp_json['error']:
                    msg = resp_json['error']['message']
                    logger.error(f"type={resp_json['error']['type']}, message={msg}")
                    raise Exception(msg)

                time1 = time.time()
                latency = (time1 - time0) * 1_000
                logger.info(
                    f'{trace_id}  # {self.model_name} [{self.server_source}] BaseLLM time elapsed: {latency}')

                usage = resp_json.get("usage", {})
                print(f"usage: {usage}")
                reasoning_content = ""
                if self.model_name == "deepseek-r1":
                    reasoning_content = resp_json["choices"][0]['message']['reasoning_content']
                answer = resp_json["choices"][0]['message']['content']

                result = ""
                if reasoning_content:
                    reasoning_content = html.unescape(reasoning_content)
                    reasoning_content = reasoning_content.strip()
                    result += reasoning_content + "\n<thinking>\n"

                if answer:
                    answer = html.unescape(answer)
                    result += answer.strip()

                return result

            except Exception as e:
                logger.error(f"[{self.server_source}] API request error: {e}, retrying...")
                time.sleep(1)

        raise TimeoutError("Max retry exceeded")

    def create_messages(self, query, history, instruct=None):
        messages = []
        if instruct:
            messages.append({"role": "system", "content": instruct})

        min_len = min(self.max_msg_num, len(history))
        history = history[-min_len:]
        if history:
            for msg in history:
                if msg.get("role") == "assistant" or msg.get("role") == "user":
                    messages.append(msg)
                else:
                    messages.append({"role": "assistant", "content": msg.get("content")})

        messages.append({"role": "user", "content": query})
        return messages

    def chat(self, query, history=[], system_prompt=None, trace_id='', cache=False):
        messages = self.create_messages(query, history, system_prompt)

        # logger.info(f'OpenaiLLM messages: {messages}')

        return self.send(messages, trace_id=trace_id, cache=cache)


class DeepSeekLLM_hk(BaseLLM):
    server_source = 'OpenAI'

    def __init__(self,
                 model_name='gpt-4o',
                 temperature=0.3,
                 top_p=1.0,
                 max_token=8192,
                 max_msg_num=20,
                 timeout=60):
        self.set_para(model_name, temperature, top_p, max_token, max_msg_num)
        self.timeout = timeout

    def set_para(self,
                 model_name,
                 temperature,
                 top_p,
                 max_token,
                 max_msg_num):
        self.max_msg_num = max_msg_num
        self.model_name = 'gpt-4o'
        self.temperature = temperature
        self.top_p = top_p
        self.max_token = max_token

        self.token = os.getenv("OpenAILLM_KEY_hekai") 
        self.url = "https://api.openai.com/v1/chat/completions"

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }

    def _format_request_data_(self, messages, stream=False, cache=False, **kwargs):
        temperature = self.temperature if not cache else 0

        model_name = kwargs.get("model_name", self.model_name)

        return {
            "model": model_name,
            "temperature": temperature,
            "top_p": 1,
            "max_tokens": self.max_token,
            "messages": messages,
            "stream": stream,
        }

    def send(self, messages, trace_id='', cache=False, retry=3, **kwargs):
        for _ in range(retry):
            try:
                time0 = time.time()

                data = self._format_request_data_(
                    messages, stream=False, trace_id=trace_id, cache=cache, **kwargs)

                response = requests.post(self.url,
                                         headers=self.headers,
                                         data=json.dumps(data),
                                         verify=True,
                                         timeout=self.timeout)
                
                if response.status_code == 200:
                    resp_json = response.json()
                else:
                    raise Exception(f'status_code={response.status_code}, {response.text}')

                if 'error' in resp_json and 'type' in resp_json['error'] and 'message' in resp_json['error']:
                    msg = resp_json['error']['message']
                    logger.error(f"type={resp_json['error']['type']}, message={msg}")
                    raise Exception(msg)

                time1 = time.time()
                latency = (time1 - time0) * 1_000

                usage = resp_json.get("usage", {})
                answer = resp_json["choices"][0]['message']['content']
                if answer:
                    answer = html.unescape(answer)
                    answer = answer.strip()

                return answer

            except Exception as e:
                logger.error(f"[{self.server_source}] API request error: {e}, retrying...")
                time.sleep(0.5)

        raise TimeoutError(f"# {trace_id} # [{self.server_source}] {kwargs.get('model_name', self.model_name)} request error: Max retry exceeded")

    def send_stream(self, messages, trace_id='', cache=False, retry=3, **kwargs) -> Generator:
        for _ in range(retry):
            try:
                response_stream = self.remote_call_stream(
                        messages, trace_id=trace_id, cache=cache, **kwargs)
                peekable_stream = peekable(response_stream)
                response = self.decode_stream(peekable_stream)

                return response

            except Exception as e:
                logger.error(f"[{self.server_source}] stream request error: {e}, retrying...")
                time.sleep(0.5)

        raise TimeoutError(f"# {trace_id} # [{self.server_source}] {kwargs.get('model_name', self.model_name)} stream request error: Max retry exceeded")

    def remote_call_stream(self, messages, trace_id='', cache=False, **kwargs):

        data = self._format_request_data_(
            messages,
            stream=True,
            trace_id=trace_id,
            cache=cache,
            **kwargs)

        time0 = time.time()
        try:
            response = requests.post(
                    self.url,
                    headers=self.headers,
                    json=data,
                    stream=True)
        except Exception as e:
            logger.error(f"[{self.server_source}] {kwargs.get('model_name', self.model_name)} API request error: {e}")
            raise e
        
        return self._yield_stream(response, trace_id, time0, **kwargs)
    
    def _yield_stream(self, response: Generator, trace_id: str, start_time: float, **kwargs):
        first_token = False
        for line in response.iter_lines():
            if line == b'':
                continue
            if not first_token:
                first_token = True
                latency = (time.time() - start_time) * 1_000
                
            yield line + bytes("\n", encoding="utf-8")

    def decode_stream(self, peekable_stream) -> Generator:
        # handle error
        try:
            line = peekable_stream.peek()
            line_str = line.decode(encoding="utf-8")
        except Exception as e:
            line_str = ""
        if line_str.startswith("{"):
            err_lines = []
            for line in peekable_stream:
                err_lines += [str(line, encoding="utf-8").rstrip("\n")]
            content = "".join(err_lines)
            try:
                resp_json = json.loads(content)
                err = resp_json.get("error", {})
                raise Exception(message=err.get("code", "Unparsable error"), param="param")
            except Exception as e:
                logger.error(f"[{self.server_source}] API error: {e}")
                raise e

        # assume normal message stream with assistant response only
        stream_prefix = "data: "
        prefix_len = len(stream_prefix)
        for line in peekable_stream:
            line_str = line.decode(encoding="utf-8").rstrip("\n")
            if line_str == "":
                continue
            if line_str.startswith(stream_prefix) and line_str != "{}[DONE]".format(stream_prefix):
                try:
                    body = json.loads(line_str[prefix_len:].rstrip("\n"))
                    delta = body.get("choices", [{}])[0].get("delta", {})
                    if delta.get("content", "") != "":
                        yield str(delta["content"])
                except GeneratorExit:
                    return
                except BaseException:
                    logger.error(f"[{self.server_source}] API error: {e}")
                    continue

    def create_messages(self, query, history, instruct=None):
        messages = []
        if instruct:
            messages.append({"role": "system", "content": instruct})

        min_len = min(self.max_msg_num, len(history))
        history = history[-min_len:]
        if history:
            for msg in history:
                if msg.get("role") == "assistant" or msg.get("role") == "user":
                    messages.append(msg)
                else:
                    messages.append({"role": "assistant", "content": msg.get("content")})

        messages.append({"role": "user", "content": query})
        return messages

    def chat(
            self,
            query: str,
            history: list = [],
            system_prompt: str = None,
            trace_id: str = '',
            cache: bool = False,
            stream: bool = False,
            **kwargs) -> Union[str, Generator]:
        messages = self.create_messages(query, history, system_prompt)

        if stream:
            return self.send_stream(messages, trace_id=trace_id, cache=cache, **kwargs)
        else:
            return self.send(messages, trace_id=trace_id, cache=cache, **kwargs)



def random_string(length):
    import random
    import string
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


if __name__ == "__main__":
    llm = DeepSeekLLM()
    res = llm.chat("Hello", history=[{"role": "user", "content": "Hello"}])
    print(res)
    
    # llm = AzureLLM()
    # llm.chat("Hello", history=[{"role": "user", "content": "Hello"}])

    # llm = DeepSeekLLM()
    # trace_id = "run_role-" + random_string(8)
    # print(trace_id)
    # print(llm.chat("Hello, how to be a better worker?", history=[], trace_id=""))


# if __name__ == "__main__":
#     llm = DeepSeekLLM()
#     trace_id = "run_role-" + random_string(8)
#     print(trace_id)
    
#     # 在查询中直接要求 JSON 格式
#     query = """Please provide advice on how to be a better worker. 
#     Respond in JSON format with the following structure:
#     {
#         "advice": "your main advice",
#         "tips": ["tip1", "tip2", "tip3"]
#     }"""
    
#     result = llm.chat(
#         query=query, 
#         history=[], 
#         trace_id=trace_id
#     )
    
#     print("Response:")
#     print(result)
