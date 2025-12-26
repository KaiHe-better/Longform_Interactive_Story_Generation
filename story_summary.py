import json
from llm import GPTProxyLLM, DeepSeekLLM, AzureLLM, DeepSeekLLM_hk
from utils import parse_json_from_string, remove_first_sentence
import random
from template_factory import get_template, get_query_template, ID2LANG


class StorySummary():
    def __init__(self, language='en'):
        self.language = language
        # self.llm = GPTProxyLLM(model_name='gpt-4o', temperature=0.)
        # self.llm = DeepSeekLLM(model_name='deepseek-chat', temperature=0.0)
        # self.llm = DeepSeekLLM_hk(model_name='deepseek-chat', temperature=0.9)

        self.template = get_template(language, "summary")
        self.query_template = get_query_template(language, "summary")

    def __format_template__(self):
        template = self.template.replace('{language}', ID2LANG[self.language])
        return template

    def __format_query__(self, history):
        query = self.query_template.format(
            story=history
        )
        return query

    def generate(self, history):
        system_prompt = self.__format_template__()
        story_plot = self.__format_query__(history)
        # print(f"***** Story Summary *****")
        # print(f"system_prompt: {system_prompt}")
        # print(f"story_plot: {story_plot}")
        # print(f"****************************")
        for _ in range(3):
            try:
                response = self.llm.chat(story_plot, [], system_prompt)
                # if isinstance(self.llm, DeepSeekLLM) and self.llm.model_name == "deepseek-reasoner":
                if isinstance(self.llm, DeepSeekLLM) and  "deepseek" in self.llm.model_name :
                    thinking = response.split('\n<thinking>\n')[0]
                    answer = response.split('\n<thinking>\n')[1]
                    resp_json = parse_json_from_string(answer)
                    if not resp_json:
                        raise Exception("No response")
                    resp_json['thinking'] = remove_first_sentence(thinking, self.language)
                else:
                    resp_json = parse_json_from_string(response)
                    if not resp_json:
                        raise Exception("No response")
                if 'summary' in resp_json:
                    return resp_json['summary']
                else:
                    raise Exception("No summary")
            except Exception as e:
                print(f"Story Summary Error: {e}")
        return ''
