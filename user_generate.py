import json
from llm import GPTProxyLLM, DeepSeekLLM, AzureLLM, DeepSeekLLM_hk
from utils import parse_json_from_string, remove_first_sentence
import random
from template_factory import get_template, get_query_template, get_user_actions, ID2LANG


class UserGenerator():
    def __init__(self, language='en'):
        self.language = language
        # self.llm = GPTProxyLLM(model_name='gpt-4o', temperature=0.9)
        # self.llm = DeepSeekLLM(model_name='deepseek-chat', temperature=0.9)
        # self.llm = DeepSeekLLM_hk(model_name='deepseek-chat', temperature=0.9)

        self.template = {
            1: get_template(language, "user_normal"),
            0: get_template(language, "user_abnormal")
        }
        self.query_template = get_query_template(language, "user")
        self.probability_list = [0.0, 0.0, 1.0]

    def __format_template__(self, story_setting, episode_setting, template_id=1):
        story_name = story_setting.get('story_name', '')
        story_style = ', '.join(story_setting.get('story_style', []))
        story_desc = story_setting.get('story_desc', '')
        leading_role = story_setting.get('leading_role', '')
        role_desc = story_setting.get('story_chars', {}).get(leading_role, '')
        episode_desc = episode_setting.get('episode_desc', '')
        episode_goal = episode_setting.get('episode_goal', '')
        episode_scene = episode_setting.get('episode_scene', '')
        history = episode_setting.get('history', "")

        template = self.template.get(template_id, get_template(self.language, "user_normal"))
        template = template.replace('{story_name}', story_name)
        template = template.replace('{story_style}', story_style)
        template = template.replace('{story_desc}', story_desc)
        template = template.replace('{leading_role}', leading_role)
        template = template.replace('{role_desc}', role_desc)
        template = template.replace('{episode_desc}', episode_desc)
        template = template.replace('{episode_goal}', episode_goal)
        template = template.replace('{episode_scene}', episode_scene)
        template = template.replace('{history}', history)
        template = template.replace('{language}', ID2LANG[self.language])
        return template

    def __format_query__(self, current_episode):
        query = self.query_template.format(
            story=current_episode
        )
        return query

    def generate(self, story_setting: dict, episode_setting: dict, current_plot: str):
        if not current_plot:
            episode_desc = episode_setting.get('episode_desc', '')
            role_name = story_setting.get('leading_role', '')
            nike_name = story_setting.get('nike_name', '')
            # 找到第一句话
            if self.language == 'zh':
                first_sentence = episode_desc.split('。')[0]
                action = first_sentence.replace(role_name, '我')
                action = action.replace(nike_name, '我') if nike_name !="" else action
            else:
                first_sentence = episode_desc.split('.')[0]
                action = first_sentence.replace(role_name, 'I')
                action = action.replace(nike_name, 'I') if nike_name !="" else action
            return {
                "type": "normal",
                "action": action
            }
        random_seed = random.random()
        if random_seed < self.probability_list[0]:
            return {
                "type": "hacking",
                "action": get_user_actions(self.language)
            }
        elif random_seed < self.probability_list[1]:
            template_id = 0
        else:
            template_id = 1
        system_prompt = self.__format_template__(
            story_setting, episode_setting, template_id=template_id)
        current_plot = self.__format_query__(current_plot)
        # print(f"***** User Generation *****")
        # print(f"system_prompt: {system_prompt}")
        # print(f"plot: {current_plot}")
        # print(f"****************************")

        for _ in range(3):
            try:
                response = self.llm.chat(current_plot, [], system_prompt)
                # if isinstance(self.llm, DeepSeekLLM) and self.llm.model_name == "deepseek-reasoner":
                if isinstance(self.llm, DeepSeekLLM) and  "deepseek" in self.llm.model_name :
                    thinking = response.split('\n<thinking>\n')[0]
                    answer = response.split('\n<thinking>\n')[1]
                    resp_json = parse_json_from_string(answer)
                    if not resp_json:
                        raise Exception("No response")
                    return {
                        "type": "abnormal" if template_id == 0 else "normal",
                        "thinking": remove_first_sentence(thinking, self.language),
                        "action": resp_json.get('action', ''),
                    }
                else:
                    resp_json = parse_json_from_string(response)
                    if not resp_json:
                        raise Exception("No response")
                    return {
                        "type": "abnormal" if template_id == 0 else "normal",
                        "action": resp_json.get('action', ''),
                    }
            except Exception as e:
                print(f"User Generation Error: {e}")
        return {}
