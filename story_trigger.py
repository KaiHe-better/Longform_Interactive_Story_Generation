import json
from llm import GPTProxyLLM, DeepSeekLLM, AzureLLM, DeepSeekLLM_hk
from utils import parse_json_from_string, remove_first_sentence
import random
from template_factory import get_template, get_query_template, ID2LANG


class StoryTrigger():
    def __init__(self, language='en'):
        self.language = language
        # self.llm = GPTProxyLLM(model_name='deepseek-chat', temperature=0.0)
        # self.llm = DeepSeekLLM(model_name='deepseek-chat', temperature=0.0)
        # self.llm = DeepSeekLLM_hk(model_name='deepseek-chat', temperature=0.9)

        self.template = get_template(language, "trigger")
        self.query_template = get_query_template(language, "trigger")

    def __format_template__(self, episode_setting):
        episode_id = episode_setting.get('episode_id', '')

        if self.language == 'zh':
            triggers = ""
            # for item in episode_setting.get('triggers', []):
            #     condition, gen_polt = item['condition'].split('<THEN>')
            #     condition = condition.replace('<IF>', '').replace('<THEN>', '')
            #     gen_polt = gen_polt.replace('<IF>', '').replace('<THEN>', '')
            #     triggers += f"- 当且仅当“{condition}”和“{gen_polt}”的故事情节内容同时满足，则设置next_episode = {item['next_episode']}\n"
            for item in episode_setting.get('triggers', []):
                conditions = item['condition'].split('<TRIGGER_CONDITION>')
                conditions = [condition.strip() for condition in conditions if condition.strip()]
                conditions = [f"“{condition}”" for condition in conditions if condition.strip()]
                conditions = "和".join(conditions)
                triggers += f"- 当且仅当生成的故事内容同时满足<condition>{conditions}</condition>的故事情节时，则设置next_episode = {item['next_episode']}\n"
            
            triggers = triggers.strip()
        else:
            triggers = ""
            # for item in episode_setting.get('triggers', []):
            #     condition, gen_polt = item['condition'].split('<THEN>')
            #     condition = condition.replace('<IF>', '').replace('<THEN>', '')
            #     gen_polt = gen_polt.replace('<IF>', '').replace('<THEN>', '')
            #     triggers += f"- If and only if the generated story plot includes content: “{condition}” and “{gen_polt}”, then set next_episode = {item['next_episode']}\n"
            for item in episode_setting.get('triggers', []):
                conditions = item['condition'].split('<TRIGGER_CONDITION>')
                conditions = [condition.strip() for condition in conditions if condition.strip()]
                conditions = [f"“{condition}”" for condition in conditions if condition.strip()]
                conditions = " and ".join(conditions)
                triggers += f"- If and only if the generated story plot includes content: <condition>{conditions}</condition>, then set next_episode = {item['next_episode']}\n"
            
            triggers = triggers.strip()
        
        template = self.template
        template = template.replace('{episode_id}', str(episode_id))
        template = template.replace('{triggers}', triggers)
        return template
    
    def __format_plot__(self, plot_list):
        plots = ""

        for plot in plot_list:
            if "narrative" in plot and plot["narrative"]:
                plots += f"{plot['narrative']}\n"
            if "role_dialogue" in plot and plot["role_dialogue"]:
                plots += f"{plot['role_dialogue']['name']}: {plot['role_dialogue']['utterance']}\n"
        return plots.strip()

    def __format_query__(self, history, user_input, generated_story, story_setting):
        leading_role = story_setting.get('leading_role', '')
        history_list = history.split('\n')
        story_plot = ""
        for hist in history_list:
            hist = str(hist)
            if not hist:
                continue
            if self.language == "zh":
                # 查看hist是否包含“第 {i} 轮”， i 为数字
                if hist.startswith("第 "): continue
                if hist.startswith("情节"): continue
            else:
                if hist.startswith("Round"): continue
                if hist.startswith("Plot"): continue
            story_plot += f"{hist}\n"
        
        story_plot += f"{leading_role}: {user_input}\n"
        story_plot +=  self.__format_plot__(generated_story)
        query = self.query_template.format(
            story=story_plot
        )
        return query

    def generate(self, story_setting, episode_setting, history, user_input, generated_story):
        system_prompt = self.__format_template__(episode_setting)
        
        story_plot = self.__format_query__(history, user_input, generated_story, story_setting)
        # print(f"***** Story Trigger *****")
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
                print(resp_json)
                if 'next_episode' in resp_json:
                    return resp_json['next_episode']
                else:
                    raise Exception("No next_episode")
            except Exception as e:
                print(f"Story Trigger Error: {e}")
        return ''
