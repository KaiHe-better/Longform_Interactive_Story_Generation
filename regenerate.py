from llm import GPTProxyLLM, DeepSeekLLM, AzureLLM, DeepSeekLLM_hk
from utils import parse_json_from_string, remove_first_sentence
from template_factory import get_template, get_query_template, ID2LANG


class Regenerator():
    def __init__(self, language='en'):
        self.language = language
        # self.llm = DeepSeekLLM(temperature=1.5)
        # self.llm = AzureLLM(model_name='gpt-4o', temperature=0.0)
        # self.llm = GPTProxyLLM(model_name='deepseek-chat', temperature=0.9)
        # self.llm = DeepSeekLLM(model_name='deepseek-chat', temperature=0.9)
        # self.llm = DeepSeekLLM_hk(model_name='deepseek-chat', temperature=0.9)

        self.template = get_template(language, "regen")
        self.query_template = get_query_template(language, "regen")

    def __format_template__(self, story_setting, episode_setting):
        story_name = story_setting.get('story_name', '')
        story_style = ', '.join(story_setting.get('story_style', []))
        story_desc = story_setting.get('story_desc', '')
        story_chars = '\n'.join(
            [f"- {k}: {v}" for k, v in story_setting.get('story_chars', {}).items()])
        leading_role = story_setting.get('leading_role', '')
        story_goal = story_setting.get('story_goal', '')
        episode_id = episode_setting.get('episode_id', '')
        episode_desc = episode_setting.get('episode_desc', '')
        episode_chars = '\n'.join(
            [f"- {v}" for v in episode_setting.get('episode_chars', [])])
        episode_goal = episode_setting.get('episode_goal', '')
        episode_scene = episode_setting.get('episode_scene', '')
        
        if self.language == 'zh':
            triggers = ""
            special_requirements = ""
            # for item in episode_setting.get('triggers', []):
            #     condition, gen_polt = item['condition'].split('<THEN>')
            #     condition = condition.replace('<IF>', '').replace('<THEN>', '')
            #     gen_polt = gen_polt.replace('<IF>', '').replace('<THEN>', '')
            #     special_requirements += f"- 生成故事内容时，只要生成了“{condition}”的情节，则必须跟着生成以下情节：“{gen_polt}”\n"
            #     triggers += f"- 当且仅当“{condition}”和“{gen_polt}”的故事情节内容同时满足，则设置next_episode = {item['next_episode']}\n"
            for item in episode_setting.get('triggers', []):
                conditions = item['condition'].split('<TRIGGER_CONDITION>')
                conditions = [condition.strip() for condition in conditions if condition.strip()]
                conditions = [f"“{condition}”" for condition in conditions if condition.strip()]
                conditions = "和".join(conditions)
                triggers += f"- 当且仅当生成的故事内容同时满足<condition>{conditions}</condition>的故事情节时，则设置next_episode = {item['next_episode']}\n"
            
            special_requirements = special_requirements.strip()
            triggers = triggers.strip()
        else:
            triggers = ""
            special_requirements = ""
            # for item in episode_setting.get('triggers', []):
            #     condition, gen_polt = item['condition'].split('<THEN>')
            #     condition = condition.replace('<IF>', '').replace('<THEN>', '')
            #     gen_polt = gen_polt.replace('<IF>', '').replace('<THEN>', '')
            #     special_requirements += f"- When generating story content, if the plot “{condition}” is generated, then the following plot must be generated next: “{gen_polt}”\n"
            #     triggers += f"- If and only if the generated story plot includes content: “{condition}” and “{gen_polt}”, then set next_episode = {item['next_episode']}\n"
            for item in episode_setting.get('triggers', []):
                conditions = item['condition'].split('<TRIGGER_CONDITION>')
                conditions = [condition.strip() for condition in conditions if condition.strip()]
                conditions = [f"“{condition}”" for condition in conditions if condition.strip()]
                conditions = " and ".join(conditions)
                triggers += f"- If and only if the generated story plot includes content: <condition>{conditions}</condition>, then set next_episode = {item['next_episode']}\n"
            
            special_requirements = special_requirements.strip()
            triggers = triggers.strip()
            
        history = episode_setting.get('history', "")
        template = self.template.replace('{story_name}', story_name)
        template = template.replace('{story_style}', story_style)
        template = template.replace('{story_desc}', story_desc)
        template = template.replace('{story_chars}', story_chars)
        template = template.replace('{leading_role}', leading_role)
        template = template.replace('{story_goal}', story_goal)
        template = template.replace('{episode_id}', str(episode_id))
        template = template.replace('{episode_desc}', episode_desc)
        template = template.replace('{episode_chars}', episode_chars)
        template = template.replace('{episode_goal}', episode_goal)
        template = template.replace('{episode_scene}', episode_scene)
        template = template.replace('{special_requirements}', special_requirements)
        template = template.replace('{triggers}', triggers)
        template = template.replace('{history}', history)
        template = template.replace('{language}', ID2LANG[self.language])
        return template

    def __format_query__(self, user_input, generated_story, review):
        query = self.query_template.format(
            user_input=user_input,
            generated_story=generated_story,
            review=review
        )
        return query

    def generate(
            self,
            story_setting: dict,
            episode_setting: dict,
            user_input: str,
            generated_story: str,
            review: str):

        system_prompt = self.__format_template__(story_setting, episode_setting)
        query = self.__format_query__(user_input, generated_story, review)
        # print(f"**** Story Re-Generation ****")
        # print(f"system_prompt: {system_prompt}")
        # print(f"story and review: {query}")
        # print(f"****************************")

        for _ in range(3):
            try:
                response = self.llm.chat(query, [], system_prompt)
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
                return resp_json
            except Exception as e:
                print(f"Regenerate Error: {e}")
        return {}
