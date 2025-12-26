import jsonlines
import json
from copy import deepcopy
from colorama import Fore, Style, init

from generate import Generator
from refine import Refiner
from regenerate import Regenerator
from utils import parse_json_from_string, PRINT_COLOR_LIST
from my_threading import ContextThreadExecutor
from template_factory import ID2LANG

# 初始化 colorama
init()


class StoryGenerator():
    def __init__(self, story_file, story_id, output_file="", language="en", concurrent_num=5):
        self.story, self.plot_id = self.__load_story_files(story_file, story_id)
        self.language = ID2LANG.get(language, "English")
        self.generator = Generator(self.language)
        self.refiner = Refiner(self.language)
        self.regenerator = Regenerator(self.language)
        self.print_story_desc()

        self.output_file = output_file
        self.concurrent_num = concurrent_num

    def __load_story_files(self, story_file, story_id):
        with jsonlines.open(story_file) as reader:
            story_info = None
            for obj in reader:
                if obj['Settings']['story_id'] == story_id:
                    story_info = obj
                    break
            if not story_info:
                raise ValueError(f"Story with id {story_id} not found in {story_file}")

            story_setting = story_info.get('Settings', {})
            episodes = story_info.get('Episodes', [])
            story_episodes = {}
            for obj in episodes:
                story_episodes[obj['plot_id']] = obj
            return {
                'Settings': story_setting,
                'Episodes': story_episodes
            }, 1

    def print_story_desc(self):
        print('==' * 20)
        print("Story Name:", self.story['Settings']['story_name'])
        print("Story Style:", ', '.join(self.story['Settings']['story_style']))
        print("Story Description:", self.story['Settings']['story_desc'])
        print(self.story['Settings']['story_target'])
        print(
            "Player will play the role of:",
            f"{self.story['Settings']['leading_role']}, ({self.story['Settings']['story_chars'][self.story['Settings']['leading_role']]})")
        print('==' * 20)

    def sync_output(self, story, plot, user_input, response):
        output = {
            'Settings': story['Settings'],
            'Episodes': []
        }
        message_pairs = {
            "user": user_input,
            "bot": response
        }
        plot['message_pairs'].append(message_pairs)
        output['Episodes'].append(plot)
        if self.output_file:
            with jsonlines.open(self.output_file, 'a') as writer:
                writer.write(output)

    # 对hostory进行递归总结
    def recursive_summary(self, history):
        if len(history) == 0:
            return ''
        if len(history) == 1:
            return history[0]
        return history[0] + self.recursive_summary(history[1:])

    def _generate(self, user_input, story, plot, history):
        generated_story = self.generator.generate(user_input, story, plot, history)
        print(f"Generated Story:\n{json.dumps(generated_story, indent=4, ensure_ascii=False)}\n")
        review = self.refiner.generate(
            user_input, story, plot, json.dumps(
                generated_story, indent=4, ensure_ascii=False))
        print(f"Review:\n{json.dumps(review, indent=4, ensure_ascii=False)}\n")
        new_story = self.regenerator.generate(
            user_input, story, plot, history, json.dumps(
                generated_story, indent=4, ensure_ascii=False), json.dumps(
                review, indent=4, ensure_ascii=False))
        print(f"New Story:\n{json.dumps(new_story, indent=4, ensure_ascii=False)}\n")
        return new_story

    def _print(self, content, color=Fore.BLACK):
        # BLACK           = 30
        # RED             = 31
        # GREEN           = 32
        # YELLOW          = 33
        # BLUE            = 34
        # MAGENTA         = 35
        # CYAN            = 36
        # WHITE           = 37
        # RESET           = 39
        print(f"{color}{content}{Style.RESET_ALL}")

    def generate(self, user_input, history):
        plot = self.story['Episodes'].get(self.plot_id, {})

        item_results = {}
        with ContextThreadExecutor() as executor:
            for i in range(self.concurrent_num):
                item_results[i] = executor.submit(self._generate,
                                                  user_input=user_input,
                                                  story=self.story,
                                                  plot=plot,
                                                  history=history)

        for i in range(len(item_results)):
            cur_result = item_results[i].result()
            print_content = f"Threading {i}:\n{json.dumps(cur_result, ensure_ascii=False, indent=4)}\n"
            self._print(print_content, PRINT_COLOR_LIST[i % len(PRINT_COLOR_LIST)])

        user_choose = input("Choose Best Answer: ")

        while not user_choose.isdigit() or int(user_choose) < 0 and int(user_choose) >= len(item_results) - 1:
            user_choose = input(f"Choose Best Answer: (Enter number 0～{len(item_results)-1})")
        final_result = item_results[int(user_choose)].result()

        if final_result['next_plot']:
            self.plot_id = final_result['next_plot']

        print_content = f"The Best Answer:\n{final_result}\n"
        self._print(print_content, Fore.RED)

        self.sync_output(self.story, plot, user_input, final_result)
        return json.dumps(final_result, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    story_file = 'data/game_gen_11_23_SM.jsonl'
    output_file = 'data/game_gen_11_23_SM_output.jsonl'
    story_id = '1'
    sg = StoryGenerator(story_file, story_id, output_file, language="zh", concurrent_num=1)
    history = []
    while True:
        user_input = input("User: ")
        if user_input == 'exit':
            break
        resp = sg.generate(user_input, history)
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": resp})
