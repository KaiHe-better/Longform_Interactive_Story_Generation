import jsonlines
import json
import os
from copy import deepcopy
from colorama import Fore, Style, init

from copy import deepcopy
from user_generate import UserGenerator
from generate import Generator
from refine import Refiner
from regenerate import Regenerator
from story_summary import StorySummary
from story_trigger import StoryTrigger
from utils import parse_json_from_string, PRINT_COLOR_LIST
from my_threading import ContextThreadExecutor
from template_factory import ID2LANG, LANG2ID
import random

# 初始化 colorama
init()


class StoryOrganizer():

    def __init__(self, story_info):
        self.story_id = story_info.get('Settings', {}).get('story_id', "")
        if not self.story_id:
            self.story_id = str(random.randint(100000, 999999))
        self.story = self._load_story_(story_info)
        self.episode_lines = self.organize_episode_lines(self.story)

    def _load_story_(self, story_info):
        if not story_info:
            raise ValueError(f"Story not found with story_id: {self.story_id}")

        story_setting = story_info.get('Settings', {})
        episodes = story_info.get('Episodes', [])

        if not story_setting:
            raise ValueError(f"Story setting not found with story_id: {self.story_id}")
        if not episodes:
            raise ValueError(f"Episodes not found with story_id: {self.story_id}")

        if story_setting.get("leading_role", "") not in story_setting.get("story_chars", {}):
            for name, char in story_setting.get("story_chars", {}).items():
                if story_setting.get("leading_role", "") in name:
                    story_setting["nike_name"] = story_setting["leading_role"]
                    story_setting["leading_role"] = name
                    break
            if story_setting.get("leading_role", "") not in story_setting.get("story_chars", {}):
                raise ValueError(f"Leading role not found with story_id: {self.story_id}")

        if isinstance(story_setting["states"], dict):
            # 将story_setting中的story_setting["states"]中所有value["seq"]转换为字符串
            story_setting["states"] = [[str(state) for state in value["seq"]]
                                       for value in story_setting["states"].values()]
        elif isinstance(story_setting["states"], list):
            story_setting["states"] = [[str(state) for state in states]
                                       for states in story_setting["states"]]
        else:
            raise ValueError(f"Invalid states format with story_id: {self.story_id}")

        story_episodes = {}
        for obj in episodes:
            obj['episode_id'] = str(obj['episode_id'])
            if isinstance(obj['pre_episode_id'], list):
                obj['pre_episode_id'] = [str(pre_episode_id)
                                         for pre_episode_id in obj['pre_episode_id']]
            else:
                obj['pre_episode_id'] = [str(obj['pre_episode_id'])]
            # triggers中的next_episode转换为字符串
            for trigger in obj['triggers']:
                trigger['next_episode'] = str(trigger['next_episode'])

            story_episodes[obj['episode_id']] = obj

        return {
            'Settings': story_setting,
            'Episodes': story_episodes
        }

    def organize_episode_lines(self, story):
        episodes = deepcopy(story['Episodes'])
        episode_lines = []
        for states in story['Settings']['states']:
            episode_line = {"start_episode_id": "1"}
            episode_line['states'] = states
            for i, state in enumerate(states[:-1]):
                episode = deepcopy(episodes[state])
                episode['triggers'] = [trigger for trigger in episode['triggers']
                                       if trigger.get('next_episode', "") == states[i + 1]]
                if i == 0:
                    episode_line['start_episode_id'] = episode['episode_id']
                episode_line[episode['episode_id']] = episode
            episode_line[states[-1]] = deepcopy(episodes[states[-1]])
            episode_lines.append(episode_line)
        return episode_lines


class HistoryManager():
    def __init__(self):
        self.summary = ""
        self.history = []
        self.last_user_input = ""
        self.last_story_plot = []

    def append(self, role, content):
        assert role in ["user", "plot", "summary"], "Role must be 'user', 'plot' or 'summary'."
        if role == "summary":
            self.summary = content
        else:
            self.history.append({"role": role,
                                "content": content})
            if role == "user":
                self.last_user_input = content["action"]
            elif role == "plot":
                self.last_story_plot = content["plot_list"]

    def _get_summary(self):
        return self.summary

    def _format_plot(self, plot_list):
        plots = ""

        for plot in plot_list:
            if "narrative" in plot and plot["narrative"]:
                plots += f"{plot['narrative']}\n"
            if "role_dialogue" in plot and plot["role_dialogue"]:
                plots += f"{plot['role_dialogue']['name']}: {plot['role_dialogue']['utterance']}\n"
        return plots.strip()

    def get(self, role_name="", language="en"):
        history = ""
        i = 0
        if self.summary:
            i += 1
            if language == "zh":
                history += f"第 {i} 轮\n"
            else:
                history += f"Round {i}\n"
            history += f"Plot: {self._get_summary()}\n\n"

        for hist in self.history:
            if hist.get("role", "") == "user":
                i += 1
                if language == "zh":
                    history += f"第 {i} 轮\n"
                else:
                    history += f"Round {i}\n"

                history += f"{role_name}: {hist['content']['action']}\n"
            elif hist.get("role", "") == "plot":
                if language == "zh":
                    if i == 0:
                        i += 1
                        history += f"第 {i} 轮\n"
                    history += f"情节:\n{self._format_plot(hist['content']['plot_list'])}\n\n"
                else:
                    if i == 0:
                        i += 1
                        history += f"Round {i}\n"
                    history += f"Plot:\n{self._format_plot(hist['content']['plot_list'])}\n\n"
        return history, i

    def get_last_story_plot(self):
        if self.last_story_plot:
            return self._format_plot(self.last_story_plot)
        return ""

    def get_last_user_input(self):
        if self.last_user_input:
            return self.last_user_input
        return ""

    def load(self, episode):
        if 'prequel_episode_summary' in episode and episode['prequel_episode_summary']:
            self.append("summary", episode['prequel_episode_summary'])
        if 'message_pairs' in episode and episode['message_pairs']:
            for message_pair in episode['message_pairs']:
                if message_pair['role'] == "user":
                    self.append("user", message_pair['content'])
                elif message_pair['role'] == "assistant":
                    self.append("plot", message_pair['content'])

    def clear_history(self):
        self.history = []

    def clear_summary(self):
        self.summary = ""

    def clear(self):
        self.clear_history()
        self.clear_summary()

    def __len__(self):
        return len(self.history)

    def __getitem__(self, index):
        return self.history[index]


class StoryGenerator():
    def __init__(
            self,
            story_info,
            story_output_dir="",
            refine_output_dir="",
            language="en"):
        self.story_organizer = StoryOrganizer(story_info)
        self.user_generator = UserGenerator(language)
        self.generator = Generator(language)
        self.refiner = Refiner(language)
        self.regenerator = Regenerator(language)
        self.story_summary = StorySummary(language)
        self.story_trigger = StoryTrigger(language)
        self.history_managers = {}
        self.print_story_desc()

        self.story_output_dir = story_output_dir
        os.makedirs(self.story_output_dir, exist_ok=True)
        os.makedirs(refine_output_dir, exist_ok=True)
        self.refine_output_file = os.path.join(refine_output_dir, f"{language}.jsonl")
        self.language = language

    def set_story_organizer(self, story_info):
        self.story_organizer = StoryOrganizer(story_info)

    def print_story_desc(self):
        print('==' * 20)
        print("Story Name:", self.story_organizer.story['Settings']['story_name'])
        print("Story Style:", ', '.join(self.story_organizer.story['Settings']['story_style']))
        print("Story Description:", self.story_organizer.story['Settings']['story_desc'])
        print(self.story_organizer.story['Settings']['story_goal'])
        print(
            "Player will play the role of:", 
            f"{self.story_organizer.story['Settings']['leading_role']}, ({self.story_organizer.story['Settings']['story_chars'][self.story_organizer.story['Settings']['leading_role']]})")
        print('==' * 20)

    def sync_story_output(self, episode_setting, episode_id, user_output, story_output, series_id=""):
        print(f"Sync story output:  {series_id}, {episode_id}")
        data = {}
        # 读取json文件
        output_file = os.path.join(self.story_output_dir,
                                   f"story-{self.story_organizer.story_id}-{series_id}.json")
        try:
            if os.path.exists(output_file):
                with open(output_file, 'r', encoding='utf-8') as file:
                    content = file.read().strip()
                    if content:  # 确保文件不是空的
                        data = json.loads(content)

        except json.JSONDecodeError:
            # 如果文件损坏或格式不正确，使用空的数据字典
            data = {}

        # 更新数据
        if not data:
            data["Settings"] = self.story_organizer.story['Settings']
            data["Episodes"] = {str(key): value for key,
                                value in self.story_organizer.story['Episodes'].items()}

        if "prequel_episode_summary" not in data["Episodes"][str(episode_id)]:
            data["Episodes"][str(episode_id)]["prequel_episode_summary"] = ""
        data["Episodes"][str(episode_id)]["prequel_episode_summary"] = self.history_managers[series_id]._get_summary()


        if "current_round_num" not in data["Episodes"][str(episode_id)]:
            data["Episodes"][str(episode_id)]["current_round_num"] = ""
        data["Episodes"][str(episode_id)]["current_round_num"] = episode_setting['round']

        if "total_round_num" not in data["Episodes"][str(episode_id)]:
            data["Episodes"][str(episode_id)]["total_round_num"] = ""
        data["Episodes"][str(episode_id)]["total_round_num"] = episode_setting['total_round_num']

        if "message_pairs" not in data["Episodes"][str(episode_id)]:
            data["Episodes"][str(episode_id)]["message_pairs"] = []
        data["Episodes"][str(episode_id)]["message_pairs"].append(
            {"role": "user", "content": user_output})
        data["Episodes"][str(episode_id)]["message_pairs"].append(
            {"role": "assistant", "content": story_output})

        # 写入json文件
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        print(f"Sync story output:  {series_id}, {episode_id} done")

    def sync_refine_output(
            self,
            story_setting,
            episode_setting,
            user_input,
            story_output,
            review,
            series_id=""):
        print(f"Sync refine output:  {series_id}")

        if self.language == "zh":
            role_name = "玩家"
        else:
            role_name = "Player"
        history = self.history_managers[series_id].get(role_name=role_name, language=self.language)

        sync_info = {
            "Settings": story_setting,
            "Episode_Settings": episode_setting,
            "History": history,
            "User": user_input,
            "Story": story_output,
            "Review": review
        }
        with jsonlines.open(self.refine_output_file, mode='a') as writer:
            writer.write(sync_info)
        print(f"Sync refine output:  {series_id} done")

    def _judge_quality_(self, review, current_next_episode, plot_round, total_round_num, try_num, next_episode_ids=[]):
        threshold = 3.5

        if plot_round >= total_round_num:
            if current_next_episode in next_episode_ids:
                threshold = threshold-0.5-try_num

        for key, value in review.items():
            if isinstance(value, dict):
                score = value.get('score', 0)
                if score < threshold:
                    return False
        return True

    def _generate(self, story_setting, episode_setting, series_id=""):
        role_name = story_setting.get('leading_role', '')
        history, round = self.history_managers[series_id].get(role_name=role_name, language=self.language)
        episode_setting['history'] = history
        episode_setting['round'] = round

        if 'total_round_num' not in episode_setting:
            # total_round_num = random.randint(2, 3)
            total_round_num = random.randint(5, 10)
            episode_setting['total_round_num'] = total_round_num

        user_input = self.user_generator.generate(
            story_setting, episode_setting, self.history_managers[series_id].get_last_story_plot())
        self._print(f"User Input:\n{user_input}\n", color=Fore.GREEN)
        if int(episode_setting['total_round_num']) <= int(round):
            self._print( f"========== Adding emergency_note ==========  \n"*3, color=Fore.YELLOW)
        generated_story = self.generator.generate(
            story_setting, episode_setting, user_input['action'])
        next_episode = self.story_trigger.generate(
            story_setting,
            episode_setting,
            history,
            user_input['action'],
            generated_story['plot_list'])
        if next_episode:
            generated_story['next_episode'] = next_episode
        self._print(
            f"Generated Story:\n{json.dumps(generated_story, indent=4, ensure_ascii=False)}\n",
            color=Fore.BLUE)
        story_output = generated_story

        review = self.refiner.generate(story_setting,
                                       episode_setting,
                                       user_input['action'],
                                       json.dumps({"plot_list": generated_story["plot_list"],
                                                   "next_episode": generated_story["next_episode"]},
                                                  indent=4,
                                                  ensure_ascii=False))
        self._print(
            f"Review:\n{json.dumps(review, indent=4, ensure_ascii=False)}\n",
            color=Fore.RED)
        
        best_quality = review.get('quality', 0)
        self.sync_refine_output(story_setting,
                                episode_setting,
                                user_input['action'],
                                json.dumps({"plot_list": generated_story["plot_list"],
                                            "next_episode": generated_story["next_episode"]},
                                           indent=4,
                                           ensure_ascii=False),
                                review,
                                series_id)
        try_num = 1
        next_episode_ids = [item['next_episode'] for item in episode_setting.get('triggers', [])]
        if (int(round)==int(episode_setting['total_round_num'])+5) and (generated_story['next_episode'] not in next_episode_ids):
        # if (int(round)==int(episode_setting['total_round_num'])+1) and (generated_story['next_episode'] not in next_episode_ids):
            generated_story['next_episode'] = next_episode_ids[0]
            self._print(f"========== force jump to: {next_episode_ids[0]} with good results ==========  \n"*3, color=Fore.YELLOW)
        while not self._judge_quality_(
                review,
                generated_story['next_episode'],
                round,
                episode_setting['total_round_num'],
                try_num,
                next_episode_ids) and try_num <= 3:
            
            if "thinking" in review:
                review.pop("thinking")
            # generated_story = self.regenerator.generate(
            #     story_setting, episode_setting, user_input['action'],
            #     json.dumps(
            #         {
            #             "plot_list": story_output["plot_list"],
            #             "next_episode": story_output["next_episode"]
            #         }, indent=4, ensure_ascii=False),
            #     json.dumps(
            #         {
            #             "plot": review['plot'],
            #             "guidance": review['guidance'],
            #             "narration": review['narration'],
            #             "characters": review['characters'],
            #         }, indent=4, ensure_ascii=False))
            self._print(f"========== No good, Re-gegenerating story: {try_num} times ==========  \n"*3, color=Fore.YELLOW)
            generated_story = self.generator.generate(
                story_setting, episode_setting, user_input['action'])
            next_episode = self.story_trigger.generate(
                story_setting, episode_setting, history, user_input, generated_story)
            if next_episode:
                generated_story['next_episode'] = next_episode
            # if (int(round)==int(episode_setting['total_round_num'])+5) and (generated_story['next_episode'] not in next_episode_ids):
            if (int(round)==int(episode_setting['total_round_num'])+1) and (generated_story['next_episode'] not in next_episode_ids):
                generated_story['next_episode'] = next_episode_ids[0]
                self._print(f"========== force jump to: {next_episode_ids[0]} with refined results ==========  \n"*2, color=Fore.YELLOW)
            self._print(
                f"Generated Story:\n{json.dumps(generated_story, indent=4, ensure_ascii=False)}\n",
                color=Fore.BLUE)

            review = self.refiner.generate(story_setting,
                                           episode_setting,
                                           user_input['action'],
                                           json.dumps({"plot_list": generated_story["plot_list"],
                                                       "next_episode": generated_story["next_episode"]},
                                                      indent=4,
                                                      ensure_ascii=False))
            self._print(
                f"Review:\n{json.dumps(review, indent=4, ensure_ascii=False)}\n",
                color=Fore.RED)
            self._print(f"========== Re-gegenerating story: {try_num} times finshed ! ==========  \n"*3, color=Fore.YELLOW)
            self.sync_refine_output(story_setting,
                                    episode_setting,
                                    user_input['action'],
                                    {"plot_list": generated_story["plot_list"],
                                     "next_episode": generated_story["next_episode"]},
                                    review,
                                    series_id)

            story_output['plot_list'] = generated_story['plot_list']
            story_output['next_episode'] = generated_story['next_episode']

            try_num += 1

        return user_input, story_output, episode_setting

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

    def _load_checkpoint(self, series_id, episode_line):
        history_manager = HistoryManager()
        output_file = os.path.join(self.story_output_dir,
                                   f"story-{self.story_organizer.story_id}-{series_id}.json")
        content = {}
        if os.path.exists(output_file):
            content = json.load(open(output_file, 'r', encoding='utf-8'))

        start_episode_id = episode_line['start_episode_id']
        if content:
            for i, state in enumerate(episode_line['states']):
                if not content['Episodes'][str(state)]['message_pairs']:
                    if i == 0:
                        start_episode_id = state
                    else:
                        next_episode = content['Episodes'][str(episode_line['states'][
                            i - 1])]['message_pairs'][-1]['content']['next_episode']
                        start_episode_id = next_episode
                        if next_episode == state and 'prequel_episode_summary' not in content['Episodes'][str(
                                next_episode)]:
                            history_manager.load(content['Episodes']
                                                 [str(episode_line['states'][i - 1])])
                            plot_summary = self.story_summary.generate(history_manager.get())
                            content['Episodes'][str(next_episode)
                                                ]['prequel_episode_summary'] = plot_summary
                            history_manager.clear()
                    break
            history_manager.load(content['Episodes'][str(start_episode_id)])

        return history_manager, start_episode_id

    def story_generate(self, story_setting, episode_line, series_id=""):
        self.history_managers[series_id], current_episode_id = self._load_checkpoint(series_id, episode_line)
        print(f"Start the current episode line: story-{self.story_organizer.story_id}-{series_id}")
        print("==" * 20)
        while current_episode_id != "-1":
            user_output, story_output, episode_setting = self._generate(story_setting, episode_line[current_episode_id], series_id)
            self.sync_story_output(
                episode_setting,
                current_episode_id,
                user_output,
                story_output,
                series_id=series_id)
            self.history_managers[series_id].append("user", user_output)
            self.history_managers[series_id].append("plot", story_output)
            if story_output['next_episode'] != current_episode_id:
                history = self.history_managers[series_id].get()
                summary = self.story_summary.generate(history)
                self.history_managers[series_id].clear()
                self.history_managers[series_id].append("summary", summary)
                current_episode_id = story_output['next_episode']
        print(f"Exit the current episode line: story-{self.story_organizer.story_id}-{series_id}")
        print("==" * 20)
        print("==" * 20)

    def run(self, renditions=2):
        story_setting = self.story_organizer.story["Settings"]
        episode_lines = self.story_organizer.episode_lines

        # with ContextThreadExecutor() as executor:
        #     for i in range(renditions):
        #         for j, episode_line in enumerate(episode_lines):
        #             executor.submit(self.story_generate,
        #                             story_setting=story_setting,
        #                             episode_line=episode_line,
        #                             series_id=f"{i}-{j}")

        for i, episode_line in enumerate(episode_lines):
            for j in range(renditions):
                self.story_generate(story_setting=story_setting,
                                    episode_line=episode_line,
                                    series_id=f"{i}-{j}")


def load_story_file(story_file):
    story_dict = {}
    with jsonlines.open(story_file) as reader:
        for obj in reader:
            story_dict[obj['Settings']['story_id']] = obj
    return story_dict


def generate_story(story_file, story_output_dir, refine_output_file, language="en"):
    story_dict = load_story_file(story_file)

    for story_id, story_info in story_dict.items():
        sg = StoryGenerator(story_info, story_output_dir, refine_output_file, language)
        sg.run()


def generate_story_parallel(
        story_file,
        story_output_dir,
        refine_output_dir,
        num_threads=1,
        num_runs=1):
    """
    生成故事的主函数

    Args:
        story_file: 故事框架文件路径
        story_output_dir: 故事输出目录
        refine_output_file: 优化输出文件路径
        language: 语言设置
        num_threads: 并行线程数
        num_runs: 每个state_line运行的次数
    """
    # 加载所有故事
    story_dict = load_story_file(story_file)

    # 创建线程池
    with ContextThreadExecutor(max_workers=num_threads) as executor:
        # 存储所有任务
        futures = []

        # 遍历所有story
        for i, story_info in enumerate(story_dict.values()):

            story_organizer = StoryOrganizer(story_info)
            episode_lines = story_organizer.episode_lines
            story_setting = story_organizer.story["Settings"]

            for j, episode_line in enumerate(episode_lines):
                for run_idx in range(num_runs):
                    # 提交任务到线程池
                    language = LANG2ID[story_info.get('Settings', {}).get(
                        'story_info', {}).get('language', 'English')]
                    sg = StoryGenerator(story_info, story_output_dir, refine_output_dir, language)
                    future = executor.submit(
                        sg.story_generate,
                        story_setting=story_setting,
                        episode_line=episode_line,
                        series_id=f"{j}-{run_idx}"
                    )
                    futures.append(future)

        # 等待所有任务完成
        for future in futures:
            future.result()


if __name__ == '__main__':
    story_file = os.path.join(os.path.dirname(__file__), 'storyline_generator/story_framework_data/all_data_file_new_triggers-0811.jsonl')
    # story_file = os.path.join(os.path.dirname(__file__), 'storyline_generator/story_framework_data/all_data_file_new_triggers-0811-test.jsonl')

    story_output_dir = 'data/story_output'
    refine_output_dir = 'data/refine_output'
    generate_story_parallel(story_file,
                            story_output_dir,
                            refine_output_dir,
                            num_threads=10,
                            num_runs=3)
    # generate_story_parallel(story_file,
    #                         story_output_dir,
    #                         refine_output_dir,
    #                         num_threads=1,
    #                         num_runs=1)
