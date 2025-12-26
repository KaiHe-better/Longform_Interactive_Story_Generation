import os
import re
import sys
import time
import json
import copy
import logging
import argparse
import jsonlines
from tqdm import tqdm
from datetime import datetime
from langchain.prompts import PromptTemplate 
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from loguru import logger
import random

logger.remove()
logger.add(sys.stdout, level="INFO")
logger.debug("This is a debug message")  # 不会显示
logger.info("This is an info message")   # 会显示
logger.warning("This is a warning message")  # 会显示
logger.error("This is an error message")  # 会显示
logger.critical("This is a critical message")  # 会显示
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 将上一级目录添加到 sys.path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from llm import AzureLLM, GPTProxyLLM, DeepSeekLLM
llm_gpt = GPTProxyLLM()
llm_deepseek = DeepSeekLLM()




# 拆分think
def ds_think_analysis(model_chat:str):
    model_chat_list = model_chat.split("<thinking>") 
    if len(model_chat_list) == 2:
        model_think = model_chat_list[0].strip()
        model_ans = model_chat_list[1].strip()
    elif len(model_chat_list) == 1:
        model_think = ""
        model_ans = model_chat_list[0].strip()
        # print("Warning! no <thinking> in res")
    else:
        logger.error("Warning! too many <thinking> in res")
        
    return model_think, model_ans

# llm及处理
def llm_chat_analysis(llm_model_name:str, llm_prompt:str, res_json_type:str):
    # print(llm_prompt)
    
    if llm_model_name == "deepseek-r1":
        llm_model = DeepSeekLLM()
    elif llm_model_name == "gpt-4o":
        llm_model = GPTProxyLLM()
    else:
        logger.error("wrong llm model type")

    try_count = 0
    while try_count<3:
        try_count += 1
        try:

            if type(llm_prompt) == str:
                res_llm = llm_model.chat(llm_prompt)
                # print(res_llm)
            elif type(llm_prompt) == dict:
                res_llm = llm_model.chat(llm_prompt["query"],llm_prompt["example"],llm_prompt["system_prompt"])

            else:
                logger.error(f"wrong prompt type {type(llm_prompt)}: {llm_prompt}")


            if llm_model_name == "deepseek-r1":
                res_think, res_ans = ds_think_analysis(res_llm)
                res_think = res_think.replace('\n\n','\n').strip()
            else:
                res_think = ""
                res_ans = res_llm
            res_ans = res_ans.strip()

            try:
                cleaned_json_string = res_ans.split('```json')[1].split('```')[0].strip()
            except:
                try:
                    cleaned_json_string = res_ans.split('```')[1].strip()
                except:
                    cleaned_json_string = res_ans
            cleaned_json_string = cleaned_json_string.strip()

            if res_json_type == "str":
                return res_think, cleaned_json_string
            if res_json_type == "int":
                return res_think, int(cleaned_json_string)
            # print(cleaned_json_string)
                    
            try:
                res_json_load = json.loads(cleaned_json_string)
            except:
                try:
                    cleaned_json_string1 = cleaned_json_string
                    if not cleaned_json_string1.startswith("{"):
                        cleaned_json_string1 = "{\n" +cleaned_json_string1
                    if not cleaned_json_string1.endswith("}"):
                        cleaned_json_string1 = cleaned_json_string1 + "\n}" 
                    res_json_load = json.loads(cleaned_json_string1)
                except Exception as e:
                    try:
                        cleaned_json_string1 = cleaned_json_string
                        cleaned_json_string1 = cleaned_json_string1.strip("\"").strip("main").strip("\"").strip(":")
                        res_json_load = json.loads(cleaned_json_string1)
                    except:
                        cleaned_json_string1 = cleaned_json_string.replace('[',"{").replace.replace(']',"}")
                        
                
            if type(res_json_load) == dict:
                try:
                    res_json_load = res_json_load["main"] 
                except:
                    pass
            
            
            if res_json_type == "dict":
                res_json = {}
                if type(res_json_load) == dict:
                    res_json = res_json_load
                elif type(res_json_load) == list:
                    for item in res_json_load:
                        # 多个str组成的list：['伊森·温特斯: 主角...',...]
                        if type(item) == str:
                            item1 = item.split(":")
                            res_json[item1[0].strip()] = item1[1].strip()
                        # 多个dict组成的list：[{'伊森·温特斯': '主角...'},...,{}]
                        elif type(item) == dict:
                            res_json.update(item)
                # 几率低, 比较粗糙的处理方法, 可以改进
                # elif type(res_json_load) == str:
                #     pass
                else:
                    logger.error(f"Warning! res_json_load type error {type(res_json_load)}, need {res_json_type}")

            elif res_json_type == "list":
                res_json = []
                if type(res_json_load) == list:
                    res_json = res_json_load
                elif type(res_json_load) == dict:
                    res_json = []
                    for key in res_json_load.keys():
                        res_json.append(f"{key}: {res_json_load[key]}")
                # 几率低, 比较粗糙的处理方法, 可以改进
                elif type(res_json_load) == str:
                    res_json.append(res_json_load)
                else:
                    logger.error(f"Warning! res_json_load type error {type(res_json_load)}, need {res_json_type}")

            return res_think, res_json
        except Exception as e:
            logger.error(f"重新生成, 发生错误：{e}")
                            





prompt_gen_story_template = {
    # 提取背景
    "create_storyline_keywords": PromptTemplate(
        template = """你是一个资深作家, 接下来请根据我的关键词设计一个符合关键词的一个故事大纲
关键词如下：** {keywords} **

要求：
1. 故事大纲需要包含关键词, 应该将不同关键词的剧情进行融合，不要出现关键词缺失， 根据关键词确定故事题材、情绪
2. 剧情合理有逻辑，不要出现剧情矛盾
3. 剧情大纲有深度有新意，突出反转与冲突，
4. 有吸引力，情感准确传达，确保有代入感，开头应该足够吸引人
5. 单主角，以主角的视角展开剧情，每个剧情主角都应该出现
6. 故事大概有十个章节，每个章节都应该有一些核心事件发生，且不同章节的事件应该有所差异
7. 像故事一样叙述，不要出现大纲的格式，不要出现“本故事”，“主角”这种令人出戏的词汇


输出格式:使用json格式输出，最终是一个list，每个list项是一个str格式，为一个章节的详细剧情"
示例如下：
[
    "(章节1具体剧情)",
    "(章节2具体剧情)",
    ...
]
    """,
        input_variables=["keywords"],
    ),
    "sumary_title": PromptTemplate(
        template = """有一段剧情和这段剧情的关键词，将它起一个标题
要求：精简，15个字以内，文艺且能体现故事，不要与现有的小说或者游戏等标题重复
只返回标题，不要返回其他内容

关键词如下：
{keywords}
剧情如下：
{plot}
        """,
        input_variables=["plot", "keywords"],
    ),
    "summary_title_plot": PromptTemplate(
        template = """有一段剧情，将它精简成一句话,作为章节标题，不要书名号等其他内容
剧情如下：
{plot}
        """,
        input_variables=["plot"],
    ),

}



all_keyword = []

def keywords_generator():
    confront_keywords = [
        ["好结局","悲剧"],
        ["男性","女性"],
        ["人类主角","非人类主角"],
        ["现代","历史","未来","穿越"],
        ["现实世界","平行世界","架空世界"],
        ["黑暗压抑","轻松治愈","沙雕搞笑","黑色幽默","热血励志","荒诞讽刺"]
    ]
    fusion_keywords = [
        ["爱情","亲情","友情","背叛","虐恋","仇恨","救赎","复仇", "成长", "自我救赎", "牺牲", "奉献", "孤独", "依赖"],
        ["悬疑推理","权谋斗争","恐怖惊悚","军事战争","克苏鲁诡谲","生存竞技","青春","魔法","科幻","玄幻仙侠","都市","现实","游戏","言情","末世","冒险", "竞技", "校园", "职场", "谍战"],
        ["天才", "废柴逆袭", "杀手", "侦探", "特工/士兵", "医生", "学生", "商人", "艺术家", "武者","扮猪吃老虎"],
        ["高智商","反套路","道德困境","黑色幽默","轻松","养成","逆袭","硬核"]
    ]

    confront_keywords_list = []
    for i in range(len(confront_keywords)):
        confront_keywords_list.append(confront_keywords[i][random.randint(0, len(confront_keywords[i])-1)])

    story_requirements = f"故事主角是{confront_keywords_list[2]}({confront_keywords_list[1]}); 故事发生于{confront_keywords_list[3]}的{confront_keywords_list[4]}, 整体是个{confront_keywords_list[0]}, 故事风格/氛围是{confront_keywords_list[5]}的\n故事类型关键词如下:\n"
    # print(story_keywords_list)

    for i in range(len(fusion_keywords)):
        random_type = random.randint(0,1)
        if random_type == 0:
            random_num = random.randint(0, min(4,len(fusion_keywords[i])))
        else:
            random_num = random.randint(0, len(fusion_keywords[i])-1)
            
        random_selection = random.sample(fusion_keywords[i], random_num)
        story_requirements += " ".join(random_selection)+" "

    story_requirements = story_requirements.strip()

    # if story_requirements in all_keyword:
    #     pass
    # else:
    #     print(story_requirements)
    #     all_keyword.append(story_requirements)

    return story_requirements


def create_storyline(keywords:str):
    prompt1 = prompt_gen_story_template['create_storyline_keywords'].format(keywords=keywords)
    _,story_plot = llm_chat_analysis("gpt-4o", prompt1, "list")

    prompt2 = prompt_gen_story_template['sumary_title'].format(plot='\n'.join(story_plot), keywords=keywords)
    _,story_title = llm_chat_analysis("gpt-4o", prompt2, "str")

    data_output_file = f"{data_output_folder}{story_title}_gen.jsonl"
    # print(story_plot)
    print(len(story_plot))
    for plot_i in tqdm(story_plot):

        # print(plot_i)
        prompt3 = prompt_gen_story_template['summary_title_plot'].format(plot=plot_i)
        _,sum_plot_i = llm_chat_analysis("gpt-4o", prompt3, "str")
        # print(sum_plot_i)
        content_plot_sum = sum_plot_i.strip('') + '\n\n' + plot_i.replace('\n\n','\n').strip('')

        with jsonlines.open(data_output_file, mode='a') as writer:
            writer.write(content_plot_sum)


data_output_folder = f"storyline_generator/source_stories/extra_story_output/create_storyline/"



def main():
    if not os.path.exists(data_output_folder):
        os.makedirs(data_output_folder)
    for i in tqdm(range(80)):
        keywords = keywords_generator()
        create_storyline(keywords)
        # break

if __name__ == "__main__":
    main()