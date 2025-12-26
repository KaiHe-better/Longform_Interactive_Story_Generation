import os
import shutil
import json
from tqdm import tqdm
import re
import time
import pandas as pd
import csv
import numpy as np
import copy
import chardet
import tiktoken
import argparse
import jsonlines
import zipfile
from collections import Counter
from ebooklib import epub
import ebooklib
from bs4 import BeautifulSoup
import string
import sys
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate 
# from langchain_openai import AzureChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
# from kor.extraction import create_extraction_chain
# from kor.nodes import Object, Text, Number

from loguru import logger
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

parser = argparse.ArgumentParser(description="配置参数解析")
parser.add_argument("--data_base_folder", type=str, default="storyline_generator/source_stories/", help="小说数据文件夹") 
parser.add_argument("--source_story_name", type=str, default="神探狄仁杰1-使团喋血记", help="小说名") 
# "HarryPotter1-TheSorcerersStone"
# 重要！！语言
parser.add_argument("--use_language", type=int, default=0, choices=[0, 1], help="语言，0代表中文，1代表英文")
parser.add_argument("--if_novel_series", type=bool, default=False, help="是否是系列故事，多个故事以同一个story名作为开头，需要整合") 
parser.add_argument("--novel_text_type", type=str, default="txt", choices=["txt","pdf","epub"], help="原格式是否需要处理") 
parser.add_argument("--repair_raw_text", type=bool, default=True, help="") 
parser.add_argument("--integrate_max_token", type=int, default=2000, help="文本整合后最大输入") 
parser.add_argument("--chapter_type_en", type=int, default=1, help="英文章节，设置为1比较稳妥，不然可能会因为行首的 'I ***' 而误判，但是要求格式比较单一") 
# 一般应用于一个小说中有多个章节的小故事，例如神探狄仁杰第一部，里面有三个章节《使团喋血记》、《蓝衫记》、《滴血雄鹰》，只抽取其中一个章节
parser.add_argument("--extract_chapters_start", type=int, default=0, help="小说从第几章开始截取，0是从头开始，其他的是具体章节数") 
parser.add_argument("--extract_chapters_end", type=int, default=-1, help="小说需要截取多少章节，-1是全书，其他的是具体章节数") 
# 抽取断点重连，gpt断连后修改
parser.add_argument("--extract_begin", type=int, default=0, help="抽取断点重连，从这个值开始抽取直到最后") 
parser.add_argument("--extract_end", type=int, default=-1, help="最少抽取多少章节，-1为抽全篇") 
parser.add_argument("--chapter_cut_type", type=str, default="chapter", choices=["chapter","token"], help="1是按章节划分，0是按token划分") 

args = parser.parse_args()
if args.use_language == 0:
    llm_ans_language = "中文"
elif args.use_language == 1:
    llm_ans_language = "English"
else:
    logger.error(f"warning! wrong parameter for language {args.use_language}")

enc = tiktoken.get_encoding("cl100k_base")



# messages_multichat_template ={"system_prompt": "", "example_query":"","example_ans":"","query":""}
prompt_extra_story_template = {
    # 提取背景
    "summary_story_coarse": {
        "system_prompt":"Summarize the key points of the following text in a concise way, using bullet points.",
        "example":[
            {
                "role":"assistant",
                "content":"""###
    Text:
    洪七公、周伯通、郭靖、黄蓉四人乘了小船，向西驶往陆地。黄蓉不住向周伯通详问骑鲨游海之事，周伯通兴起，当场就要设法捕捉鲨鱼，与黄蓉大玩一场。
    洪七公被欧阳锋点中之后，穴道虽已解开，内伤却又加深了一层。黄蓉喂他服了几颗九花玉露丸，痛楚稍减，气喘仍是甚急。
    老顽童不顾别人死活，仍是嚷着要下海捉鱼，黄蓉向他连使眼色，要他安安静静的，别吵得洪七公心烦。周伯通并不理会，只闹个不休。黄蓉皱眉道：“你要捉鲨鱼，又没饵引得鱼来，吵些甚么”

    Summarize in BULLET POINTS form:
                """
            },
            {
                "role":"user",
                "content":"""
    - 洪七公等四人乘船西行,洪七公因受内伤加重而气喘不止
    - 周伯通要捉鲨鱼玩,被黄蓉阻止
                """
            }
        ],
        "query":"",
    },

    "summary_story_fine_example": {
        "system_prompt":"Summarize the above key points into a short paragraph, as short as possible",
        "example":[
            {
                "role":"assistant",
                "content":"""###
Text:
- 郡主企图刺杀狄公，虎敬晖牺牲自己保护狄公。
- 郡主揭露自己是真正的金木兰，旨在推翻武则天的统治。
- 郡主批评太子软弱，表明自己利用外力对抗内敌的策略。
- 狄公和郡主就忠诚和策略发生争执，郡主强调她的目的是推翻武则天。
- 郡主努力三年联络各种势力营救刘金，曾策划失败的营救行动
- 武则天的圣旨让郡主嫁给突厥可汗，郡主视为救刘金及自身脱离牢笼的机会
- 郡主与吉利可汗的叔叔莫度密谋，计划在团进京时假死逃脱
- 郡主自认为是李青霞，狄仁杰因此未能识破其真实身份
- 狄仁杰愤怒郡主勾结突厥，郡主则声称要成为女皇帝
- 虎敬晖为保护狄仁杰而攻击郡主，最后牺牲
Summarize in BULLET POINTS form:
                """
            },
            {
                "role":"user",
                "content":"郡主企图刺杀狄公，虎敬晖因此牺牲。她自称金木兰，计划利用外力推翻武则天，成为女皇帝。之前营救刘金失败，此次武则天令她嫁给突厥可汗，她借此机会与可汗叔叔密谋假死逃脱。狄仁杰虽未识破她的真实身份，但对她勾结外敌表示愤怒。"
            }
        ],
        "query":"",
    },

    "summary_story_fine": PromptTemplate(
        template = """接下来我将给你一段摘要，你需要将这些摘要再次总结精炼成一段叙述性的话
            用{language}回答,不要提到"好的","明白了"这种令人出戏，没有意义的字眼
            原始摘要如下：{long_summary}

        """,
        input_variables=["language","long_summary"],
    ),
}




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
def llm_chat_analysis(llm_model_name:str, llm_prompt, res_json_type:str):
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
        except:
            break
    return res_llm.strip()

# 读取epub格式小说
def epub_to_txt(epub_path, txt_path):
    book = epub.read_epub(epub_path)
    text = ''
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            text += soup.get_text() + '\n\n'
    with open(txt_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(text)

def save_epub_2_txt(story_name:str):
    if args.if_novel_series:
        series_folder = f'{args.folder_path}/novel_source/epub/{story_name}'
        output_folder = f'{args.folder_path}/novel_source/{story_name}_epub2'
        os.makedirs(output_folder, exist_ok=True)
        with tqdm(total=len(os.listdir(series_folder)), desc="Processing .epub files") as pbar:  
            for filename in os.listdir(series_folder):
                if filename.endswith(".epub"):
                    epub_path = os.path.join(series_folder, filename)
                    txt_name = os.path.splitext(filename)[0] + "_epub2.txt"
                    txt_path = os.path.join(output_folder, txt_name)
                    epub_to_txt(epub_path, txt_path)
                if filename.endswith(".txt"):
                    shutil.copy(os.path.join(series_folder, filename),os.path.join(output_folder, filename))
                pbar.update(1)  
    else:
        print("epub 格式转换中...")
        epub_path = f'{args.folder_path}/novel_source/epub/{story_name}.epub'
        txt_path = f'{args.folder_path}/novel_source/{story_name}_epub2.txt'
        epub_to_txt(epub_path, txt_path)
        print(f"{args.name}文件 epub 格式转换完成...")

    novel_name = f"{story_name}_epub2"
    return novel_name


def establish_cor_folder(story_name):
# 小说文件应该在 {base_folder}/novel_source/"中)
    story_history_folder = f"{args.data_base_folder}/novel_data_history/{story_name}"       # 处理后的文件夹
    save_output_folder = f"{story_history_folder}/{story_name}_extract"                    # 抽取的文件
    intermediate_data_folder =  f"{story_history_folder}/reorganized_story_{story_name}"    # 第二部分用的
    # save_variables = f'{novel_folder}/middle_variable'                      # 存储一些中间变量
    if not os.path.exists(story_history_folder):
        os.makedirs(story_history_folder)
        print(f"已创建_{story_name}_文件夹")
    # 创建相应存储文件夹

    if not args.if_novel_series:        # 单本小说
        story_file = f'{story_history_folder}/{story_name}.txt'  
        novel_source = f"{args.data_base_folder}/novel_source/{story_name}.txt"
        if not os.path.exists(novel_source):
            print(f"注意，{novel_source}文件不存在!")
        if not os.path.exists(story_file):
            print("复制小说ing...")
            shutil.copy(novel_source,story_file)
    else:                       # 多部系列小说在一个文件夹下
        story_file = f'{story_history_folder}/{story_name}' 
        novel_source = f"{args.data_base_folder}/novel_source/{story_name}"
        if not os.path.exists(novel_source):
            print(f"注意，{novel_source}文件夹不存在!")
        else:
            if not os.listdir(novel_source):
                print(f"注意，文件夹 {novel_source} 为空!")
        if not os.path.exists(story_file):
            print("复制文件夹ing...")
            shutil.copytree(novel_source,story_file)

    if not os.path.exists(save_output_folder):
        os.makedirs(save_output_folder)
        print(f"已创建{story_name}_extract文件夹")
    else:
        print(f'文件夹 {save_output_folder} 已经存在')

    data_output_folder = f"{args.data_base_folder}/extra_story_output/extra_storyline/"
    if not os.path.exists(data_output_folder):
        os.makedirs(data_output_folder)
        print(f"已创建{data_output_folder}文件夹")
    else:
        print(f'文件夹 {data_output_folder} 已经存在')

    return story_history_folder, save_output_folder, intermediate_data_folder,story_file,data_output_folder




def detect_encoding(filename):
    with open(filename, 'rb') as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    return result['encoding']
    
#print("is novel series? ",args.if_novel_series)

def contains_roman_numerals(text):
    pattern = r"\b([IVX]+)\b"  #IVXLCDM
    matches = re.findall(pattern, text, re.IGNORECASE)
    if matches:
        return True
    return False

def has_digit(text):
    for char in text:
        if char.isdigit():
            return True
    return False

def contains_all_english_digits(text):
    english_digits = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                      "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                      "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
                      "eighty", "ninety", "hundred", "thousand", "million", "billion"]
    text = text.lower()
    if contains_roman_numerals(text): # 罗马数字
      return True
    text = text.replace(" ", "")
    for digit in english_digits:

      if text.find(digit) != -1:
        return True
    if has_digit(text): # 阿拉伯数字
      return True
    return False

def is_chapter_line(line,chapter_type):
    # 英文设置为1比较稳妥，不然可能会因为行首的I ***而误判，但是要求格式比较单一
    if chapter_type == 0:
        line_judge = line.replace(" ", "").replace("　", "").replace(",", "").replace(".", "").replace("'", "").replace("“", "").replace("’", "").replace("”", "").replace(";", "").replace(":", "").replace("?", "").replace("‘", "").replace("—", "").replace("-", "").replace("(", "").replace(")", "").replace("\"", "").replace("*", "").replace("!", "")
        return line.strip().startswith('CHAPTER') or line.strip().startswith('Chapter') or contains_roman_numerals(line_judge) or line.strip().startswith('VOLUME')
    elif chapter_type == 1:
        return line.strip().startswith('CHAPTER') or line.strip().startswith('Chapter') or line.strip().startswith('VOLUME')
    elif chapter_type == 2:
        line_judge = line.replace(" ", "").replace(",", "").replace(".", "").replace("'", "").replace("“", "").replace("’", "").replace("”", "").replace(";", "").replace(":", "").replace("?", "").replace("‘", "").replace("—", "").replace("-", "").replace("(", "").replace(")", "").replace("\"", "").replace("*", "").replace("!", "")
        return contains_roman_numerals(line_judge)
    elif chapter_type == 3:         # 针对的是1./n这种类型
        pattern = r"\d+\."
        return re.fullmatch(pattern, line.strip())
    else:
        # 如果 chapter_type 不是 0、1、2 中的一个值，可能需要根据实际情况进行处理
        raise ValueError("Invalid chapter_type value")


# 用正则简单看下对话数
def count_dialogues(raw_text):
    # 使用正则表达式匹配任意跨行的对话
    raw_text1 = raw_text.replace('\n','')
    dialogue_pattern = re.compile(r'["\'“‘](.+?)["\'”’]', re.DOTALL)
    # 找到所有匹配的对话
    dialogues = re.findall(dialogue_pattern, raw_text)

    # 返回对话的数量
    return len(dialogues)
# 定义divide函数，用来切分超长文本
def divide_str(s, sep=['\n', '.', '。']):
    mid_len = len(s) // 2  # 中心点位置
    best_sep_pos = len(s) + 1  # 最接近中心点的分隔符位置
    best_sep = None  # 最接近中心点的分隔符
    for curr_sep in sep:
        sep_pos = s.rfind(curr_sep, 0, mid_len)  # 从中心点往左找分隔符
        if sep_pos > 0 and abs(sep_pos - mid_len) < abs(best_sep_pos - mid_len):
            best_sep_pos = sep_pos
            best_sep = curr_sep
    if not best_sep:  # 没有找到分隔符
        return s, ''
    return s[:best_sep_pos + 1], s[best_sep_pos + 1:]
def strong_divide(s):
    left, right = divide_str(s)
    if right != '':
        return left, right
    whole_sep = ['\n', '.', '，', '、', ';', ',', '；',\
                 '：', '！', '？', '(', ')', '”', '“', \
                 '’', '‘', '[', ']', '{', '}', '<', '>', \
                 '/', '''\''', '|', '-', '=', '+', '*', '%', \
               '$', '''#''', '@', '&', '^', '_', '`', '~',\
                 '·', '…']
    left, right = divide_str(s, sep=whole_sep)
    if right != '':
        return left, right
    mid_len = len(s) // 2
    return s[:mid_len], s[mid_len:]


# 切片 - 切章节和chunk
# 读文件
def read_raw_text(story_file):
    if not args.if_novel_series:   
        # 检测文件编码
        try:
            try:
                raw_text = open(story_file, encoding="gbk").read()
                logger.info("default gbk")
            except:
                raw_text = open(story_file, encoding="utf-8").read()
                logger.info("default utf-8")
        except:
            logger.warning("not utf-8")
            file_encoding = detect_encoding(story_file)
            logger.info(f"file_encoding: {file_encoding}")

            # 处理 raw_text
            # 根据检测到的编码来打开文件
            raw_text = open(story_file, encoding=file_encoding).read()
            # print(raw_text[:100])
    else:
        raw_text = ""
        files_series = os.listdir(story_file)
        files_series.sort()  # 按文件名排序
        num = 0
        with tqdm(total=len(files_series)) as pbar:
            for file_name in files_series:
                file_series_path = os.path.join(story_file, file_name)
                if os.path.isfile(file_series_path):
                    num += 1 
                    raw_text = raw_text+ f"\n VOLUME {num} - {os.path.splitext(file_name)[0]} \n"

                    encoding = detect_encoding(file_series_path)
                    with open(file_series_path, 'r', encoding=encoding) as f:
                        raw_text += f.read()
                pbar.update(1)  # 更新进度条
        logger.info(f"文件夹共{num}本小说")
    return raw_text

# 处理英文语料的一些换行符
def filiter_txt_line(raw_text):
    lines = raw_text.splitlines()
    raw_text_new = ""
    # 逐行处理
    judge_space= 0
    chapter_num = 0
    logger.info(f"原始文本行数: {len(lines)}")
    for i in range(len(lines)):
        if is_chapter_line(lines[i], args.chapter_type):

            raw_text_new += "\n" + lines[i] + "\n"
            chapter_num += 1
        else:
            if lines[i].strip() == "":
                
                if i<len(lines)-1 and  lines[i+1].strip() == "":
                    
                    continue
                elif judge_space == 1:
                    judge_space = 0
                    continue
                else:
                    raw_text_new += lines[i] + "\n"
            else:
                #print(i,lines[i])
                if lines[i].strip()[-1].isdigit() or (lines[i].strip()[-1] in string.punctuation):
                    #print(i)
                    raw_text_new += lines[i] + "\n"
                else:
                    #print(i,lines[i])
                    raw_text_new += lines[i]
                    if lines[i][-1] !=" ":
                        raw_text_new += " "
                    if i<len(lines)-2 and lines[i+1].strip() == "":
                        judge_space = 1

    lines = raw_text_new.splitlines()
    logger.info(f"原始文本行数: {len(lines)}")
    logger.info(f"chapter_num: {chapter_num}")
    return raw_text


# 切 chapter
def split_chapter(raw_text):
    chapters = []
    chapter_contents = []
    chapter_num_flag = 0
    if args.use_language == 0:
        logger.info("故事语言: 中文")
        # 中文切章节
        # 我觉得这段代码现在比正则表达式好用， 当然这个代码是针对 第xx章 xxxx 这种格式的
        # 如果有的时候你是 ---第xx章--， 要改一下那个startswith那个 或者把前面的特殊符号给strip掉
        for line in raw_text.split('\n'):
            Flag = False
            if line.strip().startswith('第'):
                # 遇到章节标题,将之前章节内容添加到结果列表
                head = line.strip()
                # print(head)
                #head = head[:min(8,len(head))]
                head = head[:min(30,len(head))]
                if head.find('章',1)>0 or head.find('部',1)>0:
                    logger.info(head)
                    Flag = True
            if Flag:
                if chapter_contents:
                    chapters.append('\n'.join(chapter_contents))
                    chapter_contents = []
                    chapter_contents.extend([head, '\n'])
                # 记录当前章节标题
                # chapters.append(line)
            else:
                # 累积章节内容
                chapter_contents.append(line)
        # 添加最后一个章节内容
        if chapter_contents:
            chapters.append('\n'.join(chapter_contents))

    elif args.use_language == 1:          #英语抽章节
        logger.info("story language: English")
        chapters_name = []
        i = 0
        Flag_content = 2
        for line in raw_text.split('\n'):
            line = line.replace('\u3000',' ')
            Flag = False
            line_judge = line.replace(" ", "").replace("　", "").replace(",", "").replace(".", "").replace("'", "").replace("“", "").replace("’", "").replace("”", "").replace(";", "").replace(":", "").replace("?", "").replace("‘", "").replace("—", "").replace("-", "").replace("(", "").replace(")", "").replace("\"", "").replace("*", "").replace("!", "")
            if is_chapter_line(line, args.chapter_type):
                i = i+1
                # 遇到章节标题,将之前章节内容添加到结果列表
                head = line.strip()
                #print(head)

                if args.chapter_type ==  3:
                    head_num = head[0:-1]
                    #print(head_num)
                else:
                    head_num = head[7:min(20,len(head))]
                    #print(head_num)
                #print(head)
                if (contains_all_english_digits(head_num)) or contains_roman_numerals(line_judge): #or has_digit(head[:11])): #and (head.find("'") == -1):
                    Flag = True
                    if Flag and Flag_content<=1:
                        Flag = False
                    Flag_content = 0 # 记录上一行是不是标题，0代表是
            if line:
                Flag_content += 1
            if Flag:
                if chapter_contents:
                    chapters.append('\n'.join(chapter_contents))
                    chapter_contents = []
                    chapter_contents.extend([head, '\n'])
                    chapter_num_flag += 1
                # 记录当前章节标题
                chapters_name.append(line)
            else:
                # 累积章节内容
                chapter_contents.append(line)
        # 添加最后一个章节内容
        if chapter_contents:
            chapters.append('\n'.join(chapter_contents))
        for i in range(len(chapters_name)):
           logger.info(chapters_name[i])
        # 注意，这里只改了中文切分章节把标题加入的逻辑，英文需要额外修改
    else:
        print("warning! Currently not supported for other languages")
    logger.info(f"len(chapters):{len(chapters)}")
    # logger.info(f"flag:{chapter_num_flag}")
    return chapters



# 切块 chunk
def split_chunk(chapters):
    
    # 以1500 token为限，切分chunk，输出总chunk数量
    # max_token_len = 24000 #基本是一章，太大了
    max_token_len = args.integrate_max_token   # 对话抽取5000效果可以,但是有点犯病
    chapter_num = -1
    extract_chapters_start = args.extract_chapters_start
    extract_chapters_end = args.extract_chapters_end
    # 这里要注释掉
    # 这里是你要截取多少章节，比如说一部小说有3个故事，其中，想要第二个故事，则设置章节数开头和结尾
    # 选取所有就设置为-1或者注释（默认是-1）
    # extract_chapters_end = 17
    #extract_chapters_start = 2

    chunk_text = []
    chapter_text_token = []
    for chapter in chapters:
        # 如果截取章节从0开始，0一般是第一章前的介绍或者序章
        if (not extract_chapters_end == -1) and chapter_num >= extract_chapters_end:
            logger.info(f"截取章节数：{extract_chapters_start} - {extract_chapters_end}")
            break
        chapter_num += 1
        if chapter_num < extract_chapters_start:
            continue
        # 找到第一个换行符的位置
        newline_index = chapter.find('\n')
        # 如果找到了换行符，就取从开始到换行符之前的部分，否则取前30个字符
        # logger.info(f"当前章节标题/前30字符: {chapter[:min(30, newline_index) if newline_index != -1 else 30]}")
        split_text = chapter.split('\n')
        tmp = []
        for line in split_text:
            # 有无影响？
            if not line.strip():
                continue
            line_len = len(enc.encode( line ))
            if line_len <= max_token_len - 5:
                tmp.append(line)
            else:
                path = [line]
                tmp_res = []
                while path:
                    my_str = path.pop()
                    left, right = strong_divide(my_str)
                    len_left = len(enc.encode( left ))
                    len_right = len(enc.encode( right ))
                    if len_left > max_token_len - 15:
                        path.append(left)
                    else:
                        tmp_res.append(left)

                    if len_right > max_token_len - 15:
                        path.append(right)
                    else:
                        tmp_res.append(right)
                for line in tmp_res:
                    tmp.append(line)
        split_text = tmp

        curr_len = 0
        curr_chunk = ''
        chunk_text_per_chapter = []
        for line in split_text:
            line_len = len(enc.encode( line ))
            if line_len > max_token_len:
                logger.warning('warning line_len = ', line_len)
            if curr_len + line_len <= max_token_len:
                curr_chunk += line + '\n'
                curr_len += line_len + 1
            else:
                chunk_text_per_chapter.append(curr_chunk)
                curr_chunk = line
                curr_len = line_len
        if curr_chunk:
            chunk_text_per_chapter.append(curr_chunk)
        chunk_text.extend(chunk_text_per_chapter) 
        # break
        if args.chapter_cut_type == 'chapter':
            chapter_text_token.append(chunk_text_per_chapter)
            #print("chapter与len",chapter_num,len(chapter_text_token))   # 如果是从头开始计算，chapter_num 应该比 len（chapter）少1
            logger.info(f"chapter_num: {chapter_num}, 当前章节chunk数: {len(chapter_text_token[-1])}, 累计章节chunk数: {len(chunk_text)}")
            #chunk_text = []


    logger.info(f"分块后块数: {len(chunk_text)}")
    return chunk_text,chapter_text_token

# 按章节和token总结
def summary_story_coarse(story_raw_chunks_chapter:list,save_output_folder:str):
        # 按token
    # 和下面的二选一
    # 按章节可能过长，时间异常长
    # 尽量按选章节
    sum_cor_time = time.time()
    if False:
        args.extract_end = -1
        # 最后一个章节数+1
        args.extract_end = 10
        # 全篇
        if args.extract_end <= 0:
            extract_part = len(chunk_text)  # 全篇
        else :
            extract_part = args.extract_end

        print(len(chunk_text))

        for i in tqdm(range(args.extract_begin, min(len(chunk_text),extract_part))):  #断点重连
            
            save_name_sum = os.path.join(save_folder, f"{i}_sum.txt")
            # 文本过短，跳过
            if len(chunk_text[i])<30:
                continue

            if not os.path.exists(save_name_sum) or os.path.getsize(save_name_sum) < 5:
                if os.path.exists(save_name_sum):
                    print('re-summarize id = ',i )
                #dealing with summarize
                messages = [SystemMessage( content = system_prompt_0),
                    HumanMessage( content = q_example_0),
                    AIMessage( content = a_example_0)]

                new_input = f"###\nText:\n{chunk_text[ i ]}\nSummarize in BULLET POINTS form:"

                messages.append( HumanMessage(content = new_input) )

                summarize_response = llm_agent.invoke(messages).content

                with open(save_name_sum, 'w', encoding='utf-8') as f:
                    f.write( summarize_response )

            raw_text_save_name = os.path.join(save_folder, f"{i}_raw.txt")
            if not os.path.exists(raw_text_save_name) or os.path.getsize(raw_text_save_name) < 5:
                with open(raw_text_save_name, 'w', encoding='utf-8') as f:
                    f.write( chunk_text[i] )
            #break

    else:
        # 按章节
        args.extract_end = -1
        #args.extract_end = 10
        # 全篇
        if args.extract_end <= 0:
            extract_part = len(story_raw_chunks_chapter)  # 全篇
        else :
            extract_part = args.extract_end
        # print(len(story_raw_chunks_chapter))
        for i in tqdm(range(args.extract_begin, min(len(story_raw_chunks_chapter),extract_part))):  #断点重连

            save_name_sum = os.path.join(save_output_folder, f"{i}_sum.txt")
            # logger.info(f"save_name_sum   {save_name_sum}")
            # 断点重连

            # 跳过已存在文档
            if os.path.exists(save_name_sum):
                continue
                pass
            query_text_sum = ""
            summary_text_sum = ""
            for chapter_cut_text in story_raw_chunks_chapter[i]:

                #print(len(chapter_cut_text))
                query_text_sum += chapter_cut_text
                new_input = f"###\nText:\n{chapter_cut_text}\nSummarize in BULLET POINTS form:"

                prompt_SumCoa = copy.deepcopy(prompt_extra_story_template["summary_story_coarse"])
                prompt_SumCoa['query'] = new_input
                summarize_response = llm_chat_analysis("gpt-4o",prompt_SumCoa,"str")
                summary_text_sum += summarize_response
            # 文本过短，跳过
            if len(query_text_sum)<30:
                continue
            with open(save_name_sum, 'w', encoding='utf-8') as f:
                f.write( summary_text_sum )
            raw_text_save_name = os.path.join(save_output_folder, f"{i}_raw.txt")
            if not os.path.exists(raw_text_save_name) or os.path.getsize(raw_text_save_name) < 5:
                with open(raw_text_save_name, 'w', encoding='utf-8') as f:
                    f.write( query_text_sum )

            # break
    sum_cor_time_cost = time.time()-sum_cor_time
    logger.info(f"粗总结全文耗时: {sum_cor_time_cost}")


def split_string_by_lines(input_string, max_token_len):

# 按行与token切分
    cut_proportion = 2
    # 初始化结果列表
    str_result = []
    input_token = len(enc.encode(input_string))
    if input_token < max_token_len:
    #if len(input_string) < max_token_len:    
        str_result = [input_string]
        print(f"input char = {len(input_string)}, input token = {input_token}")
    else:
        len_str_ori = len(enc.encode(input_string))
        #len_str_ori = len(input_string)

        lines = input_string.split('\n')

        #for line in lines:
        #    print(line)
            
        str_result = [""]
        i = 0 
        if len_str_ori > max_token_len*cut_proportion:
            while i<len(lines):

                line_len = len(enc.encode( lines[i] ))  # token长度，可以换成字符长度
                if (len(enc.encode(str_result[-1]))+ line_len) < max_token_len:
                #if (len(str_result[-1])+ len(lines[i])) < max_token_len: 
                #    
                    str_result[-1] += lines[i] + '\n'
                else:

                    if len_str_ori <= max_token_len*cut_proportion:
                        break
                    str_result.append(lines[i])
                len_str_ori -= len(enc.encode(lines[i]))
                len_str_ori -= len(lines[i])
                i += 1
            
        #print("len1:", len_str_ori)
        len_str_ori /= 2
        str_result.append("")
        
        while (len(enc.encode(str_result[-1]))+len(enc.encode( lines[i] )) < len_str_ori):
        #while (len(str_result[-1])+len(lines[i]) < len_str_ori):
            str_result[-1] += lines[i] + '\n'
            i += 1
        str_result.append("")
        #print(str_result)
        
        for j in range(i,len(lines)):
            str_result[-1] += lines[j] + '\n'
    for i in range(len(str_result)):
        str_result[i] = str_result[i].strip('\n')

    cleaned_list = [item for item in str_result if item]
    return cleaned_list


def summary_story_fine(len_chapter_text,save_output_folder:str):
    # 对总结内容重新继续总结
    # 初始化一个列表来存储文件内容
    # 遍历文件夹中的文件
    sum_fine_time = time.time()
    k_num = 0
    for filename in sorted(os.listdir(save_output_folder)):
        
        # 检查文件是否以 '_sum.txt' 结尾
        if filename.endswith('_sum.txt'):
            k_num += 1
            file_path = os.path.join(save_output_folder, filename)
            file_path_plot = file_path.replace("_sum.txt","_resum.txt")

            if os.path.exists(file_path_plot):
                continue

            # 打开并读取文件
            logger.info(f"{filename} {k_num}/{len_chapter_text}")
            with open(file_path, 'r', encoding='utf-8') as file:
                content_plot = file.read()  # 读取文件内容
            con_list = split_string_by_lines(content_plot,args.integrate_max_token)
            #print(json.dumps(con_list,indent = 2,ensure_ascii=False))
            #break
            logger.info(f"该段文件总长度：{len(content_plot)}, 分为{len(con_list)}段")
            plot_text_sum = ""
            #"""
            for list_1 in con_list:
                logger.info(len(list_1))    # 输出每段文件长度
                if True:
                    prompt_SumFine = prompt_extra_story_template["summary_story_fine"].format(language=llm_ans_language,long_summary=list_1)
                else:
                    new_input = f"###\nText:\n{list_1}\nSummarize in BULLET POINTS form:"
                    prompt_SumFine = copy.deepcopy(prompt_extra_story_template["summary_story_fine_example"])
                    prompt_SumFine['query'] = new_input
                summarize_response = llm_chat_analysis("gpt-4o",prompt_SumFine,"str")
                plot_text_sum += summarize_response +'\n'
            #print(plot_text_sum)
            #break
            
            with open(file_path_plot, 'w', encoding='utf-8') as f:
                f.write( plot_text_sum )
            #break
            #"""
    sum_fine_time_cost = time.time()-sum_fine_time
    logger.info(f"精总结全文耗时: {sum_fine_time_cost}")

def combine_t2_jsonl(save_output_folder,data_output_file):
    # 将总结文件和原始文本标题(原文本第一行)结合
    # combine_list = []
    for filename in sorted(os.listdir(save_output_folder)):
        # 检查文件是否以 '_plot.txt'或者"resum.txt" 结尾
        if filename.endswith('_resum.txt'):
            content_plot_sum = ""
            #print("a")
            novel_plot_path = os.path.join(save_output_folder, filename)
            novel_raw_path = novel_plot_path.replace("_resum.txt","_raw.txt")
            with open(novel_plot_path, 'r', encoding='utf-8') as file:
                content_plot = file.read()  # 读取重新总结文件内容
            with open(novel_raw_path, 'r', encoding='utf-8') as file:
                content_raw = file.read()  # 读取源文件章节

            content_raw_list = content_raw.split('\n')
            i = 0
            while i < len(content_raw_list) and (not content_raw_list[i].strip()):
                i += 1
            #print( content_raw_list[i])
            content_plot_sum += content_raw_list[i]
            content_plot_sum += '\n\n'

            content_plot_sum += content_plot.strip('\n')
            # content_plot_sum += '\n\n\n'

            content_plot_sum = content_plot_sum.strip('\n')
            #print(content_plot_sum)
            with open(novel_plot_path, 'w', encoding='utf-8') as f:
                f.write( content_plot_sum )

            # combine_list.append(content_plot_sum)

            with jsonlines.open(data_output_file, mode='a') as writer:
                writer.write(content_plot_sum)
    logger.info(f"已存储{data_output_file}")


def extra_story(story_name):

    if args.novel_text_type == "epub":
        story_name = save_epub_2_txt(story_name)
    save_history_folder, save_output_folder, intermediate_data_folder,story_file,data_output_folder = establish_cor_folder(story_name)

    # 切片 - 切章节和chunk
    # 读取文件
    story_raw_text = read_raw_text(story_file)
    # 处理一下txt中出现的换行问题 
    if args.repair_raw_text  and args.use_language == 1:
        logger.info("英文语料，分隔符换行")
        story_raw_text = filiter_txt_line(story_raw_text) # 这里可能有问题·没有写返回

    raw_dialogue_count = count_dialogues(story_raw_text)
    story_raw_text1 = story_raw_text.replace('\n', '')
    logger.info(f"Total number of dialogues: {raw_dialogue_count}")
    logger.info(f"故事总字数: {len(story_raw_text1)}")

    # 切章节
    story_raw_chapters = split_chapter(story_raw_text)
    # 切块
    story_raw_chunks_sum, story_raw_chunks_chapter= split_chunk(story_raw_chapters,)
    # 剧情摘要
    summary_story_coarse(story_raw_chunks_chapter,save_output_folder)
    summary_story_fine(len(story_raw_chunks_chapter),save_output_folder)
    # 将总结文件和原始文本标题(原文本第一行)结合
    data_output_file = f"{data_output_folder}{story_name}_extra.jsonl"
    combine_t2_jsonl(save_output_folder,data_output_file)
    



def main():

    extra_story(args.source_story_name)


    pass

if __name__ == "__main__":
    main()