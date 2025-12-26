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
parser.add_argument("--data_output_folder", type=str, default="storyline_generator/story_framework_data", help="输出数据文件夹路径")
parser.add_argument("--combine_episode_num", type=int, default=1, choices=range(0, 5), help="合并粒度，x个章节合并成一个新章节，用于章节数很多的小说或者游戏，默认为1，推荐1-4，0代表全文合并，1代表保持原主线结构不变")
parser.add_argument("--seg_main_episode_num", type=int, default=10, choices=range(6, 15),help="主线划分为多少章节，推荐6-10")
# 重要！！语言
parser.add_argument("--use_language", type=int, default=0, choices=[0, 1], help="语言，0代表中文，1代表英文")
parser.add_argument("--story_branch_num", type=int, default=2, help="多线剧情发散度，生成结局数量等，(生成范围/+1)")
# 重要！！是否生成多线剧情
parser.add_argument("--if_single_episode", type=bool, default=False, help="是否生成单线剧情，不传此参数默认为多线剧情")
parser.add_argument("--if_role_known", type=bool, default=False, help="是否已知角色信息，不传此参数默认为 False")
parser.add_argument("--story_source", type=str, default="game", help="故事来源：小说文件/生成文件/电影电视剧名/游戏名/动漫名")
parser.add_argument("--story_source_type", type=str, default="name", help="故事来源类型：文件/名称")
parser.add_argument("--use_rag_know", type=int, default=0, help="是根据llm内在知识/rag生成/通过长文本抽取") # if_use_novel_doc = False



#if_use_novel_doc = True
# 小说中是否具有额外信息还是需要全部自己抽取
#if_novel_extra_info = False
args = parser.parse_args()

if args.combine_episode_num > args.seg_main_episode_num or args.combine_episode_num < 0:
    logger.error(f"warning! wrong parameter for combine_episode_num {args.combine_episode_num}")
if args.use_language == 0:
    llm_ans_language = "中文"
elif args.use_language == 1:
    llm_ans_language = "English"
else:
    logger.error(f"warning! wrong parameter for language {args.use_language}")


# 游戏列表,推荐剧情为主的rpg(Role-playing game),act(Action Game),avg(Adventure Game)

# rts 会有操作方面的提示，建议删除
game_name_list = [

    "生化危机7(Resident Evil 7)", 
    "生化危机8(Resident Evil 8)", 
    "生化危机2重制版(Resident Evil 2 Remake)", 
    "生化危机3重制版(Resident Evil 3 Remake)", 
    "生化危机4重制版(Resident Evil 4 Remake)", 
     
    "古墓丽影9(Tomb Raider)",
    "古墓丽影：崛起(Rise of The Tomb Raider)",
    "古墓丽影：暗影(Shadow of The Tomb Raider)",

    "死亡搁浅(Death Stranding)",
    "双人成行(It Takes Two)",
    
    # 10
    "行尸走肉(The Walking Dead)", 
    "夜之城:2077(Cyberpunk 2077)",
     
    "巫师三:狂猎(The Witcher 3: Wild Hunt)", 
    "巫师二:刺客之王(The Witcher 2: Assassins of Kings)", 

    "战神：诸神黄昏(God of War: Ragnarok)",
    "鬼泣5(Devil May Cry 5)",
    "尼尔：机械纪元(NieR: Automata)",
     
    "荒野大镖客2(Red Dead Redemption 2)",
    
    "逃生(Outlast)",
    "逃生2(Outlast 2)",
    "恶灵附身(The Evil Within)",
    "控制(Control)",

    "消逝的光芒1(Dying Light)",
    "消逝的光芒2:人与仁之战(DyingLight2:StayHuman)",
    "最后生还者(The Last of Us)",
    "最后生还者2(The Last of Us Part 2)",
    "看门狗2(Watch_Dogs 2)",
    "只狼：影逝二度(Sekiro：Shadows Die Twice)",
     
    "龙腾世纪:审判(Dragon Age: Inquisition)",  
    "质量效应(Mass Effect)",  # 有一系列
    "异界锁链(Chains of Promathia)", 
    "最终幻想 XV(Final Fantasy XV)", 
    "辐射：新维加斯(Fallout: New Vegas)",  

    "星际争霸2：自由之翼(StarCraft 2: Wings of Liberty)",
    "星际争霸2：虫群之心(StarCraft 2: Heart of the Swarm)",
    "星际争霸2：虚空之遗(StarCraft 2: Legacy of the Void)",
    "魔兽争霸3：混乱之治(Warcraft 3: Reign of Chaos)",
    "魔兽争霸3：冰封王座(Warcraft 3: The Frozen Throne)",

    "仙剑奇侠传1(Chinese Paladin: Sword and Fairy1)",
    "仙剑奇侠传2(Chinese Paladin: Sword and Fairy2)", 
    "仙剑奇侠传3(Chinese Paladin: Sword and Fairy3)", 
    "仙剑奇侠传4(Chinese Paladin: Sword and Fairy4)", 
    "仙剑奇侠传5(Chinese Paladin: Sword and Fairy5)", 

    "地铁：离去(Metro Exodus)",
    "地铁：2033(Metro 2033)",
    "最终幻想7(Final Fantasy VII)",
    "神秘海域(Uncharted)",
    "神秘海域2(Uncharted 2: Among Thieves)",
    "猎天使魔女(Bayonetta)",
    "小小梦魇(Little Nightmares)",
    "小小梦魇2(Little Nightmares II)",
    "寂静岭(Silent Hill)",
    "侠盗猎车手：罪恶都市(Grand Theft Auto: Vice City)",
    "侠盗猎车手：圣安地列斯(Grand Theft Auto: San Andreas)",
    "极乐迪斯科(Disco Elysium)",
    "黑暗之魂(Dark Souls)",
    "生化奇兵(BioShock)",
    "生化奇兵2(BioShock 2)",
    "生化奇兵3：无限(BioShock Infinite)",
    "半条命(Half-Life)",
    "半条命2(Half-Life 2)",
    "战神(God of War)",
    "刺客信条:起源(Assassin's Creed: Origins)",
    "刺客信条:奥德赛(Assassin's Creed: Odyssey)",
    "使命召唤4:现代战争(Call of Duty 4: Modern Warfare)",
    "使命召唤7:黑色行动(Call of Duty: Black Ops)",
    "使命召唤10:幽灵(Call of Duty: Ghosts)",

]

movie_name_list = [

    "肖申克的救赎(The Shawshank Redemption)",
    "楚门的世界(The Truman Show)",
    "变相怪杰(The Mask)",
    "冒牌天神(Bruce Almighty)",
    "这个杀手不太冷(Léon)",
    "泰坦尼克号(Titanic)",
    
    "蝙蝠侠：侠影之谜(Batman Begins)", 
    "蝙蝠侠：黑暗骑士(Batman The Dark Knight)", 
    "蝙蝠侠：黑暗骑士崛起(Batman The Dark Knight Rises)",

    "蜘蛛侠：平行宇宙(Spider-Man: Into the Spider-Verse)", 
    "变形金刚(Transformers(2007))",
    "变形金刚2(Transformers: Revenge of the Fallen(2009))",
    "变形金刚3：月黑之时(Transformers: Dark of the Moon (2011))",
    "变形金刚4：绝迹重生(Transformers: Age of Extinction (2014))",
    "变形金刚5：最后的骑士(Transformers: The Last Knight (2017))",

    "黑客帝国(The Matrix)",
    "黑客帝国2：重装上阵(The Matrix Reloaded)",
    "黑客帝国3：矩阵革命(The Matrix Revolution)",

    "逃出绝命镇(Get Out)",
    "寄生虫(Parasite)",  
    "异次元杀阵(心慌方, Cube)", 
    "疾速追杀(John Wick)",   
    "盗梦空间(Inception)",
    #22
    "当幸福来敲门(The Pursuit of Happyness)",
    "飞屋环游记(Up)",
    "飞越疯人院(One Flew Over the Cuckoo's Nest)",
    "千与千寻(千と千尋の神隠し, Spirited Away)",

    "2001太空漫游(2001: A Space Odyssey)",
    "银翼杀手(Blade Runner)",
    "阿凡达(Avatar)",
    "疯狂的麦克斯：狂暴之路(Mad Max: Fury Road)",  

    "疯狂动物城(Zootopia)",
    "杀死比尔(Kill Bill)",
    "查理的巧克力工厂(Charlie and the Chocolate Factory)",
    "驯龙高手(How to Train Your Dragon)",
    "狮子王(The Lion King)",
]




# 键值对模板
# story，plot，messages
template_dict_story = {
    "Settings":{
        "story_id": "",         # int, 唯一id，用于查找
        "story_name": "",       # str, 故事名
        "story_style": "",      # 故事风格，尽可能超过5个短语
        "leading_role":"",      # 每个episode中都要出现主角
        "story_desc": "",       # 故事的主要描述
        "story_goal":"",        # 故事主角需要做的目标
        "story_chars": {},      # dict, 故事有哪些主角,性格以及人物描述 
        "storylines":[],        # list, 故事主线梗概
        "story_info":{
            "epi_struct":"",    # 结局结构(单线/多线)
            "gen_end_num":0,        # int, 实际结局数量
            "pre_epi_node":{},  # dict, 前驱节点
            "language":"",      # 故事语言(中文/英文)
            "sources":"",      # 故事来源，从哪里获得的，game/novel...
        },     
        "states":[],            # 多个dict组成的list, 每个dict包含顺序，以及先后评分)
                
        "gen_parameter":{}      # 生成故事使用的超参数
    },
    "Episodes":[]   # plots节点, plot 是list，多个下面的plot
}

template_dict_episode = {
    "episode_id": "",
    "pre_episode_id":[],
    "episode_goal": "",
    "episode_scene": "",
    "episode_desc": "",
    "episode_chars": {},
    "extra_info": [],
    "triggers": [
#        {
#            "condition": "",
#            "examples": [],
#            "next_episode": "",
#        },
    ],
    "message_pairs":[],
}
template_dict_message =  {
    "task": ["transition", "role-play", ""],
    "user": "@npc content_text",
    "bot": {
        "narrative": "", 
        "role_dialogue": [
            {"name": "", 
             "utterance": "",
             },
        ],
        "next_episode": ""
    }                       
}
prompt_gen_story_template = {
    # 提取背景
    "summary_background": PromptTemplate(
        template = """请依次回答介绍在{novel_name}中以下内容：
故事风格(使用几个短语形容，使用逗号隔开,7个词以上)，发生的背景(时间地点人物, 以第三人称风格介绍)，剧情中主角的主要目标，主角(有且仅能有一个，如果是多主角，请选择一个作为主角)分别是什么
介绍时不要出现"玩家"，"游戏"等令人出戏的词语
保持结果的格式和例子中的格式一致, 并以键值对的json格式: '{{key:value, ...}} '重构。
用{language}回答,不要提到玩家等令人出戏的字眼
下面是一个例子（取自游戏"巫师3"）
{{
    "story_style":"奇幻,中世纪，成人向叙事,复杂阴谋,黑暗幻想,史诗冒险,传奇主义色彩,哥特式神秘感,魔法与剑",
    "background":"这是一个辽阔的幻想世界，这个世界充满了魔法、怪兽和战争，猎魔人杰洛特是这个世界中的一名传奇人物，他的养女希里拥有强大的魔法能力，被来自另一个维度，拥有强大的力量且神秘的狂猎势力追捕。杰洛特为了寻找并保护希里，踏上了一段跨越战争蹂躏的大陆的冒险旅程，同时卷入了北方王国与尼尔夫加德帝国之间的冲突。猎魔人是被训练来对抗怪物的变种人类，尽管他们拥有超自然力量，却常被社会边缘化。",
    "target":"杰洛特的主要目标是是寻找并保护他的养女希里，她是一个拥有强大力量的源术士，目前正在被神秘的狂猎骑士团追捕，同时杰洛特还需要揭露狂猎的计划，最终阻止他们。在这个过程中需要探索一个充满危险和奇幻元素的广阔开放世界，解决各种任务和委托，与怪物战斗，同时做出影响剧情走向的决策。杰洛特还需要不断提升自己的能力，包括剑术、魔法和装备，以应对越来越强大的挑战。",
    "leading_role":"杰洛特"
}}
    """,
        input_variables=["novel_name","language"],
    ),

    # 从已有故事数据中提取背景
    "summary_background_with_data": PromptTemplate(
        template = """请根据以下故事内容，提取关键信息：

故事名称：{novel_name}

故事内容：
{story_data}

请提取以下信息并以JSON格式返回：
1. 故事风格（使用7个以上的短语描述，用逗号分隔）
2. 故事背景（时间、地点、主要人物，以第三人称叙述）
3. 主角的主要目标
4. 主角（如果有多个主角，请选择最重要的一个）

注意：
- 不要出现"玩家"、"游戏"等令人出戏的词语
- 保持客观叙述 
- 使用{language}回答
- 返回格式：{{"story_style": "风格1,风格2,...", "background": "背景描述", "target": "目标描述", "leading_role": "主角名称"}}

下面是一个例子（取自游戏"巫师3"）：
{{
    "story_style":"奇幻,中世纪，成人向叙事,复杂阴谋,黑暗幻想,史诗冒险,传奇主义色彩,哥特式神秘感,魔法与剑",
    "background":"这是一个辽阔的幻想世界，这个世界充满了魔法、怪兽和战争，猎魔人杰洛特是这个世界中的一名传奇人物，他的养女希里拥有强大的魔法能力，被来自另一个维度，拥有强大的力量且神秘的狂猎势力追捕。杰洛特为了寻找并保护希里，踏上了一段跨越战争蹂躏的大陆的冒险旅程，同时卷入了北方王国与尼尔夫加德帝国之间的冲突。猎魔人是被训练来对抗怪物的变种人类，尽管他们拥有超自然力量，却常被社会边缘化。",
    "target":"杰洛特的主要目标是是寻找并保护他的养女希里，她是一个拥有强大力量的源术士，目前正在被神秘的狂猎骑士团追捕，同时杰洛特还需要揭露狂猎的计划，最终阻止他们。在这个过程中需要探索一个充满危险和奇幻元素的广阔开放世界，解决各种任务和委托，与怪物战斗，同时做出影响剧情走向的决策。杰洛特还需要不断提升自己的能力，包括剑术、魔法和装备，以应对越来越强大的挑战。",
    "leading_role":"杰洛特"
}}
    """,
        input_variables=["novel_name","language","story_data"],
    ),

    # 抽取剧情主要剧情线
    "summary_main_plot":PromptTemplate(
        template="""在游戏/电影{novel_name}中，与主角「{leading_role}」相关的主要剧情线是什么? 请完整介绍最主要的剧情线（单线）,请尊重原始剧情，不要遗漏重要情节
需要每章节内容均衡，所有章节总体可以概括整个剧情，维持总结后的章节数在{seg_main_plot_num}章左右
plot的顺序应该按时间顺序（线性叙事），不要出现倒叙、插叙、多时间线，第一个plot应该是自然时间最早的节点，最后一个plot是时间最晚
在各个plot的情节中应该有所差异，但是临近的plot应该有所关联
确保每个情节点之间有逻辑连接, 但是不同情节中的剧情不能重复
注意,在每个plot中都需要出现story中最重要的主角{leading_role},需要强调主角

使用如下格式`总结:详细内容`，保持结果的格式和例子中的格式一致, 并整体以json格式: '{{"main":[xx:xx, xx:xx, ...]}} '重构。 
使用{language},下面是一个例子：
寻找希里:主角杰洛特寻找他的养女希里，发现她被狂猎追捕。
追踪狂猎:杰洛特开始调查狂猎的活动，了解他们的目标和动机。
政治纷争:杰洛特在调查狂猎时卷入北方王国之间的战争与政治斗争，选择支持的势力。
...
        """,
        # 维持总结后的章节数在{seg_main_plot_num}章左右
        # 不能低于{seg_main_plot_num}个plot，根据剧情长度适当增加plot数量
        #input_variables=["novel_name","language"],
        input_variables=["novel_name","seg_main_plot_num","leading_role","language"],
    ),   

    # 剧情重组织之后生成，目前未使用
    "summary_combine_plot":PromptTemplate(
        template = """进一步的，在{novel_name}游戏中，上述答案中的 {plot_combine}部分，更加详细的剧情是什么? 重新划分段落详细叙述，每行所占的剧情数应该相同
            每个剧情前用(plot *),*是对应序号，表示这是第几幕出现的，序号应该连续递增，从1开始
            剧情大概有{seg_main_plot_num}个
                
            仅仅解释{plot_combine}的部分，不要与其他部分内容重合
            使用如下格式`总结:详细内容`, 保持结果的格式和例子中的格式一致, 并以json格式: 'main:[xx:xx, xx:xx, ...] '重构。 
            用{language}回答,下面是一个例子:

            (plot 1)调查遗迹:探索古代遗迹，了解希里与古老力量的关系。
            (plot 2)遭遇敌人:面对追寻希里的敌对势力，保护自己和寻找线索。 
            (plot 3)发现藏身处:最终找到希里的一些藏身之处，获取更深入的信息。
            (plot 4)...
            """,
        input_variables=["novel_name","plot_combine","seg_main_plot_num","language"],
    ),

    # 重写多线剧情prompt
    # 跳转合理：如果两个分支指向同一个剧情，那么这两个分支的跳转必须符合不同分支的剧情逻辑。所有结局要和对应剧情逻辑保持一致。
    # 在产生分支剧情时原剧情应该有所提示，分支剧情的输出位置尽可能贴近分支节点; 比如在之前的剧情中间插入分支，而不是只在末尾加入分支，并将目前整体的剧情重构
    "rewrite_multi_plot":PromptTemplate(
        template = """
你是一个剧情策划，需要将我给你的单线剧情重构为多分支叙事结构，单线剧情是{novel_name}游戏/电影中的一部分

# 核心原则
1. 剧情重写
 - 新剧情需要在原有剧情的基础上进行扩展和分支，产生额外分支剧情和新结局
 - 详细描述每个情节和结局，包括发生的事件、跳转原因、触发条件
 - 所有的剧情都应该围绕主角{leading_role}展开,且主角应该向游戏目标{game_target}努力，但是应该保持原有故事风格{game_style}
 - 你的输出包含两种节点: plot节点和end节点。plot节点是中间剧情节点; end是结局节点, 作为一条故事线的结束
2. 保持原始骨架
 - 必须保留原有剧情和结局，不能完全重写或忽略原剧情
 - 新分支应从原剧情的中间节点开始产生，而不是只在原剧情结束后才产生
 - 原最终节点必须作为新分支的第一个结局(end 1)
3. 三维叙事设计
 - 横向拓展: 每个节点不超过2个分支, 不同plot之间的内容不能重复或者近似
 - 纵向延伸: 每个分支需包含4-6个递进plot节点后到达结局, 在原有前期剧情(前3个章节)中间, 应该至少出现一个新分支plot, 即第一个你创造的节点分支在plot2-3里面
 - 垂直深化: 确保每个结局体现: 人物命运的根本性差异(生存/死亡/蜕变)/世界观的不同侧面展现, 结局不能同质化，确保不同结局的人物命运和事件结果有显著差异;
 - 树状延伸: 在分支节点有核心冲突的差异化方案，做出不同抉择; 但不同结局只是由于行动或者抉择导致的不同，不应该偏离原始角色设定，不同结局需要逻辑自洽
4. 因果链约束
 - 每个选择必须触发【蝴蝶效应】，使后续剧情产生本质区别
 - 当一个plot面临抉择时, 其必须产生不同节点的分支
 - 产生的选择必须满足: 前序决策逻辑不冲突; 角色行为模式保持一致; 世界观设定无矛盾; 剧情保持连贯
 - 每个分支的跳转逻辑必须清晰合理，确保玩家能够理解不同选择导致的剧情走向变化。

# 技术规范
1. 结构要求
 - 节点数量: plot不要超过18个, end不能超过3个(即剧情线不要超过3条)
 - 路径唯一: 不能产生环结构, 除起始节点(没有from)外, 其他每个节点有且只能有一个前置节点(from只有一个)；结局的数量就代表一共有几条剧情线
 - 树状结构: 确保从起始事件plot1(根节点)可以达到其他所有事件, 而且所有plot都可以通过路径到达至少一个end节点(plot可以看作是非叶子节点, end是叶子节点); (即没有前置节点或后续节点的plot)只有end才能作为最终结局
 - 禁止出现孤立plot: 对于除了plot1的plot(非叶子节点), 必须被一个其他节点指向(有且只能有1个前置节点); 必须至少有一个后续节点(至少有一个后续节点, 也就是个plot必须可以跳转到其他事件)
2. 内容规范
 - 每个分支plot需明确: 核心冲突(人物 vs 环境/他人/自我); 关键决策点(必须影响后续分支走向); 状态转变（角色能力/关系/认知的变化）
 - 每个end需体现: 核心矛盾的终结方式; 主角{leading_role}的最终状态; 对游戏目标{game_target}的达成程度

# 输出格式
严格遵循以下结构：
* 仅是一个int数
(plot *)【章节标题】: 核心事件描述。关键决策点(from plot *)
(end *)【结局名称】: 结局描述及对{game_target}的影响程度(from plot *)
具体要求: 
 - 在剧情后仅用(from plot *)指示从何plot跳转而来, 不要在from后写跳转原因
 - 所有行的开头都应该是(plot *)或者是(end *)，代表剧情节点和结局节点
 - 除了plot节点和end节点不要输出其他内容(比如设计原因, 总结等)
 - 你的输出应该从plot1开始, 序号需要连贯; 先输出完所有plot, 再输出end

示例(参考格式而非内容): 
```
(plot 1)寻找希里:杰洛特寻找他的养女希里，她被狂猎追捕。(begin)
(plot 2)战场的鲜血:希里情绪低落，需要做出选择以将决定希里是否能够控制自己的力量。(from plot 1)
(plot 3)希里女王支线:"战场上的鲜血"任务选择去带希里参见皇帝。(from plot 2)
(plot 4)希里猎魔人支线:"战场上的鲜血"任务选择去追杀狂猎。(from plot 2)
(plot 5)为她铸剑:杰洛特为希里铸造一把钢剑，并在百果园的酒馆将剑送给她，象征着她成为一名真正的猎魔人。(from plot 4)
(plot 6)各奔东西:希里会独立去探索世界，而杰洛特则会继续他的猎魔人生涯。(from plot 5)
(plot 7)共同作战:希里与杰洛特一起成为猎魔人，为追杀狂猎做准备(from plot 5)
(plot 8)寻访猎魔大师:杰洛特听说陶森特有一位猎魔大师结束了混乱的吸血鬼屠杀事件，决定去找他交流经验(from plot 6)
(end 1)希里的堕落:希里在成为女王后，被权力所诱惑，与狂猎达成交易，最终同流合污(from plot 3)
(end 2)胜利与牺牲:虽然最终击败了狂猎，但是希里为了保护她所爱的人选择了牺牲自己(from plot 7)
(end 3)重见故友:杰洛特发现猎魔大师就是希里。她在庄园的树下与杰洛特重逢(from plot 8)
```
参考内容：
游戏发生的背景如下:{game_background}
用{language}回答,原单线剧情树如下:{main_plot_str}
        """,
        input_variables=["novel_name","main_plot_str","game_background","game_target","game_style","language","leading_role"],
    ),
    # 1. 生成具体plot的events，详细剧情
    # 内容形式像讲故事一样，
    "expand_plot_detail": PromptTemplate(
        template = """请根据摘要和先前剧情编写当前摘要发生的故事（详细介绍当前摘要剧情的具体情节）
要求如下：
- 延续游戏{novel_name}中的人物，剧情，背景设定，但是可能并不是游戏中实际的剧情
- 请严格按照摘要的范围，不要扩展到未提及的后续剧情。
- 采用第三人称白描手法，禁用元叙述词汇（例如"整个故事中","在游戏中","好的","以下是关于xx的详细介绍"这种令人出戏的回复）
- 与之前的剧情连贯性，没有突兀的剧情跳跃，但是不要出现重复情节和冲突
- 用{language}回答,保持回答长度在300字符左右。

该段剧情是整体剧情的{plot_num_frac}部分,该段剧情摘要如下: {plot_str}
可供参考先前的剧情如下{pre_plot}       
    """,
        input_variables=["novel_name","plot_str","plot_num_frac","language","pre_plot"],
    ),

    # eva_score, 对生成的详细剧情进行评估
    # - 人物信息: {story_chars}
    "evaluate_plot_detail": PromptTemplate(
        template = """你作为故事分析师, 请根据以下标准严格评估剧情树, 严格寻找问题与不足, 尽可能严厉, 给出专业评分(0-100整型数), 评分须严格遵循标准，扣分必须注明具体原因。

### 评分维度及权重(总分100)
  **基础结构(20分)**: 
    - 每个章节是否以主角[{leading_role}]视角展开
    - 主角的核心目标是否明确且贯穿始终
    - 不同章节发生事件是否保持唯一(含非相邻章节), 没有重复事件
    - 时间轴是否严格单向流动(无倒叙/插叙)
  **连贯性(20分)**: 
    - 整体章节剧情是否连贯
    - 相邻章节是否保持时间/逻辑连续性, 剧情不跳跃
    - 相邻章节是否没有重叠内容(相邻的章节剧情应该连续但不相交)
    - 非相邻章节是否通过伏笔或线索形成隐性关联  
  **合理性(20分)**: 
    - 剧情整体是否形成完整闭环
    - 前后剧情逻辑是否合理, 事件发生是否自然; 
    - 剧情是否存在矛盾或者发生重复事件
    - 剧情是否符合世界观以及对应人物性格/行为
  **完整性(20分)**: 
    - 整体剧情是否完整(例如: 后期突兀出现没有提到的关键道具/人物; 结局结束过于突然)
    - 有无节奏问题(例如: 拖沓/突兀转折/不自然中断)
    - 前期剧情是否对后期剧情有所影响(例如: 前期细节/道具在后期转化为关键线索/发生作用/反转, 首尾呼应)
  **吸引性(20分)**: 
    - 整体故事是否有趣/引人入胜
    - 整体剧情是否节奏合理(例如: 起承转合)
    - 是否有引人入胜的转折点,悬念和高潮
  **注意**: 剧情树的序号可能不连贯, 仅是作为剧情的唯一标识, 不作为连贯性的评判标准, 无需修改, 只需关注序号后的实际剧情

### 输出格式
返回一个json格式的字典, 请用{language}输出JSON   
包含两个键, 评分的原因以及最终的具体分数
{{
    "reason": "(各个指标的（评分/总分）以及扣分原因)"
    "score": 最终评分(仅数字)
}}

### 上下文信息
- 故事风格: {story_style}
- 世界观设定: {story_desc}
- 核心目标: {story_goal}
- 当前剧情树: {episode_seq}   
    """,
        input_variables=["leading_role","episode_seq","story_style","story_desc","story_goal","language"],
    ),

    # refine_score, 对生成的详细剧情进行优化
    "optimize_plot_detail": PromptTemplate(
        template = """你作为专业作家, 接下来我将给你一个剧情大纲评价, 请根据剧情大纲以及评价对你剧情的不足之处进行修改
首先根据总体大纲评价找到不合理的章节, 若发现不合格章节, 请输出包含问题章节的JSON对象, 说明对应具体章节的不合理原因，改进建议以及改进后的章节内容(只需指出和改进不合理的章节) 

# 修改约束
1. 严禁删除/新增章节！仅可修改现有章节内容
2. 章节剧情冲突/重复/逻辑矛盾时优先修改后续章节
3. 如果你建议修改结局/最后一个章节, 那么你改进的结局需与其他存在的结局有本质差异。 
4. 无冲突/重复: 修改后前后剧情一致, 不冲突不重复; 修改后的剧情是否符合人物行为性格和世界观; 
5. 剧情合理: 对故事进行修改时，应保持其整体风格, 基调, 人物整体命运不变; 后期关键道具/人物应该提前铺垫
6. 剧情连贯: 相邻的章节剧情应该连续但不相交, 内容不重叠不跳跃; 时间轴严格单向流动(无倒叙/插叙); 
7. 剧情主线一致(以主角[{leading_role}]视角展开)
 - 注意: 剧情树的序号可能不连贯, 仅是作为剧情的唯一标识, 不作为连贯性的评判标准, 无需修改, 只需关注序号后的实际剧情

# 输出格式(json),请用{language}输出JSON:
{{"(章节序号)":{{
        "存在问题": "(对应章节的不合理原因)",
        "改进建议": "(如何修改这部分章节)",
        "改进后章节": "(改进后的章节内容)",
    }},
    ...(多个有问题的章节)
}} 

# 上下文信息
- 故事风格: {story_style}
- 世界观设定: {story_desc}
- 核心目标: {story_goal}
- 其他已经存在的结局: {other_ans}
- 当前剧情大纲: {episode_seq}
- 剧情大纲的不足之处: {outline_refine_reason}
    """,
        input_variables=["leading_role","language","outline_refine_reason","episode_seq","story_style","story_desc","story_goal","other_ans"],
    ),
    "supple_plot_info": PromptTemplate(
        template = """请用{language}整理「{novel_name}」游戏中的某段plot中的以下内容
plot摘要如下:{plot_str}
(该段剧情是整体剧情的{plot_num_frac}部分)
# 整理内容
▶ target: 以主角「{leading_role}」视角说明本段plot的核心目的(仅限目的，不涉及本段剧情实际结果)
▶ scene: 本段plot的背景和它发生的地点/时间/环境等, 需要结合当前plot中即将发生的剧情进行介绍,但是不要涉及到当前plot发生的内容
▶ role: 本段plot中出现的主要人物, 需要介绍性格与当前情节的状态等关键信息, 人物数量小于5个，注意：必须包含主角「{leading_role}」且名称应该完全一致，并且出现在role列表中的第一个

# 格式规范
 - 使用如下格式`**:详细内容`, 保持结果的格式和例子中的格式一致
 - 输出以json格式: '{{key:value}}(键值对的形式dict) '重构
 - 禁止使用「玩家」「游戏」「剧情中」等出戏表述
 - 保持叙事内视角的一致性
        
# 示例结构
{{
    "target":"劳拉的主要目标是荒岛上生存下来，找到并解救她的团队成员，同时探索荒岛。",
    "scene":"劳拉和她的团队在寻找邪马台古国的途中遭遇风暴，船只失事，劳拉漂流到一个未知的岛屿上。这个岛屿是日本以东、福尔摩沙龙三角中的一座小岛——邪马台。那座岛屿以及曾经存在于岛上的王国充满着神秘的传说，许多飞机和船只都曾在附近遭遇狂风暴雨而失事，残骸布满着它的海岸，因而闻名。",
    "role":[
        "劳拉·克劳馥: 幼稚单纯且充满自由精神和荒野情怀。此时劳拉还只是一个21岁的刚踏出校园的小女孩。对亡故的父亲的不解和对冒险的渴望促使劳拉和志同道和的同窗好友珊曼莎一起乘上了驶往邪马台古国的坚忍号，在寻找邪马台古国的途中，劳拉所在的船只遭遇风暴失事，她漂流到一个未知的岛屿上，孤身一人在岛上醒来，被一个野蛮人从背后袭击。醒来后发现自己被吊在一个类似祭台的环境中，周围满是尸骨，开始她的求生之旅。",
        "康拉德·罗斯: 睿智，理性，坚韧而一丝不苟。坚忍号探险船的船长，52岁的资深冒险家，曾在皇家海军陆战队突击旅服役。他与劳拉的父母私交颇深，对队员们的安危非常关心，是劳拉的老师和精神领导者。在风暴中与劳拉一同遭遇船难，之后在岛上与劳拉重逢，共同面对岛上的危险。",
        "珊曼莎·西村: 大大咧咧，习惯过奢侈阔绰的生活。实习纪录片制片人，也是劳拉的同窗好友。她拥有葡萄牙和日本的双重国籍，家境殷实。珊曼莎是太阳女王卑弥呼的后裔，珊曼莎下落不明，但是的背包被劳拉找到，里面有一台DV、火柴和对讲机。",
    ]
}}""",
        input_variables=["novel_name","plot_str","plot_num_frac","leading_role","language"],
    ),   
    "supple_plot_trigger": PromptTemplate(
        template = """基于「{novel_name}」世界观，用{language}分析从当前剧情向下一情节过渡条件(trigger)
# 核心要求
 - 使用<IF>(标注情节转换的关键原因)和<THEN>(标注触发原因后产生的即时结果)作为标识符; <IF>必须直接关联当前核心目标「{plot_target}」, <THEN>必须自然衔接下一情节
 - 简洁明了地阐述从当前情节跳转到下一个情节的关键原因, 如果有其他过渡条件, 当前跳转原因应该与其有较大差异
 - 人物动机需符合性格设定, 事件发展需符合世界观规则; 采用「内因驱动>外因推动」的优先级
 - 避免提及「玩家」「游戏」「剧情中」元叙事词汇
   
# 格式示例
<IF> **在当前剧情中跳转需要达成的条件** <THEN> **达成条件后触发的关键事件(事件承上启下，发生于下一段剧情之前)，导致剧情向下一个情节发展**(请根据实际情况填写**中的内容)

# 上下文内容
 - 该段剧情是整体剧情的{plot_num_frac}部分
 - 当前plot摘要:{plot_str}
 - 待衔接plot摘要:{next_episode_str}
 - 其他过渡条件如下: {else_triggers}
 - 故事风格: {story_style}; 世界观: {story_desc}; 主要人物: {story_chars}
    """,
        input_variables=["novel_name","plot_str","next_episode_str","plot_target","plot_num_frac","else_triggers","story_style","story_desc","story_chars","language"],
    ),
    "supple_end_trigger": PromptTemplate(
        template = """你是一个作家, 我将会给你一个结局剧情, 基于「{novel_name}」世界观，用{language}分析
# 核心要求
 - 你需要生成三个部分「最终章流程」「结局触发点」「结局演出」
 - 完全基于当前结局剧情,不能改写，只能扩写
 - 「最终章流程」是可互动剧情
 - 「结局触发点」只是一个条件或者触发的行为等,不包括后续内容, 尽可能短(例如:击败boss(注意只是击败这个条件而非开始对战), 主角或者某个npc阵亡, 找到某个关键道具, 对战持续多久后等情况, 需要和当前剧情摘要保持一致))
 - 「结局演出」应是最终一个结局，单一事件(「结局触发点」产生的结果),主角不能做出影响事件的抉择, 这部分可以留有悬念
 - 如果原结局剧情包含了结尾，需要将原结局剧情拆成这三个部分; 如果没有包含结尾, 将原结局剧情整理成「最终章流程」,在流程之后续写「结局触发点」「结局演出」两个部分
 - 简洁明了地阐述从当前故事结束的关键原因
 - 人物动机需符合性格设定, 事件发展需符合世界观规则
 - 避免提及「玩家」「游戏」「剧情中」「故事」元叙事词汇
   
# 格式示例
json格式的dict
```
{{
    "最终章流程":"(保留所有可操作内容（战斗/解谜/对话分支）)",
    "结局触发点":"(达成某项事件或者触发某个条件",
    "结局演出":"(达到结局触发点后触发的最终结局，不能互动,需要与主角强相关「{leading_role}」)",
}}
```

# 上下文内容
 - 当前plot如下: {episode_desc}
 - 故事风格: {story_style}; 世界观: {story_desc}; 主要人物: {story_chars}
    """,
        input_variables=["novel_name","leading_role","episode_desc", "story_style","story_desc","story_chars","language"],
    ),


    "evaluate_plot_info": PromptTemplate(
        template = """请根据总体章节的信息严格分析当前章节的目标，场景和剧情摘要，并说明原因（尽可能严格，寻找问题与不足之处）
# 评价内容(可以参考但不限于以下内容)
1. 当前章节的目标，背景和剧情是否按照总目标前进
2. 是否总是以主角{leading_role}作为展开
3. 是否符合人物总体的行事风格以及观念
4. 剧情展开是否合理，有没有逻辑漏洞或者难以理解的发展过程
5. 剧情是否符合故事的总体风格

# 输出规范
 - 用{language}回答
 - 以dict的json格式重构
 - 格式如下: "main":{{"episode_goal":"**", "episode_scene":"**", "episode_desc":"**"}} '(**是对应的具体评价及原因)

# 上下文信息
 - 需要评估的原章节内容如下：
原章节的目标: {episode_goal}; 原章节的场景: {episode_scene}; 原章节的剧情摘要: {episode_desc},
 - 其他补充信息:
故事风格: {story_style}; 总体背景: {story_desc}; 主要目标: {story_goal}; 人物信息: {story_chars}
    """,
        input_variables=["leading_role","language","episode_goal","episode_scene","episode_desc","story_style","story_desc","story_goal","story_chars"],
    ),

    # triggers eva only
    "evaluate_trigger": PromptTemplate(
        template = """你是一位资深故事结构分析师，专门负责负责章节间的逻辑衔接(跳转事件trigger)的评分, 根据总体章节信息严格评价当前章节跳转到下一情节的原因及跳转事件(验证是否符合「 当前plot → <IF>触发 → <THEN>衔接 → 下个plot 」时序链), 严格寻找问题与不足, 尽可能严厉, 给出专业评分(0-100整型数), 评分须严格遵循标准，扣分必须注明具体原因。
    
# 重点关注项
 - 时间连贯性(20分): <IF>是否存在于当前章结尾处延展的时间点? <THEN>是否构成独立于下章开篇的衔接事件? trigger是否起到承上启下的作用?
 - 剧情流程性(20分): 两段plot通过当前跳转事件连接是否流畅自然? <IF>是否在当前情节之中可以达成，但是发生在当前情节之后的跳转条件?
 - 事件合理性(20分): 跳转事件是否合理、符合逻辑，与故事背景/情节, 人物行为/风格一致?
 - 主体关联性(20分): 触发条件是否由主角「{leading_role}」的行为直接/间接引发？
 - 逻辑完整性(20分): <IF>是否为单一可验证条件, 且仅为条件而非事件描述? <THEN>是否包含独立事件而非章节内容复述?
 - 注意: 仅关注触发器<IF><THEN>结构, 忽略文学性, 保持分析客观性
 
# 上下文背景
当前plot如下: {episode_desc}
下一plot摘要如下: {next_episode_desc}
需要修改的跳转事件如下: {trigger_condition}
故事风格: {story_style}; 世界观: {story_desc}; 主要人物: {story_chars}

# 输出格式
 - 返回一个json格式的字典,包含两个键, 评分的原因以及最终的具体分数
示例如下:
{{
    "reason": "(各个指标的（评分/总分）以及扣分原因)"
    "score": 最终评分(仅数字)
}}
    """,
        input_variables=["leading_role","episode_desc","next_episode_desc","trigger_condition","story_style","story_desc","story_chars"],
    ),
    # triggers opt only
    # 
    "optimize_trigger": PromptTemplate(
        template = """你是一位资深作家，专门负责分析改进章节间的逻辑衔接(跳转事件trigger), 接下来我将给你一个当前的trigger评价, 根据总体章节信息与评价，严格分析当前章节跳转到下一情节的原因及跳转事件中断问题与不足, 提供trigger的改写办法(保持当前格式，不修改章节内容), 精简回答

# 修改规范
仅关注触发器<IF><THEN>结构, 忽略文学性, 保持分析客观性
<IF>必须满足:
    - 包含主角行为影响(触发条件由主角「{leading_role}」的行为直接/间接引发)
    - 是<THEN>发生的直接原因, 是当前情节后可能产生的行为, 不与当前plot内容重叠, 与当前plot流畅衔接
    - <IF>存在于当前章结尾处延展的时间点, 即<IF>是在当前情节之中可以达成，但是发生在当前情节之后的跳转条件
    - <IF>为单一可验证条件, 且仅为条件而非事件描述(只是一个触发的原因, 即仅包含触发条件, 而非具体事件)
    - <IF>是<THEN>及以后事件发生的关键原因, 应该仅包含主角/或主角导致的一个关键动作
<THEN>必须满足:
    - 是<IF>产生的直接结果, 是下一章节发生之前的内容, 不与下一plot内容重叠, 与下一plot流畅衔接
    - <THEN>是构成独立于下章开篇的衔接事件
    - <THEN>包含独立事件而非章节内容复述
    - <THEN>是<IF>达成后产生的事件结果, 只是一个触发原因导致的事件(直接后继节点)
整体上: 
    - 不要复述输入, 只需要介绍当前trigger的不足之处并说明原因, 提供改进办法
    - 「 当前plot → <IF>触发 → <THEN>衔接 → 下个plot 」这个顺序必须流畅连贯自然; 不要产生突兀转折, 重复剧情以及剧情断裂/脱节/中断
    - 如果有一个通用的语句: "当***, 则***", 那么<IF>就是「当***」的部分,<THEN>就是「则***」的部分
    - trigger需要起到承上启下的作用, 两段plot通过当前跳转事件连接的流畅自然
    - 修改后的跳转事件合理、符合逻辑，与故事背景/情节, 人物行为/风格一致
    - <IF>与<THEN>不要与同章节的其他跳转条件重复, 触发原因和产生的结果都应该具有较大差异
    
# 上下文背景
当前plot如下: {episode_desc}
下一plot摘要如下: {next_episode_desc}
需要修改的跳转事件如下: {trigger_condition}
跳转事件评估结果如下: {trigger_eva_think}
{other_triggers}
故事风格: {story_style}; 世界观: {story_desc}; 主要人物: {story_chars}

# 输出格式
 - 严格使用JSON格式,仅输出一个dict, 不要输出其他内容
 - 使用「{language}」回答
 - dict的键名称请严格按照示例,有且仅有三个key，key使用英文并保持大写,且key名称为:"OPTIMIZE","IF","THEN"
示例如下:
```
{{
    "OPTIMIZE":"(提出你的优化方法与依据)",
    "IF":"(改进后的IF)",
    "THEN":"(改进后的THEN)"
}}
```
    """,
        input_variables=["leading_role","language","episode_desc","next_episode_desc","trigger_condition","story_style","story_desc","story_chars","trigger_eva_think","other_triggers"],
    ),

    # 总体人物介绍
    "supple_all_role":PromptTemplate(
        template = """依次介绍游戏「{novel_name}」中的核心人物"{role_list}"，角色输出顺序按剧情权重降序，满足：
# 处理流程
    1. 名称预处理(名称标准化): 将人物的名称统一为最通用完整名称(可能包括不同译名,别名,昵称,英文名等)
    2. 重要性筛选: 第一个人物应该是故事中的主角「{leading_role}」,其他人物按重要性/剧情出现频率排序, 最终保留不超过7个角色
    3. 信息结构化: 根据故事中的信息，按身份(基础身份+隐藏身份(若存在)), 性格(显性特质+隐性矛盾), 经历(人物背景,出场状态,关键转折点,持续状态等方面)三大方面依次介绍

# 输出规范
    - 使用json结构, 使用"(角色名)":"(角色介绍)"的格式， dict格式(角色名作为键，具体介绍作为值), '{{"xx":"xx", "xx":"xx", ...}}'重构
    - 条目数≤7(主要角色数小于等于7), 按综合权重值排序
    - 不要提到"玩家","在xx游戏中", "在剧情中"等令人出戏的字眼, 不要提到本游戏具体名称
    - 用「{language}」回答

# 输出示例
```
{{
    "劳拉·克劳馥": "主角，21岁，刚踏出校园。充满自由精神和荒野情怀，坚韧不拔，作风实际，头脑冷静。在寻找邪马台古国的途中，劳拉所在的船只遭遇风暴失事，她漂流到一个未知的岛屿上，孤身一人在岛上醒来，开始她的求生之旅,在岛上，劳拉学会了生存，经历了从一名经验不足的年轻女孩到一个能够独当一面的冒险家的成长过程。在这个过程中，劳拉不仅要面对凶猛的野生动物和敌对的势力，还要揭开岛上的秘密，寻找失踪的同伴，并最终战胜邪教组织。",
    "珊曼莎·西村": "实习纪录片制片人，也是与劳拉从寄宿学校一直读到伦敦大学的同窗好友。为人大大咧咧，习惯过着奢侈阔绰的生活。拥有葡萄牙和日本的双重国籍，家境殷实。由于她是太阳女王卑弥呼的后裔，所以马蒂亚斯企图将卑弥呼的灵魂植入她体内以复活太阳女王。",
    "詹姆斯·惠特曼": "考古学家，也是当红考古电视节目「惠特曼的世界」主持人。久负盛名但是自私冷漠，对财富和名声有着异呼寻常的渴望，心胸狭窄，为了私利不择手段。坚忍号的这次探险计划其实就是为「惠特曼的世界」下一季做准备。劳拉原来十分崇拜惠特曼博士，但在荒岛遇险过程中惠特曼所表现出的自私冷漠以及背叛使劳拉看清了他的真面目。",
    "马蒂亚斯": "反派boss，太阳兄弟会的首领。神经怪异，但行事却条理清晰，冷酷无情，为了实现自己的目标不惜牺牲他人。1982年，马蒂亚斯驾机迫降在小岛附近海域。在逃亡过程中，他发现了岛上隐藏的邪马台古国遗址，并得知太阳女王拥有不可思议的魔力。31年的孤岛生活使他变得疯狂而且偏执，他采用威胁和恐吓手段笼统了一帮遇难者组成太阳兄弟会，还在女王遗址外修建了一座破烂小城。当他知道坚忍号失事船员里的珊曼莎是卑弥呼后裔后，就开始了丧心病狂的活人献祭阴谋。",
}}
```
    """,
        input_variables=["novel_name","role_list","leading_role","language"],
    )


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
                            


# 将多线剧情树结构反向组织 函数组
# 从form plot 变成指向下一个的dict
# 多线情况下，反向索引
def filter_plot_end(ans:str):
    new_ans = ""
    temp_list = ans.split('\n')
    for temp in temp_list:
        temp = temp.replace('（','(').replace('）',')').replace("：",":").strip()
        if temp:
            if temp.startswith('(') and temp.endswith(')'):
                new_ans += temp + '\n'
    return new_ans.strip()

def extra_plot_end(item_str:str, plot_sum_last:int):
    item_str = item_str.replace("(P","(p").replace("(E","(e").replace("(F","(f").strip()
    assert item_str != "" and item_str[0]=="("
    
    from_plot = item_str.split("from plot ")[1].split(")")[0] if "from plot " in item_str else 0        # 提取来源节点编号
    episode_key_abstract = item_str.split(")",1)[1].rsplit("(")[0].strip()          # 提取实际内容
    from_plot_list = []
    # from 多个plot的情况
    try:
        from_plot_list = [int(from_plot)] if from_plot else [0]
    except:
        num_temp = re.findall(r'\d+', from_plot)
        from_plot_list = [int(item) for item in num_temp]
        try:
            from_plot_list = [int(from_plot)] if from_plot else [0]
        except:
            num_temp = re.findall(r'\d+', from_plot)
            from_plot_list = [int(item) for item in num_temp]
        
    if item_str.startswith("(plot"):
        plot_num = int(item_str.split("plot ")[1].split(")")[0])        # 提取剧情编号
        # 判断一下，最大的plot是多少,这一步是判断一下gpt输出
        if plot_num > plot_sum_last:
            plot_sum_last = plot_num
        elif plot_num == plot_sum_last:
            logger.error(f"注意: 出现相同plot序号 {plot_num}")
        plot_dict = {
            "node_type":"plot",
            'plot_num': plot_num,
            'from_plot': from_plot_list,
            'episode_key_abstract': episode_key_abstract
        }
    elif item_str.startswith("(end"):
        plot_num = int(item_str.split("end ")[1].split(")")[0])      # 提取结局剧情编号
        plot_dict = {
            "node_type":"end",
            'end_num': plot_num,
            'from_plot': from_plot_list,
            'episode_key_abstract': episode_key_abstract
        }
    else:
        logger.error(f"no right type, the string is: \"{item_str}\"")
        plot_dict = {}
        plot_num = -1
    return plot_dict,plot_num,plot_sum_last

# 具体函数：将多线剧情树结构反向组织
# 从form plot 变成指向下一个的dict, 多线情况下，反向索引
def reverse_index_pro_stats(res_gen_multi:str):
    story_states = []
    pre_plot_dict = {}

    if not args.if_single_episode:
        new_gen_multi_plot = filter_plot_end(res_gen_multi)
        new_gen_multi_list = new_gen_multi_plot.split('\n')
        plot_sum_last = 0
        plot_temp_list = []
        end_temp_list = []

        for i in range(len(new_gen_multi_list)):
            item = new_gen_multi_list[i]
            item_dict, plot_num, plot_sum_last = extra_plot_end(item,plot_sum_last)
            if plot_num>-1:
                if item_dict["node_type"] == "plot":
                    plot_temp_list.append(item_dict)
                    pre_plot_dict[int(plot_num)] = item_dict["from_plot"]
                elif item_dict["node_type"] == "end":
                    end_temp_list.append(item_dict)
                    pre_plot_dict[plot_sum_last + int(plot_num)] = item_dict["from_plot"]

                else:
                    logger.error(f"Warning, no correct node_type :", item_dict["node_type"])
        # 这里要看一下 gpt的输出序号不一定连续
        if not len(plot_temp_list)==plot_sum_last:
            logger.error(f"剧情数量和序号未对齐, 剧情总数: {plot_temp_list}, 剧情序号最大值: {plot_sum_last}")
        multi_plot_dict = {}
        for i in range(len(plot_temp_list)):
            plot_temp = plot_temp_list[i]
            multi_plot_dict[plot_temp["plot_num"]] = {"episode_key_abstract":plot_temp["episode_key_abstract"],"next_episode":[],"info":"plot"}
            if not plot_temp["from_plot"][0] == 0:
                for from_plot_num in plot_temp["from_plot"]:
                    multi_plot_dict[from_plot_num]["next_episode"].append({plot_temp["plot_num"] : plot_temp["episode_key_abstract"]})
        for i in range(len(end_temp_list)):
            plot_temp = end_temp_list[i]
            multi_plot_dict[plot_sum_last + i + 1] = {"episode_key_abstract":plot_temp["episode_key_abstract"],"next_episode":[],"info":"end"}
            if not plot_temp["from_plot"][0] == 0:
                for from_plot_num in plot_temp["from_plot"]:
                    multi_plot_dict[from_plot_num]["next_episode"].append({plot_temp["end_num"]+plot_sum_last : plot_temp["episode_key_abstract"]})
        # 这里gpt输出plot不一定是plot
        for key in multi_plot_dict:
            if multi_plot_dict[key]["next_episode"] == [] and multi_plot_dict[key]["info"] == "plot":
                multi_plot_dict[key]["info"] = "end"
        # 通过bfs记录所有节点list
        states_temp = [[1]]         # 记录第一个节点必定是1      
        while states_temp:
            temp = states_temp.pop()
            if multi_plot_dict[temp[-1]]["info"] == "end":
                story_states.append(temp)
                #print("end:", temp)
                continue
            next_episodes = multi_plot_dict[temp[-1]]["next_episode"] # list,每个list是一个dict
            next_episodes_list = [int(next(iter(item))) for item in next_episodes]

            # 记录所有的新节点
            for next_episode in next_episodes_list:
                temp_new = copy.deepcopy(temp)
                temp_new.append(next_episode)
                states_temp.append(temp_new)
            #break
        gen_end_num = len(end_temp_list)
        gen_initial_episode = multi_plot_dict


    # 单线剧情重组织，反向索引
    # 这里的res_gen_multi是main_plot_list_re
    elif args.if_single_episode:
        single_plot_dict = {}
        
        pre_plot_dict[1] = 0
        for i in range(len(res_gen_multi)-1):
            single_plot_dict[i+1] = {"episode_key_abstract":res_gen_multi[i],"next_episode":[{i+2:res_gen_multi[i+1]}],"info": "plot"}
            pre_plot_dict[i+2] = [i+1]
            #print(main_plot_list["main"][i])
        
        # 补充结局节点
        single_plot_dict[len(res_gen_multi)] = {"episode_key_abstract":res_gen_multi[len(res_gen_multi)-1],"next_episode":[],"info": "end"}
        # 顺序记录所有节点
        story_states = [[i+1 for i in range(len(res_gen_multi))]]
        gen_end_num = 1  
        gen_initial_episode = single_plot_dict

    # print(json.dumps(gen_initial_episode,indent=2, ensure_ascii=False))  
    logger.info(f"剧情树: {story_states}")
    # logger.info(f"前驱节点集合: {pre_plot_dict}")

    return gen_initial_episode,story_states,gen_end_num,pre_plot_dict

  

def Summary_Background(story_name:str, llm_model_name:str="gpt-4o", story_data:str=""):
    # 背景 等内容dict填充
    res_dic = {}
    time_background_begin = time.time()
    res0_dict = {}
    if not story_data:
        prompt_SumBkgd = prompt_gen_story_template['summary_background'].format(novel_name=story_name, language=llm_ans_language)
    else:
        prompt_SumBkgd = prompt_gen_story_template['summary_background_with_data'].format(novel_name=story_name, language=llm_ans_language, story_data=story_data)
    # 中文游戏考虑deepseek
    res_think, res_SumBkgd = llm_chat_analysis(llm_model_name, prompt_SumBkgd, "dict")

    story_dict = copy.deepcopy(template_dict_story)
    story_dict["Settings"]["story_name"] = story_name
    style_items = re.split(r'[，,]', res_SumBkgd["story_style"])
    story_dict["Settings"]['story_style'] = [item.strip() for item in style_items if item.strip()]
    story_dict["Settings"]['story_desc'] = res_SumBkgd["background"]
    story_dict["Settings"]["story_goal"] = res_SumBkgd["target"]
    story_dict["Settings"]["leading_role"] = res_SumBkgd["leading_role"]
    story_dict["Settings"]["gen_parameter"] = vars(args)

    story_dict["Settings"]["story_info"]["epi_struct"] = "single" if args.if_single_episode else "multi"
    story_dict["Settings"]["story_info"]["language"] = llm_ans_language
    story_dict["Settings"]["story_info"]["sources_type"] = args.story_source_type
    story_dict["Settings"]["story_info"]["sources"] = args.story_source

    time_background = time.time() - time_background_begin
    logger.info(f"背景信息抽取时间：{time_background:.2f}秒")
    # print(json.dumps(story_dict,indent=2, ensure_ascii=False))
    return story_dict

def Summary_Mainplot(story_name:str,story_leading_role:str,model_name:str="gpt-4o"):
    time_mainplot_begin = time.time()

    prompt_SumMp = prompt_gen_story_template["summary_main_plot"].format(novel_name=story_name, seg_main_plot_num=args.story_branch_num, leading_role=story_leading_role, language=llm_ans_language)
    res_think, res_SumMp = llm_chat_analysis(model_name, prompt_SumMp, "list")

    # logger.info("主线剧情: ",json.dumps(res_SumMp,indent=2, ensure_ascii=False))
    time_mainplot = time.time() - time_mainplot_begin
    logger.info(f"主线剧情抽取时间：{time_mainplot:.2f}秒")
    return res_SumMp

# ### 多线剧情重组织
# 1. 将单线剧情重新组合，重写多线剧情
# 2. 将多线剧情线通过dict进行连接
def Rewrite_Multiplot(story_name:str,main_plot_str:str,story_rel_info:dict,llm_model_name:str="deepseek-r1"):
    time_gen_multi_begin = time.time()

    prompt_RwMp = prompt_gen_story_template["rewrite_multi_plot"].format(novel_name=story_name, main_plot_str=main_plot_str, game_background=story_rel_info["description"], game_target=story_rel_info["goal"], game_style=story_rel_info["style"], language=llm_ans_language, leading_role=story_rel_info["leading_role"])
    res_think, res_RwMp = llm_chat_analysis(llm_model_name, prompt_RwMp, "str")

    # print(res_think)
    logger.info(f"重写后多线剧情:\n{res_RwMp}")
    time_gen_multi = time.time() - time_gen_multi_begin
    logger.info(f"多线剧情生成时间：{time_gen_multi:.2f}秒")
    return res_RwMp

# 单线/多线
# 生成具体剧情
def Expand_Detailplot(story_name:str,gen_initial_episode:dict,states:list,llm_model_name:str="gpt-4o"):

    GenInit_Epi_dict = {k: gen_initial_episode[k] for k in sorted(gen_initial_episode)}
    len_main_Epi = len(GenInit_Epi_dict)
    logger.info(f"剧情长度: {len_main_Epi}")
    # print(json.dumps(GenInit_Epi_dict,indent=2,ensure_ascii=False))

    for key in tqdm(GenInit_Epi_dict.keys()):
        GenEpi_dict_item = GenInit_Epi_dict[key]

        if key > len_main_Epi+1:
            logger.error(f"The key {key} > max_len_plot {len_main_Epi+1}")
        GenEpi_PosFrac = f"{key}/{len_main_Epi+1}"  # 当前情节进程数

        # 1. 情节生成 详细情节 - detail_events
        # 前置剧情
        states_temp = []
        pre_plot = ""
        for states_i in states:
            if int(key) in states_i:
                states_temp = states_i
                break
        if states_temp == []:
            logger.error(f"Warning! states_temp is []")
        else:
            for state in states_temp:
                if not state == int(key):
                    pre_plot += GenInit_Epi_dict[state]["episode_desc"] + "\n"
                else:
                    break
        prompt_ExpDp = prompt_gen_story_template["expand_plot_detail"].format(novel_name=story_name, plot_str=GenEpi_dict_item["episode_key_abstract"],plot_num_frac=GenEpi_PosFrac,language=llm_ans_language,pre_plot=pre_plot)
        res_think, res_gen_plot_detail = llm_chat_analysis(llm_model_name, prompt_ExpDp, "str")

        res_gen_plot_detail = res_gen_plot_detail.replace("\n"," ").strip()
        logger.info(f"key: {key}, res_gen_plot_detail: {res_gen_plot_detail}")
        GenInit_Epi_dict[key]["episode_desc"] = res_gen_plot_detail
        # break
    return GenInit_Epi_dict
        
        
        
def Eva_Opt_Detailplot(states:list, GenInit_Epi_dict:dict, story_rel_info:dict,plotinfo_opt_threshold:int=85, eva_llm_model_name = "gpt-4o",opt_llm_model_name="deepseek-r1"):
    new_states_list = {}
    other_ans = ""
    other_ans_num = []
    # plotinfo_opt_threshold = 0
    # 评估连贯性
    for j in tqdm(range(len(states))):
        state = {}
        if type(states[j]) == dict:
            state["seq"] = states[j]["seq"]
        elif type(states[j]) == list:
            state["seq"] = states[j]
        else:
            logger.error(f"Warning, wrong type in state {j}: {type(states[j])}")

        episode_seq = ""
        episode_seq_refine = ""

        # 剧情序列
        logger.info(state["seq"])
        
        # print(json.dumps(GenInit_Epi_dict,indent=2,ensure_ascii=False))
        episode_seq += "\n".join(f"(episode {k}) {GenInit_Epi_dict[k]['episode_desc']}\n" for k in state["seq"])
        other_ans_num.append(state["seq"][-1]) if not state["seq"][-1] in other_ans_num else None

        # 判断总体章节信息
        prompt_EvaDp_init = prompt_gen_story_template["evaluate_plot_detail"].format(leading_role=story_rel_info["leading_role"],episode_seq=episode_seq,story_style=story_rel_info["style"],story_desc=story_rel_info["description"],story_goal=story_rel_info["goal"],language=llm_ans_language)
        res_Eva_think_1, res_Eva_Score_Epi1 = llm_chat_analysis(eva_llm_model_name, prompt_EvaDp_init, "dict")

        res_Eva_Score_Epi1["score"] = int(res_Eva_Score_Epi1["score"])
        state["Eva_outline_ori_think"] = res_Eva_Score_Epi1["reason"]
        state["Eva_outline_ori_score"] = res_Eva_Score_Epi1["score"]

        logger.info(f'原始章节框架「{j}」评分: {res_Eva_Score_Epi1["score"]}')

        refine_try_count = 0
        opt_reason = res_Eva_Score_Epi1["reason"]

        while res_Eva_Score_Epi1["score"] < plotinfo_opt_threshold and refine_try_count<3:
            refine_try_count += 1
            logger.info(f"第{refine_try_count}次优化章节框架")
            try_count = 0
            while try_count <3:
                try_count+=1
                logger.info(f"分数低于<{plotinfo_opt_threshold}>,重新生成当前章节, 第{refine_try_count}-{try_count}次")
                try:
                    prompt_OpiDp = prompt_gen_story_template["optimize_plot_detail"].format(leading_role=story_rel_info["leading_role"],language=llm_ans_language,outline_refine_reason=opt_reason,episode_seq=episode_seq,story_style=story_rel_info["style"],story_desc=story_rel_info["description"],story_goal=story_rel_info["goal"],other_ans=other_ans)
                    res_think, res_Eva_Opt_Epi = llm_chat_analysis(opt_llm_model_name, prompt_OpiDp, "dict")
                    
                    if not state["seq"][-1] in other_ans_num:
                        other_ans_num.append(state["seq"][-1])
                        other_ans += "Other end: " + GenInit_Epi_dict[state["seq"][-1]]["episode_desc"]

                    for key in res_Eva_Opt_Epi:
                        pass
                        key_num = key
                        
                        if type(key) == str:
                            key_num  = re.findall(r'\d+', key_num)
                            key_num = [int(num) for num in key_num][0]

                        # 有点问题
                        GenInit_Epi_dict[int(key_num)]["episode_desc_old"] = GenInit_Epi_dict[int(key_num)]["episode_desc"]
                        GenInit_Epi_dict[int(key_num)]["episode_desc"] = res_Eva_Opt_Epi[key]["改进后章节"]
                        GenInit_Epi_dict[int(key_num)]["episode_old_question"] = res_Eva_Opt_Epi[key]["存在问题"] + " " + res_Eva_Opt_Epi[key]["改进建议"]
                        
                    break
                except Exception as e:
                    logger.error(f"detail plot refine failed: {str(e)}")
                    if try_count == 3:
                        logger.error("Maximum retry attempts reached for plot refinement")

            logger.info("refine结束, 重新评分ing...")
            episode_seq_refine += "\n".join(f"(episode {k}) {GenInit_Epi_dict[k]['episode_desc']}\n" for k in state["seq"])
            prompt_EvaDp_opt = prompt_gen_story_template["evaluate_plot_detail"].format(leading_role=story_rel_info["leading_role"],episode_seq=episode_seq,story_style=story_rel_info["style"],story_desc=story_rel_info["description"],story_goal=story_rel_info["goal"],language=llm_ans_language)
            res_Eva_think_2, res_Eva_Score_Epi2 = llm_chat_analysis(eva_llm_model_name, prompt_EvaDp_opt, "dict")

            # 有点问题，回退
            res_Eva_Score_Epi2["score"] = int(res_Eva_Score_Epi2["score"])
            logger.info(f'refine后章节框架评分: {res_Eva_Score_Epi2["score"]}')
            if res_Eva_Score_Epi2["score"]>=res_Eva_Score_Epi1["score"]:
                state["Eva_outline_refine_think"] = res_Eva_Score_Epi2["reason"]
                state["Eva_outline_refine_score"] = res_Eva_Score_Epi2["score"]
                break
            opt_reason = res_Eva_Score_Epi2["reason"]

        new_states_list[j] = state
        # break
    return new_states_list,GenInit_Epi_dict



def Supple_Info_Detailplot(story_name:str,GenInit_Epi_dict:dict, pre_plot_dict:dict, story_rel_info:dict,llm_model_name="gpt-4o"):
    Episodes_all = []
    len_main_Epi = len(GenInit_Epi_dict) 

    for key in tqdm(GenInit_Epi_dict.keys()):
        count_try_while = 0
        while count_try_while<3:
            count_try_while += 1
            if count_try_while > 1:
                logger.info(f"第「{count_try_while}」try生成失败, 重试ing...")
            try:     
            # if True:  
                plot_dict_item = GenInit_Epi_dict[key]
                plot_dict = copy.deepcopy(template_dict_episode)
                plot_dict["episode_id"] = key
                plot_dict["pre_episode_id"] = pre_plot_dict[key]
                plot_dict["episode_desc"] = plot_dict_item["episode_desc"]
                plot_dict["episdoe_desc_brief"] = plot_dict_item["episode_key_abstract"]

                if key > len_main_Epi+1:
                    logger.error(f"The key {key} > max_len_plot {len_main_Epi+1}")
                GenEpi_PosFrac = f"{key}/{len_main_Epi+1}"
                
                # 2. 其他故事信息  - target,scene,role
                retry_count_leading_role = 0
                while retry_count_leading_role < 3:

                    retry_count_leading_role += 1
                    if retry_count_leading_role > 1:
                        logger.info(f"第「{retry_count_leading_role}」次补充其他信息")


                    res_gen_plot_info = {"target":"","scene":"","role":{}}
                    
                    logger.info(f"对应章节「{key}」其他信息补充ing...")
                    prompt_SupPlot = prompt_gen_story_template["supple_plot_info"].format(novel_name=story_name, plot_str=plot_dict_item["episode_key_abstract"],plot_num_frac=GenEpi_PosFrac,language=llm_ans_language,leading_role=story_rel_info["leading_role"])
                    res_think, res_gen_plot_info = llm_chat_analysis(llm_model_name, prompt_SupPlot, "dict")
                    # print(json.dumps(res_gen_plot_info,indent=2,ensure_ascii=False))

                    plot_dict["episode_goal"] = res_gen_plot_info["target"]
                    plot_dict["episode_scene"] = res_gen_plot_info["scene"]

                    episode_role = res_gen_plot_info["role"]
                    if type(episode_role) == dict:
                        plot_dict["episode_chars"] = episode_role
                    elif type(episode_role) == list:
                        epi_role_dict = {}
                        for role_item in episode_role:
                            if type(role_item) == str:
                                role_item = role_item.replace("：",":")
                                epi_role_dict[role_item.split(":")[0].strip()] = role_item.split(":")[1].strip()
                            else:
                                logger.error(f"warning, episode_role type is list[{type(role_item)}]")
                                logger.error("llm能力应该不会有这种问题")
                        plot_dict["episode_chars"] = epi_role_dict
                    else:
                        logger.error(f"warning, episode_role type is {type(episode_role)}")
                        logger.error("llm能力应该不会有这种问题")
                        
                    if not story_rel_info["leading_role"] in plot_dict["episode_chars"]:
                        logger.warning(f"warning, {story_rel_info['leading_role']} not in episode_chars, retry...")
                        continue
                    break

                if retry_count_leading_role >= 3:
                    logger.error(f"warning, {story_rel_info['leading_role']} not in episode_chars, retry 3 times, exit...")
                
                # 3. 跳转情节生成 - triggers
                # 需要当前情节的目标
                logger.info(f"triggers补充ing...")
                plot_target = plot_dict["episode_goal"]
                plot_scene = plot_dict["episode_scene"]
                # 非结局
                if not plot_dict_item["next_episode"] == []:
                    # break
                    
                    plot_dict["triggers"] = []
                    # 对于每个trigger生成? 但是不确定这样会不会与不同的选项冲突
                    # 将生成过的triggers记录一下
                    else_triggers = ""
                    for item in plot_dict_item["next_episode"]:
                        next_key = list(item.keys())[0]
                        next_value = item[next_key]
                        trigger_plot = {"condition":"","next_episode": next_key}
                        
                        prompt_SupTri = prompt_gen_story_template["supple_plot_trigger"].format(novel_name=story_name, plot_str=plot_dict_item["episode_key_abstract"], next_episode_str=next_value, plot_target=plot_target, plot_num_frac=GenEpi_PosFrac, else_triggers=else_triggers,  story_style=story_rel_info['style'],story_desc=plot_scene,story_chars=episode_role,language=llm_ans_language)

                        res_think, res_gen_plot_triggers = llm_chat_analysis(llm_model_name, prompt_SupTri, "str")
                        # print(json.dumps(res_gen_plot_triggers,indent=2,ensure_ascii=False))

                        if else_triggers == "":
                            else_triggers = """
            需要注意的是, 在这段剧情中, 除了你现在需要编写的trigger以外, 还存在其他triggers, 它们可以导致当前情节跳转到不同的分支。
            你需要确保你编写的触发条件与其他触发条件有显著差别，以确保剧情能够清晰地沿着不同的路径发展
            具体来说, <IF>之后的条件和<THEN>之后触发的剧情都应与其他triggers的<IF>和<THEN>的内容产生较大的不同
            以下是这个情节中的其他triggers: \n
                            """
                        else_triggers += res_gen_plot_triggers + "\n"
                        trigger_plot["condition"] = res_gen_plot_triggers
                        plot_dict["triggers"].append(trigger_plot)
                # 结局
                else: 
                    prompt_SupEnd = prompt_gen_story_template["supple_end_trigger"].format(novel_name= story_name,leading_role=story_rel_info["leading_role"],episode_desc=plot_dict["episode_desc"],story_style=story_rel_info["style"],story_desc=story_rel_info["description"],story_chars=episode_role,language = llm_ans_language)
                    res_Gen_think_4, res_Gen_End= llm_chat_analysis(llm_model_name, prompt_SupEnd, "dict")
                    print(json.dumps(res_Gen_End,indent=2,ensure_ascii=False))
                    plot_dict["episode_desc"] = res_Gen_End["最终章流程"]
                    plot_dict["triggers"].append({})
                    plot_dict["triggers"][0]["condition"] = "<IF>" +res_Gen_End["结局触发点"] + " <THEN>" + res_Gen_End["结局演出"] 
                    plot_dict["triggers"][0]["next_episode"] = -1
                
                break
            except  Exception as e:
                logger.error(f"发生错误: {e}")
                count_try_while += 1
        Episodes_all.append(plot_dict)        
        #break
    return Episodes_all

# 非结局plot triggers评估与优化
def Eva_Opt_Trigger(Episodes:list,story_rel_info:dict,trigger_opt_threshold:int=80,eva_llm_model_name="gpt-4o",opt_llm_model_name="deepseek-r1"):
    episodes_new = copy.deepcopy(Episodes)
    # 用于判断优化阈值
    # trigger_opt_threshold = 80
    for j in tqdm(range(len(Episodes))):
        episode = Episodes[j]
        # 判断triggers
        triggers = episode["triggers"]
        if (not triggers == []) and (not triggers == [{}]):
            # continue
            other_triggers = ""
            for k in range(len(triggers)):
                trigger = triggers[k]
                # 这里看一下要不要-1
                next_episode = Episodes[trigger["next_episode"]-1]
                prompt_EvaTri_init = prompt_gen_story_template["evaluate_trigger"].format(leading_role=story_rel_info["leading_role"],episode_desc=episode["episode_desc"],next_episode_desc=next_episode["episode_desc"],trigger_condition=trigger["condition"],story_style=story_rel_info["style"],story_desc=story_rel_info["description"],story_chars=episode["episode_chars"])

                res_Eva_think_1, res_Eva_Score_Tri1 = llm_chat_analysis(eva_llm_model_name, prompt_EvaTri_init, "dict")
                res_Eva_Score_Tri1["score"] = int(res_Eva_Score_Tri1["score"])
                episodes_new[j]["triggers"][k]["eva_reason_ori"] = res_Eva_Score_Tri1["reason"]
                episodes_new[j]["triggers"][k]["eva_score_ori"] = res_Eva_Score_Tri1["score"]
                logger.info(f'章节{j}-trigger{k}-原始评分: {res_Eva_Score_Tri1["score"]}')

                refine_try_count = 0
                
                while res_Eva_Score_Tri1["score"]<trigger_opt_threshold and refine_try_count<3:
                    refine_try_count +=1
                    logger.info(f"第{refine_try_count}次优化triggers")

                    gpt_try_count=0
                    while gpt_try_count<3:
                        gpt_try_count+=1
                        try:
                            prompt_OptTri = prompt_gen_story_template["optimize_trigger"].format(leading_role=story_rel_info["leading_role"],language=llm_ans_language,episode_desc=episode["episode_desc"],next_episode_desc=next_episode["episode_desc"],trigger_condition=trigger["condition"],story_style=story_rel_info["style"],story_desc=story_rel_info["description"],story_chars=episode["episode_chars"],trigger_eva_think=res_Eva_Score_Tri1["reason"],other_triggers=other_triggers)
                            res_Eva_think_2, res_Eva_Opt_Tri1 = llm_chat_analysis(opt_llm_model_name, prompt_OptTri, "dict")
                            # break
                            new_trigger_con = "<IF>" + res_Eva_Opt_Tri1["IF"] + " <THEN>" + res_Eva_Opt_Tri1["THEN"]
                            prompt_EvaTri_opt = prompt_gen_story_template["evaluate_trigger"].format(leading_role=story_rel_info["leading_role"],episode_desc=episode["episode_desc"],next_episode_desc=next_episode["episode_desc"],trigger_condition=new_trigger_con,story_style=story_rel_info["style"],story_desc=story_rel_info["description"],story_chars=episode["episode_chars"])
                            res_Eva_think_3, res_Eva_Score_Tri2 = llm_chat_analysis(eva_llm_model_name, prompt_EvaTri_opt, "dict")
                            
                            res_Eva_Score_Tri2["score"] = int(res_Eva_Score_Tri2["score"])
                            logger.info(f'优化后trigger评分{res_Eva_Score_Tri2["score"]}')
                            break 
                        except Exception as e:
                            logger.warning(f"refine fail, retrying...{e}")

                    # 分数合适保存
                    if res_Eva_Score_Tri2["score"]>=res_Eva_Score_Tri1["score"]:
                        episodes_new[j]["triggers"][k]["eva_reason_opt"] = res_Eva_Score_Tri2["reason"]
                        episodes_new[j]["triggers"][k]["eva_score_opt"] = res_Eva_Score_Tri2["score"]

                        episodes_new[j]["triggers"][k]["eva_opt_ori"] = res_Eva_Opt_Tri1["OPTIMIZE"]
                        episodes_new[j]["triggers"][k]["old_con"] = trigger["condition"]
                        episodes_new[j]["triggers"][k]["condition"] = new_trigger_con
                        break

                    # break

                if not other_triggers:
                    other_triggers = "当前章节其他的的triggers如下:\n"
                other_triggers+=episodes_new[j]["triggers"][k]["condition"]+"\n"
                # break
        # break
    return episodes_new

# 最后整体补齐story所有人物信息, 单线多线都有
def Supple_Info_Allrole(story_name:str, Episodes:list, story_rel_info:dict,llm_model_name="deepssek-r1"):
    # 统计一下总人物
    role_str = ""  
    logger.info(f"if_role_known: {args.if_role_known}")
    # 第一种是根据人物列表直接总结，认为llm知道所有人物信息
    if not args.if_role_known:
        role_str_set = set()  
        for plot in Episodes:  
            for role in plot["episode_chars"]:  
                if role not in role_str_set: 
                    role_str += f"{role}, " 
                    role_str_set.add(role)  #
        role_str.strip().strip(",")
        logger.info(f"role_list: {role_str}")

    # 第二种更适用于小说llm, llm不了解人物时根据所有人物状态总结
    else:
        for i in range(len(Episodes)):  
            plot = Episodes[i]
            role_str += "故事的第{i}幕出场的人物以及状态如下:\n"
            for role in plot["episode_chars"]:  
                role_str += role + ": " + plot["episode_chars"][role] + "\n"
        role_str.strip().strip("\n")
        logger.info(f"role_list: {role_str}")
    prompt_SupAr = prompt_gen_story_template["supple_all_role"].format(novel_name=story_name, role_list=role_str,leading_role=story_rel_info["leading_role"],language=llm_ans_language)
    res_think, res_story_all_role = llm_chat_analysis(llm_model_name, prompt_SupAr, "dict")

    return res_story_all_role





# 这个是整体流程
def gen_abstract_plot_flow(story_name:str,other_info:dict,story_data_list:list=[]):
    time_all_link = time.time()
    logger.info(f"故事名称: {story_name}")

    # 背景生成ds耗时 60s - 100s; gpt耗时 5s - 10s
    if not story_data_list:
        logger.info("故事基础信息生成ing...")   
        gen_story_dict = Summary_Background(story_name,"gpt-4o")
        story_leading_role = gen_story_dict["Settings"]["leading_role"]
        story_desc = gen_story_dict["Settings"]["story_desc"]
        story_goal = gen_story_dict["Settings"]["story_goal"]
        story_style = gen_story_dict["Settings"]["story_style"]
        story_rel_info = {"leading_role":gen_story_dict["Settings"]["leading_role"],"description":story_desc,"goal":story_goal,"style":story_style}

        # 主线剧情生成ds耗时 60s - 100s; gpt耗时 6s - 10s
        logger.info("主线剧情大纲生成ing...")
        gen_story_dict["Settings"]["storylines"] = Summary_Mainplot(story_name,story_leading_role,"gpt-4o")
    else:
        story_data_list = [item.replace("\n\n",": ").replace("\n"," ") for item in story_data_list]
        story_data = "\n".join(story_data_list)
        
        logger.info("从已有数据中提取故事基础信息...")
        gen_story_dict = Summary_Background(story_name, "gpt-4o", story_data)
        story_leading_role = gen_story_dict["Settings"]["leading_role"]
        story_desc = gen_story_dict["Settings"]["story_desc"]
        story_goal = gen_story_dict["Settings"]["story_goal"]
        story_style = gen_story_dict["Settings"]["story_style"]
        story_rel_info = {"leading_role":gen_story_dict["Settings"]["leading_role"],"description":story_desc,"goal":story_goal,"style":story_style}
        
        # 使用已有故事数据作为主线剧情
        logger.info("使用已有故事数据作为主线剧情...")
        gen_story_dict["Settings"]["storylines"] = story_data_list

    # print(gen_story_dict["Settsdings"])
    # 多线剧情生成ds耗时 120s - 180s; gpt耗时 10s - 20s
    gen_count = 0
    while gen_count<3:
        try:
            gen_count += 1
            if not args.if_single_episode:
                logger.info("分支剧情大纲生成ing...")
                main_plot_str = "\n".join(item for item in gen_story_dict["Settings"]["storylines"])
                # Reorganize_struct_plot_str = Rewrite_Multiplot(story_name,main_plot_str,story_rel_info,"gpt-4o")
                Reorganize_struct_plot_str = Rewrite_Multiplot(story_name,main_plot_str,story_rel_info,"deepseek-r1")
            else:
                Reorganize_struct_plot_str = gen_story_dict["Settings"]["storylines"]

            # 反向索引
            gen_initial_episode, states, gen_story_dict["Settings"]["story_info"]['gen_end_num'], pre_plot_dict = reverse_index_pro_stats(Reorganize_struct_plot_str)
            if len(states) == gen_story_dict["Settings"]["story_info"]['gen_end_num']:
                break
            logger.error(f"Warning, plot_tree({len(states)}) and end_num are not matched")
        except Exception as e:
            logger.error(f"剧情生成报错: {e}")
            logger.info("重新构建剧情ing...")

    gen_story_dict["Settings"]["story_info"]['pre_epi_node'] = pre_plot_dict 


    # 逐个剧情，生成对应具体节点（在多线剧情生成后，单线剧情跳过上一部分）
    # gpt len(剧情长度)*5s
    logger.info("具体剧情内容补充ing...")
    gen_expand_episode = Expand_Detailplot(story_name,gen_initial_episode,states,"gpt-4o")

    # refine
    # 一轮deepseek耗时 200-240s
    logger.info("剧情大纲eva/refine...")
    # gen_story_dict["Settings"]["states"],gen_optimize_episode = Eva_Opt_Detailplot(states, gen_expand_episode, story_rel_info,0,"gpt-4o","gpt-4o")
    gen_story_dict["Settings"]["states"],gen_optimize_episode = Eva_Opt_Detailplot(states, gen_expand_episode, story_rel_info,85,"gpt-4o","deepseek-r1")

    # plot其他内容生成: 目标,背景,角色以及跳转条件
    logger.info("具体剧情其他信息生成ing...")
    gen_story_dict["Episodes"] = Supple_Info_Detailplot(story_name,gen_optimize_episode, pre_plot_dict, story_rel_info,"gpt-4o")

    # 存储中间结果
    temp_save_file_path = '../data/temp_save.jsonl'
    if True:
        logger.info("中间结果已保存")
        with jsonlines.open(temp_save_file_path, mode='a') as writer:
            writer.write(gen_story_dict)
    else:
        with jsonlines.open(temp_save_file_path, mode='r') as reader:
            story_dicts = [obj for obj in reader]
        logger.info(len(story_dicts))

        gen_story_dict = story_dicts[0]
        Settings = gen_story_dict["Settings"]
        story_leading_role = Settings["leading_role"]
        story_desc = Settings["story_desc"]
        story_goal = Settings["story_goal"]
        story_style = Settings["story_style"]
        story_rel_info = {"leading_role":story_leading_role,"description":story_desc,"goal":story_goal,"style":story_style}
        states = Settings["states"]

    Episodes = gen_story_dict["Episodes"]
    # 判断总体章节信息, 不太需要, 效果本身还可以

    # 判断triggers
    # deepseek 评价单情节评估评价时间30s
    logger.info("triggers eva/refine ...")
    # gen_story_dict["Episodes"] = Eva_Opt_Trigger(Episodes,story_rel_info,0,"gpt-4o","gpt-4o")
    gen_story_dict["Episodes"] = Eva_Opt_Trigger(Episodes,story_rel_info,80,"gpt-4o","deepseek-r1")
    # 最后整体补齐story所有人物信息, 单线多线都有
    # deepseek 用时 60-100s
    logger.info("总体人物信息补充ing...")
    gen_story_dict["Settings"]["story_chars"] = Supple_Info_Allrole(story_name, gen_story_dict["Episodes"], story_rel_info,"deepseek-r1")
    # gen_story_dict["Settings"]["story_chars"] = Supple_Info_Allrole(story_name, gen_story_dict["Episodes"], story_rel_info,"gpt-4o")
    logger.info(json.dumps(gen_story_dict["Settings"]["story_chars"],indent=2, ensure_ascii=False)) 

    for i in range(len(gen_story_dict["Episodes"])):
        gen_story_dict["Episodes"][i]["episode_id"] = str(gen_story_dict["Episodes"][i]["episode_id"])

    time_all_link = time.time() - time_all_link
    min_time = int(time_all_link//60)
    sec_timr = (time_all_link%60)
    logger.info(f"「{story_name}」总耗时: {min_time}min {sec_timr:.2f}s")

    return gen_story_dict

def main():
    if not os.path.exists(args.data_output_folder):
        os.makedirs(args.data_output_folder)
        logger.info(f"创建数据文件夹：{args.data_output_folder}")
    date_md = str(datetime.now().strftime("%m-%d")).replace("-","_")
    logger.info(f"data_now: {date_md}")
    story_out_filepath = f'{args.data_output_folder}/StoryGen_{args.story_source}_{args.story_source_type}_{date_md}.jsonl'
    other_info = {}

    if args.story_source_type == "name":
        if args.story_source == "game":
            name_list = game_name_list
        elif args.story_source == "movie":
            name_list = movie_name_list
        else:
            raise Exception("wrong type name!")  # 抛出异常并停止程序

        for i in range(0,len(name_list)):
            # if i %2 == 0:
            #     continue
            story_name = name_list[i]
            story_name = f"{args.story_source}:「{story_name}」"
            
            count_try = 0
            while count_try<3:
                count_try+=1
                try:
                    story_dict = gen_abstract_plot_flow(story_name,other_info)
                    with jsonlines.open(story_out_filepath, mode='a') as writer:
                        writer.write(story_dict)
                    logger.info(f"{story_name}抽取成功，save file path: {story_out_filepath}")
                    break
                except:
                    logger.warning(f"{story_name}抽取失败，重新抽取")
            # break
    elif args.story_source_type == "file":
        input_all_folder = "storyline_generator/source_stories/extra_story_output/"
        if args.story_source == "novel":
            storyline_input_folder = input_all_folder + "extra_storyline/"
        elif args.story_source == "keyword_create":
            storyline_input_folder = input_all_folder + "create_storyline/"
        else:
            raise Exception("wrong type name!")  # 抛出异常并停止程序
        
        i = 0
        begin = 61
        end = 6000
        for filename in os.listdir(storyline_input_folder):
            if not filename.endswith(".jsonl"):
                continue

            i+=1
            if i<begin:
                continue
            if i>end:
                break

            print(i,filename)

            file_path = os.path.join(storyline_input_folder, filename)  # 构建完整文件路径
            print(f"正在读取文件: {os.path.basename(file_path)}")  # 打印当前文件名
            story_data = []
            story_name = os.path.basename(file_path).split("_")[0]
            with jsonlines.open(file_path, mode='r') as reader:
                for line in reader:
                    story_data.append(line)

            count_try = 0
            while count_try<3:
                count_try+=1
                try:
                    story_dict = gen_abstract_plot_flow(story_name,other_info,story_data_list=story_data)
                    with jsonlines.open(story_out_filepath, mode='a') as writer:
                        writer.write(story_dict)
                    logger.info(f"{story_name}抽取成功，save file path: {story_out_filepath}")
                    break
                except:
                    # story_dict = gen_abstract_plot_flow(story_name,other_info,story_data=story_data)
                    logger.warning(f"{story_name}抽取失败，重新抽取")





if __name__ == "__main__":
    main()