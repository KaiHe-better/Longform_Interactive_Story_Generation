import random
from templates_zh import *
from templates_en import *

ID2LANG = {
    "en": "English",
    "zh": "中文"
}

LANG2ID = {
    "English": "en",
    "中文": "zh"
}

TEMPLATE_DICT = {
    "zh": {
        "gen": GENERATOR_TEMPLATE_ZH,
        "refine": REFINE_TEMPLATE_ZH,
        "regen": REGENERATOR_TEMPLATE_ZH,
        "user_normal": USER_TEMPLATE_NORMAL_ZH,
        "user_abnormal": USER_TEMPLATE_ABNORMAL_ZH,
        "summary": SUMMARY_TEMPLATE_ZH,
        "trigger": TRIGGER_TEMPLATE_ZH,
    },
    "en": {
        "gen": GENERATOR_TEMPLATE_EN,
        "refine": REFINE_TEMPLATE_EN,
        "regen": REGENERATOR_TEMPLATE_EN,
        "user_normal": USER_TEMPLATE_NORMAL_EN,
        "user_abnormal": USER_TEMPLATE_ABNORMAL_EN,
        "summary": SUMMARY_TEMPLATE_EN,
        "trigger": TRIGGER_TEMPLATE_EN,
    }
}

QUERY_TEMPLATE_DICT = {
    "zh": {
        "gen": GEN_QUERY_ZH,
        "refine": REFINE_QUERY_ZH,
        "regen": REGEN_QUERY_ZH,
        "user": USER_QUERY_ZH,
        "summary": SUMMARY_QUERY_ZH,
        "trigger": TRIGGER_QUERY_ZH,
    },
    "en": {
        "gen": GEN_QUERY_EN,
        "refine": REFINE_QUERY_EN,
        "regen": REGEN_QUERY_EN,
        "user": USER_QUERY_EN,
        "summary": SUMMARY_QUERY_EN,
        "trigger": TRIGGER_QUERY_EN,
    }
}

USER_ACTIONS_DICT = {
    "zh": USER_ACTION_HACKING_ZH + USER_ACTION_INSTRUCT_ZH,
    "en": USER_ACTION_HACKING_EN + USER_ACTION_INSTRUCT_EN,
}


def get_template(language, template_type):
    return TEMPLATE_DICT[language][template_type]


def get_query_template(language, template_type):
    return QUERY_TEMPLATE_DICT[language][template_type]


def get_user_actions(language):
    return random.choice(USER_ACTIONS_DICT[language])
