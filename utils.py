import json
import re
import string
from colorama import Fore, Style

PRINT_COLOR_LIST = [Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN, Fore.WHITE]


def parse_json_from_string(text: string):
    try:
        if not text:
            return {}
        # 找到第一个{和最后一个}的位置
        start_index = str(text).find("{")
        end_index = str(text).rfind("}")
        # 截取json字符串
        text = text[start_index:end_index + 1]
        json_str = json.loads(text)
        return json_str
    except Exception as e:
        print(f'{e}, {text}')
        return {}


def remove_first_sentence(text: str, language: str = "en"):
    # 去掉text的第一句话，剩下的话保留
    # 如果text为空，则返回空字符串
    if not text:
        return ""

    # 中文以句号、问号、感叹号作为分隔符
    if language == "zh":
        return re.sub(r'^[^。？！]+[。？！]', '', text).strip()
    # 英文以句号、问号、感叹号作为分隔符
    else:
        return re.sub(r'^[^.!?]+[.!?]', '', text).strip()


if __name__ == "__main__":
    print(remove_first_sentence("Hello, world! How are you?", "en"))
    print(remove_first_sentence("你好，世界。你好吗？", "zh"))
