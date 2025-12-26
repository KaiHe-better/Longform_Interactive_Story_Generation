import torch
from typing import Any, List, Dict
import re
import math
from json_repair import repair_json 


def count_tokens(text: str) -> int:
    if not isinstance(text, str) or not text:
        return 0
    
    # 1. 匹配中文字符 (每个汉字计 1)
    chinese_chars = re.findall(r'[\u4e00-\u9fa5]', text)
    
    # 2. 匹配英文单词 (连续的字母/数字序列计 1)
    # 这里的 [a-zA-Z0-9]+ 会把 "apple" 识别为一个整体
    english_words = re.findall(r'[a-zA-Z0-9]+', text)
    
    return len(chinese_chars) + len(english_words)

def compute_score(
    reward_inputs: List[Dict[str, Any]], 
    global_step: int = -1,
    beta: float = 1.0,
    gamma: float = 1.0,
    epsilon: float = 1e-8,
    rho: float = 0.2, 
    format_weight: float = 0.2,
    accuracy_weight: float = 0.4,
    length_weight: float = 0.1, 
    use_format_gate: bool = False
) -> List[Dict[str, float]]:

    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    
    for reward_input in reward_inputs:
        # ================= 1. CRE Computation (Normalized to [0, 1]) =================
        log_probs = reward_input["old_log_probs"]
        entropy_with_r = reward_input["old_entropy"]
        entropy_wo_r = reward_input["entropy_wo_r"]
        info_flow = reward_input["info_flow"]

        if not isinstance(log_probs, torch.Tensor):
            log_probs = torch.tensor(log_probs)
            entropy_with_r = torch.tensor(entropy_with_r)
            entropy_wo_r = torch.tensor(entropy_wo_r)
            info_flow = torch.tensor(info_flow)
            
        device = log_probs.device
        T = log_probs.shape[0]
        token_mask = torch.ones_like(entropy_wo_r)

        # 计算权重 alpha_prime (逻辑保持不变)
        raw_d_i = (entropy_with_r - entropy_wo_r) / (entropy_wo_r + epsilon)
        d_i = torch.clamp(raw_d_i, min=-10.0, max=10.0) * token_mask
        masked_info_flow = info_flow * token_mask
        raw_alpha_i = (1.0 / T) * (1.0 + beta * d_i + gamma * masked_info_flow)
        
        sum_alpha = torch.sum(raw_alpha_i)
        alpha_prime = raw_alpha_i / (sum_alpha + epsilon) if abs(sum_alpha) > epsilon else token_mask / (T + epsilon)

        # CRE 映射：将原本为负的 log_prob 期望通过 exp 映射到 [0, 1]

        cre_score_val = -torch.sum(alpha_prime * log_probs).item() 
        cre_reward = math.exp(-cre_score_val) 

        # ================= 2. Format & JSON Parsing (Robust) =================
        response_str = reward_input.get("response", "")
        has_markdown = "```json" in response_str and response_str.count("```") >= 2
        
        # 移除 <think> 标签进行解析
        clean_content = re.sub(r'<think>.*?</think>', '', response_str, flags=re.DOTALL).strip()
        
        r_format = 0.0
        if has_markdown:
            r_format += 0.15

        pred_data = None
        try:
            # 使用 json_repair 处理 None/True/False 及残缺括号
            pred_data = repair_json(clean_content, return_objects=True)
            if isinstance(pred_data, dict):
                r_format += 0.25 # 成功解析为字典
        except:
            pred_data = None

        # Schema 验证
        has_valid_structure = False
        if isinstance(pred_data, dict):
            if "plot_list" in pred_data and "next_episode" in pred_data:
                # 简单校验 plot_list 是否为非空列表
                if isinstance(pred_data["plot_list"], list) and len(pred_data["plot_list"]) > 0:
                    r_format += 0.60
                    has_valid_structure = True

        # ================= 3. Task Metric (Accuracy) =================
        ground_truth_str = reward_input.get("ground_truth", "")
        is_correct_next_ep = 0.0
        
        if ground_truth_str and has_valid_structure:
            try:
                # 尝试解析 ground_truth
                gt_dict = repair_json(ground_truth_str, return_objects=True)
                if isinstance(gt_dict, dict):
                    gt_next = str(gt_dict.get('next_episode', "")).strip()
                    pred_next = str(pred_data.get("next_episode", "")).strip()
                    if pred_next and pred_next == gt_next:
                        is_correct_next_ep = 1.0
            except:
                pass

        # ================= 4. Length Reward (Strict Content-Based) =================
        effective_char_count = 0
        effective_count = 0
        r_length = 0.0
        
        if has_valid_structure:
            for plot in pred_data["plot_list"]:
                if isinstance(plot, dict):
                    # 提取内容
                    narrative = plot.get("narrative", "")
                    utterance = ""
                    role_dialogue = plot.get("role_dialogue", {})
                    if isinstance(role_dialogue, dict):
                        utterance = role_dialogue.get("utterance", "")
                    
                    effective_count += count_tokens(str(narrative))
                    effective_count += count_tokens(str(utterance))

            effective_char_count = effective_count
            # 长度奖励函数 (Gaussian: 80-100 字符)
            min_target, max_target, sigma = 80, 100, 20
            if min_target <= effective_char_count <= max_target:
                r_length = 1.0
            else:
                target = min_target if effective_char_count < min_target else max_target
                r_length = math.exp(-0.5 * ((effective_char_count - target) / sigma) ** 2)
        else:
            # 结构不正确时，不再 fallback 到 raw string，直接判定有效长度为 0
            r_length = 0.0
            effective_char_count = 0

        # ================= 5. Final Reward Weighted Sum =================
        # 动态计算 CRE 权重
        current_cre_weight = 1.0 - format_weight - accuracy_weight - length_weight
        
        if use_format_gate:
            # 只有格式达到高分，才开启其它奖励，否则只给格式分
            if r_format >= 0.8:
                final_reward = (current_cre_weight * cre_reward + 
                                format_weight * r_format + 
                                accuracy_weight * is_correct_next_ep + 
                                length_weight * r_length)
            else:
                final_reward = r_format
        else:
            final_reward = (current_cre_weight * cre_reward + 
                            format_weight * r_format + 
                            accuracy_weight * is_correct_next_ep + 
                            length_weight * r_length)

        # Record metrics
        scores.append({
            "overall": final_reward,
            "cre_reward": cre_reward,
            "format_reward": r_format,
            "accuracy_reward": is_correct_next_ep,
            "length_reward": r_length,
            "effective_char_count": effective_char_count,
            "has_markdown": float(has_markdown),
            "val/acc_next_episode": is_correct_next_ep,
        })

    return scores