import torch
from typing import Any, List, Dict
import json
import re
import ast
import math

def compute_score(
    reward_inputs: List[Dict[str, Any]], 
    global_step: int = -1,
    beta: float = 1.0,
    gamma: float = 1.0,
    epsilon: float = 1e-8,
    rho: float = 0.2, 
    format_weight: float = 0.3,
    accuracy_weight: float = 0.2,
    length_weight: float = 0.15, # 新增长度奖励权重
    use_format_gate: bool = False
) -> List[Dict[str, float]]:
    """
    Compute reward scores composed of:
    1. Contextual Reasoning Entropy (CRE)
    2. Format Reward
    3. Next Episode Accuracy
    4. Length Constraint (Gaussian distribution target: 80-100 chars)
    """
    
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    
    for reward_input in reward_inputs:
        # ================= 1. CRE Computation =================
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
        entropy_with_r = entropy_with_r.to(device)
        entropy_wo_r = entropy_wo_r.to(device)
        info_flow = info_flow.to(device)

        T = log_probs.shape[0]

        # token_mask = torch.ones_like(entropy_wo_r)
        k = max(1, int(T * rho))
        topk_values, _ = torch.topk(entropy_wo_r, k)
        entropy_threshold = topk_values[-1]
        token_mask = (entropy_wo_r >= entropy_threshold).float()
        
        num_valid_tokens = token_mask.sum().item()

        # Compute Context Dependency (d_i)
        raw_d_i = (entropy_with_r - entropy_wo_r) / (entropy_wo_r + epsilon)
        d_i = torch.clamp(raw_d_i, min=-10.0, max=10.0) * token_mask

        # Compute weights alpha_i
        masked_info_flow = info_flow * token_mask
        raw_alpha_i = (1.0 / T) * (1.0 + beta * d_i + gamma * masked_info_flow)
        alpha_i = raw_alpha_i * token_mask
        
        # Normalize weights
        sum_alpha = torch.sum(alpha_i)
        
        if abs(sum_alpha) < epsilon:
            alpha_prime = token_mask / (token_mask.sum() + epsilon)
        else:
            alpha_prime = alpha_i / sum_alpha

        # Compute CRE
        weighted_log_prob = alpha_prime * log_probs
        cre_score = -torch.sum(weighted_log_prob)
        cre_reward = -cre_score.item()

        # ================= 2. Format Reward Parsing =================
        response_str = reward_input.get("response", "")
        r_format = 0.0
        pred_data = None
        is_valid_json = False
        has_valid_structure = False

        # Remove <think> tags
        response_str = re.sub(r'<think>.*?</think>', '', response_str, flags=re.DOTALL).strip()

        # Check markdown
        has_markdown = "```json" in response_str and response_str.count("```") >= 2
        if has_markdown:
            r_format += 0.15

        # Parse JSON
        ast_str = None
        try:
            if has_markdown:
                ast_match = re.search(r"```json\s*(.*?)\s*```", response_str, re.DOTALL)
                if ast_match:
                    ast_str = ast_match.group(1).strip()
            
            if not ast_str:
                first_brace = response_str.find('{')
                last_brace = response_str.rfind('}')
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    ast_str = response_str[first_brace:last_brace + 1]
            
            if ast_str:
                pred_data = ast.literal_eval(ast_str)
                r_format += 0.25
                is_valid_json = True
            else:
                r_format = 0.0
                
        except (json.JSONDecodeError, AttributeError, SyntaxError, ValueError):
            try:
                if ast_str:
                    pred_data = json.loads(ast_str)
                    r_format += 0.25
                    is_valid_json = True
            except:
                pred_data = None
                r_format = 0.0

        # Schema Validation
        if is_valid_json and isinstance(pred_data, dict):
            if "plot_list" in pred_data and "next_episode" in pred_data:
                plot_list = pred_data["plot_list"]
                
                if isinstance(plot_list, list) and len(plot_list) > 0:
                    valid_plots = True
                    for plot in plot_list:
                        if not isinstance(plot, dict):
                            valid_plots = False; break
                        if "narrative" not in plot or "role_dialogue" not in plot:
                            valid_plots = False; break
                        if not isinstance(plot["narrative"], str):
                            valid_plots = False; break
                        role_dialogue = plot["role_dialogue"]
                        if not isinstance(role_dialogue, dict):
                            valid_plots = False; break
                        if role_dialogue:
                            if "name" not in role_dialogue or "utterance" not in role_dialogue:
                                valid_plots = False; break
                    
                    if valid_plots and isinstance(pred_data["next_episode"], str):
                        r_format += 0.60
                        has_valid_structure = True
                    else:
                        r_format = 0.0
                else:
                    r_format = 0.0
            else:
                r_format = 0.0
        else:
            r_format = 0.0

        # ================= 3. Task Metric (Accuracy) =================
        ground_truth_str = reward_input.get("ground_truth", "")
        is_correct_next_ep = 0.0
        
        if ground_truth_str and has_valid_structure:
            try:
                gt_dict = ast.literal_eval(ground_truth_str)
                if gt_dict:
                    gt_next = str(gt_dict.get('next_episode', "")).strip()
                    pred_next = str(pred_data.get("next_episode", "")).strip()
                    if pred_next == gt_next:
                        is_correct_next_ep = 1.0
            except Exception:
                pass

        # ================= 4. Length Reward (New) =================
        # 目标：有效字数在 [80, 100] 之间奖励最大
        r_length = 0.0
        effective_char_count = 0
        
        if has_valid_structure and pred_data:
            # 策略A: 既然结构正确，统计 plot_list 中的 narrative 和 utterance 总字数
            try:
                for plot in pred_data["plot_list"]:
                    effective_char_count += len(plot.get("narrative", ""))
                    if plot.get("role_dialogue"):
                        effective_char_count += len(plot["role_dialogue"].get("utterance", ""))
            except:
                effective_char_count = len(response_str) # Fallback
        else:
            # 策略B: 结构不正确，统计去除 think 标签后的 raw string 长度
            # 注意：这种情况下包含 json 符号，可能偏大，但作为惩罚依据足够
            effective_char_count = len(response_str)

        # 定义长度奖励函数 (Flat-top Gaussian)
        min_target = 80
        max_target = 100
        sigma = 20.0  # 控制衰减速度，sigma越小，超出范围后分数下降越快

        if min_target <= effective_char_count <= max_target:
            r_length = 1.0
        elif effective_char_count < min_target:
            # 过短：以 min_target 为中心衰减
            r_length = math.exp(-0.5 * ((effective_char_count - min_target) / sigma) ** 2)
        else:
            # 过长：以 max_target 为中心衰减
            r_length = math.exp(-0.5 * ((effective_char_count - max_target) / sigma) ** 2)
        
        if use_format_gate:
            # 废弃
            if r_format >= 0.8:
                remaining_weight = 1.0 - format_weight
                
                current_cre_weight = 1.0 - format_weight - accuracy_weight - length_weight
                
                if current_cre_weight < 0:
                    total_specified = format_weight + accuracy_weight + length_weight
                    norm_factor = 1.0 / total_specified
                    final_reward = (format_weight * norm_factor * r_format + 
                                    accuracy_weight * norm_factor * is_correct_next_ep + 
                                    length_weight * norm_factor * r_length)
                else:
                    final_reward = (current_cre_weight * cre_reward + 
                                    format_weight * r_format + 
                                    accuracy_weight * is_correct_next_ep + 
                                    length_weight * r_length)
            else:
                final_reward = r_format
        else:

            current_cre_weight = 1.0 - format_weight - accuracy_weight - length_weight
            final_reward = (current_cre_weight * cre_reward + 
                            format_weight * r_format + 
                            accuracy_weight * is_correct_next_ep + 
                            length_weight * r_length)

        scores.append({
            "overall": final_reward,
            "cre": cre_score.item(),
            "cre_reward": cre_reward,
            "format_reward": r_format,
            "accuracy_reward": is_correct_next_ep,
            "length_reward": r_length,
            "char_count": effective_char_count,
            "avg_entropy_with_r": torch.mean(entropy_with_r).item(),
            "avg_entropy_wo_r": torch.mean(entropy_wo_r).item(),
            "avg_dependency_d": torch.mean(d_i).item(),
            "avg_info_flow_f": torch.mean(info_flow).item(),
            "num_total_tokens": T,
            "num_valid_tokens": num_valid_tokens,
            "sum_alpha": sum_alpha.item(),
            "val/acc_next_episode": is_correct_next_ep,
        })

    return scores