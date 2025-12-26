import torch
from typing import Any, List, Dict
import json
import re
import ast
import math
from json_repair import repair_json

def fix_dirty_json(text: str):
    # 预处理：去掉可能导致解析器混淆的 Python 风格 None/True/False
    text = text.replace("None", "null").replace(
        "True", "true").replace("False", "false")

    # 尝试使用 json_repair 直接修复并加载
    try:
        # return_objects=True 会直接返回字典，而不是 json 字符串
        return repair_json(text, return_objects=True)
    except Exception as e:
        # 如果 json_repair 失败，尝试最后的兜底：强制转义所有内部单引号后再次尝试
        # 这种情况极少发生，但在 Story Generation 中模型可能会生成极其破碎的文本
        pass

    raise ValueError(f"Unable to repair JSON: {str(e)}\nInput: {text}")

def compute_score(
    reward_inputs: List[Dict[str, Any]], 
    global_step: int = -1,
    beta: float = 1.0,
    gamma: float = 1.0,
    epsilon: float = 1e-8,
    rho: float = 0.2, 
    format_weight: float = 0.15,
    accuracy_weight: float = 0.3,
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

        # 使用全 1 mask (User strategy)
        token_mask = torch.ones_like(entropy_wo_r)
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
        # cre_reward = -cre_score.item()
        # 映射到0-1
        cre_reward = torch.exp(-cre_score).item()
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
                try:
                    pred_data = ast.literal_eval(ast_str)
                    r_format += 0.25
                    is_valid_json = True
                except:
                    pred_data = None
                    r_format = 0.0
                
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
        r_length = 0.0
        
        parsed_data = None
        try:
            parsed_data = repair_json(response_str, return_objects=True)
        except:
            parsed_data = None

        # 只有当解析出正确的字典结构时才计算长度
        if isinstance(parsed_data, dict) and "plot_list" in parsed_data:
            # 仅统计用户真正关心的“实质内容”
            for plot in parsed_data["plot_list"]:
                if isinstance(plot, dict):
                    # 统计剧情描写
                    effective_char_count += len(str(plot.get("narrative", "")))
                    # 统计对话内容
                    role_dialogue = plot.get("role_dialogue", {})
                    if isinstance(role_dialogue, dict):
                        effective_char_count += len(str(role_dialogue.get("utterance", "")))
            
            min_target, max_target, sigma = 80, 100, 10
            if min_target <= effective_char_count <= max_target:
                r_length = 1.0
            else:
                target = min_target if effective_char_count < min_target else max_target
                r_length = math.exp(-0.5 * ((effective_char_count - target) / sigma) ** 2)
        else:
            r_length = 0.0
            effective_char_count = 0

        if use_format_gate:
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

        # Record metrics
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