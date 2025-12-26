import torch
from typing import Any, List, Dict
import json
import re
import ast
def compute_score(
    reward_inputs: List[Dict[str, Any]], 
    global_step: int = -1,
    beta: float = 1.0,
    gamma: float = 1.0,
    epsilon: float = 1e-8,
    rho: float = 0.2, # 虽然不再使用rho进行筛选，但保留参数以维持接口兼容
    format_weight: float = 0.3,
    accuracy_weight: float = 0.2,
    use_format_gate: bool = False
) -> List[Dict[str, float]]:
    """
    Compute reward scores based on Contextual Reasoning Entropy (CRE) WITHOUT masking (using all tokens).
    """
    
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    
    for reward_input in reward_inputs:
        # Extract and preprocess data
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

        # ================= MODIFICATION START =================
        # 原有的 Top-k mask 逻辑已被移除
        # k = max(1, int(T * rho))
        # topk_values, _ = torch.topk(entropy_wo_r, k)
        # entropy_threshold = topk_values[-1]
        # token_mask = (entropy_wo_r >= entropy_threshold).float()
        
        # 现在直接使用全 1 的 mask，即计算时考虑所有 Token
        token_mask = torch.ones_like(entropy_wo_r)
        num_valid_tokens = token_mask.sum().item() # 这里现在等于 T
        # ================= MODIFICATION END ===================

        # Compute Context Dependency (d_i)
        # 保持原有逻辑，token_mask 为全1时相当于没有mask
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

        # Format reward computation
        response_str = reward_input.get("response", "")
        r_format = 0.0
        pred_data = None
        is_valid_json = False
        has_valid_structure = False

        # Remove <think> tags if present
        response_str = re.sub(r'<think>.*?</think>', '', response_str, flags=re.DOTALL).strip()

        # Check markdown wrapper (optional now)
        has_markdown = "```json" in response_str and response_str.count("```") >= 2
        if has_markdown:
            r_format += 0.15

        # Parse JSON with multiple strategies
        ast_str = None
        try:
            # Strategy 1: Try markdown wrapper first
            if has_markdown:
                ast_match = re.search(r"```json\s*(.*?)\s*```", response_str, re.DOTALL)
                if ast_match:
                    ast_str = ast_match.group(1).strip()
            
            # Strategy 2: Extract content between outermost curly braces
            if not ast_str:
                # Find the first '{' and last '}' to extract the JSON object
                first_brace = response_str.find('{')
                last_brace = response_str.rfind('}')
                
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    ast_str = response_str[first_brace:last_brace + 1]
            
            # Try to parse the extracted string
            if ast_str:
                pred_data = ast.literal_eval(ast_str)
                r_format += 0.25
                is_valid_json = True
            else:
                r_format = 0.0
                
        except (json.JSONDecodeError, AttributeError, SyntaxError, ValueError) as e:
            # Fallback: try json.loads if ast.literal_eval fails
            try:
                if ast_str:
                    pred_data = json.loads(ast_str)
                    r_format += 0.25
                    is_valid_json = True
            except:
                pred_data = None
                r_format = 0.0


        # Strict schema validation
        if is_valid_json and isinstance(pred_data, dict):
            if "plot_list" in pred_data and "next_episode" in pred_data:
                plot_list = pred_data["plot_list"]
                
                if isinstance(plot_list, list) and len(plot_list) > 0:
                    valid_plots = True
                    
                    for plot in plot_list:
                        if not isinstance(plot, dict):
                            valid_plots = False
                            break
                        
                        if "narrative" not in plot or "role_dialogue" not in plot:
                            valid_plots = False
                            break
                        
                        if not isinstance(plot["narrative"], str):
                            valid_plots = False
                            break
                        
                        role_dialogue = plot["role_dialogue"]
                        if not isinstance(role_dialogue, dict):
                            valid_plots = False
                            break
                        
                        if role_dialogue:
                            if "name" not in role_dialogue or "utterance" not in role_dialogue:
                                valid_plots = False
                                break
                            if not isinstance(role_dialogue["name"], str) or not isinstance(role_dialogue["utterance"], str):
                                valid_plots = False
                                break
                    
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

        # Task metrics computation
        ground_truth_str = reward_input.get("ground_truth", "")
        is_correct_next_ep = 0.0
        pred_next = "N/A"
        gt_next = "Unknown"

        if ground_truth_str and has_valid_structure:
            try:
                # gt_match = re.search(r'"next_episode"\s*:\s*"([^"]+)"', ground_truth_str)
                gt_dict = ast.literal_eval(ground_truth_str)
                
                if gt_dict:
                    # gt_next = gt_match.group(1).strip()
                    gt_next = str(gt_dict.get('next_episode', "")).strip()
                    pred_next = str(pred_data.get("next_episode", "")).strip()
                    
                    if pred_next == gt_next:
                        is_correct_next_ep = 1.0
            except Exception:
                pass

        # Combine final reward with accuracy
        if use_format_gate:
            if r_format >= 0.8:
                # Normalize weights to sum to 1
                cre_weight = 1.0 - format_weight - accuracy_weight
                final_reward = cre_weight * cre_reward + format_weight * r_format + accuracy_weight * is_correct_next_ep
            else:
                final_reward = r_format
        else:
            # Adjust format weight based on training progress
            current_format_weight = format_weight
            
            # Normalize weights to sum to 1
            cre_weight = 1.0 - current_format_weight - accuracy_weight
            final_reward = cre_weight * cre_reward + current_format_weight * r_format + accuracy_weight * is_correct_next_ep

        # Record metrics
        # 注意：这里所有的key都原样保留。
        # 由于 token_mask 现在是全1，_masked 的指标将与非 masked 指标数值基本一致，
        # num_valid_tokens 将等于 num_total_tokens。
        scores.append({
            "overall": final_reward,
            "cre": cre_score.item(),
            "cre_reward": cre_reward,
            "format_reward": r_format,
            "accuracy_reward": is_correct_next_ep,
            "avg_entropy_with_r": torch.mean(entropy_with_r).item(),
            "avg_entropy_wo_r": torch.mean(entropy_wo_r).item(),
            "avg_entropy_with_r_masked": (torch.sum(entropy_with_r * token_mask) / (num_valid_tokens + epsilon)).item(),
            "avg_entropy_wo_r_masked": (torch.sum(entropy_wo_r * token_mask) / (num_valid_tokens + epsilon)).item(),
            "avg_dependency_d": torch.mean(d_i).item(),
            "avg_info_flow_f": torch.mean(info_flow).item(),
            "num_total_tokens": T,
            "num_valid_tokens": num_valid_tokens,
            "valid_token_ratio": num_valid_tokens / T,
            "sum_alpha": sum_alpha.item(),
            "max_alpha": torch.max(alpha_prime).item(),
            "min_alpha": torch.min(alpha_prime[token_mask > 0]).item() if num_valid_tokens > 0 else 0.0,
            "val/acc_next_episode": is_correct_next_ep,
        })

    return scores