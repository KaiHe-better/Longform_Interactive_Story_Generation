import torch
from typing import Any, List, Dict
import numpy as np
import json
import ast
import re

def compute_score(
    reward_inputs: List[Dict[str, Any]], 
    beta: float = 1.0,   # Context dependency strength weight
    gamma: float = 1.0,  # Information flow coefficient weight
    epsilon: float = 1e-8
) -> List[Dict[str, float]]:
    """
    计算基于 Contextual Reasoning Entropy (CRE) 的奖励分数。
    
    Args:
        reward_inputs: 包含 'old_log_probs', 'old_entropy', 'entropy_wo_r', 'info_flow' 的列表
        beta: 公式中 d_i 的权重
        gamma: 公式中 f_i 的权重
        epsilon: 防止除零的极小值
    
    Returns:
        scores: 包含 'overall' 分数和其他统计信息的字典列表
    """
    
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    
    for reward_input in reward_inputs:
        # 1. 提取数据 (假设数据已经是 Tensor 且在同一 Device 上，或者是 CPU Tensor)
        # 注意：这里假设输入数据的长度是 T (response length)，即已经是 valid token level
        # breakpoint()
        # p(y|x, r) 的 log probability
        log_probs = reward_input["old_log_probs"]  # Shape: [T]
        
        # Entropy with context: S(p(y_i | y_<i, x, r))
        entropy_with_r = reward_input["old_entropy"] # Shape: [T]
        
        # Entropy without context: S(p(y_i | y_<i, x))
        entropy_wo_r = reward_input["entropy_wo_r"]  # Shape: [T]
        
        # Information flow: f_i
        info_flow = reward_input["info_flow"]        # Shape: [T]

        # 确保转换为 Tensor (以防万一是 list 或 numpy)
        if not isinstance(log_probs, torch.Tensor):
            log_probs = torch.tensor(log_probs)
            entropy_with_r = torch.tensor(entropy_with_r)
            entropy_wo_r = torch.tensor(entropy_wo_r)
            info_flow = torch.tensor(info_flow)
            
        # 确保数据类型一致
        device = log_probs.device
        entropy_with_r = entropy_with_r.to(device)
        entropy_wo_r = entropy_wo_r.to(device)
        info_flow = info_flow.to(device)

        T = log_probs.shape[0]

        # ============================================================
        # Component 1: Context dependency strength (d_i)
        # Formula: d_i = (S_with - S_wo) / (S_wo + epsilon)
        # ============================================================
        d_i = (entropy_with_r - entropy_wo_r) / (entropy_wo_r + epsilon)

        # ============================================================
        # Calculate Weights: alpha_i
        # Formula: alpha_i = (1/T) * (1 + beta * d_i + gamma * f_i)
        # ============================================================
        # 注意：d_i 通常是负数（因为有了 Context 熵通常变小），
        # 如果 beta 很大，1 + beta * d_i 可能会变成负数，导致权重为负。
        # 根据具体论文实现，有时需要对 alpha_i 进行 ReLU 或者 abs 操作，
        # 但这里我们严格按照你的公式实现。
        
        alpha_i = (1.0 / T) * (1.0 + beta * d_i + gamma * info_flow)
        
        # ============================================================
        # Normalize Weights: alpha'_i
        # Formula: alpha'_i = alpha_i / sum(alpha_k)
        # ============================================================
        sum_alpha = torch.sum(alpha_i)
        
        # 防止 sum_alpha 为 0
        if abs(sum_alpha) < epsilon:
            alpha_prime = torch.full_like(alpha_i, 1.0 / T)
        else:
            alpha_prime = alpha_i / sum_alpha

        # ============================================================
        # Calculate CRE
        # Formula: CRE = - sum( alpha'_i * log(p(y|...)) )
        # ============================================================
        # log_probs 本身是 log(p)，通常是负数。
        # weighted_log_prob = alpha' * log_p
        weighted_log_prob = alpha_prime * log_probs
        
        # CRE 是 Weighted Negative Log Likelihood
        cre_score = -torch.sum(weighted_log_prob)

        # ============================================================
        # Final Reward Calculation
        # ============================================================
        # 在 RL 中我们需要最大化 Reward。
        # CRE 越小（熵越低，确定性越高），生成的质量通常越好。
        # 所以 Reward 设为 -CRE (或者 exp(-CRE))
        
        final_reward = -cre_score.item()
        


        response_str = reward_input.get("response", "")
        r_format = 0.0
        pred_data = None
        is_valid_json = False
        has_required_keys = False

        # 2.1 Markdown 检查
        if "```json" in response_str and "```" in response_str:
            r_format += 0.2

        # 2.2 JSON 解析
        try:
            json_match = re.search(r"```json\s*(.*?)\s*```", response_str, re.DOTALL)
            json_str = json_match.group(1) if json_match else response_str
            pred_data = json.loads(json_str.strip())
            r_format += 0.3
            is_valid_json = True
        except (json.JSONDecodeError, AttributeError):
            pred_data = None

        # 2.3 Schema 检查
        if is_valid_json and isinstance(pred_data, dict):
            if "plot_list" in pred_data and "next_episode" in pred_data:
                if isinstance(pred_data["plot_list"], list) and len(pred_data["plot_list"]) > 0:
                    r_format += 0.5
                    has_required_keys = True

        # ============================================================
        # Part 3: Task Metric (剧情跳转准确率) - 仅用于监控
        # ============================================================
        ground_truth_str = reward_input.get("ground_truth", "")
        
        # 指标初始化
        is_correct_next_ep = 0.0
        pred_next = "N/A"
        gt_next = "Unknown"

        if ground_truth_str:
            try:
                # 尝试解析 Ground Truth
                try:
                    gt_data = ast.literal_eval(ground_truth_str)
                except (ValueError, SyntaxError):
                    gt_data = json.loads(ground_truth_str)
            except Exception:
                gt_data = None
                
            if gt_data and has_required_keys:
                # 提取预测值和真实值
                pred_next = str(pred_data.get("next_episode", "")).strip()
                gt_next = str(gt_data.get("next_episode", "")).strip()

                # 判断准确性
                if pred_next == gt_next:
                    is_correct_next_ep = 1.0
                else:
                    is_correct_next_ep = 0.0
        # 记录详细信息以便 Debug
        scores.append(
            {
                "overall": final_reward,
                "cre": cre_score.item(),
                "avg_entropy_with_r": torch.mean(entropy_with_r).item(),
                "avg_entropy_wo_r": torch.mean(entropy_wo_r).item(),
                "avg_dependency_d": torch.mean(d_i).item(),
                "avg_info_flow_f": torch.mean(info_flow).item(),
                "sum_alpha": sum_alpha.item(),
                "val/acc_next_episode": is_correct_next_ep,
            }
        )

    return scores