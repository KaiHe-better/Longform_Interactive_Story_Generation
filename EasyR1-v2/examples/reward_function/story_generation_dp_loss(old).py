import torch
from typing import Any, List, Dict
import numpy as np
import json
import ast
import re

def compute_score(
    reward_inputs: List[Dict[str, Any]], 
    global_step: int = -1,
    beta: float = 1.0,   # Context dependency strength weight
    gamma: float = 1.0,  # Information flow coefficient weight
    epsilon: float = 1e-8,
    rho: float = 0.2,    # Top-k ratio for high-entropy tokens
    format_weight: float = 0.3,  # Weight for format reward
    use_format_gate: bool = False  # Whether to use format as gate
) -> List[Dict[str, float]]:
    """
    计算基于 Contextual Reasoning Entropy (CRE) 的奖励分数。
    
    Args:
        reward_inputs: 包含 'old_log_probs', 'old_entropy', 'entropy_wo_r', 'info_flow' 的列表
        beta: 公式中 d_i 的权重
        gamma: 公式中 f_i 的权重
        epsilon: 防止除零的极小值
        rho: 选取高熵 token 的比例（default 0.2 表示 top 20%）
        format_weight: 格式奖励的权重
        use_format_gate: 是否使用格式作为门控（格式不对则不给 CRE 奖励）
    
    Returns:
        scores: 包含 'overall' 分数和其他统计信息的字典列表
    """
    
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    
    for reward_input in reward_inputs:
        # ============================================================
        # Part 1: 数据提取和预处理
        # ============================================================
        log_probs = reward_input["old_log_probs"]  # Shape: [T]
        entropy_with_r = reward_input["old_entropy"] # Shape: [T]
        entropy_wo_r = reward_input["entropy_wo_r"]  # Shape: [T]
        info_flow = reward_input["info_flow"]        # Shape: [T]

        # 确保转换为 Tensor
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
        # Part 2: Token Masking (80/20 原则)
        # ============================================================
        # 只对高熵 token 计算 CRE，过滤低熵的格式符号
        k = max(1, int(T * rho))  # 至少保留1个token
        
        # 获取第 k 大的熵值作为阈值
        topk_values, _ = torch.topk(entropy_wo_r, k)
        entropy_threshold = topk_values[-1]
        
        # 生成掩码：只有熵 >= 阈值的 token 参与计算
        token_mask = (entropy_wo_r >= entropy_threshold).float()
        
        # 统计有效 token 数量
        num_valid_tokens = token_mask.sum().item()

        # ============================================================
        # Part 3: 计算 Context Dependency (d_i)
        # ============================================================
        raw_d_i = (entropy_with_r - entropy_wo_r) / (entropy_wo_r + epsilon)
        
        # 截断极端值，防止数值不稳定
        raw_d_i = torch.clamp(raw_d_i, min=-10.0, max=10.0)
        
        # 应用 mask
        d_i = raw_d_i * token_mask

        # ============================================================
        # Part 4: 计算权重 alpha_i
        # ============================================================
        masked_info_flow = info_flow * token_mask
        
        # 修正：加上 1/T 归一化因子（虽然后面会重新归一化）
        raw_alpha_i = (1.0 / T) * (1.0 + beta * d_i + gamma * masked_info_flow)
        
        # 应用 mask，确保无效位置权重为 0
        alpha_i = raw_alpha_i * token_mask
        
        # ============================================================
        # Part 5: 归一化权重
        # ============================================================
        sum_alpha = torch.sum(alpha_i)
        
        if abs(sum_alpha) < epsilon:
            # 极端情况：平均分配给有效 token
            alpha_prime = token_mask / (token_mask.sum() + epsilon)
        else:
            alpha_prime = alpha_i / sum_alpha

        # ============================================================
        # Part 6: 计算 CRE
        # ============================================================
        weighted_log_prob = alpha_prime * log_probs
        cre_score = -torch.sum(weighted_log_prob)
        cre_reward = -cre_score.item()  # CRE 越小越好，所以取负

        # ============================================================
        # Part 7: 格式奖励计算
        # ============================================================
        response_str = reward_input.get("response", "")
        r_format = 0.0
        pred_data = None
        is_valid_json = False
        has_required_keys = False

        # 7.1 Markdown 检查
        if "```json" in response_str and "```" in response_str:
            r_format += 0.2

        # 7.2 JSON 解析
        try:
            json_match = re.search(r"```json\s*(.*?)\s*```", response_str, re.DOTALL)
            json_str = json_match.group(1) if json_match else response_str
            pred_data = json.loads(json_str.strip())
            r_format += 0.3
            is_valid_json = True
        except (json.JSONDecodeError, AttributeError):
            pred_data = None

        # 7.3 Schema 检查
        if is_valid_json and isinstance(pred_data, dict):
            if "plot_list" in pred_data and "next_episode" in pred_data:
                if isinstance(pred_data["plot_list"], list) and len(pred_data["plot_list"]) > 0:
                    r_format += 0.5
                    has_required_keys = True

        # ============================================================
        # Part 8: 组合最终奖励
        # ============================================================
        if use_format_gate:
            # 方案2: 格式门控 - 格式不对就不给 CRE 奖励
            if r_format >= 0.5:  # 至少有基本格式
                final_reward = cre_reward + r_format
            else:
                final_reward = r_format  # 只给格式分
        else:
            # 方案1/3: 加权组合（推荐）
            # 如果 format_weight=0，则退化为纯 CRE
            # 如果 format_weight=1，则只看格式
            if (global_step > 25 and r_format > 0.85):
                format_weight = 0.1
            final_reward = (1 - format_weight) * cre_reward + format_weight * r_format

        # ============================================================
        # Part 9: 任务指标计算（监控用）
        # ============================================================
        ground_truth_str = reward_input.get("ground_truth", "")
        is_correct_next_ep = 0.0
        pred_next = "N/A"
        gt_next = "Unknown"

        if ground_truth_str:
            try:
                try:
                    gt_data = ast.literal_eval(ground_truth_str)
                except (ValueError, SyntaxError):
                    gt_data = json.loads(ground_truth_str)
            except Exception:
                gt_data = None
                
            if gt_data and has_required_keys:
                pred_next = str(pred_data.get("next_episode", "")).strip()
                gt_next = str(gt_data.get("next_episode", "")).strip()

                if pred_next == gt_next:
                    is_correct_next_ep = 1.0

        # ============================================================
        # Part 10: 记录详细指标
        # ============================================================
        scores.append({
            # 主要奖励
            "overall": final_reward,
            
            # CRE 相关
            "cre": cre_score.item(),
            "cre_reward": cre_reward,
            
            # 格式相关
            "format_reward": r_format,
            
            # 熵统计
            "avg_entropy_with_r": torch.mean(entropy_with_r).item(),
            "avg_entropy_wo_r": torch.mean(entropy_wo_r).item(),
            "avg_entropy_with_r_masked": (torch.sum(entropy_with_r * token_mask) / (num_valid_tokens + epsilon)).item(),
            "avg_entropy_wo_r_masked": (torch.sum(entropy_wo_r * token_mask) / (num_valid_tokens + epsilon)).item(),
            
            # 依赖性和信息流
            "avg_dependency_d": torch.mean(d_i).item(),
            "avg_info_flow_f": torch.mean(info_flow).item(),
            
            # Token 统计
            "num_total_tokens": T,
            "num_valid_tokens": num_valid_tokens,
            "valid_token_ratio": num_valid_tokens / T,
            
            # 权重统计
            "sum_alpha": sum_alpha.item(),
            "max_alpha": torch.max(alpha_prime).item(),
            "min_alpha": torch.min(alpha_prime[token_mask > 0]).item() if num_valid_tokens > 0 else 0.0,
            
            # 任务指标
            "val/acc_next_episode": is_correct_next_ep,
        })

    return scores