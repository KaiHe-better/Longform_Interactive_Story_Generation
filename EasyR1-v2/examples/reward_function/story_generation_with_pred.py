import torch
from typing import Any, List, Dict
import numpy as np
import json
import ast
import re

def compute_score(
    reward_inputs: List[Dict[str, Any]], 
    beta: float = 0.5,    # 建议降低 beta，防止 d_i 过度主导
    gamma: float = 0.5,   
    epsilon: float = 1e-8,
    
    # 权重配置 
    weight_format: float = 0.2,    
    weight_accuracy: float = 1.0,  # 提高准确率的权重
    weight_cre: float = 1.0,       
    
    # 格式错误时的强惩罚 (Base penalty)
    format_penalty: float = -2.0,  
) -> List[Dict[str, float]]:
    """
    Refactored Reward Function with Consistent Naming.
    - 恢复了原始的 key 命名 (val/acc_next_episode, avg_dependency_d 等)
    - 保留了 d_i 截断和格式门控机制
    """
    
    scores = []
    
    for reward_input in reward_inputs:
        # ============================================================
        # Part 1: 格式与任务指标计算 (Format & Accuracy)
        # ============================================================
        response_str = reward_input.get("response", "")
        ground_truth_str = reward_input.get("ground_truth", "")
        
        # --- 1.1 格式评分 (r_format) ---
        r_format = 0.0
        pred_data = None
        is_valid_json = False
        has_required_keys = False
        
        # A. Markdown 检查
        if "```json" in response_str and "```" in response_str:
            r_format += 0.2
            
        # B. JSON 解析
        try:
            json_match = re.search(r"```json\s*(.*?)\s*```", response_str, re.DOTALL)
            json_str = json_match.group(1) if json_match else response_str
            pred_data = json.loads(json_str.strip())
            r_format += 0.3
            is_valid_json = True
        except (json.JSONDecodeError, AttributeError):
            pred_data = None
            
        # C. Schema 检查
        if is_valid_json and isinstance(pred_data, dict):
            if "plot_list" in pred_data and "next_episode" in pred_data:
                if isinstance(pred_data["plot_list"], list):
                    r_format += 0.5
                    has_required_keys = True

        # --- 1.2 准确率评分 (is_correct_next_ep) ---
        is_correct_next_ep = 0.0
        
        # 只有在解析成功时才计算准确率，否则保持 0.0
        if has_required_keys and ground_truth_str:
            try:
                try:
                    gt_data = ast.literal_eval(ground_truth_str)
                except (ValueError, SyntaxError):
                    gt_data = json.loads(ground_truth_str)
                
                if gt_data:
                    gt_next = str(gt_data.get("next_episode", "")).strip()
                    pred_next = str(pred_data.get("next_episode", "")).strip()
                    
                    if gt_next and pred_next and (gt_next == pred_next):
                        is_correct_next_ep = 1.0
            except Exception:
                pass 

        # ============================================================
        # Part 2: CRE (Contextual Reasoning Entropy) 计算
        # ============================================================
        
        # 初始化默认值，防止格式错误时变量未定义
        cre_score_val = 0.0 
        avg_entropy_with = 0.0
        avg_entropy_wo = 0.0
        avg_d_i = 0.0
        avg_info = 0.0
        sum_alpha_val = 0.0
        cre_reward_term = 0.0

        # 门控：如果连 JSON 都解析不出来，跳过 CRE 计算以节省资源并防止污染
        if is_valid_json: 
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

            # --- 2.1 计算 d_i (带截断) ---
            # 原始公式: d_i = (H_with - H_wo) / H_wo
            raw_d_i = (entropy_with_r - entropy_wo_r) / (entropy_wo_r + epsilon)
            
            # [CRITICAL] 强力截断：限制在 [-5, 5] 之间，防止 8000+ 这种情况出现
            d_i = torch.clamp(raw_d_i, min=-5.0, max=5.0) 
            
            # --- 2.2 计算 Alpha ---
            alpha_term = 1.0 + beta * d_i + gamma * info_flow
            # 使用 ReLU 保证非负
            alpha_i = (1.0 / T) * torch.relu(alpha_term)

            # --- 2.3 归一化 ---
            sum_alpha = torch.sum(alpha_i)
            if sum_alpha < epsilon:
                alpha_prime = torch.full_like(alpha_i, 1.0 / T)
            else:
                alpha_prime = alpha_i / sum_alpha

            # --- 2.4 计算最终 CRE ---
            weighted_log_prob = alpha_prime * log_probs
            cre_score = -torch.sum(weighted_log_prob)
            
            # 记录用于日志的原始数值
            cre_score_val = cre_score.item()
            avg_entropy_with = torch.mean(entropy_with_r).item()
            avg_entropy_wo = torch.mean(entropy_wo_r).item()
            avg_d_i = torch.mean(d_i).item()
            avg_info = torch.mean(info_flow).item()
            sum_alpha_val = sum_alpha.item()
            
            # 实际用于奖励的部分 (CRE 越小越好，取负)
            cre_reward_term = -cre_score_val

        # ============================================================
        # Part 3: 总分聚合 (Overall Reward)
        # ============================================================
        final_reward = 0.0
        
        if not has_required_keys:
            # 场景 A: 格式完全错误 -> 给予惩罚
            # format_penalty (-2.0) + 获得的少量格式分
            final_reward = format_penalty + (r_format * weight_format)
        else:
            # 场景 B: 格式正确 -> 综合打分
            # Overall = Format + Accuracy + CRE
            final_reward = (
                (r_format * weight_format) + 
                (is_correct_next_ep * weight_accuracy) + 
                (cre_reward_term * weight_cre)
            )

        # ============================================================
        # Part 4: 构造返回值 (保持原有命名规则)
        # ============================================================
        scores.append({
            # 核心奖励
            "overall": final_reward,
            
            # 原始指标 (用于绘图监控)
            "cre": cre_score_val,
            "avg_entropy_with_r": avg_entropy_with,
            "avg_entropy_wo_r": avg_entropy_wo,
            "avg_dependency_d": avg_d_i,
            "avg_info_flow_f": avg_info,
            "sum_alpha": sum_alpha_val,
            
            # 任务指标
            "r_format": r_format,
            "val/acc_next_episode": is_correct_next_ep,
        })
        
    return scores