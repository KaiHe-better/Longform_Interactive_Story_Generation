import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
import re
from transformers import AutoTokenizer, AutoModel
import numpy as np


def calculate_entropy(logits: torch.Tensor) -> torch.Tensor:
    """计算概率分布的熵 S(p) = - sum(p * log_p)"""
    return torch.distributions.Categorical(logits=logits).entropy()


def calculate_context_dependency(
    logits_with_r: torch.Tensor, logits_without_r: torch.Tensor, epsilon: float = 1e-8
) -> torch.Tensor:
    """计算上下文依赖强度 d_i"""
    entropy_with_r = calculate_entropy(logits_with_r)
    entropy_without_r = calculate_entropy(logits_without_r)

    d = (entropy_without_r - entropy_with_r) / (entropy_without_r + epsilon)
    return d


def calculate_info_flow(token_embeddings: torch.Tensor) -> torch.Tensor:
    """
    计算信息流系数 f_i
    """
    T, _ = token_embeddings.shape
    if T <= 1:
        return torch.zeros(T, device=token_embeddings.device)

    norm_embeds = F.normalize(token_embeddings, p=2, dim=1)
    cosine_sim_matrix = torch.matmul(norm_embeds, norm_embeds.t())

    f = torch.zeros(T, device=token_embeddings.device)
    for i in range(1, T):
        sum_sim = torch.sum(cosine_sim_matrix[i, :i])
        f[i] = torch.clamp(sum_sim / i, min=0.0)

    return f


def format_reward(response: str) -> float:
    '''理论上还是要有一个 format_reward，这是老版的使用 think 过程的，这里没用上'''
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def _as_records(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, dict):
        lists = {k: v for k, v in data.items() if isinstance(v, list)}
        if not lists:
            return [data]
        L = max(len(v) for v in lists.values())
        recs: List[Dict[str, Any]] = []
        for i in range(L):
            rec: Dict[str, Any] = {}
            for k, v in data.items():
                if isinstance(v, list):
                    rec[k] = v[i] if i < len(v) else None
                else:
                    rec[k] = v
            recs.append(rec)
        return recs
    if isinstance(data, list):
        if (
            data
            and all(isinstance(x, dict) for x in data)
            and all(any(isinstance(v, list) for v in x.values()) for x in data)
        ):
            merged: Dict[str, Any] = {}
            for d in data:
                merged.update(d)
            return _as_records(merged)
        return data
    raise ValueError("need batched inputs")


def _first_nonempty(d: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in d and d[k] is not None:
            s = str(d[k])
            if s.strip() != "":
                return s
    return None


def _normalize_text(s: str) -> str:
    s = str(s or "")
    s = re.sub(r"\s+", " ", s.strip())
    return s if s else "<EMPTY>"


def compute_score(
    reward_inputs: List[Dict[str, Any]],
    beta: float = 1.0,
    gamma: float = 1.0,
    epsilon: float = 1e-8,
    model_name: str = "/data01/sdz/models/Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418",
    device: Optional[str] = None,
    # 没用到以下
    batch_size: int = 64,
    max_input_chars: int = 2000,
    max_reply_chars: int = 1000,
) -> List[Dict[str, float]]:

    items = _as_records(reward_inputs)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cache_key = (model_name, device)
    cache = getattr(compute_score, "_embedder_cache", None)
    if cache is None or not isinstance(cache, dict):
        cache = {}
        setattr(compute_score, "_embedder_cache", cache)

    if cache_key not in cache:
        model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, output_hidden_states=True
        ).to(device)
        model.eval()
        cache[cache_key] = model

    model = cache[cache_key]
    embeddings_layer = model.get_input_embeddings()

    scores = []

    # 每个 item 的 T 长度都不同，还是得逐项处理
    for item in items:

        logits_with_r = item.get("logits_with_r")
        logits_without_r = item.get("logits_without_r")
        responses_id = item.get("responses")

        if logits_with_r is None or logits_without_r is None or responses_id is None:
            scores.append({"overall": 0.0, "cre_loss": 0.0, "format_reward": 0.0})
            continue

        try:
            if isinstance(logits_with_r, np.ndarray):
                logits_with_r = torch.tensor(logits_with_r).to(device)
            elif isinstance(logits_with_r, torch.Tensor):
                logits_with_r = logits_with_r.to(device)

            if isinstance(logits_without_r, np.ndarray):
                logits_without_r = torch.tensor(logits_without_r).to(device)
            elif isinstance(logits_without_r, torch.Tensor):
                logits_without_r = logits_without_r.to(device)

            if isinstance(responses_id, np.ndarray):
                responses_id = torch.tensor(responses_id).to(device)
            elif isinstance(responses_id, torch.Tensor):
                responses_id = responses_id.to(device)

            T = logits_with_r.shape[0]  # 当前回答的长度
            if T == 0:
                scores.append({"overall": 0.0, "cre_loss": 0.0, "format_reward": 0.0})
                continue

            # 确保 responses_id 长度与 T 匹配
            if responses_id.shape[0] != T:
                raise ValueError(
                    f"Shape mismatch: logits T={T} but responses_id T={responses_id.shape[0]}"
                )

            # 1. 计算上下文依赖 d_i
            d = calculate_context_dependency(logits_with_r, logits_without_r, epsilon)

            # (T,) -> (T, hidden_size)
            with torch.no_grad():
                token_embeddings = embeddings_layer(responses_id)

            # 2. 计算信息流系数 f_i
            f = calculate_info_flow(token_embeddings)

            # 3. 计算权重 alpha_i
            # 确保 d 和 f 都是 (T,)
            alpha = (1.0 / T) * (1 + beta * d + gamma * f)
            alpha_prime = alpha / (torch.sum(alpha) + epsilon)

            # 4. 获取生成词元的log概率
            log_probs = F.log_softmax(logits_with_r, dim=-1)
            # (T,)
            generated_log_probs = log_probs[range(T), responses_id]

            # 5. 计算 CRE
            cre_loss = -torch.sum(alpha_prime * generated_log_probs)

            final_reward = -cre_loss.item()

            scores.append(
                {
                    "overall": final_reward,
                    "cre_loss": cre_loss.item(),
                }
            )

        except Exception as e:
            print(f"Error calculating score for item: {e}")
            scores.append({"overall": 0.0, "cre_loss": 0.0})

    return scores
