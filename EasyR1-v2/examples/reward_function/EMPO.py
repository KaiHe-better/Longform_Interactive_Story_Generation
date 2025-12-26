import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

__all__ = ["compute_score"]


def _normalize_text(s: str) -> str:
    s = str(s or "")
    s = re.sub(r"\s+", " ", s.strip())
    return s if s else "<EMPTY>"


def _group_key(ri: Dict[str, Any]) -> str:
    if "group_id" in ri and ri["group_id"] is not None:
        return str(ri["group_id"])
    return "INP::" + _normalize_text(ri.get("input", ""))[:256]


def _normalized_entropy(counts: Counter, n: int) -> float:
    if n <= 1 or len(counts) <= 1:
        return 0.0
    import numpy as np
    ps = np.array([c / n for c in counts.values() if c > 0], dtype=float)
    if ps.size == 0:
        return 0.0
    H = float(-(ps * np.log(np.maximum(ps, 1e-12))).sum())
    Hmax = float(np.log(len(ps)))
    return H / max(Hmax, 1e-12)


def _compute_embeddings(texts: List[str], model, batch_size: int):
    import numpy as np
    embs = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    return embs / norms


def _cluster_by_embeddings(
    context: str,
    replies: List[str],
    model,
    *,
    batch_size: int,
    sim_threshold: float,
    max_input_chars: int,
    max_reply_chars: int,
) -> List[int]:
    import numpy as np

    def _clip(s: str, lim: int) -> str:
        s = (s or "").strip()
        return (s[:lim] + " ...") if len(s) > lim else s

    texts = [
        "Input:\n" + _clip(context, max_input_chars) +
        "\n\nReply:\n" + _clip(r, max_reply_chars)
        for r in replies
    ]

    embs = _compute_embeddings(texts, model, batch_size=batch_size)

    labels: List[int] = [-1] * len(replies)
    rep_vecs: List[Any] = []

    for i in range(len(replies)):
        v = embs[i]
        placed = False
        if rep_vecs:
            sims = np.dot(np.stack(rep_vecs, axis=0), v)
            best_cid = int(np.argmax(sims))
            if float(sims[best_cid]) >= sim_threshold:
                labels[i] = best_cid
                placed = True
        if not placed:
            labels[i] = len(rep_vecs)
            rep_vecs.append(v)

    return labels


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
        if data and all(isinstance(x, dict) for x in data) and all(
            any(isinstance(v, list) for v in x.values()) for x in data
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


def compute_score(
    reward_inputs: List[Dict[str, Any]],
    *,
    tau_low: float = 0.05,
    tau_high: float = 0.75,
    min_samples_for_entropy: int = 4,
    sim_threshold: float = 0.83,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[str] = None,
    batch_size: int = 64,
    max_input_chars: int = 2000,
    max_reply_chars: int = 1000,
    **_unused,
) -> List[Dict[str, float]]:

    items = _as_records(reward_inputs)

    norm_items: List[Dict[str, Any]] = []
    for ri in items:
        inp = _first_nonempty(
            ri,
            ["input", "prompt", "question", "instruction", "query", "source", "context"],
        )
        resp = _first_nonempty(
            ri,
            ["response", "output", "completion", "prediction", "model_output", "text", "reply"],
        )
        norm_items.append(
            {
                "input": _normalize_text(inp),
                "response": _normalize_text(resp),
                "group_id": ri.get("group_id"),
            }
        )

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError("sentence-transformers is required.") from e

    cache_key = (model_name, device or "auto")
    cache = getattr(compute_score, "_embedder_cache", None)
    if cache is None or not isinstance(cache, dict):
        cache = {}
        setattr(compute_score, "_embedder_cache", cache)
    if cache_key not in cache:
        cache[cache_key] = SentenceTransformer(model_name, device=device)
    model = cache[cache_key]

    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, ri in enumerate(norm_items):
        groups[_group_key(ri)].append(idx)

    inputs_norm = [ri["input"] for ri in norm_items]
    responses = [ri["response"] for ri in norm_items]

    out: List[Dict[str, float]] = [dict() for _ in norm_items]

    for _g, idxs in groups.items():
        n = len(idxs)
        if n == 0:
            continue

        context = inputs_norm[idxs[0]]
        replies_g = [responses[i] for i in idxs]

        labels = _cluster_by_embeddings(
            context=context,
            replies=replies_g,
            model=model,
            batch_size=batch_size,
            sim_threshold=sim_threshold,
            max_input_chars=max_input_chars,
            max_reply_chars=max_reply_chars,
        )

        counts = Counter(labels)
        H_norm = _normalized_entropy(counts, n)
        agreement = 1.0 - H_norm

        gated = False
        if n >= max(1, int(min_samples_for_entropy)):
            if H_norm < float(tau_low) or H_norm > float(tau_high):
                gated = True

        if gated:
            for i in idxs:
                out[i] = {
                    "overall": 0.0,
                    "entropy": float(agreement),
                    "consistency": 0.0,
                }
            continue

        for j, i in enumerate(idxs):
            frac = counts[labels[j]] / float(n) if n > 0 else 0.0
            frac = float(max(0.0, min(1.0, frac)))
            out[i] = {
                "overall": frac,
                "entropy": float(agreement),
                "consistency": frac,
            }

    return out
