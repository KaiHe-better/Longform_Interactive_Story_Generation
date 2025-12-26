import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

def set_gpu_device(gpu_id=0):
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        if num_devices == 0:
            print("CUDA available but no devices detected, falling back to CPU")
            return "cpu"

        if not isinstance(gpu_id, int):
            raise TypeError(f"gpu_id must be int, got {type(gpu_id)}")

        if gpu_id < 0 or gpu_id >= num_devices:
            raise ValueError(f"Requested gpu_id {gpu_id} is out of range. Available device indices: 0-{num_devices - 1}.")

        torch.cuda.set_device(gpu_id)
        device_str = f"cuda:{gpu_id}"
        print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        return device_str
    else:
        print("CUDA not available, using CPU")
        return "cpu"

def _resolve_device_map(device: str):
    if device == "auto":
        return "auto"
    if isinstance(device, str) and device.startswith("cuda"):
        return {"": device}
    return {"": device}


def load_causal_lm(model_id: str, device: str = "auto"):
    print(f"Loading model from {model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if device == "auto":
        device_map = "auto"
    elif isinstance(device, str) and device.startswith("cuda"):
        device_map = {"": device}
    else:
        device_map = {"": device}

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True
    )

    print(f"Model loaded to device: {next(model.parameters()).device}")
    return model, tokenizer


def load_model_and_tokenizer(generation_model, reward_model, device= "auto"):
    generation_model_obj, generation_tokenizer = load_causal_lm(generation_model, device=device)
    reward_model_obj, reward_tokenizer = load_causal_lm(reward_model, device=device)

    print("Model and tokenizer loaded successfully!")
    return generation_model_obj, reward_model_obj, generation_tokenizer, reward_tokenizer


def _sample_key_parts(sample: dict, fallback_idx: int) -> Tuple[str, Optional[int], int, Optional[int]]:
    extra_info = sample.get("extra_info") or {}
    split = str(extra_info.get("split", ""))
    story_id = extra_info.get("story_id")
    index = extra_info.get("index", fallback_idx)
    turn_index = extra_info.get("turn_index")
    return split, story_id, index, turn_index


def _serialize_sample_key(parts: Tuple[str, Optional[int], int, Optional[int]]) -> str:
    split, story_id, index, turn_index = parts
    return "|".join(
        "" if value is None else str(value)
        for value in (split, story_id, index, turn_index)
    )


def _normalize_role(role: Optional[str]) -> str:
    if not role:
        return "user"
    role = role.lower()
    if role in {"system", "user", "assistant"}:
        return role
    if role in {"human", "question"}:
        return "user"
    if role in {"bot", "answer"}:
        return "assistant"
    return "user"


def _extract_chat_messages(sample: dict) -> Tuple[List[Dict[str, str]], str, str]:
    """Build chat messages compatible with tokenizer chat template.

    Returns:
        messages: Full dialogue history as a list of {"role", "content"} dicts
        system_prompt: First system-level instruction if available
        user_input: Latest user utterance that model should respond to
    """

    messages: List[Dict[str, str]] = []

    def append_message(role: str, content: Optional[str]):
        if content is None:
            content = ""
        messages.append({
            "role": _normalize_role(role),
            "content": str(content)
        })

    prompt = sample.get("prompt")

    if isinstance(prompt, (list, tuple)):
        for idx, item in enumerate(prompt):
            if isinstance(item, dict):
                role = item.get("role") or item.get("from") or (
                    "system" if idx == 0 else "user"
                )
                content = item.get("content") or item.get("value") or item.get("text") or item.get("message")
                append_message(role, content)
            elif isinstance(item, str):
                role = "system" if idx == 0 else ("user" if messages and messages[-1]["role"] != "user" else "assistant")
                append_message(role, item)
            else:
                append_message("user", item)
    elif isinstance(prompt, dict):
        for key in ("system", "system_prompt", "user", "assistant"):
            if key in prompt:
                append_message(key, prompt.get(key))
    elif isinstance(prompt, str):
        append_message("system", prompt)

    if not messages:
        for alt_key in ("conversation", "conversation_history", "messages", "dialogue", "history", "system"):
            if alt_key not in sample:
                continue
            raw = sample[alt_key]
            if isinstance(raw, str):
                append_message("system", raw)
            elif isinstance(raw, dict):
                append_message(raw.get("role") or raw.get("from") or "system", raw.get("content") or raw.get("value") or raw.get("text"))
            elif isinstance(raw, (list, tuple)):
                for idx, entry in enumerate(raw):
                    if isinstance(entry, dict):
                        append_message(entry.get("role") or entry.get("from"), entry.get("content") or entry.get("value") or entry.get("text"))
                    elif isinstance(entry, str):
                        role = "system" if idx == 0 else "user"
                        append_message(role, entry)
            if messages:
                break

    if not messages:
        system_prompt = str(sample.get("system_prompt", ""))
        user_input = str(sample.get("user_input", ""))
        if not system_prompt and not user_input:
            raise KeyError("Unable to extract chat messages from sample; expected 'prompt' or chat-style fields")
        if system_prompt:
            append_message("system", system_prompt)
        if user_input:
            append_message("user", user_input)

    # Ensure a leading system prompt if available
    if all(message["role"] != "system" for message in messages):
        system_prompt_fallback = sample.get("system_prompt")
        if system_prompt_fallback is not None:
            messages.insert(0, {"role": "system", "content": str(system_prompt_fallback)})

    # Ensure the latest message is from the user to trigger generation
    if messages and messages[-1]["role"] != "user":
        last_user = sample.get("user_input")
        if last_user is not None:
            append_message("user", last_user)

    system_prompt = next((m["content"] for m in messages if m["role"] == "system"), messages[0]["content"] if messages else "")
    user_input = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

    if not user_input:
        raise ValueError("No user input found in conversation; cannot generate continuation")

    return messages, system_prompt, user_input

def load_test_data(test_source: str, test_num: int, split: Optional[str] = "train"):
    path = Path(test_source)

    if path.exists():
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            for key in ("data", "items", "records"):
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
        if test_num != -1:
            data = data[:test_num]
        print(f"Loaded {len(data)} test samples from local file")
        return data

    # Treat as Hugging Face datasets identifier
    dataset = load_dataset(test_source, split=split)
    if test_num != -1:
        dataset = dataset.select(range(min(test_num, len(dataset))))
    data = dataset.to_list()
    print(f"Loaded {len(data)} test samples from Hugging Face dataset '{test_source}' (split='{split}')")
    return data

def generate_response(model, tokenizer, system_prompt, user_input, max_length=1024):
    """Convenience wrapper that delegates to batched generation for a single example."""
    messages = [
        {"role": "system", "content": str(system_prompt)},
        {"role": "user", "content": str(user_input)}
    ]
    responses,chat_texts = generate_responses_batch(
        model,
        tokenizer,
        [messages],
        max_length=max_length
    )
    return responses[0]


def generate_responses_batch(
    model,
    tokenizer,
    messages_batch: List[List[Dict[str, str]]],
    max_length: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.8,
):
    """Generate model responses for a batch of chat message histories."""
    chat_texts: List[str] = []
    for messages in messages_batch:
        if not messages:
            raise ValueError("Each messages batch entry must contain at least one message")
        chat_texts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        )

    model_inputs = tokenizer(
        chat_texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    prompt_length = model_inputs.input_ids.shape[1]

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )

    outputs: List[str] = []
    for idx, output_ids in enumerate(generated_ids):
        new_tokens = output_ids[prompt_length:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).replace("<think>\n\n</think>\n\n", "")
        outputs.append(decoded)

    return outputs, chat_texts

def generate_reward(model, tokenizer, system_prompt, max_length=1024):
    rewards = generate_rewards_batch(
        model,
        tokenizer,
        [system_prompt],
        max_length=max_length
    )
    return rewards[0]


def generate_rewards_batch(
    model,
    tokenizer,
    system_prompts: Iterable[str],
    max_length: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.8,
):
    """Batch generate reward model outputs for provided system prompts."""
    system_prompts = list(system_prompts)
    if not system_prompts:
        return []

    messages_batch = [
        [{"role": "system", "content": str(system_prompt)}]
        for system_prompt in system_prompts
    ]

    chat_texts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        for messages in messages_batch
    ]

    model_inputs = tokenizer(
        chat_texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    prompt_length = model_inputs.input_ids.shape[1]

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )

    outputs: List[str] = []
    for idx, output_ids in enumerate(generated_ids):
        new_tokens = output_ids[prompt_length:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
        match = re.search(r"\{[\s\S]*\}", decoded)
        if match:
            json_str = match.group(0)
            outputs.append(json_str)  # 转为合法 JSON 对象
        else:
            try:
                outputs.append(str(match))
            except:
                outputs.append("Exception")

    return outputs

def predict_on_test_data(model, tokenizer, test_data, output_path, batch_size: int = 2):
    results = []

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_samples = len(test_data)
    batch_size = max(1, int(batch_size))

    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        current_batch = test_data[start_idx:end_idx]
        print(f"Generating samples {start_idx + 1}-{end_idx}/{total_samples}")

        system_prompts: List[str] = []
        user_inputs: List[str] = []
        messages_batch: List[List[Dict[str, str]]] = []
        sample_keys: List[str] = []
        key_parts_list = []
        extra_infos: List[dict] = []

        for offset, sample in enumerate(current_batch):
            index = start_idx + offset
            messages, system_prompt, user_input = _extract_chat_messages(sample)

            messages_batch.append(messages)
            system_prompts.append(system_prompt)
            user_inputs.append(user_input)

            key_parts = _sample_key_parts(sample, index)
            key_parts_list.append(key_parts)
            sample_keys.append(_serialize_sample_key(key_parts))
            extra_infos.append(sample.get("extra_info") or {})

        batch_responses, chat_texts = generate_responses_batch(
            model,
            tokenizer,
            messages_batch
        )

        for response, key_parts, sample_key, extra_info, chat_text, user_input in zip(
            batch_responses,
            key_parts_list,
            sample_keys,
            extra_infos,
            chat_texts,
            user_inputs,
        ):
            result = {
                "sample_id": key_parts[2],
                "system": chat_text,
                "user_input": user_input,
                "predicted_response": response,
                "sample_key": sample_key,
                "extra_info": extra_info,
            }
            results.append(result)
    
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {output_path}")
    return results

def score_on_test_data(
    reward_model,
    reward_tokenizer,
    test_data,
    score_output_path,
    pred_output_path,
    batch_size: int = 1,
):
    score_output_path = Path(score_output_path)
    pred_output_path = Path(pred_output_path)

    if not pred_output_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_output_path}")

    with pred_output_path.open('r', encoding='utf-8') as f:
        predictions_raw = json.load(f)

    if isinstance(predictions_raw, dict):
        if "results" in predictions_raw and isinstance(predictions_raw["results"], list):
            predictions_list = predictions_raw["results"]
        else:
            predictions_list = list(predictions_raw.values())
    elif isinstance(predictions_raw, list):
        predictions_list = predictions_raw
    else:
        raise TypeError(f"Unsupported predictions format in {pred_output_path}: {type(predictions_raw)}")

    if not predictions_list:
        raise ValueError(f"No predictions loaded from {pred_output_path}")

    predictions_by_key = {}
    predictions_by_index = {}
    predictions_by_id = {}

    for idx, item in enumerate(predictions_list):
        predictions_by_index[idx] = item
        sample_id = item.get("sample_id")
        sample_key = item.get("sample_key")
        if sample_id is not None:
            predictions_by_id[sample_id] = item
        if sample_key:
            predictions_by_key[sample_key] = item

    results = []
    score_output_path.parent.mkdir(parents=True, exist_ok=True)

    total_samples = len(test_data)
    batch_size = max(1, int(batch_size))

    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        current_batch = test_data[start_idx:end_idx]
        print(f"Scoring samples {start_idx + 1}-{end_idx}/{total_samples}")

        eval_prompts: List[str] = []
        meta_info: List[dict] = []

        for offset, sample in enumerate(current_batch):
            index = start_idx + offset
            key_parts = _sample_key_parts(sample, index)
            sample_key = _serialize_sample_key(key_parts)

            prediction_entry = predictions_by_key.get(sample_key)
            if prediction_entry is None:
                prediction_entry = predictions_by_id.get(key_parts[2])
            if prediction_entry is None:
                prediction_entry = predictions_by_index.get(index)
            if prediction_entry is None:
                raise KeyError(
                    f"Prediction for sample_key '{sample_key}' (fallback index {index}) not found in {pred_output_path}. "
                    "Ensure the prediction file aligns with the test dataset order or regenerate predictions."
                )

            messages, system_prompt, user_input = _extract_chat_messages(sample)
            predicted_response = prediction_entry.get("predicted_response", "")

            eval_prompt_template = sample.get('judge_system_prompt', "")
            eval_prompt = eval_prompt_template.replace("{user_input}", str(user_input)).replace("{generated_story}", str(predicted_response))

            eval_prompts.append(eval_prompt)
            meta_info.append({
                "sample_id": index,
                "system": system_prompt,
                "user_input": user_input,
                "predicted_response": predicted_response,
            })

        batch_scores = generate_rewards_batch(
            reward_model,
            reward_tokenizer,
            eval_prompts
        )

        for scores, info in zip(batch_scores, meta_info):
            result = {
                **info,
                "reward_scores": scores,
            }
            results.append(result)

    with score_output_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {score_output_path}")
    return results

def main():
    gpu_id = 7
    device = set_gpu_device(gpu_id)
    # generation_flag = True
    generation_flag = False
    test_num = -1
    generation_batch_size = 24
    reward_batch_size = 24

    generation_model_id = "HeAAAAA/story_generation_Qwen3_8B_SFT"
    reward_model_list = [
        # "HeAAAAA/story_reward_Qwen3_8B_v1",
        "HeAAAAA/story_reward_Llama3.1_8B_expneg",
        "HeAAAAA/story_reward_Qwen3_8B_expneg",
    ]

    test_source = "HeAAAAA/story_generation_RL"
    test_split = "test"
    generation_model_slug = generation_model_id.split("/")[-1]
    pred_output_path = Path(
        f"/raid/hpc/hekai/WorkShop/My_project/ai_story_data_generation/res/pred/eval_res_{generation_model_slug}.json"
    )

    test_data = load_test_data(test_source, test_num, split=test_split)
    total_samples = len(test_data)

    start_time = time.time()

    if generation_flag:
        generation_model, generation_tokenizer = load_causal_lm(generation_model_id, device=device)
        predict_on_test_data(generation_model, generation_tokenizer, test_data, pred_output_path, batch_size = generation_batch_size)

        # Free generation model memory if using GPU
        if torch.cuda.is_available():
            del generation_model
            torch.cuda.empty_cache()
    else:
        if not pred_output_path.exists():
            raise FileNotFoundError(f"Prediction file not found at {pred_output_path}. Set generation_flag=True to generate it first.")

    for reward_model_id in reward_model_list:
        reward_model_slug = reward_model_id.split("/")[-1]
        score_output_path = Path(f"/raid/hpc/hekai/WorkShop/My_project/ai_story_data_generation/res/scores/eval_res_{generation_model_slug}_{reward_model_slug}.json")

        reward_model, reward_tokenizer = load_causal_lm(reward_model_id, device=device)
        score_on_test_data(
            reward_model,
            reward_tokenizer,
            test_data,
            score_output_path,
            pred_output_path,
            batch_size=reward_batch_size,
        )

        if torch.cuda.is_available():
            del reward_model
            torch.cuda.empty_cache()

    end_time = time.time()

    print("Evaluation completed!")
    print(f"Total samples evaluated: {total_samples}")
    print(f"Reward models evaluated: {len(reward_model_list)}")
    elapsed = end_time - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    print(f"Time taken: {hours} hours {minutes} minutes")


if __name__ == "__main__":
    main()
