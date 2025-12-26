#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1,3
# export CUDA_VISIBLE_DEVICES=0,1
export RAY_allowed_local_fs_capacity_threshold=0.99
export RAY_local_fs_capacity_threshold=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# MODEL_PATH="/data01/sdz/models/Qwen3-1.7B/snapshots/0060bc56d46589041c1048efd1a397421b1142b5"
# MODEL_PATH=Qwen/Qwen3-8B
MODEL_PATH=deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
# MODEL_PATH=/data/kai_he/models/Deepseek-SFT
# MODEL_PATH=HeAAAAA/story_generation_Qwen3_8B_SFT

python -m verl.trainer.main \
    config=examples/deepseek_story_config_origin_len.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    