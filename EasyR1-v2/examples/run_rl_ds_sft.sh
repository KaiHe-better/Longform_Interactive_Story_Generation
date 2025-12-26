# bash examples/deepseek_sft_story_grpo_origin_len.sh
# python scripts/model_merger.py --local_dir /data/kai_he/above20/Story_RL_origin_len_deepseek/global_step_68/actor
# python scripts/model_merger.py --local_dir /data/kai_he/above20/Story_RL_origin_len_deepseek_sft/global_step_68/actor
python scripts/model_merger.py --local_dir /data/kai_he/above20/Story_RL_origin_len_llama/global_step_68/actor
python scripts/model_merger.py --local_dir /data/kai_he/above20/Story_RL_origin_len_deepseek/global_step_68/actor
python scripts/model_merger.py --local_dir /data/kai_he/above20/Story_RL_origin_len_deepseek/global_step_40/actor
python scripts/evaluation_predict_score_vllm.py