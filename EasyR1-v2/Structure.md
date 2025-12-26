# 简单记录一下项目的结构
基于verl+vllm+lora


## 数据集
给的原始的数据集都是json。但是verl需要parquet格式，其实差不多。

假如原始json是：
```json
{
  "system": "你是一个故事生成AI...",
  "conversations": [
    {"from": "user", "value": "用户动作1"},
    {"from": "assistant", "value": "助手回复1"},
    {"from": "user", "value": "用户动作2"}, 
    {"from": "assistant", "value": "助手回复2"},
    {"from": "user", "value": "用户动作3"},
    {"from": "assistant", "value": "助手回复3"},
    {"from": "user", "value": "用户动作4"},
    {"from": "assistant", "value": "助手回复4"}
  ]
}
```

脚本将它转换成4个独立的训练样本, 转换后的parquet格式差不多是这样：
```json
{
  "prompt": [
    {"role": "system", "content": "你是一个故事生成AI..."},
    {"role": "user", "content": "用户动作1"}
  ],
  "target_response": "助手回复1"
}

{
  "prompt": [
    {"role": "system", "content": "你是一个故事生成AI..."},
    {"role": "user", "content": "用户动作1"},
    {"role": "assistant", "content": "助手回复1"},
    {"role": "user", "content": "用户动作2"}
  ],
  "target_response": "助手回复2"
}

{
  "prompt": [
    {"role": "system", "content": "你是一个故事生成AI..."},
    {"role": "user", "content": "用户动作1"},
    {"role": "assistant", "content": "助手回复1"},
    {"role": "user", "content": "用户动作2"},
    {"role": "assistant", "content": "助手回复2"},
    {"role": "user", "content": "用户动作3"}
  ],
  "target_response": "助手回复3"
}

{
  "prompt": [
    {"role": "system", "content": "你是一个故事生成AI..."},
    {"role": "user", "content": "用户动作1"},
    {"role": "assistant", "content": "助手回复1"},
    {"role": "user", "content": "用户动作2"},
    {"role": "assistant", "content": "助手回复2"},
    {"role": "user", "content": "用户动作3"},
    {"role": "assistant", "content": "助手回复3"},
    {"role": "user", "content": "用户动作4"}
  ],
  "target_response": "助手回复4"
}
```

突然想到一个问题。**我们设置了target_response，作为最后回答的标准答案。但是我们的奖励函数并不包括这部分**。
如果我们要根据我们设计的奖励函数计算分数，我们要学习到的信息肯定不仅仅局限于上下文。不应该学习最后如何匹配标准答案，而是要比标准答案要好。

数据集有这些数据，可以直接打印其中一个数据的结构，`python EasyR1/scripts/print_parquet.py`
>(Pdb) first_train_data.keys()
dict_keys(['data_source', 'prompt', 'ability', 'reward_model', 'target_response', 'extra_info', 'system_prompt', 'conversation_history', 'story_content', 'user_input', 'output_format'])


```python
system_prompt = instruction+ story_content + output_format
prompt = system_prompt + conversation_history + user_input
```

## 如何把数据集塞进训练中

主要代码都在这里 EasyR1/verl/utils/dataset.py: Line406
把 _getitem_ 函数重写了

## 奖励函数
这就是multi turn story generation的奖励函数的代码。
```
/root/rl-llm/EasyR1/examples/reward_function/story_generation.py
```
其中和math.py也就是数学推理不一样的地方就是，math是有一个ground truth来评判回答正不正确的。但是，我们的奖励函数仅仅依靠模型运行状态中的中间过程。当然，这个需要进一步优化。

同样的，我们也有一个答案的格式奖励format reward。
TODO：
现在还没实现format reward。

需要添加一个格式文件jinja。
```
/root/rl-llm/EasyR1/examples/format_prompt/story_generation.jinja
```

当然，为了实现这个奖励函数，我们需要获取模型推理的中间结果并且保存下来。
这些输出的策略也就是rollout的策略，都写在`/root/rl-llm/EasyR1/verl/workers/rollout/vllm_rollout_spmd.py: Line173`中。
关键的代码就是在这里，第二版进行修改过的这个函数，只生成一遍 with_r 的 response。
```python
class vLLMRollout(BaseRollout):
    # ...
    def generate_sequences_r(self, prompts: DataProto) -> DataProto:
    # ...
```

**非常非常关键，**
实际上，经过测试，发现如果试图在生成时就返回所有的 logprob 开销极大。
因此需要在 compute_logprob ，也即是 dp_actor.py 中进行 micro_batch_forward 时，获取这些 logits。顺带直接能把 entropy 和 info_flow 全都搞定了。
之后，直接传进 reward_func 中就结束了。
compute_reward 还有一个前置函数。worker.reward.BatchFunctionRewardManager.compute_reward。这里也针对我们的任务做了修改。

## 怎么运行？
相关实验的变量都写在这个yaml中。

```
/root/rl-llm/EasyR1/examples/story_config.yaml
```

这个设置文件非常重要。另外，rollout策略的config地址为`/root/rl-llm/EasyR1/verl/workers/rollout/config.py`.
```sh
bash examples/qwen3_1.7b_story_grpo.sh
```

## 代码框架
verl文件夹下就是主要的类。
训练的步骤在`verl/trainer/ray_trainer.py`中。(Class RayPPOTrainer.fit中)


## 运行流程
1. data 已经处理好 with_r 和 without_r 的 prompt
2. 丢进去合成一个 batch 一起输出结果
3. 从结果提取出各自的 logits
4. 在 compute_score 那里，再通过 embedding model 获得其 embedding
5. 综合一下 Logits 和 embedding 得到最后的总奖励


## 注意注意
EasyR1/examples/story_config.yaml 里面我暂时把所有的 batch 大小都设置成 1 了。
而且禁用了 KL 散度。

> 运行报错可能要修改 python 某个包里面最大的返回 Logprob 的值。但我忘了是哪个了。。。。
这是以前版本的。现在不需要了。