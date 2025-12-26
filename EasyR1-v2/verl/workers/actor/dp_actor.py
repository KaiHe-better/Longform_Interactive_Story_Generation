# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implement Actor
"""

import os
from collections import defaultdict
from typing import Any, Optional

import torch
import torch.distributed as dist
from einops import rearrange
from ray.experimental.tqdm_ray import tqdm
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ...protocol import DataProto, batch_collate
from ...trainer.core_algos import average_loss, compute_kl, compute_policy_loss
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from .base import BasePPOActor
from .config import ActorConfig


try:
    from flash_attn.bert_padding import (
        index_first_axis,
        pad_input,
        rearrange,
        unpad_input,
    )
except ImportError:
    pass


__all__ = ["DataParallelPPOActor"]


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        if config.use_torch_compile:
            self.log_probs_from_logits = torch.compile(
                VF.log_probs_from_logits, dynamic=True
            )
        else:
            self.log_probs_from_logits = VF.log_probs_from_logits
        # 新增一个判断是否是 story
        self.is_story = config.is_story

    def _forward_micro_batch(
        self, micro_batch: dict[str, torch.Tensor], temperature: float
    ) -> torch.Tensor:
        """
        Returns:
            log_probs: # (bs, response_len)
        """
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(
                0, 1
            )  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

        multi_modal_inputs = defaultdict(list)
        if "multi_modal_inputs" in micro_batch:
            multi_modal_inputs = batch_collate(micro_batch["multi_modal_inputs"])
            multi_modal_inputs = {
                key: torch.cat(value, dim=0)
                for key, value in multi_modal_inputs.items()
            }
        else:
            multi_modal_inputs = {}

        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(
                input_ids.unsqueeze(-1), attention_mask
            )  # (total_nnz, 1)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(
                        rearrange(position_ids, "c b s ... -> (b s) c ..."), indices
                    )
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                    indices,
                ).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(
                input_ids_rmpad, shifts=-1, dims=1
            )  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.config.ulysses_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = (
                    ulysses_pad_and_slice_inputs(
                        input_ids_rmpad,
                        position_ids_rmpad,
                        sp_size=self.config.ulysses_size,
                    )
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(
                0
            )  # ((total_nnz / sp) + pad)

            # only pass input_ids and position_ids to enable flash_attn_varlen
            output = self.actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
            )  # prevent model thinks we are generating

            logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
            logits_rmpad.div_(temperature)
            # ((total_nnz / sp) + pad)
            log_probs = self.log_probs_from_logits(
                logits=logits_rmpad, labels=input_ids_rmpad_rolled
            )

            # gather log_prob if sp > 1
            if self.config.ulysses_size > 1:
                # gather and unpad for the ulysses sp
                log_probs = gather_outputs_and_unpad(
                    log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size
                )

            # pad back to (bsz, seqlen)
            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1),
                indices=indices,
                batch=batch_size,
                seqlen=seqlen,
            )
            log_probs = full_log_probs.squeeze(-1)[
                :, -response_length - 1 : -1
            ]  # (bsz, response_length)
        else:
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits: torch.Tensor = output.logits
            logits.div_(temperature)
            logits = logits[
                :, -response_length - 1 : -1, :
            ]  # (bsz, response_length, vocab_size)
            log_probs = self.log_probs_from_logits(
                logits, responses
            )  # (bsz, response_length)

        return log_probs

    def _forward_micro_batch_story(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        key: str = "input_ids",
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            log_probs:  (bsz, response_len)
            entropy:    (bsz, response_len)
            info_flow:  (bsz, response_len) Information Flow Coefficient
        """
        input_ids = micro_batch[key]
        batch_size, seqlen = input_ids.shape
        # print(f"key: {key}, shape:{input_ids.shape}")
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        
        if key == "input_ids_without_r":
            # 这里拼接 input_ids_wo_r 和 reponse，来实现 force_decoding
            attention_mask = micro_batch["attention_mask_without_r"]
            position_ids = micro_batch["position_ids_without_r"]
        else:
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]

        if key == "input_ids_without_r":
            # print(f"input_ids shape: {input_ids.shape}")
            # print(f"attention_mask_without_r shape: {attention_mask.shape}")
            # print(f"responses shape: {responses.shape}")

            assert input_ids.shape[1] == attention_mask.shape[1], \
                f"Length mismatch: input_ids {input_ids.shape[1]} vs attention_mask {attention_mask.shape[1]}"

        # --- Helper Function to compute the Metric ---
        def compute_information_flow_coefficient(
            hidden_states: torch.Tensor,
        ) -> torch.Tensor:
            """
            Calculates f_i = max(0, 1/i * sum_{j=1}^{i-1} sim(y_j, y_i))
            Args:
                hidden_states: (bsz, seqlen, hidden_dim)
            Returns:
                flow_coeff: (bsz, seqlen)
            """
            # 1. Normalize embeddings to compute Cosine Similarity via Dot Product
            # (bsz, seqlen, dim)
            norm_hidden = torch.nn.functional.normalize(hidden_states, p=2, dim=-1)

            # 2. Compute Similarity Matrix: (bsz, seqlen, seqlen)
            # sim_matrix[b, i, j] = cos(h[b,i], h[b,j])
            sim_matrix = torch.bmm(norm_hidden, norm_hidden.transpose(1, 2))

            # 3. Mask to keep only strictly past tokens (j < i)
            # sum_{j=1}^{i-1} implies we exclude the diagonal (j=i) and future
            seq_len = hidden_states.size(1)
            mask = torch.tril(
                torch.ones(
                    seq_len, seq_len, device=hidden_states.device, dtype=torch.bool
                ),
                diagonal=-1,
            )

            # Apply mask (set non-past correlations to 0)
            sim_matrix = sim_matrix.masked_fill(~mask, 0.0)

            # 4. Sum over j (dim=2)
            sum_sim = sim_matrix.sum(dim=2)  # (bsz, seqlen)

            # 5. Divide by i (the count of previous tokens)
            # indices: [0, 1, 2, ..., seqlen-1]
            indices = torch.arange(
                seq_len, device=hidden_states.device, dtype=hidden_states.dtype
            ).unsqueeze(0)
            denominator = indices.clamp(
                min=1.0
            )  # Avoid division by zero for the first token

            # 6. Compute f_i
            flow_coeff = sum_sim / denominator
            flow_coeff = torch.clamp(flow_coeff, min=0.0)  # max(0, ...)

            # For the very first token (i=0), the sum is 0, result is 0. Correct.
            return flow_coeff

        # ---------------------------------------------

        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)

        multi_modal_inputs = defaultdict(list)
        if "multi_modal_inputs" in micro_batch:
            multi_modal_inputs = batch_collate(micro_batch["multi_modal_inputs"])
            multi_modal_inputs = {
                key: torch.cat(value, dim=0)
                for key, value in multi_modal_inputs.items()
            }
        else:
            multi_modal_inputs = {}

        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(
                input_ids.unsqueeze(-1), attention_mask
            )
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(
                        rearrange(position_ids, "c b s ... -> (b s) c ..."), indices
                    )
                    .transpose(0, 1)
                    .unsqueeze(1)
                )
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                    indices,
                ).transpose(0, 1)

            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)

            if self.config.ulysses_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = (
                    ulysses_pad_and_slice_inputs(
                        input_ids_rmpad,
                        position_ids_rmpad,
                        sp_size=self.config.ulysses_size,
                    )
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)

            # --- [CHANGE] Add output_hidden_states=True ---
            output = self.actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
                output_hidden_states=True,
            )

            logits_rmpad = output.logits.squeeze(0)
            logits_rmpad.div_(temperature)

            # Entropy calculation (from previous step)
            probs_rmpad = torch.softmax(logits_rmpad, dim=-1)
            log_probs_rmpad_all = torch.log_softmax(logits_rmpad, dim=-1)
            entropy_rmpad = -torch.sum(probs_rmpad * log_probs_rmpad_all, dim=-1)

            # Log probs calculation
            log_probs = self.log_probs_from_logits(
                logits=logits_rmpad, labels=input_ids_rmpad_rolled
            )

            # --- [CHANGE START] Compute Information Flow (Padding Free Path) ---
            # 1. Get hidden states: (total_nnz, dim) or (total_nnz/sp, dim)
            # Use last_hidden_state which is the last layer embedding
            hidden_states_rmpad = output.hidden_states[-1].squeeze(0)

            # 2. Gather hidden states if Ulysses is on
            # We need the full sequence embedding to compute correlation with previous tokens
            if self.config.ulysses_size > 1:
                log_probs = gather_outputs_and_unpad(
                    log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size
                )
                entropy_rmpad = gather_outputs_and_unpad(
                    entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                )
                # Gather hidden states as well (Note: this can be memory intensive for very large batches)
                hidden_states_rmpad = gather_outputs_and_unpad(
                    hidden_states_rmpad,
                    gather_dim=0,
                    unpad_dim=0,
                    padding_size=pad_size,
                )

            hidden_dim = hidden_states_rmpad.shape[-1]

            # 确保 hidden_states_rmpad 是 (total_nnz, dim)
            if hidden_states_rmpad.dim() == 1:
                hidden_states_rmpad = hidden_states_rmpad.unsqueeze(-1)
            # 3. Pad hidden states back to (bsz, seqlen, dim) to do Matrix arithmetic
            full_hidden_states = pad_input(
                hidden_states=(
                    hidden_states_rmpad.unsqueeze(-1)
                    if hidden_states_rmpad.dim() == 2
                    else hidden_states_rmpad
                ),
                indices=indices,
                batch=batch_size,
                seqlen=seqlen,
            )

            # 调整形状为 (bsz, seqlen, hidden_dim)
            full_hidden_states = full_hidden_states.view(batch_size, seqlen, hidden_dim)

            # full_hidden_states = pad_input(
            #     hidden_states=hidden_states_rmpad.unsqueeze(
            #         1
            #     ),  # pad_input expects (total_nnz, 1, dim) roughly
            #     indices=indices,
            #     batch=batch_size,
            #     seqlen=seqlen,
            # ).squeeze(
            #     1
            # )  # -> (bsz, seqlen, dim)

            # 4. Compute the metric
            full_info_flow = compute_information_flow_coefficient(full_hidden_states)

            # 5. Slice output for response
            # We take [:, -response_length:] because we want the metric for the actual response tokens
            info_flow = full_info_flow[:, -response_length:]
            # --- [CHANGE END] ---

            # Pad back log_probs and entropy
            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1),
                indices=indices,
                batch=batch_size,
                seqlen=seqlen,
            )
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]

            full_entropy = pad_input(
                hidden_states=entropy_rmpad.unsqueeze(-1),
                indices=indices,
                batch=batch_size,
                seqlen=seqlen,
            )
            entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]

        else:
            # --- [CHANGE] Add output_hidden_states=True ---
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
                output_hidden_states=True,
            )
            logits: torch.Tensor = output.logits
            logits.div_(temperature)

            # Entropy
            logits_for_entropy = logits[:, -response_length - 1 : -1, :]
            probs = torch.softmax(logits_for_entropy, dim=-1)
            log_probs_all = torch.log_softmax(logits_for_entropy, dim=-1)
            entropy = -torch.sum(probs * log_probs_all, dim=-1)

            # Log probs
            logits_for_prob = logits[:, -response_length - 1 : -1, :]
            log_probs = self.log_probs_from_logits(logits_for_prob, responses)

            # --- [CHANGE START] Compute Information Flow (Standard Path) ---
            hidden_states = output.hidden_states[-1]  # (bsz, seqlen, dim)

            full_info_flow = compute_information_flow_coefficient(hidden_states)

            # Slice for response part
            info_flow = full_info_flow[:, -response_length:]
            # --- [CHANGE END] ---

        return log_probs, entropy, info_flow

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(
                self.actor_module.parameters(), max_norm=self.config.max_grad_norm
            )

        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
        else:
            self.actor_optimizer.step()

        self.actor_optimizer.zero_grad()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            if ``is_story``:
                return (
                    torch.Tensor: log_probs,
                    torch.Tensor: entropy,
                    torch.Tensor: info_flow,
                    torch.Tensor: log_probs_wo_r,
                    torch.Tensor: entropy_wo_r,
                    torch.Tensor: info_flow_wo_r,
                )

            else:
                return: torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        if self.is_story:
            select_keys = [
                "input_ids",
                "attention_mask",
                "position_ids",
                "responses",
                "input_ids_without_r",
                "position_ids_without_r",
                "attention_mask_without_r",
            ]
        else:
            select_keys = ["input_ids", "attention_mask", "position_ids", "responses"]
        non_tensor_select_keys = ["multi_modal_inputs"]

        data = data.select(select_keys, non_tensor_select_keys)
        if self.config.dynamic_batching:
            max_token_len = (
                self.config.micro_batch_size_per_device_for_experience
                * data.batch["input_ids"].size(-1)
            )
            micro_batches, batch_idx_list = prepare_dynamic_batch(
                data, max_token_len=max_token_len
            )
        else:
            micro_batches = data.split(
                self.config.micro_batch_size_per_device_for_experience
            )

        log_probs_lst = []
        entropy_lst = []
        info_flow_lst = []

        log_probs_wo_r_lst = []
        entropy_wo_r_lst = []
        # info_flow_wo_r_lst = []

        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=1)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            if self.is_story:
                log_probs, entropy, info_flow = self._forward_micro_batch_story(
                    model_inputs, temperature=temperature, key="input_ids"
                )
                # 实际上，info_flow_wo_r 并不在公式中，后续也不会在 reward 中用到
                log_probs_wo_r, entropy_wo_r, info_flow_wo_r = (
                    self._forward_micro_batch_story(
                        model_inputs, temperature=temperature, key="input_ids_without_r"
                    )
                )
                log_probs_lst.append(log_probs)
                entropy_lst.append(entropy)
                info_flow_lst.append(info_flow)

                log_probs_wo_r_lst.append(log_probs_wo_r)
                entropy_wo_r_lst.append(entropy_wo_r)
                # info_flow_wo_r_lst.append(info_flow_wo_r)
            else:
                log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature
                )
                log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)

        if self.is_story:
            entropy = torch.concat(entropy_lst, dim=0)
            info_flow = torch.concat(info_flow_lst, dim=0)

            log_probs_wo_r = torch.concat(log_probs_wo_r_lst, dim=0)
            entropy_wo_r = torch.concat(entropy_wo_r_lst, dim=0)
            # info_flow_wo_r = torch.concat(info_flow_wo_r_lst, dim=0)

        if self.config.dynamic_batching:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)

        if self.is_story:
            if self.config.dynamic_batching:
                entropy = restore_dynamic_batch(entropy, batch_idx_list)
                info_flow = restore_dynamic_batch(info_flow, batch_idx_list)

                log_probs_wo_r = restore_dynamic_batch(log_probs_wo_r, batch_idx_list)
                entropy_wo_r = restore_dynamic_batch(entropy_wo_r, batch_idx_list)
                # info_flow_wo_r = restore_dynamic_batch(info_flow_wo_r, batch_idx_list)

            return (
                log_probs,
                entropy,
                info_flow,
                log_probs_wo_r,
                entropy_wo_r,
                # info_flow_wo_r,
            )

        return log_probs

    @torch.no_grad()
    def compute_log_prob_v2(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            if ``is_story``:
                return (
                    torch.Tensor: log_probs,
                    torch.Tensor: entropy,
                    torch.Tensor: info_flow,
                    torch.Tensor: log_probs_wo_r,
                    torch.Tensor: entropy_wo_r,
                    torch.Tensor: info_flow_wo_r,
                )

            else:
                return: torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        if self.is_story:
            select_keys = [
                "input_ids",
                "attention_mask",
                "position_ids",
                "responses",
                "input_ids_without_r",
                "position_ids_without_r",
                "attention_mask_without_r",
            ]
        else:
            select_keys = ["input_ids", "attention_mask", "position_ids", "responses"]
        non_tensor_select_keys = ["multi_modal_inputs"]

        data = data.select(select_keys, non_tensor_select_keys)
        if self.config.dynamic_batching:
            max_token_len = (
                self.config.micro_batch_size_per_device_for_experience
                * data.batch["input_ids"].size(-1)
            )
            micro_batches, batch_idx_list = prepare_dynamic_batch(
                data, max_token_len=max_token_len
            )
        else:
            micro_batches = data.split(
                self.config.micro_batch_size_per_device_for_experience
            )

        log_probs_lst = []
        entropy_lst = []
        info_flow_lst = []

        log_probs_wo_r_lst = []
        entropy_wo_r_lst = []
        # info_flow_wo_r_lst = []

        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=1)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            if self.is_story:
                log_probs, entropy, info_flow = self._forward_micro_batch_story(
                    model_inputs, temperature=temperature, key="input_ids"
                )
                # 实际上，info_flow_wo_r 并不在公式中，后续也不会在 reward 中用到
                log_probs_wo_r, entropy_wo_r, info_flow_wo_r = (
                    self._forward_micro_batch_story(
                        model_inputs, temperature=temperature, key="input_ids_without_r"
                    )
                )
                log_probs_lst.append(log_probs)
                entropy_lst.append(entropy)
                info_flow_lst.append(info_flow)

                log_probs_wo_r_lst.append(log_probs_wo_r)
                entropy_wo_r_lst.append(entropy_wo_r)
                # info_flow_wo_r_lst.append(info_flow_wo_r)
            else:
                log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature
                )
                log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)

        if self.is_story:
            entropy = torch.concat(entropy_lst, dim=0)
            info_flow = torch.concat(info_flow_lst, dim=0)

            log_probs_wo_r = torch.concat(log_probs_wo_r_lst, dim=0)
            entropy_wo_r = torch.concat(entropy_wo_r_lst, dim=0)
            # info_flow_wo_r = torch.concat(info_flow_wo_r_lst, dim=0)

        if self.config.dynamic_batching:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)

        if self.is_story:
            if self.config.dynamic_batching:
                entropy = restore_dynamic_batch(entropy, batch_idx_list)
                info_flow = restore_dynamic_batch(info_flow, batch_idx_list)

                log_probs_wo_r = restore_dynamic_batch(log_probs_wo_r, batch_idx_list)
                entropy_wo_r = restore_dynamic_batch(entropy_wo_r, batch_idx_list)
                # info_flow_wo_r = restore_dynamic_batch(info_flow_wo_r, batch_idx_list)

            return (
                log_probs,
                entropy,
                info_flow,
                log_probs_wo_r,
                entropy_wo_r,
                # info_flow_wo_r,
            )

        return log_probs
    
    def update_policy(self, data: DataProto) -> dict[str, Any]:
        self.actor_module.train()

        temperature = data.meta_info[
            "temperature"
        ]  # temperature must be in the data.meta_info to avoid slient error
        select_keys = [
            "input_ids",
            "attention_mask",
            "position_ids",
            "responses",
            "response_mask",
            "old_entropy",
        ]
        select_keys.extend(["old_log_probs", "ref_log_probs", "advantages"])
        non_tensor_select_keys = ["multi_modal_inputs"]

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(
            self.config.global_batch_size_per_device
        )

        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=1)

            for mini_batch in mini_batches:
                total_response_tokens = torch.sum(mini_batch.batch["response_mask"])
                dist.all_reduce(total_response_tokens, op=dist.ReduceOp.SUM)

                if self.config.dynamic_batching:
                    max_input_len = mini_batch.batch["input_ids"].size(-1)
                    max_token_len = (
                        self.config.micro_batch_size_per_device_for_update
                        * max_input_len
                    )
                    micro_batches, _ = prepare_dynamic_batch(
                        mini_batch, max_token_len=max_token_len
                    )
                else:
                    micro_batches = mini_batch.split(
                        self.config.micro_batch_size_per_device_for_update
                    )

                if self.rank == 0:
                    micro_batches = tqdm(
                        micro_batches, desc="Update policy", position=2
                    )

                for micro_batch in micro_batches:
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_probs = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    # all return: (bsz, response_length)
                    log_probs = self._forward_micro_batch(
                        model_inputs, temperature=temperature
                    )

                    # print(f"is using dp loss")
                    if self.config.use_dp_loss:
                        rho = self.config.rho
                        old_entropy = model_inputs["old_entropy"]  # (bs, response_length)

                        k = max(1, int(rho * old_entropy.size(1)))  # 至少保留1个token
                        
                        # 对每个样本，找出 top-k 高熵位置
                        topk_values, topk_indices = torch.topk(old_entropy, k=k, dim=1)  # (bs, k)
                        
                        # 创建新的 mask，只保留高熵位置
                        high_entropy_mask = torch.zeros_like(response_mask)
                        high_entropy_mask.scatter_(1, topk_indices, 1.0)
                        
                        # 与原始 response_mask 结合（确保不训练padding位置）
                        response_mask = response_mask * high_entropy_mask
                        
                    pg_loss, pg_metrics = compute_policy_loss(
                        old_log_probs=old_log_probs,
                        log_probs=log_probs,
                        advantages=advantages,
                        response_mask=response_mask,
                        clip_ratio_low=self.config.clip_ratio_low,
                        clip_ratio_high=self.config.clip_ratio_high,
                        clip_ratio_dual=self.config.clip_ratio_dual,
                        loss_type=self.config.loss_type,
                        loss_avg_mode=self.config.loss_avg_mode,
                    )
                    if self.config.use_kl_loss and "ref_log_probs" in model_inputs:
                        ref_log_probs = model_inputs["ref_log_probs"]
                        # compute kl loss
                        kld = compute_kl(
                            log_probs=log_probs,
                            ref_log_probs=ref_log_probs,
                            kl_penalty=self.config.kl_penalty,
                        )
                        kl_loss = average_loss(
                            kld, response_mask, mode=self.config.loss_avg_mode
                        )
                        loss = pg_loss + kl_loss * self.config.kl_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_coef
                    else:
                        loss = pg_loss

                    loss = (
                        loss
                        * torch.sum(response_mask)
                        * self.world_size
                        / total_response_tokens
                    )
                    loss.backward()

                    batch_metrics = {f"actor/{k}": v for k, v in pg_metrics.items()}
                    batch_metrics["actor/pg_loss"] = pg_loss.detach().item()
                    append_to_dict(metrics, batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        return metrics
