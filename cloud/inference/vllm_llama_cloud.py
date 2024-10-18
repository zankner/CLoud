# Code taken from vllm llama model
"""Inference-only LLaMA model compatible with HuggingFace weights."""

from typing import Iterable, List, Optional, Tuple, Any
from copy import deepcopy

import torch
from torch import nn
from transformers import AutoTokenizer
from vllm.attention import AttentionMetadata
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata, SamplingTensors
from vllm.model_executor.layers.sampler import Sampler
from vllm.sequence import SamplerOutput
from vllm.model_executor.layers.sampler import _apply_penalties, _apply_top_k_top_p, _apply_min_p, _sample, _get_logprobs, _build_sampler_output

class LlamaRewardHead(nn.Module):

    def __init__(self, cfg, n_labels: int):
        super().__init__()
        self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        # use same dropout as attention dropout
        self.dropout = nn.Dropout(cfg.attention_dropout)
        self.out_proj = nn.Linear(cfg.hidden_size, n_labels)

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output = self.out_proj(hidden_states)
        return output

class RewardSampler(Sampler):


    def __init__(self, reward_token_id:int, **kwargs):
        self.reward_token_id = reward_token_id
        super().__init__(**kwargs)

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        """
        Args:
        logits: (num_tokens, vocab_size).
        sampling_metadata: Metadata for sampling.
        """
        assert logits is not None
        _, vocab_size = logits.shape

        is_reward_token = torch.sum(logits != -float('inf'), dim=1) == 1
        reward_scores = logits[:, self.reward_token_id].clone()
        batch_idx_to_reward_score = [(i, score.item()) for i, (is_reward, score) in enumerate(zip(is_reward_token, reward_scores)) if is_reward]

        # Prepare sampling tensors with pinned memory to avoid blocking.
        (sampling_tensors, do_penalties, do_top_p_top_k,
         do_min_p) = SamplingTensors.from_sampling_metadata(
             sampling_metadata, vocab_size, logits.device, logits.dtype)

        # Apply presence and frequency penalties.
        if do_penalties:
            logits = _apply_penalties(logits, sampling_tensors.prompt_tokens,
                                      sampling_tensors.output_tokens,
                                      sampling_tensors.presence_penalties,
                                      sampling_tensors.frequency_penalties,
                                      sampling_tensors.repetition_penalties)

        # Apply temperature scaling.
        # Use in-place division to avoid creating a new tensor.
        logits.div_(sampling_tensors.temperatures.unsqueeze_(dim=1))

        if do_top_p_top_k:
            logits = _apply_top_k_top_p(logits, sampling_tensors.top_ps,
                                        sampling_tensors.top_ks)

        if do_min_p:
            logits = _apply_min_p(logits, sampling_tensors.min_ps)

        # We use float32 for probabilities and log probabilities.
        # Compute the probabilities.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # Compute the log probabilities.
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

        for batch_idx, reward_score in batch_idx_to_reward_score:
            logprobs[batch_idx] = torch.zeros_like(logprobs[batch_idx])
            logprobs[batch_idx, self.reward_token_id] = reward_score

        # Sample the next tokens.
        sample_results, maybe_sampled_tokens_tensor = _sample(
            probs,
            logprobs,
            sampling_metadata,
            sampling_tensors,
            include_gpu_probs_tensor=self.include_gpu_probs_tensor,
            modify_greedy_probs=self._should_modify_greedy_probs_inplace,
        )

        if self.include_gpu_probs_tensor:
            assert maybe_sampled_tokens_tensor is not None
            on_device_tensors = (probs, logprobs, maybe_sampled_tokens_tensor)
        else:
            on_device_tensors = None

        # Get the logprobs query results.
        prompt_logprobs, sample_logprobs = _get_logprobs(
            logprobs, sampling_metadata, sample_results)
        return _build_sampler_output(sample_results,
                                     sampling_metadata,
                                     prompt_logprobs,
                                     sample_logprobs,
                                     on_device_tensors=on_device_tensors)

class LlamaCloudModel(nn.Module):

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__()
        self.reward_base_model = LlamaForCausalLM(**kwargs)
        self.reward_head = LlamaRewardHead(cfg=kwargs["config"], n_labels=1)

        tokenizer = AutoTokenizer.from_pretrained(kwargs["config"]._name_or_path)
        self.reward_token_id = tokenizer.encode("<|reserved_special_token_0|>", add_special_tokens=False)[0]
        self.eot_token_id = tokenizer.encode("<|eot_id|>", add_special_tokens=False)[0]
        self.vocab_size = kwargs["config"].vocab_size

        self.sampler = RewardSampler(self.reward_token_id)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.reward_base_model.forward(input_ids, positions, kv_caches, attn_metadata)
        num_tokens = input_ids.shape[0]
        for token_idx in range(num_tokens):
            if input_ids[token_idx] == self.eot_token_id:
                reward = self.reward_head(hidden_states[token_idx:token_idx+1])
                reward_hidden_state = torch.full_like(hidden_states[token_idx:token_idx+1], fill_value=reward.item())
                hidden_states[token_idx:token_idx+1] = reward_hidden_state
        return hidden_states
    
    # Generation fluff
    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:

        logits = self.reward_base_model.compute_logits(hidden_states, sampling_metadata)

        if logits is None:
            return logits

        for batch_idx, seq_end_idx in enumerate(sampling_metadata.selected_token_indices):
            if torch.allclose(hidden_states[seq_end_idx], hidden_states[seq_end_idx, 0]):
                reward = hidden_states[seq_end_idx, 0].item()
                logits[batch_idx] = torch.full_like(logits[batch_idx], float('-inf'))
                logits[batch_idx, self.reward_token_id] = reward
        
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return self.sampler(logits, sampling_metadata)

    # Load weights
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)