from typing import Any

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel, AutoModelForCausalLM, AutoConfig

COT_PROMPT = "The following is a break down on the correctness and usefulness of the assistant's response to my question: "

class CLoudRewardModelConfig(PretrainedConfig):
    """
    Configuration class for Reward Model.

    Args:
        base_model_name_or_path: Name of the base model
        **kwargs: Additional keyword arguments to be passed to the parent class constructor.
    """

    def __init__(self, feedback_method="vanilla", base_model_name_or_path="meta-llama/Meta-Llama-3-8B", **kwargs):
        
        assert feedback_method in ["vanilla", "teacher"]

        self.feedback_method = feedback_method
        self.base_model_name_or_path = base_model_name_or_path

        super().__init__(**kwargs)
    
class RewardHead(nn.Module):

    def __init__(self, cfg: PretrainedConfig, n_labels: int):
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

class CLoudRewardModel(PreTrainedModel):
    config_class = CLoudRewardModelConfig

    def __init__(self, config, pretrained_reward_base_model=None):
        super().__init__(config)
        self.feedback_method = config.feedback_method

        if pretrained_reward_base_model is None:
            reward_base_model_cfg = AutoConfig.from_pretrained(config.base_model_name_or_path)
            self.reward_base_model = AutoModelForCausalLM.from_config(reward_base_model_cfg)
        else:
            self.reward_base_model = pretrained_reward_base_model
        self.reward_head = RewardHead(self.reward_base_model.config, 1)

        self._no_split_modules = self.reward_base_model._no_split_modules

    # Only used during training
    def forward(self, input_ids, attention_mask):
        batch_size, _ = input_ids.shape

        output = self.reward_base_model(
            input_ids,
            attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        hidden_states = output.hidden_states[-1]
        rewards = self.reward_head(hidden_states)
        sequence_lengths = torch.sum(attention_mask, dim=-1) - 1
        rewards = rewards[torch.arange(batch_size, device=rewards.device), sequence_lengths]

        return rewards, output.logits

    def prepare_inputs_for_reward(self, user_prompts, assistant_responses, tokenizer, critique_prompt):
        formatted_prompts = []
        for user_prompt, assistant_response in zip(user_prompts, assistant_responses):
            input_prefix = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response},
                ],
                tokenize=False
            )

            if self.feedback_method == "teacher":
                critique_prefix = tokenizer.apply_chat_template(
                    [{"role": "user", "content": critique_prompt}],
                    tokenize=False
                )
                critique_prefix = critique_prefix.replace(tokenizer.decode([tokenizer.bos_token_id]), "")
                critique_prefix = critique_prefix.replace(tokenizer.decode([tokenizer.eos_token_id]), "")
                critique_prefix = critique_prefix.replace("<|eot_id|>", "") # Currently hard coded for llama3 chat template

                input_prefix += critique_prefix
            
            formatted_prompts.append(input_prefix)
        
        return formatted_prompts, tokenizer(formatted_prompts, add_special_tokens=False, return_tensors="pt", padding=True).to(self.reward_base_model.device)
    
    @torch.inference_mode()
    def predict_reward(
        self,
        user_prompts,
        assistant_responses,
        tokenizer,
        critique_prompt=COT_PROMPT,
        temp=0.0,
        max_tokens=1024,
    ):
        """
        args:
            user_prompts: List[str] -- list of user prompts
            assistant_responses: List[str] -- list of assistant responses
            tokenizer: Tokenizer -- tokenizer for the reward model
            critique_prompt: str -- prompt for generating critiques
            temp: float -- temperature for sampling critiques
            max_tokens: int -- maximum number of tokens to generate
        returns:
            rewards: torch.Tensor -- rewards for the assistant responses
            critiques: List[str] -- critiques for the assistant responses
        """
        formatted_prompts, reward_model_inputs = self.prepare_inputs_for_reward(user_prompts, assistant_responses, tokenizer, critique_prompt)
        batch_size, input_seq_len = reward_model_inputs.input_ids.shape

        eot_token_text = "<|eot_id|>"
        eot_token_id = tokenizer.encode(eot_token_text, add_special_tokens=False)[0] # Hard coded for llama3 chat template

        if self.feedback_method == "vanilla":
            outputs = self.reward_base_model(
                **reward_model_inputs,
                output_hidden_states=True,
                return_dict=True
            )
            critiques = [""] * batch_size
        elif self.feedback_method == "teacher":
            outputs = self.reward_base_model.generate(
                **reward_model_inputs,
                max_new_tokens=max_tokens,
                temperature=temp if temp > 0 else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eot_token_id,
                return_dict_in_generate=True,
                repetition_penalty=1.01
            )

            critique_ids = outputs.sequences[:, input_seq_len:]
            critique_ids[critique_ids == tokenizer.eos_token_id] = eot_token_id
            
            critiques = []
            for critique_idx in range(batch_size):
                eot_idx = (critique_ids[critique_idx] == eot_token_id).nonzero()
                if eot_idx.numel() > 0:
                    eot_idx = eot_idx[0].item()
                    critiques.append(tokenizer.decode(critique_ids[critique_idx, :eot_idx]).strip())
                else:
                    eos_idx = (critique_ids[critique_idx] == tokenizer.eos_token_id).nonzero()
                    if eos_idx.numel() > 0:
                        eos_idx = eos_idx[0].item()
                        critiques.append(tokenizer.decode(critique_ids[critique_idx, :eos_idx]).strip())
                    else:
                        critiques.append(tokenizer.decode(critique_ids[critique_idx]).strip())
            
            critique_prompts = [formatted_prompt + critique + eot_token_text for formatted_prompt, critique in zip(formatted_prompts, critiques)]
            reward_model_inputs = tokenizer(critique_prompts, add_special_tokens=False, return_tensors="pt", padding=True).to(self.reward_base_model.device)
            
            outputs = self.reward_base_model(
                **reward_model_inputs,
                output_hidden_states=True,
                return_dict=True
            )
        
        rewards = self.reward_head(outputs.hidden_states[-1][:, -1]).flatten().tolist()

        return rewards, critiques


if __name__ == "__main__":
    from transformers import AutoTokenizer

    model_name = "ankner/Llama3-8B-CLoud-RM"
    # model_name = "ankner/Llama3-8B-Classic-RM"
    model = CLoudRewardModel.from_pretrained(model_name, device_map="cuda", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    user_prompt = ["Write me a story", "What is the capital of the moon?"]
    assistant_response = ["No I don't want to do that.", "Since the moon is made out of cheese, the capital is mozzerella."]

    rewards, critiques = model.predict_reward(user_prompt, assistant_response, tokenizer)
    for prompt, response, reward, critique in zip(user_prompt, assistant_response, rewards, critiques):
        print("Prompt:")
        print(prompt)
        print("Response:")
        print(response)
        print("Critique:")
        print(critique)
        print("Reward:")
        print(reward)
        print("=" * 100)
