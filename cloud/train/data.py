import time
from functools import partial

import omegaconf
import torch
from datasets import load_dataset
from composer.core import Evaluator
from composer.utils import dist
from torch.utils.data import DataLoader

def build_chat_messages(prompt, response):
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]

# Utility functions for tokenization
def build_prompt_tokens(prompt, tokenizer):
    tokenized_prompt = tokenizer.apply_chat_template([{
        "role": "user",
        "content": prompt
    }], tokenize=True)
    return tokenized_prompt, [-100] * len(tokenized_prompt)

# TODO: Currently use the <start_header> stuff as labels as well but probably want to remove them
def build_response_tokens(response, tokenizer, keep_targets):
    tokenized_response = tokenizer.apply_chat_template([{
        "role": "assistant",
        "content": response
    }], tokenize=True)
    if tokenized_response[0] == tokenizer.bos_token_id:
        tokenized_response = tokenized_response[1:]
    if keep_targets:
        return tokenized_response, tokenized_response[1:] + [-100]
    return tokenized_response, [-100] * len(tokenized_response)

def build_feedback_tokens(feedback, feedback_prompt, tokenizer, keep_targets):
    tokenized_feedback = tokenizer.apply_chat_template([{
        "role": "user",
        "content": feedback_prompt + feedback
    }], tokenize=True)
    if tokenized_feedback[0] == tokenizer.bos_token_id:
        tokenized_feedback = tokenized_feedback[1:]
    if keep_targets:
        return tokenized_feedback, tokenized_feedback[1:] + [-100]
    return tokenized_feedback, [-100] * len(tokenized_feedback)

def transform_to_seq_len(data, pad_val, max_seq_len):
    if len(data) > max_seq_len:
        fit_data = data[:max_seq_len]
    else:
        fit_data = data + [pad_val] * (max_seq_len - len(data))
    return torch.tensor(fit_data, dtype=torch.long)

def feedback_collate_fn(
    tokenizer,
    max_seq_len,
    feedback_method,
    cot_prompt,
    data,
):
    """Collator for feedback data.

    Args:
        tokenizer (Tokenzer): The model's tokenizer.
        max_seq_len (int): The maximum sequence length of the model.
        data (dict): The preference data to collate.
    """
    batch_chosen_input_ids = []
    batch_chosen_labels = []
    batch_chosen_attention_masks = []

    batch_rejected_input_ids = []
    batch_rejected_labels = []
    batch_rejected_attention_masks = []

    for sample_idx, sample in enumerate(data):

        # Trackers that will be returned
        chosen_input_ids = []
        chosen_labels = []

        rejected_input_ids = []
        rejected_labels = []

        # Build prompt tokens and labels
        prompt, prompt_labels = build_prompt_tokens(sample["prompt"], tokenizer)

        chosen_input_ids.extend(prompt)
        chosen_labels.extend(prompt_labels)

        rejected_input_ids.extend(prompt)
        rejected_labels.extend(prompt_labels)

        # Build chosen and rejected response tokens
        chosen_response, chosen_response_labels = build_response_tokens(sample["chosen"], tokenizer, keep_targets=False)
        rejected_response, rejected_response_labels = build_response_tokens(sample["rejected"], tokenizer, keep_targets=False)

        chosen_input_ids.extend(chosen_response)
        chosen_labels.extend(chosen_response_labels)

        rejected_input_ids.extend(rejected_response)
        rejected_labels.extend(rejected_response_labels)

        # Build chosen and rejected feedback tokens
        if isinstance(sample["chosen_feedback"], str):
            chosen_feedback = sample["chosen_feedback"]
            rejected_feedback = sample["rejected_feedback"]
        elif isinstance(sample["chosen_feedback"], list):
            chosen_feedback = sample["chosen_feedback"][0]
            rejected_feedback = sample["rejected_feedback"][0]
        else:
            raise ValueError(f"Invalid feedback type: {type(sample['chosen_feedback'])}")

        if feedback_method in ["csft", "teacher"]:
            chosen_feedback, chosen_feedback_labels = build_feedback_tokens(chosen_feedback, cot_prompt, tokenizer, keep_targets=True)
            rejected_feedback, rejected_feedback_labels = build_feedback_tokens(rejected_feedback, cot_prompt, tokenizer, keep_targets=True)

            chosen_input_ids.extend(chosen_feedback)
            chosen_labels.extend(chosen_feedback_labels)

            rejected_input_ids.extend(rejected_feedback)
            rejected_labels.extend(rejected_feedback_labels)
        
        # Convert to tensor and update batch data
        chosen_attn_mask = [1] * len(chosen_input_ids)
        rejected_attn_mask = [1] * len(rejected_input_ids)

        batch_chosen_input_ids.append(transform_to_seq_len(chosen_input_ids, tokenizer.pad_token_id, max_seq_len))
        batch_chosen_labels.append(transform_to_seq_len(chosen_labels, -100, max_seq_len))
        batch_chosen_attention_masks.append(transform_to_seq_len(chosen_attn_mask, 0, max_seq_len))

        batch_rejected_input_ids.append(transform_to_seq_len(rejected_input_ids, tokenizer.pad_token_id, max_seq_len))
        batch_rejected_labels.append(transform_to_seq_len(rejected_labels, -100, max_seq_len))
        batch_rejected_attention_masks.append(transform_to_seq_len(rejected_attn_mask, 0, max_seq_len))
    
    # Stack all to be tensor of batch shape
    batch_chosen_input_ids = torch.stack(batch_chosen_input_ids, dim=0)
    batch_chosen_labels = torch.stack(batch_chosen_labels, dim=0)
    batch_chosen_attention_masks = torch.stack(batch_chosen_attention_masks, dim=0)

    batch_rejected_input_ids = torch.stack(batch_rejected_input_ids, dim=0)
    batch_rejected_labels = torch.stack(batch_rejected_labels, dim=0)
    batch_rejected_attention_masks = torch.stack(batch_rejected_attention_masks, dim=0)

    # Force last token to be eos token id regardless
    batch_chosen_input_ids[:, -1] = tokenizer.eos_token_id
    batch_rejected_input_ids[:, -1] = tokenizer.eos_token_id

    # Handle labels when overflow seq length
    for sample_idx in range(batch_chosen_labels.shape[0]):
        if batch_chosen_labels[sample_idx, -2] != -100:
            batch_chosen_labels[sample_idx, -2] = tokenizer.eos_token_id
        if batch_rejected_labels[sample_idx, -2] != -100:
            batch_rejected_labels[sample_idx, -2] = tokenizer.eos_token_id
    batch_chosen_labels[:, -1] = -100
    batch_rejected_labels[:, -1] = -100
    
    return {
        "chosen_input_ids": batch_chosen_input_ids,
        "chosen_attention_mask": batch_chosen_attention_masks,
        "chosen_lm_labels": batch_chosen_labels,
        "rejected_input_ids": batch_rejected_input_ids,
        "rejected_attention_mask": batch_rejected_attention_masks,
        "rejected_lm_labels": batch_rejected_labels,
    }


def build_feedback_dataloader(
    cfg,
    device_batch_size,
    tokenizer,
    feedback_method,
    cot_prompt,
):
    """Build a streaming dataloader for preference data.

    Args:
        cfg (DictConfig): config to initialize the streaming components.
        device_batch_size (int): batch size per device.
        dataset_class (cls): the streaming dataset class to initialize.
        tokenizer (Tokenizer): the model's tokenizer.
        collate_fn: the function used to collate data.
        pad_token (str): the pad token to use.
        left_pad (bool): indiating if we should left pad the sequences.
        add_pad_token (bool): indicating if we should add a pad token to the tokenizer.
    """
    max_seq_len = cfg.dataset.pop("max_seq_len", None)

    dataset = load_dataset(cfg.dataset.remote, split=cfg.dataset.split)
    dist_sampler = dist.get_sampler(dataset, shuffle=cfg.dataset.shuffle, drop_last=cfg.drop_last)
    dataloader = DataLoader(
        dataset,
        collate_fn=partial(
            feedback_collate_fn, tokenizer, max_seq_len, feedback_method, cot_prompt
        ),
        sampler=dist_sampler,
        batch_size=device_batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.get('pin_memory', True),
        prefetch_factor=cfg.get('prefetch_factor', 2),
        persistent_workers=cfg.get('persistent_workers', True),
    )
    return dataloader


def build_evaluators(
    eval_loader_config,
    tokenizer,
    device_eval_batch_size,
    feedback_method,
    cot_prompt,
    metric_names
):
    evaluators = []
    assert isinstance(eval_loader_config, omegaconf.ListConfig)

    for i, eval_config in enumerate(eval_loader_config):
        label = eval_config.pop('label', f'eval-{i}')
        eval_dataloader = build_feedback_dataloader(
            eval_config,
            device_eval_batch_size,
            tokenizer,
            feedback_method,
            cot_prompt
        )
        eval_loader = Evaluator(
            label=f'eval/{label}',
            dataloader=eval_dataloader,
            metric_names=metric_names,
            device_eval_microbatch_size=device_eval_batch_size,
        )
        evaluators.append(eval_loader)
    return evaluators


if __name__ == "__main__":

    from transformers import AutoTokenizer
    from cloud.train.train import COT_PROMPT

    sample_data = [{
        "prompt": "What is the capital of the moon?",
        "chosen": "The moon does not have a capital.",
        "rejected": "The moon is made out of cheese.",
        "chosen_feedback": ["This response is correct."],
        "rejected_feedback": ["This response is funny but wrong."],
    }]

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    ret = feedback_collate_fn(tokenizer, 60, "teacher", COT_PROMPT, sample_data)

    print("CHOSEN TEXT")
    print(tokenizer.decode(ret["chosen_input_ids"][0]))
    print("=" * 100)
    print("CHOSEN LABELS")
    print(tokenizer.decode(ret["chosen_lm_labels"][0][ret["chosen_lm_labels"][0] != -100]))
    print("=" * 100)
    print("REJECTED TEXT")
    print(tokenizer.decode(ret["rejected_input_ids"][0]))
    print("=" * 100)
    print("REJECTED LABELS")
    print(tokenizer.decode(ret["rejected_lm_labels"][0][ret["rejected_lm_labels"][0] != -100]))
    print("=" * 100)
