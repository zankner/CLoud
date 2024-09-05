import os
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor

from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm
from transformers import AutoTokenizer
from openai import OpenAI

from cloud.train.train import COT_PROMPT
from cloud.train.data import build_chat_messages

def build_feedback_prompts(tokenizer, example):
    bos_text = tokenizer.decode([tokenizer.bos_token_id])
    eos_text = tokenizer.decode([tokenizer.eos_token_id])
    eot_text =  "<|eot_id|>" # Hard coded for llama3 end of turn id for now but oh well

    chosen_prefix = tokenizer.apply_chat_template(build_chat_messages(example["prompt"], example["chosen"]), tokenize=False)
    rejected_prefix = tokenizer.apply_chat_template(build_chat_messages(example["prompt"], example["rejected"]), tokenize=False)
    cot_fmt = tokenizer.apply_chat_template([{"role": "user", "content": COT_PROMPT}], tokenize=False).replace(bos_text, "").replace(eos_text, "").replace(eot_text, "")

    example["chosen_feedback_prompt"] = chosen_prefix + cot_fmt
    example["rejected_feedback_prompt"] = rejected_prefix + cot_fmt
    return example

def main(args):

    # Load backend client
    if "gpt" in args.model:
        client = OpenAI()
    else:
        client = OpenAI(
            api_key="EMPTY",
            base_url="http://localhost:8000/v1"
        )

    # Load the dataset
    eot_text = "<|eot_id|>"
    if args.remote_server:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
        ref_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        tokenizer.chat_template = ref_tokenizer.chat_template
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    ds = load_dataset(args.base_dataset)
    ds = ds.map(lambda x: build_feedback_prompts(tokenizer, x), num_proc=10)

    def fetch_response(examples):
        chosen_feedback_prompts = [example["chosen_feedback_prompt"] for example in examples]
        chosen_feedback = [choice.text for choice in client.completions.create(
            model=args.model,
            prompt=chosen_feedback_prompts,
            temperature=args.temp,
            max_tokens=args.max_tokens,
            n=1,
            extra_body={
                "stop_token_ids": [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(eot_text)]
            }
        ).choices]

        rejected_feedback_prompts = [example["rejected_feedback_prompt"] for example in examples]
        rejected_feedback = [choice.text for choice in client.completions.create(
            model=args.model,
            prompt=rejected_feedback_prompts,
            temperature=args.temp,
            max_tokens=args.max_tokens,
            n=1,
            extra_body={
                "stop_token_ids": [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(eot_text)]
            }
        ).choices]

        results = [{**example, "chosen_feedback": [chosen_feedback], "rejected_feedback": [rejected_feedback]} for example, chosen_feedback, rejected_feedback in zip(examples, chosen_feedback, rejected_feedback)]
        return results

    feedback_ds = {}

    if args.splits is None:
        splits = ds.keys()
    else:
        splits = args.splits
    
    for split in splits:
        all_feedback = []
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []
            bs = 128 
            for i in range(0, len(ds[split]), bs):
                futures.append(executor.submit(fetch_response, [ds[split][i] for i in range(i, min(i+bs, len(ds[split])))]))
            for future in tqdm(futures, desc=f"Fetching {split} responses", leave=True):
                all_feedback.extend(future.result())
        feedback_ds[split] = all_feedback

    hf_feedback_ds = DatasetDict(
        {split: Dataset.from_list(feedback_ds[split]) for split in splits}
    )
    cols_to_select = ["prompt", "chosen", "rejected", "chosen_feedback", "rejected_feedback", "id"]
    hf_feedback_ds = hf_feedback_ds.select_columns(cols_to_select)

    print(f"Pushing data to {args.upload_name}")
    for split in splits:
        hf_feedback_ds[split].push_to_hub(
            args.upload_name,
            split=split
        )

if __name__ == "__main__":

    parser = ArgumentParser()
    
    # Model / data params
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--base-dataset", type=str, required=True)
    parser.add_argument("--splits", type=str, nargs="+", required=False)
    parser.add_argument("--upload-name", type=str, required=True)

    # Sampling params
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temp", type=float, default=0.0)

    # Misc
    parser.add_argument("--remote-server", action="store_true")

    args = parser.parse_args()

    main(args)
    