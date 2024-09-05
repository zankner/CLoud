import os
import json
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor

from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm
from openai import OpenAI

SOLO_SYSTEM_PROMPT = """Please act as an expert in providing feedback. You will be given a user's prompt and an assistant's response to the prompt. Your job is to think step by step and provide a thoughtful and detailed analysis of how well the response answers the user's query. \n\n When providing feedback, consider if the assistant's answer is helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.\n\nIt is not necessary to be polite when providing feedback. Think deeply to identify all the good and bad parts of the answer.\n\nDo not numerically score the resposne when evaluating it. Think deeply and provide linguistic feedback and analysis only."""
SOLO_USER_TEMPLATE = """Here is the user's prompt and the assistant's response.\n\n<|User Prompt|>\n{question}\n\n<|The Start of the Assistant's Answer|>\n{answer}\n<|The End of the Assistant's Answer|>"""


def _get_key_and_url(model_name):
    if "gpt" in model_name:
        return None, None
    else:
        return "EMPTY", "http://localhost:8000/v1"

def build_critique_prompt(example):

    prompt_args = {
        "question": example["prompt"],
    }

    chosen_critique_prompt = [
        {"role": "system", "content": SOLO_SYSTEM_PROMPT},
        {"role": "user", "content": SOLO_USER_TEMPLATE.format(answer=example["chosen"], **prompt_args)},
    ]
    rejected_critique_prompt = [
        {"role": "system", "content": SOLO_SYSTEM_PROMPT},
        {"role": "user", "content": SOLO_USER_TEMPLATE.format(answer=example["rejected"], **prompt_args)},
    ]

    return {**example, "chosen_critique_prompt": chosen_critique_prompt, "rejected_critique_prompt": rejected_critique_prompt}

model_to_fmt_name = {
    "meta-llama/Meta-Llama-3-8B-Instruct": "llama3-8B",
    "meta-llama/Meta-Llama-3.1-70B-Instruct": "llama3-70B",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-FP8": "llama3-405B",
}

def main(args):

    # Load the client
    api_key, base_url = _get_key_and_url(args.judge_model)
    judge_client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    # Load the dataset
    ds = load_dataset(args.base_dataset)
    ds = ds.map(build_critique_prompt)

    def fetch_response(example):

        chosen_critique = judge_client.chat.completions.create(
            model=args.judge_model,
            messages=example["chosen_critique_prompt"],
            temperature=args.temp,
            max_tokens=args.max_tokens,
        ).choices[0].message.content

        rejected_critique = judge_client.chat.completions.create(
            model=args.judge_model,
            messages=example["rejected_critique_prompt"],
            temperature=args.temp,
            max_tokens=args.max_tokens,
        ).choices[0].message.content


        return {
            **example,
            "chosen_feedback": chosen_critique,
            "rejected_feedback": rejected_critique
        }
    
    response_ds = {}
    if args.splits is None:
        splits = ds.keys()
    else:
        splits = args.splits
    for split in splits:
        all_responses = []
        temp_file_path = f"cloud/build_data/temp_critique/{'seed' if args.from_seed else 'no_seed'}_{split}.jsonl"
        print(f"Storing temporary critiques at: {temp_file_path}")
        if os.path.exists(temp_file_path):
            with open(temp_file_path, "r") as f:
                already_written = [json.loads(line) for line in f]
            already_written_ids = set([example["id"] for example in already_written])
            print(f"Already written: {len(already_written_ids)}")
            ds[split] = ds[split].filter(lambda x: x["id"] not in already_written_ids)
            all_responses = already_written
        else:
            all_responses = []

        with open(temp_file_path, "a") as temp_file:
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                futures = [executor.submit(fetch_response, examples) for examples in ds[split]]
                for future in tqdm(futures, desc=f"Fetching {split} responses"):
                    result = future.result()
                    all_responses.append(result)
                    json.dump(result, temp_file)
                    temp_file.write("\n")
        response_ds[split] = all_responses
    
    hf_response_ds = DatasetDict(
        {split: Dataset.from_list(response_ds[split]) for split in splits}
    )

    cols_to_select = ["prompt", "id", "chosen", "rejected", "chosen_feedback", "rejected_feedback"]
    hf_response_ds = hf_response_ds.select_columns(cols_to_select)

    for split in splits:
        hf_response_ds[split].save_to_disk(
            os.path.join("datasets", args.save_name, split),
        )

if __name__ == "__main__":

    parser = ArgumentParser()
    
    # Model / data params
    parser.add_argument("--judge-model", type=str, required=True)
    parser.add_argument("--splits", type=str, nargs="+")
    parser.add_argument("--base-dataset", type=str, required=True)
    parser.add_argument("--save-name", type=str, required=True)

    # Sampling args
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=16)

    args = parser.parse_args()

    main(args)