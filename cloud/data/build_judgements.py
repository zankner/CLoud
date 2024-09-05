import os
import re
import json
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor

from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm
from openai import OpenAI


SYSTEM_PROMPT = "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.\n\nBegin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.\n\nWhen evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.\n\nThen consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.\n\nThen consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.\n\nAfter providing your explanation, you must output only one of the following choices as your final verdict with a label:\n\n1. Assistant A is significantly better: [[A>>B]]\n2. Assistant A is slightly better: [[A>B]]\n3. Tie, relatively the same: [[A=B]]\n4. Assistant B is slightly better: [[B>A]]\n5. Assistant B is significantly better: [[B>>A]]\n\nExample output: \"My final verdict is tie: [[A=B]]\"."
USER_TEMPLATE = "<|User Prompt|>\n{question}\n\n<|The Start of Assistant A's Answer|>\n{answer_1}\n<|The End of Assistant A's Answer|>\n\n<|The Start of Assistant B's Answer|>\n{answer_2}\n<|The End of Assistant B's Answer|>"


def _get_key_and_url(model_name, port):
    if "gpt" in model_name:
        return None, None
    else:
        return "EMPTY", f"http://localhost:{port}/v1"


def build_generation_prompt(example):
    example["generation_prompt"] = [{
        "role": "user",
        "content": example["prompt"]
    }]
    return example


def get_score(judgement):
    pattern = re.compile(r"\[\[([AB<>=]+)\]\]")
    matches = pattern.findall(judgement)
    matches = [m for m in matches if m != ""]
    if len(set(matches)) == 1:
        return matches[0].strip("\n"), False
    else:
        return None, True


def get_judgements(question, response_1, response_2, client, judge_model, number_of_judgment_attempts=2):
    games = []
    for game_idx in range(2):
        conv = [{"role": "system", "content": SYSTEM_PROMPT}]
        prompt_args = {}

        if game_idx % 2 == 1:
            response_1, response_2 = response_2, response_1
        
        prompt_args["question"] = question
        prompt_args["answer_1"] = response_1
        prompt_args["answer_2"] = response_2

        user_prompt = USER_TEMPLATE.format(**prompt_args)
        conv.append({"role": "user", "content": user_prompt})

        judgment = ""
        for _ in range(number_of_judgment_attempts):
            new_judgment = client.chat.completions.create(
                model=judge_model,
                messages=conv,
                temperature=0.0,
                max_tokens=1024,
            ).choices[0].message.content

            judgment += ("\n" + new_judgment)

            score, try_again = get_score(judgment)

            conv.append({"role": "assistant", "content": new_judgment})

            if not try_again:
                break

            conv.append({"role": "user", "content": "continue your judgment and finish by outputting a final verdict label"})

        result = {
            "judgment": judgment,
            "score": score
        }
        games.append(result)
    return games

def check_winner(games):

    response_1_points = 0
    response_2_points = 0

    if games[0]["score"] == "A>B":
        response_1_points += 1
    elif games[0]["score"] == "A>>B":
        response_1_points += 2
    elif games[0]["score"] == "B>A":
        response_2_points += 1
    elif games[0]["score"] == "B>>A":
        response_2_points += 2
    
    if games[1]["score"] == "A>B":
        response_2_points += 1
    elif games[1]["score"] == "A>>B":
        response_2_points += 2
    elif games[1]["score"] == "B>A":
        response_1_points += 1
    elif games[1]["score"] == "B>>A":
        response_1_points += 2
    
    if response_1_points > response_2_points:
        return "response_1"
    elif response_1_points < response_2_points:
        return "response_2"
    else:
        return "tie"


model_to_fmt_name = {
    "meta-llama/Meta-Llama-3-8B-Instruct": "llama3-8B",
    "meta-llama/Meta-Llama-3.1-70B-Instruct": "llama3-70B",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-FP8": "llama3-405B",
}

def main(args):

    # Load the client -- if using local vllm inference assumes gen model is on port 8000 and judge model is on port 8001
    gen_api_key, gen_base_url = _get_key_and_url(args.gen_model, 8000)
    gen_client = OpenAI(
        api_key=gen_api_key,
        base_url=gen_base_url
    )

    judge_api_key, judge_base_url = _get_key_and_url(args.judge_model, 8001)
    judge_client = OpenAI(
        api_key=judge_api_key,
        base_url=judge_base_url
    )

    # Load the dataset
    ds = load_dataset(args.base_dataset)
    ds = ds.map(build_generation_prompt, num_proc=10)

    def fetch_response(example):

        responses = [choice.message.content for choice in gen_client.chat.completions.create(
            model=args.gen_model,
            messages=example["generation_prompt"],
            temperature=args.temp,
            max_tokens=args.max_tokens,
            n=args.num_responses
        ).choices]

        base_response = responses[0]

        winner_found = False
        for response in responses[1:]:
            games = get_judgements(
                example["prompt"],
                base_response,
                response,
                client=judge_client,
                judge_model=args.judge_model,
                number_of_judgment_attempts=args.num_judgment_attempts
            )
            if not all([game["score"] is not None for game in games]):
                continue

            winner = check_winner(games)
            if winner != "tie":
                winner_found = True
                break
        
        if not winner_found:
            return
        
        if winner == "response_1":
            chosen_response = base_response
            rejected_response = response
        else:
            chosen_response = response
            rejected_response = base_response
        
        return {**example, "chosen": chosen_response, "rejected": rejected_response}
    
    response_ds = {}
    
    if args.splits is None:
        splits = ds.keys()
    else:
        splits = args.splits
    for split in splits:

        os.makedirs("cloud/build_data/temp_data", exist_ok=True)
        temp_file_path = f"cloud/build_data/temp_data/{split}.jsonl"
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
                futures = [executor.submit(fetch_response, example) for example in ds[split]]
                null_count = 0
                total_count = 0
                pbar = tqdm(futures, desc=f"Fetching {split} responses")
                for future in pbar:
                    result = future.result()
                    total_count += 1
                    if result is None:
                        null_count += 1
                    else:
                        all_responses.append(result)
                        json.dump(result, temp_file)
                        temp_file.write("\n")
                    pbar.set_postfix({'No winner': f"{null_count}/{total_count} ({null_count/total_count:.2%})"}, refresh=True)
        response_ds[split] = all_responses
    
    hf_response_ds = DatasetDict(
        {split: Dataset.from_list(response_ds[split]) for split in splits}
    )

    cols_to_select = ["prompt", "id", "chosen", "rejected"]
    hf_response_ds = hf_response_ds.select_columns(cols_to_select)

    for split in splits:
        hf_response_ds[split].save_to_disk(
            os.path.join("datasets", args.save_name, split),
        )

if __name__ == "__main__":

    parser = ArgumentParser()
    
    # Model / data params
    parser.add_argument("--gen-model", type=str, required=True)
    parser.add_argument("--judge-model", type=str, required=True)
    parser.add_argument("--base-dataset", type=str, required=True)
    parser.add_argument("--save-name", type=str, required=True)
    parser.add_argument("--splits", type=str, nargs="+")

    # Sampling params
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--num-responses", type=int, default=5)
    parser.add_argument("--num-judgment-attempts", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=16)

    args = parser.parse_args()

    main(args)