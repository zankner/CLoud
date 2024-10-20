import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from cloud.model import CLoudRewardModel
from cloud.inference.api import CLoudAPI

REWARD_BENCH_TO_CATEGORY_MAPPING = {
    "Chat": [
        "alpacaeval-easy",
        "alpacaeval-length",
        "alpacaeval-hard",
        "mt-bench-easy",
        "mt-bench-med",
    ],
    "Chat Hard": [
        "mt-bench-hard",
        "llmbar-natural",
        "llmbar-adver-neighbor",
        "llmbar-adver-GPTInst",
        "llmbar-adver-GPTOut",
        "llmbar-adver-manual",
    ],
    "Safety": [
        "refusals-dangerous",
        "refusals-offensive",
        "xstest-should-refuse",
        "xstest-should-respond",
        "donotanswer",
    ],
    "Reasoning": [
        "math-prm",
        "hep-cpp",
        "hep-go",
        "hep-java",
        "hep-js",
        "hep-python",
        "hep-rust",
    ],
}

###########
# Build eval data
###########

def load_reward_bench():
    data = load_dataset("allenai/reward-bench")["filtered"]
    eval_data = []
    eval_metadata = []
    for example in data:
        eval_data.append({
            "id": f"{example['id']}-chosen",
            "prompt": example["prompt"],
            "response": example["chosen"]
        })
        eval_data.append({
            "id": f"{example['id']}-rejected",
            "prompt": example["prompt"],
            "response": example["rejected"]
        })
        eval_metadata.append({
            "id": str(example["id"]),
            "subset": example["subset"]
        })
    return eval_data, eval_metadata


###########
# Post-process Scores
###########

def post_process_reward_bench(eval_metadata, rewards):
    per_category_scores = {category: [] for category in REWARD_BENCH_TO_CATEGORY_MAPPING.keys()}
    for example in eval_metadata:
        id_ = example["id"]
        chosen_reward = rewards[id_ + "-chosen"]
        rejected_reward = rewards[id_ + "-rejected"]
        for category, subsets in REWARD_BENCH_TO_CATEGORY_MAPPING.items():
            if example["subset"] in subsets:
                per_category_scores[category].append(int(chosen_reward > rejected_reward))
                break
    per_category_scores = {category: np.mean(scores) * 100 for category, scores in per_category_scores.items()}
    per_category_scores["Average"] = np.mean([score for score in per_category_scores.values()])

    # Print scores in a pretty way
    print("\nReward Bench Scores:")
    print("=" * 40)
    max_category_length = max(len(category) for category in per_category_scores.keys())
    for category, score in per_category_scores.items():
        print(f"{category:<{max_category_length}} : {score:.2f}%")
    print("=" * 40)

    return per_category_scores



###########
# Scoring
###########

def generate_rewards_hf(model, tokenizer, eval_data, batch_size):
    rewards = {}

    for i in tqdm(range(0, len(eval_data), batch_size)):
        batch = eval_data[i:i+batch_size]
        
        prompts = [item["prompt"] for item in batch]
        responses = [item["response"] for item in batch]
        ids = [item["id"] for item in batch]

        batch_rewards, _ = model.predict_reward(prompts, responses, tokenizer)
        
        for id_, reward in zip(ids, batch_rewards):
            rewards[id_] = reward

    return rewards

def generate_rewards_vllm(client: CLoudAPI, eval_data, num_workers):
    rewards = {}

    def fetch_reward(example):
        critique, reward = client.get_reward(
            example["prompt"],
            example["response"],
        )
        return critique, reward, example["id"]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(fetch_reward, example): example["id"] for example in eval_data}
        for future in tqdm(as_completed(futures), total=len(eval_data), desc="Generating rewards"):
            critique, reward, id_ = future.result()
            rewards[id_] = reward
    
    return rewards



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--benchmark", type=str, default="reward-bench", choices=["reward-bench", "arena-hard"])

    # Vllm args
    parser.add_argument("--inference-method", type=str, default="hf", choices=["hf", "vllm"])
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--hosted", action="store_true")

    # HF args
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()


    if args.benchmark == "reward-bench":
        eval_data, eval_metadata = load_reward_bench()
    
    if args.inference_method == "hf":
        model = CLoudRewardModel.from_pretrained(args.model_path, device_map="cuda")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
        rewards = generate_rewards_hf(model, tokenizer, eval_data, batch_size=args.batch_size)
    elif args.inference_method == "vllm":
        client = CLoudAPI(model=args.model_path, tensor_parallel_size=args.tensor_parallel_size, hosted=args.hosted)
        rewards = generate_rewards_vllm(client, eval_data, num_workers=args.num_workers)

    post_process_reward_bench(eval_metadata, rewards)
