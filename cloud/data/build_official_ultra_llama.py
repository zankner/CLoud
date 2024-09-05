from argparse import ArgumentParser

from datasets import load_dataset


def build_official_ultra_llama(args):
    dataset_name = "Llama3-8b-ultra-" + args.mode + (f"-{args.model_size}" if args.mode == "self-gen" else "")
    print(dataset_name)
    ultra_llama_8b = load_dataset("ankner/" + dataset_name)
    ultra_feedback = load_dataset("openbmb/UltraFeedback")["train"]
    ultra_interact = load_dataset("openbmb/UltraInteract_pair")["train"]

    def lookup_prompt(example):
        subset = example["id"].split("-")[0]
        id_ = int(example["id"].split("-")[1])
        if subset == "feedback":
            prompt = ultra_feedback[id_]["instruction"]
        else:
            prompt = ultra_interact[id_]["trajectory"][0]["value"]
        return {**example, "prompt": prompt}
    
    ultra_llama_8b = ultra_llama_8b.map(lookup_prompt, num_proc=16)

    ultra_llama_8b.save_to_disk(f"datasets/{dataset_name}", max_shard_size="20MB") # Max shard size small for weird load_dataset bug

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["oracle", "self-gen"], required=True)
    parser.add_argument("--model-size", type=str, choices=["8b", "70b"], default="8b")
    args = parser.parse_args()

    build_official_ultra_llama(args)