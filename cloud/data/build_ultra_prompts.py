import os
import uuid
from argparse import ArgumentParser

from datasets import load_dataset, concatenate_datasets


def convert_feedback_to_pairwise():

    ds = load_dataset("openbmb/UltraFeedback")["train"]

    ds = ds.rename_column("instruction", "prompt")
    ds = ds.select_columns(["prompt"])

    return ds


def convert_interact_to_pairwise():

    ds = load_dataset("openbmb/UltraInteract_pair")["train"]

    ds = ds.filter(lambda x: len(x["trajectory"] == 1))
    ds = ds.map(lambda x: {**x, "prompt": x["trajectory"][0]["value"]})
    ds = ds.select_columns(["prompt"])

    return ds


def merge_datasets(feedback_ds, interact_ds):

    feedback_length = len(feedback_ds)
    print(f"Sample rate: {feedback_length / len(interact_ds)}")
    interact_ds = interact_ds.shuffle(seed=42).select(range(feedback_length))

    ds = concatenate_datasets([interact_ds, feedback_ds])

    ds = ds.map(lambda x: {**x, "id": str(uuid.uuid4())})

    ds = ds.shuffle(seed=42)
    ds = ds.train_test_split(test_size=args.test_size, seed=42)

    return ds

def build_ultra_prompts(args):

    feedback_ds = convert_feedback_to_pairwise()
    interact_ds = convert_interact_to_pairwise()
    ds = merge_datasets(feedback_ds, interact_ds)

    ds.save_to_disk(os.path.join("datasets", args.save_name))


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.05)
    parser.add_argument("--save-name", type=str, required=True)
    args = parser.parse_args()

    build_ultra_prompts(args)