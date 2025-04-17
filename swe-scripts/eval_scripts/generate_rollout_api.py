import os
import argparse
from secrets import token_hex
import numpy as np
import sglang as sgl
from datasets import load_dataset
from transformers import AutoTokenizer
from verl.utils.reward_score.swe_rl import swe_rl_search_replace_score
from openai import OpenAI

from openai_call import multithread_openai_call


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--base_url", type=str, default="https://localhost:30000/v1")
    parser.add_argument("--model_name", type=str, default="default")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--model_max_length", type=int, default=1024*32)
    parser.add_argument("--prompt_key", type=str, default="prompt")
    parser.add_argument("--response_key", type=str, default="prediction")
    return parser.parse_args()


def build_messages(prompt):
    return [dict(role="user", content=prompt)]


def calculate_score(example):
    gt = example["reward_model"]["ground_truth"]
    score = swe_rl_search_replace_score(None, example[args.response_key], gt, example["extra_info"])
    return {"score": score}



if __name__ == "__main__":
    args = parse_args()

    sampling_params = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    if args.dataset.endswith(".parquet"):
        dataset = load_dataset("parquet", data_files=args.dataset, split="train")
    else:
        dataset = load_dataset(args.dataset, split=args.split)

    selected_dataset = dataset.filter(lambda x: x["qwen_input_length"] <= args.model_max_length - args.max_tokens)
    print(f"Selected {len(selected_dataset)=} examples from {len(dataset)=} examples")

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"), base_url=args.base_url)
    messages = selected_dataset[args.prompt_key]
    responses = multithread_openai_call(client, messages, args.model_name, max_workers=32, **sampling_params)
    selected_dataset = selected_dataset.add_column(args.response_key, responses)

    selected_dataset = selected_dataset.map(calculate_score, num_proc=32)
    scores = selected_dataset["score"]
    print(f"Average score {np.mean(scores)=}")
    
    if args.output_path.endswith(".jsonl"):
        output_path = args.output_path.replace(".jsonl", f"_score_{np.mean(scores):.4f}.jsonl")
        selected_dataset.to_json(output_path, lines=True)
    elif args.output_path.endswith(".parquet"):
        output_path = args.output_path.replace(".parquet", f"_score_{np.mean(scores):.4f}.parquet")
        selected_dataset.to_parquet(output_path)
    else:
        raise ValueError(f"Unsupported output format: {args.output_path}, only support .jsonl and .parquet")
