import argparse
import numpy as np
import sglang as sgl
from datasets import load_dataset
from transformers import AutoTokenizer
from verl.utils.reward_score.swe_rl import swe_rl_unidiff_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--dp", type=int, default=8)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--model_max_length", type=int, default=None)
    parser.add_argument("--prompt_key", type=str, default="text")
    parser.add_argument("--response_key", type=str, default="prediction")
    return parser.parse_args()


def build_messages(prompt):
    return [dict(role="user", content=prompt)]


def make_rollout_fn(llm, tokenizer, sampling_params, prompt_key="text", response_key="prediction", model_max_length=None):
    
    def process_fn(examples, idx):
        inputs = examples[prompt_key]
        messages = [build_messages(p) for p in inputs]
        prompts = tokenizer.apply_chat_template(messages, tokenize=False)
        responses = [""] * len(prompts)
        scores = [0.0] * len(prompts)
        
        token_lens = [len(tokenizer.encode(p)) for p in prompts]
        effective_model_max_length = model_max_length if model_max_length is not None else tokenizer.model_max_length
        token_limit = effective_model_max_length - sampling_params["max_new_tokens"]
        filtered_idx = [i for i, token_len in enumerate(token_lens) if token_len > token_limit]
        generation_idx = [i for i in range(len(prompts)) if i not in filtered_idx]
        for i in filtered_idx:
            print(f"Skipping example {idx[i]} because it's too long {token_lens[i]=} > {token_limit=}")
        
        remained_prompts = [prompts[i] for i in generation_idx]
        outputs = llm.generate(remained_prompts, sampling_params=sampling_params)
        for i, output in zip(generation_idx, outputs):
            responses[i] = output["text"]
        gts = examples["patch"]
        scores = [swe_rl_unidiff_score(None, response, gt) for gt, response in zip(gts, responses)]
        
        examples["input_length"] = token_lens
        examples[response_key] = responses
        examples["score"] = scores
        return examples
    
    return process_fn


"""
Evaluation script example:
python verl-train/test_rollout.py \
    --model_path checkpoints/swe-rl-exp/coder-32b-16k-swe-oracle-unidiff-train-model-Qwen2.5-Coder-32B-Instruct-bs128-gs16-adapt-ent-0.001-d-0.001-0-0.5-tgt-0.05-kl-0-temp0.6-1e-6-no_mask-4nodes/global_step_180/actor/huggingface \
    --dataset data/swe-verified-with-oracle-prompts.parquet \
    --output_path outputs/swe-rl-exp/coder-32b-16k-swe-oracle-unidiff-train-model-Qwen2.5-Coder-32B-Instruct-bs128-gs16-adapt-ent-0.001-d-0.001-0-0.5-tgt-0.05-kl-0-temp0.6-1e-6-no_mask-4nodes/global_step_180.parquet \
    --split test \
    --temperature 0.0 \
    --max_new_tokens 4096 
"""


if __name__ == "__main__":
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if args.model_max_length is not None:
        tokenizer.model_max_length = args.model_max_length
    else:
        args.model_max_length = tokenizer.model_max_length
    llm = sgl.Engine(model_path=args.model_path, dp_size=args.dp, context_length=args.model_max_length)

    sampling_params = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
    }
    if args.dataset.endswith(".parquet"):
        dataset = load_dataset("parquet", data_files=args.dataset, split="train")
    else:
        dataset = load_dataset(args.dataset, split=args.split)

    process_fn = make_rollout_fn(llm, tokenizer, sampling_params, args.prompt_key, args.response_key, args.model_max_length)
    dataset = dataset.map(process_fn, batched=True, batch_size=100, with_indices=True)
    scores = dataset["score"]
    print(f"Average score {np.mean(scores)=}")
    if args.output_path.endswith(".jsonl"):
        output_path = args.output_path.replace(".jsonl", f"_score_{np.mean(scores):.4f}.jsonl")
        dataset.to_json(output_path, lines=True)
    elif args.output_path.endswith(".parquet"):
        output_path = args.output_path.replace(".parquet", f"_score_{np.mean(scores):.4f}.parquet")
        dataset.to_parquet(output_path)
    else:
        raise ValueError(f"Unsupported output format: {args.output_path}, only support .jsonl and .parquet")
    llm.shutdown()
