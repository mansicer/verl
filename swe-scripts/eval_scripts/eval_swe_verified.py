import re
import os
import time
import json
import numpy as np
import pandas as pd
import datasets
import argparse
import requests
import warnings

from typing import List, Dict
from pprint import pprint
from sympy import Idx
from tqdm import tqdm
from datasets import load_dataset

from verl.utils.reward_score.swe_rl.original import extract_thought_solution, parse_search_replace, apply_code_change, get_filelevel_diff, FormatError, UnidiffParseError
from verl.utils.reward_score.swe_rl.diff_utils import generate_patch_from_dicts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="data/princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--eval_url", type=str, default="http://47.251.84.48:8000/sync/run_task")
    parser.add_argument("--batch_size", type=int, default=10)
    return parser.parse_args()


def extract_patch_content(solution_str):
    patch_content = re.findall(r'<patch>(.*?)</patch>', solution_str, re.DOTALL)
    if len(patch_content) == 0:
        return "", False
    patch_content = patch_content[-1].strip()
    changes = get_filelevel_diff(patch_content)
    if len(changes) == 0:
        return "", False
    return patch_content, True


def extract_search_replace_patch(example):
    pred = example["prediction"]
    file_names = example["extra_info"]["file_names"]
    file_contents = example["extra_info"]["file_contents"]
    try:
        code_dict = {name: content for name, content in zip(file_names, file_contents)}
        thought, solution = extract_thought_solution(pred)
        code_changes = parse_search_replace(solution)
        pred_dict = apply_code_change(code_dict, code_changes)
        patch = generate_patch_from_dicts(code_dict, pred_dict)
    except (FormatError, UnidiffParseError):
        return "", False
    return patch, True


def build_request_data(ids, model_name, patches):
    data = []
    for id, patch in zip(ids, patches):
        data.append({
            "instance_id": id,
            "model_name_or_path": "gold",
            "model_patch": patch
        })
    return dict(predictions=data)


def request_example():
    example = {
        "predictions": [
            {
                "instance_id": "sympy__sympy-20590",
                "model_name_or_path": "gold",
                "model_patch": "--- a/sympy/core/_print_helpers.py\n+++ b/sympy/core/_print_helpers.py\n@@ -17,6 +17,11 @@ class Printable:\n     This also adds support for LaTeX printing in jupyter notebooks.\n     \"\"\"\n \n+    # Since this class is used as a mixin we set empty slots. That means that\n+    # instances of any subclasses that use slots will not need to have a\n+    # __dict__.\n+    __slots__ = ()\n+\n     # Note, we always use the default ordering (lex) in __str__ and __repr__,\n     # regardless of the global setting. See issue 5487.\n     def __str__(self):\n"
            },
            {
                "instance_id": "django__django-17087",
                "model_name_or_path": "gold",
                "model_patch": "+++ b/django/db/migrations/serializer.py\n@@ -168,7 +168,7 @@ def serialize(self):\n         ):\n             klass = self.value.__self__\n             module = klass.__module__\n-            return \"%s.%s.%s\" % (module, klass.__name__, self.value.__name__), {\n+            return \"%s.%s.%s\" % (module, klass.__qualname__, self.value.__name__), {\n                 \"import %s\" % module\n             }\n         # Further error checking\n"
            }
        ]
    }
    response = requests.post("http://47.251.84.48:8000/sync/run_task", json=example)
    print(response)
    print(response.json())
    return response.json()

"""
Example:
python swe-scripts/eval_scripts/eval_swe_verified.py --rollout_path outputs/swe-verified-eval/search-replace/deepseek-chat-temp-0.0-tokens-4096-maxlen-131072_score_0.1138.jsonl
"""


if __name__ == "__main__":
    status = request_example()
    if status["status"] == "error":
        print("Error: api is busy now, please try again later")
        exit()
        
    args = parse_args()
    dataset = load_dataset(args.dataset, split="test")
    rollout_path = args.rollout_path
    model_name = os.path.basename(rollout_path).split(".")[0]
    if rollout_path.endswith(".jsonl"):
        outputs = load_dataset("json", data_files=rollout_path, split="train")
    elif rollout_path.endswith(".parquet"):
        outputs = load_dataset("parquet", data_files=rollout_path, split="train")
    else:
        raise ValueError(f"Unsupported file format: {rollout_path}")

    instance_ids = dataset["instance_id"]
    selected_idx = []
    for instance_id in instance_ids:
        # assert instance_id in outputs["instance_id"], f"Instance ID {instance_id} not found in swe-verified dataset"
        if instance_id in outputs["instance_id"]:
            selected_idx.append(outputs["instance_id"].index(instance_id))

    outputs = outputs.select(selected_idx)

    patch_contents, is_valid = zip(*[extract_search_replace_patch(example) for example in outputs])
    valid_idx = [i for i, valid in enumerate(is_valid) if valid]
    print(f"Number of valid patches: {len(valid_idx)}/{len(outputs)}")

    ids_to_be_verified = []
    patches_to_be_verified = []
    for idx in valid_idx:
        ids_to_be_verified.append(instance_ids[idx])
        patches_to_be_verified.append(patch_contents[idx])
    
    # Create a list to track labels that will be updated
    labels = [False] * len(outputs)
    
    print("Start verification...")
    start_time = time.time()

    # Process patches in batches
    batch_size = args.batch_size
    total_batch = (len(ids_to_be_verified) + batch_size - 1) // batch_size
    for i in range(0, len(ids_to_be_verified), batch_size):
        batch_start_time = time.time()
        batch_ids = ids_to_be_verified[i:i + batch_size]
        batch_patches = patches_to_be_verified[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{total_batch}")
        
        request_data = build_request_data(batch_ids, model_name, batch_patches)
        response = requests.post(args.eval_url, json=request_data)
        
        results = response.json()["result"]
        print(f"Resolved {len(results['resolved_ids'])}/{len(batch_ids)} tasks")
        for resolved_id in results["resolved_ids"]:
            idx = outputs["instance_id"].index(resolved_id)
            labels[idx] = True
            
        batch_time = time.time() - batch_start_time
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time}] Time cost for batch {i//batch_size + 1}: {batch_time:.2f}s")
    
    # Update the dataset with the new labels
    outputs = outputs.add_column("label", labels)
    
    time_cost = time.time() - start_time
    current_time = time.strftime("%Y-%m-%d %H:%M:%S") 
    print(f"[{current_time}] Time cost for verification: {time_cost:.2f}s")

    mean_acc = np.sum(labels) / len(dataset)
    labeled_data_path = rollout_path.replace(".parquet", f"_acc_{mean_acc:.4f}_verified.parquet")
    outputs.to_parquet(labeled_data_path)
    print(f"Labeled data saved to {labeled_data_path}")

    selected_outputs = outputs.filter(lambda x: x["qwen_input_length"] <= 12288)
    mean_acc_16k = np.sum(selected_outputs["label"]) / len(selected_outputs)
    selected_outputs = outputs.filter(lambda x: 12288 < x["qwen_input_length"] <= 20480)
    mean_acc_24k = np.sum(selected_outputs["label"]) / len(selected_outputs)
    selected_outputs = outputs.filter(lambda x: 20480 < x["qwen_input_length"] <= 28672)
    mean_acc_32k = np.sum(selected_outputs["label"]) / len(selected_outputs)
    selected_outputs = outputs.filter(lambda x: x["qwen_input_length"] > 28672)
    mean_acc_32k_plus = np.sum(selected_outputs["label"]) / len(selected_outputs)
    print(f"Acc: {mean_acc:.4f}, 16k context Acc: {mean_acc_16k:.4f}, 24k context Acc: {mean_acc_24k:.4f}, 32k context Acc: {mean_acc_32k:.4f}, 32k+ context Acc: {mean_acc_32k_plus:.4f}")
