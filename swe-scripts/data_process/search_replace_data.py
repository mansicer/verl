import re
import json
import os
import argparse
from typing import List, Dict, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd
import datasets
from datasets import Dataset, load_dataset
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer
from unidiff import PatchSet, UnidiffParseError

from prompts import AGENTLESS_REPAIR, CODE_FILE
from apply_patch import apply_patch_to_code_dict

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def read_jsonl(file_path: str) -> List[Dict]:
    """Read a JSONL file and return a list of dictionaries.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries containing the JSON data
    """
    data = []
    with open(file_path) as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                data.append(item)
    return data


def extract_code_section(example):
    text = example["text"]
    pattern = r'<code>(.*?)</code>'
    matches = re.findall(pattern, text, re.DOTALL)
    code_text = matches[0].strip()
    # Extract file sections using regex
    file_pattern = r'\[start of (.*?)\](.*?)\[end of \1\]'
    matches = re.findall(file_pattern, code_text, re.DOTALL)
    
    # Build dictionary of file paths and contents
    files_dict = {}
    for file_path, content in matches:
        # Split content into lines and process each line
        processed_lines = []
        for line in content.split('\n'):
            # Skip empty lines or lines with just line numbers
            if not line.strip():
                continue
            # Remove line number prefix if it exists
            if ' ' in line:
                _, text = line.split(' ', 1)
                processed_lines.append(text)
            else:
                processed_lines.append(line)
        
        # Join processed lines and store in dictionary
        files_dict[file_path.strip()] = '\n'.join(processed_lines).strip()
    example["file_names"] = list(files_dict.keys())
    example["file_contents"] = list(files_dict.values())
    return example


def make_map_fn(split: str, data_source: str, tokenizer: Any) -> callable:
    """Create a mapping function for processing dataset examples.
    
    Args:
        split: Dataset split name (e.g., 'train')
        data_source: Source identifier for the data
        tokenizer: Tokenizer instance to use
        
    Returns:
        Function that processes a single example
    """
    def process_fn(example: Dict, idx: int) -> Dict:
        problem_statement = example["problem_statement"]
        file_names = example["file_names"]
        file_contents = example["file_contents"]
        code_dict = {file_name: content for file_name, content in zip(file_names, file_contents)}

        file_prompts = [CODE_FILE.format(path=name, content=content) 
                       for name, content in zip(file_names, file_contents)]
        file_prompts = "\n\n".join(file_prompts)
        prompt = AGENTLESS_REPAIR.format(problem_statement=problem_statement, content=file_prompts)
        messages = [dict(role="user", content=prompt)]
        message_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        token_length = len(tokenizer.encode(message_prompt))

        try:
            answer = apply_patch_to_code_dict(code_dict, example["patch"])
            oracle_file_names = list(answer.keys())
            oracle_file_contents = list(answer.values())
        except (ValueError, UnidiffParseError) as e:
            print(f"Found error in {example['instance_id']}: {e}")
            oracle_file_names = []
            oracle_file_contents = []
            
        return {
            "instance_id": example["instance_id"],
            "qwen_input_length": token_length,
            "data_source": data_source,
            "prompt": messages,
            "ability": "swe",
            "reward_model": {
                "style": "rule",
                "ground_truth": (oracle_file_names, oracle_file_contents),
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "file_names": file_names,
                "file_contents": file_contents
            },
        }

    return process_fn

def process_data(
    data_path: str,
    output_path: str,
    model_path: str,
    split: str
) -> None:
    """Process and save the training data.
    
    Args:
        train_data_path: Path to the input training data JSONL file
        output_path: Path where to save the processed parquet file
        model_path: Path to the model checkpoint
    """
    if data_path.endswith(".jsonl"):
        # Load and prepare data
        train_data = read_jsonl(data_path)
        print(f"Initial training data length: {len(train_data)}")
        
        # Process file contents
        for item in train_data:
            file_contents = item["file_contents"]
            item["file_names"] = list(file_contents.keys())
            item["file_contents"] = list(file_contents.values())

        # Convert to Dataset
        train_data = Dataset.from_list(train_data)
    else:
        train_data = load_dataset(data_path, split=split)
        train_data = train_data.map(extract_code_section, num_proc=32)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Process data
    train_data = train_data.map(
        make_map_fn("train", "swe-oracle-search-replace", tokenizer),
        with_indices=True,
        remove_columns=train_data.column_names,
        num_proc=32,
    )
    
    # Filter out invalid examples
    train_data = train_data.filter(lambda x: len(x["reward_model"]["ground_truth"][0]) != 0)
    print(f"Valid data length: {len(train_data)}")
    
    # Save processed data
    train_data.to_parquet(output_path)

    train_data_16k = train_data.filter(lambda x: x["qwen_input_length"] <= 4096*3)
    print(f"16k data length: {len(train_data_16k)}")
    train_data_16k.to_parquet(output_path.replace(".parquet", "-16k.parquet"))

    train_data_24k = train_data.filter(lambda x: x["qwen_input_length"] <= 4096*5)
    print(f"24k data length: {len(train_data_24k)}")
    train_data_24k.to_parquet(output_path.replace(".parquet", "-24k.parquet"))

    train_data_32k = train_data.filter(lambda x: x["qwen_input_length"] <= 4096*7)
    print(f"32k data length: {len(train_data_32k)}")
    train_data_32k.to_parquet(output_path.replace(".parquet", "-32k.parquet"))


def main():
    """Parse command line arguments and process the data."""
    parser = argparse.ArgumentParser(
        description="Process SWE benchmark data and save as parquet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/augmented-data/nebius__SWE-bench-extra__style-2__fs-oracle.train.progress.jsonl",
        help="Path to the input training data JSONL file"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split to process"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/swe-bench-extra/search-replace/train.parquet",
        help="Path where to save the processed parquet file"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="/mnt/data/models/Qwen2.5-Coder-7B-Instruct",
        help="Path to the model checkpoint"
    )
    
    args = parser.parse_args()
    
    process_data(
        data_path=args.data_path,
        output_path=args.output_path,
        model_path=args.model_path,
        split=args.split
    )


# example usage:
# python swe-scripts/data_process/search_replace_data.py --data_path data/augmented-data/SWE-Gym__SWE-Gym__style-2__fs-oracle.train.progress.jsonl --output_path data/swe-gym/search_replace/train.parquet
if __name__ == "__main__":
    main()
