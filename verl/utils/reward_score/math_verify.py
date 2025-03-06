# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from math_verify import parse, verify
from math_verify.metric import math_metric
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig


def compute_score(model_output: str, ground_truth: str) -> bool:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
    except Exception as e:
        print(e)

    return ret_score


def extract_yes_no(text: str) -> bool | None:
    """
    Search for 'Is the answer correct (Yes/No)? Yes|No' pattern in text and return True/False for Yes/No.
    Returns None if no match found, multiple matches found, or if the pattern is not at the end of the text.
    """
    text = text.strip()
    # Remove "<|im_end|>" if it's at the end of text
    if text.endswith("<|im_end|>"):
        text = text[:-len("<|im_end|>")]

    pattern = r"Is the answer correct \(Yes/No\)\?\s+(Yes|No)"
    matches = re.findall(pattern, text)
    
    # Return None if no matches or multiple matches found
    if len(matches) != 1:
        return None
    
    # Check if the pattern is at the end of the text
    last_match_pos = text.rfind(f"Is the answer correct (Yes/No)? {matches[0]}")
    if last_match_pos == -1 or last_match_pos + len(f"Is the answer correct (Yes/No)? {matches[0]}") != len(text):
        return None
    
    # Return True for Yes, False for No
    return matches[0] == "Yes"


def math_verify_reward(data_source, solution_str, ground_truth, extra_info=None):
    pred = parse(solution_str)
    gt = parse(f"${ground_truth}$")
    res = verify(gt, pred)
    return float(res)


def qwen_math_reward(data_source, solution_str, ground_truth, extra_info=None):
    pred = parse(solution_str)
    if "gsm8k" in data_source:
        gt = parse(ground_truth)
    else:
        gt = parse(f"${ground_truth}$")
    label = verify(gt, pred)
    res = float(label)
    if len(pred) == 0:
        res = -1.0

    code_blocks = re.findall(r'```python[\s\S]*?```', solution_str)
    if (len(code_blocks) > 0):
        res -= 0.5
    return float(res)


def train_verification_reward(data_source, solution_str, ground_truth, extra_info=None):
    output = extract_yes_no(solution_str)
    if output is None:
        res = -0.5
    else:
        correct = output == ground_truth
        if correct:
            # imbalanced dataset
            # res = 0.685 if output else 1.852
            res = 0.5 if output else 2.0
            # res = 1.0
        else:
            res = 0.0
    
    # penalty for code blocks
    code_blocks = re.findall(r'```python[\s\S]*?```', solution_str)
    if (len(code_blocks) > 0):
        res -= 0.5
    
    # penalty for short response
    if len(solution_str) <= 40:
        res -= 0.5
    return float(res)


def test_verification_reward(data_source, solution_str, ground_truth, extra_info=None):
    output = extract_yes_no(solution_str)
    if output is None:
        res = False
    else:
        res = output == ground_truth
    return float(res)
