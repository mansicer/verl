import os

VERIFICATION_REWARD_TYPE = os.getenv("VERIFICATION_REWARD_TYPE", "baseline")

import re
try:
    from math_verify import parse, verify
    from math_verify.metric import math_metric
    from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
    from math_verify.errors import TimeoutException
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")

import logging

# Disable all loggers starting with 'math_verify'
for name, logger in logging.Logger.manager.loggerDict.items():
    if isinstance(logger, logging.Logger) and name.startswith("math_verify"):
        logger.disabled = True
        logger.handlers.clear()


def compute_score(data_source, solution_str, ground_truth, extra_info):
    split = extra_info.get("split", "train")
    if data_source.startswith("math-verify"):
        res = math_verify_reward(data_source, solution_str, ground_truth, extra_info)

    elif data_source == "qwen-math":
        res = qwen_math_reward(data_source, solution_str, ground_truth, extra_info)

    elif data_source.startswith("verification"):
        if split == "test":
            res = test_verification_reward(data_source, solution_str, ground_truth, extra_info)
        else:
            res = train_verification_reward(data_source, solution_str, ground_truth, extra_info)

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    return res


def math_verify_reward(data_source, solution_str, ground_truth, extra_info=None):
    pred = parse(solution_str)
    if "gsm8k" in data_source:
        gt = parse(ground_truth)
    else:
        gt = parse(f"${ground_truth}$")
    res = verify(gt, pred)
    return dict(score=float(res))


def qwen_math_reward(data_source, solution_str, ground_truth, extra_info=None):
    reward = 0.0
    reward_dict = {}

    pred = parse(solution_str)
    gt = parse(f"${ground_truth}$")
    label = verify(gt, pred)
    reward += float(label)

    code_blocks = re.findall(r'```python[\s\S]*?```', solution_str)
    reward_dict["contain_code"] = len(code_blocks) > 0
    if reward_dict["contain_code"]:
        reward -= 0.5
    reward_dict["score"] = reward
    return reward_dict


def extract_yes_no(text: str) -> bool | None:
    """
    Search for 'Is the answer correct (Yes/No)? Yes|No' pattern in text and return True/False for Yes/No.
    Returns None if no match found, multiple matches found, or if the pattern is not at the end of the text.
    """
    pattern = r"Is the answer correct \(Yes/No\)\?\s+(Yes|No)"
    matches = re.findall(pattern, text)
    if len(matches) > 0:
        return matches[-1] == "Yes"
    else:
        result = None
        if "is correct" in text.lower():
            result = True
        elif "is incorrect" in text.lower() or "is not correct" in text.lower() or "is wrong" in text.lower():
            result = False
        return result


def train_verification_reward(data_source, solution_str, ground_truth, extra_info=None):
    reward = 0.0
    reward_dict = {}

    text = solution_str.strip().replace("**", "")
    label, correct_ratio = ground_truth.split("|")
    label, correct_ratio = eval(label), float(correct_ratio)
    
    output = extract_yes_no(text)
    reward_dict["valid_verification_form"] = output is not None
    if reward_dict["valid_verification_form"]:
        if VERIFICATION_REWARD_TYPE == "baseline":
            reward = label == output
        elif VERIFICATION_REWARD_TYPE == "fix_imbalance":
            if label == output:
                reward += float(output) * (1 - correct_ratio) + (1 - float(output)) * correct_ratio
                reward *= 2
        else:
            raise NotImplementedError(f"Reward function is not implemented for {VERIFICATION_REWARD_TYPE=}")

    def check_consistency(text: str) -> bool:
        pattern = r"Is the answer correct \(Yes/No\)\?\s+(Yes|No)"
        matches = re.findall(pattern, text)
        if len(matches) != 1:
            return False
        if len(set(matches)) != 1:
            return False
        # Check if the pattern is at the end of the text
        if f"Is the answer correct (Yes/No)? {matches[0]}" not in text[-60:]:
            return False

        if "is correct" in text.lower() and (not output):
            return False
        elif ("is incorrect" in text.lower() or "is not correct" in text.lower() or
              "is wrong" in text.lower()) and output:
            return False
        return True

    # check if the model gives a consistent answer
    reward_dict["consistent_verification"] = check_consistency(text)
    if not reward_dict["consistent_verification"]:
        reward -= 0.5

    # penalty for short response
    reward_dict["non_short_response"] = len(solution_str) >= 40
    if not reward_dict["non_short_response"]:
        reward -= 0.5
    reward_dict["score"] = reward
    return reward_dict


def test_verification_reward(data_source, solution_str, ground_truth, extra_info=None):
    output = extract_yes_no(solution_str)
    label = eval(ground_truth)
    if output is None:
        res = False
    else:
        res = output == label
    return dict(score=float(res))
