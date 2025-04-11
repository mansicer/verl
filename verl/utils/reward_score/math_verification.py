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
    if data_source.startswith("math-verify"):
        res = math_verify_reward(data_source, solution_str, ground_truth, extra_info)

    elif data_source == "qwen-math":
        res = qwen_math_reward(data_source, solution_str, ground_truth, extra_info)

    elif "verification" in data_source.lower():
        if data_source.startswith("test"):
            res = test_verification_reward(data_source, solution_str, ground_truth, extra_info)
        else:
            res = train_verification_reward(data_source, solution_str, ground_truth, extra_info)

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])


def math_verify_reward(data_source, solution_str, ground_truth, extra_info=None):
    pred = parse(solution_str)
    if "gsm8k" in data_source:
        gt = parse(ground_truth)
    else:
        gt = parse(f"${ground_truth}$")
    res = verify(gt, pred)
    return float(res)


def qwen_math_reward(data_source, solution_str, ground_truth, extra_info=None):
    pred = parse(solution_str)
    gt = parse(f"${ground_truth}$")
    label = verify(gt, pred)
    res = float(label)

    code_blocks = re.findall(r'```python[\s\S]*?```', solution_str)
    if (len(code_blocks) > 0):
        res -= 0.5
    return float(res)


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
    text = solution_str.strip().replace("**", "")
    # Remove "<|im_end|>" if it's at the end of text
    if text.endswith("<|im_end|>"):
        text = text[:-len("<|im_end|>")]

    output = extract_yes_no(text)
    if output is None:
        res = -1.0
    else:
        res = output == ground_truth

    def check_consistency(text: str) -> bool:
        pattern = r"Is the answer correct \(Yes/No\)\?\s+(Yes|No)"
        matches = re.findall(pattern, text)
        if len(matches) != 1:
            return False
        if len(set(matches)) != 1:
            return False
        # Check if the pattern is at the end of the text
        last_match_pos = text.rfind(f"Is the answer correct (Yes/No)? {matches[0]}")
        if last_match_pos == -1 or last_match_pos + len(f"Is the answer correct (Yes/No)? {matches[0]}") != len(text):
            return False

        if "is correct" in text.lower() and (not output):
            return False
        elif ("is incorrect" in text.lower() or "is not correct" in text.lower() or
              "is wrong" in text.lower()) and output:
            return False
        return True

    # check if the model gives a consistent answer
    if not check_consistency(text):
        res = -0.5

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
