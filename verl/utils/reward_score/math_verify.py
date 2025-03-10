import re
from math_verify import parse, verify


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
