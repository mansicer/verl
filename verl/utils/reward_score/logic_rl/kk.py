import re
from typing import Dict, Tuple, Optional

def parse_solution_text_format(solution_text: str) -> Dict[str, str]:
    """Parses ground truth solution text into status dictionary.
    
    Args:
        solution_text: Formatted solution text from dataset
        
    Returns:
        Dictionary mapping character names to their roles (knight/knave)
    """
    status_dict = {}
    # print("\n[Ground Truth Parsing]")
    
    for line in solution_text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        match = re.search(r'\b([A-Za-z]+)\b.*?\b(knight|knave)\b', line, re.IGNORECASE)
        if match:
            name, role = match.groups()
            status_dict[name] = role.lower()
            # print(f"  Found: {name} → {role}")
        else:
            print(f"  [Warning] Unparseable line: '{line}'")

    return status_dict


def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
    # Extract final answer using XML-style tags
    answer_pattern = re.compile(r'CONCLUSION:\n((?:.*\n?)*)', re.DOTALL)
    matches = answer_pattern.findall(solution_str)

    if not matches:
        print("[Error] No valid answer tags found")
        return None
        
    final_answer = matches[-1].strip()
    return final_answer


def parse_model_answer(answer_text: str, expected_names: list) -> Optional[Dict[str, str]]:
    """Parses model's answer text into status dictionary.
    
    Args:
        answer_text: Text extracted from model's <answer> tags
        expected_names: List of character names requiring identification
        
    Returns:
        Dictionary mapping character names to predicted roles, or None if incomplete
    """
    status_dict = {}
    # print("\n[Model Answer Parsing]")
    # print(f"  Expected characters: {expected_names}")

    knight_count = answer_text.lower().count('knight')
    knave_count = answer_text.lower().count('knave')

    # print(f"  Number of predicted roles: {knight_count + knave_count}")
    if knight_count + knave_count != len(expected_names):
        print(f"  [Error] Number of characters mismatch: {knight_count + knave_count} != {len(expected_names)}")
        return None

    for name in expected_names:
        pattern = re.compile(
            rf'\b{re.escape(name)}\b\s+is\s+a\s+\b(knight|knave)\b', 
            re.IGNORECASE
        )
        match = pattern.search(answer_text)
        
        if match:
            role = match.group(1).lower()
            status_dict[name] = role
        else:
            print(f"  [Error] Missing identification for {name}")
            return None
    
    return status_dict


def compute_kk_score(solution_str, ground_truth, reward_info):
    gt_dict = parse_solution_text_format(ground_truth)
    solution = extract_solution(solution_str)
    if solution is None:
        return False
    model_answer = parse_model_answer(solution, gt_dict.keys())
    if model_answer is None:
        return False
    return model_answer == gt_dict
