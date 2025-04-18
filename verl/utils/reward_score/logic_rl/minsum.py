from math_verify import parse, verify


def compute_minsum_score(solution_str, ground_truth, reward_info):
    pred = parse(solution_str)
    gt = parse(f"${ground_truth}$")
    score = verify(pred, gt)
    return score
