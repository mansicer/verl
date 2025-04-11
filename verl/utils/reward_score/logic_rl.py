from verl.utils.reward_score.logic_rl import *


def compute_score(data_source, solution_str, ground_truth, extra_info):
    if data_source.startswith("point24"):
        res = compute_point24_score(solution_str, ground_truth, extra_info)
    elif data_source.startswith("minsum"):
        res = compute_minsum_score(solution_str, ground_truth, extra_info)
    elif data_source.startswith("sudoku"):
        res = compute_sudoku_score(solution_str, ground_truth, extra_info)
    elif data_source.startswith("kk"):
        res = compute_kk_score(solution_str, ground_truth, extra_info)
    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
