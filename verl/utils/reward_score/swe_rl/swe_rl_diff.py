from verl.utils.reward_score.swe_rl.original import swe_rl_unidiff_score


def compute_score(data_source, solution_str, ground_truth, extra_info):
    return swe_rl_unidiff_score(data_source, solution_str, ground_truth, extra_info)
