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
# from . import gsm8k, math, prime_math, prime_code
from verl.utils.reward_score.math_verify import math_verify_reward, qwen_math_reward
from verl.utils.reward_score.math_verify import train_verification_reward, test_verification_reward
from verl.utils.reward_score.logic_rl import *


def _default_compute_score(data_source, solution_str, ground_truth, extra_info):
    # get the final response
    if "</think>" in solution_str:
        solution_str = solution_str.split("</think>")[-1]

    if data_source == 'openai/gsm8k':
        from . import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ['lighteval/MATH', 'DigitalLearningGmbH/MATH-lighteval']:
        from . import math
        res = math.compute_score(solution_str, ground_truth)

        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source in [
            'numina_aops_forum', 'numina_synthetic_math', 'numina_amc_aime', 'numina_synthetic_amc', 'numina_cn_k12',
            'numina_olympiads'
    ]:
        from . import prime_math
        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ['codecontests', 'apps', 'codeforces', 'taco']:
        from . import prime_code
        res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ['hiyouga/geometry3k']:
        from . import geo3k
        res = geo3k.compute_score(solution_str, ground_truth)

    elif data_source.startswith("math-verify"):
        res = math_verify_reward(data_source, solution_str, ground_truth, extra_info)

    elif data_source.endswith("verification"):
        if data_source.startswith("test"):
            res = test_verification_reward(data_source, solution_str, ground_truth, extra_info)
        else:
            res = train_verification_reward(data_source, solution_str, ground_truth, extra_info)

    elif data_source == "qwen-math":
        res = qwen_math_reward(data_source, solution_str, ground_truth, extra_info)

    elif data_source.startswith("point24"):
        res = compute_point24_score(solution_str, ground_truth, extra_info)
    elif data_source.startswith("minsum"):
        res = compute_minsum_score(solution_str, ground_truth, extra_info)
    elif data_source.startswith("sudoku"):
        res = compute_sudoku_score(solution_str, ground_truth, extra_info)
    elif data_source.startswith("kk"):
        res = compute_kk_score(solution_str, ground_truth, extra_info)

    else:
        raise NotImplementedError(f"Reward score for {data_source} is not implemented")

    if isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
