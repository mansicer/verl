import re
import sympy
from sympy import sympify
from math_verify import parse, verify


def extract_boxed_content(response):
    match = re.search(r"\\boxed{(.*?)}", response)
    if match:
        pred = match.group(1).strip()
        if "=" in pred:
            pred = pred.split("=")[0].strip()
        return pred
    return None


def problem24_answer_checker(prediction, numbers, target):

    def answer_checker_problem24(prediction):
        try:
            prediction = (prediction.replace("\\times", "*").replace("×", "*").replace("÷", "/").replace(
                "\\div", "/").replace("\\cdot", "*").replace("\\frac{", "(").replace("}{", ")/(").replace("}", ")"))

            sympy_expr = sympify(prediction)
            result = sympy_expr.evalf()
            is_equal_target_num = float(result) == target
            return is_equal_target_num
        except Exception as e:
            print(f"Error when parsing {prediction}: {e}")
            return False

    def validate_prediction(prediction, inputs_list):
        prediction_numbers = re.findall(r"\b\d+\b", prediction)
        pred_count = {}
        input_count = {}

        for num in prediction_numbers:
            num = int(num)
            pred_count[num] = pred_count.get(num, 0) + 1

        for num in inputs_list:
            input_count[num] = input_count.get(num, 0) + 1

        if pred_count != input_count:
            return False

        return True

    if not validate_prediction(prediction, numbers):
        return False

    result = answer_checker_problem24(prediction)
    return result


def compute_point24_score(solution_str, ground_truth, reward_info):
    prediction = extract_boxed_content(solution_str)
    print(prediction)
    if prediction is None:
        return False

    if not reward_info["solvable"]:
        return "无解" in prediction or "NO SOLUTION" in prediction

    target = reward_info['target']
    numbers = reward_info['numbers']
    if isinstance(numbers, str):
        numbers = eval(numbers)

    # Initialize score
    score = problem24_answer_checker(prediction, numbers, target)
    return score


if __name__ == "__main__":
    question = {"numbers": [2, 4, 4, 8], "target": 36, "solvable": True}
    solution = r"""\boxed{8 \times (4 + (2 \div 4)) = 36}"""
    solution = r"""\boxed{8 × (4 + (2 ÷ 4)) = 36}"""
    print(compute_point24_score(solution, question, question))
