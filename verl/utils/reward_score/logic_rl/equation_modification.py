import re


def evaluate_equation_modification(answer: str, ground_truth: str, **kwargs) -> float:
    """
    判断模型对一道“修改等式”问题的答案是否正确:
      1. 是否仅修改了一个运算符(+ - *)？
      2. 修改后的等式是否正确(左侧eval == 右侧整数)？
      3. 是否未改动数字（只改动运算符）？
      4. 右侧的数值是否与原式相同？

    Args:
        answer (str): 模型给出的修改后等式, 
                      例如 "2 - 3 + 9 * 8 - 7 - 4 * 1 + 5 - 6 = 59"
        ground_truth (str): 原始等式, 
                            例如 "2 * 3 + 9 * 8 - 7 - 4 * 1 + 5 - 6 = 59"
        **kwargs: 预留，不使用

    Returns:
        float: 1.0 表示答案正确, 0.0 表示答案错误
    """

    # ground_truth / answer 中必须包含 "="
    if "=" not in ground_truth or "=" not in answer:
        return 0.0

    # 拆分原式和答案的左右部分
    gt_left, gt_right_str = ground_truth.split("=")
    ans_left, ans_right_str = answer.split("=")

    gt_left = gt_left.strip()
    gt_right_str = gt_right_str.strip()
    ans_left = ans_left.strip()
    ans_right_str = ans_right_str.strip()

    # 若右侧数值不一致，直接判定为错误
    if gt_right_str != ans_right_str:
        return 0.0

    # 提取数字、运算符
    def tokenize(expr: str):
        """
        返回 (numbers, operators) 两个列表
        例如 "2 * 3 + 9 - 8" -> ([2, 3, 9, 8], ['*', '+', '-'])
        """
        tokens = re.findall(r'\d+|\+|\-|\*', expr)
        nums, ops = [], []
        for t in tokens:
            if t.isdigit():
                nums.append(int(t))
            else:
                ops.append(t)
        return nums, ops

    gt_nums, gt_ops = tokenize(gt_left)
    ans_nums, ans_ops = tokenize(ans_left)

    # 若数字序列不同，直接错误
    if gt_nums != ans_nums:
        return 0.0

    # 若运算符数量不同，直接错误
    if len(gt_ops) != len(ans_ops):
        return 0.0

    # 仅能改动一个运算符
    diff_count = sum(1 for o1, o2 in zip(gt_ops, ans_ops) if o1 != o2)
    if diff_count != 1:
        return 0.0

    # 尝试计算改后左侧的值
    try:
        left_val = eval(ans_left)
    except Exception:
        return 0.0

    # 右侧必须为整数
    try:
        right_val = int(ans_right_str)
    except ValueError:
        return 0.0

    # 对比运算结果
    return 1.0 if left_val == right_val else 0.0


if __name__ == "__main__":
    """
    根据您提供的数据重新构造 30 条测试用例。
    每条包含:
      - problem (原始表达式)
      - answer  (模型给出的“修改后”表达式)
      - correct (TRUE/FALSE) -> expected = 1.0 或 0.0
    """

    # 将 TRUE 对应 1.0，FALSE 对应 0.0
    def bool_to_float(label):
        return 1.0 if label == "TRUE" else 0.0

    test_cases = [
        {
            "name": "Case 1",
            "problem": "2 * 3 + 9 * 8 - 7 - 4 * 1 + 5 - 6 = 59",
            "answer": "2 - 3 + 9 * 8 - 7 - 4 * 1 + 5 - 6 = 59",
            "expected": bool_to_float("TRUE")
        },
        {
            "name": "Case 2",
            "problem": "8 - 1 - 3 * 9 - 4 - 6 - 5 - 2 * 7 = -39",
            "answer": "8 - 1 - 3 * 9 - 4 - 6 + 5 - 2 * 7 = -39",
            "expected": bool_to_float("TRUE")
        },
        {
            "name": "Case 3",
            "problem": "9 + 1 + 3 + 6 * 8 - 7 * 5 + 4 * 2 = 18",
            "answer": "9 + 1 + 3 + 6 * 8 - 7 * 5 - 4 * 2 = 18",
            "expected": bool_to_float("TRUE")
        },
        {
            "name": "Case 4",
            "problem": "6 * 4 - 9 - 8 - 1 * 7 + 3 - 2 * 5 = -21",
            "answer": "6 + 4 - 9 - 8 - 1 * 7 + 3 - 2 * 5 = -21",
            "expected": bool_to_float("TRUE")
        },
        {
            "name": "Case 5",
            "problem": "6 - 7 - 8 - 9 - 5 + 2 - 3 + 4 * 1 = -4",
            "answer": "6 - 7 + 8 - 9 - 5 + 2 - 3 + 4 * 1 = -4",
            "expected": bool_to_float("TRUE")
        },
        {
            "name": "Case 6",
            "problem": "8 - 7 + 9 - 3 + 5 * 6 + 4 + 1 + 2 = 65",
            "answer": "8 - 7 + 9 * 3 + 5 * 6 + 4 + 1 + 2 = 65",
            "expected": bool_to_float("TRUE")
        },
        {
            "name": "Case 7",
            "problem": "5 * 1 - 3 * 4 * 2 + 9 + 7 + 6 + 8 = -7",
            "answer": "5 * 1 - 3 * 4 * 2 - 9 + 7 + 6 + 8 = -7",
            "expected": bool_to_float("TRUE")
        },
        {
            "name": "Case 8",
            "problem": "5 - 2 + 6 + 1 - 9 + 8 + 7 + 4 - 3 = 15",
            "answer": "5 - 2 + 6 - 1 - 9 + 8 + 7 + 4 - 3 = 15",
            "expected": bool_to_float("TRUE")
        },
        {
            "name": "Case 9",
            "problem": "1 + 7 + 6 + 3 - 4 - 2 + 9 - 5 * 8 = -45",
            "answer": "1 + 7 + 6 + 3 - 4 - 2 * 9 - 5 * 8 = -45",
            "expected": bool_to_float("TRUE")
        },
        {
            "name": "Case 10",
            "problem": "4 - 2 - 9 + 3 - 7 + 8 * 6 + 1 - 5 = -63",
            "answer": "4 - 2 - 9 + 3 - 7 - 8 * 6 + 1 - 5 = -63",
            "expected": bool_to_float("TRUE")
        },
        {
            "name": "Case 11",
            "problem": "2 - 8 + 1 + 4 * 6 * 3 - 7 + 5 * 9 = 56",
            "answer": "2 - 8 + 1 + 4 * 6 * 3 - 7 + 5 - 9 = 56",
            "expected": bool_to_float("TRUE")
        },
        {
            "name": "Case 12",
            "problem": "1 - 7 * 4 + 5 - 2 + 3 + 8 + 6 - 9 = -3",
            "answer": "1 - 7 * 4 + 5 - 2 + 3 * 8 + 6 - 9 = -3",
            "expected": bool_to_float("TRUE")
        },
        {
            "name": "Case 13",
            "problem": "2 + 4 * 1 - 8 + 7 * 3 + 6 - 5 * 9 = -62",
            "answer": "2 + 4 * 1 - 8 - 7 * 3 + 6 - 5 * 9 = -62",
            "expected": bool_to_float("TRUE")
        },
        {
            "name": "Case 14",
            "problem": "9 + 5 * 8 + 7 * 1 + 2 + 3 + 4 - 6 = 55",
            "answer": "9 + 5 * 8 + 7 * 1 - 2 + 3 + 4 - 6 = 55",
            "expected": bool_to_float("TRUE")
        },
        {
            "name": "Case 15",
            "problem": "8 - 4 * 5 * 6 + 7 + 9 * 3 + 1 - 2 = -93",
            "answer": "8 - 4 * 5 * 6 - 7 + 9 * 3 + 1 - 2 = -93",
            "expected": bool_to_float("TRUE")
        },
        {
            "name": "Case 16",
            "problem": "9 * 5 + 2 + 4 - 7 * 6 + 3 + 1 + 8 = 17",
            "answer": "9 * 5 - 2 + 4 - 7 * 6 + 3 + 1 + 8 = 17",
            "expected": bool_to_float("TRUE")
        },
        {
            "name": "Case 17",
            "problem": "5 * 8 + 7 + 4 + 1 + 6 + 9 * 2 * 3 = 73",
            "answer": "5 * 8 + 7 + 4 + 1 + 6 + 9 * 2 - 3 = 73",
            "expected": bool_to_float("FALSE")
        },
        {
            "name": "Case 18",
            "problem": "9 + 3 - 2 + 7 + 5 + 1 * 4 + 8 * 6 = 85",
            "answer": "9 + 3 - 2 + 7 + 5 * 1 * 4 + 8 * 6 = 85",
            "expected": bool_to_float("TRUE")
        },
        {
            "name": "Case 19",
            "problem": "6 - 7 - 8 + 9 * 3 - 5 - 1 + 4 + 2 = 10",
            "answer": "6 - 7 - 8 + 9 * 3 - 5 - 1 - 4 + 2 = 10",
            "expected": bool_to_float("TRUE")
        },
        {
            "name": "Case 20",
            "problem": "7 + 2 * 6 + 8 * 5 + 1 + 3 + 4 - 9 = 34",
            "answer": "7 - 2 * 6 + 8 * 5 + 1 + 3 + 4 - 9 = 34",
            "expected": bool_to_float("TRUE")
        },
        {
            "name": "Case 21",
            "problem": "3 * 7 - 2 * 8 - 4 + 9 - 6 - 1 * 5 = -19",
            "answer": "3 * 7 - 2 * 8 - 4 - 9 - 6 - 1 * 5 = -19",
            "expected": bool_to_float("TRUE")
        },
        {
            "name": "Case 22",
            "problem": "3 * 9 + 7 - 8 * 2 + 1 * 6 - 5 + 4 = 8",
            "answer": "3 + 9 + 7 - 8 * 2 + 1 * 6 - 5 + 4 = 8",
            "expected": bool_to_float("TRUE")
        },
        {
            "name": "Case 23",
            "problem": "9 + 2 + 5 + 7 - 3 + 6 - 4 + 8 * 1 = 36",
            "answer": "9 + 2 + 5 + 7 + 3 + 6 - 4 + 8 * 1 = 36",
            "expected": bool_to_float("TRUE")
        },
        {
            "name": "Case 24",
            "problem": "1 + 6 - 8 * 4 * 2 - 9 * 3 - 5 + 7 = -18",
            "answer": "1 + 6 / 8 * 4 * 2 - 9 * 3 - 5 + 7 = -18",
            "expected": bool_to_float("FALSE")
        },
        {
            "name": "Case 25",
            "problem": "3 + 7 + 1 * 8 - 2 - 6 - 4 - 5 - 9 = -39",
            "answer": "3 + 7 + 1 * 8 - 2 - 6 - 4 - 5 * 9 = -39",
            "expected": bool_to_float("TRUE")
        },
        {
            "name": "Case 26",
            "problem": "7 - 4 + 1 - 6 * 3 + 5 + 2 - 8 - 9 = 1",
            "answer": "7 * 4 + 1 - 6 * 3 + 5 + 2 - 8 - 9 = 1",
            "expected": bool_to_float("TRUE")
        },
        {
            "name": "Case 27",
            "problem": "2 + 7 - 4 * 9 + 6 + 5 - 3 * 8 * 1 = -35",
            "answer": "2 * 7 - 4 * 9 + 6 + 5 - 3 * 8 * 1 = -35",
            "expected": bool_to_float("TRUE")
        },
        {
            "name": "Case 28",
            "problem": "9 * 4 + 2 - 6 - 1 + 7 - 3 * 8 - 5 = 57",
            "answer": "9 * 4 + 2 - 6 - 1 + 7 + 3 * 8 - 5 = 57",
            "expected": bool_to_float("TRUE")
        },
        {
            "name": "Case 29",
            "problem": "6 + 1 * 7 - 4 * 2 * 5 - 8 * 9 - 3 = -56",
            "answer": "6 + 1 * 7 - 4 - 2 * 5 - 8 * 9 - 3 = -56",
            "expected": bool_to_float("FALSE")
        },
        {
            "name": "Case 30",
            "problem": "6 * 9 + 3 * 4 - 5 - 8 - 7 - 1 * 2 = 54",
            "answer": "6 * 9 + 3 * 4 + 5 - 8 - 7 - 1 * 2 = 54",
            "expected": bool_to_float("TRUE")
        },
    ]

    print("开始测试 evaluate_equation_modification 函数...")
    passed = 0
    for i, tc in enumerate(test_cases, 1):
        result = evaluate_equation_modification(tc["answer"], tc["problem"])
        status = "通过" if result == tc["expected"] else "失败"
        print(f"测试 {i} ({tc['name']}): {status}")
        if status == "通过":
            passed += 1
        else:
            print(f"  期望: {tc['expected']}, 实际: {result}")
            print(f"  原式: {tc['problem']}")
            print(f"  答案: {tc['answer']}")

    print(f"\n测试结果: {passed}/{len(test_cases)} 通过")
