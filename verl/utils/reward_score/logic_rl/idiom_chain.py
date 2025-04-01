import re


def evaluate_idiom_chain(answer, ground_truth, **kwargs):
    """评估成语接龙

    Args:
        answer: 模型生成的答案
        ground_truth: 标准答案，是一个列表，包含部分隐藏的成语
        **kwargs: 其他参数，如problem（问题）

    Returns:
        float: 1.0表示正确，0.0表示错误
    """
    try:
        # 从答案中提取\boxed{}中的内容
        boxed_match = re.search(r'\\boxed{([^}]+)}', answer)
        if not boxed_match:
            return 0.0

        boxed_content = boxed_match.group(1)

        # 根据非中文字符分割成语
        response_idioms = re.split(r'[^\u4e00-\u9fff]+', boxed_content)
        # 过滤掉空字符串
        response_idioms = [idiom for idiom in response_idioms if idiom.strip()]

        # 1. 数量匹配：检查回答的成语数量是否与ground_truth相同
        if len(response_idioms) != len(ground_truth):
            return 0.0

        # 2. 逻辑匹配：检查每个成语的最后一个字是否是下一个成语的第一个字
        for i in range(len(response_idioms) - 1):
            if not response_idioms[i] or not response_idioms[i + 1]:
                return 0.0
            if response_idioms[i][-1] != response_idioms[i + 1][0]:
                return 0.0

        # 3. 指定位置字符串相等匹配：检查答案中的非星号字符是否与ground_truth匹配
        for i in range(len(ground_truth)):
            pattern = ground_truth[i]
            response = response_idioms[i]

            # 如果模式长度大于回答长度，直接失败
            if len(pattern) > len(response):
                return 0.0

            # 检查每个非星号位置的字符是否匹配
            for j in range(len(pattern)):
                if pattern[j] != '*' and j < len(response) and pattern[j] != response[j]:
                    return 0.0

        return 1.0

    except Exception as e:
        print(f"Error evaluating idiom_chain: {e}")
        return 0.0


if __name__ == "__main__":
    # 测试用例集合
    test_cases = [
        # 测试用例1: 完全正确的答案
        {
            "name": "完全正确的答案",
            "answer": "xxxxxxxx（表示推理步骤）\\boxed{山盟海誓，誓不两立，立功赎罪，罪大恶极，极乐世界}",
            "ground_truth": ["山盟**", "*不**", "***罪", "*大**", "*乐世*"],
            "expected": 1.0
        },
        # 测试用例2: 成语数量不匹配
        {
            "name": "成语数量不匹配",
            "answer": "\\boxed{山盟海誓，誓不两立，立功赎罪，罪大恶极}",
            "ground_truth": ["山盟**", "*不**", "***罪", "*大**", "*乐世*"],
            "expected": 0.0
        },
        # 测试用例3: 接龙逻辑错误
        {
            "name": "接龙逻辑错误",
            "answer": "\\boxed{山盟海誓，两立不倒，立功赎罪，罪大恶极，极乐世界}",
            "ground_truth": ["山盟**", "*不**", "***罪", "*大**", "*乐世*"],
            "expected": 0.0
        },
        # 测试用例4: 指定位置字符不匹配
        {
            "name": "指定位置字符不匹配",
            "answer": "\\boxed{山盟海誓，誓不两立，立地成佛，佛大慈悲，悲欢离合}",
            "ground_truth": ["山盟**", "*不**", "***罪", "*大**", "*乐世*"],
            "expected": 0.0
        },
        # 测试用例5: 没有\\boxed{}
        {
            "name": "没有\\boxed{}",
            "answer": "山盟海誓，誓不两立，立功赎罪，罪大恶极，极乐世界",
            "ground_truth": ["山盟**", "*不**", "***罪", "*大**", "*乐世*"],
            "expected": 0.0
        },
        # 测试用例6: 空答案
        {
            "name": "空答案",
            "answer": "",
            "ground_truth": ["山盟**", "*不**", "***罪", "*大**", "*乐世*"],
            "expected": 0.0
        },
        # 测试用例7: 另一个正确的例子
        {
            "name": "另一个正确的例子",
            "answer": "\\boxed{洞天福地，地广人稀，稀奇古怪，怪声怪气，气宇轩昂}",
            "ground_truth": ["洞天**", "*广**", "**古*", "**怪*", "*宇轩*"],
            "expected": 1.0
        },
        # 测试用例8: 模式长度大于回答长度
        {
            "name": "模式长度大于回答长度",
            "answer": "\\boxed{山盟海，誓不两立，立功赎罪，罪大恶极，极乐世界}",
            "ground_truth": ["山盟**", "*不**", "***罪", "*大**", "*乐世*"],
            "expected": 0.0
        },
        # 测试用例9: 空成语
        {
            "name": "空成语",
            "answer": "\\boxed{山盟海誓，，立功赎罪，罪大恶极，极乐世界}",
            "ground_truth": ["山盟**", "*不**", "***罪", "*大**", "*乐世*"],
            "expected": 0.0
        },
        # 测试用例10: 多余的空格和标点
        {
            "name": "多余的空格和标点",
            "answer": "\\boxed{山盟海誓, 誓不两立; 立功赎罪! 罪大恶极? 极乐世界.}",
            "ground_truth": ["山盟**", "*不**", "***罪", "*大**", "*乐世*"],
            "expected": 1.0
        },
        # 测试用例11: 多个\\boxed{}，应该只取第一个
        {
            "name": "多个\\boxed{}",
            "answer": "\\boxed{山盟海誓，誓不两立，立功赎罪，罪大恶极，极乐世界} \\boxed{错误答案}",
            "ground_truth": ["山盟**", "*不**", "***罪", "*大**", "*乐世*"],
            "expected": 1.0
        },
        # 测试用例12: 成语中包含数字和英文
        {
            "name": "成语中包含数字和英文",
            "answer": "\\boxed{山盟海誓，誓不两立，立功赎罪123，罪大恶极abc，极乐世界}",
            "ground_truth": ["山盟**", "*不**", "***罪", "*大**", "*乐世*"],
            "expected": 1.0
        },
        # 测试用例13: 使用箭头(→)分隔
        {
            "name": "使用箭头(→)分隔",
            "answer": "\\boxed{洞天福地 → 地广人稀 → 稀奇古怪 → 怪声怪气 → 气宇轩昂}",
            "ground_truth": ["洞天**", "*广**", "**古*", "**怪*", "*宇轩*"],
            "expected": 1.0
        },
        # 测试用例14: 使用逗号无空格分隔
        {
            "name": "使用逗号无空格分隔",
            "answer": "\\boxed{家喻户晓,晓以大害,害群之马,马马虎虎,虎背熊腰}",
            "ground_truth": ["家*户*", "***害", "***马", "*马**", "*背熊*"],
            "expected": 1.0
        },
        # 测试用例15: 使用分号分隔
        {
            "name": "使用分号分隔",
            "answer": "\\boxed{洞天福地;地广人稀;稀奇古怪;怪声怪气;气宇轩昂}",
            "ground_truth": ["洞天**", "*广**", "**古*", "**怪*", "*宇轩*"],
            "expected": 1.0
        },
        # 测试用例16: 使用换行符分隔
        {
            "name": "使用换行符分隔",
            "answer": "\\boxed{洞天福地\n地广人稀\n稀奇古怪\n怪声怪气\n气宇轩昂}",
            "ground_truth": ["洞天**", "*广**", "**古*", "**怪*", "*宇轩*"],
            "expected": 1.0
        },
        # 测试用例17: 使用数字编号
        {
            "name": "使用数字编号",
            "answer": "\\boxed{1. 洞天福地 2. 地广人稀 3. 稀奇古怪 4. 怪声怪气 5. 气宇轩昂}",
            "ground_truth": ["洞天**", "*广**", "**古*", "**怪*", "*宇轩*"],
            "expected": 1.0
        },
        # 测试用例18: 混合分隔符
        {
            "name": "混合分隔符",
            "answer": "\\boxed{洞天福地，地广人稀->稀奇古怪;怪声怪气 气宇轩昂}",
            "ground_truth": ["洞天**", "*广**", "**古*", "**怪*", "*宇轩*"],
            "expected": 1.0
        },
        # 测试用例19: 带解释的答案
        {
            "name": "带解释的答案",
            "answer": "\\boxed{洞天福地(第一个成语)，地广人稀(第二个成语)，稀奇古怪(第三个成语)，怪声怪气(第四个成语)，气宇轩昂(最后一个成语)}",
            "ground_truth": ["洞天**", "*广**", "**古*", "**怪*", "*宇轩*"],
            "expected": 1.0
        },
        # 测试用例20: 使用顿号分隔
        {
            "name": "使用顿号分隔",
            "answer": "\\boxed{洞天福地、地广人稀、稀奇古怪、怪声怪气、气宇轩昂}",
            "ground_truth": ["洞天**", "*广**", "**古*", "**怪*", "*宇轩*"],
            "expected": 1.0
        }
    ]

    # 运行测试用例
    print("开始测试 evaluate_idiom_chain 函数...")
    passed = 0
    for i, test in enumerate(test_cases):
        result = evaluate_idiom_chain(test["answer"], test["ground_truth"])
        status = "通过" if result == test["expected"] else "失败"
        print(f"测试 {i+1} ({test['name']}): {status}")
        if status == "通过":
            passed += 1
        else:
            print(f"  期望: {test['expected']}, 实际: {result}")
            print(f"  答案: {test['answer']}")
            print(f"  标准: {test['ground_truth']}")

    print(f"\n测试结果: {passed}/{len(test_cases)} 通过")

    # 额外测试: 从成语接龙.json中提取的真实例子
    print("\n测试真实例子...")

    # 示例1
    real_example1 = {
        "answer": "经过思考，我发现这个成语接龙序列是：\\boxed{洞天福地，地广人稀，稀奇古怪，怪声怪气，气宇轩昂}",
        "ground_truth": ["洞天**", "*广**", "**古*", "**怪*", "*宇轩*"],
    }
    result1 = evaluate_idiom_chain(real_example1["answer"], real_example1["ground_truth"])
    print(f"真实例子1: {'通过' if result1 == 1.0 else '失败'}")

    # 示例2
    real_example2 = {
        "answer": "分析这个成语接龙序列，我得到：\\boxed{倾巢而出，出乖露丑，丑态百出，出神入化，化险为夷}",
        "ground_truth": ["*巢而*", "**露*", "***出", "***化", "*险为*"],
    }
    result2 = evaluate_idiom_chain(real_example2["answer"], real_example2["ground_truth"])
    print(f"真实例子2: {'通过' if result2 == 1.0 else '失败'}")

    # 示例3 - 故意错误的例子
    real_example3 = {
        "answer": "这个成语接龙序列是：\\boxed{戴高帽子，子虚乌有，有眼无珠，珠光宝气，气宇轩昂}",
        "ground_truth": ["戴高**", "***有", "*色**", "*花**", "*明星*"],
    }
    result3 = evaluate_idiom_chain(real_example3["answer"], real_example3["ground_truth"])
    print(f"真实例子3 (故意错误): {'通过' if result3 == 0.0 else '失败'}")

    # 示例4 - 箭头分隔
    real_example4 = {
        "answer": "成语接龙序列是：\\boxed{家喻户晓 → 晓以大义 → 义不容辞 → 辞旧迎新 → 新陈代谢}",
        "ground_truth": ["家*户*", "***义", "*不容*", "*旧**", "*陈**"],
    }
    result4 = evaluate_idiom_chain(real_example4["answer"], real_example4["ground_truth"])
    print(f"真实例子4 (箭头分隔): {'通过' if result4 == 1.0 else '失败'}")

    # 示例5 - 无空格逗号
    real_example5 = {
        "answer": "经分析,成语接龙序列为:\\boxed{鼎鼎大名,名扬四海,海底捞月,月明星稀,稀奇古怪}",
        "ground_truth": ["*鼎大*", "***海", "*底**", "*明**", "*奇古*"],
    }
    result5 = evaluate_idiom_chain(real_example5["answer"], real_example5["ground_truth"])
    print(f"真实例子5 (无空格逗号): {'通过' if result5 == 1.0 else '失败'}")
