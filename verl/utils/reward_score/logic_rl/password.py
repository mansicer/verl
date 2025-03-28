def compute_password_score(solution_str, ground_truth, **kwargs):
    """
    Evaluate the answer to a password problem

    Args:
        solution_str: The answer generated by the model
        ground_truth: The standard answer
        **kwargs: Additional keyword arguments (ignored)

    Returns:
        float: 1.0 means correct, 0.0 means incorrect
    """
    import re

    # Extract content from \boxed{} in the answer
    boxed_matches = re.findall(r'\\boxed\{(.*?)\}', solution_str)
    if boxed_matches:
        # Take the last match if multiple matches exist
        answer = boxed_matches[-1].strip()
        print(answer, ground_truth)
        # Compare the answer with the standard answer (both stripped)
        return 1.0 if answer == ground_truth.strip() else 0.0
    else:
        # If no \boxed{} is found, return 0.0 directly
        return 0.0


if __name__ == "__main__":
    print(
        compute_password_score("每个字母向前移动9位，得到原文为“think like a user”。\n\n\\boxed{think like a user}",
                               "think like a user"))
