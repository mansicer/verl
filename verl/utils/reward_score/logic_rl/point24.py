import sympy
from math_verify import parse


def eval_sympy_expr(expr, target, numbers):
    """
    Evaluate if a sympy expression equals a target value.
    If the expression is an equality, use its left-hand side for comparison.
    
    Args:
        expr: A sympy expression or equality
        target: The target value to compare against
        numbers: List of numbers that should be used in the expression
        
    Returns:
        bool: True if the expression equals the target, False otherwise
    """

    # If expr is an equality, extract the left-hand side
    if isinstance(expr, sympy.core.relational.Equality):
        expr = expr.lhs

    # Try to simplify both expressions before comparison
    expr_simplified = sympy.simplify(expr)
    target_simplified = sympy.simplify(target)

    # Check if they are equal
    equality_check = expr_simplified == target_simplified

    # Check if all numbers in the list are used exactly once
    if equality_check:
        # Get all number literals in the expression
        atoms = [atom for atom in expr.atoms(sympy.Number) if atom.is_Integer]

        # Convert sympy numbers to Python integers for comparison
        expr_numbers = [int(atom) for atom in atoms]

        # Create a list to track which numbers from the input list have been used
        used_numbers = []

        # Check if each number in expr_numbers is in the original numbers list or its negative
        for num in expr_numbers:
            if num in numbers and num not in used_numbers:
                used_numbers.append(num)
            elif -num in numbers and -num not in used_numbers:
                used_numbers.append(-num)

        # Check if all numbers from the input list are used exactly once
        numbers_check = sorted(used_numbers) == sorted(numbers) and len(used_numbers) == len(numbers)

        return equality_check and numbers_check

    return False


def compute_point24_score(solution_str, ground_truth, reward_info):
    pred = parse(solution_str)
    if not reward_info["solvable"]:
        return "无解" in pred or "NO SOLUTION" in pred

    numbers = reward_info['numbers']
    if isinstance(numbers, str):
        numbers = eval(numbers)
    target = reward_info['target']

    # Initialize score
    score = False

    # Try to evaluate each prediction
    for p in pred:
        # Check if p is a valid sympy expression
        if isinstance(p, (sympy.Expr, sympy.core.relational.Equality)):
            # Evaluate if the prediction matches the target
            score = eval_sympy_expr(p, target, numbers)
            if score:  # If we found a correct prediction, we can break
                break
    return score
