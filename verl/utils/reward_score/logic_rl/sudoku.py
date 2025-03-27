import re
import numpy as np
from typing import List, Tuple


def parse_sodoku_solution(solution_str: str) -> List[List[int]]:
    """
    Parse a Sudoku solution from a string representation.
    
    This function extracts a 9x9 Sudoku grid from a string that may contain code blocks.
    It searches for code blocks using regex and looks for lines containing 9 digits.
    
    Args:
        solution_str (str): A string containing the Sudoku solution, typically with code blocks
                           formatted with triple backticks (```).
    
    Returns:
        List[List[int]]: A 9x9 grid represented as a list of lists, where each inner list
                        contains 9 integers. Returns an empty list if no valid grid is found.
    
    Example:
        >>> solution = '''```
        ... 8 5 3 | 1 4 2 | 7 6 9
        ... 2 1 4 | 7 6 9 | 3 5 8
        ... 7 9 6 | 3 8 5 | 4 1 2
        ... ------+------+------
        ... 3 6 8 | 4 9 1 | 2 7 5
        ... 1 7 5 | 6 2 8 | 9 3 4
        ... 4 2 9 | 5 7 3 | 1 8 6
        ... ------+------+------
        ... 6 3 2 | 9 5 7 | 8 4 1
        ... 5 8 1 | 2 3 4 | 6 9 7
        ... 9 4 7 | 8 1 6 | 5 2 3
        ... ```'''
        >>> parse_sodoku_solution(solution)
        [[8, 5, 3, 1, 4, 2, 7, 6, 9], [2, 1, 4, 7, 6, 9, 3, 5, 8], ...]
    """
    # Find all code blocks in the solution string
    code_blocks = re.findall(r"```([\s\S]+?)```", solution_str)

    if not code_blocks:
        return []

    # Process each code block to find valid rows
    for block in code_blocks:
        # Use regex to find lines with exactly 9 digits
        # This pattern looks for lines containing exactly 9 digits (possibly with other characters)
        digit_lines = re.findall(
            r"[^\n]*?(\d)[^\n]*?(\d)[^\n]*?(\d)[^\n]*?(\d)[^\n]*?(\d)[^\n]*?(\d)[^\n]*?(\d)[^\n]*?(\d)[^\n]*?(\d)[^\n]*",
            block)

        # If we found at least 9 lines with 9 digits each
        if len(digit_lines) >= 9:
            # Take the first 9 valid lines and convert tuples of digit characters to lists of integers
            grid = [[int(d) for d in row] for row in digit_lines[:9]]
            return grid

    # If we get here, we couldn't find a valid grid
    return []


def validate_sodoku_solution(solution: List[List[int]], problem: List[List[int]]) -> Tuple[bool, dict]:
    """
    Validate a Sudoku solution against the original problem and Sudoku rules.
    
    This function checks whether a proposed solution to a Sudoku puzzle is valid by verifying:
    1. The solution and problem have valid 9x9 dimensions
    2. The solution respects the initial values given in the problem (non-zero values)
    3. Each row contains all numbers 1-9 without repetition
    4. Each column contains all numbers 1-9 without repetition
    5. Each 3x3 box contains all numbers 1-9 without repetition
    
    Args:
        solution (List[List[int]]): A 9x9 grid representing the proposed solution
        problem (List[List[int]]): A 9x9 grid representing the original problem,
                                  where 0 indicates an empty cell
    
    Returns:
        Tuple[bool, dict]: A tuple containing:
            - bool: True if the solution is valid, False otherwise
            - dict: A dictionary with the following keys:
                - info: A message explaining the validation result or the reason for failure
                - num_filled: Number of cells that needed to be filled in the problem
                - num_correct: Number of cells that were correctly filled in the solution
                             (counts partially correct solutions)
    
    Example:
        >>> problem = [[0, 5, 0, 1, 0, 2, 0, 0, 9], ...]
        >>> solution = [[8, 5, 3, 1, 4, 2, 7, 6, 9], ...]
        >>> validate_sodoku_solution(solution, problem)
        (True, {"info": "Valid solution", "num_filled": 45, "num_correct": 45})
    """
    # Calculate number of cells to be filled (zeros in problem)
    num_filled = sum(row.count(0) for row in problem)

    # Initialize number of correctly filled cells
    num_correct = 0

    # Check if solution and problem have valid dimensions
    if len(solution) != 9 or len(problem) != 9:
        return False, {
            "info": "Invalid dimensions: expected 9x9 grid",
            "num_filled": num_filled,
            "num_correct": num_correct
        }

    for i in range(9):
        if len(solution[i]) != 9 or len(problem[i]) != 9:
            return False, {
                "info": f"Invalid row length at row {i}: expected 9 elements",
                "num_filled": num_filled,
                "num_correct": num_correct
            }

    # Check if non-zero values in problem match solution and count correct filled cells
    for i in range(9):
        for j in range(9):
            if problem[i][j] != 0:
                if problem[i][j] != solution[i][j]:
                    return False, {
                        "info":
                            f"Solution doesn't match problem at position ({i}, {j}): expected {problem[i][j]}, got {solution[i][j]}",
                        "num_filled":
                            num_filled,
                        "num_correct":
                            num_correct
                    }
            else:
                # For empty cells in problem, check if the filled value is valid
                value = solution[i][j]
                if value < 1 or value > 9:
                    continue

                # Check row
                row_valid = list(solution[i]).count(value) == 1

                # Check column
                col_values = [solution[r][j] for r in range(9)]
                col_valid = col_values.count(value) == 1

                # Check 3x3 box
                box_row, box_col = i // 3, j // 3
                box_values = []
                for r in range(3):
                    for c in range(3):
                        box_values.append(solution[box_row * 3 + r][box_col * 3 + c])
                box_valid = box_values.count(value) == 1

                # If the value is valid in all three aspects, count it as correct
                if row_valid and col_valid and box_valid:
                    num_correct += 1

    # Check if the entire solution is valid
    valid_solution = True

    # Check row constraints
    for i, row in enumerate(solution):
        if sorted(row) != list(range(1, 10)):
            valid_solution = False
            break

    # Check column constraints if still valid
    if valid_solution:
        for j in range(9):
            column = [solution[i][j] for i in range(9)]
            if sorted(column) != list(range(1, 10)):
                valid_solution = False
                break

    # Check 3x3 box constraints if still valid
    if valid_solution:
        for box_row in range(3):
            for box_col in range(3):
                box_values = []
                for i in range(3):
                    for j in range(3):
                        box_values.append(solution[box_row * 3 + i][box_col * 3 + j])
                if sorted(box_values) != list(range(1, 10)):
                    valid_solution = False
                    break
            if not valid_solution:
                break

    # Return result
    if valid_solution:
        return True, {"info": "Valid solution", "num_filled": num_filled, "num_correct": num_correct}
    else:
        return False, {
            "info": "Invalid solution: contains conflicts",
            "num_filled": num_filled,
            "num_correct": num_correct
        }


def compute_sudoku_score(solution_str, ground_truth, reward_info):
    problem = reward_info['problem_list']
    
    # prevent dataset autoconversion
    if isinstance(problem, np.ndarray):
        problem = [list(row) for row in problem]

    solution = parse_sodoku_solution(solution_str)
    # print("problem", type(problem), problem)
    # print("solution", type(solution), solution)
    if len(solution) == 0:
        return False
    is_correct, info = validate_sodoku_solution(solution, problem)
    return is_correct


if __name__ == "__main__":
    test_cases = [
        {
            "problem": [
                [0, 5, 0, 1, 0, 2, 0, 0, 9],
                [2, 0, 4, 0, 0, 0, 0, 5, 8],
                [0, 9, 6, 0, 8, 5, 4, 1, 2],
                [3, 0, 8, 4, 9, 0, 2, 0, 0],
                [0, 7, 5, 0, 0, 8, 9, 3, 4],
                [0, 0, 0, 0, 7, 0, 1, 8, 6],
                [0, 3, 2, 9, 5, 0, 8, 4, 1],
                [5, 8, 1, 2, 0, 4, 6, 0, 7],
                [0, 4, 7, 0, 1, 6, 5, 2, 3],
            ],
            "solution":
                """The solved Sudoku puzzle is as follows:

```
8 5 3 | 1 4 2 | 7 6 9
2 1 4 | 7 6 9 | 3 5 8
7 9 6 | 3 8 5 | 4 1 2
------+------+------
3 6 8 | 4 9 1 | 2 7 5
1 7 5 | 6 2 8 | 9 3 4
4 2 9 | 5 7 3 | 1 8 6
------+------+------
6 3 2 | 9 5 7 | 8 4 1
5 8 1 | 2 3 4 | 6 9 7
9 4 7 | 8 1 6 | 5 2 3
```

Each row, column, and subgrid contains the numbers 1 through 9 without repetition.

```
8 5 3 | 1 4 2 | 7 6 9
2 1 4 | 7 6 9 | 3 5 8
7 9 6 | 3 8 5 | 4 1 2
------+------+------
3 6 8 | 4 9 1 | 2 7 5
1 7 5 | 6 2 8 | 9 3 4
4 2 9 | 5 7 3 | 1 8 6
------+------+------
6 3 2 | 9 5 7 | 8 4 1
5 8 1 | 2 3 4 | 6 9 7
9 4 7 | 8 1 6 | 5 2 3
```
""",
        },
        {
            "problem": [
                [8, 0, 0, 2, 1, 5, 0, 7, 9],
                [0, 2, 7, 0, 0, 9, 1, 8, 0],
                [0, 9, 6, 8, 4, 0, 2, 3, 5],
                [3, 0, 0, 4, 0, 6, 7, 5, 0],
                [7, 6, 4, 0, 0, 8, 9, 2, 0],
                [2, 8, 0, 0, 9, 3, 4, 6, 0],
                [0, 5, 0, 9, 3, 4, 0, 1, 0],
                [4, 0, 0, 5, 8, 2, 3, 0, 6],
                [9, 0, 8, 6, 0, 0, 5, 0, 2],
            ],
            "solution":
                """Here is the step-by-step solution to the Sudoku puzzle:

1. **Fill in Box1:**
   - Cell (1,2) and (1,3) are determined by eliminating existing numbers and checking column constraints.
   - Cell (1,2) = 3, Cell (1,3) = 4.

2. **Complete Box3:**
   - Cell (1,7) = 6, Cell (2,9) = 4.

3. **Solve Box4:**
   - Cell (4,2) = 1, Cell (4,3) = 9, Cell (6,3) = 5.

4. **Fill Box5:**
   - Cell (4,5) = 2, Cell (5,4) = 1, Cell (5,5) = 5, Cell (6,4) = 7.

5. **Complete Box6:**
   - Cell (4,9) = 8, Cell (5,9) = 3, Cell (6,9) = 1.

6. **Solve Box7:**
   - Cell (7,1) = 6, Cell (7,3) = 2, Cell (8,2) = 3, Cell (9,2) = 7.

7. **Fill Box8:**
   - Cell (9,5) = 7, Cell (9,6) = 1, Cell (9,8) = 3.

8. **Complete Box9:**
   - Cell (7,7) = 7, Cell (7,9) = 8, Cell (8,8) = 9.

The final solved Sudoku puzzle is:

```
┌───────┬───────┬───────┐
│ 8 3 4 │ 2 1 5 │ 6 7 9 │
│ 5 2 7 │ 3 6 9 │ 1 8 4 │
│ 1 9 6 │ 8 4 7 │ 2 3 5 │
├───────┼───────┼───────┤
│ 3 1 9 │ 4 2 6 │ 7 5 8 │
│ 7 6 4 │ 1 5 8 │ 9 2 3 │
│ 2 8 5 │ 7 9 3 │ 4 6 1 │
├───────┼───────┼───────┤
│ 6 5 2 │ 9 3 4 │ 7 1 8 │
│ 4 3 1 │ 5 8 2 │ 3 9 6 │
│ 9 7 8 │ 6 7 1 │ 5 3 2 │
└───────┴───────┴───────┘
```

However, upon closer inspection, there seems to be a conflict in Row9 where the number 7 appears twice. This indicates an error in the solving process. A correct solution should not have duplicates in any row, column, or box. Therefore, the final solution needs to be adjusted to ensure all constraints are met without conflicts.
""",
        },
    ]

    # Test the functions with the provided test cases
    for i, test_case in enumerate(test_cases):
        print(f"Test Case {i+1}:")
        problem = test_case["problem"]
        solution_str = test_case["solution"]

        # Parse solution
        parsed_solution = parse_sodoku_solution(solution_str)
        print(f"Parsed solution grid:")
        for row in parsed_solution:
            print(row)

        # Validate solution
        is_valid, result = validate_sodoku_solution(parsed_solution, problem)
        score = compute_sudoku_score(solution_str, problem, {"problem_list": problem})
        print(f"Solution valid: {is_valid}")
        print(f"Result: {result}")
        print(f"Score: {score}")
        print("-" * 50)
