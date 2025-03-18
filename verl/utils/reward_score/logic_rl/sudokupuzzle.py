import re
from typing import List, Tuple


def parse_sodoku_solution(solution_str: str) -> List[List[int]]:
    # Find all code blocks in the solution string
    code_blocks = re.findall(r"```([\s\S]+?)```", solution_str)

    if not code_blocks:
        return []

    # Process each code block to find valid rows
    for block in code_blocks:
        # Use regex to find lines with exactly 9 digits
        # This pattern looks for lines containing exactly 9 digits (possibly with other characters)
        digit_lines = re.findall(r"[^\n]*?(\d)[^\n]*?(\d)[^\n]*?(\d)[^\n]*?(\d)[^\n]*?(\d)[^\n]*?(\d)[^\n]*?(\d)[^\n]*?(\d)[^\n]*?(\d)[^\n]*", block)

        # If we found at least 9 lines with 9 digits each
        if len(digit_lines) >= 9:
            # Take the first 9 valid lines and convert tuples of digit characters to lists of integers
            grid = [[int(d) for d in row] for row in digit_lines[:9]]
            return grid

    # If we get here, we couldn't find a valid grid
    return []


def validate_sodoku_solution(solution: List[List[int]], problem: List[List[int]]) -> Tuple[bool, str]:
    # Check if solution and problem have valid dimensions
    if len(solution) != 9 or len(problem) != 9:
        return False, "Invalid dimensions: expected 9x9 grid"

    for i in range(9):
        if len(solution[i]) != 9 or len(problem[i]) != 9:
            return False, f"Invalid row length at row {i}: expected 9 elements"

    # Check if non-zero values in problem match solution
    for i in range(9):
        for j in range(9):
            if problem[i][j] != 0 and problem[i][j] != solution[i][j]:
                return False, f"Solution doesn't match problem at position ({i}, {j}): expected {problem[i][j]}, got {solution[i][j]}"

    # Check row constraints
    for i, row in enumerate(solution):
        if sorted(row) != list(range(1, 10)):
            return False, f"Row {i} has duplicate or missing numbers"

    # Check column constraints
    for j in range(9):
        column = [solution[i][j] for i in range(9)]
        if sorted(column) != list(range(1, 10)):
            return False, f"Column {j} has duplicate or missing numbers"

    # Check 3x3 box constraints
    for box_row in range(3):
        for box_col in range(3):
            box_values = []
            for i in range(3):
                for j in range(3):
                    box_values.append(solution[box_row * 3 + i][box_col * 3 + j])
            if sorted(box_values) != list(range(1, 10)):
                return False, f"Box at ({box_row}, {box_col}) has duplicate or missing numbers"

    # All checks passed
    return True, "Valid solution"


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
            "solution": """The solved Sudoku puzzle is as follows:

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
            "solution": """Here is the step-by-step solution to the Sudoku puzzle:

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
        is_valid, reason = validate_sodoku_solution(parsed_solution, problem)
        print(f"Solution valid: {is_valid}")
        print(f"Reason: {reason}")
        print("-" * 50)
