# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import difflib
import os
import re
import warnings
from typing import TypedDict

from pprint import pprint
from unidiff import PatchedFile, PatchSet
from unidiff.errors import UnidiffParseError

THINK_START = "<think>"
THINK_END = "</think>"
ANSWER_START = "<solution>"
ANSWER_END = "</solution>"

SEARCH_REPLACE_REGEX = r"```.*?\n### (.*)\n<<<<<<< SEARCH\n([\s\S]*?)\n=======\n([\s\S]*?)\n>>>>>>> REPLACE\n```"


class FormatError(Exception):
    pass


def extract_thought_solution(output: str) -> tuple[str, str]:
    """
    Extract the thought and solution from the output. It is expected to have the following format:
    <think>
    ...
    </think>
    <solution>
    ...
    </solution>
    """
    for tag in [THINK_START, THINK_END, ANSWER_START, ANSWER_END]:
        if output.count(tag) != 1:
            raise FormatError(f"count of {tag} is not 1")

    thought = output.split(THINK_START)[1].split(THINK_END)[0].strip()
    answer = output.split(ANSWER_START)[1].split(ANSWER_END)[0].strip()
    if len(thought) == 0:
        raise FormatError("Thought is empty")
    return thought, answer


def parse_search_replace(text: str) -> dict[str, list[tuple[str, str]]]:
    """
    Parse the search/replace blocks from the text.

    Returns:
        A dictionary where the key is the file path and the value is a list of search/replace pairs.
    """
    path_search_replaces: list[tuple[str, str, str]] = re.findall(SEARCH_REPLACE_REGEX, text)
    path_search_replace_dict = dict[str, list[tuple[str, str]]]()
    for path, search, replace in path_search_replaces:
        path_search_replace_dict.setdefault(path, []).append((search, replace))
    return path_search_replace_dict


def generate_unified_diff(
    old_code: str,
    new_code: str,
    n_context: int = 3,
) -> str:
    """Generate a unified diff between two code.

    Args:
        old_code: The original code.
        new_code: The modified code.
        n_context: The number of context lines to show.

    Returns:
        A string representing the unified diff."""

    original_lines = old_code.splitlines()
    modified_lines = new_code.splitlines()

    diff = difflib.unified_diff(
        original_lines,
        modified_lines,
        fromfile="old",
        tofile="new",
        lineterm="",
        n=n_context,
    )
    try:
        next(diff)
        next(diff)
        diff_code = "\n".join(diff)
        return diff_code
    except StopIteration:
        return ""


def apply_code_change(
    code_context: dict[str, str],
    search_replace_dict: dict[str, list[tuple[str, str]]],
    silent: bool = False,
) -> dict[str, str]:
    """
    Apply the search/replace edits to the code context.

    Args:
        code_context: A dictionary containing the file path and the content of the code.
        search_replace_dict: A dictionary mapping the file path to the search/replace edits.
        silent: Whether to suppress the error messages.

    Returns:
        A dictionary containing the file path and the new content of the code.
    """
    new_content_dict = dict[str, str]()
    for path, search_replaces in search_replace_dict.items():
        new_content = "\n" + code_context.get(path, "")
        for search, replace in search_replaces:
            # Ensure search block can be matched
            # "\n" + search to ensure the indentations are correct
            if not silent and len(search) == len(replace) and search == replace:
                raise FormatError("Search and replace blocks are identical")
            search = "\n" + search
            replace = "\n" + replace
            if not silent and search not in new_content:
                raise FormatError(f"Search block not found in the code")
            new_content = new_content.replace(search, replace)
        # Remove the leading "\n"
        new_content_dict[path] = new_content[1:]
    return new_content_dict


def get_normalized_patch(
    code_context: dict[str, str],
    new_content_dict: dict[str, str],
) -> dict[str, str]:
    """
    According to the code context and new content, generate the normalized patch for each file.

    Args:
        code_context: A dictionary containing the file path and the content of the code.
        new_content_dict: A dictionary mapping the file path to the new content of the file.

    Returns:
        A dictionary containing the file path and the normalized patch.
    """
    patch_dict = dict[str, str]()
    for path, new_content in new_content_dict.items():
        old_content = code_context.get(path, "")
        patch = generate_unified_diff(old_content, new_content)
        # Only add the patch if it's not empty
        # NOTE: this should not happen due to the search == replace check in `apply_code_change`
        # but it can occur in general-purpose usages
        if patch:
            patch_dict[path] = patch
    return patch_dict


class ChangeSimilarity(TypedDict):
    path: str
    pred_change: str
    oracle_change: str
    similarity: float


def compute_change_similarities(
    pred_patch: dict[str, str],
    oracle_patch: dict[str, str],
) -> list[ChangeSimilarity]:
    all_file_paths = set(oracle_patch.keys()).union(set(pred_patch.keys()))
    similarities = list[ChangeSimilarity]()
    for path in all_file_paths:
        pred_change = pred_patch.get(path, "")
        oracle_change = oracle_patch.get(path, "")
        if oracle_change == "" or pred_change == "":
            # Both are empty changes, meaning search = replace. We should penalize this to avoid
            # the model predicting empty changes to hack the reward.
            # NOTE: this should not happen due to (1) the search == replace check in `apply_code_change`
            # and (2) the `if patch` check in `get_normalized_patch`.
            change_similarity = 0.0
        else:
            change_similarity = difflib.SequenceMatcher(
                None,
                pred_change,
                oracle_change,
                autojunk=False,
            ).ratio()
        similarities.append(
            ChangeSimilarity(
                path=path,
                pred_change=pred_change,
                oracle_change=oracle_change,
                similarity=change_similarity,
            ))
    return similarities


def calculate_reward(
    code_context: dict[str, str],
    oracle_new_content: dict[str, str],
    pred_new_content: dict[str, str],
) -> tuple[float, dict]:
    """
    Compute the SWE-RL reward given the code context, oracle patch, and the model output.
    Note that this function is a general version of the reward calculation, which can be used
    for code changes in any form, not just search/replace edits. For search/replace edits, use
    `calculate_search_replace_reward`.

    The return value is always within the range of [0, 1].

    Args:
        code_context: path -> original content of the file. It doesn't need to
            contain the entire codebase, only the files that are affected by the oracle patch.
        oracle_new_content: path -> oracle new content of the file after change.
        pred_new_content: path -> predicted new content of the file after change.

    Returns:
        A float value representing the reward, and a dictionary containing some metadata.
    """
    # Obtain a unified diff for each file, for both the predicted and the oracle patch
    oracle_patch = get_normalized_patch(code_context, oracle_new_content)
    pred_patch = get_normalized_patch(code_context, pred_new_content)
    # Calculate the reward based on the similarity between the predicted and the oracle patch
    similarities = compute_change_similarities(pred_patch, oracle_patch)
    # assert len(similarities) > 0
    # This means oracle_patch and pred_patch are both empty, then they are identical and we reward 1.0
    if len(similarities) == 0:
        assert len(oracle_patch) == 0 and len(pred_patch) == 0
        return 1.0, dict(similarities=[])
    reward = sum(map(lambda x: x["similarity"], similarities)) / len(similarities)
    return reward, dict(similarities=similarities)


def calculate_search_replace_reward(
    code_context: dict[str, str],
    oracle_new_content: dict[str, str],
    output: str,
) -> tuple[float, dict]:
    """
    The search/replace version of the reward calculation. It expects the output to contain
    the thought and solution in the following format:
    <think>
    ...
    </think>
    <solution>
    ...
    </solution>

    Args:
        code_context: path -> original content of the file.
        oracle_new_content: path -> oracle new content of the file after change.
        output: The output from the model containing the thought and solution.

    Returns:
        A float value representing the reward, and a dictionary containing some metadata.
    """
    try:
        # Extract the thought and solution from the output
        thought, answer = extract_thought_solution(output)
        # Parse the search/replace edits from the solution
        pred_search_replaces = parse_search_replace(answer)
        if len(pred_search_replaces) == 0:
            raise FormatError("No valid search blocks found")
        # Get the new content of each file after applying the search/replace edits
        pred_new_content = apply_code_change(code_context, pred_search_replaces)
        reward, metadata = calculate_reward(code_context, oracle_new_content, pred_new_content)
        metadata["thought"] = thought
        metadata["answer"] = answer
        return reward, metadata
    except FormatError as e:
        return -1.0, dict(error=str(e))


def get_filelevel_diff(patch_text: str) -> dict[str, str]:
    """
    Convert a unified diff text into a dictionary of file patches.
    """
    try:
        patch = PatchSet(patch_text)
    except UnidiffParseError:
        return {}
    except Exception as e:
        # NOTE: sometimes unidiff throws other exceptions (e.g. UnboundLocalError) than
        # UnidiffParseError, which is unexpected, but we should still handle it.
        warnings.warn(f"Unexpected unidiff parsing error: {str(e)}")
        return {}
    result = dict[str, str]()
    for patchfile in patch:
        patchfile: PatchedFile = patchfile
        if patchfile.is_binary_file:
            # We don't consider binary files
            continue
        if patchfile.is_rename:
            # Add a special header for renamed files
            source_file = patchfile.source_file
            target_file = patchfile.target_file
            if source_file.startswith("a/"):
                source_file = source_file[2:]
            if target_file.startswith("b/"):
                target_file = target_file[2:]
            header = f"rename from {source_file} to {target_file}"
            path = source_file
        else:
            header = ""
            path = patchfile.path
        body = "\n".join(str(hunk).strip() for hunk in patchfile)
        content = header + "\n" + body
        content = content.strip()
        result[path] = content
    return result


def calculate_reward_unidiff(oracle_patches: list[str], pred_patches: list[str]) -> tuple[float, dict]:
    """
    Compute the SWE-RL reward given two sets of unified diffs.

    The return value is always within the range of [0, 1].

    Args:
        oracle_patches: A list of oracle diffs.
        pred_patches: A list of predicted diffs.

    Returns:
        A float value representing the reward, and a dictionary containing some metadata.
    """
    # Calculate the reward based on the similarity between the predicted and the oracle patch
    pred_patch_dict = dict[str, str]()
    oracle_patch_dict = dict[str, str]()

    for patch_text in oracle_patches:
        oracle_patch_dict.update(get_filelevel_diff(patch_text))

    for patch_text in pred_patches:
        pred_patch_dict.update(get_filelevel_diff(patch_text))

    similarities = compute_change_similarities(pred_patch_dict, oracle_patch_dict)
    if len(similarities) == 0:
        assert len(pred_patch_dict) == 0 and len(oracle_patch_dict) == 0
        return 1.0, dict(similarities=[])
    reward = sum(map(lambda x: x["similarity"], similarities)) / len(similarities)
    return reward, dict(similarities=similarities)


def swe_rl_unidiff_score(data_source, solution_str, ground_truth, extra_info=None):

    def extract_patch_content(solution_str):
        patch_content = re.findall(r'<patch>(.*?)</patch>', solution_str, re.DOTALL)
        return patch_content

    patch_content = extract_patch_content(solution_str)

    if len(patch_content) == 0:
        return 0.0
    else:
        solution = f"<patch>{patch_content[-1]}</patch>"
        return calculate_reward_unidiff([ground_truth], [solution])[0]


def swe_rl_search_replace_score(data_source, solution_str, ground_truth, extra_info=None):
    code_dict = {name: content for name, content in zip(extra_info["file_names"], extra_info["file_contents"])}
    oracle_dict = {name: content for name, content in zip(ground_truth[0], ground_truth[1])}
    try:
        thought, answer = extract_thought_solution(solution_str)
        patch = parse_search_replace(answer)
        if len(patch) == 0:
            raise FormatError("No valid search blocks found")
        pred_dict = apply_code_change(code_dict, patch)
        reward, info = calculate_reward(code_dict, oracle_dict, pred_dict)
    except FormatError as e:
        reward, info = 0.0, dict(error=str(e))
        print(f"Format error in swe_rl_search_replace_score: {str(e)}")
    return reward


if __name__ == "__main__":
    # single file
    gt1 = open(os.path.join(os.path.dirname(__file__), "test_cases", "gt1.txt"), "r").read()
    pred1 = open(os.path.join(os.path.dirname(__file__), "test_cases", "pred1.txt"), "r").read()
    pprint(calculate_reward_unidiff([gt1], [pred1])[0])

    # wrong file change
    gt2 = open(os.path.join(os.path.dirname(__file__), "test_cases", "gt2.txt"), "r").read()
    pred2 = open(os.path.join(os.path.dirname(__file__), "test_cases", "pred2.txt"), "r").read()
    pprint(calculate_reward_unidiff([gt2], [pred2])[0])

    # multi files
    gt3 = open(os.path.join(os.path.dirname(__file__), "test_cases", "gt3.txt"), "r").read()
    pred3 = open(os.path.join(os.path.dirname(__file__), "test_cases", "pred3.txt"), "r").read()
    pprint(calculate_reward_unidiff([gt3], [pred3])[0])

    # search replace
    import json
    sample = json.load(open(os.path.join(os.path.dirname(__file__), "test_cases", "search-replace-1.json"), "r"))
    soultion = sample["response"]
    extra_info = sample["extra_info"]
    ground_truth = sample["reward_model"]["ground_truth"]
    pprint(swe_rl_search_replace_score(extra_info, soultion, ground_truth, extra_info))
