import difflib

def generate_patch_from_dicts(original_dict, pred_dict):
    patch_lines = []

    # Only generate diffs for files that exist in both
    for file_path in original_dict.keys() & pred_dict.keys():
        old_content = original_dict[file_path].splitlines(keepends=True)
        new_content = pred_dict[file_path].splitlines(keepends=True)
        diff = difflib.unified_diff(
            old_content,
            new_content,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
        )
        patch_lines.extend(diff)

    return "".join(patch_lines)
