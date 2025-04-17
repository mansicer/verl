import re
import json
import io
from typing import List, Dict, Tuple, Any
from pathlib import Path

from unidiff import PatchSet, UnidiffParseError


def apply_patch_to_code_dict(code_dict: Dict[str, str], patch_text: str) -> Dict[str, str]:
    """Applies a unidiff patch to the given code dictionary.
    
    Args:
        code_dict: Dictionary where keys are file paths and values are file contents
        patch_text: Unified diff patch in string format
        
    Returns:
        Updated code_dict after applying the patch
        
    Raises:
        ValueError: If a file referenced in the patch is not found in code_dict
    """
    patched_code_dict = code_dict.copy()
    patch_text = patch_text.strip()
    
    # Clean up patch text
    if patch_text.startswith('<patch>'):
        patch_text = patch_text[7:]
    if patch_text.endswith('</patch>'):
        patch_text = patch_text[:-8]
        
    patch = PatchSet(io.StringIO(patch_text))

    for patched_file in patch:
        path = patched_file.path
        original_text = code_dict.get(path)
        if original_text is None:
            raise ValueError(f"File {path} not found in code_dict {code_dict.keys()}.")

        original_lines = original_text.splitlines(keepends=True)
        patched_lines = []
        i = 0  # Pointer for original_lines

        for hunk in patched_file:
            # Add lines before hunk
            while i < hunk.source_start - 1 and i < len(original_lines):
                patched_lines.append(original_lines[i])
                i += 1

            # Apply hunk changes
            for line in hunk:
                if line.is_context:
                    if i < len(original_lines):
                        patched_lines.append(original_lines[i])
                        i += 1
                elif line.is_removed:
                    if i < len(original_lines):
                        i += 1  # Skip the line (removal)
                elif line.is_added:
                    patched_lines.append(line.value)

        # Add remaining lines after last hunk
        patched_lines.extend(original_lines[i:])
        patched_code_dict[path] = ''.join(patched_lines)

    return patched_code_dict