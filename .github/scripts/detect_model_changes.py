#!/usr/bin/env python3
"""
Detect model-affecting changes in src/train_and_convert.py.

This script analyzes git diffs and identifies changes that require
model regeneration based on predefined patterns.

Usage:
    python .github/scripts/detect_model_changes.py [base_branch] [head_branch]

Environment Variables:
    BASE_BRANCH: Git branch to compare against (default: origin/main)
    HEAD_BRANCH: Git branch to compare (default: HEAD)

Output:
    JSON format with keys:
    - model_changed: boolean
    - matched_patterns: list of matched patterns
    - message: human-readable summary
"""

import subprocess
import json
import sys
from pathlib import Path

# Patterns that require model regeneration
MODEL_AFFECTING_PATTERNS = [
    "def compute_stats",  # Feature engineering logic
    "feature_cols",  # Feature selection/order
    "def build_model",  # Model architecture
    "keras.layers",  # Layer changes
    "learning_rate",  # Hyperparameter: learning rate
    "BATCH_SIZE",  # Hyperparameter: batch size
    "EPOCHS",  # Hyperparameter: epochs
    "optimizer",  # Optimizer changes
    "loss=",  # Loss function changes
    "BinaryCrossentropy",  # Loss function config
    "class_weight",  # Training logic
    "validation_split",  # Training data handling
    "tf.lite.Optimize",  # Quantization changes
    "inference_input_type",  # Quantization config
    "inference_output_type",  # Quantization config
]


def get_git_diff(base_branch: str, head_branch: str) -> str:
    """
    Get git diff between base and head branches.

    Args:
        base_branch: Base branch to compare against
        head_branch: Head branch to compare

    Returns:
        Diff content as string
    """
    try:
        result = subprocess.run(
            ["git", "diff", f"{base_branch}...{head_branch}", "--", "src/train_and_convert.py"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout
    except Exception as e:
        print(f"Error getting git diff: {e}", file=sys.stderr)
        return ""


def check_for_model_changes(diff_content: str) -> tuple[bool, list[str]]:
    """
    Check if diff contains model-affecting changes.

    Args:
        diff_content: Git diff content

    Returns:
        Tuple of (model_changed, matched_patterns)
    """
    matched_patterns = []

    for pattern in MODEL_AFFECTING_PATTERNS:
        if pattern in diff_content:
            matched_patterns.append(pattern)

    return len(matched_patterns) > 0, matched_patterns


def main():
    """Main entry point."""
    # Get branch arguments or use environment variables
    base_branch = sys.argv[1] if len(sys.argv) > 1 else "origin/main"
    head_branch = sys.argv[2] if len(sys.argv) > 2 else "HEAD"

    # For GitHub Actions, override with environment variables if set
    base_branch = subprocess.os.environ.get("BASE_BRANCH", base_branch)
    head_branch = subprocess.os.environ.get("HEAD_BRANCH", head_branch)

    print(f"Comparing {base_branch}...{head_branch} in src/train_and_convert.py")
    print()

    # Get the diff
    diff_content = get_git_diff(base_branch, head_branch)

    if diff_content:
        print("=== Changes detected ===")
        print(diff_content[:500])  # Print first 500 chars for debugging
        if len(diff_content) > 500:
            print(f"... ({len(diff_content)} chars total)")
        print()
    else:
        print("No changes found in src/train_and_convert.py")

    # Check for model-affecting changes
    model_changed, matched_patterns = check_for_model_changes(diff_content)

    # Prepare output
    output = {
        "model_changed": model_changed,
        "matched_patterns": matched_patterns,
        "message": (
            f"Model regeneration required: {len(matched_patterns)} pattern(s) detected"
            if model_changed
            else "No model regeneration required"
        ),
    }

    # Print as JSON for GitHub Actions to parse
    print(json.dumps(output))

    # Also print human-readable output
    print()
    if model_changed:
        print(f"✓ Model-affecting changes detected ({len(matched_patterns)} pattern(s)):")
        for pattern in matched_patterns:
            print(f"  - {pattern}")
    else:
        print("✓ No model-affecting changes detected")

    return 0 if model_changed else 1  # Exit 0 if changes found, 1 otherwise


if __name__ == "__main__":
    sys.exit(main())
