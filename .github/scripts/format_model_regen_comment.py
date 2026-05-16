#!/usr/bin/env python3
"""
Generate PR comment for model regeneration requirements.

This script generates a formatted GitHub comment with instructions
for model regeneration when model-affecting changes are detected.

Usage:
    python .github/scripts/format_model_regen_comment.py <comma-separated-patterns>

Environment Variables:
    MATCHED_PATTERNS: Comma-separated list of matched patterns

Output:
    Formatted markdown comment for GitHub PR
"""

import sys
import textwrap


def generate_comment(patterns: list[str]) -> str:
    """
    Generate markdown comment for model regeneration.

    Args:
        patterns: List of matched pattern strings

    Returns:
        Formatted markdown comment
    """
    pattern_list = "\n".join(f"  - `{p.strip()}`" for p in patterns)

    comment = f"""⚠️ **Model Regeneration Required**

This PR contains changes to model architecture, features, training logic, or hyperparameters:

{pattern_list}

**Action Required After Merge:**

1. After this PR is merged to `main`, regenerate the model:
   ```bash
   python -m src.train_and_convert
   ```

2. Verify the new model works with the smoke tests:
   ```bash
   bash tools/check_ci.sh
   ```

3. Create a model update PR to commit the new `models/imx8-compatible-sepsis.tflite` artifact.

4. Tag the model commit:
   ```bash
   git tag models/$(cat synthea_version.txt)-$(date +%s)
   git push origin models/$(cat synthea_version.txt)-$(date +%s)
   ```

See [pipeline-internals.md](docs/pipeline-internals.md) for full CI/CD model update workflow."""

    return comment


def main():
    """Main entry point."""
    patterns_input = sys.argv[1] if len(sys.argv) > 1 else ""

    # Parse patterns from comma-separated input
    patterns = [p.strip() for p in patterns_input.split(",") if p.strip()]

    if not patterns:
        print("Error: No patterns provided", file=sys.stderr)
        sys.exit(1)

    comment = generate_comment(patterns)
    print(comment)


if __name__ == "__main__":
    main()
