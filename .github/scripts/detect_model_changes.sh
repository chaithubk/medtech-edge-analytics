#!/bin/bash
# Parse model detection output and set GitHub Actions outputs
# 
# Usage: bash .github/scripts/detect_model_changes.sh [base_branch] [head_branch]
#
# Reads from detect_model_changes.py and exports GitHub Actions outputs
# If GITHUB_OUTPUT is not set (local testing), outputs to stdout.

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run the Python detection script
BASE_BRANCH="${1:-origin/main}"
HEAD_BRANCH="${2:-HEAD}"

DETECTION_OUTPUT=$(python "$SCRIPT_DIR/detect_model_changes.py" "$BASE_BRANCH" "$HEAD_BRANCH" 2>&1 | tail -1)

echo "Detection output: $DETECTION_OUTPUT"

# Parse JSON output
MODEL_CHANGED=$(echo "$DETECTION_OUTPUT" | python -c "import sys, json; print(json.load(sys.stdin)['model_changed'])" 2>/dev/null || echo "false")
MATCHED_PATTERNS=$(echo "$DETECTION_OUTPUT" | python -c "import sys, json; print(','.join(json.load(sys.stdin)['matched_patterns']))" 2>/dev/null || echo "")

# Set GitHub Actions outputs or print for local testing
if [ -n "$GITHUB_OUTPUT" ]; then
  echo "model_changed=$MODEL_CHANGED" >> "$GITHUB_OUTPUT"
  echo "matched_patterns=$MATCHED_PATTERNS" >> "$GITHUB_OUTPUT"
fi

# Print for debugging
echo ""
echo "=== Output ==="
echo "model_changed=$MODEL_CHANGED"
echo "matched_patterns=$MATCHED_PATTERNS"

exit 0
