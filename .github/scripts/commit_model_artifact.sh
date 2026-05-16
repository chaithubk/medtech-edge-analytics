#!/usr/bin/env bash
set -e

MODEL_PATH="models/imx8-compatible-sepsis.tflite"

if git diff --quiet "$MODEL_PATH"; then
  echo "Model unchanged, skipping commit."
  echo "changed=false" >> "$GITHUB_OUTPUT"
else
  echo "Model changed, committing..."
  git add "$MODEL_PATH"
  git commit -m "ci: update sepsis model from training run ${{ GITHUB_RUN_ID }}

Synthea version: $(cat synthea_version.txt)
Trigger: ${{ GITHUB_EVENT_NAME }}
Workflow: ${{ GITHUB_SERVER_URL }}/${{ GITHUB_REPOSITORY }}/actions/runs/${{ GITHUB_RUN_ID }}"
  echo "changed=true" >> "$GITHUB_OUTPUT"
fi
