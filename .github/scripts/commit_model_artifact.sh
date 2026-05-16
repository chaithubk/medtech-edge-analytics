#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="models/imx8-compatible-sepsis.tflite"

write_output() {
  local key="$1"
  local value="$2"
  if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    echo "${key}=${value}" >> "${GITHUB_OUTPUT}"
  fi
}

if git diff --quiet "$MODEL_PATH"; then
  echo "Model unchanged, skipping commit."
  write_output "changed" "false"
else
  echo "Model changed, committing..."
  git add "$MODEL_PATH"

  run_id="${GITHUB_RUN_ID:-local}"
  event_name="${GITHUB_EVENT_NAME:-manual}"
  server_url="${GITHUB_SERVER_URL:-https://github.com}"
  repository="${GITHUB_REPOSITORY:-local/repo}"
  synthea_version="unknown"
  if [[ -f "synthea_version.txt" ]]; then
    synthea_version="$(cat synthea_version.txt)"
  fi

  git commit -m "ci: update sepsis model from training run ${run_id}

Synthea version: ${synthea_version}
Trigger: ${event_name}
Workflow: ${server_url}/${repository}/actions/runs/${run_id}"

  write_output "changed" "true"
fi
