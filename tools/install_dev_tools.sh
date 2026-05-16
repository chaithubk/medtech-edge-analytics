#!/usr/bin/env bash
set -euo pipefail

# Install actionlint
if ! command -v actionlint >/dev/null 2>&1; then
  echo "Installing actionlint..."
  version="1.7.7"
  curl -fsSL "https://github.com/rhysd/actionlint/releases/download/v${version}/actionlint_${version}_linux_amd64.tar.gz" \
    | tar -xz -C /usr/local/bin actionlint
fi

# Install shellcheck
if ! command -v shellcheck >/dev/null 2>&1; then
  echo "Installing shellcheck..."
  apt-get update && apt-get install -y shellcheck
fi

echo "actionlint and shellcheck are installed."
