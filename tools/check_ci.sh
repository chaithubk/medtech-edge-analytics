#!/bin/bash
# Run all CI checks locally, matching the pipeline
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

fail=0

function check() {
  echo -e "\n${GREEN}== $1 ==${NC}"
  if eval "$2"; then
    echo -e "${GREEN}PASS${NC}"
  else
    echo -e "${RED}FAIL${NC}"
    fail=1
  fi
}

# Black formatting
check "Black formatting" "black --check src tests"

# isort import order
check "isort import order" "isort --check-only src tests"

# Flake8 lint
check "Flake8 lint" "flake8 src tests --max-line-length=100"

# MyPy type check
check "MyPy type check" "mypy src --ignore-missing-imports"

# Pytest
check "Pytest" "pytest tests/ -v"

if [[ $fail -ne 0 ]]; then
  echo -e "\n${RED}Some checks failed. See above for details.${NC}"
  echo -e "\nTo auto-fix most issues, run:"
  echo -e "  black src tests"
  echo -e "  isort src tests"
  echo -e "  flake8 src tests --max-line-length=100"
  echo -e "  mypy src --ignore-missing-imports"
  exit 1
else
  echo -e "\n${GREEN}All checks passed!${NC}"
fi
