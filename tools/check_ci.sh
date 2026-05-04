#!/usr/bin/env bash
# Run local quality gates with optional auto-fix before validation.
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

MODE="fix"
TOTAL=0
FAILED=0

usage() {
  cat <<'EOF'
Usage: tools/check_ci.sh [--fix|--check-only] [--help]

Modes:
  --fix         Apply safe auto-fixes (black + isort) first, then run all checks (default)
  --check-only  Run checks only (no modifications)
  --help        Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fix)
      MODE="fix"
      shift
      ;;
    --check-only)
      MODE="check"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo -e "${RED}Unknown argument: $1${NC}"
      usage
      exit 2
      ;;
  esac
done

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo -e "${RED}Missing required command: $1${NC}"
    echo -e "Install development dependencies first: pip install -r requirements-dev.txt"
    exit 2
  fi
}

run_step() {
  local title="$1"
  local cmd="$2"
  TOTAL=$((TOTAL + 1))

  echo -e "\n${BLUE}== ${title} ==${NC}"
  if eval "$cmd"; then
    echo -e "${GREEN}PASS${NC}"
  else
    echo -e "${RED}FAIL${NC}"
    FAILED=$((FAILED + 1))
  fi
}

require_cmd black
require_cmd isort
require_cmd flake8
require_cmd mypy
require_cmd pytest

echo -e "${BLUE}Running local quality gates (mode: ${MODE})${NC}"

if [[ "$MODE" == "fix" ]]; then
  echo -e "\n${YELLOW}Applying safe auto-fixes before checks...${NC}"
  black src tests
  isort src tests
fi

run_step "Black formatting" "black --check src tests"
run_step "isort import order" "isort --check-only src tests"
run_step "Flake8 lint" "flake8 src tests --max-line-length=100"
run_step "MyPy type check" "mypy src --ignore-missing-imports"
run_step "Pytest" "pytest tests/ -v"

echo -e "\n${BLUE}Summary: $((TOTAL - FAILED))/${TOTAL} checks passed${NC}"

if [[ $FAILED -ne 0 ]]; then
  echo -e "${RED}Quality gates failed.${NC}"
  echo -e "${YELLOW}Auto-fix applied where safe (formatting/import order). Remaining failures require code changes.${NC}"
  exit 1
fi

echo -e "${GREEN}All checks passed.${NC}"
