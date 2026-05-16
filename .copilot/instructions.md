---
name: CI Check Runner
applyTo: '**'
description: |
  Enables running the tools/check_ci.sh script via natural language prompts like "Run full ci checks" or "Run and fix full ci checks".
  - "Run full ci checks": Runs the CI check script and reports the result.
  - "Run and fix full ci checks": Runs the CI check script, and if issues are found, suggests or applies fixes if possible.
examples:
  - Run full ci checks
  - Run and fix full ci checks
---

# CI Check Runner Instructions

## Purpose
Enable easy execution of the tools/check_ci.sh script using natural language prompts.

## Behaviors
- On prompt "Run full ci checks":
  - Execute tools/check_ci.sh
  - Report the output and status
- On prompt "Run and fix full ci checks":
  - Execute tools/check_ci.sh
  - If issues are found, suggest or apply fixes if possible, then re-run the check

## Notes
- Use the standard shell execution environment
- Output should be shown to the user
- If fixes are not possible, provide guidance
