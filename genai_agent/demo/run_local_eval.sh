#!/usr/bin/env bash
# ARGUS Local Evaluation — runs mock evaluation without API keys
#
# Usage:
#   bash genai_agent/demo/run_local_eval.sh
#
# Produces genai_agent/demo/results/eval_report.json
#
# For live LLM evaluation, create .env with GROQ_API_KEY and GOOGLE_API_KEY.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== ARGUS Mock Evaluation ==="
echo "Running in local simulation mode (no API keys required)"
echo ""

python "$SCRIPT_DIR/mock_evaluator.py"

echo ""
echo "=== Evaluation complete ==="
echo "Results in: $SCRIPT_DIR/results/eval_report.json"
