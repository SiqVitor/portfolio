#!/usr/bin/env bash
# Fraud Detection Demo — single-command entry point
#
# Usage:
#   bash fraud_detection/demo/run_demo.sh
#
# Produces fraud_detection/demo/results/:
#   synthetic_data.csv      — 10,000 synthetic transactions
#   summary.json            — evaluation metrics (ROC-AUC, Brier, ECE, etc.)
#   calibration_curve.png   — reliability diagram

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Step 1/2: Generating synthetic dataset ==="
python "$SCRIPT_DIR/generate_synthetic.py"

echo ""
echo "=== Step 2/2: Training model and evaluating ==="
python "$SCRIPT_DIR/train_model.py"

echo ""
echo "=== Demo complete ==="
echo "Results in: $SCRIPT_DIR/results/"
