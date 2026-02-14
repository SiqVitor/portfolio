#!/usr/bin/env bash
# Real-Time ML System Demo — single-command entry point
#
# Usage:
#   bash realtime_ml_system/demo/run_demo.sh
#
# Produces realtime_ml_system/demo/results/:
#   summary.json — events processed, latency stats, AUC, throughput

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Real-Time ML System Demo ==="
echo "Batch training → online inference → latency metrics"
echo ""

python "$SCRIPT_DIR/online_inference.py"

echo ""
echo "=== Demo complete ==="
echo "Results in: $SCRIPT_DIR/results/summary.json"
