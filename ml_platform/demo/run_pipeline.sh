#!/usr/bin/env bash
# ML Platform Pipeline Demo — single-command entry point
#
# Usage:
#   bash ml_platform/demo/run_pipeline.sh
#
# Produces ml_platform/demo/results/:
#   metrics.json            — evaluation metrics + champion selection
#   validation_report.json  — data quality checks
#   model_registry/v1/      — versioned model artifact + metadata

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== ML Platform Pipeline Demo ==="
echo "Validate → Engineer → Train → Evaluate → Register"
echo ""

python "$SCRIPT_DIR/pipeline.py"

echo ""
echo "=== Demo complete ==="
echo "Results in: $SCRIPT_DIR/results/"
