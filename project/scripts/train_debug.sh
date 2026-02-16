#!/bin/bash
# Train HuBERT ASR model with debug configuration
# Quick test run with minimal resources

set -e
cd "$(dirname "$0")/.."

# Disable MPS backend (use CPU if no CUDA available)
# This avoids MPS-specific issues like missing CTC loss implementation
export PYTORCH_MPS_ENABLED=0

python -m src.train --config configs/debug.yaml "$@"
