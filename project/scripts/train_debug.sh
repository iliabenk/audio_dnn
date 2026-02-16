#!/bin/bash
# Train HuBERT ASR model with debug configuration
# Quick test run with minimal resources
# Run from repository root: ./project/scripts/train_debug.sh

set -e

# Disable MPS backend (use CPU if no CUDA available)
# This avoids MPS-specific issues like missing CTC loss implementation
export PYTORCH_MPS_ENABLED=0

python -m project.src.train --config project/configs/debug.yaml "$@"
