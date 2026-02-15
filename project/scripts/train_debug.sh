#!/bin/bash
# Train HuBERT ASR model with debug configuration
# Quick test run with minimal resources

set -e
cd "$(dirname "$0")/.."

python -m src.train --config configs/debug.yaml "$@"
