#!/bin/bash
# Evaluate HuBERT ASR model on test splits
# Run from repository root: ./project/scripts/evaluate.sh <model_path> [--splits test.clean test.other]

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <model_path> [additional args...]"
    echo "Example: $0 project/outputs/hubert-gpu/final --splits test.clean test.other"
    exit 1
fi

python -m project.src.evaluate --model "$@"
