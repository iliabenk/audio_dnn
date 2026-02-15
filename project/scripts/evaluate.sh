#!/bin/bash
# Evaluate HuBERT ASR model on test splits
# Usage: ./scripts/evaluate.sh <model_path> [--splits test.clean test.other]

set -e
cd "$(dirname "$0")/.."

if [ -z "$1" ]; then
    echo "Usage: $0 <model_path> [additional args...]"
    echo "Example: $0 outputs/hubert-gpu/final --splits test.clean test.other"
    exit 1
fi

python -m src.evaluate --model "$@"
