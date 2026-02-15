#!/bin/bash
# Train HuBERT ASR model with Mac configuration
# Optimized for Apple Silicon with MPS

set -e
cd "$(dirname "$0")/.."

python -m src.train --config configs/mac.yaml "$@"
