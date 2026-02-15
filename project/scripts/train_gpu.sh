#!/bin/bash
# Train HuBERT ASR model with GPU configuration
# Optimized for large GPUs (24GB+ VRAM)

set -e
cd "$(dirname "$0")/.."

python -m src.train --config configs/gpu.yaml "$@"
