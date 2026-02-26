#!/bin/bash
# Train HuBERT ASR model with GPU configuration
# Optimized for large GPUs (24GB+ VRAM)
# Run from repository root: ./project/scripts/train_gpu.sh

set -e

python -m project.src.train --config project/configs/libri-100h.yaml "$@"
