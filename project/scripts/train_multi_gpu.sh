#!/bin/bash
# Train HuBERT ASR model with multi-GPU DDP
# Uses all available GPUs with DistributedDataParallel via Accelerate

set -e
cd "$(dirname "$0")/.."

accelerate launch \
    --multi_gpu \
    -m src.train \
    --config configs/gpu.yaml \
    "$@"
