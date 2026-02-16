#!/bin/bash
# Train HuBERT ASR model with multi-GPU DDP
# Uses all available GPUs with DistributedDataParallel via Accelerate
# Run from repository root: ./project/scripts/train_multi_gpu.sh

set -e

accelerate launch \
    --multi_gpu \
    -m project.src.train \
    --config project/configs/gpu.yaml \
    "$@"
