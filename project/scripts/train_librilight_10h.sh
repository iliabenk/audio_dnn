#!/bin/bash
# Train HuBERT ASR model on Libri-Light 10h subset with multi-GPU DDP
# Uses all available GPUs with DistributedDataParallel via Accelerate
# Run from repository root: ./project/scripts/train_librilight_10h.sh

set -e

echo "=========================================="
echo "Training HuBERT on Libri-Light 10h subset"
echo "=========================================="

accelerate launch \
    --multi_gpu \
    -m project.src.train \
    --config project/configs/librilight-10h.yaml \
    "$@"
