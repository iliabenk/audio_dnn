#!/bin/bash
# Train HuBERT ASR model with Mac configuration
# Optimized for Apple Silicon with MPS
# Run from repository root: ./project/scripts/train_mac.sh

set -e

python -m project.src.train --config project/configs/mac.yaml "$@"
