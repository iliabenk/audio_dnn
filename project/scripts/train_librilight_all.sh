#!/bin/bash
# Train HuBERT ASR model on ALL Libri-Light subsets sequentially
# Runs 10h, 1h, and 10min training in sequence with multi-GPU DDP
# Each model is saved to its own unique output directory
#
# Run from repository root: ./project/scripts/train_librilight_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "Training HuBERT on ALL Libri-Light subsets (10h, 1h, 10min)"
echo "============================================================"
echo ""
echo "Output directories:"
echo "  - 10h:    ./project/outputs/hubert-librilight-10h"
echo "  - 1h:     ./project/outputs/hubert-librilight-1h"
echo "  - 10min:  ./project/outputs/hubert-librilight-10min"
echo ""

# Train on 10h subset
echo ""
echo "=========================================="
echo "[1/3] Starting Libri-Light 10h training"
echo "=========================================="
"${SCRIPT_DIR}/train_librilight_10h.sh" "$@"
echo ""
echo "[1/3] Libri-Light 10h training COMPLETE"
echo ""

# Train on 1h subset
echo ""
echo "=========================================="
echo "[2/3] Starting Libri-Light 1h training"
echo "=========================================="
"${SCRIPT_DIR}/train_librilight_1h.sh" "$@"
echo ""
echo "[2/3] Libri-Light 1h training COMPLETE"
echo ""

# Train on 10min subset
echo ""
echo "============================================"
echo "[3/3] Starting Libri-Light 10min training"
echo "============================================"
"${SCRIPT_DIR}/train_librilight_10min.sh" "$@"
echo ""
echo "[3/3] Libri-Light 10min training COMPLETE"
echo ""

echo "============================================================"
echo "ALL TRAINING COMPLETE!"
echo "============================================================"
echo ""
echo "Models saved to:"
echo "  - 10h:    ./project/outputs/hubert-librilight-10h/final"
echo "  - 1h:     ./project/outputs/hubert-librilight-1h/final"
echo "  - 10min:  ./project/outputs/hubert-librilight-10min/final"
