#!/bin/bash
# Train HuBERT ASR model on ALL LibriSpeech/Libri-Light subsets sequentially
# Runs from shortest to longest: 10min → 1h → 10h → 100h with multi-GPU DDP
# Each model is saved to its own unique output directory
#
# Run from repository root: ./project/scripts/train_librilight_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================================================"
echo "Training HuBERT on ALL subsets (10min → 1h → 10h → 100h)"
echo "========================================================================"
echo ""
echo "Output directories:"
echo "  - 10min:  ./project/outputs/hubert-librilight-10min"
echo "  - 1h:     ./project/outputs/hubert-librilight-1h"
echo "  - 10h:    ./project/outputs/hubert-librilight-10h"
echo "  - 100h:   ./project/outputs/hubert-libri-100h"
echo ""

# Train on 10min subset (shortest)
echo ""
echo "============================================"
echo "[1/4] Starting Libri-Light 10min training"
echo "============================================"
"${SCRIPT_DIR}/train_librilight_10min.sh" "$@"
echo ""
echo "[1/4] Libri-Light 10min training COMPLETE"
echo ""

# Train on 1h subset
echo ""
echo "=========================================="
echo "[2/4] Starting Libri-Light 1h training"
echo "=========================================="
"${SCRIPT_DIR}/train_librilight_1h.sh" "$@"
echo ""
echo "[2/4] Libri-Light 1h training COMPLETE"
echo ""

# Train on 10h subset
echo ""
echo "=========================================="
echo "[3/4] Starting Libri-Light 10h training"
echo "=========================================="
"${SCRIPT_DIR}/train_librilight_10h.sh" "$@"
echo ""
echo "[3/4] Libri-Light 10h training COMPLETE"
echo ""

# Train on 100h subset (longest)
echo ""
echo "============================================"
echo "[4/4] Starting LibriSpeech 100h training"
echo "============================================"
"${SCRIPT_DIR}/train_multi_gpu.sh" "$@"
echo ""
echo "[4/4] LibriSpeech 100h training COMPLETE"
echo ""

echo "========================================================================"
echo "ALL TRAINING COMPLETE!"
echo "========================================================================"
echo ""
echo "Models saved to:"
echo "  - 10min:  ./project/outputs/hubert-librilight-10min/final"
echo "  - 1h:     ./project/outputs/hubert-librilight-1h/final"
echo "  - 10h:    ./project/outputs/hubert-librilight-10h/final"
echo "  - 100h:   ./project/outputs/hubert-libri-100h/final"
