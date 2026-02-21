#!/bin/bash
# Tune beam search + LM parameters for optimal WER
# Usage: ./project/scripts/tune_lm_params.sh

set -e

MODEL="${MODEL:-project/final_prev}"
LM_PATH="${LM_PATH:-project/lm/4-gram-lower.arpa}"
SPLIT="${SPLIT:-test.clean}"
OUTPUT_DIR="${OUTPUT_DIR:-project/outputs/lm_tuning}"

mkdir -p "$OUTPUT_DIR"
RESULTS_FILE="$OUTPUT_DIR/tuning_results.csv"

echo "=== LM Parameter Tuning ==="
echo "Model: $MODEL"
echo "LM: $LM_PATH"
echo "Split: $SPLIT"
echo "Results: $RESULTS_FILE"
echo ""

# Initialize results file
echo "beam_width,alpha,beta,wer" > "$RESULTS_FILE"

# Parameter ranges
BEAM_WIDTHS=(50 100 200)
ALPHAS=(0.3 0.5 0.7 1.0)
BETAS=(0.5 1.0 1.5 2.0)

total=$((${#BEAM_WIDTHS[@]} * ${#ALPHAS[@]} * ${#BETAS[@]}))
current=0

for beam_width in "${BEAM_WIDTHS[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
        for beta in "${BETAS[@]}"; do
            current=$((current + 1))
            echo "[$current/$total] beam_width=$beam_width, alpha=$alpha, beta=$beta"

            # Run evaluation and capture output
            output=$(accelerate launch -m project.src.evaluate \
                --model "$MODEL" \
                --lm_path "$LM_PATH" \
                --beam_width "$beam_width" \
                --alpha "$alpha" \
                --beta "$beta" \
                --splits "$SPLIT" 2>&1)

            # Extract WER from output (looks for "WER = X.XX%")
            wer=$(echo "$output" | grep -oP "WER = \K[0-9]+\.[0-9]+" | tail -1)

            if [[ -n "$wer" ]]; then
                echo "  WER: $wer%"
                echo "$beam_width,$alpha,$beta,$wer" >> "$RESULTS_FILE"
            else
                echo "  Failed to extract WER"
                echo "$beam_width,$alpha,$beta,ERROR" >> "$RESULTS_FILE"
            fi
            echo ""
        done
    done
done

echo "=== Tuning Complete ==="
echo ""

# Find best result
echo "Top 5 configurations:"
sort -t',' -k4 -n "$RESULTS_FILE" | head -6 | tail -5

echo ""
echo "Full results saved to: $RESULTS_FILE"
