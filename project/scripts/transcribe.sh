#!/bin/bash
# Transcribe audio files with HuBERT ASR model
# Usage: ./scripts/transcribe.sh <model_path> <audio_file(s)>

set -e
cd "$(dirname "$0")/.."

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <model_path> <audio_file> [audio_file2 ...]"
    echo "Example: $0 outputs/hubert-gpu/final sample.wav"
    exit 1
fi

MODEL=$1
shift

python -m src.transcribe --model "$MODEL" --audio "$@"
