#!/bin/bash
# Transcribe audio files with HuBERT ASR model
# Run from repository root: ./project/scripts/transcribe.sh <model_path> <audio_file(s)>

set -e

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <model_path> <audio_file> [audio_file2 ...]"
    echo "Example: $0 project/outputs/hubert-gpu/final sample.wav"
    exit 1
fi

MODEL=$1
shift

python -m project.src.transcribe --model "$MODEL" --audio "$@"
