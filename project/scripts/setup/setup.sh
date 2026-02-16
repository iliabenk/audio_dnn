#!/bin/bash
# Setup script for HuBERT ASR Fine-tuning
# Installs system dependencies and Python packages

set -e

echo "=== HuBERT ASR Setup ==="

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected: macOS"

    # Check for Homebrew
    if ! command -v brew &> /dev/null; then
        echo "Error: Homebrew not found. Install from https://brew.sh"
        exit 1
    fi

    # Install FFmpeg (required for audio decoding)
    echo "Installing FFmpeg..."
    brew install ffmpeg

elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected: Linux"

    # Install FFmpeg
    echo "Installing FFmpeg..."
    if command -v conda &> /dev/null; then
        conda install -y -c conda-forge ffmpeg
    elif command -v apt-get &> /dev/null && command -v sudo &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y ffmpeg
    elif command -v yum &> /dev/null && command -v sudo &> /dev/null; then
        sudo yum install -y ffmpeg
    elif command -v ffmpeg &> /dev/null; then
        echo "FFmpeg already installed."
    else
        echo "Warning: Could not install FFmpeg automatically (no sudo/conda)."
        echo "FFmpeg may already be available, or install it manually."
    fi
else
    echo "Warning: Unknown OS. Please install FFmpeg manually."
fi

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Try to install apple-bolt (optional - to run on Bolt)
echo ""
echo "Attempting to install apple-bolt (optional)..."
pip install apple-bolt 2>/dev/null || echo "Note: apple-bolt not installed (optional, only needed fto run on Bolt)"

# Verify installation
echo ""
echo "=== Verifying Installation ==="

python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import datasets; print(f'Datasets: {datasets.__version__}')"

# Check for GPU
python -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA: Available ({torch.cuda.get_device_name(0)})')
elif torch.backends.mps.is_available():
    print('MPS: Available (Apple Silicon)')
else:
    print('GPU: Not available (CPU only)')
"

echo ""
echo "=== Setup Complete ==="
echo "Run training with: ./scripts/train_debug.sh"
