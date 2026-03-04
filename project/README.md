# HuBERT ASR Fine-tuning

Reproduce HuBERT ASR results by fine-tuning on LibriSpeech using CTC loss.

## Overview

This project implements a modular system for fine-tuning the HuBERT (Hidden-Unit BERT) model on the LibriSpeech dataset for Automatic Speech Recognition (ASR). It reproduces key results from the paper "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units" by Hsu et al.

## Project Structure

```
project/
├── configs/                   # Configuration files
│   ├── default.yaml          # Default 100h training
│   ├── debug.yaml            # Quick debug config
│   ├── libri-100h.yaml        # Optimized for large GPUs (24GB+)
│   └── mac.yaml              # Optimized for Mac with MPS
├── src/                      # Source code
│   ├── train.py              # Training entry point
│   ├── evaluate.py           # Evaluation entry point
│   ├── transcribe.py         # Transcription entry point
│   ├── config.py             # Configuration dataclasses
│   ├── data/                 # Data loading modules
│   │   ├── dataset.py        # LibriSpeech dataset
│   │   └── collator.py       # CTC data collator
│   ├── model/                # Model modules
│   │   └── hubert_ctc.py     # HuBERT CTC wrapper
│   ├── training/             # Training modules
│   │   └── trainer.py        # HuggingFace Trainer setup
│   ├── evaluation/           # Evaluation modules
│   │   └── metrics.py        # WER computation
│   └── utils/                # Utility modules
│       └── device.py         # Device management
├── scripts/                  # Bash scripts with predefined configs
│   ├── setup.sh              # Setup script (installs FFmpeg + deps)
│   ├── train_debug.sh        # Quick debug training
│   ├── train_gpu.sh          # GPU training
│   ├── train_mac.sh          # Mac training
│   ├── evaluate.sh           # Evaluation wrapper
│   └── transcribe.sh         # Transcription wrapper
├── outputs/                  # Training outputs (gitignored)
└── requirements.txt          # Dependencies
```

## Installation

### Quick Setup (Recommended)

```bash
cd project
./scripts/setup.sh
```

This installs FFmpeg (required for audio decoding) and all Python dependencies.

### Manual Setup

1. Install FFmpeg:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Or via conda
conda install -c conda-forge ffmpeg
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

**Using bash scripts (recommended):**
```bash
# Debug run (quick test)
./scripts/train_debug.sh

# Full training on GPU
./scripts/train_gpu.sh

# Training on Mac
./scripts/train_mac.sh
```

**Using Python module directly:**
```bash
# With any config
python -m src.train --config configs/default.yaml

# Resume from checkpoint
python -m src.train --config configs/libri-100h.yaml --resume outputs/checkpoint-500
```

### Evaluation

**Using bash script:**
```bash
./scripts/evaluate.sh outputs/hubert-gpu/final --splits test.clean test.other
```

**Using Python module:**
```bash
python -m src.evaluate --model outputs/hubert-finetuned/final --splits test.clean test.other
python -m src.evaluate --model outputs/hubert-finetuned/final --output results.json
```

### Transcription

**Using bash script:**
```bash
./scripts/transcribe.sh outputs/hubert-gpu/final sample.wav
```

**Using Python module:**
```bash
python -m src.transcribe --model outputs/hubert-finetuned/final --audio sample.wav
python -m src.transcribe --model outputs/hubert-finetuned/final --audio file1.wav file2.wav
```

## Configuration

Configuration is done via YAML files. Key parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model.name` | HuggingFace model name | `facebook/hubert-base-ls960` |
| `model.freeze_feature_encoder` | Freeze CNN layers | `true` |
| `dataset.train_split` | Training split | `train.100` |
| `training.num_train_epochs` | Number of epochs | `30` |
| `training.learning_rate` | Learning rate | `3e-5` |
| `training.per_device_train_batch_size` | Batch size per device | varies by config |
| `training.fp16` | Mixed precision training | `true` (GPU only) |

### Available Training Splits

- `train.100` - 100 hours (train-clean-100)
- `train.360` - 360 hours (train-clean-360)
- `train.other.500` - 500 hours (train-other-500)

## Results

### Our Results

**100h training (train-clean-100):**

| Decoding | dev-clean | dev-other | test-clean | test-other |
|----------|-----------|-----------|------------|------------|
| LM (best) | 4.30% | 11.75% | 4.62% | 11.63% |
| LM        | 4.44% | 12.48% | 4.69% | 12.61% |
| Greedy    | 5.45% | 14.80% | 5.74% | 15.05% |

**10h training (train-10h subset):**

| Decoding | dev-clean | dev-other | test-clean | test-other |
|----------|-----------|-----------|------------|------------|
| LM       | 8.62% | 20.83% | 9.15% | 21.78% |
| Greedy   | 10.31% | 23.79% | 10.90% | 25.01% |

### HuBERT Paper Baseline (Table 5)

| Training Data | dev-clean | dev-other | test-clean | test-other |
|---------------|-----------|-----------|------------|------------|
| 100h          | ~4.3%     | ~9.4%     | ~4.6%      | ~9.5%      |
| 960h          | ~3.4%     | ~6.6%     | ~3.8%      | ~6.8%      |

*Note: Results may vary slightly due to random initialization and hardware differences.*

## Device Support

The code automatically detects and uses the best available device:

1. **CUDA** - NVIDIA GPUs (recommended for training)
2. **MPS** - Apple Silicon Macs (M1/M2/M3)
3. **CPU** - Fallback option (slow)

Use the `device.prefer` config option to force a specific device:
```yaml
device:
  prefer: "cuda"  # or "mps" or "cpu" or "auto"
```

## Technical Details

### Architecture

- **Model**: HuBERT-Base (90M parameters)
- **Loss**: Connectionist Temporal Classification (CTC)
- **Decoding**: Greedy CTC decoding
- **Metric**: Word Error Rate (WER)

### Training Strategy

Following the HuBERT paper:
1. Start from pre-trained HuBERT model
2. Freeze the CNN feature encoder
3. Fine-tune transformer layers with CTC head
4. Use AdamW optimizer with linear warmup

## References

- [HuBERT Paper](https://arxiv.org/abs/2106.07447): Hsu et al., "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units"
- [HuggingFace Model](https://huggingface.co/facebook/hubert-base-ls960)
- [LibriSpeech Dataset](https://www.openslr.org/12)

## Authors

- Gal Barak (211699707)
- Adam Fleisher (211603469)
- Ilia Benkovitch (316857820)

## License

This project is for educational purposes as part of a course assignment.
