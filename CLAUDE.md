# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HuBERT ASR fine-tuning system that reproduces results from the HuBERT paper. Fine-tunes pre-trained HuBERT on LibriSpeech using CTC loss for automatic speech recognition.

## Essential Commands

```bash
# Setup (installs FFmpeg + Python deps)
cd project && ./scripts/setup/setup.sh

# Training
./scripts/train_debug.sh                    # Quick test (1 epoch)
./scripts/train_gpu.sh                      # Full training on NVIDIA GPU
./scripts/train_mac.sh                      # Apple Silicon training
python -m src.train --config configs/default.yaml
python -m src.train --config configs/libri-100h.yaml --resume outputs/checkpoint-500

# Evaluation
python -m src.evaluate --model outputs/hubert-finetuned/final --splits test.clean test.other

# Transcription
python -m src.transcribe --model outputs/hubert-finetuned/final --audio sample.wav
```

## Architecture

The implementation lives in `project/src/` with three entry points:
- **train.py** - Training pipeline with HuggingFace Trainer
- **evaluate.py** - WER evaluation on LibriSpeech splits
- **transcribe.py** - Audio file transcription

### Core Components

| Module | Purpose |
|--------|---------|
| `config.py` | Dataclass-based YAML configuration (ModelConfig, DatasetConfig, TrainingConfig, etc.) |
| `data/dataset.py` | LibriSpeechDataset - loads data, filters by duration, extracts features |
| `data/collator.py` | CTCDataCollator - dynamic padding with -100 ignore index for CTC loss |
| `model/hubert_ctc.py` | HuBERTForASR - model initialization, processor setup, vocab file creation |
| `training/trainer.py` | ASRTrainerSetup - wraps HuggingFace Trainer with CTC-specific config |
| `evaluation/metrics.py` | WERCalculator - greedy CTC decoding + WER computation via jiwer |
| `utils/device.py` | DeviceManager - auto-detects CUDA/MPS/CPU, handles FP16 compatibility |

### Data Flow

1. YAML config loaded → dataclass hierarchy (Config → ModelConfig, DatasetConfig, etc.)
2. DeviceManager selects best device (CUDA > MPS > CPU)
3. HuBERTForASR creates model + processor (writes temp vocab.json to /tmp)
4. LibriSpeechDataset loads split, filters duration (0.5-20s), extracts features
5. CTCDataCollator pads batches dynamically
6. HuggingFace Trainer runs training with WER metric
7. Final model saved to `outputs/{config_name}/final/`

### Configuration System

Four pre-defined configs in `project/configs/`:
- **debug.yaml** - 1 epoch, batch_size=2, for testing
- **default.yaml** - 30 epochs, batch_size=8, standard training
- **libri-100h.yaml** - BF16 enabled, batch_size=16, 8 workers
- **mac.yaml** - MPS device, no FP16, 0 workers

Key parameters: `model.freeze_feature_encoder=true` (CNN frozen), `training.learning_rate=3e-5`, `audio.max_duration_sec=20.0`

### Model Details

- Base model: `facebook/hubert-base-ls960` (90M params, ~18M trainable when CNN frozen)
- Vocabulary: 30 tokens (`<pad>`, `<unk>`, `|`, a-z, apostrophe)
- Decoding: Greedy CTC (argmax, no language model)

## Device-Specific Behavior

- **CUDA**: FP16 enabled, 4-8 dataloader workers
- **MPS**: FP16 disabled (stability), 0 workers
- **CPU**: FP16 disabled, 0 workers
