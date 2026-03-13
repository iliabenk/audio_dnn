HuBERT ASR Fine-tuning - Project README
========================================

Requirements:
  - Python 3.10+
  - FFmpeg (for audio decoding)

Installation:
  pip install -r requirements.txt

Training:
  python -m src.train --config configs/default.yaml

  Other training configs:
    python -m src.train --config configs/debug.yaml          # Quick test (1 epoch)
    python -m src.train --config configs/mac.yaml            # Apple Silicon
    python -m src.train --config configs/libri-100h.yaml     # Large GPU (24GB+)

  Resume from checkpoint:
    python -m src.train --config configs/default.yaml --resume outputs/checkpoint-500

Evaluation:
  python -m src.evaluate --model outputs/hubert-finetuned/final --splits test.clean test.other

Transcription:
  python -m src.transcribe --model outputs/hubert-finetuned/final --audio sample.wav

Authors:
  Gal Barak (211699707)
  Adam Fleisher (211603469)
  Ilia Benkovitch (316857820)
