"""Word Error Rate (WER) computation for ASR evaluation."""

import os
from typing import TYPE_CHECKING, Dict, List, Optional

import jiwer
import numpy as np
from transformers import Wav2Vec2Processor

if TYPE_CHECKING:
    from .decoder import CTCDecoder


class WERCalculator:
    """Word Error Rate computation for ASR evaluation."""

    def __init__(
        self,
        processor: Wav2Vec2Processor,
        decoder: Optional["CTCDecoder"] = None,
    ):
        """Initialize WER calculator.

        Args:
            processor: HuggingFace processor for decoding predictions.
            decoder: Optional CTC decoder for beam search + LM decoding.
                    If None, uses greedy decoding.
        """
        self.processor = processor
        self.decoder = decoder

        # Text normalization transformation
        # Note: We don't remove punctuation because apostrophes are valid
        # characters in our vocabulary (e.g., "don't", "it's")
        self.transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
        ])

    def decode_predictions(self, logits: np.ndarray) -> List[str]:
        """Decode CTC logits to text.

        Uses beam search + LM if decoder is configured, otherwise greedy.

        Args:
            logits: Model output logits of shape (batch, seq_len, vocab_size).

        Returns:
            List of decoded text strings.
        """
        if self.decoder is not None:
            return self.decoder.decode(logits)

        # Greedy decoding fallback
        predicted_ids = np.argmax(logits, axis=-1)
        predictions = self.processor.batch_decode(predicted_ids)
        return predictions

    def compute_wer(
        self,
        predictions: List[str],
        references: List[str],
        normalize: bool = True,
    ) -> Dict[str, float]:
        """Compute Word Error Rate and related metrics.

        Args:
            predictions: List of predicted transcriptions.
            references: List of reference transcriptions.
            normalize: Whether to normalize text before computing WER.

        Returns:
            Dictionary with WER and detailed metrics.
        """
        if normalize:
            predictions = [self.transformation(p) for p in predictions]
            references = [self.transformation(r) for r in references]

        # Filter out empty references (can cause issues)
        valid_pairs = [
            (p, r) for p, r in zip(predictions, references) if r.strip()
        ]

        if not valid_pairs:
            return {
                "wer": 0.0,
                "substitutions": 0,
                "insertions": 0,
                "deletions": 0,
                "hits": 0,
            }

        predictions, references = zip(*valid_pairs)

        # Compute WER
        wer = jiwer.wer(list(references), list(predictions))

        # Compute detailed measures using process_words (newer jiwer API)
        output = jiwer.process_words(list(references), list(predictions))

        return {
            "wer": wer,
            "substitutions": output.substitutions,
            "insertions": output.insertions,
            "deletions": output.deletions,
            "hits": output.hits,
        }

    def compute_metrics_for_trainer(self, pred) -> Dict[str, float]:
        """Compute metrics in the format expected by HuggingFace Trainer.

        This method is compatible with the Trainer's compute_metrics callback.

        Args:
            pred: EvalPrediction object with predictions and label_ids.

        Returns:
            Dictionary with 'wer' metric.
        """
        pred_logits = pred.predictions

        # Replace -100 with pad token id for proper decoding
        label_ids = pred.label_ids.copy()  # Don't modify original
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id

        # Decode predictions using decoder (beam search + LM) or greedy
        predictions = self.decode_predictions(pred_logits)

        # Decode references
        references = self.processor.batch_decode(label_ids, group_tokens=False)

        # Debug: print first 3 samples (only on main process)
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is None or int(local_rank) == 0:
            print("\n" + "=" * 50)
            print("DEBUG: Sample predictions vs references")
            for i in range(min(3, len(predictions))):
                print(f"  [{i}] PRED: '{predictions[i][:100]}'")
                print(f"  [{i}] REF:  '{references[i][:100]}'")
            print("=" * 50 + "\n")

        # Compute WER
        metrics = self.compute_wer(predictions, references)

        return {"wer": metrics["wer"]}

    @staticmethod
    def format_wer(wer: float) -> str:
        """Format WER as percentage string.

        Args:
            wer: WER value (0-1 scale).

        Returns:
            Formatted string (e.g., "5.23%").
        """
        return f"{wer * 100:.2f}%"
