"""Word Error Rate (WER) computation for ASR evaluation."""

from typing import Dict, List

import jiwer
import numpy as np
from transformers import Wav2Vec2Processor


class WERCalculator:
    """Word Error Rate computation for ASR evaluation."""

    def __init__(self, processor: Wav2Vec2Processor):
        """Initialize WER calculator.

        Args:
            processor: HuggingFace processor for decoding predictions.
        """
        self.processor = processor

        # Text normalization transformation
        self.transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemovePunctuation(),
        ])

    def decode_predictions(self, logits: np.ndarray) -> List[str]:
        """Decode CTC logits to text using greedy decoding.

        Args:
            logits: Model output logits of shape (batch, seq_len, vocab_size).

        Returns:
            List of decoded text strings.
        """
        # Get predicted token IDs (argmax along vocabulary dimension)
        predicted_ids = np.argmax(logits, axis=-1)

        # Decode to text
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

        # Compute detailed measures
        measures = jiwer.compute_measures(list(references), list(predictions))

        return {
            "wer": wer,
            "substitutions": measures["substitutions"],
            "insertions": measures["insertions"],
            "deletions": measures["deletions"],
            "hits": measures["hits"],
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
        pred_ids = np.argmax(pred_logits, axis=-1)

        # Replace -100 with pad token id for proper decoding
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id

        # Decode predictions and labels
        predictions = self.processor.batch_decode(pred_ids)
        references = self.processor.batch_decode(label_ids, group_tokens=False)

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
