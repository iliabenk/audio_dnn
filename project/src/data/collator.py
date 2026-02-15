"""Data collator for CTC training with dynamic padding."""

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
from transformers import Wav2Vec2Processor


@dataclass
class CTCDataCollator:
    """Data collator for CTC training with dynamic padding.

    This collator handles:
    - Padding input_values to the maximum length in the batch
    - Padding labels with -100 (ignore index for CTC loss)
    - Creating attention masks for padded inputs
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: int = None
    pad_to_multiple_of: int = None

    def __call__(
        self, features: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """Collate batch of features with proper padding.

        Args:
            features: List of feature dictionaries with 'input_values' and 'labels'.

        Returns:
            Batch dictionary with padded tensors.
        """
        # Separate inputs and labels
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad input features
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Pad labels with -100 (CTC loss ignore index)
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Replace padding token id with -100 for CTC loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels

        return batch
