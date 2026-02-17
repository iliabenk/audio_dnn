"""HuBERT model initialization for CTC-based ASR."""

import json
import tempfile
from pathlib import Path

from transformers import (
    HubertConfig,
    HubertForCTC,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

from ..config import ModelConfig


class HuBERTForASR:
    """Wrapper for HuBERT CTC model initialization and configuration."""

    # Vocabulary for LibriSpeech (lowercase letters + special tokens)
    VOCAB = {
        "<pad>": 0,
        "<unk>": 1,
        "|": 2,  # Word delimiter
        "a": 3,
        "b": 4,
        "c": 5,
        "d": 6,
        "e": 7,
        "f": 8,
        "g": 9,
        "h": 10,
        "i": 11,
        "j": 12,
        "k": 13,
        "l": 14,
        "m": 15,
        "n": 16,
        "o": 17,
        "p": 18,
        "q": 19,
        "r": 20,
        "s": 21,
        "t": 22,
        "u": 23,
        "v": 24,
        "w": 25,
        "x": 26,
        "y": 27,
        "z": 28,
        "'": 29,
    }

    def __init__(self, config: ModelConfig):
        """Initialize HuBERT ASR model builder.

        Args:
            config: Model configuration.
        """
        self.config = config

    def build_processor(self) -> Wav2Vec2Processor:
        """Build feature extractor and tokenizer processor.

        Returns:
            Wav2Vec2Processor for preprocessing audio and text.
        """
        # Load feature extractor from pretrained model
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.config.name
        )

        # Create a temporary vocab file for the tokenizer
        vocab_file = Path(tempfile.gettempdir()) / "hubert_vocab.json"
        with open(vocab_file, "w") as f:
            json.dump(self.VOCAB, f)

        # Create tokenizer with vocabulary file
        tokenizer = Wav2Vec2CTCTokenizer(
            vocab_file=str(vocab_file),
            unk_token="<unk>",
            pad_token="<pad>",
            word_delimiter_token="|",
        )

        # Combine into processor
        processor = Wav2Vec2Processor(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
        )

        return processor

    def build_model(self, vocab_size: int = None) -> HubertForCTC:
        """Build HuBERT model with CTC head.

        Args:
            vocab_size: Size of vocabulary. If None, uses default vocab size.

        Returns:
            HubertForCTC model ready for fine-tuning.
        """
        if vocab_size is None:
            vocab_size = len(self.VOCAB)

        # Load model configuration
        model_config = HubertConfig.from_pretrained(
            self.config.name,
            attention_dropout=self.config.attention_dropout,
            hidden_dropout=self.config.hidden_dropout,
            feat_proj_dropout=self.config.feat_proj_dropout,
            layerdrop=self.config.layerdrop,
            ctc_loss_reduction="mean",
            pad_token_id=0,  # <pad> token
            vocab_size=vocab_size,
        )

        # Load pretrained model with CTC head
        model = HubertForCTC.from_pretrained(
            self.config.name,
            config=model_config,
            ignore_mismatched_sizes=True,  # For different vocab size
            use_safetensors=True,  # Use safetensors to avoid torch.load security issue
        )

        # Freeze feature encoder if configured
        if self.config.freeze_feature_encoder:
            self.freeze_feature_encoder(model)

        return model

    @staticmethod
    def freeze_feature_encoder(model: HubertForCTC) -> None:
        """Freeze the CNN feature encoder layers.

        This is recommended for fine-tuning as the feature encoder
        is already well-trained and freezing it helps stability.

        Args:
            model: HuBERT model to modify.
        """
        model.freeze_feature_encoder()

    @staticmethod
    def count_parameters(model: HubertForCTC) -> dict:
        """Count trainable and total parameters.

        Args:
            model: Model to count parameters for.

        Returns:
            Dictionary with 'trainable' and 'total' parameter counts.
        """
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        return {
            "trainable": trainable,
            "total": total,
            "trainable_percent": 100 * trainable / total if total > 0 else 0,
        }
