"""CTC Decoder with optional Language Model support."""

import logging
from typing import List, Optional

import numpy as np
from transformers import Wav2Vec2Processor

logger = logging.getLogger(__name__)


class CTCDecoder:
    """CTC Decoder supporting greedy and beam search with optional LM."""

    def __init__(
        self,
        processor: Wav2Vec2Processor,
        use_lm: bool = False,
        lm_path: Optional[str] = None,
        beam_width: int = 100,
        alpha: float = 0.5,
        beta: float = 1.0,
    ):
        """Initialize CTC decoder.

        Args:
            processor: HuggingFace processor with tokenizer.
            use_lm: Whether to use language model decoding.
            lm_path: Path to KenLM language model (.arpa or .bin file).
            beam_width: Beam width for beam search (default: 100).
            alpha: LM weight (default: 0.5).
            beta: Word insertion bonus (default: 1.0).
        """
        self.processor = processor
        self.use_lm = use_lm
        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta
        self.lm_path = lm_path

        self._decoder = None

        if use_lm:
            self._init_beam_search_decoder(lm_path)

    def _init_beam_search_decoder(self, lm_path: Optional[str] = None):
        """Initialize pyctcdecode beam search decoder with optional LM.

        Args:
            lm_path: Path to KenLM language model file.
        """
        try:
            from pyctcdecode import build_ctcdecoder
        except ImportError:
            logger.warning(
                "pyctcdecode not installed. Install with: pip install pyctcdecode"
            )
            self.use_lm = False
            return

        # Build vocabulary list from processor tokenizer
        vocab = self.processor.tokenizer.get_vocab()
        # Sort by token ID to get correct order
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        labels = [token for token, _ in sorted_vocab]

        # pyctcdecode expects specific tokens:
        # - CTC blank should be "" (empty string) - this is typically <pad> at index 0
        # - Word boundary | should be " " (space)
        # - All other tokens must be unique
        labels_for_decoder = []
        for label in labels:
            if label == "<pad>":
                # CTC blank token
                labels_for_decoder.append("")
            elif label == "|":
                # Word delimiter -> space
                labels_for_decoder.append(" ")
            elif label == "<unk>":
                # Keep <unk> as a special token (pyctcdecode handles this)
                labels_for_decoder.append("⁇")  # Use a unique placeholder
            else:
                labels_for_decoder.append(label)

        # Build decoder
        if lm_path:
            try:
                import kenlm  # Check if kenlm is available
                self._decoder = build_ctcdecoder(
                    labels=labels_for_decoder,
                    kenlm_model_path=lm_path,
                    alpha=self.alpha,
                    beta=self.beta,
                )
                logger.info(f"Loaded LM from {lm_path}")
            except ImportError:
                logger.warning(
                    f"kenlm not installed. Cannot load LM from {lm_path}. "
                    "Install with: pip install kenlm"
                )
                logger.info("Falling back to beam search without LM")
                self._decoder = build_ctcdecoder(labels=labels_for_decoder)
            except Exception as e:
                logger.warning(f"Failed to load LM from {lm_path}: {e}")
                logger.info("Falling back to beam search without LM")
                self._decoder = build_ctcdecoder(labels=labels_for_decoder)
        else:
            # Beam search without LM
            self._decoder = build_ctcdecoder(labels=labels_for_decoder)
            logger.info("Using beam search decoder without LM")

    def decode_greedy(self, logits: np.ndarray) -> List[str]:
        """Decode using greedy (argmax) decoding.

        Args:
            logits: Model output logits of shape (batch, seq_len, vocab_size).

        Returns:
            List of decoded text strings.
        """
        predicted_ids = np.argmax(logits, axis=-1)
        predictions = self.processor.batch_decode(predicted_ids)
        return predictions

    def decode_beam_search(self, logits: np.ndarray) -> List[str]:
        """Decode using beam search with optional LM.

        Args:
            logits: Model output logits of shape (batch, seq_len, vocab_size).

        Returns:
            List of decoded text strings.
        """
        if self._decoder is None:
            logger.warning("Beam search decoder not initialized, falling back to greedy")
            return self.decode_greedy(logits)

        # Apply softmax to get probabilities
        from scipy.special import softmax
        probs = softmax(logits, axis=-1)

        predictions = []
        for prob in probs:
            # Decode single sequence
            text = self._decoder.decode(
                prob,
                beam_width=self.beam_width,
            )
            predictions.append(text)

        return predictions

    def decode(self, logits: np.ndarray) -> List[str]:
        """Decode logits using configured method.

        Args:
            logits: Model output logits of shape (batch, seq_len, vocab_size).

        Returns:
            List of decoded text strings.
        """
        if self.use_lm:
            return self.decode_beam_search(logits)
        else:
            return self.decode_greedy(logits)
