"""LibriSpeech dataset loading and preprocessing."""

from typing import List, Optional

from datasets import Audio, Dataset, load_dataset
from transformers import Wav2Vec2Processor

from ..config import AudioConfig, DatasetConfig


class LibriSpeechDataset:
    """Wrapper for loading and preparing LibriSpeech data for HuBERT fine-tuning."""

    def __init__(
        self,
        dataset_config: DatasetConfig,
        audio_config: AudioConfig,
        processor: Wav2Vec2Processor,
        eval_splits: Optional[List[str]] = None,
    ):
        """Initialize LibriSpeech dataset loader.

        Args:
            dataset_config: Dataset configuration.
            audio_config: Audio processing configuration.
            processor: HuggingFace processor for feature extraction and tokenization.
            eval_splits: List of evaluation splits to load (e.g., ["validation.clean"]).
        """
        self.dataset_config = dataset_config
        self.audio_config = audio_config
        self.processor = processor
        self.eval_splits = eval_splits or ["validation"]
        self._loaded_splits: dict = {}

    def _load_split(self, split: str) -> Dataset:
        """Load a single split from LibriSpeech.

        Args:
            split: Split name (e.g., "train.100", "validation.clean").

        Returns:
            Dataset for the specified split.
        """
        if split in self._loaded_splits:
            return self._loaded_splits[split]

        dataset = load_dataset(
            self.dataset_config.name,
            self.dataset_config.subset,
            split=split,
            cache_dir=self.dataset_config.cache_dir,
        )

        # Cast audio column to correct sampling rate
        dataset = dataset.cast_column(
            "audio", Audio(sampling_rate=self.audio_config.sampling_rate)
        )

        self._loaded_splits[split] = dataset
        return dataset

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """Apply preprocessing to dataset: filter by duration and extract features.

        Args:
            dataset: Raw dataset to preprocess.

        Returns:
            Preprocessed dataset ready for training.
        """
        # Filter by duration
        dataset = dataset.filter(
            self._filter_by_duration,
            desc="Filtering by duration",
        )

        # Preprocess: extract features and tokenize transcriptions
        dataset = dataset.map(
            self._preprocess_function,
            remove_columns=dataset.column_names,
            desc="Preprocessing",
            num_proc=1,  # Processor not picklable, use single process
        )

        return dataset

    def _filter_by_duration(self, example: dict) -> bool:
        """Filter samples by audio duration.

        Args:
            example: Dataset example with audio data.

        Returns:
            True if example is within duration bounds, False otherwise.
        """
        audio = example["audio"]
        duration = len(audio["array"]) / audio["sampling_rate"]
        return (
            self.audio_config.min_duration_sec
            <= duration
            <= self.audio_config.max_duration_sec
        )

    def _preprocess_function(self, example: dict) -> dict:
        """Preprocess a single example: extract features and tokenize.

        Args:
            example: Dataset example with audio and text.

        Returns:
            Preprocessed example with input_values and labels.
        """
        audio = example["audio"]

        # Extract input features from audio
        inputs = self.processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors=None,
        )

        # Tokenize transcription
        labels = self.processor.tokenizer(
            example["text"],
            return_tensors=None,
        )

        # input_values comes as nested list [[...]], flatten to 1D list
        input_values = inputs["input_values"]
        if isinstance(input_values, list) and len(input_values) == 1:
            input_values = input_values[0]

        return {
            "input_values": input_values,
            "labels": labels["input_ids"],
        }

    def get_train_dataset(self) -> Dataset:
        """Get prepared training dataset.

        Returns:
            Preprocessed training dataset.
        """
        train_split = self.dataset_config.train_split
        train_data = self._load_split(train_split)
        return self.prepare_dataset(train_data)

    def get_eval_dataset(self, split: str) -> Dataset:
        """Get prepared evaluation dataset for a specific split.

        Args:
            split: Dataset split name (e.g., "validation.clean", "test.other").

        Returns:
            Preprocessed evaluation dataset.
        """
        eval_data = self._load_split(split)
        return self.prepare_dataset(eval_data)

    def get_all_eval_datasets(self, splits: list) -> dict:
        """Get prepared evaluation datasets for multiple splits.

        Args:
            splits: List of split names.

        Returns:
            Dictionary mapping split names to preprocessed datasets.
        """
        return {split: self.get_eval_dataset(split) for split in splits}
