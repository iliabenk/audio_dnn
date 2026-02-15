"""LibriSpeech dataset loading and preprocessing."""

from typing import Optional

from datasets import Audio, Dataset, DatasetDict, load_dataset
from transformers import Wav2Vec2Processor

from ..config import AudioConfig, DatasetConfig


class LibriSpeechDataset:
    """Wrapper for loading and preparing LibriSpeech data for HuBERT fine-tuning."""

    def __init__(
        self,
        dataset_config: DatasetConfig,
        audio_config: AudioConfig,
        processor: Wav2Vec2Processor,
    ):
        """Initialize LibriSpeech dataset loader.

        Args:
            dataset_config: Dataset configuration.
            audio_config: Audio processing configuration.
            processor: HuggingFace processor for feature extraction and tokenization.
        """
        self.dataset_config = dataset_config
        self.audio_config = audio_config
        self.processor = processor
        self._dataset: Optional[DatasetDict] = None

    def load_dataset(self) -> DatasetDict:
        """Load LibriSpeech dataset from HuggingFace.

        Returns:
            DatasetDict containing train/validation/test splits.
        """
        dataset = load_dataset(
            self.dataset_config.name,
            self.dataset_config.subset,
            cache_dir=self.dataset_config.cache_dir,
            trust_remote_code=True,
        )

        # Cast audio column to correct sampling rate
        dataset = dataset.cast_column(
            "audio", Audio(sampling_rate=self.audio_config.sampling_rate)
        )

        self._dataset = dataset
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

        return {
            "input_values": inputs["input_values"],
            "labels": labels["input_ids"],
        }

    def get_train_dataset(self) -> Dataset:
        """Get prepared training dataset.

        Returns:
            Preprocessed training dataset.
        """
        if self._dataset is None:
            self.load_dataset()

        # Get the appropriate training split
        train_split = self.dataset_config.train_split
        if train_split in self._dataset:
            train_data = self._dataset[train_split]
        else:
            # Handle different split naming conventions
            train_data = self._dataset["train"]

        return self.prepare_dataset(train_data)

    def get_eval_dataset(self, split: str) -> Dataset:
        """Get prepared evaluation dataset for a specific split.

        Args:
            split: Dataset split name (e.g., "validation.clean", "test.other").

        Returns:
            Preprocessed evaluation dataset.
        """
        if self._dataset is None:
            self.load_dataset()

        if split in self._dataset:
            eval_data = self._dataset[split]
        else:
            # Try without the subset suffix
            base_split = split.split(".")[0]
            if base_split in self._dataset:
                eval_data = self._dataset[base_split]
            else:
                raise ValueError(f"Split '{split}' not found in dataset")

        return self.prepare_dataset(eval_data)

    def get_all_eval_datasets(self, splits: list) -> dict:
        """Get prepared evaluation datasets for multiple splits.

        Args:
            splits: List of split names.

        Returns:
            Dictionary mapping split names to preprocessed datasets.
        """
        return {split: self.get_eval_dataset(split) for split in splits}
