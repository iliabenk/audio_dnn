"""LibriSpeech dataset loading and preprocessing."""

import os
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
        num_proc: Optional[int] = None,
    ):
        """Initialize LibriSpeech dataset loader.

        Args:
            dataset_config: Dataset configuration.
            audio_config: Audio processing configuration.
            processor: HuggingFace processor for feature extraction and tokenization.
            eval_splits: List of evaluation splits to load (e.g., ["validation.clean"]).
            num_proc: Number of processes for dataset mapping. Defaults to number of CPUs.
        """
        self.dataset_config = dataset_config
        self.audio_config = audio_config
        self.processor = processor
        self.eval_splits = eval_splits or ["validation"]
        self._loaded_splits: dict = {}

        # Set number of processes for parallel mapping
        # In distributed training, reduce num_proc to avoid resource exhaustion
        if num_proc is None:
            local_rank = os.environ.get("LOCAL_RANK")
            if local_rank is not None:
                # Distributed training: use fewer workers to avoid resource issues
                # Only rank 0 uses multiprocessing, others use single process
                if int(local_rank) == 0:
                    self.num_proc = min(os.cpu_count() or 1, 8)
                else:
                    self.num_proc = 1
            else:
                # Single process training: use all available CPUs, cap at 16
                self.num_proc = min(os.cpu_count() or 1, 16)
        else:
            self.num_proc = num_proc

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
        # Filter by duration (can use multiprocessing)
        dataset = dataset.filter(
            self._filter_by_duration,
            num_proc=self.num_proc,
            desc="Filtering by duration",
        )

        # Preprocess: extract features and tokenize transcriptions
        # Use batched processing for better performance
        dataset = dataset.map(
            self._preprocess_function_batched,
            remove_columns=dataset.column_names,
            batched=True,
            batch_size=100,
            num_proc=self.num_proc,
            desc="Preprocessing",
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

    def _preprocess_function_batched(self, examples: dict) -> dict:
        """Preprocess a batch of examples: extract features and tokenize.

        Args:
            examples: Batch of dataset examples with audio and text.

        Returns:
            Preprocessed batch with input_values and labels.
        """
        # Extract audio arrays
        audio_arrays = [audio["array"] for audio in examples["audio"]]
        sampling_rates = [audio["sampling_rate"] for audio in examples["audio"]]

        # Process all audio in batch
        inputs = self.processor(
            audio_arrays,
            sampling_rate=sampling_rates[0],  # All should be same rate
            return_tensors=None,
            padding=False,
        )

        # Tokenize all transcriptions in batch
        labels = self.processor.tokenizer(
            examples["text"],
            return_tensors=None,
            padding=False,
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
