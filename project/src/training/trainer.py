"""HuggingFace Trainer setup for HuBERT ASR fine-tuning."""

from pathlib import Path
from typing import Callable, Optional

import torch
from datasets import Dataset
from transformers import (
    HubertForCTC,
    Trainer,
    TrainingArguments,
    Wav2Vec2Processor,
)

from ..config import TrainingConfig
from ..utils.device import DeviceManager


class ASRTrainerSetup:
    """Setup and configuration for HuggingFace Trainer."""

    def __init__(
        self,
        model: HubertForCTC,
        processor: Wav2Vec2Processor,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        config: TrainingConfig,
        device: torch.device,
        compute_metrics: Optional[Callable] = None,
    ):
        """Initialize ASR trainer setup.

        Args:
            model: HuBERT model with CTC head.
            processor: Wav2Vec2 processor for tokenization.
            train_dataset: Prepared training dataset.
            eval_dataset: Prepared evaluation dataset.
            config: Training configuration.
            device: Compute device.
            compute_metrics: Optional function for computing metrics during eval.
        """
        self.model = model
        self.processor = processor
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config
        self.device = device
        self.compute_metrics = compute_metrics

    def get_training_args(self) -> TrainingArguments:
        """Create TrainingArguments from configuration.

        Returns:
            Configured TrainingArguments for HuggingFace Trainer.
        """
        # Check if FP16 is supported on this device
        use_fp16 = self.config.fp16 and DeviceManager.is_fp16_supported(self.device)

        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        return TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            fp16=use_fp16,
            logging_dir=str(output_dir / "logs"),
            logging_steps=self.config.logging_steps,
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            dataloader_num_workers=self.config.dataloader_num_workers,
            seed=self.config.seed,
            report_to=["tensorboard"],
            push_to_hub=False,
            # Group samples by length for efficient batching
            group_by_length=True,
            length_column_name="input_length",
        )

    def get_data_collator(self):
        """Create data collator for CTC training.

        Returns:
            CTCDataCollator for dynamic padding.
        """
        from ..data.collator import CTCDataCollator
        return CTCDataCollator(processor=self.processor)

    def get_trainer(self) -> Trainer:
        """Create configured HuggingFace Trainer instance.

        Returns:
            Trainer ready for training.
        """
        training_args = self.get_training_args()
        data_collator = self.get_data_collator()

        # Add input_length column for group_by_length
        def add_length_column(example):
            example["input_length"] = len(example["input_values"])
            return example

        train_dataset = self.train_dataset.map(add_length_column)
        eval_dataset = self.eval_dataset.map(add_length_column)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=self.processor.feature_extractor,
        )

        return trainer


def create_trainer(
    model: HubertForCTC,
    processor: Wav2Vec2Processor,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    config: TrainingConfig,
    device: torch.device,
    compute_metrics: Optional[Callable] = None,
) -> Trainer:
    """Convenience function to create a configured Trainer.

    Args:
        model: HuBERT model with CTC head.
        processor: Wav2Vec2 processor for tokenization.
        train_dataset: Prepared training dataset.
        eval_dataset: Prepared evaluation dataset.
        config: Training configuration.
        device: Compute device.
        compute_metrics: Optional function for computing metrics during eval.

    Returns:
        Configured Trainer ready for training.
    """
    setup = ASRTrainerSetup(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config,
        device=device,
        compute_metrics=compute_metrics,
    )
    return setup.get_trainer()
