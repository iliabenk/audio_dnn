"""HuggingFace Trainer setup for HuBERT ASR fine-tuning."""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import torch
from datasets import Dataset
from transformers import (
    HubertForCTC,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    Wav2Vec2Processor,
)

from ..config import TrainingConfig
from ..utils.bolt import get_artifact_dir, is_on_bolt
from ..utils.device import DeviceManager
from .callbacks import get_bolt_callback


class ASRTrainerSetup:
    """Setup and configuration for HuggingFace Trainer."""

    def __init__(
        self,
        model: HubertForCTC,
        processor: Wav2Vec2Processor,
        train_dataset: Dataset,
        eval_datasets: Union[Dataset, Dict[str, Dataset]],
        config: TrainingConfig,
        device: torch.device,
        compute_metrics: Optional[Callable] = None,
        metric_for_best_model: Optional[str] = None,
    ):
        """Initialize ASR trainer setup.

        Args:
            model: HuBERT model with CTC head.
            processor: Wav2Vec2 processor for tokenization.
            train_dataset: Prepared training dataset.
            eval_datasets: Prepared evaluation dataset(s). Can be a single Dataset
                or a dict mapping split names to datasets for multi-split eval.
            config: Training configuration.
            device: Compute device.
            compute_metrics: Optional function for computing metrics during eval.
            metric_for_best_model: Override metric name for load_best_model_at_end.
        """
        self.model = model
        self.processor = processor
        self.train_dataset = train_dataset
        self.eval_datasets = eval_datasets
        self.config = config
        self.device = device
        self.compute_metrics = compute_metrics
        self.metric_for_best_model = metric_for_best_model

    def get_training_args(self) -> TrainingArguments:
        """Create TrainingArguments from configuration.

        Returns:
            Configured TrainingArguments for HuggingFace Trainer.
        """
        # Determine precision settings
        # bf16 requires CUDA with Ampere+ architecture (compute capability >= 8.0)
        use_bf16 = False
        use_fp16 = False

        if self.device.type == "cuda":
            if self.config.bf16:
                # Check if bf16 is supported (Ampere+)
                import torch
                if torch.cuda.get_device_capability()[0] >= 8:
                    use_bf16 = True
                else:
                    # Fall back to fp16 if bf16 not supported
                    use_fp16 = self.config.fp16 and DeviceManager.is_fp16_supported(self.device)
            else:
                use_fp16 = self.config.fp16 and DeviceManager.is_fp16_supported(self.device)

        # Determine device settings for TrainingArguments
        use_cpu = self.device.type == "cpu"

        # Use ARTIFACT_DIR when on Bolt, otherwise use config output_dir
        if is_on_bolt():
            artifact_dir = get_artifact_dir()
            output_dir = Path(artifact_dir) if artifact_dir else Path(self.config.output_dir)
        else:
            output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine LR scheduler settings
        lr_scheduler_type = self.config.lr_scheduler_type
        lr_scheduler_kwargs = {}

        # If min LR is specified with cosine, use cosine_with_min_lr
        if self.config.lr_min is not None:
            if lr_scheduler_type == "cosine":
                lr_scheduler_type = "cosine_with_min_lr"
            lr_scheduler_kwargs["min_lr"] = self.config.lr_min

        return TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            lr_scheduler_kwargs=lr_scheduler_kwargs if lr_scheduler_kwargs else None,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            fp16=use_fp16,
            bf16=use_bf16,
            logging_dir=str(output_dir / "logs"),
            logging_steps=self.config.logging_steps,
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.metric_for_best_model or self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            dataloader_num_workers=self.config.dataloader_num_workers,
            seed=self.config.seed,
            report_to=["tensorboard"],
            push_to_hub=False,
            # Device settings - explicitly control device selection
            use_cpu=use_cpu,
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

        # Build callbacks list
        callbacks: List[TrainerCallback] = []
        bolt_callback = get_bolt_callback(metric_prefix=self.config.bolt_metric_prefix)
        if bolt_callback is not None:
            callbacks.append(bolt_callback)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_datasets,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            processing_class=self.processor.feature_extractor,
            callbacks=callbacks if callbacks else None,
        )

        return trainer


def create_trainer(
    model: HubertForCTC,
    processor: Wav2Vec2Processor,
    train_dataset: Dataset,
    eval_datasets: Union[Dataset, Dict[str, Dataset]],
    config: TrainingConfig,
    device: torch.device,
    compute_metrics: Optional[Callable] = None,
    metric_for_best_model: Optional[str] = None,
) -> Trainer:
    """Convenience function to create a configured Trainer.

    Args:
        model: HuBERT model with CTC head.
        processor: Wav2Vec2 processor for tokenization.
        train_dataset: Prepared training dataset.
        eval_datasets: Prepared evaluation dataset(s). Can be a single Dataset
            or a dict mapping split names to datasets for multi-split eval.
        config: Training configuration.
        device: Compute device.
        compute_metrics: Optional function for computing metrics during eval.
        metric_for_best_model: Override metric name for load_best_model_at_end.

    Returns:
        Configured Trainer ready for training.
    """
    setup = ASRTrainerSetup(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        eval_datasets=eval_datasets,
        config=config,
        device=device,
        compute_metrics=compute_metrics,
        metric_for_best_model=metric_for_best_model,
    )
    return setup.get_trainer()
