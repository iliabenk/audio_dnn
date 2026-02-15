"""Configuration dataclasses and YAML loader for HuBERT ASR fine-tuning."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class ModelConfig:
    """Model configuration."""

    name: str = "facebook/hubert-base-ls960"
    freeze_feature_encoder: bool = True
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    feat_proj_dropout: float = 0.0
    layerdrop: float = 0.1


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    name: str = "librispeech_asr"
    subset: str = "clean"
    train_split: str = "train.100"
    cache_dir: Optional[str] = None


@dataclass
class AudioConfig:
    """Audio processing configuration."""

    sampling_rate: int = 16000
    max_duration_sec: float = 20.0
    min_duration_sec: float = 0.5


@dataclass
class TrainingConfig:
    """Training configuration."""

    output_dir: str = "./outputs/hubert-finetuned"
    num_train_epochs: int = 30
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-5
    warmup_steps: int = 500
    weight_decay: float = 0.005
    fp16: bool = True
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "wer"
    greater_is_better: bool = False
    dataloader_num_workers: int = 4
    seed: int = 42


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""

    eval_splits: List[str] = field(default_factory=lambda: ["validation"])
    test_splits: List[str] = field(default_factory=lambda: ["test"])


@dataclass
class DeviceConfig:
    """Device configuration."""

    prefer: str = "auto"


@dataclass
class Config:
    """Main configuration container."""

    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            Config instance with values from YAML file.
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Create configuration from dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            Config instance.
        """
        model_data = data.get("model", {})
        dataset_data = data.get("dataset", {})
        audio_data = data.get("audio", {})
        training_data = data.get("training", {})
        evaluation_data = data.get("evaluation", {})
        device_data = data.get("device", {})

        return cls(
            model=ModelConfig(**model_data),
            dataset=DatasetConfig(**dataset_data),
            audio=AudioConfig(**audio_data),
            training=TrainingConfig(**training_data),
            evaluation=EvaluationConfig(**evaluation_data),
            device=DeviceConfig(**device_data),
        )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration.
        """
        return {
            "model": {
                "name": self.model.name,
                "freeze_feature_encoder": self.model.freeze_feature_encoder,
                "attention_dropout": self.model.attention_dropout,
                "hidden_dropout": self.model.hidden_dropout,
                "feat_proj_dropout": self.model.feat_proj_dropout,
                "layerdrop": self.model.layerdrop,
            },
            "dataset": {
                "name": self.dataset.name,
                "subset": self.dataset.subset,
                "train_split": self.dataset.train_split,
                "cache_dir": self.dataset.cache_dir,
            },
            "audio": {
                "sampling_rate": self.audio.sampling_rate,
                "max_duration_sec": self.audio.max_duration_sec,
                "min_duration_sec": self.audio.min_duration_sec,
            },
            "training": {
                "output_dir": self.training.output_dir,
                "num_train_epochs": self.training.num_train_epochs,
                "per_device_train_batch_size": self.training.per_device_train_batch_size,
                "per_device_eval_batch_size": self.training.per_device_eval_batch_size,
                "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
                "learning_rate": self.training.learning_rate,
                "warmup_steps": self.training.warmup_steps,
                "weight_decay": self.training.weight_decay,
                "fp16": self.training.fp16,
                "logging_steps": self.training.logging_steps,
                "eval_steps": self.training.eval_steps,
                "save_steps": self.training.save_steps,
                "save_total_limit": self.training.save_total_limit,
                "load_best_model_at_end": self.training.load_best_model_at_end,
                "metric_for_best_model": self.training.metric_for_best_model,
                "greater_is_better": self.training.greater_is_better,
                "dataloader_num_workers": self.training.dataloader_num_workers,
                "seed": self.training.seed,
            },
            "evaluation": {
                "eval_splits": self.evaluation.eval_splits,
                "test_splits": self.evaluation.test_splits,
            },
            "device": {
                "prefer": self.device.prefer,
            },
        }

    def save_yaml(self, path: str) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save YAML file.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
