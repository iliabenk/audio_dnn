"""HuggingFace Trainer callback for logging metrics to TuriBolt."""

import logging
import time
from typing import Dict, List, Optional

import apple_bolt as bolt
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from ...utils.bolt import is_on_bolt

logger = logging.getLogger(__name__)


class BoltCallback(TrainerCallback):
    """
    TrainerCallback that logs metrics to TuriBolt.

    This callback logs training metrics to Bolt.

    Args:
        metric_keys: List of metric names to log. If None, all numeric metrics are logged.
        metric_prefix: Optional prefix to add to metric names.
        log_non_specified_metrics: Whether to log metrics not in metric_keys.
    """

    def __init__(
        self,
        metric_keys: Optional[List[str]] = None,
        metric_prefix: Optional[str] = None,
        log_non_specified_metrics: bool = True,
    ):
        self.metric_keys = metric_keys
        self.metric_prefix = metric_prefix
        self.log_non_specified_metrics = log_non_specified_metrics
        self._initialized = False
        self._step_start_time: Optional[float] = None
        self._step_times: List[float] = []
        self._window_size = 1

    def setup(self) -> None:
        """Set up the callback if not already initialized."""
        if self._initialized:
            return

        if not is_on_bolt():
            logger.warning("Not running on TuriBolt. BoltCallback will not log metrics.")
            return

        self._initialized = True
        logger.info("BoltCallback initialized for metric logging.")

    def _format_metric_name(self, name: str) -> str:
        """Add prefix to metric name if specified."""
        if self.metric_prefix:
            return f"{self.metric_prefix}/{name}"
        return name

    def _should_log_metric(self, metric_name: str) -> bool:
        """Determine if a metric should be logged based on configuration."""
        if self.metric_keys is None:
            return True
        if metric_name in self.metric_keys:
            return True
        return self.log_non_specified_metrics

    def on_init_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Called at the end of Trainer initialization."""
        self.setup()

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        """Called when the trainer logs metrics."""
        if not state.is_world_process_zero or not self._initialized or logs is None:
            return

        metrics = {}
        for k, v in logs.items():
            if isinstance(v, (int, float)) and self._should_log_metric(k):
                metrics[self._format_metric_name(k)] = v

        if metrics:
            bolt.send_metrics(metrics)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        """Called after an evaluation phase."""
        if not state.is_world_process_zero or not self._initialized or metrics is None:
            return

        eval_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and self._should_log_metric(k):
                eval_metrics[self._format_metric_name(k)] = v

        if eval_metrics:
            bolt.send_metrics(eval_metrics)

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Record the start time of each training step."""
        if not state.is_world_process_zero or not self._initialized:
            return
        self._step_start_time = time.time()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Calculate step time and log metrics."""
        if not state.is_world_process_zero or not self._initialized or self._step_start_time is None:
            return

        step_time = time.time() - self._step_start_time
        self._step_times.append(step_time)
        if len(self._step_times) > self._window_size:
            self._step_times.pop(0)

        avg_step_time = sum(self._step_times) / len(self._step_times)
        samples_per_step = (
            args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * getattr(args, "world_size", 1)
        )
        avg_time_per_sample = avg_step_time / samples_per_step if samples_per_step > 0 else 0

        metrics = {
            self._format_metric_name("avg_step_time_seconds"): round(avg_step_time, 3),
            self._format_metric_name("avg_time_per_sample_seconds"): round(avg_time_per_sample, 3),
        }
        bolt.send_metrics(metrics)


def get_bolt_callback(metric_prefix: Optional[str] = None) -> Optional[BoltCallback]:
    """Get a BoltCallback if running on Bolt, otherwise None.

    Args:
        metric_prefix: Optional prefix to add to all metric names.

    Returns:
        BoltCallback instance if on Bolt, None otherwise.
    """
    if is_on_bolt():
        return BoltCallback(metric_prefix=metric_prefix)
    return None
