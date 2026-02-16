"""Device detection and management utilities."""

from typing import Any, Dict

import torch


class DeviceManager:
    """Handle device detection and configuration for training."""

    @staticmethod
    def get_device(prefer: str = "auto") -> torch.device:
        """Detect and return appropriate compute device.

        Args:
            prefer: Device preference. Options:
                - "auto": Automatically detect best available (CUDA > MPS > CPU)
                - "auto_no_mps": Automatically detect (CUDA > CPU, skips MPS)
                - "cuda": Force CUDA (raises error if unavailable)
                - "mps": Force MPS (raises error if unavailable)
                - "cpu": Force CPU

        Returns:
            torch.device for the selected compute device.

        Raises:
            RuntimeError: If requested device is not available.
        """
        if prefer == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")

        elif prefer == "auto_no_mps":
            # Skip MPS, use CUDA if available, otherwise CPU
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")

        elif prefer == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            return torch.device("cuda")

        elif prefer == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS requested but not available")
            return torch.device("mps")

        elif prefer == "cpu":
            return torch.device("cpu")

        else:
            raise ValueError(f"Unknown device preference: {prefer}")

    @staticmethod
    def is_fp16_supported(device: torch.device) -> bool:
        """Check if FP16 (mixed precision) training is supported.

        Args:
            device: The compute device to check.

        Returns:
            True if FP16 is fully supported, False otherwise.
        """
        if device.type == "cuda":
            return True
        elif device.type == "mps":
            # MPS has limited FP16 support, disable for stability
            return False
        else:
            # CPU doesn't benefit from FP16
            return False

    @staticmethod
    def get_device_specific_training_args(device: torch.device) -> Dict[str, Any]:
        """Get device-specific training arguments.

        Args:
            device: The compute device.

        Returns:
            Dictionary of training arguments specific to the device.
        """
        args = {}

        if device.type == "cuda":
            # CUDA supports FP16 and multiple workers
            args["fp16"] = True
            args["dataloader_num_workers"] = 4

        elif device.type == "mps":
            # MPS: disable FP16, use fewer workers for stability
            args["fp16"] = False
            args["dataloader_num_workers"] = 0
            # MPS may need specific settings
            args["use_mps_device"] = True

        else:  # CPU
            args["fp16"] = False
            args["dataloader_num_workers"] = 0

        return args

    @staticmethod
    def get_device_info(device: torch.device) -> str:
        """Get human-readable device information.

        Args:
            device: The compute device.

        Returns:
            String with device information.
        """
        if device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(device)
            gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
            return f"CUDA: {gpu_name} ({gpu_memory:.1f} GB)"

        elif device.type == "mps":
            return "Apple MPS (Metal Performance Shaders)"

        else:
            return "CPU"
