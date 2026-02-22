#!/usr/bin/env python3
"""
Hyperparameter tuning script for HuBERT ASR training.

Runs training with different hyperparameter combinations and tracks the best results.

Usage:
    python project/scripts/tune_hparams.py
    python project/scripts/tune_hparams.py --base-config project/configs/gpu.yaml --epochs 10
"""

import argparse
import csv
import itertools
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


# Hyperparameter search space
HPARAM_GRID = {
    "learning_rate": [5e-5, 1e-4, 2e-4],
    "lr_scheduler_type": ["cosine", "linear"],
    "weight_decay": [0.0, 0.005, 0.01],
    "warmup_steps": [200, 500, 1000],
}


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_config(config: dict, config_path: str):
    """Save config to YAML file."""
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def run_training(config_path: str, run_name: str) -> float | None:
    """Run training and return best validation WER."""
    cmd = [
        "accelerate", "launch", "--multi_gpu",
        "-m", "project.src.train",
        "--config", config_path,
    ]

    print(f"\n{'='*60}")
    print(f"Running: {run_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600 * 12,  # 12 hour timeout
        )

        # Print output
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        # Extract best WER from output
        # Look for lines like "Best WER: 0.0523" or "wer': 0.0523"
        best_wer = None
        for line in result.stdout.split("\n"):
            if "best" in line.lower() and "wer" in line.lower():
                # Try to extract number
                import re
                match = re.search(r"(\d+\.\d+)", line)
                if match:
                    best_wer = float(match.group(1))
            elif "'wer':" in line or '"wer":' in line:
                import re
                match = re.search(r"(\d+\.\d+)", line)
                if match:
                    wer = float(match.group(1))
                    if best_wer is None or wer < best_wer:
                        best_wer = wer

        return best_wer

    except subprocess.TimeoutExpired:
        print(f"ERROR: Training timed out for {run_name}")
        return None
    except Exception as e:
        print(f"ERROR: Training failed for {run_name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for HuBERT ASR")
    parser.add_argument(
        "--base-config",
        type=str,
        default="project/configs/gpu.yaml",
        help="Base configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="project/outputs/hparam_tuning",
        help="Output directory for results",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs (for faster tuning)",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Results tracking
    results_file = output_dir / "tuning_results.csv"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load base config
    base_config = load_config(args.base_config)

    # Generate all combinations
    hparam_names = list(HPARAM_GRID.keys())
    hparam_values = list(HPARAM_GRID.values())
    combinations = list(itertools.product(*hparam_values))

    print(f"=== Hyperparameter Tuning ===")
    print(f"Base config: {args.base_config}")
    print(f"Total combinations: {len(combinations)}")
    print(f"Parameters: {hparam_names}")
    print(f"Results file: {results_file}")
    print()

    # Initialize results file
    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(hparam_names + ["wer", "status"])

    # Track best result
    best_wer = float("inf")
    best_hparams = None

    for i, combo in enumerate(combinations, 1):
        hparams = dict(zip(hparam_names, combo))

        # Create run name
        run_name = f"run_{i:03d}_lr{hparams['learning_rate']}_wd{hparams['weight_decay']}"

        print(f"\n{'#'*60}")
        print(f"# Combination {i}/{len(combinations)}")
        print(f"# {hparams}")
        print(f"{'#'*60}")

        # Modify config
        config = base_config.copy()
        config["training"] = config.get("training", {}).copy()
        config["training"]["learning_rate"] = hparams["learning_rate"]
        config["training"]["lr_scheduler_type"] = hparams["lr_scheduler_type"]
        config["training"]["weight_decay"] = hparams["weight_decay"]
        config["training"]["warmup_steps"] = hparams["warmup_steps"]
        config["training"]["output_dir"] = str(output_dir / run_name)

        if args.epochs:
            config["training"]["num_train_epochs"] = args.epochs

        # Save temp config
        temp_config_path = output_dir / f"config_{run_name}.yaml"
        save_config(config, str(temp_config_path))

        # Run training
        wer = run_training(str(temp_config_path), run_name)

        # Record result
        status = "success" if wer is not None else "failed"
        with open(results_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(list(combo) + [wer if wer else "N/A", status])

        # Update best
        if wer is not None and wer < best_wer:
            best_wer = wer
            best_hparams = hparams.copy()

        # Print current best
        print(f"\n{'='*60}")
        print(f"Completed {i}/{len(combinations)}")
        if best_hparams:
            print(f"BEST SO FAR: WER = {best_wer:.4f}")
            print(f"  learning_rate: {best_hparams['learning_rate']}")
            print(f"  lr_scheduler_type: {best_hparams['lr_scheduler_type']}")
            print(f"  weight_decay: {best_hparams['weight_decay']}")
            print(f"  warmup_steps: {best_hparams['warmup_steps']}")
        print(f"{'='*60}")

    # Final summary
    print(f"\n{'='*60}")
    print("TUNING COMPLETE")
    print(f"{'='*60}")
    print(f"Total runs: {len(combinations)}")
    print(f"Results saved to: {results_file}")
    if best_hparams:
        print(f"\nOPTIMAL HYPERPARAMETERS:")
        print(f"  WER: {best_wer:.4f}")
        for k, v in best_hparams.items():
            print(f"  {k}: {v}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
