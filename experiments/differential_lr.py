"""Differential learning rate experiments.

Test different LR ratios between within-chunk and cross-chunk components.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import logging
from datetime import datetime
from typing import List, Dict, Any

import torch

from src.config import ModelConfig, TrainingConfig, ExperimentConfig
from src.train import train_from_config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_lr_ratio_sweep(
    lr_ratios: List[float] = [0.1, 0.5, 1.0, 2.0, 10.0],
    base_lr: float = 3e-4,
    chunk_size: int = 64,
    gamma: float = 0.95,
    max_steps: int = 10000,
    output_base: str = "outputs/lr_sweep",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Sweep over learning rate ratios.

    cross_chunk_lr = base_lr * ratio

    Ratios:
    - < 1: Cross-chunk components learn slower
    - = 1: Equal learning rates (baseline)
    - > 1: Cross-chunk components learn faster

    Args:
        lr_ratios: Ratios to test (cross_lr / within_lr)
        base_lr: Base learning rate for within-chunk
        chunk_size: Chunk size to use
        gamma: Gamma value to use
        max_steps: Training steps per configuration
        output_base: Base output directory
        seed: Random seed

    Returns:
        Dict with results for each ratio
    """
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_base) / timestamp

    for ratio in lr_ratios:
        cross_lr = base_lr * ratio

        logger.info(f"\n{'='*50}")
        logger.info(f"LR Ratio experiment: ratio={ratio}")
        logger.info(f"  within_lr={base_lr:.2e}, cross_lr={cross_lr:.2e}")
        logger.info(f"{'='*50}\n")

        model_config = ModelConfig(
            chunk_size=chunk_size,
            gamma=gamma,
        )

        training_config = TrainingConfig(
            within_chunk_lr=base_lr,
            cross_chunk_lr=cross_lr,
            max_steps=max_steps,
            output_dir=str(output_dir),
        )

        config = ExperimentConfig(
            name=f"lr_ratio_{ratio}",
            model=model_config,
            training=training_config,
            seed=seed,
        )

        try:
            metrics = train_from_config(config)
            results[f"ratio_{ratio}"] = {
                "lr_ratio": ratio,
                "within_lr": base_lr,
                "cross_lr": cross_lr,
                "chunk_size": chunk_size,
                "gamma": gamma,
                "best_eval_loss": metrics["best_eval_loss"],
                "final_train_loss": metrics["final_train_loss"],
                "status": "success",
            }

            logger.info(f"Completed ratio={ratio}: "
                       f"best_loss={metrics['best_eval_loss']:.4f}")

        except Exception as e:
            logger.error(f"Failed for ratio={ratio}: {e}")
            results[f"ratio_{ratio}"] = {
                "lr_ratio": ratio,
                "status": "failed",
                "error": str(e),
            }

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = output_dir / "lr_sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    # Analysis
    successful = [r for r in results.values() if r.get("status") == "success"]
    if successful:
        best = min(successful, key=lambda x: x["best_eval_loss"])
        logger.info(f"\nBest LR ratio: {best['lr_ratio']} "
                   f"with loss {best['best_eval_loss']:.4f}")

    return results


def run_lr_scale_experiment(
    scales: List[float] = [0.1, 0.3, 1.0, 3.0, 10.0],
    chunk_size: int = 64,
    gamma: float = 0.95,
    max_steps: int = 10000,
    output_base: str = "outputs/lr_scale",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Scale both learning rates together to find optimal absolute values.

    Args:
        scales: Multipliers for default LR (3e-4)
        chunk_size: Chunk size to use
        gamma: Gamma value to use
        max_steps: Training steps per configuration
        output_base: Base output directory
        seed: Random seed

    Returns:
        Dict with results for each scale
    """
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_base) / timestamp
    base = 3e-4

    for scale in scales:
        lr = base * scale

        logger.info(f"\n{'='*50}")
        logger.info(f"LR Scale experiment: scale={scale}, lr={lr:.2e}")
        logger.info(f"{'='*50}\n")

        model_config = ModelConfig(
            chunk_size=chunk_size,
            gamma=gamma,
        )

        training_config = TrainingConfig(
            within_chunk_lr=lr,
            cross_chunk_lr=lr,
            max_steps=max_steps,
            output_dir=str(output_dir),
        )

        config = ExperimentConfig(
            name=f"lr_scale_{scale}",
            model=model_config,
            training=training_config,
            seed=seed,
        )

        try:
            metrics = train_from_config(config)
            results[f"scale_{scale}"] = {
                "scale": scale,
                "lr": lr,
                "best_eval_loss": metrics["best_eval_loss"],
                "final_train_loss": metrics["final_train_loss"],
                "status": "success",
            }

        except Exception as e:
            logger.error(f"Failed for scale={scale}: {e}")
            results[f"scale_{scale}"] = {
                "scale": scale,
                "status": "failed",
                "error": str(e),
            }

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = output_dir / "lr_scale_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_combined_sweep(
    lr_ratios: List[float] = [0.1, 0.5, 1.0, 2.0],
    chunk_sizes: List[int] = [32, 64, 128],
    gamma: float = 0.95,
    base_lr: float = 3e-4,
    max_steps: int = 5000,
    output_base: str = "outputs/combined_sweep",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Combined sweep over LR ratios and chunk sizes.

    Args:
        lr_ratios: LR ratios to test
        chunk_sizes: Chunk sizes to test
        gamma: Fixed gamma
        base_lr: Base learning rate
        max_steps: Steps per config
        output_base: Output directory
        seed: Random seed

    Returns:
        Dict with all results
    """
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_base) / timestamp

    total = len(lr_ratios) * len(chunk_sizes)
    current = 0

    for chunk_size in chunk_sizes:
        for ratio in lr_ratios:
            current += 1
            cross_lr = base_lr * ratio

            logger.info(f"\n{'='*50}")
            logger.info(f"Run {current}/{total}: "
                       f"chunk={chunk_size}, ratio={ratio}")
            logger.info(f"{'='*50}\n")

            model_config = ModelConfig(
                chunk_size=chunk_size,
                gamma=gamma,
            )

            training_config = TrainingConfig(
                within_chunk_lr=base_lr,
                cross_chunk_lr=cross_lr,
                max_steps=max_steps,
                output_dir=str(output_dir),
            )

            config = ExperimentConfig(
                name=f"chunk_{chunk_size}_ratio_{ratio}",
                model=model_config,
                training=training_config,
                seed=seed,
            )

            try:
                metrics = train_from_config(config)
                key = f"chunk_{chunk_size}_ratio_{ratio}"
                results[key] = {
                    "chunk_size": chunk_size,
                    "lr_ratio": ratio,
                    "within_lr": base_lr,
                    "cross_lr": cross_lr,
                    "best_eval_loss": metrics["best_eval_loss"],
                    "final_train_loss": metrics["final_train_loss"],
                    "status": "success",
                }

            except Exception as e:
                logger.error(f"Failed: {e}")
                results[f"chunk_{chunk_size}_ratio_{ratio}"] = {
                    "chunk_size": chunk_size,
                    "lr_ratio": ratio,
                    "status": "failed",
                    "error": str(e),
                }

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary_path = output_dir / "combined_sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Differential LR experiments")
    parser.add_argument(
        "--mode", type=str, default="ratio",
        choices=["ratio", "scale", "combined"],
        help="Experiment mode"
    )
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.mode == "ratio":
        run_lr_ratio_sweep(
            chunk_size=args.chunk_size,
            max_steps=args.max_steps,
            output_base=f"{args.output_dir}/lr_ratio",
            seed=args.seed,
        )
    elif args.mode == "scale":
        run_lr_scale_experiment(
            chunk_size=args.chunk_size,
            max_steps=args.max_steps,
            output_base=f"{args.output_dir}/lr_scale",
            seed=args.seed,
        )
    else:
        run_combined_sweep(
            max_steps=args.max_steps,
            output_base=f"{args.output_dir}/combined",
            seed=args.seed,
        )
