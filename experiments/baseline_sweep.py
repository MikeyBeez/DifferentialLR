"""Baseline sweep experiment: vary chunk sizes with fixed LR."""

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


def run_chunk_size_sweep(
    chunk_sizes: List[int] = [16, 32, 64, 128, 256, 512],
    gamma: float = 0.95,
    max_steps: int = 10000,
    output_base: str = "outputs/chunk_sweep",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run sweep over chunk sizes with fixed gamma.

    Args:
        chunk_sizes: List of chunk sizes to test
        gamma: Fixed gamma value
        max_steps: Training steps per configuration
        output_base: Base output directory
        seed: Random seed

    Returns:
        Dict with results for each chunk size
    """
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_base) / timestamp

    for chunk_size in chunk_sizes:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running experiment: chunk_size={chunk_size}, gamma={gamma}")
        logger.info(f"{'='*50}\n")

        # Create config
        model_config = ModelConfig(
            chunk_size=chunk_size,
            gamma=gamma,
        )

        training_config = TrainingConfig(
            max_steps=max_steps,
            output_dir=str(output_dir),
        )

        config = ExperimentConfig(
            name=f"chunk_{chunk_size}_gamma_{gamma}",
            model=model_config,
            training=training_config,
            seed=seed,
        )

        try:
            # Train
            metrics = train_from_config(config)
            results[f"chunk_{chunk_size}"] = {
                "chunk_size": chunk_size,
                "gamma": gamma,
                "best_eval_loss": metrics["best_eval_loss"],
                "final_train_loss": metrics["final_train_loss"],
                "status": "success",
            }

            logger.info(f"Completed chunk_size={chunk_size}: "
                       f"best_loss={metrics['best_eval_loss']:.4f}")

        except Exception as e:
            logger.error(f"Failed for chunk_size={chunk_size}: {e}")
            results[f"chunk_{chunk_size}"] = {
                "chunk_size": chunk_size,
                "gamma": gamma,
                "status": "failed",
                "error": str(e),
            }

        # Clear GPU memory between runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save summary
    summary_path = output_dir / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nSweep complete. Results saved to {summary_path}")

    return results


def run_gamma_sweep(
    gammas: List[float] = [0.9, 0.95, 0.99, 0.999],
    chunk_size: int = 64,
    max_steps: int = 10000,
    output_base: str = "outputs/gamma_sweep",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run sweep over gamma values with fixed chunk size.

    Args:
        gammas: List of gamma values to test
        chunk_size: Fixed chunk size
        max_steps: Training steps per configuration
        output_base: Base output directory
        seed: Random seed

    Returns:
        Dict with results for each gamma
    """
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_base) / timestamp

    for gamma in gammas:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running experiment: chunk_size={chunk_size}, gamma={gamma}")
        logger.info(f"{'='*50}\n")

        model_config = ModelConfig(
            chunk_size=chunk_size,
            gamma=gamma,
        )

        training_config = TrainingConfig(
            max_steps=max_steps,
            output_dir=str(output_dir),
        )

        config = ExperimentConfig(
            name=f"chunk_{chunk_size}_gamma_{gamma}",
            model=model_config,
            training=training_config,
            seed=seed,
        )

        try:
            metrics = train_from_config(config)
            results[f"gamma_{gamma}"] = {
                "chunk_size": chunk_size,
                "gamma": gamma,
                "best_eval_loss": metrics["best_eval_loss"],
                "final_train_loss": metrics["final_train_loss"],
                "status": "success",
            }

        except Exception as e:
            logger.error(f"Failed for gamma={gamma}: {e}")
            results[f"gamma_{gamma}"] = {
                "chunk_size": chunk_size,
                "gamma": gamma,
                "status": "failed",
                "error": str(e),
            }

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = output_dir / "gamma_sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_full_grid(
    chunk_sizes: List[int] = [32, 64, 128],
    gammas: List[float] = [0.9, 0.95, 0.99],
    max_steps: int = 5000,
    output_base: str = "outputs/full_grid",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run full grid search over chunk sizes and gammas.

    Args:
        chunk_sizes: List of chunk sizes
        gammas: List of gamma values
        max_steps: Training steps per configuration
        output_base: Base output directory
        seed: Random seed

    Returns:
        Dict with results for each configuration
    """
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_base) / timestamp

    total_runs = len(chunk_sizes) * len(gammas)
    current_run = 0

    for chunk_size in chunk_sizes:
        for gamma in gammas:
            current_run += 1
            logger.info(f"\n{'='*50}")
            logger.info(f"Run {current_run}/{total_runs}: "
                       f"chunk_size={chunk_size}, gamma={gamma}")
            logger.info(f"{'='*50}\n")

            model_config = ModelConfig(
                chunk_size=chunk_size,
                gamma=gamma,
            )

            training_config = TrainingConfig(
                max_steps=max_steps,
                output_dir=str(output_dir),
            )

            config = ExperimentConfig(
                name=f"chunk_{chunk_size}_gamma_{gamma}",
                model=model_config,
                training=training_config,
                seed=seed,
            )

            try:
                metrics = train_from_config(config)
                key = f"chunk_{chunk_size}_gamma_{gamma}"
                results[key] = {
                    "chunk_size": chunk_size,
                    "gamma": gamma,
                    "best_eval_loss": metrics["best_eval_loss"],
                    "final_train_loss": metrics["final_train_loss"],
                    "status": "success",
                }

            except Exception as e:
                logger.error(f"Failed: {e}")
                results[f"chunk_{chunk_size}_gamma_{gamma}"] = {
                    "chunk_size": chunk_size,
                    "gamma": gamma,
                    "status": "failed",
                    "error": str(e),
                }

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary_path = output_dir / "grid_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    # Find best configuration
    best_config = None
    best_loss = float("inf")
    for key, result in results.items():
        if result.get("status") == "success":
            if result["best_eval_loss"] < best_loss:
                best_loss = result["best_eval_loss"]
                best_config = key

    logger.info(f"\nBest configuration: {best_config} with loss {best_loss:.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline sweep experiments")
    parser.add_argument(
        "--mode", type=str, default="chunk",
        choices=["chunk", "gamma", "grid"],
        help="Sweep mode: chunk sizes, gamma values, or full grid"
    )
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.mode == "chunk":
        run_chunk_size_sweep(
            max_steps=args.max_steps,
            output_base=f"{args.output_dir}/chunk_sweep",
            seed=args.seed,
        )
    elif args.mode == "gamma":
        run_gamma_sweep(
            max_steps=args.max_steps,
            output_base=f"{args.output_dir}/gamma_sweep",
            seed=args.seed,
        )
    else:
        run_full_grid(
            max_steps=args.max_steps,
            output_base=f"{args.output_dir}/full_grid",
            seed=args.seed,
        )
