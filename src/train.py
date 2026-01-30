"""Training loop with diagnostics integration."""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, Any, List
import time
import json
import logging
from pathlib import Path
from tqdm import tqdm
import math

from .config import ModelConfig, TrainingConfig, ExperimentConfig
from .model import ChunkedTransformerModel, create_model
from .data import create_tokenizer, create_dataloaders, InfiniteDataLoader
from .diagnostics import DiagnosticsManager


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Training loop with full diagnostic support.

    Features:
    - Mixed precision training
    - Gradient accumulation
    - Differential learning rates
    - Checkpoint saving
    - Integrated diagnostics
    """

    def __init__(
        self,
        model: ChunkedTransformerModel,
        config: TrainingConfig,
        model_config: ModelConfig,
        output_dir: str,
        device: torch.device = None,
    ):
        self.model = model
        self.config = config
        self.model_config = model_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Setup optimizer with differential LR
        self.optimizer = self._create_optimizer()

        # Setup scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision
        self.scaler = GradScaler() if config.use_amp else None

        # Diagnostics
        self.diagnostics = DiagnosticsManager(
            model=model,
            output_dir=str(self.output_dir / "diagnostics"),
            track_gradients=config.track_gradients,
            track_magnitudes=config.track_magnitudes,
        )

        # Training state
        self.global_step = 0
        self.best_eval_loss = float("inf")
        self.train_losses: List[float] = []
        self.eval_losses: List[float] = []

    def _create_optimizer(self) -> AdamW:
        """Create optimizer with parameter groups for differential LR."""
        param_groups = self.model.get_parameter_groups(
            within_chunk_lr=self.config.within_chunk_lr,
            cross_chunk_lr=self.config.cross_chunk_lr,
            weight_decay=self.config.weight_decay,
        )

        return AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.max_steps is not None:
            total_steps = self.config.max_steps
        else:
            # Estimate from epochs (will be updated when we have dataloader)
            total_steps = 100000

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.config.warmup_steps,
        )

        if self.config.lr_scheduler == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - self.config.warmup_steps,
                eta_min=self.config.within_chunk_lr * 0.1,
            )
        elif self.config.lr_scheduler == "linear":
            main_scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=total_steps - self.config.warmup_steps,
            )
        else:  # constant
            main_scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=1.0,
                total_iters=total_steps - self.config.warmup_steps,
            )

        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[self.config.warmup_steps],
        )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()

        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Forward pass with optional intermediates
        track_this_step = (
            self.global_step % self.config.diagnostic_interval == 0
            and self.config.track_magnitudes
        )

        with autocast(enabled=self.config.use_amp):
            outputs = self.model(
                input_ids=input_ids,
                labels=labels,
                chunk_size=self.model_config.chunk_size,
                return_intermediates=track_this_step,
            )
            loss = outputs["loss"] / self.config.gradient_accumulation_steps

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Track intermediates
        if track_this_step and "intermediates" in outputs:
            self.diagnostics.on_forward(outputs["intermediates"])

        return {"loss": loss.item() * self.config.gradient_accumulation_steps}

    def optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.scheduler.step()
        self.optimizer.zero_grad()

        # Track learning speed
        self.diagnostics.on_optimizer_step()

    @torch.no_grad()
    def evaluate(self, eval_loader) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()

        total_loss = 0.0
        total_tokens = 0

        for batch in tqdm(eval_loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            with autocast(enabled=self.config.use_amp):
                outputs = self.model(
                    input_ids=input_ids,
                    labels=labels,
                    chunk_size=self.model_config.chunk_size,
                )

            # Count non-padding tokens
            num_tokens = (labels != -100).sum().item()
            total_loss += outputs["loss"].item() * num_tokens
            total_tokens += num_tokens

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)

        return {
            "eval_loss": avg_loss,
            "perplexity": perplexity,
        }

    def save_checkpoint(self, name: str = "checkpoint"):
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / f"{name}.pt"

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "config": {
                "model": self.model_config.__dict__,
                "training": self.config.__dict__,
            },
            "train_losses": self.train_losses,
            "eval_losses": self.eval_losses,
            "best_eval_loss": self.best_eval_loss,
        }, checkpoint_path)

        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.train_losses = checkpoint.get("train_losses", [])
        self.eval_losses = checkpoint.get("eval_losses", [])
        self.best_eval_loss = checkpoint.get("best_eval_loss", float("inf"))

        logger.info(f"Loaded checkpoint from {checkpoint_path} at step {self.global_step}")

    def train(
        self,
        train_loader,
        eval_loader,
        max_steps: Optional[int] = None,
    ):
        """
        Main training loop.

        Args:
            train_loader: Training dataloader
            eval_loader: Evaluation dataloader
            max_steps: Maximum training steps (overrides config)
        """
        max_steps = max_steps or self.config.max_steps

        # Make train loader infinite for step-based training
        train_iter = InfiniteDataLoader(train_loader)

        # Progress bar
        pbar = tqdm(total=max_steps, initial=self.global_step, desc="Training")

        accumulation_loss = 0.0
        start_time = time.time()

        while self.global_step < max_steps:
            for accum_step in range(self.config.gradient_accumulation_steps):
                batch = next(train_iter)
                step_output = self.train_step(batch)
                accumulation_loss += step_output["loss"]

            # Optimizer step after accumulation
            self.optimizer_step()
            self.global_step += 1

            # Average loss over accumulation steps
            avg_loss = accumulation_loss / self.config.gradient_accumulation_steps
            self.train_losses.append(avg_loss)
            accumulation_loss = 0.0

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

            # Logging
            if self.global_step % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                steps_per_sec = self.global_step / elapsed

                log_entry = self.diagnostics.log_step(
                    step=self.global_step,
                    loss=avg_loss,
                    steps_per_sec=steps_per_sec,
                    lr=self.optimizer.param_groups[0]["lr"],
                )

                logger.info(
                    f"Step {self.global_step}: loss={avg_loss:.4f}, "
                    f"lr={log_entry['lr']:.2e}, steps/s={steps_per_sec:.2f}"
                )

            # Evaluation
            if self.global_step % self.config.eval_interval == 0:
                eval_metrics = self.evaluate(eval_loader)
                self.eval_losses.append(eval_metrics["eval_loss"])

                logger.info(
                    f"Eval at step {self.global_step}: "
                    f"loss={eval_metrics['eval_loss']:.4f}, "
                    f"ppl={eval_metrics['perplexity']:.2f}"
                )

                # Save best model
                if eval_metrics["eval_loss"] < self.best_eval_loss:
                    self.best_eval_loss = eval_metrics["eval_loss"]
                    self.save_checkpoint("best_model")

            # Periodic checkpoint
            if self.global_step % self.config.save_interval == 0:
                self.save_checkpoint(f"checkpoint_step_{self.global_step}")

            # Reset per-step diagnostics to save memory
            if self.global_step % self.config.diagnostic_interval == 0:
                self.diagnostics.reset_step_trackers()

        pbar.close()

        # Final save
        self.save_checkpoint("final_model")
        self.diagnostics.save_all(prefix="final_")

        # Save training curves
        self._save_training_curves()

        logger.info(f"Training complete. Best eval loss: {self.best_eval_loss:.4f}")

    def _save_training_curves(self):
        """Save training and eval loss curves."""
        curves = {
            "train_losses": self.train_losses,
            "eval_losses": self.eval_losses,
            "eval_steps": list(range(
                self.config.eval_interval,
                len(self.eval_losses) * self.config.eval_interval + 1,
                self.config.eval_interval,
            )),
        }

        with open(self.output_dir / "training_curves.json", "w") as f:
            json.dump(curves, f)

    def cleanup(self):
        """Clean up resources."""
        self.diagnostics.cleanup()


def train_from_config(config: ExperimentConfig):
    """
    Train model from experiment configuration.

    Args:
        config: Full experiment configuration

    Returns:
        Dict with final metrics
    """
    # Set seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # Create output directory
    output_dir = Path(config.training.output_dir) / config.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config.save(output_dir / "config.yaml")

    # Create tokenizer
    tokenizer = create_tokenizer()

    # Create model
    model = create_model(config.model)
    num_params = model.get_num_params()
    logger.info(f"Created model with {num_params:,} parameters")

    # Create dataloaders
    dataloaders = create_dataloaders(
        tokenizer=tokenizer,
        seq_length=config.training.seq_length,
        batch_size=config.training.batch_size,
        chunk_size=config.model.chunk_size,
        dataset_name=config.training.dataset,
        num_workers=config.training.num_workers,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        config=config.training,
        model_config=config.model,
        output_dir=str(output_dir),
    )

    # Train
    trainer.train(
        train_loader=dataloaders["train"],
        eval_loader=dataloaders["eval"],
    )

    # Cleanup
    trainer.cleanup()

    return {
        "best_eval_loss": trainer.best_eval_loss,
        "final_train_loss": trainer.train_losses[-1] if trainer.train_losses else None,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train chunked linear attention model")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--chunk-size", type=int, default=64, help="Chunk size")
    parser.add_argument("--gamma", type=float, default=0.95, help="Gamma decay")
    parser.add_argument("--max-steps", type=int, default=10000, help="Max training steps")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--name", type=str, default="default", help="Experiment name")

    args = parser.parse_args()

    if args.config:
        config = ExperimentConfig.load(args.config)
    else:
        model_config = ModelConfig(
            chunk_size=args.chunk_size,
            gamma=args.gamma,
        )
        training_config = TrainingConfig(
            max_steps=args.max_steps,
            output_dir=args.output_dir,
        )
        config = ExperimentConfig(
            name=args.name,
            model=model_config,
            training=training_config,
        )

    train_from_config(config)
