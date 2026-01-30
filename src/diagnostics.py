"""Gradient and magnitude tracking for research diagnostics.

This module provides tools to analyze training dynamics:
- GradientTracker: Track gradient norms per parameter group
- MagnitudeTracker: Track forward pass output magnitudes
- LearningSpeedTracker: Track parameter update magnitudes over time
"""

import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Optional, Any, Callable
import json
from pathlib import Path


class GradientTracker:
    """
    Track gradient norms during training.

    Registers hooks on parameters to capture gradient information:
    - Per-parameter gradient norms
    - Within-chunk vs cross-chunk gradient magnitude comparison
    - Gradient flow through chunk boundaries
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.grad_norms: Dict[str, List[float]] = defaultdict(list)
        self.grad_stats: Dict[str, Dict[str, float]] = {}
        self._hooks = []
        self._enabled = True

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register gradient hooks on all parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(
                    lambda grad, n=name: self._grad_hook(n, grad)
                )
                self._hooks.append(hook)

    def _grad_hook(self, name: str, grad: torch.Tensor):
        """Hook function called during backward pass."""
        if self._enabled and grad is not None:
            norm = grad.norm().item()
            self.grad_norms[name].append(norm)

    def enable(self):
        """Enable gradient tracking."""
        self._enabled = True

    def disable(self):
        """Disable gradient tracking."""
        self._enabled = False

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all parameters."""
        summary = {}
        for name, norms in self.grad_norms.items():
            if norms:
                summary[name] = {
                    "mean": sum(norms) / len(norms),
                    "max": max(norms),
                    "min": min(norms),
                    "last": norms[-1],
                    "count": len(norms),
                }
        return summary

    def get_grouped_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary grouped by within-chunk vs cross-chunk."""
        within_norms = []
        cross_norms = []

        for name, norms in self.grad_norms.items():
            if "cross_" in name or "gate" in name or "state" in name:
                cross_norms.extend(norms)
            else:
                within_norms.extend(norms)

        summary = {}
        if within_norms:
            summary["within_chunk"] = {
                "mean": sum(within_norms) / len(within_norms),
                "max": max(within_norms),
                "min": min(within_norms),
            }
        if cross_norms:
            summary["cross_chunk"] = {
                "mean": sum(cross_norms) / len(cross_norms),
                "max": max(cross_norms),
                "min": min(cross_norms),
            }

        return summary

    def reset(self):
        """Reset tracked gradients."""
        self.grad_norms = defaultdict(list)

    def remove_hooks(self):
        """Remove all gradient hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def save(self, path: str):
        """Save gradient history to JSON."""
        with open(path, "w") as f:
            json.dump({
                "grad_norms": dict(self.grad_norms),
                "summary": self.get_summary(),
                "grouped_summary": self.get_grouped_summary(),
            }, f, indent=2)


class MagnitudeTracker:
    """
    Track output magnitudes during forward passes.

    Monitors:
    - Within-chunk attention output magnitude
    - Cross-chunk state contribution magnitude
    - Layer-wise output magnitudes
    """

    def __init__(self):
        self.within_chunk_magnitudes: List[float] = []
        self.cross_chunk_magnitudes: List[float] = []
        self.layer_magnitudes: Dict[int, List[float]] = defaultdict(list)
        self.state_magnitudes: List[List[float]] = []  # Per chunk position

    def track_forward(
        self,
        intermediates: Dict[str, Any],
        layer_idx: Optional[int] = None,
    ):
        """
        Track magnitudes from forward pass intermediates.

        Args:
            intermediates: Dict from attention layer with 'within_chunk_output',
                          'cross_magnitudes', 'state_history', etc.
            layer_idx: Optional layer index for layer-wise tracking
        """
        if intermediates is None:
            return

        # Track within-chunk output magnitude
        if "within_chunk_output" in intermediates:
            within_out = intermediates["within_chunk_output"]
            mag = within_out.norm(dim=-1).mean().item()
            self.within_chunk_magnitudes.append(mag)
            if layer_idx is not None:
                self.layer_magnitudes[layer_idx].append(mag)

        # Track cross-chunk contribution magnitudes
        if "cross_magnitudes" in intermediates:
            self.cross_chunk_magnitudes.extend(intermediates["cross_magnitudes"])

        # Track state history
        if "state_history" in intermediates:
            state_hist = intermediates["state_history"]
            mags = state_hist.norm(dim=-1).mean(dim=0).tolist()
            self.state_magnitudes.append(mags)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of tracked magnitudes."""
        summary = {}

        if self.within_chunk_magnitudes:
            summary["within_chunk"] = {
                "mean": sum(self.within_chunk_magnitudes) / len(self.within_chunk_magnitudes),
                "max": max(self.within_chunk_magnitudes),
                "min": min(self.within_chunk_magnitudes),
            }

        if self.cross_chunk_magnitudes:
            summary["cross_chunk"] = {
                "mean": sum(self.cross_chunk_magnitudes) / len(self.cross_chunk_magnitudes),
                "max": max(self.cross_chunk_magnitudes),
                "min": min(self.cross_chunk_magnitudes),
            }

        if self.within_chunk_magnitudes and self.cross_chunk_magnitudes:
            within_mean = sum(self.within_chunk_magnitudes) / len(self.within_chunk_magnitudes)
            cross_mean = sum(self.cross_chunk_magnitudes) / len(self.cross_chunk_magnitudes)
            summary["ratio"] = cross_mean / (within_mean + 1e-8)

        if self.layer_magnitudes:
            summary["per_layer"] = {
                layer: {
                    "mean": sum(mags) / len(mags),
                }
                for layer, mags in self.layer_magnitudes.items()
            }

        return summary

    def get_state_magnitude_by_position(self) -> Dict[str, List[float]]:
        """Get average state magnitude per chunk position."""
        if not self.state_magnitudes:
            return {}

        # Transpose to get per-position data
        num_positions = len(self.state_magnitudes[0]) if self.state_magnitudes else 0
        per_position = defaultdict(list)

        for step_mags in self.state_magnitudes:
            for pos, mag in enumerate(step_mags):
                per_position[pos].append(mag)

        return {
            f"position_{pos}": {
                "mean": sum(mags) / len(mags),
                "values": mags,
            }
            for pos, mags in per_position.items()
        }

    def reset(self):
        """Reset all tracked magnitudes."""
        self.within_chunk_magnitudes = []
        self.cross_chunk_magnitudes = []
        self.layer_magnitudes = defaultdict(list)
        self.state_magnitudes = []

    def save(self, path: str):
        """Save magnitude history to JSON."""
        with open(path, "w") as f:
            json.dump({
                "summary": self.get_summary(),
                "state_by_position": self.get_state_magnitude_by_position(),
                "within_chunk_history": self.within_chunk_magnitudes,
                "cross_chunk_history": self.cross_chunk_magnitudes,
            }, f, indent=2)


class LearningSpeedTracker:
    """
    Track parameter update magnitudes over training.

    Compares:
    - Update magnitude (new - old) relative to parameter magnitude
    - Within-chunk vs cross-chunk learning speeds
    - Per-layer learning speeds
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.prev_params: Dict[str, torch.Tensor] = {}
        self.update_magnitudes: Dict[str, List[float]] = defaultdict(list)
        self.relative_updates: Dict[str, List[float]] = defaultdict(list)

        # Store initial parameters
        self._store_params()

    def _store_params(self):
        """Store current parameter values."""
        for name, param in self.model.named_parameters():
            self.prev_params[name] = param.data.clone().detach()

    def step(self):
        """
        Track updates from last stored parameters to current.
        Call this after optimizer.step().
        """
        for name, param in self.model.named_parameters():
            if name in self.prev_params:
                # Compute update magnitude
                update = param.data - self.prev_params[name]
                update_mag = update.norm().item()
                param_mag = param.data.norm().item()

                self.update_magnitudes[name].append(update_mag)

                # Relative update (update / param magnitude)
                if param_mag > 1e-8:
                    self.relative_updates[name].append(update_mag / param_mag)

        # Store new params for next step
        self._store_params()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of learning speeds."""
        summary = {"per_param": {}, "grouped": {}}

        # Per-parameter summary
        for name, updates in self.update_magnitudes.items():
            if updates:
                summary["per_param"][name] = {
                    "mean_update": sum(updates) / len(updates),
                    "max_update": max(updates),
                }

        # Grouped summary
        within_updates = []
        cross_updates = []

        for name, updates in self.update_magnitudes.items():
            if "cross_" in name or "gate" in name or "state" in name:
                cross_updates.extend(updates)
            else:
                within_updates.extend(updates)

        if within_updates:
            summary["grouped"]["within_chunk"] = {
                "mean": sum(within_updates) / len(within_updates),
                "total": sum(within_updates),
            }
        if cross_updates:
            summary["grouped"]["cross_chunk"] = {
                "mean": sum(cross_updates) / len(cross_updates),
                "total": sum(cross_updates),
            }

        if within_updates and cross_updates:
            summary["grouped"]["ratio"] = (
                sum(cross_updates) / len(cross_updates)
            ) / (
                sum(within_updates) / len(within_updates) + 1e-8
            )

        return summary

    def get_relative_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of relative update magnitudes."""
        summary = {}

        within_rel = []
        cross_rel = []

        for name, updates in self.relative_updates.items():
            if updates:
                if "cross_" in name or "gate" in name or "state" in name:
                    cross_rel.extend(updates)
                else:
                    within_rel.extend(updates)

        if within_rel:
            summary["within_chunk"] = {
                "mean_relative_update": sum(within_rel) / len(within_rel),
            }
        if cross_rel:
            summary["cross_chunk"] = {
                "mean_relative_update": sum(cross_rel) / len(cross_rel),
            }

        return summary

    def reset(self):
        """Reset tracking."""
        self.update_magnitudes = defaultdict(list)
        self.relative_updates = defaultdict(list)
        self._store_params()

    def save(self, path: str):
        """Save learning speed history to JSON."""
        with open(path, "w") as f:
            json.dump({
                "summary": self.get_summary(),
                "relative_summary": self.get_relative_summary(),
            }, f, indent=2)


class DiagnosticsManager:
    """
    Unified manager for all diagnostic tracking.

    Provides:
    - Easy setup of all trackers
    - Coordinated logging
    - Combined analysis
    """

    def __init__(
        self,
        model: nn.Module,
        output_dir: str,
        track_gradients: bool = True,
        track_magnitudes: bool = True,
        track_learning_speed: bool = True,
    ):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.gradient_tracker = GradientTracker(model) if track_gradients else None
        self.magnitude_tracker = MagnitudeTracker() if track_magnitudes else None
        self.learning_speed_tracker = LearningSpeedTracker(model) if track_learning_speed else None

        self.step_count = 0
        self.log_history: List[Dict[str, Any]] = []

    def on_forward(self, intermediates: List[Dict[str, Any]]):
        """Call after forward pass with intermediates."""
        if self.magnitude_tracker is not None:
            for layer_idx, layer_intermediates in enumerate(intermediates):
                self.magnitude_tracker.track_forward(layer_intermediates, layer_idx)

    def on_backward(self):
        """Call after backward pass."""
        # Gradients are tracked automatically via hooks
        pass

    def on_optimizer_step(self):
        """Call after optimizer step."""
        if self.learning_speed_tracker is not None:
            self.learning_speed_tracker.step()
        self.step_count += 1

    def log_step(self, step: int, loss: float, **kwargs):
        """Log metrics for a training step."""
        log_entry = {
            "step": step,
            "loss": loss,
            **kwargs,
        }

        if self.gradient_tracker is not None:
            log_entry["grad_summary"] = self.gradient_tracker.get_grouped_summary()

        if self.magnitude_tracker is not None:
            log_entry["magnitude_summary"] = self.magnitude_tracker.get_summary()

        if self.learning_speed_tracker is not None:
            log_entry["learning_speed"] = self.learning_speed_tracker.get_summary()

        self.log_history.append(log_entry)
        return log_entry

    def reset_step_trackers(self):
        """Reset per-step trackers (call periodically to avoid memory buildup)."""
        if self.gradient_tracker is not None:
            self.gradient_tracker.reset()
        if self.magnitude_tracker is not None:
            self.magnitude_tracker.reset()

    def save_all(self, prefix: str = ""):
        """Save all diagnostic data."""
        if self.gradient_tracker is not None:
            self.gradient_tracker.save(self.output_dir / f"{prefix}gradients.json")

        if self.magnitude_tracker is not None:
            self.magnitude_tracker.save(self.output_dir / f"{prefix}magnitudes.json")

        if self.learning_speed_tracker is not None:
            self.learning_speed_tracker.save(self.output_dir / f"{prefix}learning_speed.json")

        # Save full log history
        with open(self.output_dir / f"{prefix}training_log.json", "w") as f:
            json.dump(self.log_history, f, indent=2)

    def cleanup(self):
        """Clean up hooks and resources."""
        if self.gradient_tracker is not None:
            self.gradient_tracker.remove_hooks()
