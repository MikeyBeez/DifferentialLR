"""Post-hoc analysis and visualization of experiment results."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import logging
from typing import Dict, Any, List, Optional
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Optional: matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available, plotting disabled")


def load_experiment_results(experiment_dir: str) -> Dict[str, Any]:
    """Load all results from an experiment directory."""
    exp_path = Path(experiment_dir)
    results = {}

    # Load sweep summary if exists
    for summary_file in exp_path.glob("*_summary.json"):
        with open(summary_file) as f:
            results["summary"] = json.load(f)

    # Load individual run results
    for run_dir in exp_path.iterdir():
        if run_dir.is_dir():
            run_results = {}

            # Training curves
            curves_file = run_dir / "training_curves.json"
            if curves_file.exists():
                with open(curves_file) as f:
                    run_results["curves"] = json.load(f)

            # Diagnostics
            diag_dir = run_dir / "diagnostics"
            if diag_dir.exists():
                for diag_file in diag_dir.glob("*.json"):
                    with open(diag_file) as f:
                        run_results[diag_file.stem] = json.load(f)

            if run_results:
                results[run_dir.name] = run_results

    return results


def analyze_chunk_size_sweep(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze chunk size sweep results."""
    analysis = {
        "chunk_sizes": [],
        "eval_losses": [],
        "train_losses": [],
    }

    summary = results.get("summary", {})
    for key, data in summary.items():
        if data.get("status") == "success":
            analysis["chunk_sizes"].append(data["chunk_size"])
            analysis["eval_losses"].append(data["best_eval_loss"])
            if data.get("final_train_loss"):
                analysis["train_losses"].append(data["final_train_loss"])

    # Sort by chunk size
    if analysis["chunk_sizes"]:
        sorted_indices = np.argsort(analysis["chunk_sizes"])
        analysis["chunk_sizes"] = [analysis["chunk_sizes"][i] for i in sorted_indices]
        analysis["eval_losses"] = [analysis["eval_losses"][i] for i in sorted_indices]
        if analysis["train_losses"]:
            analysis["train_losses"] = [analysis["train_losses"][i] for i in sorted_indices]

        # Find optimal
        best_idx = np.argmin(analysis["eval_losses"])
        analysis["optimal_chunk_size"] = analysis["chunk_sizes"][best_idx]
        analysis["optimal_loss"] = analysis["eval_losses"][best_idx]

    return analysis


def analyze_gamma_sweep(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze gamma sweep results."""
    analysis = {
        "gammas": [],
        "eval_losses": [],
    }

    summary = results.get("summary", {})
    for key, data in summary.items():
        if data.get("status") == "success":
            analysis["gammas"].append(data["gamma"])
            analysis["eval_losses"].append(data["best_eval_loss"])

    if analysis["gammas"]:
        sorted_indices = np.argsort(analysis["gammas"])
        analysis["gammas"] = [analysis["gammas"][i] for i in sorted_indices]
        analysis["eval_losses"] = [analysis["eval_losses"][i] for i in sorted_indices]

        best_idx = np.argmin(analysis["eval_losses"])
        analysis["optimal_gamma"] = analysis["gammas"][best_idx]
        analysis["optimal_loss"] = analysis["eval_losses"][best_idx]

    return analysis


def analyze_lr_ratio_sweep(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze LR ratio sweep results."""
    analysis = {
        "ratios": [],
        "eval_losses": [],
    }

    summary = results.get("summary", {})
    for key, data in summary.items():
        if data.get("status") == "success":
            analysis["ratios"].append(data["lr_ratio"])
            analysis["eval_losses"].append(data["best_eval_loss"])

    if analysis["ratios"]:
        sorted_indices = np.argsort(analysis["ratios"])
        analysis["ratios"] = [analysis["ratios"][i] for i in sorted_indices]
        analysis["eval_losses"] = [analysis["eval_losses"][i] for i in sorted_indices]

        best_idx = np.argmin(analysis["eval_losses"])
        analysis["optimal_ratio"] = analysis["ratios"][best_idx]
        analysis["optimal_loss"] = analysis["eval_losses"][best_idx]

    return analysis


def analyze_gradient_dynamics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze gradient dynamics across training."""
    analysis = {
        "within_chunk_grads": [],
        "cross_chunk_grads": [],
        "grad_ratios": [],
    }

    for run_name, run_data in results.items():
        if run_name == "summary":
            continue

        if "training_log" in run_data:
            for entry in run_data["training_log"]:
                grad_summary = entry.get("grad_summary", {})
                within = grad_summary.get("within_chunk", {})
                cross = grad_summary.get("cross_chunk", {})

                if within and cross:
                    within_mean = within.get("mean", 0)
                    cross_mean = cross.get("mean", 0)

                    analysis["within_chunk_grads"].append(within_mean)
                    analysis["cross_chunk_grads"].append(cross_mean)
                    if within_mean > 1e-10:
                        analysis["grad_ratios"].append(cross_mean / within_mean)

    if analysis["grad_ratios"]:
        analysis["mean_grad_ratio"] = np.mean(analysis["grad_ratios"])
        analysis["std_grad_ratio"] = np.std(analysis["grad_ratios"])

    return analysis


def analyze_magnitude_dynamics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze output magnitude dynamics."""
    analysis = {
        "within_magnitudes": [],
        "cross_magnitudes": [],
        "magnitude_ratios": [],
    }

    for run_name, run_data in results.items():
        if run_name == "summary":
            continue

        if "training_log" in run_data:
            for entry in run_data["training_log"]:
                mag_summary = entry.get("magnitude_summary", {})
                within = mag_summary.get("within_chunk", {})
                cross = mag_summary.get("cross_chunk", {})

                if within and cross:
                    analysis["within_magnitudes"].append(within.get("mean", 0))
                    analysis["cross_magnitudes"].append(cross.get("mean", 0))

                ratio = mag_summary.get("ratio")
                if ratio is not None:
                    analysis["magnitude_ratios"].append(ratio)

    if analysis["magnitude_ratios"]:
        analysis["mean_magnitude_ratio"] = np.mean(analysis["magnitude_ratios"])
        analysis["final_magnitude_ratio"] = analysis["magnitude_ratios"][-1]

    return analysis


def plot_chunk_size_sweep(analysis: Dict[str, Any], output_path: str):
    """Plot chunk size vs loss."""
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(analysis["chunk_sizes"], analysis["eval_losses"],
            'o-', linewidth=2, markersize=8, label="Eval Loss")

    if analysis.get("train_losses"):
        ax.plot(analysis["chunk_sizes"], analysis["train_losses"],
                's--', linewidth=2, markersize=6, label="Train Loss")

    ax.set_xlabel("Chunk Size", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Perplexity vs Chunk Size", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"Saved chunk size plot to {output_path}")


def plot_gamma_sweep(analysis: Dict[str, Any], output_path: str):
    """Plot gamma vs loss."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(analysis["gammas"], analysis["eval_losses"],
            'o-', linewidth=2, markersize=8)

    ax.set_xlabel("Gamma (decay factor)", fontsize=12)
    ax.set_ylabel("Eval Loss", fontsize=12)
    ax.set_title("Loss vs Gamma", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_lr_ratio_sweep(analysis: Dict[str, Any], output_path: str):
    """Plot LR ratio vs loss."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(analysis["ratios"], analysis["eval_losses"],
            'o-', linewidth=2, markersize=8)

    ax.set_xlabel("LR Ratio (cross/within)", fontsize=12)
    ax.set_ylabel("Eval Loss", fontsize=12)
    ax.set_title("Loss vs Learning Rate Ratio", fontsize=14)
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label="Equal LR")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_gradient_heatmap(results: Dict[str, Any], output_path: str):
    """Plot gradient magnitude heatmap over training."""
    if not HAS_MATPLOTLIB:
        return

    # Collect gradient data
    within_grads = []
    cross_grads = []
    steps = []

    for run_name, run_data in results.items():
        if run_name == "summary":
            continue

        if "training_log" in run_data:
            for entry in run_data["training_log"]:
                grad_summary = entry.get("grad_summary", {})
                within = grad_summary.get("within_chunk", {})
                cross = grad_summary.get("cross_chunk", {})

                if within and cross:
                    within_grads.append(within.get("mean", 0))
                    cross_grads.append(cross.get("mean", 0))
                    steps.append(entry.get("step", len(steps)))

    if not steps:
        logger.warning("No gradient data found for heatmap")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(steps, within_grads, linewidth=1, alpha=0.7)
    ax1.set_ylabel("Within-chunk Grad Norm", fontsize=11)
    ax1.set_title("Gradient Magnitudes Over Training", fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, cross_grads, linewidth=1, alpha=0.7, color='orange')
    ax2.set_xlabel("Training Step", fontsize=12)
    ax2.set_ylabel("Cross-chunk Grad Norm", fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def generate_report(
    results: Dict[str, Any],
    experiment_type: str,
    output_dir: str,
) -> str:
    """Generate a text report of the analysis."""
    report_lines = [
        f"# Chunked Linear Attention Experiment Report",
        f"## Experiment Type: {experiment_type}",
        "",
    ]

    if experiment_type == "chunk_sweep":
        analysis = analyze_chunk_size_sweep(results)
        report_lines.extend([
            "### Chunk Size Sweep Results",
            "",
            "| Chunk Size | Eval Loss |",
            "|------------|-----------|",
        ])
        for cs, loss in zip(analysis["chunk_sizes"], analysis["eval_losses"]):
            report_lines.append(f"| {cs} | {loss:.4f} |")

        report_lines.extend([
            "",
            f"**Optimal Chunk Size**: {analysis.get('optimal_chunk_size', 'N/A')}",
            f"**Best Loss**: {analysis.get('optimal_loss', 'N/A'):.4f}",
        ])

        if HAS_MATPLOTLIB:
            plot_chunk_size_sweep(analysis, f"{output_dir}/chunk_size_plot.png")

    elif experiment_type == "gamma_sweep":
        analysis = analyze_gamma_sweep(results)
        report_lines.extend([
            "### Gamma Sweep Results",
            "",
            "| Gamma | Eval Loss |",
            "|-------|-----------|",
        ])
        for g, loss in zip(analysis["gammas"], analysis["eval_losses"]):
            report_lines.append(f"| {g} | {loss:.4f} |")

        report_lines.extend([
            "",
            f"**Optimal Gamma**: {analysis.get('optimal_gamma', 'N/A')}",
            f"**Best Loss**: {analysis.get('optimal_loss', 'N/A'):.4f}",
        ])

        if HAS_MATPLOTLIB:
            plot_gamma_sweep(analysis, f"{output_dir}/gamma_plot.png")

    elif experiment_type == "lr_ratio":
        analysis = analyze_lr_ratio_sweep(results)
        report_lines.extend([
            "### Learning Rate Ratio Sweep Results",
            "",
            "| LR Ratio | Eval Loss |",
            "|----------|-----------|",
        ])
        for r, loss in zip(analysis["ratios"], analysis["eval_losses"]):
            report_lines.append(f"| {r} | {loss:.4f} |")

        report_lines.extend([
            "",
            f"**Optimal Ratio**: {analysis.get('optimal_ratio', 'N/A')}",
            f"**Best Loss**: {analysis.get('optimal_loss', 'N/A'):.4f}",
        ])

        if HAS_MATPLOTLIB:
            plot_lr_ratio_sweep(analysis, f"{output_dir}/lr_ratio_plot.png")

    # Gradient analysis
    grad_analysis = analyze_gradient_dynamics(results)
    if grad_analysis.get("mean_grad_ratio"):
        report_lines.extend([
            "",
            "### Gradient Dynamics",
            f"- Mean Gradient Ratio (cross/within): {grad_analysis['mean_grad_ratio']:.4f}",
            f"- Std Gradient Ratio: {grad_analysis['std_grad_ratio']:.4f}",
        ])

        if HAS_MATPLOTLIB:
            plot_gradient_heatmap(results, f"{output_dir}/gradient_heatmap.png")

    # Magnitude analysis
    mag_analysis = analyze_magnitude_dynamics(results)
    if mag_analysis.get("mean_magnitude_ratio"):
        report_lines.extend([
            "",
            "### Output Magnitude Dynamics",
            f"- Mean Magnitude Ratio (cross/within): {mag_analysis['mean_magnitude_ratio']:.4f}",
            f"- Final Magnitude Ratio: {mag_analysis['final_magnitude_ratio']:.4f}",
        ])

    report = "\n".join(report_lines)

    # Save report
    report_path = f"{output_dir}/analysis_report.md"
    with open(report_path, "w") as f:
        f.write(report)

    logger.info(f"Saved analysis report to {report_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("experiment_dir", type=str, help="Path to experiment directory")
    parser.add_argument(
        "--type", type=str, default="chunk_sweep",
        choices=["chunk_sweep", "gamma_sweep", "lr_ratio", "grid"],
        help="Type of experiment to analyze"
    )
    parser.add_argument("--output-dir", type=str, help="Output directory for plots/reports")

    args = parser.parse_args()

    output_dir = args.output_dir or args.experiment_dir

    # Load results
    results = load_experiment_results(args.experiment_dir)

    # Generate report
    report = generate_report(results, args.type, output_dir)
    print(report)
