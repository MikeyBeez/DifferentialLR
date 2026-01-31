# Phased Specialization for Hybrid Sequence Models

**Training dynamics, not architecture, determine whether hybrid transformers succeed.**

This repository contains code and experiments demonstrating that a Mamba-Transformer hybrid performs 14% worse than baseline with joint training, but 6% better with Phased Specialization. The 34-point perplexity improvement comes entirely from the training strategy.

## Key Findings

| Configuration | PPL | vs Baseline |
|--------------|-----|-------------|
| Full Softmax (baseline) | 167.3 | - |
| 4S+4M Joint Training | 191.5 | -14% (worse) |
| 4S+4M Phased Specialization | 157.5 | +6% (better) |

**The same architecture swings 34 PPL points based solely on training strategy.**

### Why Joint Training Fails

Under uniform learning rates, softmax attention learns faster due to simpler gradient paths. It monopolizes the gradient signal, leaving Mamba undertrained. This is the **Redundancy Bottleneck**.

### Why Phased Specialization Works

By giving each module a protected learning window with differential LRs:
- **Phase 1 (Softmax Lead):** Softmax LR 3e-4, Mamba LR 1e-5
- **Phase 2 (Mamba Catchup):** Softmax LR 1e-5, Mamba LR 3e-4

Neither module is ever frozen. The minimal LR (1e-5) maintains module compatibility.

## Ablation Results

| Experiment | PPL | What It Proves |
|------------|-----|----------------|
| Frozen Mamba (Softmax only) | 200.9 | Softmax alone can't reach baseline |
| Frozen Softmax (Mamba only) | 247.3 | Mamba alone is even worse |
| Sequential (hard freeze) | 189.1 | Separation alone ≈ joint training |
| **Phased Specialization** | **157.5** | Minimal LR provides compatibility signal |

The 32-point gap between Sequential (189.1) and Phased (157.5) proves the 1e-5 LR is essential.

## Throughput

The hybrid is actually **faster** than pure softmax due to Mamba's linear complexity:

| Sequence Length | Full Softmax | 4S+4M Hybrid | Ratio |
|-----------------|--------------|--------------|-------|
| 512 | 295k tok/s | 317k tok/s | 107% |
| 1024 | 215k tok/s | 257k tok/s | 119% |

## Installation

```bash
pip install torch transformers datasets triton
```

## Quick Start

### Run Multi-Seed Validation
```bash
python experiments/multi_seed_validation.py
```

### Run Frozen Module Ablation
```bash
python experiments/frozen_ablation.py
```

### Run Gated Skip Experiments
```bash
python experiments/gated_skip_test.py      # On well-trained model
python experiments/gated_skip_uniform.py   # On poorly-trained model
```

### Run Throughput Benchmark
```bash
python experiments/benchmark_tps.py
```

## Project Structure

```
DifferentialLR/
├── src/
│   ├── config.py              # Model configuration
│   ├── model.py               # Hybrid transformer model
│   ├── chunked_attention.py   # Softmax attention implementation
│   └── mamba.py               # Mamba SSM implementation
├── experiments/
│   ├── multi_seed_validation.py    # 5-seed validation (key result)
│   ├── frozen_ablation.py          # Module ceiling experiments
│   ├── differential_mamba.py       # Phased training implementation
│   ├── gated_skip_test.py          # Gated skip on well-trained model
│   ├── gated_skip_uniform.py       # Gated skip on poorly-trained model
│   ├── coordination_lr_ablation.py # Coordination phase analysis
│   └── benchmark_tps.py            # Throughput measurement
├── paper_neurips.txt          # Full paper (plain text)
└── paper_final.txt            # Earlier version
```

## Model Architecture

8-layer transformer, ~45M parameters:
- Hidden dim: 512
- Heads: 8, Head dim: 64
- FFN dim: 2048
- Layers 0-3: Softmax attention
- Layers 4-7: Mamba SSM

## Training Protocol

```python
# Phase 1: Softmax Lead (4 epochs)
optimizer.param_groups[1]['lr'] = 3e-4  # softmax
optimizer.param_groups[2]['lr'] = 1e-5  # mamba

# Phase 2: Mamba Catchup (4 epochs)
optimizer.param_groups[1]['lr'] = 1e-5  # softmax
optimizer.param_groups[2]['lr'] = 3e-4  # mamba

# Stop at best validation PPL (typically epoch 7-8)
# Do NOT add coordination phases
```

## Key Insights

1. **Never freeze completely.** Use minimal LR (1e-5) to maintain module compatibility.

2. **No coordination phase.** Extended training causes overfitting. Stop early.

3. **Gated skips don't rescue.** When added to poorly-trained models, gates learn to keep undertrained Mamba (gate → 0.92), not bypass it.

4. **Validate with multiple seeds.** Single-run results are unreliable. High variance indicates gradient interference.

## Citation

If you use this code, please cite:

```
@misc{differentiallr2025,
  title={Phased Specialization: Unlocking Hybrid Sequence Models via Optimization-Aware Training},
  author={},
  year={2025},
  url={https://github.com/MikeyBeez/DifferentialLR}
}
```

## Related Work

- [Mamba](https://github.com/state-spaces/mamba) - Linear-time sequence modeling
- [PCGrad](https://arxiv.org/abs/2001.06782) - Gradient surgery for multi-task learning
- [Switch Transformers](https://arxiv.org/abs/2101.03961) - MoE load balancing

## License

MIT
