# Phased Specialization for Hybrid Sequence Models

**Training dynamics, not architecture, determine whether hybrid transformers succeed.**

This repository contains code and experiments demonstrating that a Mamba-Transformer hybrid performs 14% worse than baseline with joint training, but 6% better with Phased Specialization. The 34-point perplexity improvement comes entirely from the training strategy.

Paper: [paper_neurips.txt](paper_neurips.txt)

## Key Findings

| Configuration | PPL | Std | vs Baseline |
|--------------|-----|-----|-------------|
| Full Softmax (baseline) | 167.3 | 1.1 | - |
| 4S+4M Joint Training | 191.5 | 4.3 | -14% (worse) |
| 4S+4M Phased Specialization | 157.5 | 1.3 | +6% (better) |

**The same architecture swings 34 PPL points based solely on training strategy.**

Note the variance: joint training has 3x higher std (4.3 vs 1.3), indicating unstable optimization from gradient interference.

### Why Joint Training Fails

Joint optimization of heterogeneous modules creates a zero-sum game. Attention mechanisms have shorter effective gradient paths and more stable optimization. They resolve local linguistic features rapidly, monopolizing gradient signal before SSM modules can specialize.

We term this **learning-speed asymmetry**. It explains why hybrid architectures often underperform their homogeneous counterparts despite theoretical advantages.

### Why Phased Specialization Works

Phased Specialization can be understood as **Approximate Block Coordinate Descent**. By giving each module a protected learning window:

- **Phase 1 (Attention Lead, 4 epochs):** Attention LR 3e-4, Mamba LR 1e-5
- **Phase 2 (SSM Specialization, 4 epochs):** Attention LR 1e-5, Mamba LR 3e-4

Neither module is ever frozen. The minimal learning rate (1e-5) maintains compatibility and allows minor adjustments while preventing gradient monopolization.

## Training Dynamics

Per-epoch validation PPL during phased training:

```
Attention Lead:      499.6 → 347.0 → 278.5 → 234.6
SSM Specialization:  187.9 → 170.8 → 162.3 → 158.4 → 187.2 (overfitting)
```

Stop at epoch 8 (158.4). Extended training causes rapid overfitting (epoch 9: 187.2).

## Ablation Results

| Experiment | PPL | What It Proves |
|------------|-----|----------------|
| Frozen Mamba (Softmax only) | 200.9 | Softmax alone can't reach baseline (167.3) |
| Frozen Softmax (Mamba only) | 247.3 | Mamba alone is even worse |
| Sequential (hard freeze) | 189.1 | Separation alone ≈ joint training (191.5) |
| **Phased Specialization** | **157.5** | Minimal LR provides compatibility signal |

The 32-point gap between Sequential (189.1) and Phased (157.5) proves the 1e-5 LR is not just "slow training" but provides essential module compatibility signal.

## Gated Skip Connections Don't Rescue

On jointly-trained model (189.7 PPL):
- Gate-only training: Gate → 0.92, PPL stays 189.6
- Full training: Gate → 0.50, PPL improves slightly to 188.0

The gate does NOT learn to bypass undertrained Mamba. It keeps it because Mamba output is "better than nothing." The 30-point gap to phased training (157.5) confirms architectural modifications cannot substitute for proper optimization.

On phased-trained model (158.4 PPL): Gate → 1.0, confirming properly trained Mamba is fully useful.

## Throughput

The hybrid is actually **faster** than pure softmax due to Mamba's linear complexity:

| Sequence Length | Full Softmax | 4S+4M Hybrid | Ratio |
|-----------------|--------------|--------------|-------|
| 512 | 295k tok/s | 317k tok/s | 107% |
| 1024 | 215k tok/s | 257k tok/s | 119% |

GPU: NVIDIA GeForce RTX 5070 Ti

## Installation

```bash
pip install torch transformers datasets triton
```

## Quick Start

### Run Multi-Seed Validation (Key Result)
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

### Run Coordination Ablation
```bash
python experiments/coordination_lr_ablation.py
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
│   └── mamba.py               # Mamba SSM with Triton kernel
├── experiments/
│   ├── multi_seed_validation.py    # 5-seed validation (key result)
│   ├── frozen_ablation.py          # Module ceiling experiments
│   ├── differential_mamba.py       # Phased training implementation
│   ├── gated_skip_test.py          # Gated skip on well-trained model
│   ├── gated_skip_uniform.py       # Gated skip on poorly-trained model
│   ├── coordination_lr_ablation.py # Coordination phase analysis
│   └── benchmark_tps.py            # Throughput measurement
├── paper_neurips.txt          # Full paper (NeurIPS style)
└── paper_final.txt            # Earlier version
```

## Model Architecture

8-layer transformer, ~45M parameters:
- Hidden dim: 512
- Attention heads: 8, Head dim: 64
- FFN dim: 2048
- Vocab size: 50,257 (GPT-2 tokenizer)
- Layers 0-3: Softmax attention
- Layers 4-7: Mamba SSM (d_state=16, d_conv=4, expand=2)

## Training Protocol

```python
# Separate parameters by module type
embed_params, softmax_params, mamba_params = get_layer_params(model)

optimizer = torch.optim.AdamW([
    {'params': embed_params, 'lr': 1e-4, 'weight_decay': 0.0},
    {'params': softmax_params, 'lr': 3e-4, 'weight_decay': 0.1},
    {'params': mamba_params, 'lr': 1e-5, 'weight_decay': 0.1},
])

# Phase 1: Attention Lead (4 epochs)
optimizer.param_groups[1]['lr'] = 3e-4  # attention
optimizer.param_groups[2]['lr'] = 1e-5  # mamba

# Phase 2: SSM Specialization (4 epochs)
optimizer.param_groups[1]['lr'] = 1e-5  # attention
optimizer.param_groups[2]['lr'] = 3e-4  # mamba

# Stop at best validation PPL (typically epoch 7-8)
# Do NOT add coordination phases - they cause degradation
```

## Practical Recommendations

1. **Always validate with multiple seeds.** High variance indicates gradient interference.

2. **Use phased training.** Train the slower-learning module last on stable representations from faster modules.

3. **Never fully freeze.** Use minimal learning rates (1e-5) to maintain module compatibility.

4. **Stop early.** Monitor validation loss and stop when improvement stalls; coordination phases do not help.

5. **Do not rely on architectural fixes.** Gated skips and similar modifications cannot rescue poor training.

## Related Work

- [Mamba](https://github.com/state-spaces/mamba) - Linear-time sequence modeling with selective state spaces
- [PCGrad](https://arxiv.org/abs/2001.06782) - Gradient surgery for multi-task learning
- [Switch Transformers](https://arxiv.org/abs/2101.03961) - MoE load balancing
- [Curriculum Learning](https://ronan.collobert.com/pub/matos/2009_curriculum_icml.pdf) - Bengio et al. 2009

## Citation

```bibtex
@misc{phasedspecialization2025,
  title={Phased Specialization: Unlocking Hybrid Sequence Models via Optimization-Aware Training},
  author={},
  year={2025},
  url={https://github.com/MikeyBeez/DifferentialLR}
}
```

## License

MIT
