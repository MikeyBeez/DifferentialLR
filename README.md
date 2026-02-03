# Linear Attention Research

This repository contains three research directions on efficient sequence modeling:

1. **Attention Ablation** - What's actually necessary in attention? O(N) beats O(N²) (NEW)
2. **Golden Ratio Engram** - Learnable EMA filter that complements attention
3. **Phased Specialization** - Training strategy for hybrid Mamba-Transformer models

---

## Attention May Not Be What You Need (NEW)

**Paper: [papers/attention_may_not_be_what_you_need.txt](papers/attention_may_not_be_what_you_need.txt)**

A series of ablations asking what's actually necessary in transformer attention. Each step removed something thought to be essential. Nothing broke. It got better.

### Key Result

**O(N) learned causal convolution beats O(N²) softmax attention** on both perplexity AND throughput:

| Model | PPL | Change | TPS | Change |
|-------|-----|--------|-----|--------|
| True O(N) K=64 | 8.08 | **-3.2%** | 378,066 | **+18.9%** |
| Standard QKV | 8.34 | baseline | 317,968 | baseline |

### Experimental Progression

1. **Dot product replacement** ([mlp_attention_test.py](experiments/mlp_attention_test.py)): Any comparison function works. MLP, L2, bilinear all within 1.2% of baseline.

2. **V projection elimination** ([no_value_attention_test.py](experiments/no_value_attention_test.py)): V=K works with only 1.5% degradation. The value projection is mostly redundant.

3. **Asymmetry hypothesis** ([asymmetry_hypothesis_test.py](experiments/asymmetry_hypothesis_test.py)): Q=K forces symmetric scores, hurting performance. But relative position bias provides asymmetry AND beats standard attention (-1.7%).

4. **Content-independent scores** ([positional_only_attention_test.py](experiments/positional_only_attention_test.py)): Learned positional patterns without content-dependent Q·K scores. Every variant beat standard attention.

5. **True O(N) implementation** ([true_linear_attention_test.py](experiments/true_linear_attention_test.py)): Causal convolution with learned kernel. 3.2% better perplexity, 19% higher throughput.

### Conclusions

- The dot product isn't special - any differentiable comparison works
- Content-dependent routing is unnecessary - positional patterns suffice
- The MLP does the real work - attention is just routing infrastructure
- O(N) beats O(N²) at this scale

See the paper for full details, caveats, and connections to Mamba/Hyena/Synthesizer.

---

## Golden Ratio Engram (Corrected)

**Paper: [papers/golden_engram_corrected.md](papers/golden_engram_corrected.md)**

**⚠️ CORRECTION: Original "memory system" claims were invalid. The corrected version is more useful.**

We originally claimed a Golden Ratio-based memory system. That was wrong—we had data leakage. After correction, we found something simpler but real: a **learnable EMA filter** that complements attention.

### What We Actually Built

Not a memory system—a low-pass temporal filter (exponentially weighted moving average). This places it in the lineage of S4, Hyena, and classical signal processing.

| Model | Val PPL | Change |
|-------|---------|--------|
| Pure Attention (8 layers) | 8.19 | baseline |
| Interleaved 4A + 4E | 7.86 | **-4.0%** |

| Model | TPS | Change |
|-------|-----|--------|
| Pure Attention | 230,992 | baseline |
| Fast Engram + compile | 341,396 | **+48%** |

### Key Insights

1. **Engrams provide momentum, not retrieval.** They're rank-1 with respect to time—they collapse history into one direction. No content-addressable lookup.

2. **Pure engrams fail catastrophically** (PPL 3000+). This is expected and increases confidence in the corrected results.

3. **Interleaved attention + engram works.** Attention handles retrieval; engram handles temporal smoothing. They solve different problems.

4. **The golden ratio is not special.** φ = 0.618 gives a half-life of ~1.44 tokens. It's a hyperparameter, not magic. The name is historical.

5. **The cumsum trick makes it fast.** Same vectorization used in S4/Hyena. No sequential loop needed.

### Code

```python
class FastEngramLayer(nn.Module):
    def __init__(self, dim):
        self.proj = nn.Linear(dim, dim)
        self.gate_proj = nn.Linear(dim, dim)
        self.phi = 0.618

    def forward(self, x):
        B, L, D = x.shape
        gate = torch.sigmoid(self.gate_proj(x))
        x_gated = gate * self.proj(x)

        powers = self.phi ** torch.arange(L, device=x.device)
        x_weighted = x_gated * (1.0 / powers).view(1, -1, 1)
        cum_sum = torch.cumsum(x_weighted, dim=1)
        return cum_sum * powers.view(1, -1, 1)
```

See [experiments/fast_engram_test.py](experiments/fast_engram_test.py) for full implementation.

---

## Phased Specialization for Hybrid Sequence Models

**In our experiments, training strategy dominated architectural effects for hybrid transformers.**

This repository contains code and experiments demonstrating that a Mamba-Transformer hybrid performs 14% worse than baseline with joint training, but 6% better with Phased Specialization. Holding architecture fixed, changing only the training schedule produced a 34-point perplexity swing.

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

| Experiment | PPL | Interpretation |
|------------|-----|----------------|
| Frozen Mamba (4 softmax layers only) | 200.9 | 4 layers underperform 8-layer baseline |
| Frozen Softmax (4 mamba layers only) | 247.3 | Mamba alone learns slower |
| Sequential (hard freeze) | 189.1 | Separation alone ≈ joint training (191.5) |
| **Phased Specialization** | **157.5** | Both modules contribute when properly trained |

The 32-point gap between Sequential (189.1) and Phased (157.5) suggests that maintaining a small learning rate (1e-5) improves module compatibility, though the exact mechanism requires further study.

## Gated Skip Connections Don't Rescue

On jointly-trained model (189.7 PPL):
- Gate-only training: Gate → 0.92, PPL stays 189.6
- Full training: Gate → 0.50, PPL improves slightly to 188.0

The gate does NOT learn to bypass undertrained Mamba. It keeps it because Mamba output is "better than nothing." The 30-point gap to phased training (157.5) confirms architectural modifications cannot substitute for proper optimization.

On phased-trained model (158.4 PPL): Gate → 1.0, confirming properly trained Mamba is fully useful.

## Throughput

The hybrid is **faster** than pure softmax at tested sequence lengths due to Mamba's linear complexity replacing quadratic attention in layers 4-7:

| Sequence Length | Full Softmax | 4S+4M Hybrid | Ratio |
|-----------------|--------------|--------------|-------|
| 512 | 295k tok/s | 317k tok/s | 107% |
| 1024 | 215k tok/s | 257k tok/s | 119% |

**Methodology:**
- GPU: NVIDIA GeForce RTX 5070 Ti (16GB)
- Batch size: 8
- Precision: FP16 (torch.amp.autocast)
- Warmup: 10 iterations, benchmark: 100 iterations
- No torch.compile (measuring raw PyTorch + Triton kernel)

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
│   │
│   │ # Attention Ablation (NEW)
│   ├── mlp_attention_test.py            # Dot product replacement
│   ├── no_value_attention_test.py       # V projection elimination
│   ├── asymmetry_hypothesis_test.py     # Q=K and positional bias tests
│   ├── positional_only_attention_test.py # Content-independent scoring
│   ├── true_linear_attention_test.py    # True O(N) causal convolution
│   │
│   │ # Golden Ratio Engram (Corrected)
│   ├── fast_engram_test.py         # Fast vectorized engram implementation
│   ├── interleaved_4a4e_test.py    # Interleaved attention + engram
│   ├── pure_attention_tps.py       # Attention baseline benchmarks
│   │
│   │ # Golden Ratio Crystallization (Superseded)
│   ├── multi_dataset_test.py       # WikiText-2/103 validation
│   ├── infinite_context_torture.py # Memory scaling up to 65k tokens
│   ├── kv_cache_death.py           # Inference state comparison
│   ├── recall_test.py              # Content-dependent retrieval test
│   ├── spiral_no_pos.py            # Positional encoding ablation
│   ├── damped_multihead_test.py    # Multi-resolution decay heads
│   ├── phi_powers_test.py          # Decay rate experiments
│   ├── damped_spiral_test.py       # Basic spiral tests
│   ├── recursive_golden_spiral.py  # Recursive formulation
│   ├── spiral_deep_mlp.py          # Deeper MLP experiments
│   │
│   │ # Phased Specialization (Mamba-Transformer)
│   ├── multi_seed_validation.py    # 5-seed validation (key result)
│   ├── frozen_ablation.py          # Module ceiling experiments
│   ├── differential_mamba.py       # Phased training implementation
│   ├── gated_skip_test.py          # Gated skip on well-trained model
│   ├── gated_skip_uniform.py       # Gated skip on poorly-trained model
│   ├── coordination_lr_ablation.py # Coordination phase analysis
│   └── benchmark_tps.py            # Throughput measurement
│
├── papers/
│   ├── attention_may_not_be_what_you_need.txt  # Attention ablation paper (NEW)
│   ├── golden_engram_corrected.md  # Corrected Golden Ratio Engram paper
│   └── golden_engram_corrected.txt # Plain text version
├── paper_end_of_attention.txt # Golden Crystallization paper (superseded)
├── paper_neurips.txt          # Phased Specialization paper
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

- [PEER](https://arxiv.org/abs/2407.04153) - DeepMind's Parameter Efficient Expert Retrieval (inspired our phased training)
- [mHC](https://arxiv.org/abs/2512.24880) - DeepSeek's Manifold-Constrained Hyper-Connections (tested, didn't rescue joint training)
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
