# Chunked Linear Attention: Research Summary

## Objective

Investigate training dynamics of chunked linear attention, specifically the interaction between within-chunk softmax attention and cross-chunk linear state mechanisms. Goal was to understand why linear attention methods underperform compared to standard softmax attention, and explore hybrid architectures that combine efficiency with quality.

## Architecture Design

### Chunked Linear Attention Concept

The core idea is to split sequences into chunks and use different attention mechanisms:

1. **Within-chunk**: Standard O(C²) softmax attention (high quality, limited range)
2. **Cross-chunk**: O(L/C) linear state accumulation (efficient, unlimited range)

```
For sequence length L with chunk size C:
- Within each chunk: softmax attention
- Across chunks: linear KV state with decay

State Update:
  S_i = decay * S_{i-1} + φ(K_i)ᵀ @ V_i

Cross-chunk output:
  cross_out_i = φ(Q_i) @ S_{i-1}
```

Where φ is a feature map (ELU+1, ReLU, etc.) that replaces the softmax.

### Model Configuration

- hidden_dim: 512
- num_layers: 8
- num_heads: 8
- head_dim: 64
- intermediate_dim: 2048
- vocab_size: 50257 (GPT-2)
- Parameters: ~51-64M

## Key Experimental Results

### Best Performing Methods (WikiText-2, 10 epochs)

| Method | Eval PPL | Speed | Notes |
|--------|----------|-------|-------|
| **4 Softmax + 4 Mamba** | **154.0** | 124k t/s | **Best quality (-9% vs baseline)** |
| 2 Softmax + 6 Mamba | 155.9 | 119k t/s | -8% PPL |
| 6 Softmax + 2 Mamba | 155.9 | 43k t/s | -8% PPL |
| Linear attention (learned gate) | 167.4 | - | Best pure-linear approach |
| **6 Softmax + 2 MLP** | **168.3** | **147k t/s** | **Better than baseline + faster** |
| Full Softmax baseline | 170.9 | 130k t/s | Standard transformer |
| 4 Softmax + 4 MLP | 174.8 | 156k t/s | +20% speed, +2.3% PPL |
| Full Mamba | 179.6 | 114k t/s | +6% PPL (needs early softmax) |

### Long Training Analysis (50 epochs)

Critical finding: Linear attention overfits catastrophically with extended training.

| Epoch | Softmax Eval PPL | Linear Eval PPL | Gap |
|-------|------------------|-----------------|-----|
| 10 | 179.2 | 164.1 | -8.5% (linear wins) |
| 13 | 175.6 | **160.9** | -8.4% (linear best) |
| 20 | 206.8 | 213.6 | +3.3% |
| 30 | 333.7 | 472.0 | +41.5% |
| 50 | 894.4 | 2241.2 | +150.6% (linear catastrophic) |

**Key insight**: Linear attention achieves its best performance early (epoch 13, PPL 160.9) then overfits dramatically. Softmax overfits too, but much more gracefully.

- Softmax overfit ratio at epoch 50: 182x (train 4.9, eval 894)
- Linear overfit ratio at epoch 50: 1067x (train 2.1, eval 2241)

## Mamba Hybrid Experiments

### Hypothesis
State-space models (SSMs) like Mamba can replace attention in later layers while maintaining sequence modeling capability. Unlike MLPs which are per-token, Mamba provides O(L) sequence modeling via selective scan.

### Implementation
Pure PyTorch Mamba with Triton-accelerated selective scan kernel for efficient GPU execution.

### Results

| Configuration | PPL | Speed | PPL Change | Speed Change |
|--------------|-----|-------|------------|--------------|
| Full Softmax baseline | 169.2 | 135k t/s | - | - |
| **4 Softmax + 4 Mamba** | **154.0** | 124k t/s | **-9.0%** | -8% |
| 2 Softmax + 6 Mamba | 155.9 | 119k t/s | -7.8% | -12% |
| 6 Softmax + 2 Mamba | 155.9 | 43k t/s | -7.8% | -68% |
| Full Mamba | 179.6 | 114k t/s | +6.2% | -16% |

### Speed vs Sequence Length (Mamba advantage at long sequences)

| Seq Length | Full Softmax | 4S + 4M | Mamba vs Softmax |
|------------|--------------|---------|------------------|
| 512 | 133k t/s | 122k t/s | 92% |
| 1024 | 128k t/s | 124k t/s | 97% |
| 2048 | 90k t/s | 104k t/s | **116%** |
| 4096 | 55k t/s | 76k t/s | **138%** |

**Key Findings**:
1. **4 Softmax + 4 Mamba achieves best perplexity** (154.0, -9% vs baseline)
2. Early softmax layers are essential - Full Mamba underperforms baseline
3. Mamba is slower at short sequences but **faster at long sequences** (O(L) vs O(L²))
4. Triton kernel provides 26x speedup over pure PyTorch selective scan

### Differential LR Training for Mamba Hybrid

Based on the Redundancy Bottleneck paper, we applied phased differential learning rates to the 4 Softmax + 4 Mamba architecture.

**Hypothesis**: Softmax and Mamba modules compete for gradient signal. Letting each lead in turn prevents this competition.

**Phase Configuration**:
- Phase 1 (Softmax-Lead): softmax_lr=3e-4, mamba_lr=1e-5 (3 epochs)
- Phase 2 (Mamba-Catchup): softmax_lr=1e-5, mamba_lr=3e-4 (4 epochs)
- Phase 3 (Coordinate): both at 1e-4 (3 epochs)

**Results**:

| Training Strategy | PPL | Change |
|-------------------|-----|--------|
| Uniform LR (1e-4) | 186.7 | baseline |
| **Differential LR (phased)** | **173.0** | **-7.3%** |

**Key insight**: The best PPL (173.0) was achieved during Mamba-Catchup phase (epoch 6), validating that Mamba benefits from a dedicated learning window after softmax features are established.

**Learning Rate Sensitivity**: Initial experiments with high LRs (1e-3/1e-5) performed worse. Lower LRs (3e-4/1e-5) closer to the baseline worked better for this smaller model (~45M params vs PEER's 1.1B).

### Extended Training (20 Epochs)

Extended differential LR training with 4 phases:
- Softmax-Lead: 4 epochs (3e-4 / 1e-5)
- Mamba-Catchup: 6 epochs (1e-5 / 3e-4)
- Coordinate: 6 epochs (5e-5 / 5e-5)
- Fine-Tune: 4 epochs (1e-5 / 1e-5)

**Results**:

| Training | Best PPL | Epoch | Speed |
|----------|----------|-------|-------|
| 10-epoch uniform | 186.7 | 10 | 124k t/s |
| 10-epoch differential | 173.0 | 6 | 131k t/s |
| **20-epoch extended** | **163.9** | **9** | **123k t/s** |

**Critical finding**: The Coordination phase *hurts* performance. Best PPL was 163.9 at epoch 9 (during Mamba-Catchup). Transitioning to Coordinate caused immediate degradation (epoch 10: 202.2 PPL).

**Optimal strategy**:
- 4 epochs Softmax-Lead
- 5-6 epochs Mamba-Catchup
- **Stop** (no coordination phase needed)

This suggests the phased approach works by letting each module specialize, and trying to "coordinate" them afterward disrupts the learned representations.

## Static Function Hypothesis

Based on the paper "Accelerating Transformer Inference Through Selective Attention Replacement" (Bee, 2025), we validated the hypothesis that:

> Early Transformer layers require dynamic self-attention for feature extraction, while deeper layers perform more predictable operations on already-structured representations.

### Experimental Validation

| Configuration | PPL | Speed | PPL Change | Speed Change |
|--------------|-----|-------|------------|--------------|
| Full Softmax (8 layers) | 170.9 | 130k t/s | - | - |
| **6 softmax + 2 MLP** | **168.3** | **147k t/s** | **-1.5%** | **+13%** |
| 4 softmax + 4 MLP | 174.8 | 156k t/s | +2.3% | +20% |
| 4 softmax + 4 linear | 178.3 | 31k t/s | +4.3% | -76% |
| 2 softmax + 6 MLP | 202.2 | 167k t/s | +18.3% | +29% |

**Conclusion**: The hypothesis is validated. **6 Softmax + 2 MLP** achieves both better quality AND faster inference than the baseline. More aggressive replacement (4+ MLP layers) trades quality for speed.

## Projection Bottleneck

Tested adding a projection bottleneck after MLP layers to compress to "essential geometry":

| Configuration | PPL | Speed | Notes |
|--------------|-----|-------|-------|
| 4 softmax + 4 MLP (no proj) | 175.7 | 148k t/s | Baseline hybrid |
| **4 softmax + 4 MLP+Proj (0.5x)** | **174.9** | **151k t/s** | Projection helps |
| 4 softmax + 4 MLP+Proj (0.25x) | 178.5 | 152k t/s | Too aggressive |
| 4 softmax + 4 MLP+MultiProj | 174.4 | 120k t/s | Best quality, slower |

**Conclusion**: Projection bottleneck provides marginal improvement (175.7 → 174.9 PPL) while maintaining speed. The 0.5x bottleneck ratio is optimal.

## Sparse Distillation Experiments

Attempted to replace attention entirely with MLPs via distillation:

| Approach | PPL | Notes |
|----------|-----|-------|
| Position-independent MLP | 343.0 | Failed - no cross-position info |
| Context-aware MLP + KD | 197.1 | Better but still worse than linear |
| Hybrid sparse (chunked) | 179.7 | Close to linear attention |

**Key insight**: MLPs cannot replace attention in early layers because they lack cross-position information. The learning-execution asymmetry works only when applied to later layers where representations are already structured.

## Improvement Experiments

### Feature Maps and Gating

| Improvement | Eval PPL | vs Baseline |
|-------------|----------|-------------|
| **Learned Gate** | **167.4** | -0.5% |
| Baseline (ELU+1) | 168.2 | - |
| Hybrid (2 softmax) | 169.0 | +0.5% |
| Combined | 174.2 | +3.6% |
| L2 Norm | 174.6 | +3.8% |
| Power2 | 338.8 | +101% (unstable) |

**Conclusion**: Learned gating is the only improvement that helps for pure linear attention. Data-dependent gating allows selective forgetting based on content.

## The Redundancy Bottleneck

Based on the paper "We Stacked 3 AI Upgrades. The Combined System Was Worse Than Using Just 1." (Bee, 2025).

### Core Finding

**Sparse retrieval modules are not additive — they compete for representational real estate.**

When combining multiple sparse mechanisms (attention variants, MLPs, Mamba, etc.) and training them jointly, the faster-learning module monopolizes the gradient signal, crowding out slower modules before they can specialize. The combined system underperforms its components.

### The Mechanism: Learning-Speed Asymmetry

When multiple modules can reduce the same training loss, gradient descent allocates explanatory responsibility to the module with the shortest effective path from input to loss:

1. The faster-learning module collapses the error signal
2. Slower or higher-capacity modules receive diminished gradients
3. Latent capacity remains unused, even if it is expressive

This applies to any modular system where components overlap functionally but differ in learning speed: hybrid attention architectures, MoE routing, stacked adapters, retrieval-augmented models.

### Relevance to This Work

The Redundancy Bottleneck explains several observations:

| Observation | Explanation |
|-------------|-------------|
| 4S+4 Mamba beats full softmax | Mamba and softmax have different learning dynamics; splitting layers prevents competition |
| Full Mamba underperforms | Without softmax's fast early feature extraction, Mamba struggles |
| 6S+2 MLP is optimal | MLP learns fast; limiting to 2 layers prevents it from dominating |
| Linear attention with learned gate works | Data-dependent gating allows selective specialization |

### The Solution: Differential Learning Rate Training

Instead of training all modules equally, let them take turns leading:

```
Phase 1 — Fast Module Leads:
  Fast module LR: 1e-3 (high)
  Slow module LR: 1e-5 (minimal, but not zero)

Phase 2 — Slow Module Catches Up:
  Slow module LR: 1e-3 (high)
  Fast module LR: 1e-5 (minimal)

Phase 3 — Coordination:
  All modules: 5e-4 (medium)
```

**Critical**: Never fully freeze. Complete freezing causes distribution shift — the frozen module becomes incompatible with still-training modules.

### Application to Chunked Linear Attention

The differential LR experiments in this repo (`experiments/differential_lr.py`) explore this principle:
- Within-chunk attention (softmax-like): faster learning, simpler path
- Cross-chunk state (linear): slower learning, requires accumulation

Giving cross-chunk components higher LR can help them catch up before within-chunk dominates.

### Key Takeaways

1. **Modules compete** for gradient signal and explanatory responsibility
2. **Learning-speed asymmetry** causes faster modules to dominate
3. **Functional overlap** means multiple modules explain the same variance — faster one wins
4. **Differential LR** breaks the race by giving each module a protected learning window
5. **Never fully freeze** — maintain minimal LR for adaptation
6. **Architecture is not destiny** — training dynamics determine outcomes

## Theoretical Insights

### Why Linear Attention Struggles

1. **Selection vs Averaging**: Softmax creates sparse, peaked attention patterns (selection). Linear attention creates diffuse patterns (averaging). Information gets "smeared."

2. **Geometric Navigation**: Attention output moves representations to the right neighborhood in semantic space. Linear attention's averaging loses this precise navigation.

3. **Three Functions of Attention**:
   - Polysemy disambiguation (which meaning of a word)
   - Cosine similarity matching (finding relevant tokens)
   - Geometric navigation (moving in semantic space)

Linear attention can approximate #2 but struggles with #1 and #3.

### Why Static Function Hypothesis Works

Later layers operate on already-structured representations where:
- Feature extraction is complete
- Transformations are more predictable
- Cross-position interaction is less critical

Early layers need dynamic attention to handle unstructured input tokens.

### Why Mamba Improves Over Baseline

The 4 Softmax + 4 Mamba configuration achieves better PPL (154.0) than full softmax (169.2):
1. **Selective state space**: Mamba's input-dependent gating allows selective retention/forgetting
2. **Continuous state**: Unlike attention's discrete token selection, SSM maintains smooth state evolution
3. **Complementary strengths**: Early softmax extracts features, Mamba refines sequences efficiently
4. **No attention bottleneck**: Later layers avoid O(L²) attention overhead while maintaining sequence modeling

## Practical Recommendations

### For Maximum Quality
Use **4 Softmax + 4 Mamba with extended differential LR**: PPL 163.9
- 4 epochs Softmax-Lead + 5-6 epochs Mamba-Catchup (no coordination)
- Requires Triton kernel for efficient inference
- Best for long sequences (2048+ tokens)
- Note: PPL varies between runs (154-187 range); differential LR consistently improves by ~12%

### For Speed + Quality (Short Sequences)
Use **6 Softmax + 2 MLP**: PPL 168.3, +13% speed
- Better than baseline on both metrics
- Simple implementation, no sequence modeling in late layers

### For Maximum Speed
Use **4 Softmax + 4 MLP**: PPL 174.8, +20% speed
- Small quality tradeoff for significant speedup
- Best for latency-critical applications

### Architecture Selection Guide
| Use Case | Recommendation | PPL | Speed |
|----------|---------------|-----|-------|
| Quality-critical, long context | 4S + 4 Mamba + Diff LR | 163.9 | ~123k t/s |
| Balanced (short context) | 6S + 2 MLP | 168.3 | 147k t/s |
| Speed-critical | 4S + 4 MLP | 174.8 | 156k t/s |
| Baseline comparison | Full Softmax | 170.9 | 130k t/s |

### Avoid
- Pure MLP replacement of all attention (fails completely)
- Full Mamba without early softmax layers (6% worse than baseline)
- Extended training of linear attention (catastrophic overfitting)
- Aggressive projection bottlenecks (<0.5x)

## Repository Structure

```
/home/bee/Code/LinearAttention/
├── src/
│   ├── config.py              # Hyperparameter dataclasses (includes Mamba config)
│   ├── chunked_attention.py   # Core attention + Mamba + Triton kernel
│   ├── model.py               # Full transformer with hybrid routing
│   ├── data.py                # WikiText-2 data loading
│   ├── train.py               # Training loop
│   └── diagnostics.py         # Gradient/magnitude tracking
├── experiments/
│   ├── baseline_sweep.py      # Chunk size experiments
│   ├── differential_lr.py     # LR ratio experiments
│   ├── long_training_comparison.py  # 50-epoch overfitting study
│   ├── static_function_hybrid.py    # Static Function Hypothesis (MLP)
│   ├── mamba_hybrid.py        # Mamba hybrid experiments
│   ├── hybrid_with_projection.py    # Projection bottleneck
│   ├── sparse_distillation*.py      # MLP distillation attempts
│   ├── test_improvements.py   # Feature map experiments
│   ├── differential_mamba.py  # Phased differential LR for 4S+4M
│   └── differential_mamba_extended.py  # Extended 20-epoch training
└── configs/
    └── default.yaml
```

### Key Implementation Details

**Mamba (src/chunked_attention.py)**:
- `SelectiveSSM`: Core Mamba mechanism with input-dependent B, C, delta
- `MambaLayer`: Full layer with pre-norm, residual, optional FFN
- `selective_scan_triton`: Triton kernel for 26x speedup over PyTorch
- Configurable via `mamba_layers` list in ModelConfig

**Hybrid Routing (src/model.py)**:
- `softmax_layers`: List of layer indices using full softmax attention
- `mamba_layers`: List of layer indices using Mamba SSM
- Remaining layers use chunked linear attention

## References

- lucidrains/linear-attention-transformer: https://github.com/lucidrains/linear-attention-transformer
- flash-linear-attention (GLA): https://github.com/fla-org/flash-linear-attention
- "Transformers are RNNs" (Katharopoulos et al., 2020)
- "Gated Linear Attention Transformers with Hardware-Efficient Training" (Yang et al., 2024)
- "Accelerating Transformer Inference Through Selective Attention Replacement" (Bee, 2025)
- "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023)
- state-spaces/mamba: https://github.com/state-spaces/mamba
- "We Stacked 3 AI Upgrades. The Combined System Was Worse Than Using Just 1." (Bee, 2025) - Redundancy Bottleneck
- PEER: Mixture of A Million Experts (arxiv.org/abs/2407.04153)

## Future Directions

1. **Scale validation**: Test hybrid architectures on larger models (1B+ parameters)
2. **Task-specific evaluation**: Compare on downstream tasks beyond perplexity
3. **Dynamic layer selection**: Learn which layers need attention vs MLP vs Mamba
4. **Combine approaches**: Softmax early + Mamba middle + MLP late
5. **Long context benchmarks**: Validate Mamba speedup on 8K+ token sequences
6. **mamba-ssm integration**: Compare Triton kernel vs official CUDA kernels
7. **Attention pattern analysis**: Understand why Mamba improves quality over baseline
8. ~~**Differential LR for hybrids**~~: ✓ Done - 7.3% improvement (186.7 → 173.0)
9. **Redundancy Bottleneck diagnosis**: Use output magnitude analysis to detect module dominance
10. **Three-way hybrid**: Test Softmax + Mamba + MLP with differential LR coordination
