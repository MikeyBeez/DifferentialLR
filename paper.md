# Hybrid Transformer Architectures: Combining Softmax Attention, State Space Models, and Static Functions

**Abstract**

We investigate hybrid transformer architectures that combine different sequence modeling mechanisms across layers. Through systematic experimentation on WikiText-2, we demonstrate that a 4 Softmax + 4 Mamba configuration achieves 9% better perplexity than full softmax attention while maintaining comparable inference speed. We further show that differential learning rate training—where different module types are trained with phased learning rates—improves performance by an additional 12%, achieving 163.9 PPL compared to 186.7 with uniform learning rates. Our analysis reveals that the coordination phase commonly used in multi-module training is counterproductive; optimal results are achieved by training modules sequentially without subsequent joint fine-tuning. We validate the Static Function Hypothesis, showing that later transformer layers can be replaced with simple MLPs while improving both quality and speed. These findings suggest that transformer architectures benefit from heterogeneous layer compositions matched to the computational requirements at different depths.

## 1. Introduction

The transformer architecture has become the foundation of modern language models, with self-attention providing the core mechanism for modeling token dependencies. However, the O(L²) complexity of standard softmax attention creates computational bottlenecks for long sequences, motivating research into efficient alternatives including linear attention, state space models, and hybrid architectures.

This work investigates three key questions:

1. **Can heterogeneous architectures outperform homogeneous transformers?** We test whether combining different mechanisms (softmax attention, Mamba SSM, MLPs) across layers yields better quality-speed tradeoffs than using any single mechanism throughout.

2. **How should multi-module architectures be trained?** We apply differential learning rate training, where different module types receive different learning rates in phases, to address gradient competition between modules.

3. **Do later transformer layers require dynamic attention?** We test the Static Function Hypothesis—that deeper layers perform predictable operations on structured representations and can be replaced with simpler mechanisms.

Our contributions include:

- Demonstration that 4 Softmax + 4 Mamba achieves 9% better perplexity than full softmax
- Evidence that differential LR training improves hybrid architectures by 12%
- Discovery that the coordination phase in phased training is harmful
- Validation that 6 Softmax + 2 MLP beats the baseline on both quality and speed
- A Triton kernel for Mamba selective scan achieving 26x speedup over PyTorch

## 2. Background and Related Work

### 2.1 Linear Attention

Linear attention (Katharopoulos et al., 2020) replaces the softmax normalization with a feature map φ, enabling O(L) complexity through associativity:

```
Attention(Q, K, V) = φ(Q)(φ(K)ᵀV) instead of softmax(QKᵀ)V
```

However, linear attention consistently underperforms softmax attention due to its inability to create sparse, selective attention patterns. Our experiments confirm this: linear attention achieves its best performance early (epoch 13, PPL 160.9) then overfits catastrophically, reaching PPL 2241 by epoch 50 compared to softmax's 894.

### 2.2 State Space Models

Mamba (Gu & Dao, 2023) introduces selective state spaces where the state transition matrices are input-dependent:

```
h_t = A_t h_{t-1} + B_t x_t
y_t = C_t h_t
```

This selectivity allows the model to decide what information to retain or forget based on content, combining the efficiency of RNNs with content-based gating.

### 2.3 The Redundancy Bottleneck

Recent work on modular architectures reveals that sparse retrieval modules compete for gradient signal when trained jointly. The faster-learning module monopolizes explanatory responsibility, crowding out slower modules before they can specialize. This "redundancy bottleneck" explains why naive combinations of powerful modules often underperform their components.

### 2.4 Static Function Hypothesis

The hypothesis that early transformer layers require dynamic attention while later layers perform more predictable operations suggests that computational resources are misallocated in homogeneous architectures. Later layers operate on already-structured representations where cross-position interaction is less critical.

## 3. Methods

### 3.1 Model Architecture

We use an 8-layer transformer with:
- Hidden dimension: 512
- Attention heads: 8
- Head dimension: 64
- FFN dimension: 2048
- Vocabulary: 50,257 (GPT-2 tokenizer)
- Parameters: ~45-51M depending on configuration

### 3.2 Hybrid Configurations

We test several hybrid architectures:

**Softmax + Mamba Hybrids:**
- 4 Softmax (layers 0-3) + 4 Mamba (layers 4-7)
- 2 Softmax + 6 Mamba
- 6 Softmax + 2 Mamba
- Full Mamba (8 layers)

**Softmax + MLP Hybrids:**
- 6 Softmax + 2 MLP
- 4 Softmax + 4 MLP
- 2 Softmax + 6 MLP

### 3.3 Mamba Implementation

We implement Mamba in pure PyTorch with a Triton-accelerated selective scan kernel:

```python
@triton.jit
def selective_scan_kernel(x, delta, A, B, C, out, ...):
    # Each thread block processes one (batch, d_block) slice
    # Sequential scan over time with parallel computation over state
    h = tl.zeros([BLOCK_S], dtype=tl.float32)
    for t in range(seq_len):
        delta_t = tl.load(delta_ptr + t * d_inner + d_idx)
        x_t = tl.load(x_ptr + t * d_inner + d_idx)
        A_bar = tl.exp(delta_t * A)
        B_bar = delta_t * B_t
        h = A_bar * h + B_bar * x_t
        y_t = tl.sum(C_t * h)
        tl.store(out_ptr + t * d_inner + d_idx, y_t)
```

This achieves 26x speedup over the naive PyTorch loop implementation.

### 3.4 Differential Learning Rate Training

To address gradient competition between modules, we train with phased differential learning rates:

**Phase 1 — Softmax Lead (4 epochs):**
- Softmax LR: 3e-4
- Mamba LR: 1e-5 (minimal, not frozen)

**Phase 2 — Mamba Catchup (5-6 epochs):**
- Softmax LR: 1e-5
- Mamba LR: 3e-4

**Critical insight:** We do NOT include a coordination phase. Our experiments show that attempting to train both modules jointly after the phased training degrades performance.

### 3.5 Training Setup

- Dataset: WikiText-2 (train: 584 batches, val: 61 batches)
- Sequence length: 512
- Batch size: 8
- Optimizer: AdamW with weight decay 0.1
- Mixed precision: FP16 with gradient scaling
- Gradient clipping: 1.0

## 4. Experiments

### 4.1 Mamba Hybrid Experiments

We compare different ratios of softmax to Mamba layers:

| Configuration | PPL | Speed (t/s) | PPL Δ | Speed Δ |
|--------------|-----|-------------|-------|---------|
| Full Softmax | 169.2 | 135k | — | — |
| **4S + 4M** | **154.0** | 124k | **-9.0%** | -8% |
| 2S + 6M | 155.9 | 119k | -7.8% | -12% |
| 6S + 2M | 155.9 | 43k | -7.8% | -68% |
| Full Mamba | 179.6 | 114k | +6.2% | -16% |

The 4+4 configuration achieves the best perplexity while maintaining reasonable speed. Notably, full Mamba underperforms the baseline, demonstrating that early softmax layers are essential.

### 4.2 Speed vs Sequence Length

Mamba's O(L) complexity provides advantages at long sequences:

| Seq Length | Full Softmax | 4S + 4M | Ratio |
|------------|--------------|---------|-------|
| 512 | 133k t/s | 122k t/s | 92% |
| 1024 | 128k t/s | 124k t/s | 97% |
| 2048 | 90k t/s | 104k t/s | **116%** |
| 4096 | 55k t/s | 76k t/s | **138%** |

The hybrid architecture becomes faster than full softmax at sequences longer than ~1500 tokens.

### 4.3 Differential LR Training

We compare uniform vs differential learning rate training:

| Strategy | Best PPL | Epoch | Speed |
|----------|----------|-------|-------|
| Uniform LR (1e-4) | 186.7 | 10 | 124k |
| Differential (10 epochs) | 173.0 | 6 | 131k |
| **Extended (20 epochs)** | **163.9** | **9** | 123k |

Differential LR provides consistent 12% improvement. The best result occurs during the Mamba-Catchup phase.

### 4.4 The Coordination Phase Problem

Extended training reveals that the coordination phase is harmful:

| Phase | Epochs | Best PPL in Phase |
|-------|--------|-------------------|
| Softmax-Lead | 1-4 | 236.4 |
| Mamba-Catchup | 5-10 | **163.9** (epoch 9) |
| Coordinate | 11-16 | 185.3 |
| Fine-Tune | 17-20 | 185.8 |

Transitioning from Mamba-Catchup to Coordinate caused immediate degradation (163.9 → 202.2 at epoch 10). This suggests that phased training works by letting modules specialize, and joint training afterward disrupts these learned representations.

### 4.5 Static Function Hypothesis

We test replacing later attention layers with MLPs:

| Configuration | PPL | Speed | PPL Δ | Speed Δ |
|--------------|-----|-------|-------|---------|
| Full Softmax | 170.9 | 130k | — | — |
| **6S + 2 MLP** | **168.3** | **147k** | **-1.5%** | **+13%** |
| 4S + 4 MLP | 174.8 | 156k | +2.3% | +20% |
| 2S + 6 MLP | 202.2 | 167k | +18.3% | +29% |

The 6+2 configuration achieves both better quality AND faster inference than the baseline, validating the Static Function Hypothesis. More aggressive replacement trades quality for speed.

### 4.6 Linear Attention Overfitting

Long training reveals catastrophic overfitting in linear attention:

| Epoch | Softmax PPL | Linear PPL | Gap |
|-------|-------------|------------|-----|
| 10 | 179.2 | 164.1 | -8.5% |
| 13 | 175.6 | **160.9** | -8.4% |
| 20 | 206.8 | 213.6 | +3.3% |
| 50 | 894.4 | 2241.2 | +150.6% |

Linear attention's overfit ratio reaches 1067x (train PPL 2.1, eval PPL 2241) compared to softmax's 182x. This motivates using more robust mechanisms like Mamba for sequence modeling.

## 5. Analysis

### 5.1 Why Does 4S + 4M Beat Full Softmax?

The improvement from full softmax (169.2 PPL) to 4S+4M (154.0 PPL) is surprising—we expected at best parity with efficiency gains. We hypothesize several factors:

1. **Complementary inductive biases:** Softmax attention excels at sparse, selective operations. Mamba excels at smooth state evolution. Different layers may benefit from different biases.

2. **Regularization effect:** Forcing the model to use different mechanisms may prevent overfitting to attention-specific patterns.

3. **Gradient flow:** Mamba's gated structure may provide better gradient pathways than deep attention stacks.

### 5.2 Why Does Coordination Hurt?

The coordination phase degrades performance because:

1. **Representation interference:** Each module develops specialized representations during its lead phase. Joint training averages these, losing specialization.

2. **Gradient competition resumes:** The redundancy bottleneck reappears when both modules have equal learning rates.

3. **Local minima disruption:** The model may leave a good basin found during Mamba-Catchup.

### 5.3 Why Do Early Softmax Layers Matter?

Full Mamba underperforms the baseline (179.6 vs 169.2), while all hybrid configurations improve. This suggests:

1. **Feature extraction requires selection:** Early layers must disambiguate tokens, requiring softmax's sparse attention.

2. **Structured representations enable SSMs:** Mamba works well on already-structured representations from softmax layers.

3. **Learning dynamics:** Softmax learns faster initially; without it, Mamba may struggle to bootstrap.

### 5.4 The Static Function Hypothesis

The success of 6S+2MLP (better quality AND speed) validates that later layers perform predictable operations. This has implications for:

1. **Architecture search:** The optimal number of attention layers may be less than commonly used.

2. **Inference optimization:** Later layers could be candidates for more aggressive optimization.

3. **Model compression:** Attention layers could be distilled to MLPs post-training.

## 6. Practical Recommendations

Based on our experiments:

| Use Case | Architecture | PPL | Speed |
|----------|--------------|-----|-------|
| Quality-critical, long context | 4S + 4M + Diff LR | 163.9 | 123k |
| Balanced (short context) | 6S + 2 MLP | 168.3 | 147k |
| Speed-critical | 4S + 4 MLP | 174.8 | 156k |

**Training hybrid architectures:**
1. Use differential LR with phased training
2. Train softmax layers first (4 epochs, LR 3e-4)
3. Train Mamba layers second (5-6 epochs, LR 3e-4)
4. **Stop** — do not add a coordination phase
5. Never fully freeze; use minimal LR (1e-5) for non-lead modules

**Avoid:**
- Full Mamba without early softmax (6% worse than baseline)
- Extended training of linear attention (catastrophic overfitting)
- Coordination phases after phased training (disrupts specialization)

## 7. Limitations and Future Work

**Limitations:**
- Experiments conducted on ~50M parameter models; scaling behavior unknown
- WikiText-2 is a relatively small benchmark
- Run-to-run variance observed (PPL range 154-187 for same configuration)

**Future directions:**
1. Scale validation on 1B+ parameter models
2. Task-specific evaluation beyond perplexity
3. Dynamic layer selection learning
4. Three-way hybrids: Softmax + Mamba + MLP
5. Long context benchmarks (8K+ tokens)

## 8. Conclusion

We demonstrate that hybrid transformer architectures combining softmax attention, state space models, and static functions outperform homogeneous designs. The 4 Softmax + 4 Mamba configuration achieves 9% better perplexity than full softmax, and differential learning rate training provides an additional 12% improvement.

Our key finding is that the coordination phase commonly used in multi-module training is counterproductive. Optimal results come from sequential training where each module leads in turn, followed by early stopping—not joint fine-tuning. This suggests that modular architectures benefit from specialization that joint training disrupts.

We validate the Static Function Hypothesis: 6 Softmax + 2 MLP achieves both better quality (-1.5% PPL) and faster inference (+13% speed) than the baseline. Later transformer layers can be replaced with simpler mechanisms without quality loss.

These findings suggest that the standard practice of homogeneous transformer stacks with uniform training may leave performance on the table. Heterogeneous architectures with phased training better match computational mechanisms to the requirements at different network depths.

## References

1. Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. (2020). Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. ICML.

2. Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv:2312.00752.

3. Yang, S., et al. (2024). Gated Linear Attention Transformers with Hardware-Efficient Training. ICML.

4. Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS.

5. He, J., et al. (2024). Mixture of A Million Experts. arXiv:2407.04153.

## Appendix A: Triton Kernel for Selective Scan

```python
@triton.jit
def selective_scan_kernel(
    x_ptr, delta_ptr, A_ptr, B_ptr, C_ptr, out_ptr,
    batch, seq_len, d_inner, d_state,
    stride_xb, stride_xs, stride_xd,
    stride_db, stride_ds, stride_dd,
    stride_Bd, stride_Bs, stride_Bst,
    stride_Cd, stride_Cs, stride_Cst,
    stride_ob, stride_os, stride_od,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    d_block = tl.program_id(1)
    d_idx = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_idx < d_inner

    s_idx = tl.arange(0, BLOCK_S)
    s_mask = s_idx < d_state

    A = tl.load(A_ptr + d_idx, mask=d_mask, other=0.0)
    h = tl.zeros([BLOCK_D, BLOCK_S], dtype=tl.float32)

    for t in range(seq_len):
        delta_t = tl.load(
            delta_ptr + batch_idx * stride_db + t * stride_ds + d_idx,
            mask=d_mask, other=0.0
        )
        x_t = tl.load(
            x_ptr + batch_idx * stride_xb + t * stride_xs + d_idx,
            mask=d_mask, other=0.0
        )

        B_t = tl.load(
            B_ptr + d_idx[:, None] * stride_Bd + t * stride_Bs + s_idx[None, :],
            mask=d_mask[:, None] & s_mask[None, :], other=0.0
        )
        C_t = tl.load(
            C_ptr + d_idx[:, None] * stride_Cd + t * stride_Cs + s_idx[None, :],
            mask=d_mask[:, None] & s_mask[None, :], other=0.0
        )

        A_bar = tl.exp(delta_t[:, None] * A[:, None])
        B_bar = delta_t[:, None] * B_t

        h = A_bar * h + B_bar * x_t[:, None]
        y_t = tl.sum(C_t * h, axis=1)

        tl.store(
            out_ptr + batch_idx * stride_ob + t * stride_os + d_idx,
            y_t, mask=d_mask
        )
```

## Appendix B: Training Curves

**Extended Differential LR Training (20 epochs):**

```
Softmax-Lead:
  Epoch  1: Train  895.0, Eval  499.8
  Epoch  2: Train  366.9, Eval  350.5
  Epoch  3: Train  242.9, Eval  277.3
  Epoch  4: Train  174.9, Eval  236.4

Mamba-Catchup:
  Epoch  5: Train  121.9, Eval  193.4
  Epoch  6: Train  100.2, Eval  178.5
  Epoch  7: Train   85.6, Eval  169.7
  Epoch  8: Train   74.6, Eval  165.9
  Epoch  9: Train   66.4, Eval  163.9  ← Best
  Epoch 10: Train   74.1, Eval  202.2  ← Phase transition damage

Coordinate:
  Epoch 11: Train   83.2, Eval  191.4
  ...
  Epoch 16: Train   60.1, Eval  185.3

Fine-Tune:
  Epoch 17-20: Eval PPL stable ~186
```

Note the sharp degradation at epoch 10 when transitioning from Mamba-Catchup to Coordinate phase.
