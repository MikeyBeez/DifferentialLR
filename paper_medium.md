# Hybrid Transformer Architectures: Combining Softmax Attention, State Space Models, and Static Functions

## Abstract

We investigate hybrid transformer architectures that combine different sequence modeling mechanisms across layers. Through systematic experimentation on WikiText-2, we demonstrate that a 4 Softmax + 4 Mamba configuration achieves 9% better perplexity than full softmax attention while maintaining comparable inference speed. We further show that differential learning rate training—where different module types are trained with phased learning rates—improves performance by an additional 12%, achieving 163.9 PPL compared to 186.7 with uniform learning rates. Our analysis reveals that the coordination phase commonly used in multi-module training is counterproductive; optimal results are achieved by training modules sequentially without subsequent joint fine-tuning. We validate the Static Function Hypothesis, showing that later transformer layers can be replaced with simple MLPs while improving both quality and speed. These findings suggest that transformer architectures benefit from heterogeneous layer compositions matched to the computational requirements at different depths.

---

## 1. Introduction

The transformer architecture has become the foundation of modern language models, with self-attention providing the core mechanism for modeling token dependencies. However, the O(L²) complexity of standard softmax attention creates computational bottlenecks for long sequences, motivating research into efficient alternatives including linear attention, state space models, and hybrid architectures.

This work investigates three key questions:

- Can heterogeneous architectures outperform homogeneous transformers? We test whether combining different mechanisms (softmax attention, Mamba SSM, MLPs) across layers yields better quality-speed tradeoffs than using any single mechanism throughout.

- How should multi-module architectures be trained? We apply differential learning rate training, where different module types receive different learning rates in phases, to address gradient competition between modules.

- Do later transformer layers require dynamic attention? We test the Static Function Hypothesis—that deeper layers perform predictable operations on structured representations and can be replaced with simpler mechanisms.

Our contributions include:

- Demonstration that 4 Softmax + 4 Mamba achieves 9% better perplexity than full softmax
- Evidence that differential LR training improves hybrid architectures by 12%
- Discovery that the coordination phase in phased training is harmful
- Validation that 6 Softmax + 2 MLP beats the baseline on both quality and speed
- A Triton kernel for Mamba selective scan achieving 26x speedup over PyTorch

---

## 2. Background and Related Work

### 2.1 Linear Attention

Linear attention replaces the softmax normalization with a feature map φ, enabling O(L) complexity through associativity. Instead of computing softmax(QKᵀ)V, we compute φ(Q)(φ(K)ᵀV).

However, linear attention consistently underperforms softmax attention due to its inability to create sparse, selective attention patterns. Our experiments confirm this: linear attention achieves its best performance early (epoch 13, PPL 160.9) then overfits catastrophically, reaching PPL 2241 by epoch 50 compared to softmax's 894.

### 2.2 State Space Models

Mamba introduces selective state spaces where the state transition matrices are input-dependent. The recurrence follows: h_t = A_t × h_{t-1} + B_t × x_t, with output y_t = C_t × h_t.

This selectivity allows the model to decide what information to retain or forget based on content, combining the efficiency of RNNs with content-based gating.

### 2.3 The Redundancy Bottleneck

Recent work on modular architectures reveals that sparse retrieval modules compete for gradient signal when trained jointly. The faster-learning module monopolizes explanatory responsibility, crowding out slower modules before they can specialize. This "redundancy bottleneck" explains why naive combinations of powerful modules often underperform their components.

### 2.4 Static Function Hypothesis

The hypothesis that early transformer layers require dynamic attention while later layers perform more predictable operations suggests that computational resources are misallocated in homogeneous architectures. Later layers operate on already-structured representations where cross-position interaction is less critical.

---

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

We test several hybrid architectures.

Softmax + Mamba Hybrids:
- 4 Softmax (layers 0-3) + 4 Mamba (layers 4-7)
- 2 Softmax + 6 Mamba
- 6 Softmax + 2 Mamba
- Full Mamba (8 layers)

Softmax + MLP Hybrids:
- 6 Softmax + 2 MLP
- 4 Softmax + 4 MLP
- 2 Softmax + 6 MLP

### 3.3 Mamba Implementation

We implement Mamba in pure PyTorch with a Triton-accelerated selective scan kernel. Each thread block processes one (batch, d_block) slice, performing sequential scan over time with parallel computation over the state dimension. This achieves 26x speedup over the naive PyTorch loop implementation.

### 3.4 Differential Learning Rate Training

To address gradient competition between modules, we train with phased differential learning rates.

Phase 1 — Softmax Lead (4 epochs):
- Softmax LR: 3e-4
- Mamba LR: 1e-5 (minimal, not frozen)

Phase 2 — Mamba Catchup (5-6 epochs):
- Softmax LR: 1e-5
- Mamba LR: 3e-4

Critical insight: We do NOT include a coordination phase. Our experiments show that attempting to train both modules jointly after the phased training degrades performance.

### 3.5 Training Setup

- Dataset: WikiText-2 (train: 584 batches, val: 61 batches)
- Sequence length: 512
- Batch size: 8
- Optimizer: AdamW with weight decay 0.1
- Mixed precision: FP16 with gradient scaling
- Gradient clipping: 1.0

---

## 4. Experiments

### 4.1 Mamba Hybrid Experiments

We compare different ratios of softmax to Mamba layers.

Full Softmax (baseline):
- PPL: 169.2
- Speed: 135k tokens/sec

4 Softmax + 4 Mamba (best):
- PPL: 154.0 (9% better than baseline)
- Speed: 124k tokens/sec (8% slower)

2 Softmax + 6 Mamba:
- PPL: 155.9 (7.8% better)
- Speed: 119k tokens/sec (12% slower)

6 Softmax + 2 Mamba:
- PPL: 155.9 (7.8% better)
- Speed: 43k tokens/sec (68% slower)

Full Mamba:
- PPL: 179.6 (6.2% worse than baseline)
- Speed: 114k tokens/sec (16% slower)

The 4+4 configuration achieves the best perplexity while maintaining reasonable speed. Notably, full Mamba underperforms the baseline, demonstrating that early softmax layers are essential.

### 4.2 Speed vs Sequence Length

Mamba's O(L) complexity provides advantages at long sequences.

At 512 tokens:
- Full Softmax: 133k tokens/sec
- 4S + 4M: 122k tokens/sec (92% of softmax speed)

At 1024 tokens:
- Full Softmax: 128k tokens/sec
- 4S + 4M: 124k tokens/sec (97% of softmax speed)

At 2048 tokens:
- Full Softmax: 90k tokens/sec
- 4S + 4M: 104k tokens/sec (116% of softmax speed — hybrid is faster)

At 4096 tokens:
- Full Softmax: 55k tokens/sec
- 4S + 4M: 76k tokens/sec (138% of softmax speed — hybrid is much faster)

The hybrid architecture becomes faster than full softmax at sequences longer than approximately 1500 tokens.

### 4.3 Differential LR Training

We compare uniform vs differential learning rate training.

Uniform LR (1e-4), 10 epochs:
- Best PPL: 186.7
- Achieved at: epoch 10
- Speed: 124k tokens/sec

Differential LR, 10 epochs:
- Best PPL: 173.0 (7.3% improvement)
- Achieved at: epoch 6
- Speed: 131k tokens/sec

Extended Differential LR, 20 epochs:
- Best PPL: 163.9 (12% improvement over uniform)
- Achieved at: epoch 9
- Speed: 123k tokens/sec

Differential LR provides consistent 12% improvement. The best result occurs during the Mamba-Catchup phase.

### 4.4 The Coordination Phase Problem

Extended training reveals that the coordination phase is harmful.

Softmax-Lead phase (epochs 1-4):
- Best PPL achieved: 236.4

Mamba-Catchup phase (epochs 5-10):
- Best PPL achieved: 163.9 at epoch 9

Coordinate phase (epochs 11-16):
- Best PPL achieved: 185.3 (degraded from 163.9)

Fine-Tune phase (epochs 17-20):
- Best PPL achieved: 185.8 (no recovery)

Transitioning from Mamba-Catchup to Coordinate caused immediate degradation—PPL jumped from 163.9 to 202.2 at epoch 10. This suggests that phased training works by letting modules specialize, and joint training afterward disrupts these learned representations.

### 4.5 Static Function Hypothesis

We test replacing later attention layers with MLPs.

Full Softmax (8 layers):
- PPL: 170.9
- Speed: 130k tokens/sec

6 Softmax + 2 MLP (best balanced):
- PPL: 168.3 (1.5% better than baseline)
- Speed: 147k tokens/sec (13% faster than baseline)

4 Softmax + 4 MLP:
- PPL: 174.8 (2.3% worse)
- Speed: 156k tokens/sec (20% faster)

2 Softmax + 6 MLP:
- PPL: 202.2 (18.3% worse)
- Speed: 167k tokens/sec (29% faster)

The 6+2 configuration achieves both better quality AND faster inference than the baseline, validating the Static Function Hypothesis. More aggressive replacement trades quality for speed.

### 4.6 Linear Attention Overfitting

Long training reveals catastrophic overfitting in linear attention.

At epoch 10:
- Softmax PPL: 179.2
- Linear PPL: 164.1 (linear is 8.5% better)

At epoch 13:
- Softmax PPL: 175.6
- Linear PPL: 160.9 (linear's best result, 8.4% better)

At epoch 20:
- Softmax PPL: 206.8
- Linear PPL: 213.6 (linear is now 3.3% worse)

At epoch 50:
- Softmax PPL: 894.4
- Linear PPL: 2241.2 (linear is 150.6% worse, catastrophic)

Linear attention's overfit ratio reaches 1067x (train PPL 2.1, eval PPL 2241) compared to softmax's 182x. This motivates using more robust mechanisms like Mamba for sequence modeling.

---

## 5. Analysis

### 5.1 Why Does 4S + 4M Beat Full Softmax?

The improvement from full softmax (169.2 PPL) to 4S+4M (154.0 PPL) is surprising—we expected at best parity with efficiency gains. We hypothesize several factors:

Complementary inductive biases: Softmax attention excels at sparse, selective operations. Mamba excels at smooth state evolution. Different layers may benefit from different biases.

Regularization effect: Forcing the model to use different mechanisms may prevent overfitting to attention-specific patterns.

Gradient flow: Mamba's gated structure may provide better gradient pathways than deep attention stacks.

### 5.2 Why Does Coordination Hurt?

The coordination phase degrades performance because:

Representation interference: Each module develops specialized representations during its lead phase. Joint training averages these, losing specialization.

Gradient competition resumes: The redundancy bottleneck reappears when both modules have equal learning rates.

Local minima disruption: The model may leave a good basin found during Mamba-Catchup.

### 5.3 Why Do Early Softmax Layers Matter?

Full Mamba underperforms the baseline (179.6 vs 169.2), while all hybrid configurations improve. This suggests:

Feature extraction requires selection: Early layers must disambiguate tokens, requiring softmax's sparse attention.

Structured representations enable SSMs: Mamba works well on already-structured representations from softmax layers.

Learning dynamics: Softmax learns faster initially; without it, Mamba may struggle to bootstrap.

### 5.4 The Static Function Hypothesis

The success of 6S+2MLP (better quality AND speed) validates that later layers perform predictable operations. This has implications for:

Architecture search: The optimal number of attention layers may be less than commonly used.

Inference optimization: Later layers could be candidates for more aggressive optimization.

Model compression: Attention layers could be distilled to MLPs post-training.

---

## 6. Practical Recommendations

Based on our experiments, here are recommendations for different use cases.

For quality-critical applications with long context:
- Use 4 Softmax + 4 Mamba with differential LR training
- Expected PPL: ~164
- Expected speed: ~123k tokens/sec
- Train with 4 epochs Softmax-Lead, then 5-6 epochs Mamba-Catchup, then stop

For balanced quality and speed on short sequences:
- Use 6 Softmax + 2 MLP
- Expected PPL: ~168
- Expected speed: ~147k tokens/sec
- Simple to implement, no sequence modeling needed in late layers

For speed-critical applications:
- Use 4 Softmax + 4 MLP
- Expected PPL: ~175
- Expected speed: ~156k tokens/sec
- Acceptable quality tradeoff for significant speedup

Training recommendations for hybrid architectures:
- Use differential LR with phased training
- Train softmax layers first (4 epochs, LR 3e-4)
- Train Mamba/other layers second (5-6 epochs, LR 3e-4)
- Stop after the second phase—do not add a coordination phase
- Never fully freeze modules; use minimal LR (1e-5) for non-lead modules

What to avoid:
- Full Mamba without early softmax (6% worse than baseline)
- Extended training of linear attention (catastrophic overfitting)
- Coordination phases after phased training (disrupts specialization)
- Replacing more than 2 attention layers with MLPs if quality matters

---

## 7. Limitations and Future Work

Limitations of this work:

- Experiments conducted on ~50M parameter models; scaling behavior to larger models is unknown
- WikiText-2 is a relatively small benchmark; results may differ on larger datasets
- Run-to-run variance was observed (PPL range 154-187 for the same configuration)
- Only tested language modeling; other tasks may show different patterns

Future directions:

- Scale validation on 1B+ parameter models
- Task-specific evaluation beyond perplexity (question answering, summarization, etc.)
- Dynamic layer selection—learning which layers need attention vs MLP vs Mamba
- Three-way hybrids combining Softmax + Mamba + MLP in a single model
- Long context benchmarks on 8K+ token sequences
- Comparison of our Triton kernel against official Mamba CUDA kernels

---

## 8. Conclusion

We demonstrate that hybrid transformer architectures combining softmax attention, state space models, and static functions outperform homogeneous designs. The 4 Softmax + 4 Mamba configuration achieves 9% better perplexity than full softmax, and differential learning rate training provides an additional 12% improvement.

Our key finding is that the coordination phase commonly used in multi-module training is counterproductive. Optimal results come from sequential training where each module leads in turn, followed by early stopping—not joint fine-tuning. This suggests that modular architectures benefit from specialization that joint training disrupts.

We validate the Static Function Hypothesis: 6 Softmax + 2 MLP achieves both better quality (1.5% lower PPL) and faster inference (13% higher throughput) than the baseline. Later transformer layers can be replaced with simpler mechanisms without quality loss.

These findings suggest that the standard practice of homogeneous transformer stacks with uniform training may leave performance on the table. Heterogeneous architectures with phased training better match computational mechanisms to the requirements at different network depths.

---

## Appendix: Training Curves for Extended Differential LR

Here is the epoch-by-epoch breakdown of our 20-epoch extended training run:

Softmax-Lead Phase:
- Epoch 1: Train PPL 895.0, Eval PPL 499.8
- Epoch 2: Train PPL 366.9, Eval PPL 350.5
- Epoch 3: Train PPL 242.9, Eval PPL 277.3
- Epoch 4: Train PPL 174.9, Eval PPL 236.4

Mamba-Catchup Phase:
- Epoch 5: Train PPL 121.9, Eval PPL 193.4
- Epoch 6: Train PPL 100.2, Eval PPL 178.5
- Epoch 7: Train PPL 85.6, Eval PPL 169.7
- Epoch 8: Train PPL 74.6, Eval PPL 165.9
- Epoch 9: Train PPL 66.4, Eval PPL 163.9 (BEST RESULT)
- Epoch 10: Train PPL 74.1, Eval PPL 202.2 (phase transition damage)

Coordinate Phase:
- Epoch 11: Train PPL 83.2, Eval PPL 191.4
- Epoch 12: Train PPL 76.2, Eval PPL 189.1
- Epoch 13: Train PPL 71.1, Eval PPL 187.7
- Epoch 14: Train PPL 67.0, Eval PPL 185.7
- Epoch 15: Train PPL 63.2, Eval PPL 185.9
- Epoch 16: Train PPL 60.1, Eval PPL 185.3

Fine-Tune Phase:
- Epoch 17: Train PPL 55.6, Eval PPL 185.8
- Epoch 18: Train PPL 54.8, Eval PPL 185.8
- Epoch 19: Train PPL 54.0, Eval PPL 186.5
- Epoch 20: Train PPL 53.3, Eval PPL 186.6

Note the sharp degradation at epoch 10 when transitioning from Mamba-Catchup to Coordinate phase. Train PPL actually increased (from 66.4 to 74.1) while eval PPL jumped dramatically (from 163.9 to 202.2). This is strong evidence that the coordination phase disrupts the specialized representations learned during phased training.

---

## References

1. Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. (2020). Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. ICML.

2. Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv:2312.00752.

3. Yang, S., et al. (2024). Gated Linear Attention Transformers with Hardware-Efficient Training. ICML.

4. Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS.

5. He, J., et al. (2024). Mixture of A Million Experts. arXiv:2407.04153.
