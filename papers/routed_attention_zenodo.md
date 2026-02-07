# Routed Attention: Learning When to Think Hard

**Mike Bonsignore**

February 2026

---

## Abstract

We present routed attention, an architecture that learns to dynamically select between O(N) causal convolution and O(N²) softmax attention on a per-position basis. A lightweight router network examines each position and routes it to the appropriate computational pathway. We demonstrate that naive joint optimization of task performance and attention minimization fails due to insufficient exploration, and propose a curriculum learning solution: first train with no attention penalty (λ=0) to learn task-relevant routing patterns, then gradually increase λ to minimize attention usage while maintaining accuracy. On an associative recall benchmark requiring long-range retrieval, routed attention achieves 100% accuracy with only 0.3% attention usage at distance 126 (99.7% compute savings), and 100% accuracy with 25% attention usage at distance 510 (75% compute savings). The approach provides a principled method for combining efficient local processing with expensive global attention only where necessary.

**Keywords:** attention mechanisms, efficient transformers, mixture of experts, curriculum learning, sequence modeling

---

## 1. Introduction

Transformer attention (Vaswani et al., 2017) computes pairwise interactions between all positions in a sequence, achieving O(N²) time and space complexity. This quadratic scaling limits practical sequence lengths and motivates extensive research into efficient alternatives.

Recent work demonstrates that O(N) causal convolution can match or exceed softmax attention on language modeling perplexity (Bonsignore, 2026; Poli et al., 2023). However, convolution is fundamentally local: it can only aggregate information within its kernel size. Tasks requiring precise long-range retrieval—such as matching pronouns to antecedents or looking up key-value pairs—cannot be solved by local operations alone.

This creates a dilemma. Convolution wins on efficiency and often on quality for tasks with high local correlation, but fails completely on retrieval tasks. Attention can perform retrieval but scales poorly. Practitioners must choose one or the other.

We propose routed attention, which eliminates this choice. A learned router examines each position and decides whether to use cheap convolution or expensive attention. Positions that can be predicted locally use convolution; positions requiring global context use attention. The model learns this routing end-to-end.

Our main contributions are:

1. **Architecture**: A per-position routing mechanism using Gumbel-softmax for differentiable discrete selection between O(N) and O(N²) pathways.

2. **Training methodology**: A curriculum learning approach that first learns task-relevant routing (λ=0), then optimizes for efficiency (increasing λ). We show that naive joint optimization fails due to premature convergence to the cheap pathway.

3. **Empirical results**: On associative recall, routed attention achieves 75-99% compute savings while matching full attention accuracy, up to distance 510.

---

## 2. Related Work

### 2.1 Efficient Attention

Linear attention (Katharopoulos et al., 2020) replaces softmax with kernel feature maps, achieving O(N) complexity. Performer (Choromanski et al., 2021) approximates softmax attention via random features. Flash Attention (Dao et al., 2022) reduces memory through tiling but maintains O(N²) computation.

These approaches apply uniformly to all positions. Routed attention instead selectively applies expensive computation only where needed.

### 2.2 State Space Models

Mamba (Gu & Dao, 2023) and S4 (Gu et al., 2022) achieve O(N) sequence modeling through structured state spaces. Hyena (Poli et al., 2023) uses long convolutions. These models excel at language modeling but, like convolution, cannot perform content-addressable retrieval beyond their effective memory horizon.

### 2.3 Mixture of Experts

Mixture of Experts (Shazeer et al., 2017; Fedus et al., 2022) routes tokens to different expert networks. Switch Transformer (Fedus et al., 2022) uses top-1 routing for efficiency. MoE typically routes between experts of similar computational cost for capacity scaling.

Routed attention differs in that experts have asymmetric costs (O(N) vs O(N²)), and the goal is efficiency rather than capacity. The routing objective explicitly minimizes usage of the expensive pathway.

### 2.4 Adaptive Computation

Adaptive Computation Time (Graves, 2016) allows variable computation per position. Universal Transformers (Dehghani et al., 2019) apply variable depth. Early exit strategies (Schwartz et al., 2020) skip layers for easy examples.

Routed attention applies this principle to attention specifically, routing between qualitatively different operations rather than adjusting depth.

---

## 3. Method

### 3.1 Architecture

Given input hidden states $\mathbf{x} \in \mathbb{R}^{B \times N \times D}$, routed attention computes:

**Router:**
$$\mathbf{r} = \text{Router}(\mathbf{x}) \in \mathbb{R}^{B \times N \times 2}$$

The router is a small MLP:
$$\text{Router}(\mathbf{x}) = \mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 \mathbf{x})$$

where $\mathbf{W}_1 \in \mathbb{R}^{D \times D/4}$ and $\mathbf{W}_2 \in \mathbb{R}^{D/4 \times 2}$.

**Routing decision:**
$$\mathbf{g} = \text{Gumbel-Softmax}(\mathbf{r}, \tau) \in \{0, 1\}^{B \times N \times 2}$$

During training, Gumbel-softmax (Jang et al., 2017) provides differentiable discrete sampling. During inference, we use hard argmax.

**Pathway computation:**
$$\mathbf{y}_{\text{conv}} = \text{ConvAttention}(\mathbf{x})$$
$$\mathbf{y}_{\text{attn}} = \text{SoftmaxAttention}(\mathbf{x})$$

**Output:**
$$\mathbf{y} = \mathbf{g}_{:,:,0} \odot \mathbf{y}_{\text{conv}} + \mathbf{g}_{:,:,1} \odot \mathbf{y}_{\text{attn}}$$

The convolution pathway uses learned causal convolution with kernel size 64. The attention pathway uses standard multi-head softmax attention.

### 3.2 Training Objective

The loss combines task performance with an attention cost penalty:

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \cdot \frac{1}{N} \sum_{i=1}^{N} g_{i,1}$$

where $g_{i,1}$ indicates attention usage at position $i$. Higher $\lambda$ penalizes attention more aggressively.

### 3.3 Curriculum Learning

Naive training with $\lambda > 0$ from initialization fails. The randomly-initialized router has no knowledge of which positions benefit from attention. The cost penalty immediately pushes all routing toward convolution. Since convolution cannot solve long-range retrieval, the model never learns the task, and the router never discovers which positions need attention.

We propose two-phase curriculum learning:

**Phase 1: Task learning ($\lambda = 0$)**
Train until the model solves the task. With no attention penalty, the router freely explores and discovers which positions benefit from attention.

**Phase 2: Efficiency optimization ($\lambda \to \lambda_{\text{target}}$)**
Gradually increase $\lambda$ over training. The router already knows which positions need attention; it now minimizes attention usage on positions that don't.

The gradual $\lambda$ increase prevents collapse. Jumping directly to high $\lambda$ causes the router to abandon attention entirely, losing the task solution.

---

## 4. Experiments

### 4.1 Task: Associative Recall

We use associative recall as a benchmark requiring precise long-range retrieval. The model sees a sequence of key-value token pairs, followed by a query key, and must output the corresponding value:

$$[K_1:V_1] \; [K_2:V_2] \; \ldots \; [K_n:V_n] \; [Q] \; [?]$$

where $Q = K_i$ for some $i$, and the target is $V_i$.

This task is impossible for purely local models when the query-key distance exceeds the receptive field. It requires content-addressable memory.

### 4.2 Setup

- **Model**: 6-layer transformer, 256 dimensions, 8 heads
- **Vocabulary**: 100 tokens (50 keys, 50 values)
- **Sequence lengths**: 128, 256, 512, 1024, 2048 tokens
- **Training**: AdamW, learning rate 3e-4, phase 1 up to 25 epochs, phase 2 up to 20 epochs
- **Evaluation**: Accuracy on held-out sequences

We compare four configurations:
1. **Attention Only**: Standard O(N²) attention (baseline)
2. **Conv Only**: O(N) learned convolution
3. **Routed λ=0.1**: Conservative attention penalty
4. **Routed λ=0.5**: Aggressive attention penalty

### 4.3 Results

| Distance | Attn Only | Conv Only | Routed λ=0.1 | Routed λ=0.5 |
|----------|-----------|-----------|--------------|--------------|
| 126      | 100%      | 1%        | 100% (16%)   | 100% (0.3%)  |
| 254      | 100%      | 2%        | 100% (25%)   | 99% (0.2%)   |
| 510      | 100%      | 2%        | 100% (25%)   | 100% (25%)   |
| 1022     | 99%       | 2%        | 36% (74%)    | 98% (90%)    |
| 2046     | 82%       | 2%        | 2% (35%)     | 97% (38%)    |

*Table 1: Associative recall accuracy. Percentages in parentheses indicate attention usage (fraction of positions using the attention pathway).*

**Short distances (126-254)**: Routed attention achieves 100% accuracy with near-zero attention usage (0.2-0.3%), representing 99.7% compute savings.

**Medium distances (510)**: Both routed configurations achieve 100% accuracy with 25% attention usage, representing 75% compute savings.

**Long distances (1024+)**: Phase 1 requires more epochs to converge. With λ=0.5, routed attention reaches 97-98% accuracy, approaching the 82-99% of full attention despite using only 38-90% attention.

### 4.4 Routing Patterns

The learned routing is interpretable. For a sequence $[K_1:V_1] \ldots [Q] [?]$:
- Key-value pair positions predominantly route to convolution (local processing suffices)
- Query and answer positions route to attention (global search required)

This approaches the theoretical optimum: O(N²) computation only for positions requiring global context.

### 4.5 Ablation: Curriculum Learning

| Training Strategy | Distance 126 Accuracy | Distance 254 Accuracy |
|-------------------|----------------------|----------------------|
| Joint (λ=0.1 from start) | 1% | 1% |
| Joint (λ=0.5 from start) | 1% | 1% |
| Curriculum (λ=0 → 0.1) | 100% | 100% |
| Curriculum (λ=0 → 0.5) | 100% | 99% |

*Table 2: Curriculum learning is essential. Joint optimization from initialization fails completely.*

Without curriculum learning, the model achieves only random-chance accuracy regardless of $\lambda$ value. The attention cost penalty prevents exploration of the attention pathway, and the router never discovers its utility.

---

## 5. Analysis

### 5.1 Compute Savings

The effective complexity of routed attention is O(N + pN²), where p is the fraction of positions using attention.

For p = 0.003 (distance 126, λ=0.5), complexity approaches O(N).
For p = 0.25 (distance 510), complexity is O(N + 0.25N²), a 4× reduction in attention operations.

Savings increase with sequence length as the O(N²) term dominates.

### 5.2 The 25% Floor

At distance 510+, even aggressive λ=0.5 cannot push attention below 25%. This suggests approximately one quarter of positions genuinely require global context for this task. The router has found the minimum attention necessary to maintain accuracy.

### 5.3 λ Tradeoff

Higher λ produces more aggressive attention minimization:
- λ=0.1: Conservative. Achieves 100% accuracy, 16-25% attention.
- λ=0.5: Aggressive. Pushes toward 0% attention where possible, occasionally at slight accuracy cost (99% vs 100%).

The optimal λ depends on the accuracy-efficiency tradeoff requirements of the application.

---

## 6. Inference with KV Cache

A practical concern for autoregressive generation: how does routed attention interact with KV caching?

### 6.1 The Challenge

Standard transformer inference caches key and value projections to avoid recomputation. With routed attention:
- If position i routes to conv, it has no K/V cached
- If position j routes to attention, it may need to attend to position i
- The cache becomes inconsistent

### 6.2 Solution: Always Cache, Route Aggregation

We adopt the strategy: **cache K/V for all positions, route only the aggregation**.

Every position computes and caches K, V (cheap O(1) projections). Only the expensive O(N) attention aggregation (Q·K^T) is routed:
- **Attention-routed positions**: Full aggregation over cached K/V
- **Conv-routed positions**: Skip aggregation, use O(1) convolution over last `kernel_size` hidden states

### 6.3 Empirical Results

We measure per-token latency on GPU (RTX 5070 Ti):

| Operation | Latency | Scaling |
|-----------|---------|---------|
| Attention aggregation | ~335 μs | O(N) |
| Conv (kernel=64) | ~148 μs | O(1) |
| **Speedup** | **2.3x** | - |

The 2.3x speedup is constant across context lengths (tested up to 8192 tokens).

### 6.4 Effective Speedup

Net inference speedup depends on routing fraction:

| Routing | Attention % | Conv % | Effective Speedup |
|---------|-------------|--------|-------------------|
| Distance 126 (λ=0.5) | 0.3% | 99.7% | ~2.2x |
| Distance 510 | 25% | 75% | ~1.45x |
| Distance 1024 | 90% | 10% | ~1.06x |

### 6.5 Memory

Memory usage is **unchanged** from standard attention. The full KV cache must be retained because any future position might route to attention and require the complete history.

The savings are purely computational: fewer positions perform the expensive O(N) aggregation.

---

## 8. Limitations

**Long-range convergence**: At distances beyond 1024, phase 1 requires more than 25 epochs to learn the task. Architectural modifications (larger conv kernels, more capacity) may help.

**Task specificity**: We evaluate on associative recall, an artificial benchmark. Natural language may have different routing patterns.

**Training cost**: Both pathways execute during training regardless of routing, as gradients flow through both. Inference benefits from selective execution; training does not.

**Single-layer routing**: The router decides independently at each layer. Hierarchical routing—conv in early layers, attention in later layers—might improve efficiency.

---

## 9. Conclusion

Routed attention demonstrates that the choice between efficient O(N) models and accurate O(N²) attention is a false dichotomy. A learned router can dynamically select the appropriate computation per position, achieving 75-99% compute savings while matching full attention accuracy.

The key insight is curriculum learning: train to solve the task first (λ=0), then optimize for efficiency (increasing λ). This allows the router to discover which positions genuinely require expensive computation before being penalized for using it.

Routed attention provides a principled path toward efficient sequence models that maintain the retrieval capabilities of full attention while avoiding unnecessary quadratic computation.

---

## Code Availability

All code and experiments are available at:
https://github.com/MikeyBeez/DifferentialLR

---

## References

Bonsignore, M. (2026). Attention May Not Be What You Need. *Preprint*.

Choromanski, K., Likhosherstov, V., Dohan, D., Song, X., Gane, A., Sarlos, T., ... & Weller, A. (2021). Rethinking Attention with Performers. *ICLR*.

Dao, T., Fu, D., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. *NeurIPS*.

Dehghani, M., Gouws, S., Vinyals, O., Uszkoreit, J., & Kaiser, Ł. (2019). Universal Transformers. *ICLR*.

Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. *JMLR*.

Graves, A. (2016). Adaptive Computation Time for Recurrent Neural Networks. *arXiv:1603.08983*.

Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *arXiv:2312.00752*.

Gu, A., Goel, K., & Ré, C. (2022). Efficiently Modeling Long Sequences with Structured State Spaces. *ICLR*.

Jang, E., Gu, S., & Poole, B. (2017). Categorical Reparameterization with Gumbel-Softmax. *ICLR*.

Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. (2020). Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. *ICML*.

Poli, M., Massaroli, S., Nguyen, E., Fu, D. Y., Dao, T., Baccus, S., ... & Ré, C. (2023). Hyena Hierarchy: Towards Larger Convolutional Language Models. *ICML*.

Schwartz, R., Stanovsky, G., Swayamdipta, S., Dodge, J., & Smith, N. A. (2020). The Right Tool for the Job: Matching Model and Instance Complexities. *ACL*.

Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. *ICLR*.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention Is All You Need. *NeurIPS*.
