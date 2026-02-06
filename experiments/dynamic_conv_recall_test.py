#!/usr/bin/env python3
"""
Dynamic Convolution Associative Recall Test

The key insight: static conv kernels can't do retrieval because the weights
are the same regardless of input. We need DATA-DEPENDENT kernels.

This tests several approaches:
1. Hyena-style: current token generates its own kernel
2. Linear attention kernel trick: phi(Q) @ (phi(K)^T @ V)
3. Additive attention: W_q*q + W_k*k (Bahdanau style, O(N) with tricks)

The hypothesis: making weights dynamic will enable associative recall
while keeping O(N) complexity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass


@dataclass
class TestConfig:
    vocab_size: int = 256
    dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.0
    kernel_size: int = 64


# --- Attention Variants ---

class StandardAttention(nn.Module):
    """Standard O(N²) attention - baseline."""
    def __init__(self, config: TestConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.dim // config.num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(config.dim, config.dim * 3)
        self.out = nn.Linear(config.dim, config.dim)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class LinearAttention(nn.Module):
    """
    Linear attention with kernel trick.

    Instead of softmax(Q @ K^T) @ V, we use:
    phi(Q) @ (phi(K)^T @ V)

    Where phi(x) = elu(x) + 1 (ensures positivity)

    This is O(N) because we compute (K^T @ V) first: (D x N) @ (N x D) = (D x D)
    Then Q @ that: (N x D) @ (D x D) = (N x D)

    For causal: we use cumulative sum trick.
    """
    def __init__(self, config: TestConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.dim // config.num_heads
        self.dim = config.dim

        self.qkv = nn.Linear(config.dim, config.dim * 3)
        self.out = nn.Linear(config.dim, config.dim)

    def feature_map(self, x):
        """elu(x) + 1 ensures positivity for valid attention weights."""
        return F.elu(x) + 1

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (B, H, T, D)

        # Apply feature map
        q = self.feature_map(q)
        k = self.feature_map(k)

        # Causal linear attention via cumsum
        # For each position i, we want sum_{j<=i} phi(q_i) @ phi(k_j)^T @ v_j
        # = phi(q_i) @ sum_{j<=i} (phi(k_j) outer v_j)
        # = phi(q_i) @ cumsum(k.unsqueeze(-1) * v.unsqueeze(-2))

        # k: (B, H, T, D), v: (B, H, T, D)
        # kv: (B, H, T, D, D) - outer product
        kv = k.unsqueeze(-1) * v.unsqueeze(-2)  # (B, H, T, D_k, D_v)
        kv_cumsum = torch.cumsum(kv, dim=2)  # Causal: position i sees 0..i

        # For normalization: sum of keys
        k_cumsum = torch.cumsum(k, dim=2)  # (B, H, T, D)

        # q @ kv_cumsum: (B, H, T, D) @ (B, H, T, D, D) -> (B, H, T, D)
        # Use einsum for clarity
        out = torch.einsum('bhtd,bhtde->bhte', q, kv_cumsum)

        # Normalize by sum of attention weights
        normalizer = torch.einsum('bhtd,bhtd->bht', q, k_cumsum).unsqueeze(-1)
        out = out / (normalizer + 1e-6)

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class DynamicConvAttention(nn.Module):
    """
    Hyena-style dynamic convolution.

    The current token generates its own local kernel weights.
    This makes the "attention" content-dependent while staying O(N).

    For each position i:
    1. Generate a small kernel from x[i]
    2. Apply that kernel to the local neighborhood
    """
    def __init__(self, config: TestConfig, kernel_size: int = 32):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.dim // config.num_heads
        self.kernel_size = kernel_size
        self.dim = config.dim

        self.v_proj = nn.Linear(config.dim, config.dim)

        # Generate kernel from current position
        self.kernel_gen = nn.Sequential(
            nn.Linear(config.dim, config.dim),
            nn.GELU(),
            nn.Linear(config.dim, config.num_heads * kernel_size)
        )

        self.out = nn.Linear(config.dim, config.dim)

    def forward(self, x):
        B, T, C = x.shape

        v = self.v_proj(x)  # (B, T, C)

        # Generate position-specific kernels
        kernels = self.kernel_gen(x)  # (B, T, H * K)
        kernels = kernels.view(B, T, self.num_heads, self.kernel_size)
        kernels = F.softmax(kernels, dim=-1)  # Normalize per head

        # Apply dynamic conv - each position has its own kernel
        # This is O(N * K) which is O(N) for fixed K
        v_heads = v.view(B, T, self.num_heads, self.head_dim)

        # Pad for causal conv
        v_padded = F.pad(v_heads, (0, 0, 0, 0, self.kernel_size - 1, 0))  # Pad T dimension

        # For each position, apply its kernel
        outputs = []
        for t in range(T):
            # Get the kernel for this position
            k = kernels[:, t]  # (B, H, K)

            # Get the window of values [t, t+K) in padded space = [t-K+1, t+1) in original
            window = v_padded[:, t:t + self.kernel_size]  # (B, K, H, D)
            window = window.permute(0, 2, 1, 3)  # (B, H, K, D)

            # Weighted sum
            out_t = torch.einsum('bhk,bhkd->bhd', k, window)  # (B, H, D)
            outputs.append(out_t)

        out = torch.stack(outputs, dim=1)  # (B, T, H, D)
        out = out.reshape(B, T, C)

        return self.out(out)


class DynamicConvAttentionFast(nn.Module):
    """
    Faster dynamic convolution using unfold.
    """
    def __init__(self, config: TestConfig, kernel_size: int = 64):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.dim // config.num_heads
        self.kernel_size = kernel_size
        self.dim = config.dim

        self.v_proj = nn.Linear(config.dim, config.dim)

        # Generate kernel from current position
        self.kernel_gen = nn.Sequential(
            nn.Linear(config.dim, config.dim // 2),
            nn.GELU(),
            nn.Linear(config.dim // 2, config.num_heads * kernel_size)
        )

        self.out = nn.Linear(config.dim, config.dim)

    def forward(self, x):
        B, T, C = x.shape

        v = self.v_proj(x)  # (B, T, C)
        v = v.view(B, T, self.num_heads, self.head_dim)  # (B, T, H, D)

        # Generate position-specific kernels
        kernels = self.kernel_gen(x)  # (B, T, H * K)
        kernels = kernels.view(B, T, self.num_heads, self.kernel_size)
        kernels = F.softmax(kernels, dim=-1)  # (B, T, H, K)

        # Pad for causal: we want position t to see [t-K+1, t]
        v_padded = F.pad(v.permute(0, 2, 3, 1), (self.kernel_size - 1, 0))  # (B, H, D, T+K-1)

        # Unfold to get windows
        # unfold(dim, size, step) -> (B, H, D, T, K)
        v_windows = v_padded.unfold(-1, self.kernel_size, 1)  # (B, H, D, T, K)
        v_windows = v_windows.permute(0, 3, 1, 4, 2)  # (B, T, H, K, D)

        # Apply kernels: (B, T, H, K) @ (B, T, H, K, D) -> (B, T, H, D)
        out = torch.einsum('bthk,bthkd->bthd', kernels, v_windows)
        out = out.reshape(B, T, C)

        return self.out(out)


class GatedLinearAttention(nn.Module):
    """
    Linear attention with gating (GLA-style).

    Uses forget gate to control how much history to retain.
    This allows selective "forgetting" which enables retrieval.
    """
    def __init__(self, config: TestConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.dim // config.num_heads
        self.dim = config.dim

        self.qkv = nn.Linear(config.dim, config.dim * 3)
        self.gate = nn.Linear(config.dim, config.num_heads)  # Per-head gate
        self.out = nn.Linear(config.dim, config.dim)

    def feature_map(self, x):
        return F.elu(x) + 1

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (B, H, T, D)

        # Forget gate per position per head
        gate = torch.sigmoid(self.gate(x))  # (B, T, H)
        gate = gate.permute(0, 2, 1).unsqueeze(-1)  # (B, H, T, 1)

        q = self.feature_map(q)
        k = self.feature_map(k)

        # Gated cumulative sum
        # state[t] = gate[t] * state[t-1] + k[t] outer v[t]
        kv = k.unsqueeze(-1) * v.unsqueeze(-2)  # (B, H, T, D, D)

        # Apply gate with scan (sequential for correctness)
        states = []
        state = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=x.device)

        for t in range(T):
            state = gate[:, :, t] * state + kv[:, :, t]
            states.append(state)

        kv_cumsum = torch.stack(states, dim=2)  # (B, H, T, D, D)

        # Same for normalizer
        k_states = []
        k_state = torch.zeros(B, self.num_heads, self.head_dim, device=x.device)
        for t in range(T):
            k_state = gate[:, :, t].squeeze(-1) * k_state + k[:, :, t]
            k_states.append(k_state)
        k_cumsum = torch.stack(k_states, dim=2)

        out = torch.einsum('bhtd,bhtde->bhte', q, kv_cumsum)
        normalizer = torch.einsum('bhtd,bhtd->bht', q, k_cumsum).unsqueeze(-1)
        out = out / (normalizer + 1e-6)

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class AdditiveLinearAttention(nn.Module):
    """
    Bahdanau-style additive attention made linear.

    score(q, k) = v^T tanh(W_q @ q + W_k @ k)

    With linear kernel trick for O(N).
    """
    def __init__(self, config: TestConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.dim // config.num_heads
        self.dim = config.dim
        self.attn_dim = 64

        self.q_proj = nn.Linear(config.dim, config.dim)
        self.k_proj = nn.Linear(config.dim, config.dim)
        self.v_proj = nn.Linear(config.dim, config.dim)

        self.W_q = nn.Linear(self.head_dim, self.attn_dim, bias=False)
        self.W_k = nn.Linear(self.head_dim, self.attn_dim, bias=False)
        self.score_v = nn.Parameter(torch.randn(config.num_heads, self.attn_dim) * 0.02)

        self.out = nn.Linear(config.dim, config.dim)

    def forward(self, x):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim)

        # Project to attention space
        q_attn = self.W_q(q)  # (B, T, H, A)
        k_attn = self.W_k(k)  # (B, T, H, A)

        # This is still O(N²) as written - need the full additive comparison
        # For O(N), we'd need to restructure, but let's test if additive helps first
        q_attn = q_attn.permute(0, 2, 1, 3)  # (B, H, T, A)
        k_attn = k_attn.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Additive attention (still O(N²) but different scoring)
        scores = torch.tanh(q_attn.unsqueeze(3) + k_attn.unsqueeze(2))  # (B, H, T, T, A)
        scores = torch.einsum('bhtsa,ha->bhts', scores, self.score_v)  # (B, H, T, T)

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)


# --- Model ---

class TransformerBlock(nn.Module):
    def __init__(self, config: TestConfig, attention_class, **kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.dim)
        self.attn = attention_class(config, **kwargs)
        self.norm2 = nn.LayerNorm(config.dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.dim, config.dim * 4),
            nn.GELU(),
            nn.Linear(config.dim * 4, config.dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class AssociativeRecallModel(nn.Module):
    def __init__(self, config: TestConfig, attention_class, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(config, attention_class, **kwargs)
            for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


# --- Data Generation ---

def generate_batch(batch_size, seq_len, vocab_size, device):
    KEY_START, KEY_END = 10, 110
    VAL_START, VAL_END = 110, 210
    NOISE_START = 210

    sequences = []
    targets = []

    for _ in range(batch_size):
        seq = torch.zeros(seq_len, dtype=torch.long, device=device)
        key = torch.randint(KEY_START, KEY_END, (1,), device=device)
        val = torch.randint(VAL_START, VAL_END, (1,), device=device)
        seq[0] = key
        seq[1] = val
        seq[2:-1] = torch.randint(NOISE_START, vocab_size, (seq_len - 3,), device=device)
        seq[-1] = key
        sequences.append(seq)
        targets.append(val)

    return torch.stack(sequences), torch.tensor(targets, device=device).squeeze()


def train_epoch(model, config, seq_len, num_batches=100, batch_size=32, device='cuda'):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    total_correct = 0
    total_samples = 0

    for _ in range(num_batches):
        seqs, targets = generate_batch(batch_size, seq_len, config.vocab_size, device)

        optimizer.zero_grad()
        logits = model(seqs)
        query_logits = logits[:, -1]

        loss = F.cross_entropy(query_logits, targets)
        loss.backward()
        optimizer.step()

        preds = query_logits.argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size

    return total_correct / total_samples


@torch.no_grad()
def evaluate(model, config, seq_len, num_batches=50, batch_size=32, device='cuda'):
    model.eval()
    total_correct = 0
    total_samples = 0

    for _ in range(num_batches):
        seqs, targets = generate_batch(batch_size, seq_len, config.vocab_size, device)
        logits = model(seqs)
        query_logits = logits[:, -1]
        preds = query_logits.argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size

    return total_correct / total_samples


def test_model(name, attention_class, config, seq_lengths, device, epochs=20, **kwargs):
    print(f"\n{'='*60}", flush=True)
    print(f"{name}", flush=True)
    print(f"{'='*60}", flush=True)

    results = {}

    for seq_len in seq_lengths:
        print(f"\n--- Distance: {seq_len - 2} ---", flush=True)

        model = AssociativeRecallModel(config, attention_class, **kwargs).to(device)
        if seq_len == seq_lengths[0]:
            params = sum(p.numel() for p in model.parameters())
            print(f"Parameters: {params:,}", flush=True)

        best_acc = 0
        for epoch in range(1, epochs + 1):
            train_acc = train_epoch(model, config, seq_len, device=device)
            val_acc = evaluate(model, config, seq_len, device=device)

            if val_acc > best_acc:
                best_acc = val_acc

            if epoch % 5 == 0 or val_acc > 0.99:
                print(f"Epoch {epoch}: Train {train_acc:.1%}, Val {val_acc:.1%}", flush=True)

            if val_acc > 0.99:
                print(f"SOLVED!", flush=True)
                break

        results[seq_len] = best_acc

        del model
        torch.cuda.empty_cache()

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Dynamic Convolution Associative Recall Test", flush=True)
    print(f"Testing if data-dependent kernels enable long-range retrieval", flush=True)

    config = TestConfig()
    seq_lengths = [64, 128, 256, 512]

    all_results = {}

    # 1. Standard attention (baseline)
    all_results['Standard'] = test_model(
        "Standard Attention O(N²)", StandardAttention, config, seq_lengths, device
    )

    # 2. Linear attention with kernel trick
    all_results['Linear'] = test_model(
        "Linear Attention (elu+1 kernel)", LinearAttention, config, seq_lengths, device
    )

    # 3. Dynamic conv (Hyena-style)
    all_results['DynConv'] = test_model(
        "Dynamic Conv (data-dependent kernel)", DynamicConvAttentionFast, config, seq_lengths, device,
        kernel_size=64
    )

    # 4. Gated Linear Attention
    all_results['GLA'] = test_model(
        "Gated Linear Attention", GatedLinearAttention, config, seq_lengths, device
    )

    # Summary
    print(f"\n{'='*70}", flush=True)
    print("DYNAMIC ATTENTION RESULTS", flush=True)
    print(f"{'='*70}", flush=True)

    print(f"\n{'Distance':<10}", end="", flush=True)
    for name in all_results.keys():
        print(f"{name:<12}", end="", flush=True)
    print(flush=True)
    print("-" * 60, flush=True)

    for seq_len in seq_lengths:
        distance = seq_len - 2
        print(f"{distance:<10}", end="", flush=True)
        for name, results in all_results.items():
            acc = results.get(seq_len, 0)
            status = "✓" if acc > 0.95 else "✗" if acc < 0.5 else "~"
            print(f"{acc:>5.0%} {status:<5}", end="", flush=True)
        print(flush=True)

    print(f"\n{'='*70}", flush=True)

    # Conclusion
    print("\nCONCLUSION:", flush=True)

    linear_works = all([all_results['Linear'].get(sl, 0) > 0.9 for sl in seq_lengths])
    dynconv_works = all([all_results['DynConv'].get(sl, 0) > 0.9 for sl in seq_lengths])
    gla_works = all([all_results['GLA'].get(sl, 0) > 0.9 for sl in seq_lengths])

    if linear_works:
        print("LINEAR ATTENTION: Kernel trick enables O(N) retrieval!", flush=True)
    else:
        print("LINEAR ATTENTION: Kernel trick alone insufficient", flush=True)

    if dynconv_works:
        print("DYNAMIC CONV: Data-dependent kernels work!", flush=True)
    else:
        print("DYNAMIC CONV: Still fails at long range", flush=True)

    if gla_works:
        print("GATED LINEAR: Forget gates enable selective retrieval!", flush=True)
    else:
        print("GATED LINEAR: Gates alone insufficient", flush=True)


if __name__ == "__main__":
    main()
