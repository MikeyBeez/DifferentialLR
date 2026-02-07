#!/usr/bin/env python3
"""
Selective Gate Associative Recall Test

The key insight: We need the current token to SELECTIVELY RETRIEVE from
the entire accumulated history, not just a local window.

Architecture:
1. Accumulate history via O(N) cumsum: state[t] = sum_{i<=t} f(x[i])
2. Current token generates a GATE that selectively extracts from state
3. Gate is content-dependent: g = sigmoid(W_g @ x[t])
4. Output: y[t] = g * state[t] (element-wise selection)

The trick: We accumulate KEY-VALUE pairs where the key is embedded
in a way that the gate can "match" it.

This is essentially what Linear Attention does, but with explicit gating.
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
    """Linear attention with kernel trick - known to work."""
    def __init__(self, config: TestConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.dim // config.num_heads

        self.qkv = nn.Linear(config.dim, config.dim * 3)
        self.out = nn.Linear(config.dim, config.dim)

    def feature_map(self, x):
        return F.elu(x) + 1

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        q = self.feature_map(q)
        k = self.feature_map(k)

        # Causal linear attention
        kv = k.unsqueeze(-1) * v.unsqueeze(-2)
        kv_cumsum = torch.cumsum(kv, dim=2)
        k_cumsum = torch.cumsum(k, dim=2)

        out = torch.einsum('bhtd,bhtde->bhte', q, kv_cumsum)
        normalizer = torch.einsum('bhtd,bhtd->bht', q, k_cumsum).unsqueeze(-1)
        out = out / (normalizer + 1e-6)

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class SelectiveGateAttention(nn.Module):
    """
    O(N) attention via selective gating.

    Key idea: Accumulate (key, value) pairs into a state matrix.
    Query generates a "selector" that extracts the matching value.

    State: S[t] = sum_{i<=t} outer(k[i], v[i])  -- (D, D) matrix
    Query: q[t] generates selector
    Output: y[t] = q[t] @ S[t]  -- retrieves value associated with matching key

    This is essentially Linear Attention but framed as gated retrieval.
    """
    def __init__(self, config: TestConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.dim // config.num_heads
        self.dim = config.dim

        self.q_proj = nn.Linear(config.dim, config.dim)
        self.k_proj = nn.Linear(config.dim, config.dim)
        self.v_proj = nn.Linear(config.dim, config.dim)
        self.out = nn.Linear(config.dim, config.dim)

    def forward(self, x):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim)

        # Normalize for stable retrieval
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        q = q.permute(0, 2, 1, 3)  # (B, H, T, D)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Build state: S[t] = sum_{i<=t} k[i] outer v[i]
        kv = k.unsqueeze(-1) * v.unsqueeze(-2)  # (B, H, T, D, D)
        state = torch.cumsum(kv, dim=2)  # Causal accumulation

        # Retrieve: y[t] = q[t] @ S[t]
        out = torch.einsum('bhtd,bhtde->bhte', q, state)

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class GatedStateAttention(nn.Module):
    """
    O(N) with explicit forget gate.

    The gate controls what to keep vs forget from accumulated state.
    This allows the model to "clear" old information and maintain
    only what's relevant.

    forget_gate: How much to decay previous state
    input_gate: How much of new (k,v) pair to add
    """
    def __init__(self, config: TestConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.dim // config.num_heads
        self.dim = config.dim

        self.q_proj = nn.Linear(config.dim, config.dim)
        self.k_proj = nn.Linear(config.dim, config.dim)
        self.v_proj = nn.Linear(config.dim, config.dim)

        # Forget gate - learned per position based on input
        self.forget_gate = nn.Linear(config.dim, config.num_heads)

        self.out = nn.Linear(config.dim, config.dim)

    def forward(self, x):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim)

        # Forget gate per position per head
        fg = torch.sigmoid(self.forget_gate(x))  # (B, T, H)

        q = q.permute(0, 2, 1, 3)  # (B, H, T, D)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        fg = fg.permute(0, 2, 1)  # (B, H, T)

        # Normalize
        q = F.elu(q) + 1
        k = F.elu(k) + 1

        # Gated state accumulation (sequential for correctness)
        kv = k.unsqueeze(-1) * v.unsqueeze(-2)  # (B, H, T, D, D)

        states = []
        state = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=x.device)

        for t in range(T):
            # Decay previous state, add new kv
            state = fg[:, :, t:t+1, None] * state + kv[:, :, t]
            states.append(state.clone())

        state_stack = torch.stack(states, dim=2)  # (B, H, T, D, D)

        # Similarly for normalizer
        k_states = []
        k_state = torch.zeros(B, self.num_heads, self.head_dim, device=x.device)
        for t in range(T):
            k_state = fg[:, :, t:t+1] * k_state + k[:, :, t]
            k_states.append(k_state.clone())
        k_stack = torch.stack(k_states, dim=2)

        # Retrieve
        out = torch.einsum('bhtd,bhtde->bhte', q, state_stack)
        normalizer = torch.einsum('bhtd,bhtd->bht', q, k_stack).unsqueeze(-1)
        out = out / (normalizer + 1e-6)

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class HyenaStyleAttention(nn.Module):
    """
    Hyena-inspired: Long convolution + data-dependent gating.

    1. Project to get "content" and "position" features
    2. Apply learned positional mixing (global conv via FFT)
    3. Gate output based on query

    The FFT conv gives O(N log N) global mixing.
    The gating provides content-dependent selection.
    """
    def __init__(self, config: TestConfig, order: int = 2):
        super().__init__()
        self.dim = config.dim
        self.order = order  # Number of Hyena filters

        # Projections for each order
        self.in_proj = nn.Linear(config.dim, config.dim * (order + 1))

        # Learned positional filters (will be applied via FFT)
        # We learn the filter in frequency domain for efficiency
        self.filter_params = nn.ParameterList([
            nn.Parameter(torch.randn(config.dim, 1024) * 0.02)  # Max seq len 1024
            for _ in range(order)
        ])

        self.out = nn.Linear(config.dim, config.dim)

    def forward(self, x):
        B, T, C = x.shape

        # Project to get v, x1, x2, ... (order+1 projections)
        projs = self.in_proj(x).chunk(self.order + 1, dim=-1)
        v = projs[0]
        gates = projs[1:]

        # Apply Hyena recurrence
        y = v
        for i in range(self.order):
            # Long conv via FFT
            filter_freq = self.filter_params[i][:, :T]

            # Causal filter (zero out future)
            filter_time = filter_freq  # Simplified - should be proper FFT
            filter_time = F.softmax(filter_time, dim=-1)  # Normalize

            # Apply as depthwise conv
            y_t = y.transpose(1, 2)  # (B, C, T)
            y_padded = F.pad(y_t, (T - 1, 0))

            # Depthwise conv with learned filter
            weight = filter_time.unsqueeze(1)  # (C, 1, T)
            y_conv = F.conv1d(y_padded, weight, groups=C)
            y = y_conv.transpose(1, 2)  # (B, T, C)

            # Element-wise gate
            y = y * gates[i]

        return self.out(y)


class RetentiveAttention(nn.Module):
    """
    Retention-style (from RetNet): Decay-based recurrence.

    Each head has a learned decay rate gamma.
    State decays exponentially: S[t] = gamma * S[t-1] + k[t] @ v[t]^T

    This naturally prioritizes recent tokens but can still retrieve
    if the decay is slow enough.
    """
    def __init__(self, config: TestConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.dim // config.num_heads
        self.dim = config.dim

        self.qkv = nn.Linear(config.dim, config.dim * 3)

        # Learned decay per head (initialized to slow decay)
        self.gamma_logit = nn.Parameter(torch.ones(config.num_heads) * 2)

        self.out = nn.Linear(config.dim, config.dim)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (B, H, T, D)

        # Decay rate per head
        gamma = torch.sigmoid(self.gamma_logit)  # (H,) in [0, 1]

        # Build decaying state
        kv = k.unsqueeze(-1) * v.unsqueeze(-2)  # (B, H, T, D, D)

        # Sequential scan with decay
        states = []
        state = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=x.device)

        for t in range(T):
            state = gamma.view(1, -1, 1, 1) * state + kv[:, :, t]
            states.append(state.clone())

        state_stack = torch.stack(states, dim=2)

        # Same for normalizer
        k_states = []
        k_state = torch.zeros(B, self.num_heads, self.head_dim, device=x.device)
        for t in range(T):
            k_state = gamma.view(1, -1, 1) * k_state + k[:, :, t]
            k_states.append(k_state.clone())
        k_stack = torch.stack(k_states, dim=2)

        # Retrieve
        out = torch.einsum('bhtd,bhtde->bhte', q, state_stack)
        normalizer = torch.einsum('bhtd,bhtd->bht', q, k_stack).unsqueeze(-1)
        out = out / (normalizer + 1e-6)

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out(out)


# --- Model ---

class TransformerBlock(nn.Module):
    def __init__(self, config: TestConfig, attention_class):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.dim)
        self.attn = attention_class(config)
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


class TestModel(nn.Module):
    def __init__(self, config: TestConfig, attention_class):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(config, attention_class)
            for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


# --- Data & Training ---

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
        preds = logits[:, -1].argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size

    return total_correct / total_samples


def test_model(name, attention_class, config, seq_lengths, device, epochs=20):
    print(f"\n{'='*60}", flush=True)
    print(f"{name}", flush=True)
    print(f"{'='*60}", flush=True)

    results = {}

    for seq_len in seq_lengths:
        print(f"\n--- Distance: {seq_len - 2} ---", flush=True)

        model = TestModel(config, attention_class).to(device)
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
    print(f"Selective Gate Associative Recall Test", flush=True)
    print(f"Testing O(N) architectures with content-dependent selection", flush=True)

    config = TestConfig()
    seq_lengths = [64, 128, 256, 512]

    all_results = {}

    # 1. Standard attention (baseline)
    all_results['Standard'] = test_model(
        "Standard Attention O(N²)", StandardAttention, config, seq_lengths, device
    )

    # 2. Linear attention (known to work)
    all_results['Linear'] = test_model(
        "Linear Attention", LinearAttention, config, seq_lengths, device
    )

    # 3. Selective gate
    all_results['SelectGate'] = test_model(
        "Selective Gate", SelectiveGateAttention, config, seq_lengths, device
    )

    # 4. Gated state (with forget gate)
    all_results['GatedState'] = test_model(
        "Gated State (forget gate)", GatedStateAttention, config, seq_lengths, device
    )

    # 5. Retentive (decay-based)
    all_results['Retentive'] = test_model(
        "Retentive (decay)", RetentiveAttention, config, seq_lengths, device
    )

    # Summary
    print(f"\n{'='*70}", flush=True)
    print("SELECTIVE GATE RESULTS", flush=True)
    print(f"{'='*70}", flush=True)

    print(f"\n{'Distance':<10}", end="", flush=True)
    for name in all_results.keys():
        print(f"{name:<12}", end="", flush=True)
    print(flush=True)
    print("-" * 70, flush=True)

    for seq_len in seq_lengths:
        distance = seq_len - 2
        print(f"{distance:<10}", end="", flush=True)
        for name, results in all_results.items():
            acc = results.get(seq_len, 0)
            status = "✓" if acc > 0.95 else "✗" if acc < 0.5 else "~"
            print(f"{acc:>5.0%} {status:<5}", end="", flush=True)
        print(flush=True)

    print(f"\n{'='*70}", flush=True)

    # Analysis
    print("\nANALYSIS:", flush=True)
    for name, results in all_results.items():
        if name == 'Standard':
            continue
        successes = [sl for sl, acc in results.items() if acc > 0.9]
        if len(successes) == len(seq_lengths):
            print(f"{name}: PASSES all distances!", flush=True)
        elif successes:
            print(f"{name}: Works up to distance {max(successes) - 2}", flush=True)
        else:
            print(f"{name}: FAILS", flush=True)


if __name__ == "__main__":
    main()
