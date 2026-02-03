#!/usr/bin/env python3
"""
True O(N) Positional Attention Experiment

Previous experiment removed content-dependent scoring but still had O(N²)
matmul for `attn @ v`. This experiment implements TRUE linear-time attention
using depthwise causal convolution.

Key insight: positional-only attention is mathematically equivalent to
causal convolution with learned kernel weights.

out_t = sum_{j <= t} w_{t-j} * v_j  ->  depthwise conv1d

This removes BOTH:
1. Content-dependent O(N²) scoring
2. Quadratic compute/memory

Now if perplexity stays similar, that's strong evidence that dynamic
similarity is unnecessary at this scale.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from datasets import load_dataset
from transformers import GPT2Tokenizer


class StandardAttention(nn.Module):
    """Standard O(N²) attention for baseline."""
    def __init__(self, dim, num_heads=8, dropout=0.1, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_dropout(self.out(out))


class PositionalOnlyQuadratic(nn.Module):
    """
    Previous "positional only" - still O(N²) due to attn @ v matmul.
    Included for comparison.
    """
    def __init__(self, dim, num_heads=8, dropout=0.1, max_len=128, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.v_proj = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        self.rel_pos_pattern = nn.Parameter(torch.zeros(num_heads, 2 * max_len - 1))
        self.max_len = max_len

    def forward(self, x):
        B, T, C = x.shape
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        positions = torch.arange(T, device=x.device)
        rel_pos = positions.unsqueeze(1) - positions.unsqueeze(0)
        rel_pos_idx = rel_pos + self.max_len - 1
        attn_logits = self.rel_pos_pattern[:, rel_pos_idx]

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn_logits = attn_logits.masked_fill(mask, float('-inf'))

        attn = F.softmax(attn_logits, dim=-1)
        attn = self.attn_dropout(attn)
        attn = attn.unsqueeze(0)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_dropout(self.out(out))


class TrueLinearPositionalAttention(nn.Module):
    """
    True O(N) positional-only attention.

    Implements positional attention as depthwise causal convolution.
    Complexity: O(B * H * T * K) instead of O(B * H * T²)

    K = kernel_size (receptive field)
    """
    def __init__(self, dim, num_heads=8, dropout=0.1, kernel_size=64, **kwargs):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel_size = kernel_size

        self.v_proj = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.out_dropout = nn.Dropout(dropout)

        # Learned causal kernel per head: (H, K)
        self.kernel_logits = nn.Parameter(torch.zeros(num_heads, kernel_size))

    def forward(self, x):
        B, T, C = x.shape

        # Project values
        v = self.v_proj(x)  # (B, T, C)

        # Reshape for depthwise conv: (B*H, head_dim, T)
        v = v.view(B, T, self.num_heads, self.head_dim)
        v = v.permute(0, 2, 3, 1).reshape(B * self.num_heads, self.head_dim, T)

        # Softmax over kernel to mimic attention normalization
        kernel = F.softmax(self.kernel_logits, dim=-1)  # (H, K)

        # Depthwise causal conv: each head has its own kernel
        # We need to apply each head's kernel to corresponding channels
        # Use groups=num_heads for true depthwise

        # Reshape kernel for conv1d: (out_channels, in_channels/groups, kernel_size)
        # For depthwise: out_channels = in_channels = B*H*head_dim, groups = B*H
        # But conv1d groups must divide evenly, so we do it per-sample

        # Simpler approach: unfold + matmul
        # Pad for causal conv
        v_padded = F.pad(v, (self.kernel_size - 1, 0))  # (B*H, D, T+K-1)

        # Unfold to get sliding windows: (B*H, D, T, K)
        v_unfolded = v_padded.unfold(dimension=2, size=self.kernel_size, step=1)

        # Apply kernel per head
        # kernel: (H, K) -> (B*H, 1, K)
        kernel_expanded = kernel.unsqueeze(1).repeat(B, 1, 1)  # (B*H, 1, K)

        # Weighted sum: (B*H, D, T, K) @ (B*H, K, 1) -> (B*H, D, T, 1)
        out = torch.einsum('hdtk,hok->hdto', v_unfolded, kernel_expanded.unsqueeze(-1))
        out = out.squeeze(-1)  # (B*H, D, T)

        # Reshape back: (B, T, C)
        out = out.view(B, self.num_heads, self.head_dim, T)
        out = out.permute(0, 3, 1, 2).reshape(B, T, C)

        return self.out_dropout(self.out(out))


class TrueLinearPositionalAttentionV2(nn.Module):
    """
    True O(N) using conv1d directly.
    Cleaner implementation.
    """
    def __init__(self, dim, num_heads=8, dropout=0.1, kernel_size=64, **kwargs):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel_size = kernel_size
        self.dim = dim

        self.v_proj = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.out_dropout = nn.Dropout(dropout)

        # Learned causal kernel per head: (H, K)
        self.kernel_logits = nn.Parameter(torch.zeros(num_heads, kernel_size))

    def forward(self, x):
        B, T, C = x.shape

        # Project values: (B, T, C)
        v = self.v_proj(x)

        # Reshape to (B, C, T) for conv1d
        v = v.transpose(1, 2)

        # Pad for causal conv
        v_padded = F.pad(v, (self.kernel_size - 1, 0))

        # Get softmax kernel weights
        kernel = F.softmax(self.kernel_logits, dim=-1)  # (H, K)

        # Expand kernel for all head dimensions
        # Each head applies same kernel across its head_dim channels
        # kernel shape for depthwise: (C, 1, K) where each group of head_dim shares weights
        kernel_expanded = kernel.unsqueeze(1).expand(-1, self.head_dim, -1)  # (H, D, K)
        kernel_expanded = kernel_expanded.reshape(self.dim, 1, self.kernel_size)  # (C, 1, K)

        # Depthwise conv: groups=C means each channel has its own filter
        out = F.conv1d(v_padded, kernel_expanded, groups=self.dim)

        # Reshape back: (B, C, T) -> (B, T, C)
        out = out.transpose(1, 2)

        return self.out_dropout(self.out(out))


class FixedDecayAttention(nn.Module):
    """
    Fixed exponential decay kernel (not learned).
    Tests whether learned positional patterns matter or any decay works.
    """
    def __init__(self, dim, num_heads=8, dropout=0.1, kernel_size=64, decay=0.95, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel_size = kernel_size
        self.dim = dim

        self.v_proj = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.out_dropout = nn.Dropout(dropout)

        # Fixed exponential decay kernel (not learned)
        # Most recent token gets highest weight
        decay_weights = torch.tensor([decay ** i for i in range(kernel_size - 1, -1, -1)])
        decay_weights = decay_weights / decay_weights.sum()  # Normalize
        self.register_buffer('kernel', decay_weights)

    def forward(self, x):
        B, T, C = x.shape

        v = self.v_proj(x).transpose(1, 2)  # (B, C, T)
        v_padded = F.pad(v, (self.kernel_size - 1, 0))

        # Expand fixed kernel for all channels
        kernel_expanded = self.kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, K)
        kernel_expanded = kernel_expanded.expand(self.dim, 1, -1)  # (C, 1, K)

        out = F.conv1d(v_padded, kernel_expanded, groups=self.dim)
        out = out.transpose(1, 2)

        return self.out_dropout(self.out(out))


class TransformerBlock(nn.Module):
    def __init__(self, dim, attention_type='standard', num_heads=8, dropout=0.1,
                 kernel_size=64, ffn_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)

        attention_classes = {
            'standard': StandardAttention,
            'pos_quadratic': PositionalOnlyQuadratic,
            'true_linear': TrueLinearPositionalAttentionV2,
            'fixed_decay': FixedDecayAttention,
        }

        self.attn = attention_classes[attention_type](
            dim, num_heads, dropout, kernel_size=kernel_size
        )

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, dim, vocab_size, num_layers=6, attention_type='standard',
                 num_heads=8, dropout=0.1, kernel_size=64, ffn_mult=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, attention_type, num_heads, dropout, kernel_size, ffn_mult)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def load_wikitext(max_length=128):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length, padding="max_length")

    train_data = dataset["train"].map(tokenize, batched=True, remove_columns=["text"])
    val_data = dataset["validation"].map(tokenize, batched=True, remove_columns=["text"])

    train_data.set_format(type="torch", columns=["input_ids"])
    val_data.set_format(type="torch", columns=["input_ids"])

    return train_data, val_data, tokenizer.vocab_size


def train_epoch(model, data, optimizer, scaler, device, batch_size=32):
    model.train()
    total_loss = 0
    num_batches = 0

    indices = torch.randperm(len(data)).tolist()
    for i in range(0, len(indices) - batch_size, batch_size):
        batch_indices = indices[i:i+batch_size]
        batch = torch.stack([data[j]["input_ids"] for j in batch_indices]).to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

    return math.exp(total_loss / num_batches)


@torch.no_grad()
def evaluate(model, data, device, batch_size=32):
    model.eval()
    total_loss = 0
    num_batches = 0

    for i in range(0, len(data) - batch_size, batch_size):
        batch = torch.stack([data[j]["input_ids"] for j in range(i, i+batch_size)]).to(device)
        with torch.amp.autocast('cuda'):
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
        total_loss += loss.item()
        num_batches += 1

    return math.exp(total_loss / num_batches) if num_batches > 0 else float('inf')


@torch.no_grad()
def benchmark_tps(model, device, seq_len=128, batch_size=8, num_runs=100):
    model.eval()
    dummy = torch.randint(0, 50257, (batch_size, seq_len), device=device)

    for _ in range(20):
        with torch.amp.autocast('cuda'):
            _ = model(dummy)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_runs):
        with torch.amp.autocast('cuda'):
            _ = model(dummy)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    return (num_runs * batch_size * seq_len) / elapsed


def train_and_benchmark(name, model, train_data, val_data, device, epochs=5):
    print(f"\n{'='*60}", flush=True)
    print(f"{name}", flush=True)
    print(f"{'='*60}", flush=True)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')

    best_val_ppl = float('inf')
    for epoch in range(1, epochs + 1):
        train_ppl = train_epoch(model, train_data, optimizer, scaler, device)
        val_ppl = evaluate(model, val_data, device)

        marker = " *" if val_ppl < best_val_ppl else ""
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl

        print(f"Epoch {epoch}: Train PPL {train_ppl:7.1f}, Val PPL {val_ppl:7.1f}{marker}", flush=True)

    tps = benchmark_tps(model, device)
    print(f"Throughput: {tps:,.0f} TPS", flush=True)

    return best_val_ppl, tps


def main():
    device = torch.device("cuda")
    print(f"Device: {device}", flush=True)
    print(f"True O(N) Positional Attention Experiment", flush=True)
    print(f"Testing ACTUAL linear-time attention via causal convolution", flush=True)

    train_data, val_data, vocab_size = load_wikitext()
    print(f"Vocab size: {vocab_size}", flush=True)

    results = {}

    dim = 256
    num_layers = 6
    num_heads = 8

    # 1. Standard O(N²) attention
    model = TransformerLM(dim, vocab_size, num_layers, 'standard', num_heads).to(device)
    ppl, tps = train_and_benchmark("Standard QKV O(N²)", model, train_data, val_data, device)
    results['Standard'] = (ppl, tps)
    del model
    torch.cuda.empty_cache()

    # 2. Positional-only but still O(N²) matmul (previous experiment)
    model = TransformerLM(dim, vocab_size, num_layers, 'pos_quadratic', num_heads).to(device)
    ppl, tps = train_and_benchmark("Positional O(N²) (prev)", model, train_data, val_data, device)
    results['Pos-Quad'] = (ppl, tps)
    del model
    torch.cuda.empty_cache()

    # 3. True O(N) with kernel_size=64
    model = TransformerLM(dim, vocab_size, num_layers, 'true_linear', num_heads,
                          kernel_size=64).to(device)
    ppl, tps = train_and_benchmark("True O(N) K=64", model, train_data, val_data, device)
    results['Linear-64'] = (ppl, tps)
    del model
    torch.cuda.empty_cache()

    # 4. True O(N) with kernel_size=128 (full context)
    model = TransformerLM(dim, vocab_size, num_layers, 'true_linear', num_heads,
                          kernel_size=128).to(device)
    ppl, tps = train_and_benchmark("True O(N) K=128", model, train_data, val_data, device)
    results['Linear-128'] = (ppl, tps)
    del model
    torch.cuda.empty_cache()

    # 5. True O(N) with kernel_size=32 (limited context)
    model = TransformerLM(dim, vocab_size, num_layers, 'true_linear', num_heads,
                          kernel_size=32).to(device)
    ppl, tps = train_and_benchmark("True O(N) K=32", model, train_data, val_data, device)
    results['Linear-32'] = (ppl, tps)
    del model
    torch.cuda.empty_cache()

    # 6. Fixed decay (not learned) - tests if learned patterns matter
    model = TransformerLM(dim, vocab_size, num_layers, 'fixed_decay', num_heads,
                          kernel_size=64).to(device)
    ppl, tps = train_and_benchmark("Fixed Decay K=64", model, train_data, val_data, device)
    results['Fixed-64'] = (ppl, tps)
    del model
    torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}", flush=True)
    print("RESULTS SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)

    baseline_ppl, baseline_tps = results['Standard']
    print(f"{'Model':<25} {'PPL':>8} {'PPL Δ':>8} {'TPS':>12} {'TPS Δ':>8} {'Complexity':<10}", flush=True)
    print("-" * 80, flush=True)

    complexity = {
        'Standard': 'O(N²)',
        'Pos-Quad': 'O(N²)',
        'Linear-128': 'O(N)',
        'Linear-64': 'O(N)',
        'Linear-32': 'O(N)',
        'Fixed-64': 'O(N)',
    }

    for name, (ppl, tps) in sorted(results.items(), key=lambda x: x[1][0]):
        ppl_diff = (ppl - baseline_ppl) / baseline_ppl * 100
        tps_diff = (tps - baseline_tps) / baseline_tps * 100
        comp = complexity.get(name, '?')
        print(f"{name:<25} {ppl:8.2f} {ppl_diff:+7.1f}% {tps:>12,.0f} {tps_diff:+7.1f}% {comp:<10}", flush=True)

    print(f"\n{'='*70}", flush=True)

    # Analysis
    print("\nANALYSIS:", flush=True)

    linear_128_ppl = results['Linear-128'][0]
    linear_64_ppl = results['Linear-64'][0]
    linear_32_ppl = results['Linear-32'][0]
    fixed_ppl = results['Fixed-64'][0]

    print(f"\n1. Does TRUE O(N) work?", flush=True)
    linear_vs_std = (linear_128_ppl - baseline_ppl) / baseline_ppl * 100
    if abs(linear_vs_std) < 5:
        print(f"   YES! Linear-128 is only {linear_vs_std:+.1f}% vs Standard", flush=True)
        print(f"   → O(N) attention is viable!", flush=True)
    else:
        print(f"   Degradation: {linear_vs_std:+.1f}% vs Standard", flush=True)

    print(f"\n2. How much context is needed?", flush=True)
    print(f"   K=128: {linear_128_ppl:.2f} PPL", flush=True)
    print(f"   K=64:  {linear_64_ppl:.2f} PPL", flush=True)
    print(f"   K=32:  {linear_32_ppl:.2f} PPL", flush=True)

    print(f"\n3. Does learned kernel matter?", flush=True)
    learned_vs_fixed = (linear_64_ppl - fixed_ppl) / fixed_ppl * 100
    if learned_vs_fixed < -2:
        print(f"   YES! Learned is {-learned_vs_fixed:.1f}% better than fixed decay", flush=True)
    else:
        print(f"   Not much: only {-learned_vs_fixed:.1f}% difference", flush=True)
        print(f"   → Even simple exponential decay works!", flush=True)

    print(f"\n4. Throughput comparison:", flush=True)
    for name in ['Standard', 'Linear-64', 'Linear-128']:
        tps = results[name][1]
        tps_diff = (tps - baseline_tps) / baseline_tps * 100
        print(f"   {name}: {tps:,.0f} TPS ({tps_diff:+.1f}%)", flush=True)


if __name__ == "__main__":
    main()
