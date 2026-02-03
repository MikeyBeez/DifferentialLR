#!/usr/bin/env python3
"""
Asymmetry Hypothesis Test

Hypothesis: Q=K=V degrades because it forces symmetric attention scores.
With Q=K, score(i,j) = h_i·(W·W^T)·h_j^T, and W·W^T is always symmetric.

Key test: Q=K with V separate
- If it degrades ~7%, the problem is symmetric scoring
- If it works fine, the problem was V sharing

Also measures attention symmetry in trained models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from datasets import load_dataset
from transformers import GPT2Tokenizer


class StandardAttention(nn.Module):
    """Standard attention with separate Q, K, V projections."""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        self.last_attn_weights = None  # For symmetry measurement

    def forward(self, x, store_attn=False):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        if store_attn:
            self.last_attn_weights = attn.detach()

        attn = self.attn_dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_dropout(self.out(out))


class QEqualsKAttention(nn.Module):
    """Q = K (shared projection), V separate. Tests symmetric scoring."""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q and K share one projection, V is separate
        self.qk = nn.Linear(dim, dim)  # Shared for Q and K
        self.v = nn.Linear(dim, dim)   # Separate for V
        self.out = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        self.last_attn_weights = None

    def forward(self, x, store_attn=False):
        B, T, C = x.shape

        # Q = K = x @ W_qk
        qk = self.qk(x).reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Score is symmetric: (h·W)·(h·W)^T = h·(W·W^T)·h^T
        attn = (qk @ qk.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        if store_attn:
            self.last_attn_weights = attn.detach()

        attn = self.attn_dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_dropout(self.out(out))


class QEqualsKEqualsVAttention(nn.Module):
    """Q = K = V (single projection for all)."""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.proj = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        self.last_attn_weights = None

    def forward(self, x, store_attn=False):
        B, T, C = x.shape
        p = self.proj(x).reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (p @ p.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        if store_attn:
            self.last_attn_weights = attn.detach()

        attn = self.attn_dropout(attn)
        out = (attn @ p).transpose(1, 2).reshape(B, T, C)
        return self.out_dropout(self.out(out))


class AsymmetricSingleProjAttention(nn.Module):
    """
    Single projection with forced asymmetry via positional bias.
    Q = K = h @ W, but add learned relative position bias to break symmetry.
    """
    def __init__(self, dim, num_heads=8, dropout=0.1, max_len=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.proj = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        # Learned relative position bias per head
        # bias[head, relative_pos] where relative_pos = i - j
        self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, 2 * max_len - 1))
        self.max_len = max_len

        self.last_attn_weights = None

    def forward(self, x, store_attn=False):
        B, T, C = x.shape
        p = self.proj(x).reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (p @ p.transpose(-2, -1)) * self.scale

        # Add relative position bias to break symmetry
        # For position (i, j), bias index is (i - j) + max_len - 1
        positions = torch.arange(T, device=x.device)
        rel_pos = positions.unsqueeze(1) - positions.unsqueeze(0)  # (T, T)
        rel_pos_idx = rel_pos + self.max_len - 1
        bias = self.rel_pos_bias[:, rel_pos_idx]  # (H, T, T)
        attn = attn + bias.unsqueeze(0)  # (B, H, T, T)

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        if store_attn:
            self.last_attn_weights = attn.detach()

        attn = self.attn_dropout(attn)
        out = (attn @ p).transpose(1, 2).reshape(B, T, C)
        return self.out_dropout(self.out(out))


class TransformerBlock(nn.Module):
    def __init__(self, dim, attention_type='standard', num_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)

        attention_classes = {
            'standard': StandardAttention,
            'q_equals_k': QEqualsKAttention,
            'q_equals_k_equals_v': QEqualsKEqualsVAttention,
            'asymmetric_single': AsymmetricSingleProjAttention,
        }

        self.attn = attention_classes[attention_type](dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, store_attn=False):
        x = x + self.attn(self.norm1(x), store_attn=store_attn)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, dim, vocab_size, num_layers=6, attention_type='standard',
                 num_heads=8, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, attention_type, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x, store_attn=False):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x, store_attn=store_attn)
        return self.head(self.norm(x))

    def get_attention_weights(self):
        """Get stored attention weights from all layers."""
        return [block.attn.last_attn_weights for block in self.blocks]


def measure_attention_symmetry(model, data, device, num_batches=50, batch_size=16):
    """
    Measure how asymmetric the attention patterns are.

    For causal attention, we can only measure symmetry in the lower triangle.
    Computes mean |A[i,j] - A[j,i]| for i > j (valid causal positions).

    Higher = more asymmetric (which should correlate with better performance).
    """
    model.eval()

    all_asymmetry = []

    with torch.no_grad():
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            if start >= len(data):
                break

            batch = torch.stack([
                data[i]["input_ids"] for i in range(start, min(start + batch_size, len(data)))
            ]).to(device)

            # Forward pass storing attention weights
            with torch.amp.autocast('cuda'):
                _ = model(batch[:, :-1], store_attn=True)

            # Get attention weights from all layers
            attn_weights = model.get_attention_weights()

            for layer_attn in attn_weights:
                if layer_attn is None:
                    continue
                # layer_attn: (B, H, T, T)
                B, H, T, _ = layer_attn.shape

                # Compute |A[i,j] - A[j,i]| for valid causal pairs (i > j)
                # After softmax with causal mask, A[i,j] = 0 for i < j
                # So we compare A[i,j] with A[j,i] where both are in lower triangle

                # Create mask for valid comparison pairs (both i>j and j>i visible)
                # This is actually the strictly lower triangle excluding the last row/col
                # But since causal mask zeros out upper triangle, A[j,i] = 0 when j < i
                # So asymmetry = |A[i,j] - 0| = A[i,j] for i > j, which isn't useful

                # Instead, let's measure: for positions where A[i,j] > 0 and A[j,i] > 0,
                # how different are they? This only applies to the lower triangle.

                # Actually, better metric: how much does each position attend backwards
                # vs how much it's attended to from later positions?
                # sum_j A[i,j] vs sum_j A[j,i] for each i

                # Simpler: just measure how non-uniform the attention is
                # For symmetric scoring, nearby positions should have similar attention

                # Best metric: correlation between A[i,j] and A[j,i] across all valid pairs
                # For Q=K, scores should be symmetric, but softmax might break this

                # Let's compute raw score symmetry before softmax would be ideal,
                # but we don't have access. Instead measure attention concentration.

                # Metric: entropy of attention distribution (higher = more spread out)
                # Asymmetric models might show different patterns

                # Actually, simplest useful metric:
                # Take lower triangle, compare A[i,j] vs A[j,i] where both > threshold
                lower = torch.tril(layer_attn, diagonal=-1)
                upper_t = torch.tril(layer_attn.transpose(-1, -2), diagonal=-1)

                # Both should be looking at same positions now
                diff = (lower - upper_t).abs()
                # Mean over positions where at least one is non-negligible
                mask = (lower > 0.01) | (upper_t > 0.01)
                if mask.sum() > 0:
                    asym = diff[mask].mean().item()
                    all_asymmetry.append(asym)

    return sum(all_asymmetry) / len(all_asymmetry) if all_asymmetry else 0.0


def measure_attention_entropy(model, data, device, num_batches=50, batch_size=16):
    """
    Measure attention entropy (how spread out attention is).
    Lower entropy = more concentrated attention.
    """
    model.eval()

    all_entropy = []

    with torch.no_grad():
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            if start >= len(data):
                break

            batch = torch.stack([
                data[i]["input_ids"] for i in range(start, min(start + batch_size, len(data)))
            ]).to(device)

            with torch.amp.autocast('cuda'):
                _ = model(batch[:, :-1], store_attn=True)

            attn_weights = model.get_attention_weights()

            for layer_attn in attn_weights:
                if layer_attn is None:
                    continue
                # Compute entropy: -sum(p * log(p))
                # Add small epsilon to avoid log(0)
                eps = 1e-10
                entropy = -(layer_attn * (layer_attn + eps).log()).sum(dim=-1)
                all_entropy.append(entropy.mean().item())

    return sum(all_entropy) / len(all_entropy) if all_entropy else 0.0


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

    # Measure attention properties
    print("Measuring attention patterns...", flush=True)
    asymmetry = measure_attention_symmetry(model, val_data, device)
    entropy = measure_attention_entropy(model, val_data, device)
    print(f"Attention asymmetry: {asymmetry:.4f}", flush=True)
    print(f"Attention entropy: {entropy:.4f}", flush=True)

    tps = benchmark_tps(model, device)
    print(f"Throughput: {tps:,.0f} TPS", flush=True)

    return best_val_ppl, tps, asymmetry, entropy


def main():
    device = torch.device("cuda")
    print(f"Device: {device}", flush=True)
    print(f"Asymmetry Hypothesis Test", flush=True)
    print(f"Does Q=K degrade because it forces symmetric attention?", flush=True)

    train_data, val_data, vocab_size = load_wikitext()
    print(f"Vocab size: {vocab_size}", flush=True)

    results = {}

    dim = 256
    num_layers = 6
    num_heads = 8

    # 1. Baseline: Standard attention
    model = TransformerLM(dim, vocab_size, num_layers, 'standard', num_heads).to(device)
    ppl, tps, asym, ent = train_and_benchmark(
        "Standard (Q, K, V separate)", model, train_data, val_data, device
    )
    results['Standard'] = (ppl, tps, asym, ent)
    del model
    torch.cuda.empty_cache()

    # 2. KEY TEST: Q = K, but V separate
    model = TransformerLM(dim, vocab_size, num_layers, 'q_equals_k', num_heads).to(device)
    ppl, tps, asym, ent = train_and_benchmark(
        "Q = K, V separate (symmetric scoring)", model, train_data, val_data, device
    )
    results['Q=K, V sep'] = (ppl, tps, asym, ent)
    del model
    torch.cuda.empty_cache()

    # 3. Q = K = V (previous result, for comparison)
    model = TransformerLM(dim, vocab_size, num_layers, 'q_equals_k_equals_v', num_heads).to(device)
    ppl, tps, asym, ent = train_and_benchmark(
        "Q = K = V (all shared)", model, train_data, val_data, device
    )
    results['Q=K=V'] = (ppl, tps, asym, ent)
    del model
    torch.cuda.empty_cache()

    # 4. Single projection with forced asymmetry via relative position bias
    model = TransformerLM(dim, vocab_size, num_layers, 'asymmetric_single', num_heads).to(device)
    ppl, tps, asym, ent = train_and_benchmark(
        "Single proj + rel pos bias (forced asymmetry)", model, train_data, val_data, device
    )
    results['Single+RelPos'] = (ppl, tps, asym, ent)
    del model
    torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}", flush=True)
    print("RESULTS SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)

    baseline_ppl, baseline_tps, _, _ = results['Standard']
    print(f"{'Model':<25} {'PPL':>8} {'PPL Δ':>8} {'Asymm':>8} {'Entropy':>8}", flush=True)
    print("-" * 70, flush=True)

    for name, (ppl, tps, asym, ent) in sorted(results.items(), key=lambda x: x[1][0]):
        ppl_diff = (ppl - baseline_ppl) / baseline_ppl * 100
        print(f"{name:<25} {ppl:8.2f} {ppl_diff:+7.1f}% {asym:8.4f} {ent:8.2f}", flush=True)

    print(f"\n{'='*70}", flush=True)

    # Analysis
    print("\nHYPOTHESIS TEST:", flush=True)
    qk_ppl = results['Q=K, V sep'][0]
    qkv_ppl = results['Q=K=V'][0]
    standard_ppl = results['Standard'][0]

    qk_deg = (qk_ppl - standard_ppl) / standard_ppl * 100
    qkv_deg = (qkv_ppl - standard_ppl) / standard_ppl * 100

    print(f"Q=K with V separate: {qk_deg:+.1f}% vs baseline", flush=True)
    print(f"Q=K=V (all shared):  {qkv_deg:+.1f}% vs baseline", flush=True)

    if qk_deg > 3:  # Significant degradation
        print("\n→ Q=K degrades even with V separate!", flush=True)
        print("→ HYPOTHESIS SUPPORTED: Symmetric scoring hurts performance", flush=True)
    else:
        print("\n→ Q=K works fine when V is separate", flush=True)
        print("→ HYPOTHESIS REFUTED: Problem was V sharing, not symmetry", flush=True)

    # Check if forced asymmetry helps
    relpos_ppl = results['Single+RelPos'][0]
    relpos_deg = (relpos_ppl - standard_ppl) / standard_ppl * 100
    print(f"\nSingle proj + rel pos bias: {relpos_deg:+.1f}% vs baseline", flush=True)
    if relpos_deg < qkv_deg - 2:
        print("→ Forced asymmetry helps recover some performance!", flush=True)
    else:
        print("→ Forced asymmetry doesn't help much", flush=True)


if __name__ == "__main__":
    main()
