#!/usr/bin/env python3
"""
Positional-Only Attention Experiment

Question: How much work is the content-dependent score (p @ p.T) doing
vs the positional bias?

Previous finding: Single Projection + Relative Position Bias beats
standard QKV attention (8.16 vs 8.30 PPL).

Hypothesis: The MLP after attention can handle content-dependent mixing.
Attention only needs to route based on position. Content scores may be
redundant.

Variants tested:
1. Standard QKV - baseline
2. Single+RelPos - previous best (8.16 PPL)
3. Positional Only - no content scores, just learned positional patterns
4. Gated Positional - positional + per-position content gate (O(N))
5. Conv Mixing - no attention at all, just causal convolution
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
    def __init__(self, dim, num_heads=8, dropout=0.1, max_len=128):
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


class SingleProjRelPosAttention(nn.Module):
    """Previous best: single projection + relative position bias."""
    def __init__(self, dim, num_heads=8, dropout=0.1, max_len=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.proj = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, 2 * max_len - 1))
        self.max_len = max_len

    def forward(self, x):
        B, T, C = x.shape
        p = self.proj(x).reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (p @ p.transpose(-2, -1)) * self.scale

        positions = torch.arange(T, device=x.device)
        rel_pos = positions.unsqueeze(1) - positions.unsqueeze(0)
        rel_pos_idx = rel_pos + self.max_len - 1
        bias = self.rel_pos_bias[:, rel_pos_idx]
        attn = attn + bias.unsqueeze(0)

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ p).transpose(1, 2).reshape(B, T, C)
        return self.out_dropout(self.out(out))


class PositionalOnlyAttention(nn.Module):
    """
    Pure positional attention - no content scores at all.

    Attention pattern is purely positional (learned but input-independent).
    Only V projection remains. O(N) weighted sum instead of O(N²) scores.
    """
    def __init__(self, dim, num_heads=8, dropout=0.1, max_len=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Value projection only
        self.v_proj = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        # Learned attention pattern per head, based on relative position
        self.rel_pos_pattern = nn.Parameter(torch.zeros(num_heads, 2 * max_len - 1))
        self.max_len = max_len

    def forward(self, x):
        B, T, C = x.shape

        # Project values
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention is purely positional - same pattern for all inputs
        positions = torch.arange(T, device=x.device)
        rel_pos = positions.unsqueeze(1) - positions.unsqueeze(0)
        rel_pos_idx = rel_pos + self.max_len - 1
        attn_logits = self.rel_pos_pattern[:, rel_pos_idx]  # (H, T, T)

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn_logits = attn_logits.masked_fill(mask, float('-inf'))

        attn = F.softmax(attn_logits, dim=-1)  # (H, T, T)
        attn = self.attn_dropout(attn)
        attn = attn.unsqueeze(0)  # (1, H, T, T) - broadcast over batch

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_dropout(self.out(out))


class GatedPositionalAttention(nn.Module):
    """
    Positional attention + per-position content gate.

    The gate provides minimal content dependence (O(N), not O(N²)).
    Gate modulates how much each position participates in the output.
    """
    def __init__(self, dim, num_heads=8, dropout=0.1, max_len=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.v_proj = nn.Linear(dim, dim)
        self.gate_proj = nn.Linear(dim, num_heads)  # Per-position gate
        self.out = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        self.rel_pos_pattern = nn.Parameter(torch.zeros(num_heads, 2 * max_len - 1))
        self.max_len = max_len

    def forward(self, x):
        B, T, C = x.shape

        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Content-dependent gate per position (NOT per pair)
        gate = torch.sigmoid(self.gate_proj(x))  # (B, T, H)
        gate = gate.permute(0, 2, 1).unsqueeze(-1)  # (B, H, T, 1)

        # Positional attention pattern
        positions = torch.arange(T, device=x.device)
        rel_pos = positions.unsqueeze(1) - positions.unsqueeze(0)
        rel_pos_idx = rel_pos + self.max_len - 1
        attn_logits = self.rel_pos_pattern[:, rel_pos_idx]

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn_logits = attn_logits.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn_logits, dim=-1)
        attn = self.attn_dropout(attn)
        attn = attn.unsqueeze(0)

        # Gate modulates how much each position contributes
        v_gated = v * gate
        out = (attn @ v_gated).transpose(1, 2).reshape(B, T, C)
        return self.out_dropout(self.out(out))


class ConvMixing(nn.Module):
    """
    Replace attention with causal convolution.
    No attention at all - just local mixing.
    """
    def __init__(self, dim, num_heads=8, dropout=0.1, max_len=128, kernel_size=16):
        super().__init__()
        self.kernel_size = kernel_size
        # Depthwise separable convolution
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size-1, groups=dim)
        self.out = nn.Linear(dim, dim)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, C)
        x = x.transpose(1, 2)  # (B, C, T)
        x = self.conv(x)[:, :, :-(self.kernel_size-1)]  # Causal: remove future
        x = x.transpose(1, 2)  # (B, T, C)
        return self.out_dropout(self.out(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim, attention_type='standard', num_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)

        attention_classes = {
            'standard': StandardAttention,
            'single_relpos': SingleProjRelPosAttention,
            'positional_only': PositionalOnlyAttention,
            'gated_positional': GatedPositionalAttention,
            'conv': ConvMixing,
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

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
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
    print(f"Positional-Only Attention Experiment", flush=True)
    print(f"Can we eliminate content-dependent scores entirely?", flush=True)

    train_data, val_data, vocab_size = load_wikitext()
    print(f"Vocab size: {vocab_size}", flush=True)

    results = {}

    dim = 256
    num_layers = 6
    num_heads = 8

    # 1. Standard QKV attention (baseline)
    model = TransformerLM(dim, vocab_size, num_layers, 'standard', num_heads).to(device)
    ppl, tps = train_and_benchmark("Standard QKV", model, train_data, val_data, device)
    results['Standard'] = (ppl, tps)
    del model
    torch.cuda.empty_cache()

    # 2. Single projection + relative position bias (previous best)
    model = TransformerLM(dim, vocab_size, num_layers, 'single_relpos', num_heads).to(device)
    ppl, tps = train_and_benchmark("Single+RelPos (prev best)", model, train_data, val_data, device)
    results['Single+RelPos'] = (ppl, tps)
    del model
    torch.cuda.empty_cache()

    # 3. Pure positional attention (no content scores)
    model = TransformerLM(dim, vocab_size, num_layers, 'positional_only', num_heads).to(device)
    ppl, tps = train_and_benchmark("Positional Only (no content)", model, train_data, val_data, device)
    results['Positional'] = (ppl, tps)
    del model
    torch.cuda.empty_cache()

    # 4. Gated positional (positional + O(N) content gate)
    model = TransformerLM(dim, vocab_size, num_layers, 'gated_positional', num_heads).to(device)
    ppl, tps = train_and_benchmark("Gated Positional (O(N) gate)", model, train_data, val_data, device)
    results['Gated'] = (ppl, tps)
    del model
    torch.cuda.empty_cache()

    # 5. Conv mixing (no attention at all)
    model = TransformerLM(dim, vocab_size, num_layers, 'conv', num_heads).to(device)
    ppl, tps = train_and_benchmark("Conv Mixing (local only)", model, train_data, val_data, device)
    results['Conv'] = (ppl, tps)
    del model
    torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}", flush=True)
    print("RESULTS SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)

    baseline_ppl, baseline_tps = results['Standard']
    print(f"{'Model':<30} {'PPL':>8} {'PPL Δ':>8} {'TPS':>12} {'TPS Δ':>8}", flush=True)
    print("-" * 70, flush=True)

    for name, (ppl, tps) in sorted(results.items(), key=lambda x: x[1][0]):
        ppl_diff = (ppl - baseline_ppl) / baseline_ppl * 100
        tps_diff = (tps - baseline_tps) / baseline_tps * 100
        print(f"{name:<30} {ppl:8.2f} {ppl_diff:+7.1f}% {tps:>12,.0f} {tps_diff:+7.1f}%", flush=True)

    print(f"\n{'='*70}", flush=True)

    # Analysis
    print("\nANALYSIS:", flush=True)

    pos_ppl = results['Positional'][0]
    single_ppl = results['Single+RelPos'][0]
    gated_ppl = results['Gated'][0]
    conv_ppl = results['Conv'][0]

    pos_vs_single = (pos_ppl - single_ppl) / single_ppl * 100
    gated_vs_single = (gated_ppl - single_ppl) / single_ppl * 100
    conv_vs_baseline = (conv_ppl - baseline_ppl) / baseline_ppl * 100

    print(f"\n1. Does Positional Only work?", flush=True)
    if pos_vs_single < 5:
        print(f"   YES! Only {pos_vs_single:+.1f}% vs Single+RelPos", flush=True)
        print(f"   → Content scores may be expendable!", flush=True)
    else:
        print(f"   Partial: {pos_vs_single:+.1f}% vs Single+RelPos", flush=True)
        print(f"   → Content dependence matters", flush=True)

    print(f"\n2. Does the gate help?", flush=True)
    gate_improvement = pos_ppl - gated_ppl
    if gate_improvement > 0.1:
        print(f"   YES! Gated is {gate_improvement:.2f} PPL better than Positional", flush=True)
        print(f"   → Minimal content signal helps, but O(N) is enough", flush=True)
    else:
        print(f"   Not much: only {gate_improvement:.2f} PPL difference", flush=True)

    print(f"\n3. Is attention even necessary?", flush=True)
    if conv_vs_baseline < 10:
        print(f"   Conv is only {conv_vs_baseline:+.1f}% worse than attention", flush=True)
        print(f"   → Local mixing might suffice!", flush=True)
    else:
        print(f"   Conv is {conv_vs_baseline:+.1f}% worse", flush=True)
        print(f"   → Position-dependent global routing matters", flush=True)


if __name__ == "__main__":
    main()
