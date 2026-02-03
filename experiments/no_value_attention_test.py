#!/usr/bin/env python3
"""
Value Projection Elimination Experiment

If the comparison function doesn't matter, maybe we can simplify further.
Can we eliminate the value projection entirely?

Variants:
1. Standard attention (Q, K, V all separate) - baseline
2. V = K - reuse K as values (saves 1/3 of projection params)
3. V = X - no value projection, use raw input
4. V = Q - reuse Q as values
5. Only Q and K, concat weighted keys - aggregate K directly
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


class VEqualsKAttention(nn.Module):
    """V = K: Reuse K as values. Saves W_V projection."""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Only Q and K projections (2/3 of standard)
        self.qk = nn.Linear(dim, dim * 2)
        self.out = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qk = self.qk(x).reshape(B, T, 2, self.num_heads, self.head_dim)
        q, k = qk.permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # V = K
        out = (attn @ k).transpose(1, 2).reshape(B, T, C)
        return self.out_dropout(self.out(out))


class VEqualsXAttention(nn.Module):
    """V = X: No value projection, use raw input."""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Only Q and K projections
        self.qk = nn.Linear(dim, dim * 2)
        self.out = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qk = self.qk(x).reshape(B, T, 2, self.num_heads, self.head_dim)
        q, k = qk.permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # V = X (reshaped to heads)
        v = x.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_dropout(self.out(out))


class VEqualsQAttention(nn.Module):
    """V = Q: Reuse Q as values."""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Only Q and K projections
        self.qk = nn.Linear(dim, dim * 2)
        self.out = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qk = self.qk(x).reshape(B, T, 2, self.num_heads, self.head_dim)
        q, k = qk.permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # V = Q (at each position, we look back at other Q's)
        out = (attn @ q).transpose(1, 2).reshape(B, T, C)
        return self.out_dropout(self.out(out))


class SingleProjectionAttention(nn.Module):
    """
    Just one projection matrix. X → P, then use P for Q, K, and V.
    This is the extreme case - can one projection do it all?
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Single projection
        self.proj = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        p = self.proj(x).reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q = K = V = P
        attn = (p @ p.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ p).transpose(1, 2).reshape(B, T, C)
        return self.out_dropout(self.out(out))


class TransformerBlock(nn.Module):
    def __init__(self, dim, attention_type='standard', num_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)

        attention_classes = {
            'standard': StandardAttention,
            'v_equals_k': VEqualsKAttention,
            'v_equals_x': VEqualsXAttention,
            'v_equals_q': VEqualsQAttention,
            'single_proj': SingleProjectionAttention,
        }

        if attention_type not in attention_classes:
            raise ValueError(f"Unknown attention type: {attention_type}")

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
    print(f"Value Projection Elimination Experiment", flush=True)
    print(f"Can we remove W_V and still get good results?", flush=True)

    train_data, val_data, vocab_size = load_wikitext()
    print(f"Vocab size: {vocab_size}", flush=True)

    results = {}

    # Config
    dim = 256
    num_layers = 6
    num_heads = 8

    # 1. Baseline: Standard attention with Q, K, V
    model = TransformerLM(dim, vocab_size, num_layers, 'standard', num_heads).to(device)
    ppl, tps = train_and_benchmark("Standard (Q, K, V separate)", model, train_data, val_data, device)
    results['Standard'] = (ppl, tps)
    del model
    torch.cuda.empty_cache()

    # 2. V = K
    model = TransformerLM(dim, vocab_size, num_layers, 'v_equals_k', num_heads).to(device)
    ppl, tps = train_and_benchmark("V = K (reuse K as values)", model, train_data, val_data, device)
    results['V=K'] = (ppl, tps)
    del model
    torch.cuda.empty_cache()

    # 3. V = X
    model = TransformerLM(dim, vocab_size, num_layers, 'v_equals_x', num_heads).to(device)
    ppl, tps = train_and_benchmark("V = X (raw input as values)", model, train_data, val_data, device)
    results['V=X'] = (ppl, tps)
    del model
    torch.cuda.empty_cache()

    # 4. V = Q
    model = TransformerLM(dim, vocab_size, num_layers, 'v_equals_q', num_heads).to(device)
    ppl, tps = train_and_benchmark("V = Q (reuse Q as values)", model, train_data, val_data, device)
    results['V=Q'] = (ppl, tps)
    del model
    torch.cuda.empty_cache()

    # 5. Single projection (Q = K = V)
    model = TransformerLM(dim, vocab_size, num_layers, 'single_proj', num_heads).to(device)
    ppl, tps = train_and_benchmark("Single projection (Q=K=V)", model, train_data, val_data, device)
    results['Q=K=V'] = (ppl, tps)
    del model
    torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}", flush=True)
    print("RESULTS SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)

    baseline_ppl, baseline_tps = results['Standard']
    for name, (ppl, tps) in sorted(results.items(), key=lambda x: x[1][0]):
        ppl_diff = (ppl - baseline_ppl) / baseline_ppl * 100
        tps_diff = (tps - baseline_tps) / baseline_tps * 100
        print(f"{name:<30}: {ppl:6.2f} PPL ({ppl_diff:+5.1f}%), {tps:>10,.0f} TPS ({tps_diff:+6.1f}%)", flush=True)

    print(f"\n{'='*70}", flush=True)

    # Analysis
    print("\nANALYSIS:", flush=True)
    within_5pct = [name for name, (ppl, _) in results.items()
                   if ppl <= baseline_ppl * 1.05 and name != 'Standard']
    if within_5pct:
        print(f"Within 5% of baseline: {', '.join(within_5pct)}", flush=True)
        print("-> Value projection may be redundant!", flush=True)
    else:
        print("No alternatives within 5% of baseline", flush=True)
        print("-> Value projection appears necessary", flush=True)


if __name__ == "__main__":
    main()
