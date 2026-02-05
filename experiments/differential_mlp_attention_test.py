#!/usr/bin/env python3
"""
Differential MLP Attention Benchmark

Tests a fixed version of the DifferentialMLPAttention concept:
- Original had no causal masking (position 0 sees future tokens)
- Original had same context for all positions
- Fixed version uses cumulative sum for causal aggregation

Compares:
1. Standard O(N²) attention
2. Original DifferentialMLPAttention (non-causal, broken for LM)
3. Fixed causal version (cumsum)
4. Our best: O(N) learned causal convolution

Benchmarks both throughput and actual LM perplexity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from datasets import load_dataset
from transformers import GPT2Tokenizer


# --- Attention Variants ---

class StandardAttention(nn.Module):
    """Standard O(N²) attention."""
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


class DifferentialMLPAttentionOriginal(nn.Module):
    """
    Original DifferentialMLPAttention (NON-CAUSAL).
    Included for comparison but BROKEN for autoregressive LM:
    - Sums over all positions (future leakage)
    - Same context for every position
    """
    def __init__(self, dim, num_heads=8, dropout=0.1, **kwargs):
        super().__init__()
        self.relevance_mlp = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim)
        )
        self.q_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x):
        q = self.q_proj(x)
        v = self.v_proj(x)

        # NON-CAUSAL: sums over ALL positions including future
        k_rel = self.relevance_mlp(x)
        context = torch.sum(k_rel * v, dim=1, keepdim=True)  # Same for all positions

        out = q * context
        return self.out_dropout(self.out(out))


class DifferentialMLPAttentionFixed(nn.Module):
    """
    Fixed causal DifferentialMLPAttention.
    Uses cumulative sum so position i only sees positions 0..i.
    Each position gets a different context (running sum of history).
    """
    def __init__(self, dim, num_heads=8, dropout=0.1, **kwargs):
        super().__init__()
        self.relevance_mlp = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim)
        )
        self.q_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x):
        q = self.q_proj(x)
        v = self.v_proj(x)

        # Learn non-linear relevance
        k_rel = self.relevance_mlp(x)

        # CAUSAL: cumulative sum so position i only sees 0..i
        kv = k_rel * v  # (B, T, D)
        context = torch.cumsum(kv, dim=1)  # (B, T, D) - each position different

        out = q * context
        return self.out_dropout(self.out(out))


class LearnedCausalConvAttention(nn.Module):
    """Our best: O(N) learned causal convolution (from true_linear_attention_test)."""
    def __init__(self, dim, num_heads=8, dropout=0.1, kernel_size=64, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel_size = kernel_size
        self.dim = dim

        self.v_proj = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.out_dropout = nn.Dropout(dropout)

        self.kernel_logits = nn.Parameter(torch.zeros(num_heads, kernel_size))

    def forward(self, x):
        B, T, C = x.shape

        v = self.v_proj(x).transpose(1, 2)
        v_padded = F.pad(v, (self.kernel_size - 1, 0))

        kernel = F.softmax(self.kernel_logits, dim=-1)
        kernel_expanded = kernel.unsqueeze(1).expand(-1, self.head_dim, -1)
        kernel_expanded = kernel_expanded.reshape(self.dim, 1, self.kernel_size)

        out = F.conv1d(v_padded, kernel_expanded, groups=self.dim)
        out = out.transpose(1, 2)

        return self.out_dropout(self.out(out))


# --- Model ---

class TransformerBlock(nn.Module):
    def __init__(self, dim, attention_type='standard', num_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)

        attention_classes = {
            'standard': StandardAttention,
            'diff_original': DifferentialMLPAttentionOriginal,
            'diff_fixed': DifferentialMLPAttentionFixed,
            'conv': LearnedCausalConvAttention,
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


# --- Data ---

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


# --- Training ---

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


@torch.no_grad()
def benchmark_scaling(model, device, seq_lengths=[128, 512, 1024, 2048],
                      batch_size=4, num_runs=50):
    """Benchmark throughput at different sequence lengths."""
    model.eval()
    results = {}
    for seq_len in seq_lengths:
        try:
            dummy = torch.randint(0, 50257, (batch_size, seq_len), device=device)

            for _ in range(10):
                with torch.amp.autocast('cuda'):
                    _ = model(dummy)
            torch.cuda.synchronize()

            start = time.time()
            for _ in range(num_runs):
                with torch.amp.autocast('cuda'):
                    _ = model(dummy)
            torch.cuda.synchronize()
            elapsed = time.time() - start

            results[seq_len] = (num_runs * batch_size * seq_len) / elapsed
            del dummy
            torch.cuda.empty_cache()
        except RuntimeError:
            results[seq_len] = 0  # OOM
            torch.cuda.empty_cache()

    return results


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
    print(f"Throughput (128 tokens): {tps:,.0f} TPS", flush=True)

    # Scaling benchmark
    print("Scaling benchmark...", flush=True)
    scaling = benchmark_scaling(model, device)
    for seq_len, tps_val in scaling.items():
        if tps_val > 0:
            print(f"  {seq_len} tokens: {tps_val:,.0f} TPS", flush=True)
        else:
            print(f"  {seq_len} tokens: OOM", flush=True)

    return best_val_ppl, tps, scaling


def main():
    device = torch.device("cuda")
    print(f"Device: {device}", flush=True)
    print(f"Differential MLP Attention Benchmark", flush=True)
    print(f"Comparing original (broken), fixed (causal), and learned conv", flush=True)

    train_data, val_data, vocab_size = load_wikitext()
    print(f"Vocab size: {vocab_size}", flush=True)

    results = {}

    dim = 256
    num_layers = 6
    num_heads = 8

    # 1. Standard attention (baseline)
    model = TransformerLM(dim, vocab_size, num_layers, 'standard', num_heads).to(device)
    ppl, tps, scaling = train_and_benchmark(
        "Standard O(N²) Attention", model, train_data, val_data, device
    )
    results['Standard'] = (ppl, tps, scaling)
    del model
    torch.cuda.empty_cache()

    # 2. Original DifferentialMLPAttention (non-causal - should "cheat")
    model = TransformerLM(dim, vocab_size, num_layers, 'diff_original', num_heads).to(device)
    ppl, tps, scaling = train_and_benchmark(
        "DiffMLP Original (NON-CAUSAL, sees future)", model, train_data, val_data, device
    )
    results['DiffMLP-Orig'] = (ppl, tps, scaling)
    del model
    torch.cuda.empty_cache()

    # 3. Fixed causal DifferentialMLPAttention
    model = TransformerLM(dim, vocab_size, num_layers, 'diff_fixed', num_heads).to(device)
    ppl, tps, scaling = train_and_benchmark(
        "DiffMLP Fixed (causal cumsum)", model, train_data, val_data, device
    )
    results['DiffMLP-Fixed'] = (ppl, tps, scaling)
    del model
    torch.cuda.empty_cache()

    # 4. Learned causal convolution (our best)
    model = TransformerLM(dim, vocab_size, num_layers, 'conv', num_heads).to(device)
    ppl, tps, scaling = train_and_benchmark(
        "Learned Causal Conv O(N)", model, train_data, val_data, device
    )
    results['Conv'] = (ppl, tps, scaling)
    del model
    torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}", flush=True)
    print("RESULTS SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)

    baseline_ppl, baseline_tps, _ = results['Standard']
    print(f"\n{'Model':<30} {'PPL':>8} {'PPL Δ':>8} {'TPS':>12} {'Causal?':<8}", flush=True)
    print("-" * 75, flush=True)

    causal = {'Standard': 'Yes', 'DiffMLP-Orig': 'NO!', 'DiffMLP-Fixed': 'Yes', 'Conv': 'Yes'}

    for name, (ppl, tps, _) in sorted(results.items(), key=lambda x: x[1][0]):
        ppl_diff = (ppl - baseline_ppl) / baseline_ppl * 100
        print(f"{name:<30} {ppl:8.2f} {ppl_diff:+7.1f}% {tps:>12,.0f} {causal[name]:<8}", flush=True)

    print(f"\n{'='*70}", flush=True)

    # Scaling comparison
    print("\nSCALING COMPARISON (TPS at different sequence lengths):", flush=True)
    print(f"{'Seq Len':<10}", end="", flush=True)
    for name in ['Standard', 'DiffMLP-Fixed', 'Conv']:
        print(f"{name:<20}", end="", flush=True)
    print(flush=True)
    print("-" * 70, flush=True)

    seq_lengths = [128, 512, 1024, 2048]
    for seq_len in seq_lengths:
        print(f"{seq_len:<10}", end="", flush=True)
        for name in ['Standard', 'DiffMLP-Fixed', 'Conv']:
            tps_val = results[name][2].get(seq_len, 0)
            if tps_val > 0:
                print(f"{tps_val:<20,.0f}", end="", flush=True)
            else:
                print(f"{'OOM':<20}", end="", flush=True)
        print(flush=True)

    # Analysis
    print(f"\n{'='*70}", flush=True)
    print("\nANALYSIS:", flush=True)

    orig_ppl = results['DiffMLP-Orig'][0]
    fixed_ppl = results['DiffMLP-Fixed'][0]
    conv_ppl = results['Conv'][0]

    print(f"\n1. Original DiffMLP (non-causal):", flush=True)
    if orig_ppl < baseline_ppl * 0.8:
        print(f"   PPL {orig_ppl:.2f} - suspiciously good (likely cheating via future tokens)", flush=True)
    else:
        print(f"   PPL {orig_ppl:.2f}", flush=True)

    print(f"\n2. Fixed DiffMLP (causal):", flush=True)
    fixed_vs_std = (fixed_ppl - baseline_ppl) / baseline_ppl * 100
    print(f"   {fixed_vs_std:+.1f}% vs standard attention", flush=True)

    print(f"\n3. Learned Causal Conv (our best):", flush=True)
    conv_vs_std = (conv_ppl - baseline_ppl) / baseline_ppl * 100
    print(f"   {conv_vs_std:+.1f}% vs standard attention", flush=True)

    if conv_ppl < fixed_ppl:
        diff = (fixed_ppl - conv_ppl) / conv_ppl * 100
        print(f"   Conv beats DiffMLP-Fixed by {diff:.1f}%", flush=True)
    else:
        diff = (conv_ppl - fixed_ppl) / fixed_ppl * 100
        print(f"   DiffMLP-Fixed beats Conv by {diff:.1f}%", flush=True)


if __name__ == "__main__":
    main()
