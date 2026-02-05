#!/usr/bin/env python3
"""
Long Context Experiment

Tests whether O(N) convolution advantage holds at longer sequence lengths.
Previous experiments used 128 tokens. This tests 256, 512, 1024, 2048.

Uses Wikipedia articles for real-world long-form content.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from datasets import load_dataset
from transformers import GPT2Tokenizer


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


class TrueLinearAttention(nn.Module):
    """True O(N) attention via causal convolution."""
    def __init__(self, dim, num_heads=8, dropout=0.1, kernel_size=64, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel_size = kernel_size
        self.dim = dim

        self.v_proj = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.out_dropout = nn.Dropout(dropout)

        # Learned causal kernel per head
        self.kernel_logits = nn.Parameter(torch.zeros(num_heads, kernel_size))

    def forward(self, x):
        B, T, C = x.shape

        v = self.v_proj(x).transpose(1, 2)  # (B, C, T)
        v_padded = F.pad(v, (self.kernel_size - 1, 0))

        kernel = F.softmax(self.kernel_logits, dim=-1)  # (H, K)
        kernel_expanded = kernel.unsqueeze(1).expand(-1, self.head_dim, -1)
        kernel_expanded = kernel_expanded.reshape(self.dim, 1, self.kernel_size)

        out = F.conv1d(v_padded, kernel_expanded, groups=self.dim)
        out = out.transpose(1, 2)

        return self.out_dropout(self.out(out))


class AdaptiveLinearAttention(nn.Module):
    """
    O(N) attention with kernel size adapting to sequence length.
    Uses larger kernel for longer sequences.
    """
    def __init__(self, dim, num_heads=8, dropout=0.1, max_kernel_size=256, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.max_kernel_size = max_kernel_size
        self.dim = dim

        self.v_proj = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.out_dropout = nn.Dropout(dropout)

        # Learned kernel up to max size
        self.kernel_logits = nn.Parameter(torch.zeros(num_heads, max_kernel_size))

    def forward(self, x):
        B, T, C = x.shape

        # Adapt kernel size to sequence length (at most T, at most max_kernel_size)
        kernel_size = min(T, self.max_kernel_size)

        v = self.v_proj(x).transpose(1, 2)
        v_padded = F.pad(v, (kernel_size - 1, 0))

        # Use only first kernel_size weights
        kernel = F.softmax(self.kernel_logits[:, :kernel_size], dim=-1)
        kernel_expanded = kernel.unsqueeze(1).expand(-1, self.head_dim, -1)
        kernel_expanded = kernel_expanded.reshape(self.dim, 1, kernel_size)

        out = F.conv1d(v_padded, kernel_expanded, groups=self.dim)
        out = out.transpose(1, 2)

        return self.out_dropout(self.out(out))


class TransformerBlock(nn.Module):
    def __init__(self, dim, attention_type='standard', num_heads=8, dropout=0.1,
                 kernel_size=64):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)

        if attention_type == 'standard':
            self.attn = StandardAttention(dim, num_heads, dropout)
        elif attention_type == 'linear':
            self.attn = TrueLinearAttention(dim, num_heads, dropout, kernel_size)
        elif attention_type == 'adaptive':
            self.attn = AdaptiveLinearAttention(dim, num_heads, dropout, kernel_size)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

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
                 num_heads=8, dropout=0.1, kernel_size=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, attention_type, num_heads, dropout, kernel_size)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def load_wikitext103(max_length):
    """Load WikiText-103 (longer documents than WikiText-2)."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading WikiText-103 dataset...", flush=True)
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )

    print(f"Tokenizing (max_length={max_length})...", flush=True)
    train_data = dataset["train"].map(tokenize, batched=True, remove_columns=["text"])
    val_data = dataset["validation"].map(tokenize, batched=True, remove_columns=["text"])

    train_data.set_format(type="torch", columns=["input_ids"])
    val_data.set_format(type="torch", columns=["input_ids"])

    print(f"Train: {len(train_data)}, Val: {len(val_data)}", flush=True)
    return train_data, val_data, tokenizer.vocab_size


def train_epoch(model, data, optimizer, scaler, device, batch_size=8):
    model.train()
    total_loss = 0
    num_batches = 0

    indices = torch.randperm(len(data)).tolist()
    for i in range(0, min(len(indices) - batch_size, 500 * batch_size), batch_size):
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

    return math.exp(total_loss / num_batches) if num_batches > 0 else float('inf')


@torch.no_grad()
def evaluate(model, data, device, batch_size=8):
    model.eval()
    total_loss = 0
    num_batches = 0

    for i in range(0, min(len(data) - batch_size, 100 * batch_size), batch_size):
        batch = torch.stack([data[j]["input_ids"] for j in range(i, i+batch_size)]).to(device)
        with torch.amp.autocast('cuda'):
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
        total_loss += loss.item()
        num_batches += 1

    return math.exp(total_loss / num_batches) if num_batches > 0 else float('inf')


@torch.no_grad()
def benchmark_tps(model, device, seq_len, batch_size=4, num_runs=50):
    model.eval()
    dummy = torch.randint(0, 50257, (batch_size, seq_len), device=device)

    # Warmup
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

    return (num_runs * batch_size * seq_len) / elapsed


def test_sequence_length(seq_len, device, epochs=3):
    """Test both models at a specific sequence length."""
    print(f"\n{'='*70}", flush=True)
    print(f"SEQUENCE LENGTH: {seq_len}", flush=True)
    print(f"{'='*70}", flush=True)

    # Load data for this sequence length
    train_data, val_data, vocab_size = load_wikitext103(max_length=seq_len)

    dim = 256
    num_layers = 6
    num_heads = 8

    # Adjust batch size for memory
    if seq_len <= 512:
        batch_size = 16
    elif seq_len <= 1024:
        batch_size = 8
    else:
        batch_size = 4

    results = {}

    # 1. Standard attention
    print(f"\n--- Standard Attention O(N²) ---", flush=True)
    model = TransformerLM(dim, vocab_size, num_layers, 'standard', num_heads).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')

    best_ppl = float('inf')
    for epoch in range(1, epochs + 1):
        train_ppl = train_epoch(model, train_data, optimizer, scaler, device, batch_size)
        val_ppl = evaluate(model, val_data, device, batch_size)
        marker = " *" if val_ppl < best_ppl else ""
        if val_ppl < best_ppl:
            best_ppl = val_ppl
        print(f"Epoch {epoch}: Train PPL {train_ppl:7.1f}, Val PPL {val_ppl:7.1f}{marker}", flush=True)

    tps = benchmark_tps(model, device, seq_len, batch_size=4)
    print(f"Throughput: {tps:,.0f} TPS", flush=True)
    results['Standard'] = (best_ppl, tps)

    del model, optimizer, scaler
    torch.cuda.empty_cache()

    # 2. True O(N) linear attention with adaptive kernel
    kernel_size = min(seq_len, 256)  # Cap kernel at 256
    print(f"\n--- O(N) Linear (adaptive kernel up to {kernel_size}) ---", flush=True)
    model = TransformerLM(dim, vocab_size, num_layers, 'adaptive', num_heads,
                          kernel_size=kernel_size).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')

    best_ppl = float('inf')
    for epoch in range(1, epochs + 1):
        train_ppl = train_epoch(model, train_data, optimizer, scaler, device, batch_size)
        val_ppl = evaluate(model, val_data, device, batch_size)
        marker = " *" if val_ppl < best_ppl else ""
        if val_ppl < best_ppl:
            best_ppl = val_ppl
        print(f"Epoch {epoch}: Train PPL {train_ppl:7.1f}, Val PPL {val_ppl:7.1f}{marker}", flush=True)

    tps = benchmark_tps(model, device, seq_len, batch_size=4)
    print(f"Throughput: {tps:,.0f} TPS", flush=True)
    results['Linear'] = (best_ppl, tps)

    del model, optimizer, scaler
    torch.cuda.empty_cache()

    return results


def main():
    device = torch.device("cuda")
    print(f"Device: {device}", flush=True)
    print(f"Long Context Experiment", flush=True)
    print(f"Testing O(N) vs O(N²) at different sequence lengths", flush=True)
    print(f"Using WikiText-103 for longer documents", flush=True)

    # Test different sequence lengths
    sequence_lengths = [256, 512, 1024]

    all_results = {}

    for seq_len in sequence_lengths:
        results = test_sequence_length(seq_len, device, epochs=3)
        if results:
            all_results[seq_len] = results

    # Summary
    print(f"\n{'='*70}", flush=True)
    print("SUMMARY: O(N) vs O(N²) Across Sequence Lengths", flush=True)
    print(f"{'='*70}", flush=True)

    print(f"\n{'Seq Len':<10} {'Standard PPL':<15} {'Linear PPL':<15} {'PPL Δ':<10} {'Std TPS':<12} {'Lin TPS':<12} {'TPS Δ':<10}", flush=True)
    print("-" * 90, flush=True)

    for seq_len in sequence_lengths:
        if seq_len not in all_results:
            continue

        std_ppl, std_tps = all_results[seq_len]['Standard']
        lin_ppl, lin_tps = all_results[seq_len]['Linear']

        ppl_diff = (lin_ppl - std_ppl) / std_ppl * 100
        tps_diff = (lin_tps - std_tps) / std_tps * 100

        print(f"{seq_len:<10} {std_ppl:<15.2f} {lin_ppl:<15.2f} {ppl_diff:+8.1f}% {std_tps:<12,.0f} {lin_tps:<12,.0f} {tps_diff:+8.1f}%", flush=True)

    print(f"\n{'='*70}", flush=True)

    # Analysis
    print("\nANALYSIS:", flush=True)

    if all_results:
        # Check if linear maintains advantage at longer sequences
        advantages = []
        for seq_len, results in all_results.items():
            std_ppl, _ = results['Standard']
            lin_ppl, _ = results['Linear']
            advantages.append((seq_len, (lin_ppl - std_ppl) / std_ppl * 100))

        print("\nPPL advantage by sequence length:", flush=True)
        for seq_len, adv in advantages:
            status = "BETTER" if adv < 0 else "WORSE"
            print(f"  {seq_len}: {adv:+.1f}% ({status})", flush=True)

        # Check throughput scaling
        print("\nThroughput scaling:", flush=True)
        for seq_len, results in all_results.items():
            std_tps = results['Standard'][1]
            lin_tps = results['Linear'][1]
            ratio = lin_tps / std_tps
            print(f"  {seq_len}: Linear is {ratio:.2f}x Standard TPS", flush=True)


if __name__ == "__main__":
    main()
