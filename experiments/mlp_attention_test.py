#!/usr/bin/env python3
"""
Dot Product Replacement Experiment #1: Concatenation + MLP

Replace Q·K^T with concat(Q, K) → small MLP → scalar

Hypothesis: If the projections can compensate for anything,
the specific comparison function shouldn't matter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from datasets import load_dataset
from transformers import GPT2Tokenizer


class DotProductAttention(nn.Module):
    """Standard dot product attention for baseline."""
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
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)

        # Standard dot product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_dropout(self.out(out))


class MLPAttention(nn.Module):
    """
    Replace dot product with concat(Q, K) → MLP → scalar.

    For each query position i and key position j:
    - Concatenate q_i and k_j
    - Pass through small MLP to get scalar score
    - Apply softmax over j dimension
    """
    def __init__(self, dim, num_heads=8, dropout=0.1, mlp_hidden=64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.mlp_hidden = mlp_hidden

        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        # MLP for computing attention scores
        # Input: concat(q, k) = 2 * head_dim
        # Output: scalar score
        self.score_mlp = nn.Sequential(
            nn.Linear(self.head_dim * 2, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, 1)
        )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)

        # Compute pairwise scores using MLP
        # q: (B, H, T, D), k: (B, H, T, D)
        # Need to compute score(q_i, k_j) for all i, j

        # Expand for pairwise comparison
        # q_expanded: (B, H, T, 1, D) -> broadcast over key positions
        # k_expanded: (B, H, 1, T, D) -> broadcast over query positions
        q_expanded = q.unsqueeze(3)  # (B, H, T, 1, D)
        k_expanded = k.unsqueeze(2)  # (B, H, 1, T, D)

        # Broadcast to (B, H, T, T, D) for both
        q_broadcast = q_expanded.expand(B, self.num_heads, T, T, self.head_dim)
        k_broadcast = k_expanded.expand(B, self.num_heads, T, T, self.head_dim)

        # Concatenate: (B, H, T, T, 2*D)
        qk_concat = torch.cat([q_broadcast, k_broadcast], dim=-1)

        # Apply MLP to get scores: (B, H, T, T, 1) -> (B, H, T, T)
        attn = self.score_mlp(qk_concat).squeeze(-1)

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_dropout(self.out(out))


class NegativeL2Attention(nn.Module):
    """
    Replace dot product with negative L2 distance.
    score(q, k) = -||q - k||^2

    Closer vectors get higher scores (inverted from dot product semantics).
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # Still scale for numerical stability

        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        # Negative L2 distance: -||q - k||^2
        # ||q - k||^2 = ||q||^2 + ||k||^2 - 2*q·k
        q_norm = (q ** 2).sum(dim=-1, keepdim=True)  # (B, H, T, 1)
        k_norm = (k ** 2).sum(dim=-1, keepdim=True)  # (B, H, T, 1)

        # Compute pairwise L2 distances
        # (B, H, T, 1) + (B, H, 1, T) - 2 * (B, H, T, T)
        dist_sq = q_norm + k_norm.transpose(-2, -1) - 2 * (q @ k.transpose(-2, -1))

        # Negative distance scaled
        attn = -dist_sq * self.scale

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_dropout(self.out(out))


class BilinearAttention(nn.Module):
    """
    Replace dot product with learned bilinear form.
    score(q, k) = q^T · W · k

    W is a learned matrix separate from projections.
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        # Learned bilinear matrix per head
        self.W_bilinear = nn.Parameter(torch.randn(num_heads, self.head_dim, self.head_dim) * 0.02)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)

        # Bilinear: q^T W k for each head
        # q: (B, H, T, D), W: (H, D, D), k: (B, H, T, D)
        # First: q @ W -> (B, H, T, D)
        q_transformed = torch.einsum('bhtd,hde->bhte', q, self.W_bilinear)
        # Then: (q @ W) @ k^T -> (B, H, T, T)
        attn = (q_transformed @ k.transpose(-2, -1)) * self.scale

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_dropout(self.out(out))


class AdditiveAttention(nn.Module):
    """
    Bahdanau-style additive attention.
    score(q, k) = v^T · tanh(W_q · q + W_k · k)
    """
    def __init__(self, dim, num_heads=8, dropout=0.1, attn_dim=64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_dim = attn_dim

        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        # Additive attention parameters per head
        self.W_q = nn.Linear(self.head_dim, attn_dim, bias=False)
        self.W_k = nn.Linear(self.head_dim, attn_dim, bias=False)
        self.v = nn.Parameter(torch.randn(num_heads, attn_dim) * 0.02)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v_values = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)

        # Project q and k
        q_proj = self.W_q(q)  # (B, H, T, attn_dim)
        k_proj = self.W_k(k)  # (B, H, T, attn_dim)

        # Additive combination for all pairs
        # q_proj: (B, H, T, 1, attn_dim)
        # k_proj: (B, H, 1, T, attn_dim)
        combined = torch.tanh(q_proj.unsqueeze(3) + k_proj.unsqueeze(2))  # (B, H, T, T, attn_dim)

        # Score with v: (B, H, T, T)
        attn = torch.einsum('bhija,ha->bhij', combined, self.v)

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v_values).transpose(1, 2).reshape(B, T, C)
        return self.out_dropout(self.out(out))


class TransformerBlock(nn.Module):
    def __init__(self, dim, attention_type='dot_product', num_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)

        if attention_type == 'dot_product':
            self.attn = DotProductAttention(dim, num_heads, dropout)
        elif attention_type == 'mlp':
            self.attn = MLPAttention(dim, num_heads, dropout)
        elif attention_type == 'negative_l2':
            self.attn = NegativeL2Attention(dim, num_heads, dropout)
        elif attention_type == 'bilinear':
            self.attn = BilinearAttention(dim, num_heads, dropout)
        elif attention_type == 'additive':
            self.attn = AdditiveAttention(dim, num_heads, dropout)
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
    def __init__(self, dim, vocab_size, num_layers=6, attention_type='dot_product',
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
    print(f"Dot Product Replacement Experiment", flush=True)
    print(f"Testing if the comparison function matters", flush=True)

    train_data, val_data, vocab_size = load_wikitext()
    print(f"Vocab size: {vocab_size}", flush=True)

    results = {}

    # Config for small model
    dim = 256
    num_layers = 6
    num_heads = 8

    # 1. Baseline: Standard dot product attention
    model = TransformerLM(dim, vocab_size, num_layers, 'dot_product', num_heads).to(device)
    ppl, tps = train_and_benchmark("Dot Product (baseline)", model, train_data, val_data, device)
    results['Dot Product'] = (ppl, tps)
    del model
    torch.cuda.empty_cache()

    # 2. MLP attention (concat + feedforward)
    model = TransformerLM(dim, vocab_size, num_layers, 'mlp', num_heads).to(device)
    ppl, tps = train_and_benchmark("MLP Attention (concat + FF)", model, train_data, val_data, device)
    results['MLP'] = (ppl, tps)
    del model
    torch.cuda.empty_cache()

    # 3. Negative L2 distance
    model = TransformerLM(dim, vocab_size, num_layers, 'negative_l2', num_heads).to(device)
    ppl, tps = train_and_benchmark("Negative L2 Distance", model, train_data, val_data, device)
    results['Negative L2'] = (ppl, tps)
    del model
    torch.cuda.empty_cache()

    # 4. Bilinear attention
    model = TransformerLM(dim, vocab_size, num_layers, 'bilinear', num_heads).to(device)
    ppl, tps = train_and_benchmark("Bilinear (learned W)", model, train_data, val_data, device)
    results['Bilinear'] = (ppl, tps)
    del model
    torch.cuda.empty_cache()

    # 5. Additive attention (Bahdanau)
    model = TransformerLM(dim, vocab_size, num_layers, 'additive', num_heads).to(device)
    ppl, tps = train_and_benchmark("Additive (Bahdanau)", model, train_data, val_data, device)
    results['Additive'] = (ppl, tps)
    del model
    torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}", flush=True)
    print("RESULTS SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)

    baseline_ppl, baseline_tps = results['Dot Product']
    for name, (ppl, tps) in sorted(results.items(), key=lambda x: x[1][0]):
        ppl_diff = (ppl - baseline_ppl) / baseline_ppl * 100
        tps_diff = (tps - baseline_tps) / baseline_tps * 100
        print(f"{name:<30}: {ppl:6.2f} PPL ({ppl_diff:+5.1f}%), {tps:>10,.0f} TPS ({tps_diff:+6.1f}%)", flush=True)

    print(f"\n{'='*70}", flush=True)

    # Analysis
    print("\nANALYSIS:", flush=True)
    within_10pct = [name for name, (ppl, _) in results.items()
                    if ppl <= baseline_ppl * 1.10 and name != 'Dot Product']
    if within_10pct:
        print(f"Within 10% of baseline: {', '.join(within_10pct)}", flush=True)
        print("-> The specific comparison function may not matter much", flush=True)
    else:
        print("No alternatives within 10% of baseline", flush=True)
        print("-> Dot product may have special properties", flush=True)


if __name__ == "__main__":
    main()
