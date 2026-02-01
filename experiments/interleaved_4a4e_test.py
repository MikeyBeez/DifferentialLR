#!/usr/bin/env python3
"""
Test: 4 Attention + 4 Engram (Attention first, then Engram)
Opposite of previous: A→E→A→E→A→E→A→E
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI


class GoldenEngramLayer(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.phi_inv = PHI_INV
        self.relo_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        state = torch.zeros(batch_size, self.dim, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(seq_len):
            current_x = x[:, t, :]
            alignment = torch.sum(current_x * state, dim=-1, keepdim=True) / math.sqrt(self.dim)
            gate = torch.sigmoid(alignment)
            new_info = self.relo_proj(current_x)
            state = (self.phi_inv * state) + (gate * new_info)
            outputs.append(state.unsqueeze(1))

        return self.dropout(torch.cat(outputs, dim=1))


class CausalAttentionLayer(nn.Module):
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


class InterleavedBlock(nn.Module):
    def __init__(self, dim, is_attention, num_heads=8, dropout=0.1):
        super().__init__()
        self.is_attention = is_attention

        self.norm1 = nn.LayerNorm(dim)
        if is_attention:
            self.core = CausalAttentionLayer(dim, num_heads, dropout)
        else:
            self.core = GoldenEngramLayer(dim, dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.core(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Interleaved4A4E(nn.Module):
    """Attention first: A→E→A→E→A→E→A→E"""
    def __init__(self, dim, vocab_size, num_layers=8, num_heads=8, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)

        # Attention on even layers (0,2,4,6), Engram on odd (1,3,5,7)
        self.blocks = nn.ModuleList([
            InterleavedBlock(dim, is_attention=(i % 2 == 0), num_heads=num_heads, dropout=dropout)
            for i in range(num_layers)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

        self.layer_types = ['Attention' if i % 2 == 0 else 'Engram' for i in range(num_layers)]

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


from datasets import load_dataset
from transformers import AutoTokenizer


def get_wikitext_data(tokenizer, seq_length=128):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    val_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=False, padding=False)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    val_tokenized = val_dataset.map(tokenize, batched=True, remove_columns=["text"])

    all_tokens, val_tokens = [], []
    for item in tokenized:
        all_tokens.extend(item["input_ids"])
    for item in val_tokenized:
        val_tokens.extend(item["input_ids"])

    print(f"Train: {len(all_tokens):,} tokens, Val: {len(val_tokens):,} tokens", flush=True)

    def create_sequences(tokens, seq_len):
        return torch.tensor([tokens[i:i+seq_len] for i in range(0, len(tokens)-seq_len, seq_len)])

    return create_sequences(all_tokens, seq_length), create_sequences(val_tokens, seq_length)


def train_epoch(model, train_data, optimizer, device, batch_size=16):
    model.train()
    total_loss, num_batches = 0, 0
    indices = torch.randperm(len(train_data))

    for start in range(0, len(train_data), batch_size):
        batch = train_data[indices[start:start+batch_size]].to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            logits = model(batch)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                batch[:, 1:].reshape(-1)
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, val_data, device, batch_size=16):
    model.eval()
    total_loss, num_batches = 0, 0

    for start in range(0, len(val_data), batch_size):
        batch = val_data[start:start+batch_size].to(device)
        with torch.amp.autocast('cuda'):
            logits = model(batch)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                batch[:, 1:].reshape(-1)
            )
        total_loss += loss.item()
        num_batches += 1

    return math.exp(total_loss / num_batches)


@torch.no_grad()
def benchmark_tps(model, device, seq_len=128, batch_size=8, num_runs=50):
    """Benchmark tokens per second."""
    model.eval()
    dummy = torch.randint(0, 50257, (batch_size, seq_len), device=device)

    # Warmup
    for _ in range(10):
        with torch.amp.autocast('cuda'):
            _ = model(dummy)
    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(num_runs):
        with torch.amp.autocast('cuda'):
            _ = model(dummy)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    total_tokens = num_runs * batch_size * seq_len
    return total_tokens / elapsed


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    train_data, val_data = get_wikitext_data(tokenizer, seq_length=128)

    model = Interleaved4A4E(
        dim=256,
        vocab_size=tokenizer.vocab_size,
        num_layers=8,
        num_heads=8,
        dropout=0.1
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*60}", flush=True)
    print(f"Interleaved 4A + 4E (Attention First)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Parameters: {num_params:,}", flush=True)
    print(f"Layer pattern: {' → '.join(model.layer_types)}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    best_ppl = float('inf')
    for epoch in range(8):
        train_loss = train_epoch(model, train_data, optimizer, device)
        train_ppl = math.exp(train_loss)
        val_ppl = evaluate(model, val_data, device)

        marker = " *" if val_ppl < best_ppl else ""
        best_ppl = min(best_ppl, val_ppl)

        print(f"Epoch {epoch+1}: Train PPL {train_ppl:7.1f}, Val PPL {val_ppl:7.1f}{marker}", flush=True)

    # Benchmark TPS
    print(f"\nBenchmarking throughput...", flush=True)
    tps = benchmark_tps(model, device)
    print(f"Throughput: {tps:,.0f} tokens/sec", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"FINAL: Best Val PPL = {best_ppl:.1f}, TPS = {tps:,.0f}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
