#!/usr/bin/env python3
"""
MULTI-DATASET VALIDATION

Test the Golden Crystal on different datasets to verify generalization.
- WikiText-2 (baseline - we know this works)
- WikiText-103 (larger, same domain)
- Penn Treebank (classic benchmark)
- C4 (web text, more diverse)
"""

import sys
sys.path.insert(0, '/home/bee/Code/LinearAttention')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import math

PHI = (1 + math.sqrt(5)) / 2


class GoldenCrystal(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.proj = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        decay = torch.tensor([(1/PHI) ** (i+1) for i in range(num_heads)])
        self.register_buffer('decay', decay)

    def forward(self, x):
        B, N, D = x.shape
        t = self.proj(x).view(B, N, self.num_heads, self.head_dim)
        h_fwd = torch.zeros(B, self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)
        fwd = []
        for i in range(N):
            h_fwd = t[:, i] + self.decay.view(1, -1, 1) * h_fwd
            fwd.append(h_fwd)
        fwd = torch.stack(fwd, dim=1)
        h_bwd = torch.zeros(B, self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)
        bwd = []
        for i in range(N-1, -1, -1):
            h_bwd = t[:, i] + self.decay.view(1, -1, 1) * h_bwd
            bwd.append(h_bwd)
        bwd = torch.stack(bwd[::-1], dim=1)
        y = fwd + bwd - t
        return self.out(y.reshape(B, N, D))


class StandardAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        q = self.q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)
        return self.o(out.transpose(1, 2).reshape(B, N, D))


class Transformer(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, num_heads, use_crystal=True):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.use_crystal = use_crystal

        if not use_crystal:
            self.pos_embed = nn.Embedding(2048, dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if use_crystal:
                attn = GoldenCrystal(dim, num_heads)
            else:
                attn = StandardAttention(dim, num_heads)

            self.layers.append(nn.ModuleDict({
                'attn': attn,
                'norm1': nn.LayerNorm(dim),
                'ffn': nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)),
                'norm2': nn.LayerNorm(dim)
            }))

        self.final_norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids, labels=None):
        B, N = input_ids.shape
        x = self.embed(input_ids)

        if not self.use_crystal:
            pos = torch.arange(N, device=input_ids.device).unsqueeze(0)
            x = x + self.pos_embed(pos)

        for layer in self.layers:
            x = x + layer['attn'](layer['norm1'](x))
            x = x + layer['ffn'](layer['norm2'](x))

        logits = self.lm_head(self.final_norm(x))

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1))
            return {'loss': loss, 'logits': logits}
        return {'logits': logits}


def load_data(dataset_name, tokenizer, seq_length=256, batch_size=16, max_samples=50000):
    """Load and prepare dataset."""

    if dataset_name == "wikitext-2":
        train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        val = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        text_key = "text"

    elif dataset_name == "wikitext-103":
        train = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        val = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
        text_key = "text"

    elif dataset_name == "ptb":
        train = load_dataset("ptb_text_only", "penn_treebank", split="train", trust_remote_code=True)
        val = load_dataset("ptb_text_only", "penn_treebank", split="validation", trust_remote_code=True)
        text_key = "sentence"

    elif dataset_name == "c4":
        train = load_dataset("c4", "en", split="train", streaming=True, trust_remote_code=True)
        val = load_dataset("c4", "en", split="validation", streaming=True, trust_remote_code=True)
        text_key = "text"

        # For streaming, take samples
        train_texts = []
        for i, item in enumerate(train):
            if i >= max_samples:
                break
            train_texts.append(item[text_key])

        val_texts = []
        for i, item in enumerate(val):
            if i >= max_samples // 10:
                break
            val_texts.append(item[text_key])

        # Tokenize
        train_tokens = []
        for text in train_texts:
            train_tokens.extend(tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"])

        val_tokens = []
        for text in val_texts:
            val_tokens.extend(tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"])

        def create_sequences(tokens, seq_len):
            return torch.tensor([tokens[i:i+seq_len] for i in range(0, len(tokens)-seq_len, seq_len)])

        train_loader = DataLoader(create_sequences(train_tokens, seq_length), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(create_sequences(val_tokens, seq_length), batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Non-streaming datasets
    def tokenize(examples):
        return tokenizer(examples[text_key], truncation=False, padding=False)

    train_tok = train.map(tokenize, batched=True, remove_columns=train.column_names)
    val_tok = val.map(tokenize, batched=True, remove_columns=val.column_names)

    train_tokens, val_tokens = [], []
    for item in train_tok:
        train_tokens.extend(item["input_ids"])
    for item in val_tok:
        val_tokens.extend(item["input_ids"])

    # Limit training data for speed
    train_tokens = train_tokens[:min(len(train_tokens), max_samples * seq_length)]

    def create_sequences(tokens, seq_len):
        return torch.tensor([tokens[i:i+seq_len] for i in range(0, len(tokens)-seq_len, seq_len)])

    train_loader = DataLoader(create_sequences(train_tokens, seq_length), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(create_sequences(val_tokens, seq_length), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_and_eval(model, train_loader, val_loader, device, epochs=5):
    """Train and evaluate model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out = model(batch, labels=batch)
                loss = out['loss']
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                with torch.amp.autocast('cuda'):
                    out = model(batch, labels=batch)
                val_loss += out['loss'].item()

        val_ppl = math.exp(val_loss / len(val_loader))
        train_ppl = math.exp(total_loss / len(train_loader))
        print(f"    Epoch {epoch+1}: Train PPL {train_ppl:.1f}, Val PPL {val_ppl:.1f}")

    return val_ppl


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print("\n" + "=" * 70)
    print("MULTI-DATASET VALIDATION")
    print("=" * 70)
    print("\nTesting Golden Crystal generalization across datasets")

    datasets = ["wikitext-2", "ptb", "wikitext-103"]
    results = {}

    for dataset_name in datasets:
        print(f"\n{'=' * 70}")
        print(f"DATASET: {dataset_name.upper()}")
        print("=" * 70)

        try:
            train_loader, val_loader = load_data(dataset_name, tokenizer)
            print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        except Exception as e:
            print(f"Failed to load {dataset_name}: {e}")
            continue

        # Train Crystal
        print(f"\n  Golden Crystal (no positional encoding):")
        torch.manual_seed(42)
        torch.cuda.empty_cache()

        crystal = Transformer(
            vocab_size=tokenizer.vocab_size,
            dim=256,
            num_layers=4,
            num_heads=4,
            use_crystal=True
        ).to(device)

        crystal_ppl = train_and_eval(crystal, train_loader, val_loader, device, epochs=5)
        del crystal
        torch.cuda.empty_cache()

        # Train Attention
        print(f"\n  Standard Attention:")
        torch.manual_seed(42)

        attention = Transformer(
            vocab_size=tokenizer.vocab_size,
            dim=256,
            num_layers=4,
            num_heads=4,
            use_crystal=False
        ).to(device)

        attention_ppl = train_and_eval(attention, train_loader, val_loader, device, epochs=5)
        del attention
        torch.cuda.empty_cache()

        results[dataset_name] = {
            'crystal': crystal_ppl,
            'attention': attention_ppl,
            'gap': crystal_ppl - attention_ppl
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Dataset':<15} | {'Crystal PPL':>12} | {'Attention PPL':>14} | {'Gap':>8}")
    print("-" * 55)

    for dataset_name, r in results.items():
        print(f"{dataset_name:<15} | {r['crystal']:>12.1f} | {r['attention']:>14.1f} | {r['gap']:>+8.1f}")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    avg_gap = sum(r['gap'] for r in results.values()) / len(results)
    print(f"\nAverage PPL gap: {avg_gap:+.2f}")

    if avg_gap < 0.5:
        print("Crystal generalizes well - gap is consistent across datasets!")
    else:
        print("Crystal shows varying performance across datasets.")


if __name__ == "__main__":
    main()
