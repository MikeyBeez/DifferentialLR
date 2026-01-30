#!/usr/bin/env python3
"""
Mamba Hybrid Experiment: Softmax + Mamba

Based on the Static Function Hypothesis:
- Early layers need DYNAMIC attention for feature extraction on unstructured input
- Later layers perform PREDICTABLE operations on already-structured representations

Architecture tested:
- Layers 0-3: Full softmax attention (dynamic feature extraction)
- Layers 4-7: Mamba SSM layers (efficient sequence modeling)

Compares against:
- Full softmax baseline (8 layers)
- Softmax + Linear attention hybrid
- Different softmax/mamba ratios
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
import time

from src.config import ModelConfig
from src.model import ChunkedTransformerModel, create_model


def get_wikitext_data(tokenizer, seq_length=512, batch_size=8):
    """Load and prepare WikiText-2 dataset."""
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

    def create_sequences(tokens, seq_len):
        return torch.tensor([
            tokens[i:i+seq_len]
            for i in range(0, len(tokens)-seq_len, seq_len)
        ])

    train_loader = DataLoader(
        create_sequences(all_tokens, seq_length),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        create_sequences(val_tokens, seq_length),
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, scaler, device, chunk_size=64):
    """Train for one epoch."""
    model.train()
    total_loss, num_batches = 0, 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            outputs = model(batch, labels=batch, chunk_size=chunk_size)
            loss = outputs["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, val_loader, device, chunk_size=64):
    """Evaluate and compute perplexity."""
    model.eval()
    total_loss, num_batches = 0, 0

    for batch in val_loader:
        batch = batch.to(device)
        with torch.amp.autocast('cuda'):
            outputs = model(batch, labels=batch, chunk_size=chunk_size)
        total_loss += outputs["loss"].item()
        num_batches += 1

    return math.exp(total_loss / num_batches)


@torch.no_grad()
def benchmark_speed(model, device, seq_len=512, batch_size=1, num_runs=50, chunk_size=64):
    """Benchmark inference speed."""
    model.eval()
    vocab_size = model.config.vocab_size

    # Warmup
    dummy = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    for _ in range(10):
        _ = model(dummy, chunk_size=chunk_size)

    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(num_runs):
        _ = model(dummy, chunk_size=chunk_size)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    tokens_per_sec = (num_runs * batch_size * seq_len) / elapsed
    return tokens_per_sec


def create_hybrid_config(
    num_layers: int,
    softmax_layers: list,
    mamba_layers: list,
    vocab_size: int,
    hidden_dim: int = 512,
    num_heads: int = 8,
) -> ModelConfig:
    """Create config for hybrid model."""
    return ModelConfig(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=hidden_dim // num_heads,
        intermediate_dim=hidden_dim * 4,
        vocab_size=vocab_size,
        max_seq_length=1024,
        dropout=0.1,
        chunk_size=64,
        softmax_layers=softmax_layers,
        mamba_layers=mamba_layers,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_use_ffn=False,
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n" + "="*70)
    print("MAMBA HYBRID EXPERIMENT")
    print("="*70)
    print("\nHypothesis: Early layers need dynamic attention, later layers")
    print("            can use Mamba SSM on structured representations.")

    # Load data
    print("\nLoading data...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_loader, val_loader = get_wikitext_data(
        tokenizer, seq_length=512, batch_size=8
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Configuration variants to test
    # (name, softmax_layers, mamba_layers)
    # Layers not in either list use chunked linear attention
    configs = [
        ("Full Softmax (baseline)", list(range(8)), []),
        ("4 Softmax + 4 Mamba", [0,1,2,3], [4,5,6,7]),
        ("2 Softmax + 6 Mamba", [0,1], [2,3,4,5,6,7]),
        ("6 Softmax + 2 Mamba", [0,1,2,3,4,5], [6,7]),
        ("4 Softmax + 4 Linear", [0,1,2,3], []),  # Uses chunked linear for 4-7
        ("Full Mamba", [], list(range(8))),
    ]

    results = []

    for name, softmax_layers, mamba_layers in configs:
        linear_layers = [i for i in range(8) if i not in softmax_layers and i not in mamba_layers]

        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"  Softmax layers: {softmax_layers}")
        print(f"  Mamba layers: {mamba_layers}")
        print(f"  Linear layers: {linear_layers}")
        print("="*60)

        config = create_hybrid_config(
            num_layers=8,
            softmax_layers=softmax_layers,
            mamba_layers=mamba_layers,
            vocab_size=tokenizer.vocab_size,
            hidden_dim=512,
        )

        model = create_model(config).to(device)
        num_params = model.get_num_params()
        print(f"Parameters: {num_params:,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
        scaler = torch.amp.GradScaler('cuda')

        best_ppl = float('inf')
        start = time.time()

        for epoch in range(10):
            train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
            train_ppl = math.exp(train_loss)
            eval_ppl = evaluate(model, val_loader, device)

            marker = "*" if eval_ppl < best_ppl else ""
            best_ppl = min(best_ppl, eval_ppl)

            elapsed = (time.time() - start) / 60
            print(
                f"Epoch {epoch+1:2d} | "
                f"Train PPL: {train_ppl:7.1f} | "
                f"Eval PPL: {eval_ppl:7.1f} {marker} | "
                f"{elapsed:.1f}m",
                flush=True
            )

        # Benchmark speed
        speed = benchmark_speed(model, device)

        results.append({
            'name': name,
            'ppl': best_ppl,
            'params': num_params,
            'speed': speed,
        })

        print(f"\n{name}:")
        print(f"  Best PPL: {best_ppl:.1f}")
        print(f"  Speed: {speed:.0f} tokens/sec")

    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'Configuration':<35} | {'PPL':>8} | {'Speed':>12} | {'Params':>12}")
    print("-"*70)

    baseline_ppl = results[0]['ppl']
    baseline_speed = results[0]['speed']

    for r in results:
        ppl_diff = (r['ppl'] - baseline_ppl) / baseline_ppl * 100
        speed_diff = (r['speed'] - baseline_speed) / baseline_speed * 100
        print(
            f"{r['name']:<35} | "
            f"{r['ppl']:>8.1f} | "
            f"{r['speed']:>8.0f} t/s | "
            f"{r['params']:>10,}"
        )
        if r['name'] != results[0]['name']:
            print(f"{'':35} | {ppl_diff:>+7.1f}% | {speed_diff:>+7.1f}%")

    print("\n" + "="*70)
    print("COMPARISON TO PREVIOUS RESULTS (from RESEARCH_SUMMARY.md)")
    print("="*70)
    print("  Full Softmax baseline: PPL 170.4")
    print("  Linear (learned gate): PPL 167.4")
    print("  4 Softmax + 4 MLP:     PPL 174.5, +20% speed")
    print("  4 Softmax + 4 MLP+Proj: PPL 174.9, +16% speed")


if __name__ == "__main__":
    main()
