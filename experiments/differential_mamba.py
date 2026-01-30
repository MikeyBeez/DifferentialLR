#!/usr/bin/env python3
"""
Differential Learning Rate Training for 4 Softmax + 4 Mamba

Based on the Redundancy Bottleneck paper (Bee, 2025):
- Modules compete for gradient signal when trained jointly
- The faster-learning module (softmax) dominates early training
- Differential LR lets each module lead in turn

Strategy:
1. Phase 1: Softmax leads (high LR), Mamba follows (minimal LR)
2. Phase 2: Mamba catches up (high LR), Softmax follows (minimal LR)
3. Phase 3: Coordination (medium LR for both)

Key insight: NEVER freeze completely (use 1e-5 minimum) to maintain compatibility.
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
from tqdm import tqdm

from src.config import ModelConfig
from src.model import create_model


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


def get_layer_params(model):
    """
    Separate parameters by layer type for differential LR.

    Returns:
        embed_params: Token and position embeddings
        softmax_params: Layers 0-3 (softmax attention) + final norm
        mamba_params: Layers 4-7 (Mamba SSM)
    """
    embed_params = []
    softmax_params = []
    mamba_params = []

    # Embeddings (no weight decay)
    embed_params.extend(list(model.token_embedding.parameters()))
    embed_params.extend(list(model.position_embedding.parameters()))

    # Layer-specific params
    for i, block in enumerate(model.blocks):
        if i < 4:  # Softmax layers 0-3
            softmax_params.extend(list(block.parameters()))
        else:  # Mamba layers 4-7
            mamba_params.extend(list(block.parameters()))

    # Final norm goes with softmax (after all layers)
    softmax_params.extend(list(model.final_norm.parameters()))

    # Note: lm_head is tied to token_embedding, so already counted

    return embed_params, softmax_params, mamba_params


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

    dummy = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    for _ in range(10):
        _ = model(dummy, chunk_size=chunk_size)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        _ = model(dummy, chunk_size=chunk_size)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    return (num_runs * batch_size * seq_len) / elapsed


def train_uniform(model, train_loader, val_loader, device, num_epochs=10):
    """Baseline: Train with uniform learning rate (current approach)."""
    print("\n" + "="*60)
    print("BASELINE: Uniform Learning Rate")
    print("="*60)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    scaler = torch.amp.GradScaler('cuda')

    best_ppl = float('inf')
    start_time = time.time()

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
        train_ppl = math.exp(train_loss)
        eval_ppl = evaluate(model, val_loader, device)

        marker = "*" if eval_ppl < best_ppl else ""
        best_ppl = min(best_ppl, eval_ppl)

        elapsed = (time.time() - start_time) / 60
        print(f"Epoch {epoch+1:2d} | Train PPL: {train_ppl:7.1f} | Eval PPL: {eval_ppl:7.1f} {marker} | {elapsed:.1f}m")

    return best_ppl


def train_differential(model, train_loader, val_loader, device, total_epochs=10):
    """
    Train with phased differential learning rates.

    Based on PEER train_differential.py:
    - Phase 1: Softmax leads, Mamba follows
    - Phase 2: Mamba catches up, Softmax follows
    - Phase 3: Coordination

    Key: Never go below 1e-5 to maintain compatibility.
    """
    print("\n" + "="*60)
    print("DIFFERENTIAL LR: Phased Training")
    print("="*60)

    embed_params, softmax_params, mamba_params = get_layer_params(model)

    print(f"\nParameter groups:")
    print(f"  Embeddings: {sum(p.numel() for p in embed_params):,}")
    print(f"  Softmax (layers 0-3): {sum(p.numel() for p in softmax_params):,}")
    print(f"  Mamba (layers 4-7): {sum(p.numel() for p in mamba_params):,}")

    # Create optimizer with separate param groups
    optimizer = torch.optim.AdamW([
        {'params': embed_params, 'lr': 1e-4, 'weight_decay': 0.0, 'name': 'embeddings'},
        {'params': softmax_params, 'lr': 1e-3, 'weight_decay': 0.1, 'name': 'softmax'},
        {'params': mamba_params, 'lr': 1e-3, 'weight_decay': 0.1, 'name': 'mamba'},
    ])

    scaler = torch.amp.GradScaler('cuda')

    # Phase configuration: (name, epochs, softmax_lr, mamba_lr)
    # Key insight: Never go below 1e-5 to maintain coordination
    # Using lower LRs (closer to baseline 1e-4) for smaller model
    phases = [
        ("Softmax-Lead", 3, 3e-4, 1e-5),    # Softmax establishes features
        ("Mamba-Catchup", 4, 1e-5, 3e-4),   # Mamba learns sequence patterns
        ("Coordinate", 3, 1e-4, 1e-4),       # Both refine together
    ]

    best_ppl = float('inf')
    start_time = time.time()
    global_epoch = 0

    for phase_name, num_epochs, softmax_lr, mamba_lr in phases:
        print(f"\n{'='*50}")
        print(f"PHASE: {phase_name} ({num_epochs} epochs)")
        print(f"  Softmax LR: {softmax_lr:.0e}")
        print(f"  Mamba LR: {mamba_lr:.0e}")
        print("="*50)

        # Update learning rates
        optimizer.param_groups[1]['lr'] = softmax_lr  # softmax
        optimizer.param_groups[2]['lr'] = mamba_lr    # mamba

        for epoch in range(num_epochs):
            train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
            train_ppl = math.exp(train_loss)
            eval_ppl = evaluate(model, val_loader, device)

            marker = "*" if eval_ppl < best_ppl else ""
            best_ppl = min(best_ppl, eval_ppl)

            global_epoch += 1
            elapsed = (time.time() - start_time) / 60
            print(
                f"Epoch {global_epoch:2d} ({phase_name[:7]:7s}) | "
                f"Train PPL: {train_ppl:7.1f} | Eval PPL: {eval_ppl:7.1f} {marker} | "
                f"{elapsed:.1f}m"
            )

    return best_ppl


def create_hybrid_model(vocab_size, device):
    """Create 4 Softmax + 4 Mamba hybrid model."""
    config = ModelConfig(
        hidden_dim=512,
        num_layers=8,
        num_heads=8,
        head_dim=64,
        intermediate_dim=2048,
        vocab_size=vocab_size,
        max_seq_length=1024,
        dropout=0.1,
        chunk_size=64,
        softmax_layers=[0, 1, 2, 3],
        mamba_layers=[4, 5, 6, 7],
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
    )
    return create_model(config).to(device)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n" + "="*70)
    print("DIFFERENTIAL LR TRAINING FOR 4 SOFTMAX + 4 MAMBA")
    print("="*70)
    print("\nHypothesis: Phased differential LR can improve PPL beyond 154.0")
    print("            by preventing gradient signal competition.")

    # Load data
    print("\nLoading data...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_loader, val_loader = get_wikitext_data(
        tokenizer, seq_length=512, batch_size=8
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    results = []

    # Test 1: Uniform LR baseline
    print("\n" + "="*70)
    print("TEST 1: Uniform LR (baseline)")
    print("="*70)

    model = create_hybrid_model(tokenizer.vocab_size, device)
    num_params = model.get_num_params()
    print(f"Parameters: {num_params:,}")

    uniform_ppl = train_uniform(model, train_loader, val_loader, device, num_epochs=10)
    uniform_speed = benchmark_speed(model, device)

    results.append({
        'name': 'Uniform LR (baseline)',
        'ppl': uniform_ppl,
        'speed': uniform_speed,
    })
    print(f"\nUniform LR: Best PPL = {uniform_ppl:.1f}, Speed = {uniform_speed:.0f} t/s")

    del model
    torch.cuda.empty_cache()

    # Test 2: Differential LR
    print("\n" + "="*70)
    print("TEST 2: Differential LR (phased)")
    print("="*70)

    model = create_hybrid_model(tokenizer.vocab_size, device)

    differential_ppl = train_differential(model, train_loader, val_loader, device, total_epochs=10)
    differential_speed = benchmark_speed(model, device)

    results.append({
        'name': 'Differential LR (phased)',
        'ppl': differential_ppl,
        'speed': differential_speed,
    })
    print(f"\nDifferential LR: Best PPL = {differential_ppl:.1f}, Speed = {differential_speed:.0f} t/s")

    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'Configuration':<30} | {'PPL':>8} | {'Speed':>12}")
    print("-"*55)

    baseline_ppl = results[0]['ppl']
    for r in results:
        ppl_diff = (r['ppl'] - baseline_ppl) / baseline_ppl * 100
        print(f"{r['name']:<30} | {r['ppl']:>8.1f} | {r['speed']:>8.0f} t/s")
        if r['name'] != results[0]['name']:
            print(f"{'':30} | {ppl_diff:>+7.1f}% |")

    print("\n" + "="*70)
    print("COMPARISON TO PREVIOUS RESULTS")
    print("="*70)
    print(f"  Full Softmax baseline:        PPL 169.2")
    print(f"  4S+4M Uniform LR (previous):  PPL 154.0 (-9.0%)")
    print(f"  4S+4M Uniform LR (this run):  PPL {results[0]['ppl']:.1f}")
    print(f"  4S+4M Differential LR:        PPL {results[1]['ppl']:.1f}")

    improvement = (results[0]['ppl'] - results[1]['ppl']) / results[0]['ppl'] * 100
    print(f"\n  Differential LR improvement: {improvement:+.1f}%")


if __name__ == "__main__":
    main()
