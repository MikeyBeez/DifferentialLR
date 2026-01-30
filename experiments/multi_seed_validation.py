#!/usr/bin/env python3
"""
Multi-seed validation for key configurations.

Runs 5 seeds each for:
1. 4S + 4M with uniform LR
2. 4S + 4M with differential LR (no coordination)
3. 6S + 2MLP
4. Full softmax baseline

Reports mean and std for each.
"""

import sys
sys.path.insert(0, '/home/bee/Code/LinearAttention')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import math
import time
import numpy as np

from src.config import ModelConfig
from src.model import create_model


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_wikitext_data(tokenizer, seq_length=512, batch_size=8):
    """Load WikiText-2."""
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
    """Separate parameters by layer type."""
    embed_params, softmax_params, mamba_params = [], [], []

    embed_params.extend(list(model.token_embedding.parameters()))
    embed_params.extend(list(model.position_embedding.parameters()))

    for i, block in enumerate(model.blocks):
        if i < 4:
            softmax_params.extend(list(block.parameters()))
        else:
            mamba_params.extend(list(block.parameters()))

    softmax_params.extend(list(model.final_norm.parameters()))
    return embed_params, softmax_params, mamba_params


def train_epoch(model, train_loader, optimizer, scaler, device):
    """Train for one epoch."""
    model.train()
    total_loss, num_batches = 0, 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            outputs = model(batch, labels=batch, chunk_size=64)
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
def evaluate(model, val_loader, device):
    """Evaluate and compute perplexity."""
    model.eval()
    total_loss, num_batches = 0, 0

    for batch in val_loader:
        batch = batch.to(device)
        with torch.amp.autocast('cuda'):
            outputs = model(batch, labels=batch, chunk_size=64)
        total_loss += outputs["loss"].item()
        num_batches += 1

    return math.exp(total_loss / num_batches)


def create_model_config(vocab_size, config_name):
    """Create model config for different architectures."""
    base = dict(
        hidden_dim=512,
        num_layers=8,
        num_heads=8,
        head_dim=64,
        intermediate_dim=2048,
        vocab_size=vocab_size,
        max_seq_length=1024,
        dropout=0.1,
        chunk_size=64,
    )

    if config_name == "full_softmax":
        base["softmax_layers"] = [0, 1, 2, 3, 4, 5, 6, 7]
        base["mamba_layers"] = []
    elif config_name == "4s4m":
        base["softmax_layers"] = [0, 1, 2, 3]
        base["mamba_layers"] = [4, 5, 6, 7]
        base["mamba_d_state"] = 16
        base["mamba_d_conv"] = 4
        base["mamba_expand"] = 2
    elif config_name == "6s2mlp":
        base["softmax_layers"] = [0, 1, 2, 3, 4, 5]
        base["mamba_layers"] = []
        base["mlp_only_layers"] = [6, 7]

    return ModelConfig(**base)


def train_uniform(model, train_loader, val_loader, device, num_epochs=10):
    """Train with uniform LR."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    scaler = torch.amp.GradScaler('cuda')

    best_ppl = float('inf')
    for epoch in range(num_epochs):
        train_epoch(model, train_loader, optimizer, scaler, device)
        eval_ppl = evaluate(model, val_loader, device)
        best_ppl = min(best_ppl, eval_ppl)

    return best_ppl


def train_differential_no_coord(model, train_loader, val_loader, device):
    """Train with differential LR, NO coordination phase."""
    embed_params, softmax_params, mamba_params = get_layer_params(model)

    optimizer = torch.optim.AdamW([
        {'params': embed_params, 'lr': 1e-4, 'weight_decay': 0.0},
        {'params': softmax_params, 'lr': 3e-4, 'weight_decay': 0.1},
        {'params': mamba_params, 'lr': 1e-5, 'weight_decay': 0.1},
    ])
    scaler = torch.amp.GradScaler('cuda')

    best_ppl = float('inf')

    # Phase 1: Softmax Lead (4 epochs)
    optimizer.param_groups[1]['lr'] = 3e-4
    optimizer.param_groups[2]['lr'] = 1e-5
    for epoch in range(4):
        train_epoch(model, train_loader, optimizer, scaler, device)
        eval_ppl = evaluate(model, val_loader, device)
        best_ppl = min(best_ppl, eval_ppl)

    # Phase 2: Mamba Catchup (6 epochs) - then STOP
    optimizer.param_groups[1]['lr'] = 1e-5
    optimizer.param_groups[2]['lr'] = 3e-4
    for epoch in range(6):
        train_epoch(model, train_loader, optimizer, scaler, device)
        eval_ppl = evaluate(model, val_loader, device)
        best_ppl = min(best_ppl, eval_ppl)

    return best_ppl


def run_config(config_name, train_loader, val_loader, vocab_size, device, seeds, training="uniform"):
    """Run a configuration across multiple seeds."""
    results = []

    for seed in seeds:
        set_seed(seed)

        config = create_model_config(vocab_size, config_name)
        model = create_model(config).to(device)

        if training == "uniform":
            ppl = train_uniform(model, train_loader, val_loader, device)
        elif training == "differential":
            ppl = train_differential_no_coord(model, train_loader, val_loader, device)

        results.append(ppl)
        print(f"  Seed {seed}: PPL = {ppl:.1f}")

        del model
        torch.cuda.empty_cache()

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n" + "="*70)
    print("MULTI-SEED VALIDATION (5 seeds per config)")
    print("="*70)

    # Load data once
    print("\nLoading data...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_loader, val_loader = get_wikitext_data(tokenizer)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    seeds = [42, 123, 456, 789, 1337]
    results = {}

    # Config 1: Full Softmax (baseline)
    print("\n" + "="*50)
    print("CONFIG 1: Full Softmax (uniform LR)")
    print("="*50)
    results["full_softmax"] = run_config(
        "full_softmax", train_loader, val_loader,
        tokenizer.vocab_size, device, seeds, "uniform"
    )

    # Config 2: 4S + 4M with uniform LR
    print("\n" + "="*50)
    print("CONFIG 2: 4S + 4M (uniform LR)")
    print("="*50)
    results["4s4m_uniform"] = run_config(
        "4s4m", train_loader, val_loader,
        tokenizer.vocab_size, device, seeds, "uniform"
    )

    # Config 3: 4S + 4M with differential LR (no coordination)
    print("\n" + "="*50)
    print("CONFIG 3: 4S + 4M (differential LR, no coordination)")
    print("="*50)
    results["4s4m_differential"] = run_config(
        "4s4m", train_loader, val_loader,
        tokenizer.vocab_size, device, seeds, "differential"
    )

    # Config 4: 6S + 2MLP
    print("\n" + "="*50)
    print("CONFIG 4: 6S + 2MLP (uniform LR)")
    print("="*50)
    results["6s2mlp"] = run_config(
        "6s2mlp", train_loader, val_loader,
        tokenizer.vocab_size, device, seeds, "uniform"
    )

    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY (5 seeds each)")
    print("="*70)
    print(f"{'Configuration':<40} | {'Mean PPL':>10} | {'Std':>8} | {'Min':>8} | {'Max':>8}")
    print("-"*80)

    baseline_mean = np.mean(results["full_softmax"])

    for name, ppls in results.items():
        mean_ppl = np.mean(ppls)
        std_ppl = np.std(ppls)
        min_ppl = np.min(ppls)
        max_ppl = np.max(ppls)

        change = (mean_ppl - baseline_mean) / baseline_mean * 100
        print(f"{name:<40} | {mean_ppl:>10.1f} | {std_ppl:>8.1f} | {min_ppl:>8.1f} | {max_ppl:>8.1f}")
        if name != "full_softmax":
            print(f"{'  vs baseline:':<40} | {change:>+9.1f}%")

    print("\n" + "="*70)
    print("STATISTICAL SIGNIFICANCE")
    print("="*70)

    # Simple t-test approximation
    for name in ["4s4m_uniform", "4s4m_differential", "6s2mlp"]:
        baseline = results["full_softmax"]
        treatment = results[name]

        mean_diff = np.mean(treatment) - np.mean(baseline)
        pooled_std = np.sqrt((np.var(baseline) + np.var(treatment)) / 2)
        se = pooled_std * np.sqrt(2/5)  # 5 samples each
        t_stat = mean_diff / se if se > 0 else 0

        print(f"{name}: mean diff = {mean_diff:.1f}, t = {t_stat:.2f}")
        if abs(t_stat) > 2.31:  # p < 0.05 for df=8
            print(f"  → Statistically significant (p < 0.05)")
        else:
            print(f"  → NOT statistically significant")


if __name__ == "__main__":
    main()
