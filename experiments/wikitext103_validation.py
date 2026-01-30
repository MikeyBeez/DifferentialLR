#!/usr/bin/env python3
"""
WikiText-103 Validation

Tests key findings on a larger dataset to address the
"WikiText-2 is too small" concern.

WikiText-103 is ~100x larger than WikiText-2.

Tests:
1. 4S + 4M vs Full Softmax
2. Differential LR vs Uniform LR
3. 6S + 2MLP vs Full Softmax
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

from src.config import ModelConfig
from src.model import create_model


def get_wikitext103_data(tokenizer, seq_length=512, batch_size=8, max_train_batches=2000):
    """
    Load WikiText-103.

    Note: Full WikiText-103 is very large. We use a subset for reasonable training time.
    max_train_batches limits training data while using full validation.
    """
    print("Loading WikiText-103 (this may take a moment)...")

    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    val_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=False, padding=False)

    print("Tokenizing train set...")
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    print("Tokenizing val set...")
    val_tokenized = val_dataset.map(tokenize, batched=True, remove_columns=["text"])

    all_tokens, val_tokens = [], []
    for item in tokenized:
        all_tokens.extend(item["input_ids"])
    for item in val_tokenized:
        val_tokens.extend(item["input_ids"])

    print(f"Total train tokens: {len(all_tokens):,}")
    print(f"Total val tokens: {len(val_tokens):,}")

    def create_sequences(tokens, seq_len):
        return torch.tensor([
            tokens[i:i+seq_len]
            for i in range(0, len(tokens)-seq_len, seq_len)
        ])

    train_seqs = create_sequences(all_tokens, seq_length)
    val_seqs = create_sequences(val_tokens, seq_length)

    # Limit training data if specified
    if max_train_batches and len(train_seqs) > max_train_batches * batch_size:
        train_seqs = train_seqs[:max_train_batches * batch_size]
        print(f"Limited to {len(train_seqs)} training sequences ({max_train_batches} batches)")

    train_loader = DataLoader(train_seqs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_seqs, batch_size=batch_size, shuffle=False)

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
    """Create model config."""
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


def train_uniform(model, train_loader, val_loader, device, num_epochs=5):
    """Train with uniform LR."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    scaler = torch.amp.GradScaler('cuda')

    best_ppl = float('inf')
    start_time = time.time()

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
        eval_ppl = evaluate(model, val_loader, device)
        best_ppl = min(best_ppl, eval_ppl)

        elapsed = (time.time() - start_time) / 60
        print(f"  Epoch {epoch+1}: Train PPL {math.exp(train_loss):.1f}, Eval PPL {eval_ppl:.1f}, {elapsed:.1f}m")

    return best_ppl


def train_differential(model, train_loader, val_loader, device):
    """Train with differential LR (no coordination)."""
    embed_params, softmax_params, mamba_params = get_layer_params(model)

    optimizer = torch.optim.AdamW([
        {'params': embed_params, 'lr': 1e-4, 'weight_decay': 0.0},
        {'params': softmax_params, 'lr': 3e-4, 'weight_decay': 0.1},
        {'params': mamba_params, 'lr': 1e-5, 'weight_decay': 0.1},
    ])
    scaler = torch.amp.GradScaler('cuda')

    best_ppl = float('inf')
    start_time = time.time()
    epoch = 0

    # Phase 1: Softmax Lead (2 epochs - less since more data)
    optimizer.param_groups[1]['lr'] = 3e-4
    optimizer.param_groups[2]['lr'] = 1e-5
    for _ in range(2):
        epoch += 1
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
        eval_ppl = evaluate(model, val_loader, device)
        best_ppl = min(best_ppl, eval_ppl)
        elapsed = (time.time() - start_time) / 60
        print(f"  Epoch {epoch} (softmax): Train PPL {math.exp(train_loss):.1f}, Eval PPL {eval_ppl:.1f}, {elapsed:.1f}m")

    # Phase 2: Mamba Catchup (3 epochs)
    optimizer.param_groups[1]['lr'] = 1e-5
    optimizer.param_groups[2]['lr'] = 3e-4
    for _ in range(3):
        epoch += 1
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
        eval_ppl = evaluate(model, val_loader, device)
        best_ppl = min(best_ppl, eval_ppl)
        elapsed = (time.time() - start_time) / 60
        print(f"  Epoch {epoch} (mamba): Train PPL {math.exp(train_loss):.1f}, Eval PPL {eval_ppl:.1f}, {elapsed:.1f}m")

    return best_ppl


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n" + "="*70)
    print("WIKITEXT-103 VALIDATION")
    print("="*70)
    print("\nTesting if key findings hold on larger dataset")

    # Load data
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Use subset of WikiText-103 (2000 batches = ~8M tokens)
    train_loader, val_loader = get_wikitext103_data(
        tokenizer, max_train_batches=2000
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    results = {}

    # Test 1: Full Softmax baseline
    print("\n" + "="*50)
    print("TEST 1: Full Softmax (baseline)")
    print("="*50)
    torch.manual_seed(42)
    config = create_model_config(tokenizer.vocab_size, "full_softmax")
    model = create_model(config).to(device)
    results["full_softmax"] = train_uniform(model, train_loader, val_loader, device, num_epochs=5)
    print(f"\nFull Softmax best PPL: {results['full_softmax']:.1f}")
    del model
    torch.cuda.empty_cache()

    # Test 2: 4S + 4M with uniform LR
    print("\n" + "="*50)
    print("TEST 2: 4S + 4M (uniform LR)")
    print("="*50)
    torch.manual_seed(42)
    config = create_model_config(tokenizer.vocab_size, "4s4m")
    model = create_model(config).to(device)
    results["4s4m_uniform"] = train_uniform(model, train_loader, val_loader, device, num_epochs=5)
    print(f"\n4S+4M uniform best PPL: {results['4s4m_uniform']:.1f}")
    del model
    torch.cuda.empty_cache()

    # Test 3: 4S + 4M with differential LR
    print("\n" + "="*50)
    print("TEST 3: 4S + 4M (differential LR)")
    print("="*50)
    torch.manual_seed(42)
    config = create_model_config(tokenizer.vocab_size, "4s4m")
    model = create_model(config).to(device)
    results["4s4m_differential"] = train_differential(model, train_loader, val_loader, device)
    print(f"\n4S+4M differential best PPL: {results['4s4m_differential']:.1f}")
    del model
    torch.cuda.empty_cache()

    # Test 4: 6S + 2MLP
    print("\n" + "="*50)
    print("TEST 4: 6S + 2MLP")
    print("="*50)
    torch.manual_seed(42)
    config = create_model_config(tokenizer.vocab_size, "6s2mlp")
    model = create_model(config).to(device)
    results["6s2mlp"] = train_uniform(model, train_loader, val_loader, device, num_epochs=5)
    print(f"\n6S+2MLP best PPL: {results['6s2mlp']:.1f}")
    del model
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*70)
    print("WIKITEXT-103 RESULTS SUMMARY")
    print("="*70)

    baseline = results["full_softmax"]
    print(f"\nFull Softmax (baseline): {baseline:.1f}")

    for name in ["4s4m_uniform", "4s4m_differential", "6s2mlp"]:
        ppl = results[name]
        change = (ppl - baseline) / baseline * 100
        print(f"{name}: {ppl:.1f} ({change:+.1f}% vs baseline)")

    print("\n" + "="*70)
    print("COMPARISON TO WIKITEXT-2 FINDINGS")
    print("="*70)

    findings = [
        ("4S+4M beats full softmax", results["4s4m_uniform"] < baseline),
        ("Differential LR helps 4S+4M", results["4s4m_differential"] < results["4s4m_uniform"]),
        ("6S+2MLP beats full softmax", results["6s2mlp"] < baseline),
    ]

    for claim, holds in findings:
        status = "CONFIRMED" if holds else "NOT CONFIRMED"
        print(f"  {claim}: {status}")

    # Overall
    print("\n" + "="*70)
    confirmed = sum(1 for _, holds in findings if holds)
    print(f"WikiText-103 confirms {confirmed}/{len(findings)} key findings")


if __name__ == "__main__":
    main()
