#!/usr/bin/env python3
"""
Frozen Module Ablation Study

Tests the "Module Ceiling" by training each module in isolation:
1. Frozen Mamba, train Softmax only
2. Frozen Softmax, train Mamba only
3. Sequential: Train Mamba only, then Softmax only

This establishes the performance ceiling for each module in isolation
and proves that separation (not specific LR values) is the key.
"""

import sys
sys.path.insert(0, '/home/bee/Code/LinearAttention')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import math

from src.config import ModelConfig
from src.model import create_model


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


def create_hybrid_model(vocab_size, device):
    """4 Softmax + 4 Mamba hybrid."""
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


def get_param_groups(model):
    """Separate parameters by module type."""
    embed_params = []
    softmax_params = []
    mamba_params = []
    other_params = []

    # Embeddings
    for p in model.token_embedding.parameters():
        embed_params.append(p)
    for p in model.position_embedding.parameters():
        embed_params.append(p)

    # Layer-specific params
    for i, block in enumerate(model.blocks):
        if i < 4:  # Softmax layers 0-3
            for p in block.parameters():
                softmax_params.append(p)
        else:  # Mamba layers 4-7
            for p in block.parameters():
                mamba_params.append(p)

    # Final norm and LM head
    for p in model.final_norm.parameters():
        other_params.append(p)
    for p in model.lm_head.parameters():
        other_params.append(p)

    return embed_params, softmax_params, mamba_params, other_params


def train_epoch(model, train_loader, optimizer, scaler, device):
    model.train()
    total_loss = 0
    num_batches = 0

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
    model.eval()
    total_loss = 0
    num_batches = 0

    for batch in val_loader:
        batch = batch.to(device)
        with torch.amp.autocast('cuda'):
            outputs = model(batch, labels=batch, chunk_size=64)
        total_loss += outputs["loss"].item()
        num_batches += 1

    return math.exp(total_loss / num_batches)


def run_frozen_experiment(name, model, train_loader, val_loader, device,
                          train_softmax=True, train_mamba=True,
                          epochs=8, lr=1e-4):
    """Train with specified modules frozen."""
    embed_params, softmax_params, mamba_params, other_params = get_param_groups(model)

    # Freeze first, then build optimizer only with trainable params
    if not train_softmax:
        for p in softmax_params:
            p.requires_grad = False

    if not train_mamba:
        for p in mamba_params:
            p.requires_grad = False

    # Only include trainable parameters in optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.1)
    scaler = torch.amp.GradScaler('cuda')

    print(f"\n{name}")
    print(f"  Softmax trainable: {train_softmax}, Mamba trainable: {train_mamba}")
    print("-" * 50)

    best_ppl = float('inf')
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
        eval_ppl = evaluate(model, val_loader, device)
        best_ppl = min(best_ppl, eval_ppl)
        print(f"  Epoch {epoch+1}: Train {math.exp(train_loss):.1f}, Eval {eval_ppl:.1f}")

    return best_ppl


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n" + "=" * 70)
    print("FROZEN MODULE ABLATION STUDY")
    print("=" * 70)
    print("\nPurpose: Establish module ceilings and prove separation is key")

    # Load data
    print("\nLoading data...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_loader, val_loader = get_wikitext_data(tokenizer)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    results = {}

    # Experiment 1: Train Softmax only (freeze Mamba)
    print("\n" + "=" * 50)
    print("EXPERIMENT 1: FROZEN MAMBA (Softmax-only ceiling)")
    print("=" * 50)
    torch.manual_seed(42)
    model1 = create_hybrid_model(tokenizer.vocab_size, device)
    results["frozen_mamba"] = run_frozen_experiment(
        "Frozen Mamba", model1, train_loader, val_loader, device,
        train_softmax=True, train_mamba=False, epochs=8, lr=1e-4
    )
    del model1
    torch.cuda.empty_cache()

    # Experiment 2: Train Mamba only (freeze Softmax)
    print("\n" + "=" * 50)
    print("EXPERIMENT 2: FROZEN SOFTMAX (Mamba-only ceiling)")
    print("=" * 50)
    torch.manual_seed(42)
    model2 = create_hybrid_model(tokenizer.vocab_size, device)
    results["frozen_softmax"] = run_frozen_experiment(
        "Frozen Softmax", model2, train_loader, val_loader, device,
        train_softmax=False, train_mamba=True, epochs=8, lr=1e-4
    )
    del model2
    torch.cuda.empty_cache()

    # Experiment 3: Sequential training (Softmax first, then Mamba)
    print("\n" + "=" * 50)
    print("EXPERIMENT 3: SEQUENTIAL (Softmax first, then Mamba)")
    print("=" * 50)
    torch.manual_seed(42)
    model3 = create_hybrid_model(tokenizer.vocab_size, device)

    print("\nPhase 1: Train Softmax only (4 epochs)")
    run_frozen_experiment(
        "Phase 1: Softmax", model3, train_loader, val_loader, device,
        train_softmax=True, train_mamba=False, epochs=4, lr=1e-4
    )

    # Unfreeze mamba, freeze softmax for phase 2
    embed_params, softmax_params, mamba_params, other_params = get_param_groups(model3)
    for p in softmax_params:
        p.requires_grad = False
    for p in mamba_params:
        p.requires_grad = True

    print("\nPhase 2: Train Mamba only (4 epochs)")
    # Only use trainable params
    trainable_params = [p for p in model3.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.1)
    scaler = torch.amp.GradScaler('cuda')

    best_ppl = float('inf')
    for epoch in range(4):
        train_loss = train_epoch(model3, train_loader, optimizer, scaler, device)
        eval_ppl = evaluate(model3, val_loader, device)
        best_ppl = min(best_ppl, eval_ppl)
        print(f"  Epoch {epoch+1}: Train {math.exp(train_loss):.1f}, Eval {eval_ppl:.1f}")

    results["sequential"] = best_ppl
    del model3
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nFrozen Mamba (Softmax only):     {results['frozen_mamba']:.1f} PPL")
    print(f"Frozen Softmax (Mamba only):     {results['frozen_softmax']:.1f} PPL")
    print(f"Sequential (Softmax→Mamba):      {results['sequential']:.1f} PPL")
    print("\nReference values:")
    print(f"  Joint training (uniform LR):   191.5 PPL")
    print(f"  Phased Specialization:         157.5 PPL")
    print(f"  Full Softmax baseline:         167.3 PPL")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if results["sequential"] < 170:
        print("\nSequential training succeeds!")
        print("This proves that SEPARATION, not specific LR values, is the key.")
        print("Even with uniform LR, temporal separation enables specialization.")
    else:
        print("\nSequential training underperforms phased specialization.")
        print("This suggests that minimal LR (1e-5) during non-lead phases")
        print("provides necessary gradient signal for module compatibility.")


if __name__ == "__main__":
    main()
