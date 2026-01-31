#!/usr/bin/env python3
"""
Test lower softmax LR within phased training.

Current phased approach:
  Phase 1: Softmax 3e-4, Mamba 1e-5
  Phase 2: Softmax 1e-5, Mamba 3e-4

What if we lower softmax LR in Phase 1?
  Phase 1: Softmax 1e-4 (or lower), Mamba 1e-5
  Phase 2: Softmax 1e-5, Mamba 3e-4

This tests whether throttling softmax from the start helps.
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

    train_loader = DataLoader(create_sequences(all_tokens, seq_length), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(create_sequences(val_tokens, seq_length), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def get_layer_params(model):
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


def create_hybrid_model(vocab_size, device):
    config = ModelConfig(
        hidden_dim=512, num_layers=8, num_heads=8, head_dim=64,
        intermediate_dim=2048, vocab_size=vocab_size, max_seq_length=1024,
        dropout=0.1, chunk_size=64,
        softmax_layers=[0, 1, 2, 3], mamba_layers=[4, 5, 6, 7],
        mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
    )
    return create_model(config).to(device)


def train_epoch(model, train_loader, optimizer, scaler, device):
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
    model.eval()
    total_loss, num_batches = 0, 0
    for batch in val_loader:
        batch = batch.to(device)
        with torch.amp.autocast('cuda'):
            outputs = model(batch, labels=batch, chunk_size=64)
        total_loss += outputs["loss"].item()
        num_batches += 1
    return math.exp(total_loss / num_batches)


def train_phased(name, model, train_loader, val_loader, device,
                 phase1_softmax_lr, phase1_mamba_lr,
                 phase2_softmax_lr, phase2_mamba_lr,
                 phase1_epochs=4, phase2_epochs=4):
    """Train with specified phase LRs."""
    embed_params, softmax_params, mamba_params = get_layer_params(model)

    optimizer = torch.optim.AdamW([
        {'params': embed_params, 'lr': 1e-4, 'weight_decay': 0.0},
        {'params': softmax_params, 'lr': phase1_softmax_lr, 'weight_decay': 0.1},
        {'params': mamba_params, 'lr': phase1_mamba_lr, 'weight_decay': 0.1},
    ])
    scaler = torch.amp.GradScaler('cuda')

    print(f"\n{name}")
    print(f"  Phase 1: Softmax {phase1_softmax_lr}, Mamba {phase1_mamba_lr}")
    print(f"  Phase 2: Softmax {phase2_softmax_lr}, Mamba {phase2_mamba_lr}")
    print("-" * 50)

    # Phase 1
    print("  Phase 1:")
    for epoch in range(phase1_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
        eval_ppl = evaluate(model, val_loader, device)
        print(f"    Epoch {epoch+1}: Train {math.exp(train_loss):.1f}, Eval {eval_ppl:.1f}")

    # Phase 2
    optimizer.param_groups[1]['lr'] = phase2_softmax_lr
    optimizer.param_groups[2]['lr'] = phase2_mamba_lr

    print("  Phase 2:")
    best_ppl = float('inf')
    for epoch in range(phase2_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
        eval_ppl = evaluate(model, val_loader, device)
        best_ppl = min(best_ppl, eval_ppl)
        print(f"    Epoch {epoch+1}: Train {math.exp(train_loss):.1f}, Eval {eval_ppl:.1f}")

    return best_ppl


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n" + "=" * 70)
    print("LOWER SOFTMAX LR + PHASED TRAINING")
    print("=" * 70)
    print("\nQuestion: Does throttling softmax in Phase 1 help?")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_loader, val_loader = get_wikitext_data(tokenizer)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    results = {}

    # Test configurations
    configs = [
        # (name, phase1_softmax, phase1_mamba, phase2_softmax, phase2_mamba)
        ("Original (3e-4/1e-5 → 1e-5/3e-4)", 3e-4, 1e-5, 1e-5, 3e-4),
        ("Lower P1 softmax (1e-4)", 1e-4, 1e-5, 1e-5, 3e-4),
        ("Lower P1 softmax (5e-5)", 5e-5, 1e-5, 1e-5, 3e-4),
        ("Equal P1 (1e-4/1e-4)", 1e-4, 1e-4, 1e-5, 3e-4),
        ("Mamba leads P1 (1e-5/1e-4)", 1e-5, 1e-4, 1e-5, 3e-4),
        ("Both low P1 (1e-5/1e-5)", 1e-5, 1e-5, 1e-5, 3e-4),
    ]

    for name, p1_s, p1_m, p2_s, p2_m in configs:
        torch.manual_seed(42)
        torch.cuda.empty_cache()
        model = create_hybrid_model(tokenizer.vocab_size, device)
        results[name] = train_phased(name, model, train_loader, val_loader, device,
                                     p1_s, p1_m, p2_s, p2_m)
        del model

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for name, ppl in results.items():
        print(f"  {name:40s}: {ppl:.1f} PPL")

    print(f"\nReference:")
    print(f"  Phased (multi-seed mean): 157.5 PPL")

    best_ppl = min(results.values())
    best_name = [k for k, v in results.items() if v == best_ppl][0]
    print(f"\nBest: {best_name} = {best_ppl:.1f} PPL")


if __name__ == "__main__":
    main()
