#!/usr/bin/env python3
"""
Test static differential LR (no phases).

Instead of phased training, what if we just set:
- Softmax: low LR (throttled)
- Mamba: high LR (leads)

This tests whether phases are necessary or if a fixed ratio works.
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


def test_config(name, model, train_loader, val_loader, device,
                softmax_lr, mamba_lr, epochs=8):
    embed_params, softmax_params, mamba_params = get_layer_params(model)

    optimizer = torch.optim.AdamW([
        {'params': embed_params, 'lr': 1e-4, 'weight_decay': 0.0},
        {'params': softmax_params, 'lr': softmax_lr, 'weight_decay': 0.1},
        {'params': mamba_params, 'lr': mamba_lr, 'weight_decay': 0.1},
    ])
    scaler = torch.amp.GradScaler('cuda')

    print(f"\n{name}")
    print(f"  Softmax LR: {softmax_lr}, Mamba LR: {mamba_lr}")
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
    print("STATIC DIFFERENTIAL LR TEST")
    print("=" * 70)
    print("\nQuestion: Can we just lower softmax LR instead of using phases?")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_loader, val_loader = get_wikitext_data(tokenizer)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    results = {}

    # Test different static ratios
    configs = [
        ("Uniform (baseline)", 1e-4, 1e-4),
        ("Mamba 3x", 1e-4, 3e-4),
        ("Mamba 10x", 1e-5, 1e-4),
        ("Mamba 30x", 1e-5, 3e-4),
        ("Softmax throttled", 3e-5, 3e-4),
        ("Extreme: Mamba 100x", 1e-6, 1e-4),
    ]

    for name, softmax_lr, mamba_lr in configs:
        torch.manual_seed(42)
        torch.cuda.empty_cache()
        model = create_hybrid_model(tokenizer.vocab_size, device)
        results[name] = test_config(name, model, train_loader, val_loader, device,
                                    softmax_lr, mamba_lr, epochs=8)
        del model

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for name, ppl in results.items():
        print(f"  {name:25s}: {ppl:.1f} PPL")

    print(f"\nReference:")
    print(f"  Phased Specialization:   157.5 PPL")
    print(f"  Uniform LR (multi-seed): 191.5 PPL")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    best_static = min(results.values())
    best_name = [k for k, v in results.items() if v == best_static][0]

    if best_static < 170:
        print(f"\nStatic differential LR works! Best: {best_name} = {best_static:.1f} PPL")
        print("Phases may not be necessary - just throttle softmax.")
    elif best_static < 185:
        print(f"\nStatic helps but doesn't match phased ({best_static:.1f} vs 157.5)")
        print("Phases provide additional benefit beyond static ratio.")
    else:
        print(f"\nStatic differential LR doesn't help much ({best_static:.1f} PPL)")
        print("The phased approach is necessary, not just the ratio.")


if __name__ == "__main__":
    main()
