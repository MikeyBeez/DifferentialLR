#!/usr/bin/env python3
"""Quick test of the fixed improvements."""

import sys
sys.path.insert(0, '/home/bee/Code/LinearAttention')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import time
import math

from src.config import ModelConfig
from src.model import ChunkedTransformerModel


def get_wikitext_data(tokenizer, seq_length=512, batch_size=8):
    """Load WikiText-2 dataset."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    val_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=False, padding=False)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    val_tokenized = val_dataset.map(tokenize, batched=True, remove_columns=["text"])

    all_tokens = []
    for item in tokenized:
        all_tokens.extend(item["input_ids"])

    val_tokens = []
    for item in val_tokenized:
        val_tokens.extend(item["input_ids"])

    def create_sequences(tokens, seq_len):
        sequences = []
        for i in range(0, len(tokens) - seq_len, seq_len):
            sequences.append(tokens[i:i + seq_len])
        return torch.tensor(sequences)

    train_data = create_sequences(all_tokens, seq_length)
    val_data = create_sequences(val_tokens, seq_length)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, scaler, device, max_steps=None):
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        if max_steps and batch_idx >= max_steps:
            break

        batch = batch.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            outputs = model(batch, labels=batch)
            loss = outputs["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
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
            outputs = model(batch, labels=batch)
            loss = outputs["loss"]
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return math.exp(avg_loss)


def run_experiment(config_name, config, train_loader, val_loader, device, num_epochs=5):
    print(f"\n{'='*60}", flush=True)
    print(f"Running: {config_name}", flush=True)
    print(f"{'='*60}", flush=True)

    model = ChunkedTransformerModel(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}", flush=True)

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
        print(f"Epoch {epoch+1:2d} | Train PPL: {train_ppl:7.1f} | Eval PPL: {eval_ppl:7.1f} {marker} | {elapsed:.1f}m", flush=True)

    print(f"\n{config_name} Best Eval PPL: {best_ppl:.1f}", flush=True)
    return best_ppl


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    print("Loading data...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_loader, val_loader = get_wikitext_data(tokenizer, seq_length=512, batch_size=8)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}", flush=True)

    base_config = {
        "hidden_dim": 512,
        "num_layers": 8,
        "num_heads": 8,
        "head_dim": 64,
        "intermediate_dim": 2048,
        "vocab_size": tokenizer.vocab_size,
        "max_seq_length": 1024,
        "dropout": 0.1,
        "chunk_size": 64,
        "decay": 0.9,
        "use_decay": True,
    }

    results = {}

    # Test Learned Gating (was broken before)
    config = ModelConfig(**base_config, feature_type="elu", use_learned_gate=True)
    results["Learned Gate"] = run_experiment(
        "Learned Gating", config, train_loader, val_loader, device, num_epochs=5
    )

    # Test L2 Normalization (was NaN before)
    config = ModelConfig(**base_config, feature_type="l2")
    results["L2 Norm"] = run_experiment(
        "L2 Normalization", config, train_loader, val_loader, device, num_epochs=5
    )

    # Summary
    print("\n" + "="*60, flush=True)
    print("SUMMARY", flush=True)
    print("="*60, flush=True)
    for name, ppl in sorted(results.items(), key=lambda x: x[1]):
        print(f"{name:30s}: PPL {ppl:7.1f}", flush=True)


if __name__ == "__main__":
    main()
