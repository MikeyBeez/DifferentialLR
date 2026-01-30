#!/usr/bin/env python3
"""
Extended Differential Learning Rate Training for 4 Softmax + 4 Mamba

Based on initial results:
- Best PPL (173.0) at epoch 6 during Mamba-Catchup
- Overfitting after epoch 6

This experiment tries:
1. Longer phases with LR decay
2. Extended coordination with very low LR
3. Total 20 epochs (vs 10)
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
    """
    embed_params = []
    softmax_params = []
    mamba_params = []

    embed_params.extend(list(model.token_embedding.parameters()))
    embed_params.extend(list(model.position_embedding.parameters()))

    for i, block in enumerate(model.blocks):
        if i < 4:
            softmax_params.extend(list(block.parameters()))
        else:
            mamba_params.extend(list(block.parameters()))

    softmax_params.extend(list(model.final_norm.parameters()))

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


def train_extended(model, train_loader, val_loader, device, total_epochs=20):
    """
    Extended differential LR training with more phases and LR decay.

    Strategy:
    1. Softmax-Lead: 4 epochs, decreasing LR
    2. Mamba-Catchup: 6 epochs, decreasing LR
    3. Coordination: 6 epochs, very low LR with decay
    4. Fine-tune: 4 epochs, minimal LR
    """
    print("\n" + "="*60)
    print("EXTENDED DIFFERENTIAL LR: 20 Epochs")
    print("="*60)

    embed_params, softmax_params, mamba_params = get_layer_params(model)

    print(f"\nParameter groups:")
    print(f"  Embeddings: {sum(p.numel() for p in embed_params):,}")
    print(f"  Softmax (layers 0-3): {sum(p.numel() for p in softmax_params):,}")
    print(f"  Mamba (layers 4-7): {sum(p.numel() for p in mamba_params):,}")

    optimizer = torch.optim.AdamW([
        {'params': embed_params, 'lr': 1e-4, 'weight_decay': 0.0, 'name': 'embeddings'},
        {'params': softmax_params, 'lr': 1e-3, 'weight_decay': 0.1, 'name': 'softmax'},
        {'params': mamba_params, 'lr': 1e-3, 'weight_decay': 0.1, 'name': 'mamba'},
    ])

    scaler = torch.amp.GradScaler('cuda')

    # Extended phases with LR decay within phases
    # Format: (name, epochs, softmax_lr, mamba_lr, embed_lr)
    phases = [
        # Phase 1: Softmax establishes features
        ("Softmax-Lead", 4, 3e-4, 1e-5, 1e-4),
        # Phase 2: Mamba catches up (longer, since it needs more time)
        ("Mamba-Catchup", 6, 1e-5, 3e-4, 5e-5),
        # Phase 3: Coordination with moderate LR
        ("Coordinate", 6, 5e-5, 5e-5, 3e-5),
        # Phase 4: Final fine-tuning with very low LR
        ("Fine-Tune", 4, 1e-5, 1e-5, 1e-5),
    ]

    best_ppl = float('inf')
    best_epoch = 0
    start_time = time.time()
    global_epoch = 0
    history = []

    for phase_name, num_epochs, softmax_lr, mamba_lr, embed_lr in phases:
        print(f"\n{'='*50}")
        print(f"PHASE: {phase_name} ({num_epochs} epochs)")
        print(f"  Embed LR: {embed_lr:.0e}")
        print(f"  Softmax LR: {softmax_lr:.0e}")
        print(f"  Mamba LR: {mamba_lr:.0e}")
        print("="*50)

        # Set learning rates
        optimizer.param_groups[0]['lr'] = embed_lr
        optimizer.param_groups[1]['lr'] = softmax_lr
        optimizer.param_groups[2]['lr'] = mamba_lr

        for epoch in range(num_epochs):
            train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
            train_ppl = math.exp(train_loss)
            eval_ppl = evaluate(model, val_loader, device)

            marker = "*" if eval_ppl < best_ppl else ""
            if eval_ppl < best_ppl:
                best_ppl = eval_ppl
                best_epoch = global_epoch + 1

            global_epoch += 1
            elapsed = (time.time() - start_time) / 60

            history.append({
                'epoch': global_epoch,
                'phase': phase_name,
                'train_ppl': train_ppl,
                'eval_ppl': eval_ppl,
            })

            print(
                f"Epoch {global_epoch:2d} ({phase_name[:7]:7s}) | "
                f"Train PPL: {train_ppl:7.1f} | Eval PPL: {eval_ppl:7.1f} {marker} | "
                f"{elapsed:.1f}m"
            )

    return best_ppl, best_epoch, history


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
    print("EXTENDED DIFFERENTIAL LR TRAINING (20 EPOCHS)")
    print("="*70)
    print("\nGoal: Beat previous best of 173.0 PPL with longer training")
    print("Strategy: 4 phases with LR decay to prevent overfitting")

    # Load data
    print("\nLoading data...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_loader, val_loader = get_wikitext_data(
        tokenizer, seq_length=512, batch_size=8
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    model = create_hybrid_model(tokenizer.vocab_size, device)
    num_params = model.get_num_params()
    print(f"Parameters: {num_params:,}")

    # Extended training
    best_ppl, best_epoch, history = train_extended(
        model, train_loader, val_loader, device, total_epochs=20
    )

    # Benchmark
    speed = benchmark_speed(model, device)

    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"Best PPL: {best_ppl:.1f} at epoch {best_epoch}")
    print(f"Speed: {speed:.0f} tokens/sec")

    print("\n" + "="*70)
    print("COMPARISON TO PREVIOUS RESULTS")
    print("="*70)
    print(f"  Previous 10-epoch uniform LR:     PPL 186.7")
    print(f"  Previous 10-epoch differential:   PPL 173.0")
    print(f"  This 20-epoch extended:           PPL {best_ppl:.1f}")

    improvement_vs_10 = (173.0 - best_ppl) / 173.0 * 100
    print(f"\n  vs 10-epoch differential: {improvement_vs_10:+.1f}%")

    # Print phase-by-phase summary
    print("\n" + "="*70)
    print("TRAINING HISTORY BY PHASE")
    print("="*70)

    current_phase = None
    for h in history:
        if h['phase'] != current_phase:
            current_phase = h['phase']
            print(f"\n{current_phase}:")
        print(f"  Epoch {h['epoch']:2d}: Train {h['train_ppl']:7.1f}, Eval {h['eval_ppl']:7.1f}")


if __name__ == "__main__":
    main()
