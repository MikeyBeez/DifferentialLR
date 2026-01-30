#!/usr/bin/env python3
"""
Coordination LR Ablation

Tests whether the coordination phase failure is due to:
1. Fundamental interference (our claim)
2. Just bad LR choice (alternative explanation)

We test coordination with progressively smaller LRs:
- 1e-4 (original)
- 5e-5
- 1e-5
- 5e-6

If very small LRs still hurt, the interference claim is stronger.
If very small LRs help or are neutral, it's an optimization artifact.
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


def create_hybrid_model(vocab_size, device):
    """Create 4S + 4M model."""
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


def train_with_coordination(model, train_loader, val_loader, device, coord_lr):
    """
    Train with differential LR + coordination phase at specified LR.

    Returns: (best_ppl_before_coord, best_ppl_after_coord, history)
    """
    embed_params, softmax_params, mamba_params = get_layer_params(model)

    optimizer = torch.optim.AdamW([
        {'params': embed_params, 'lr': 1e-4, 'weight_decay': 0.0},
        {'params': softmax_params, 'lr': 3e-4, 'weight_decay': 0.1},
        {'params': mamba_params, 'lr': 1e-5, 'weight_decay': 0.1},
    ])
    scaler = torch.amp.GradScaler('cuda')

    history = []
    best_before_coord = float('inf')
    best_after_coord = float('inf')
    global_epoch = 0

    # Phase 1: Softmax Lead (4 epochs)
    optimizer.param_groups[1]['lr'] = 3e-4
    optimizer.param_groups[2]['lr'] = 1e-5
    for epoch in range(4):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
        eval_ppl = evaluate(model, val_loader, device)
        global_epoch += 1
        history.append(('softmax', global_epoch, math.exp(train_loss), eval_ppl))
        best_before_coord = min(best_before_coord, eval_ppl)

    # Phase 2: Mamba Catchup (5 epochs)
    optimizer.param_groups[1]['lr'] = 1e-5
    optimizer.param_groups[2]['lr'] = 3e-4
    for epoch in range(5):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
        eval_ppl = evaluate(model, val_loader, device)
        global_epoch += 1
        history.append(('mamba', global_epoch, math.exp(train_loss), eval_ppl))
        best_before_coord = min(best_before_coord, eval_ppl)

    ppl_at_coord_start = history[-1][3]

    # Phase 3: Coordination (5 epochs) with specified LR
    optimizer.param_groups[0]['lr'] = coord_lr  # embed
    optimizer.param_groups[1]['lr'] = coord_lr  # softmax
    optimizer.param_groups[2]['lr'] = coord_lr  # mamba
    for epoch in range(5):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
        eval_ppl = evaluate(model, val_loader, device)
        global_epoch += 1
        history.append(('coord', global_epoch, math.exp(train_loss), eval_ppl))
        best_after_coord = min(best_after_coord, eval_ppl)

    return best_before_coord, best_after_coord, ppl_at_coord_start, history


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n" + "="*70)
    print("COORDINATION LR ABLATION")
    print("="*70)
    print("\nQuestion: Is coordination phase failure due to:")
    print("  A) Fundamental interference (our claim)")
    print("  B) Just bad LR choice (alternative)")
    print("\nTest: Try progressively smaller coordination LRs")

    # Load data
    print("\nLoading data...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_loader, val_loader = get_wikitext_data(tokenizer)

    # Test different coordination LRs
    coord_lrs = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
    results = []

    # Also test NO coordination (baseline)
    print("\n" + "="*50)
    print("BASELINE: No coordination phase")
    print("="*50)

    torch.manual_seed(42)
    model = create_hybrid_model(tokenizer.vocab_size, device)
    embed_params, softmax_params, mamba_params = get_layer_params(model)

    optimizer = torch.optim.AdamW([
        {'params': embed_params, 'lr': 1e-4, 'weight_decay': 0.0},
        {'params': softmax_params, 'lr': 3e-4, 'weight_decay': 0.1},
        {'params': mamba_params, 'lr': 1e-5, 'weight_decay': 0.1},
    ])
    scaler = torch.amp.GradScaler('cuda')

    best_no_coord = float('inf')
    # Softmax lead
    optimizer.param_groups[1]['lr'] = 3e-4
    optimizer.param_groups[2]['lr'] = 1e-5
    for epoch in range(4):
        train_epoch(model, train_loader, optimizer, scaler, device)
        eval_ppl = evaluate(model, val_loader, device)
        best_no_coord = min(best_no_coord, eval_ppl)
        print(f"  Epoch {epoch+1} (softmax): {eval_ppl:.1f}")

    # Mamba catchup
    optimizer.param_groups[1]['lr'] = 1e-5
    optimizer.param_groups[2]['lr'] = 3e-4
    for epoch in range(5):
        train_epoch(model, train_loader, optimizer, scaler, device)
        eval_ppl = evaluate(model, val_loader, device)
        best_no_coord = min(best_no_coord, eval_ppl)
        print(f"  Epoch {epoch+5} (mamba): {eval_ppl:.1f}")

    final_no_coord = eval_ppl
    print(f"\nNo coordination: Best PPL = {best_no_coord:.1f}, Final = {final_no_coord:.1f}")

    del model
    torch.cuda.empty_cache()

    # Test each coordination LR
    for coord_lr in coord_lrs:
        print("\n" + "="*50)
        print(f"COORDINATION LR: {coord_lr:.0e}")
        print("="*50)

        torch.manual_seed(42)  # Same seed for fair comparison
        model = create_hybrid_model(tokenizer.vocab_size, device)

        best_before, best_after, ppl_at_start, history = train_with_coordination(
            model, train_loader, val_loader, device, coord_lr
        )

        # Show coordination phase progression
        print("\nCoordination phase epochs:")
        for phase, epoch, train_ppl, eval_ppl in history:
            if phase == 'coord':
                print(f"  Epoch {epoch}: Train {train_ppl:.1f}, Eval {eval_ppl:.1f}")

        final_ppl = history[-1][3]
        change = (final_ppl - ppl_at_start) / ppl_at_start * 100

        results.append({
            'coord_lr': coord_lr,
            'best_before': best_before,
            'ppl_at_coord_start': ppl_at_start,
            'best_after': best_after,
            'final_ppl': final_ppl,
            'change': change,
        })

        print(f"\nPPL at coord start: {ppl_at_start:.1f}")
        print(f"Best after coord: {best_after:.1f}")
        print(f"Final PPL: {final_ppl:.1f}")
        print(f"Change from coord start: {change:+.1f}%")

        del model
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'Coord LR':<12} | {'Before Coord':>12} | {'At Start':>10} | {'After':>10} | {'Change':>10}")
    print("-"*65)

    print(f"{'None':<12} | {best_no_coord:>12.1f} | {final_no_coord:>10.1f} | {'N/A':>10} | {'N/A':>10}")

    for r in results:
        print(f"{r['coord_lr']:.0e}       | {r['best_before']:>12.1f} | {r['ppl_at_coord_start']:>10.1f} | {r['best_after']:>10.1f} | {r['change']:>+9.1f}%")

    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    # Check if any coordination LR helped
    helped = [r for r in results if r['best_after'] < r['ppl_at_coord_start']]
    hurt = [r for r in results if r['best_after'] > r['ppl_at_coord_start']]

    if len(hurt) == len(results):
        print("ALL coordination LRs hurt performance.")
        print("→ Strong evidence for FUNDAMENTAL INTERFERENCE")
        print("→ The claim 'coordination phase is harmful' is supported")
    elif len(helped) > 0:
        best_coord = min(results, key=lambda r: r['best_after'])
        print(f"Some coordination LRs helped. Best: {best_coord['coord_lr']:.0e}")
        print(f"→ The harm was partly an OPTIMIZATION ARTIFACT")
        print(f"→ Optimal coord LR: {best_coord['coord_lr']:.0e}")
    else:
        print("Mixed results - need more investigation")


if __name__ == "__main__":
    main()
