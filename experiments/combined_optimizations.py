#!/usr/bin/env python3
"""
Combined optimizations: Lower softmax LR + Phases + mHC

Tests whether combining all three approaches gives better results.
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

from src.config import ModelConfig
from src.model import create_model


class ManifoldConstrainedConnection(nn.Module):
    """Norm-preserving residual mixing."""
    def __init__(self, dim, init_theta=0.25):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(init_theta * math.pi))

    def forward(self, x, residual):
        x_norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        r_norm = residual.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        avg_norm = (x_norm + r_norm) / 2
        x_unit = x / x_norm
        r_unit = residual / r_norm
        mixed = torch.cos(self.theta) * x_unit + torch.sin(self.theta) * r_unit
        return mixed * avg_norm


class mHCModel(nn.Module):
    """Model with mHC applied to transition between softmax and mamba."""
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        # Single mHC at the softmax→mamba boundary (after layer 3)
        self.mhc = ManifoldConstrainedConnection(base_model.config.hidden_dim)

    def forward(self, input_ids, labels=None, chunk_size=64):
        batch_size, seq_length = input_ids.shape
        positions = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.base_model.token_embedding(input_ids) + self.base_model.position_embedding(positions)
        x = self.base_model.embed_dropout(x)

        for i, block in enumerate(self.base_model.blocks):
            residual = x
            x, _ = block(x, chunk_size)

            # Apply mHC at softmax→mamba transition
            if i == 3:
                x = self.mhc(x, residual)

        x = self.base_model.final_norm(x)
        logits = self.base_model.lm_head(x)

        result = {"logits": logits}
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            result["loss"] = loss
        return result


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
        return torch.tensor([tokens[i:i+seq_len] for i in range(0, len(tokens)-seq_len, seq_len)])

    return (DataLoader(create_sequences(all_tokens, seq_length), batch_size=batch_size, shuffle=True),
            DataLoader(create_sequences(val_tokens, seq_length), batch_size=batch_size, shuffle=False))


def get_layer_params(model):
    """Get params, handling both regular and mHC-wrapped models."""
    base = model.base_model if hasattr(model, 'base_model') else model

    embed_params, softmax_params, mamba_params = [], [], []
    embed_params.extend(list(base.token_embedding.parameters()))
    embed_params.extend(list(base.position_embedding.parameters()))

    for i, block in enumerate(base.blocks):
        if i < 4:
            softmax_params.extend(list(block.parameters()))
        else:
            mamba_params.extend(list(block.parameters()))

    softmax_params.extend(list(base.final_norm.parameters()))

    # Add mHC params to mamba group if present
    if hasattr(model, 'mhc'):
        mamba_params.extend(list(model.mhc.parameters()))

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


def train_phased(model, train_loader, val_loader, device,
                 phase1_softmax_lr, phase1_mamba_lr,
                 phase2_softmax_lr, phase2_mamba_lr,
                 phase1_epochs=4, phase2_epochs=4):
    """Train with phased LRs."""
    embed_params, softmax_params, mamba_params = get_layer_params(model)

    optimizer = torch.optim.AdamW([
        {'params': embed_params, 'lr': 1e-4, 'weight_decay': 0.0},
        {'params': softmax_params, 'lr': phase1_softmax_lr, 'weight_decay': 0.1},
        {'params': mamba_params, 'lr': phase1_mamba_lr, 'weight_decay': 0.1},
    ])
    scaler = torch.amp.GradScaler('cuda')

    print("  Phase 1:")
    for epoch in range(phase1_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
        eval_ppl = evaluate(model, val_loader, device)
        print(f"    Epoch {epoch+1}: Train {math.exp(train_loss):.1f}, Eval {eval_ppl:.1f}")

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
    print("COMBINED OPTIMIZATIONS TEST")
    print("=" * 70)
    print("\nTesting: Lower softmax LR + Phases + mHC")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_loader, val_loader = get_wikitext_data(tokenizer)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    results = {}

    # 1. Original phased (baseline)
    print("\n" + "=" * 50)
    print("1. Original Phased (3e-4/1e-5 → 1e-5/3e-4)")
    print("=" * 50)
    torch.manual_seed(42)
    model = create_hybrid_model(tokenizer.vocab_size, device)
    results["Original phased"] = train_phased(
        model, train_loader, val_loader, device,
        3e-4, 1e-5, 1e-5, 3e-4
    )
    del model
    torch.cuda.empty_cache()

    # 2. Lower softmax LR in Phase 1
    print("\n" + "=" * 50)
    print("2. Lower Softmax P1 (1e-4/1e-5 → 1e-5/3e-4)")
    print("=" * 50)
    torch.manual_seed(42)
    model = create_hybrid_model(tokenizer.vocab_size, device)
    results["Lower softmax P1"] = train_phased(
        model, train_loader, val_loader, device,
        1e-4, 1e-5, 1e-5, 3e-4
    )
    del model
    torch.cuda.empty_cache()

    # 3. Original phased + mHC
    print("\n" + "=" * 50)
    print("3. Original Phased + mHC")
    print("=" * 50)
    torch.manual_seed(42)
    base_model = create_hybrid_model(tokenizer.vocab_size, device)
    model = mHCModel(base_model)
    results["Phased + mHC"] = train_phased(
        model, train_loader, val_loader, device,
        3e-4, 1e-5, 1e-5, 3e-4
    )
    del model, base_model
    torch.cuda.empty_cache()

    # 4. Lower softmax + mHC
    print("\n" + "=" * 50)
    print("4. Lower Softmax P1 + mHC")
    print("=" * 50)
    torch.manual_seed(42)
    base_model = create_hybrid_model(tokenizer.vocab_size, device)
    model = mHCModel(base_model)
    results["Lower softmax + mHC"] = train_phased(
        model, train_loader, val_loader, device,
        1e-4, 1e-5, 1e-5, 3e-4
    )
    del model, base_model
    torch.cuda.empty_cache()

    # 5. Very low softmax + mHC
    print("\n" + "=" * 50)
    print("5. Very Low Softmax (5e-5) + mHC")
    print("=" * 50)
    torch.manual_seed(42)
    base_model = create_hybrid_model(tokenizer.vocab_size, device)
    model = mHCModel(base_model)
    results["Very low softmax + mHC"] = train_phased(
        model, train_loader, val_loader, device,
        5e-5, 1e-5, 1e-5, 3e-4
    )
    del model, base_model
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for name, ppl in results.items():
        print(f"  {name:30s}: {ppl:.1f} PPL")

    print(f"\nReference:")
    print(f"  Phased (multi-seed mean): 157.5 PPL")
    print(f"  Uniform LR:               191.5 PPL")

    best_ppl = min(results.values())
    best_name = [k for k, v in results.items() if v == best_ppl][0]

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"\nBest configuration: {best_name} = {best_ppl:.1f} PPL")

    if best_ppl < 155:
        print("Combined optimizations beat original phased training!")
    elif best_ppl < 160:
        print("Similar to original - combinations don't add much.")
    else:
        print("Something went wrong - check implementation.")


if __name__ == "__main__":
    main()
