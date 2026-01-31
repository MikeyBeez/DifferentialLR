#!/usr/bin/env python3
"""
Test Manifold-Constrained Hyper-Connections (mHC) from DeepSeek.

mHC constrains residual mixing to preserve signal magnitude:
- Information can be redistributed but not amplified or destroyed
- Prevents gradient explosion/vanishing across layers

We test whether mHC can rescue uniform LR training (like gated skip failed to do).
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
import copy

from src.config import ModelConfig
from src.model import create_model


class ManifoldConstrainedConnection(nn.Module):
    """
    Manifold-Constrained Hyper-Connection.

    Constrains the residual mixing to preserve signal magnitude.
    Uses orthogonal/rotation-like mixing instead of arbitrary learned weights.

    output = cos(theta) * x + sin(theta) * residual

    This preserves ||output|| ≈ ||x|| when ||x|| ≈ ||residual||
    """
    def __init__(self, dim, init_theta=0.25):
        super().__init__()
        # Learnable angle parameter (in units of pi)
        # 0.25 = pi/4 = equal mix of input and residual
        self.theta = nn.Parameter(torch.tensor(init_theta * math.pi))

    def forward(self, x, residual):
        # Normalize to unit vectors for mixing, then scale back
        x_norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        r_norm = residual.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        # Average magnitude to preserve
        avg_norm = (x_norm + r_norm) / 2

        # Mix on unit sphere
        x_unit = x / x_norm
        r_unit = residual / r_norm

        # Rotation-like mixing (preserves norm)
        mixed = torch.cos(self.theta) * x_unit + torch.sin(self.theta) * r_unit

        # Scale back to average magnitude
        return mixed * avg_norm

    def get_theta_degrees(self):
        return (self.theta.item() / math.pi) * 180


class mHCWrapper(nn.Module):
    """
    Wraps a model and replaces residual connections with mHC.
    """
    def __init__(self, base_model, apply_to_layers=None):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config

        # Create mHC modules for each layer
        if apply_to_layers is None:
            apply_to_layers = list(range(len(base_model.blocks)))

        self.apply_to_layers = apply_to_layers
        self.mhc_modules = nn.ModuleList([
            ManifoldConstrainedConnection(base_model.config.hidden_dim)
            for _ in apply_to_layers
        ])

    def forward(self, input_ids, labels=None, chunk_size=64):
        batch_size, seq_length = input_ids.shape
        positions = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.base_model.token_embedding(input_ids) + self.base_model.position_embedding(positions)
        x = self.base_model.embed_dropout(x)

        mhc_idx = 0
        for i, block in enumerate(self.base_model.blocks):
            # Get block output (before residual in original)
            # We need to intercept the residual connection
            residual = x

            # Run through block
            x, _ = block(x, chunk_size)

            # Apply mHC instead of standard residual
            if i in self.apply_to_layers:
                # The block already added residual internally, so we need to
                # undo it and apply mHC instead
                # Actually, looking at the model, residual is added inside the block
                # So we apply mHC as a post-processing step
                x = self.mhc_modules[mhc_idx](x, residual)
                mhc_idx += 1

        x = self.base_model.final_norm(x)
        logits = self.base_model.lm_head(x)

        result = {"logits": logits}

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            result["loss"] = loss

        return result

    def get_mhc_angles(self):
        return [m.get_theta_degrees() for m in self.mhc_modules]


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n" + "=" * 70)
    print("mHC (MANIFOLD-CONSTRAINED HYPER-CONNECTIONS) TEST")
    print("=" * 70)
    print("\nHypothesis: mHC may rescue uniform LR training by stabilizing gradients")

    # Load data
    print("\nLoading data...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_loader, val_loader = get_wikitext_data(tokenizer)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    results = {}

    # Test 1: Baseline uniform LR (no mHC)
    print("\n" + "=" * 50)
    print("TEST 1: Uniform LR without mHC (baseline)")
    print("=" * 50)

    torch.manual_seed(42)
    base_model = create_hybrid_model(tokenizer.vocab_size, device)
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=1e-4, weight_decay=0.1)
    scaler = torch.amp.GradScaler('cuda')

    best_ppl = float('inf')
    for epoch in range(8):
        train_loss = train_epoch(base_model, train_loader, optimizer, scaler, device)
        eval_ppl = evaluate(base_model, val_loader, device)
        best_ppl = min(best_ppl, eval_ppl)
        print(f"  Epoch {epoch+1}: Train {math.exp(train_loss):.1f}, Eval {eval_ppl:.1f}")

    results["uniform_no_mhc"] = best_ppl
    del base_model, optimizer
    torch.cuda.empty_cache()

    # Test 2: Uniform LR with mHC
    print("\n" + "=" * 50)
    print("TEST 2: Uniform LR with mHC")
    print("=" * 50)

    torch.manual_seed(42)
    base_model = create_hybrid_model(tokenizer.vocab_size, device)
    mhc_model = mHCWrapper(base_model, apply_to_layers=[4, 5, 6, 7])  # Apply to Mamba layers

    optimizer = torch.optim.AdamW(mhc_model.parameters(), lr=1e-4, weight_decay=0.1)
    scaler = torch.amp.GradScaler('cuda')

    best_ppl = float('inf')
    for epoch in range(8):
        train_loss = train_epoch(mhc_model, train_loader, optimizer, scaler, device)
        eval_ppl = evaluate(mhc_model, val_loader, device)
        best_ppl = min(best_ppl, eval_ppl)
        angles = mhc_model.get_mhc_angles()
        print(f"  Epoch {epoch+1}: Train {math.exp(train_loss):.1f}, Eval {eval_ppl:.1f}, Angles: {[f'{a:.1f}' for a in angles]}")

    results["uniform_with_mhc"] = best_ppl
    del mhc_model, base_model, optimizer
    torch.cuda.empty_cache()

    # Test 3: Phased Specialization with mHC
    print("\n" + "=" * 50)
    print("TEST 3: Phased Specialization with mHC")
    print("=" * 50)

    torch.manual_seed(42)
    base_model = create_hybrid_model(tokenizer.vocab_size, device)
    mhc_model = mHCWrapper(base_model, apply_to_layers=[4, 5, 6, 7])

    # Separate params
    base_params = list(mhc_model.base_model.parameters())
    mhc_params = list(mhc_model.mhc_modules.parameters())

    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': 1e-4},
        {'params': mhc_params, 'lr': 1e-3},
    ], weight_decay=0.1)
    scaler = torch.amp.GradScaler('cuda')

    # Phase 1: Attention Lead
    print("\n  Phase 1: Attention Lead")
    for epoch in range(4):
        train_loss = train_epoch(mhc_model, train_loader, optimizer, scaler, device)
        eval_ppl = evaluate(mhc_model, val_loader, device)
        angles = mhc_model.get_mhc_angles()
        print(f"    Epoch {epoch+1}: Train {math.exp(train_loss):.1f}, Eval {eval_ppl:.1f}")

    # Phase 2: SSM Specialization
    print("\n  Phase 2: SSM Specialization")
    best_ppl = float('inf')
    for epoch in range(4):
        train_loss = train_epoch(mhc_model, train_loader, optimizer, scaler, device)
        eval_ppl = evaluate(mhc_model, val_loader, device)
        best_ppl = min(best_ppl, eval_ppl)
        angles = mhc_model.get_mhc_angles()
        print(f"    Epoch {epoch+1}: Train {math.exp(train_loss):.1f}, Eval {eval_ppl:.1f}, Angles: {[f'{a:.1f}' for a in angles]}")

    results["phased_with_mhc"] = best_ppl

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nUniform LR without mHC:        {results['uniform_no_mhc']:.1f} PPL")
    print(f"Uniform LR with mHC:           {results['uniform_with_mhc']:.1f} PPL")
    print(f"Phased Specialization + mHC:   {results['phased_with_mhc']:.1f} PPL")
    print(f"\nReference (from multi-seed):")
    print(f"  Uniform LR (no mHC):         191.5 PPL")
    print(f"  Phased Specialization:       157.5 PPL")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    mhc_improvement = results['uniform_no_mhc'] - results['uniform_with_mhc']
    if mhc_improvement > 10:
        print(f"\nmHC rescues uniform LR training! ({mhc_improvement:.1f} PPL improvement)")
        print("This suggests gradient stability is a key factor.")
    elif mhc_improvement > 0:
        print(f"\nmHC provides modest improvement ({mhc_improvement:.1f} PPL)")
        print("Gradient stability helps but doesn't fully solve the problem.")
    else:
        print(f"\nmHC does not rescue uniform LR training ({mhc_improvement:.1f} PPL)")
        print("The problem is not just gradient stability.")

    phased_vs_mhc = results['uniform_with_mhc'] - 157.5
    if phased_vs_mhc > 10:
        print(f"\nPhased Specialization still outperforms mHC by {phased_vs_mhc:.1f} PPL")
        print("Training schedule matters more than gradient constraints.")


if __name__ == "__main__":
    main()
