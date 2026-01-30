#!/usr/bin/env python3
"""
Gated Skip Connection on Uniformly-Trained Model

Tests whether gated skip connections can rescue a poorly-trained hybrid model.

Hypothesis: With uniform LR training, mamba layers are undertrained.
A learned gate should discover this and bypass them (gate → 0).
"""

import sys
sys.path.insert(0, '/home/bee/Code/LinearAttention')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import math
import copy

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


class GatedSkipWrapper(nn.Module):
    """
    Wraps a trained model and adds a gated skip connection.
    """
    def __init__(self, base_model, skip_from=4, skip_to=8, init_gate_logit=3.0):
        super().__init__()
        self.base_model = base_model
        self.skip_from = skip_from
        self.skip_to = skip_to
        self.gate_logit = nn.Parameter(torch.tensor(init_gate_logit))
        self.config = base_model.config

    def get_gate_value(self):
        return torch.sigmoid(self.gate_logit).item()

    def forward(self, input_ids, labels=None, chunk_size=64):
        batch_size, seq_length = input_ids.shape
        positions = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.base_model.token_embedding(input_ids) + self.base_model.position_embedding(positions)
        x = self.base_model.embed_dropout(x)

        saved_repr = None

        for i, block in enumerate(self.base_model.blocks):
            x, _ = block(x, chunk_size)

            if i == self.skip_from - 1:
                saved_repr = x.clone()

            if i == self.skip_to - 1:
                gate = torch.sigmoid(self.gate_logit)
                x = gate * x + (1 - gate) * saved_repr

        x = self.base_model.final_norm(x)
        logits = self.base_model.lm_head(x)

        result = {"logits": logits}

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            result["loss"] = loss

        return result

    def parameters(self):
        for p in self.base_model.parameters():
            yield p
        yield self.gate_logit


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


def create_hybrid_model(vocab_size, device):
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


def train_uniform(model, train_loader, val_loader, device, num_epochs=10):
    """Train with uniform LR (the bad way)."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    scaler = torch.amp.GradScaler('cuda')

    best_ppl = float('inf')
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
        eval_ppl = evaluate(model, val_loader, device)
        best_ppl = min(best_ppl, eval_ppl)
        print(f"  Epoch {epoch+1}: Train {math.exp(train_loss):.1f}, Eval {eval_ppl:.1f}", flush=True)

    return best_ppl


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    print("\n" + "="*70, flush=True)
    print("GATED SKIP ON UNIFORMLY-TRAINED MODEL", flush=True)
    print("="*70, flush=True)
    print("\nHypothesis: Uniform LR undertrains mamba. Gate should learn to bypass.", flush=True)

    # Load data
    print("\nLoading data...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_loader, val_loader = get_wikitext_data(tokenizer)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}", flush=True)

    # Step 1: Train with UNIFORM LR (the bad way)
    print("\n" + "="*50, flush=True)
    print("STEP 1: Train 4S+4M with UNIFORM LR (bad training)", flush=True)
    print("="*50, flush=True)

    torch.manual_seed(42)
    base_model = create_hybrid_model(tokenizer.vocab_size, device)
    base_ppl = train_uniform(base_model, train_loader, val_loader, device, num_epochs=10)

    base_final_ppl = evaluate(base_model, val_loader, device)
    print(f"\nUniform-trained model final PPL: {base_final_ppl:.1f}", flush=True)

    # Step 2: Add gated skip and train gate only
    print("\n" + "="*50, flush=True)
    print("STEP 2: Add gated skip, train ONLY gate", flush=True)
    print("="*50, flush=True)

    # Start with gate at 0.5 (logit=0) to be neutral
    gated_model_a = GatedSkipWrapper(
        copy.deepcopy(base_model), skip_from=4, skip_to=8, init_gate_logit=0.0
    )

    initial_gate = gated_model_a.get_gate_value()
    print(f"Initial gate value: {initial_gate:.4f} (neutral start)", flush=True)

    initial_ppl = evaluate(gated_model_a, val_loader, device)
    print(f"Initial PPL with gate=0.5: {initial_ppl:.1f}", flush=True)

    # Train only the gate
    optimizer_a = torch.optim.Adam([gated_model_a.gate_logit], lr=0.5)
    scaler_a = torch.amp.GradScaler('cuda')

    print("\nTraining gate only for 5 epochs...", flush=True)
    for epoch in range(5):
        gated_model_a.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer_a.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = gated_model_a(batch, labels=batch, chunk_size=64)
                loss = outputs["loss"]
            scaler_a.scale(loss).backward()
            scaler_a.step(optimizer_a)
            scaler_a.update()

        eval_ppl = evaluate(gated_model_a, val_loader, device)
        gate_val = gated_model_a.get_gate_value()
        print(f"  Epoch {epoch+1}: Eval PPL {eval_ppl:.1f}, Gate = {gate_val:.4f}", flush=True)

    final_gate_a = gated_model_a.get_gate_value()
    final_ppl_a = evaluate(gated_model_a, val_loader, device)

    # Step 3: Add gated skip and train everything
    print("\n" + "="*50, flush=True)
    print("STEP 3: Add gated skip, train ALL params", flush=True)
    print("="*50, flush=True)

    gated_model_b = GatedSkipWrapper(
        copy.deepcopy(base_model), skip_from=4, skip_to=8, init_gate_logit=0.0
    )

    optimizer_b = torch.optim.AdamW([
        {'params': gated_model_b.base_model.parameters(), 'lr': 1e-5},
        {'params': [gated_model_b.gate_logit], 'lr': 0.5},
    ], weight_decay=0.01)
    scaler_b = torch.amp.GradScaler('cuda')

    print("Training all params for 5 epochs...", flush=True)
    best_ppl_b = float('inf')
    for epoch in range(5):
        gated_model_b.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer_b.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = gated_model_b(batch, labels=batch, chunk_size=64)
                loss = outputs["loss"]
            scaler_b.scale(loss).backward()
            scaler_b.unscale_(optimizer_b)
            torch.nn.utils.clip_grad_norm_(gated_model_b.parameters(), 1.0)
            scaler_b.step(optimizer_b)
            scaler_b.update()

        eval_ppl = evaluate(gated_model_b, val_loader, device)
        gate_val = gated_model_b.get_gate_value()
        best_ppl_b = min(best_ppl_b, eval_ppl)
        print(f"  Epoch {epoch+1}: Eval PPL {eval_ppl:.1f}, Gate = {gate_val:.4f}", flush=True)

    final_gate_b = gated_model_b.get_gate_value()
    final_ppl_b = evaluate(gated_model_b, val_loader, device)

    # Summary
    print("\n" + "="*70, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("="*70, flush=True)

    print(f"\nBase model (uniform LR, no skip): PPL = {base_final_ppl:.1f}", flush=True)

    print(f"\nGated skip (gate only trained):", flush=True)
    print(f"  Final PPL:  {final_ppl_a:.1f}", flush=True)
    print(f"  Final gate: {final_gate_a:.4f}", flush=True)

    print(f"\nGated skip (all params trained):", flush=True)
    print(f"  Final PPL:  {final_ppl_b:.1f} (best: {best_ppl_b:.1f})", flush=True)
    print(f"  Final gate: {final_gate_b:.4f}", flush=True)

    print("\n" + "="*70, flush=True)
    print("INTERPRETATION", flush=True)
    print("="*70, flush=True)

    if final_gate_a < 0.3:
        print(f"Gate-only went LOW ({final_gate_a:.2f}): Model prefers to BYPASS mamba", flush=True)
        print("  → Confirms mamba was undertrained with uniform LR", flush=True)
    elif final_gate_a > 0.7:
        print(f"Gate-only stayed HIGH ({final_gate_a:.2f}): Model wants to KEEP mamba", flush=True)
        print("  → Mamba may be more useful than expected", flush=True)
    else:
        print(f"Gate-only is INTERMEDIATE ({final_gate_a:.2f}): Model is hedging", flush=True)

    improvement_a = (base_final_ppl - final_ppl_a) / base_final_ppl * 100
    improvement_b = (base_final_ppl - best_ppl_b) / base_final_ppl * 100

    print(f"\nPPL improvement from gated skip:", flush=True)
    print(f"  Gate-only: {improvement_a:+.1f}% ({base_final_ppl:.1f} → {final_ppl_a:.1f})", flush=True)
    print(f"  Full train: {improvement_b:+.1f}% ({base_final_ppl:.1f} → {best_ppl_b:.1f})", flush=True)

    print("\n" + "="*70, flush=True)
    print("COMPARISON: DIFFERENTIAL LR vs GATED SKIP RESCUE", flush=True)
    print("="*70, flush=True)
    print(f"  Differential LR training:     ~157.5 PPL (from multi-seed)", flush=True)
    print(f"  Uniform LR + gated skip:      {best_ppl_b:.1f} PPL", flush=True)

    if best_ppl_b > 165:
        print("\n  → Gated skip helps but doesn't match proper training", flush=True)
    else:
        print("\n  → Gated skip successfully rescues the model!", flush=True)


if __name__ == "__main__":
    main()
