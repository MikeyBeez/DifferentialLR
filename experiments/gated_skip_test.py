#!/usr/bin/env python3
"""
Gated Skip Connection Experiment

Tests adding learned gated skip connections to an already-trained 4S+4M model.

Approach:
1. Train 4S+4M with differential LR (our best method)
2. Add gated skip from layer 4 output to layer 8 output
3. Continue training with gates
4. Analyze what the gates learn

The gate is initialized to ~0.95 (logit=3) so initial behavior matches the trained model.
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


class GatedSkipWrapper(nn.Module):
    """
    Wraps a trained model and adds a gated skip connection
    from after softmax layers to after mamba layers.
    """
    def __init__(self, base_model, skip_from=4, skip_to=8, init_gate_logit=3.0):
        super().__init__()
        self.base_model = base_model
        self.skip_from = skip_from  # Save representation after this layer index
        self.skip_to = skip_to      # Apply skip after this layer index

        # Learned gate parameter - initialized so sigmoid ≈ 0.95
        # This means we start with behavior close to the original model
        self.gate_logit = nn.Parameter(torch.tensor(init_gate_logit))

        # Store config reference
        self.config = base_model.config

    def get_gate_value(self):
        """Return current gate value (0-1)."""
        return torch.sigmoid(self.gate_logit).item()

    def forward(self, input_ids, labels=None, chunk_size=64):
        """
        Forward pass with gated skip connection.
        """
        # Get embeddings (matching base model's forward)
        batch_size, seq_length = input_ids.shape
        positions = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.base_model.token_embedding(input_ids) + self.base_model.position_embedding(positions)
        x = self.base_model.embed_dropout(x)

        saved_repr = None

        # Process through blocks
        for i, block in enumerate(self.base_model.blocks):
            x, _ = block(x, chunk_size)  # block returns (x, intermediates)

            # Save representation after softmax layers
            if i == self.skip_from - 1:  # After layer 3 (0-indexed)
                saved_repr = x.clone()

            # Apply gated skip after mamba layers
            if i == self.skip_to - 1:  # After layer 7 (0-indexed)
                gate = torch.sigmoid(self.gate_logit)
                x = gate * x + (1 - gate) * saved_repr

        # Final norm and output
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
        """Return all parameters including gate."""
        for p in self.base_model.parameters():
            yield p
        yield self.gate_logit

    def get_num_params(self):
        return self.base_model.get_num_params() + 1  # +1 for gate


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


def train_differential(model, train_loader, val_loader, device, stop_at_best=True):
    """
    Train with differential LR, returning best model.
    """
    embed_params, softmax_params, mamba_params = get_layer_params(model)

    optimizer = torch.optim.AdamW([
        {'params': embed_params, 'lr': 1e-4, 'weight_decay': 0.0},
        {'params': softmax_params, 'lr': 3e-4, 'weight_decay': 0.1},
        {'params': mamba_params, 'lr': 1e-5, 'weight_decay': 0.1},
    ])
    scaler = torch.amp.GradScaler('cuda')

    best_ppl = float('inf')
    best_state = None
    epoch = 0

    # Phase 1: Softmax Lead (4 epochs)
    optimizer.param_groups[1]['lr'] = 3e-4
    optimizer.param_groups[2]['lr'] = 1e-5
    for _ in range(4):
        epoch += 1
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
        eval_ppl = evaluate(model, val_loader, device)
        print(f"  Epoch {epoch} (softmax): Train {math.exp(train_loss):.1f}, Eval {eval_ppl:.1f}", flush=True)
        if eval_ppl < best_ppl:
            best_ppl = eval_ppl
            best_state = copy.deepcopy(model.state_dict())

    # Phase 2: Mamba Catchup (4 epochs - stop before overfitting)
    optimizer.param_groups[1]['lr'] = 1e-5
    optimizer.param_groups[2]['lr'] = 3e-4
    for _ in range(4):
        epoch += 1
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
        eval_ppl = evaluate(model, val_loader, device)
        print(f"  Epoch {epoch} (mamba): Train {math.exp(train_loss):.1f}, Eval {eval_ppl:.1f}", flush=True)
        if eval_ppl < best_ppl:
            best_ppl = eval_ppl
            best_state = copy.deepcopy(model.state_dict())

    if stop_at_best and best_state is not None:
        model.load_state_dict(best_state)

    return best_ppl


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    print("\n" + "="*70, flush=True)
    print("GATED SKIP CONNECTION EXPERIMENT", flush=True)
    print("="*70, flush=True)

    # Load data
    print("\nLoading data...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_loader, val_loader = get_wikitext_data(tokenizer)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}", flush=True)

    # Step 1: Train base model with differential LR
    print("\n" + "="*50, flush=True)
    print("STEP 1: Train base 4S+4M with differential LR", flush=True)
    print("="*50, flush=True)

    torch.manual_seed(42)
    base_model = create_hybrid_model(tokenizer.vocab_size, device)
    base_ppl = train_differential(base_model, train_loader, val_loader, device, stop_at_best=True)
    print(f"\nBase model best PPL: {base_ppl:.1f}", flush=True)

    # Evaluate base model
    base_final_ppl = evaluate(base_model, val_loader, device)
    print(f"Base model final PPL: {base_final_ppl:.1f}", flush=True)

    # Step 2: Add gated skip connection
    print("\n" + "="*50, flush=True)
    print("STEP 2: Add gated skip connection", flush=True)
    print("="*50, flush=True)

    # Wrap the trained model with gated skip
    gated_model = GatedSkipWrapper(base_model, skip_from=4, skip_to=8, init_gate_logit=3.0)

    initial_gate = gated_model.get_gate_value()
    print(f"Initial gate value: {initial_gate:.4f} (should be ~0.95)", flush=True)

    # Verify the gated model matches base model performance (gate ≈ 1)
    gated_initial_ppl = evaluate(gated_model, val_loader, device)
    print(f"Gated model initial PPL: {gated_initial_ppl:.1f} (should match base)", flush=True)

    # Step 3: Train with gates - Option A: Only train gate
    print("\n" + "="*50, flush=True)
    print("STEP 3A: Train ONLY the gate parameter", flush=True)
    print("="*50, flush=True)

    # Create fresh gated wrapper for this test
    gated_model_a = GatedSkipWrapper(
        copy.deepcopy(base_model), skip_from=4, skip_to=8, init_gate_logit=3.0
    )

    # Only optimize the gate
    optimizer_a = torch.optim.Adam([gated_model_a.gate_logit], lr=0.1)
    scaler_a = torch.amp.GradScaler('cuda')

    print("Training gate only for 3 epochs...", flush=True)
    for epoch in range(3):
        gated_model_a.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer_a.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = gated_model_a(batch, labels=batch, chunk_size=64)
                loss = outputs["loss"]
            scaler_a.scale(loss).backward()
            scaler_a.step(optimizer_a)
            scaler_a.update()
            total_loss += loss.item()

        eval_ppl = evaluate(gated_model_a, val_loader, device)
        gate_val = gated_model_a.get_gate_value()
        print(f"  Epoch {epoch+1}: Eval PPL {eval_ppl:.1f}, Gate = {gate_val:.4f}", flush=True)

    final_gate_a = gated_model_a.get_gate_value()
    final_ppl_a = evaluate(gated_model_a, val_loader, device)

    # Step 4: Train with gates - Option B: Train everything with low LR
    print("\n" + "="*50, flush=True)
    print("STEP 3B: Train ALL parameters with low LR", flush=True)
    print("="*50, flush=True)

    # Create fresh gated wrapper for this test
    gated_model_b = GatedSkipWrapper(
        copy.deepcopy(base_model), skip_from=4, skip_to=8, init_gate_logit=3.0
    )

    # Optimize all parameters with low LR
    optimizer_b = torch.optim.AdamW([
        {'params': gated_model_b.base_model.parameters(), 'lr': 1e-5},
        {'params': [gated_model_b.gate_logit], 'lr': 0.1},
    ], weight_decay=0.01)
    scaler_b = torch.amp.GradScaler('cuda')

    print("Training all params for 3 epochs...", flush=True)
    best_ppl_b = float('inf')
    for epoch in range(3):
        gated_model_b.train()
        total_loss = 0
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
            total_loss += loss.item()

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

    print(f"\nBase model (no skip):           PPL = {base_final_ppl:.1f}", flush=True)
    print(f"\nGated skip (gate only trained):", flush=True)
    print(f"  Final PPL:  {final_ppl_a:.1f}", flush=True)
    print(f"  Final gate: {final_gate_a:.4f}", flush=True)

    print(f"\nGated skip (all params trained):", flush=True)
    print(f"  Final PPL:  {final_ppl_b:.1f} (best: {best_ppl_b:.1f})", flush=True)
    print(f"  Final gate: {final_gate_b:.4f}", flush=True)

    print("\n" + "="*70, flush=True)
    print("INTERPRETATION", flush=True)
    print("="*70, flush=True)

    if final_gate_a > 0.8 and final_gate_b > 0.8:
        print("Gates stayed high (>0.8): Mamba layers are useful, skip not needed", flush=True)
    elif final_gate_a < 0.2 and final_gate_b < 0.2:
        print("Gates went low (<0.2): Model prefers to bypass Mamba", flush=True)
    elif 0.3 < final_gate_a < 0.7 or 0.3 < final_gate_b < 0.7:
        print("Gates are intermediate: Model is hedging, Mamba partially useful", flush=True)
    else:
        print("Mixed results - gate-only and full training disagree", flush=True)

    improvement_a = (base_final_ppl - final_ppl_a) / base_final_ppl * 100
    improvement_b = (base_final_ppl - best_ppl_b) / base_final_ppl * 100

    print(f"\nPPL change from adding gated skip:", flush=True)
    print(f"  Gate-only training: {improvement_a:+.1f}%", flush=True)
    print(f"  Full training:      {improvement_b:+.1f}%", flush=True)


if __name__ == "__main__":
    main()
