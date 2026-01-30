#!/usr/bin/env python3
"""Long training comparison: softmax vs linear attention."""

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


class SoftmaxTransformer(nn.Module):
    """Standard transformer with full softmax attention."""

    def __init__(self, vocab_size, hidden_dim=512, num_layers=8, num_heads=8,
                 intermediate_dim=2048, max_seq_len=1024, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=intermediate_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()

        x = self.transformer(x, mask=causal_mask, is_causal=True)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        output = {"logits": logits}

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            output["loss"] = loss

        return output


def get_wikitext_data(tokenizer, seq_length=512, batch_size=8):
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


def train_epoch(model, train_loader, optimizer, scaler, device):
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            outputs = model(batch, labels=batch)
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
            outputs = model(batch, labels=batch)
            loss = outputs["loss"]
        total_loss += loss.item()
        num_batches += 1

    return math.exp(total_loss / num_batches)


def main():
    device = torch.device("cuda")
    num_epochs = 50

    print("Loading data...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_loader, val_loader = get_wikitext_data(tokenizer, seq_length=512, batch_size=8)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}", flush=True)

    # Create both models
    print("\nCreating models...", flush=True)

    # Softmax transformer
    softmax_model = SoftmaxTransformer(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=512,
        num_layers=8,
        num_heads=8,
        intermediate_dim=2048,
        max_seq_len=1024,
        dropout=0.1,
    ).to(device)

    # Linear attention with learned gating
    linear_config = ModelConfig(
        hidden_dim=512,
        num_layers=8,
        num_heads=8,
        head_dim=64,
        intermediate_dim=2048,
        vocab_size=tokenizer.vocab_size,
        max_seq_length=1024,
        dropout=0.1,
        chunk_size=64,
        decay=0.9,
        use_decay=True,
        feature_type="elu",
        use_learned_gate=True,
    )
    linear_model = ChunkedTransformerModel(linear_config).to(device)

    softmax_params = sum(p.numel() for p in softmax_model.parameters())
    linear_params = sum(p.numel() for p in linear_model.parameters())
    print(f"Softmax params: {softmax_params:,}", flush=True)
    print(f"Linear params:  {linear_params:,}", flush=True)

    # Optimizers
    softmax_optimizer = torch.optim.AdamW(softmax_model.parameters(), lr=1e-4, weight_decay=0.1)
    linear_optimizer = torch.optim.AdamW(linear_model.parameters(), lr=1e-4, weight_decay=0.1)

    softmax_scaler = torch.amp.GradScaler('cuda')
    linear_scaler = torch.amp.GradScaler('cuda')

    print(f"\nTraining both models for {num_epochs} epochs...", flush=True)
    print("="*80, flush=True)
    print(f"{'Epoch':>5} | {'Softmax Train':>12} | {'Softmax Eval':>12} | {'Linear Train':>12} | {'Linear Eval':>12} | {'Gap':>6}", flush=True)
    print("="*80, flush=True)

    softmax_best = float('inf')
    linear_best = float('inf')
    start_time = time.time()

    for epoch in range(num_epochs):
        # Train softmax
        softmax_train_loss = train_epoch(softmax_model, train_loader, softmax_optimizer, softmax_scaler, device)
        softmax_train_ppl = math.exp(softmax_train_loss)
        softmax_eval_ppl = evaluate(softmax_model, val_loader, device)
        softmax_best = min(softmax_best, softmax_eval_ppl)

        # Train linear
        linear_train_loss = train_epoch(linear_model, train_loader, linear_optimizer, linear_scaler, device)
        linear_train_ppl = math.exp(linear_train_loss)
        linear_eval_ppl = evaluate(linear_model, val_loader, device)
        linear_best = min(linear_best, linear_eval_ppl)

        # Gap (positive = softmax better, negative = linear better)
        gap = (linear_eval_ppl - softmax_eval_ppl) / softmax_eval_ppl * 100

        elapsed = (time.time() - start_time) / 60

        print(f"{epoch+1:5d} | {softmax_train_ppl:12.1f} | {softmax_eval_ppl:12.1f} | {linear_train_ppl:12.1f} | {linear_eval_ppl:12.1f} | {gap:+5.1f}% | {elapsed:.1f}m", flush=True)

    print("="*80, flush=True)
    print(f"\nFinal Results after {num_epochs} epochs:", flush=True)
    print(f"  Softmax best eval PPL: {softmax_best:.1f}", flush=True)
    print(f"  Linear best eval PPL:  {linear_best:.1f}", flush=True)
    print(f"  Gap: {(linear_best - softmax_best) / softmax_best * 100:+.1f}%", flush=True)

    if linear_best < softmax_best:
        print(f"\n  Linear attention WINS by {softmax_best - linear_best:.1f} PPL", flush=True)
    else:
        print(f"\n  Softmax attention WINS by {linear_best - softmax_best:.1f} PPL", flush=True)


if __name__ == "__main__":
    main()
