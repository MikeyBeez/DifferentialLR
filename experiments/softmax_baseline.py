#!/usr/bin/env python3
"""Train a standard softmax transformer as a fair baseline."""

import sys
sys.path.insert(0, '/home/bee/Code/LinearAttention')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import time
import math


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
        self.lm_head.weight = self.token_embedding.weight  # Tie weights

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

        # Causal mask
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
    print("Loading data...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_loader, val_loader = get_wikitext_data(tokenizer, seq_length=512, batch_size=8)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}", flush=True)

    # Same size as our linear attention model
    model = SoftmaxTransformer(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=512,
        num_layers=8,
        num_heads=8,
        intermediate_dim=2048,
        max_seq_len=1024,
        dropout=0.1,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Softmax Transformer Parameters: {num_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    scaler = torch.amp.GradScaler('cuda')

    print("\nTraining softmax transformer for 10 epochs...", flush=True)
    print("="*60, flush=True)

    best_ppl = float('inf')
    start_time = time.time()

    for epoch in range(10):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
        train_ppl = math.exp(train_loss)
        eval_ppl = evaluate(model, val_loader, device)

        marker = "*" if eval_ppl < best_ppl else ""
        best_ppl = min(best_ppl, eval_ppl)

        elapsed = (time.time() - start_time) / 60
        print(f"Epoch {epoch+1:2d} | Train PPL: {train_ppl:7.1f} | Eval PPL: {eval_ppl:7.1f} {marker} | {elapsed:.1f}m", flush=True)

    print(f"\nSoftmax Transformer Best Eval PPL: {best_ppl:.1f}", flush=True)
    print(f"\nComparison:", flush=True)
    print(f"  Softmax (this run):     PPL {best_ppl:.1f}", flush=True)
    print(f"  Linear (learned gate):  PPL 167.4", flush=True)
    print(f"  Gap: {best_ppl / 167.4:.2f}x", flush=True)


if __name__ == "__main__":
    main()
