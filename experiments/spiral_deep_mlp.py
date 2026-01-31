#!/usr/bin/env python3
"""
Multi-Res Damped Spiral with Deeper MLP

Hypothesis: The spiral loses the Q, K, V projections (3 matrices).
A deeper MLP can compensate for this lost capacity.
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

PHI = (1 + math.sqrt(5)) / 2


class MultiResolutionDampedSpiral(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.t_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        decay_rates = torch.tensor([(1/PHI) ** (i+1) for i in range(num_heads)])
        self.register_buffer('decay_rates', decay_rates)

    def forward(self, x):
        batch, seq, dim = x.shape
        t = self.t_proj(x).view(batch, seq, self.num_heads, self.head_dim)

        fwd_outputs = []
        engram_fwd = torch.zeros(batch, self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)
        for i in range(seq):
            decay = self.decay_rates.view(1, self.num_heads, 1)
            engram_fwd = t[:, i] + decay * engram_fwd
            fwd_outputs.append(engram_fwd.clone())
        fwd = torch.stack(fwd_outputs, dim=1)

        bwd_outputs = []
        engram_bwd = torch.zeros(batch, self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)
        for i in range(seq - 1, -1, -1):
            decay = self.decay_rates.view(1, self.num_heads, 1)
            engram_bwd = t[:, i] + decay * engram_bwd
            bwd_outputs.append(engram_bwd.clone())
        bwd_outputs.reverse()
        bwd = torch.stack(bwd_outputs, dim=1)

        r = fwd + bwd - t
        output = r.reshape(batch, seq, dim)
        return self.out_proj(output)


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, num_heads, mlp_depth=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        # NO positional encoding - spiral encodes position

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            # Build deeper MLP
            if mlp_depth == 2:
                # Standard: 2 layers
                ffn = nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim)
                )
            elif mlp_depth == 4:
                # Deeper: 4 layers
                ffn = nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim)
                )
            else:
                raise ValueError(f"Unknown mlp_depth: {mlp_depth}")

            self.layers.append(nn.ModuleDict({
                'attn': MultiResolutionDampedSpiral(dim, num_heads),
                'norm1': nn.LayerNorm(dim),
                'ffn': ffn,
                'norm2': nn.LayerNorm(dim)
            }))

        self.final_norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids, labels=None):
        B, N = input_ids.shape

        x = self.embed(input_ids)
        # No positional encoding

        for layer in self.layers:
            normed = layer['norm1'](x)
            attn_out = layer['attn'](normed)
            x = x + attn_out

            normed = layer['norm2'](x)
            x = x + layer['ffn'](normed)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        result = {'logits': logits}
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            result['loss'] = loss

        return result


def get_wikitext_data(tokenizer, seq_length=256, batch_size=16):
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


def train_and_eval(model, train_loader, val_loader, device, epochs=5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out = model(batch, labels=batch)
                loss = out['loss']
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                with torch.amp.autocast('cuda'):
                    out = model(batch, labels=batch)
                val_loss += out['loss'].item()

        val_ppl = math.exp(val_loss / len(val_loader))
        print(f"  Epoch {epoch+1}: Train {math.exp(total_loss/len(train_loader)):.1f}, Val {val_ppl:.1f}")

    return val_ppl


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n" + "=" * 70)
    print("MULTI-RES SPIRAL + DEEP MLP (No Positional Encoding)")
    print("=" * 70)
    print("\nHypothesis: Deeper MLP compensates for lost Q, K, V projections.")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_loader, val_loader = get_wikitext_data(tokenizer)

    print(f"\n{'='*50}")
    print("Multi-Res Spiral + 4-Layer MLP (no pos)")
    print('='*50)

    torch.manual_seed(42)
    torch.cuda.empty_cache()

    model = SimpleTransformer(
        vocab_size=tokenizer.vocab_size,
        dim=256,
        num_layers=4,
        num_heads=4,
        mlp_depth=4  # DEEPER MLP
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")

    ppl = train_and_eval(model, train_loader, val_loader, device, epochs=5)

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"\nSpiral + 4-layer MLP (no pos): {ppl:.1f} PPL")
    print(f"Spiral + 2-layer MLP (no pos): 1.4 PPL (previous)")
    print(f"Standard Attention: 1.2 PPL (baseline)")

    if ppl < 1.4:
        improvement = (1.4 - ppl) / 1.4 * 100
        print(f"\nDeeper MLP helps! {improvement:.1f}% improvement")
    if ppl <= 1.2:
        print("\nMATCHED OR BEAT STANDARD ATTENTION!")


if __name__ == "__main__":
    main()
