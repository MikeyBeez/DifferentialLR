#!/usr/bin/env python3
"""
Test: Does attention need similarity, or just unique weights?

Hypothesis: Q·K produces unique content-dependent weights.
The "similarity" interpretation might be incidental.
What matters is that each token gets a distinct weight.

Test: Replace Q·K with simpler mechanisms that produce unique weights.
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


class SimpleWeightAttention(nn.Module):
    """
    Replace Q·K with a simple learned projection.

    weight_i = W · token_i

    Each token gets a unique weight based on its content,
    but no pairwise similarity computation.
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Single projection to get weight per token per head
        self.weight_proj = nn.Linear(dim, num_heads)

        # Value projection (still needed)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape

        # Get unique weight per token per head
        weights = self.weight_proj(x)  # [B, N, num_heads]
        weights = F.softmax(weights, dim=1)  # normalize across positions
        weights = weights.transpose(1, 2)  # [B, num_heads, N]

        # Get values
        v = self.v_proj(x)  # [B, N, D]
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, N, head_dim]

        # Weighted sum: each head produces one vector (not per-position)
        # This is different from attention - we get one output per head, not per position
        # Let's fix this to match attention's output shape

        # Actually, for fair comparison, we need per-position outputs
        # Let's do: each position attends to all positions with its weight
        weights = weights.unsqueeze(2)  # [B, heads, 1, N]
        out = torch.matmul(weights, v)  # [B, heads, 1, head_dim]

        # Hmm, this collapses positions. Let me rethink...
        # Each position needs to produce an output.
        # The "unique weight" idea might need adjustment.

        return out


class ContentHashAttention(nn.Module):
    """
    Each position gets a weight based on its content.
    Other positions' outputs are weighted sums.

    Simpler: weight_ij = f(token_j) only, not f(token_i, token_j)
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Only K and V, no Q
        # Weight comes from K alone, not Q·K
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape

        # Project to get "importance" per token
        k = self.k_proj(x)  # [B, N, D]
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Importance score per position (sum across head_dim as a simple hash)
        importance = k.sum(dim=-1)  # [B, heads, N]
        weights = F.softmax(importance * self.scale, dim=-1)  # [B, heads, N]

        # Values
        v = self.v_proj(x)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Every position gets the same weighted sum (no position-specific query)
        weights = weights.unsqueeze(2)  # [B, heads, 1, N]
        out = torch.matmul(weights, v)  # [B, heads, 1, head_dim]
        out = out.expand(-1, -1, N, -1)  # [B, heads, N, head_dim] - same for all positions

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.o_proj(out), weights.squeeze(2)


class RandomProjectionAttention(nn.Module):
    """
    Weight = random_fixed_vector · token

    Unique per token (content-dependent), but random, not learned.
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Fixed random projection (not learned)
        self.register_buffer('random_proj', torch.randn(num_heads, dim))

        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape

        # Random but fixed projection gives unique weight per token
        # [B, N, D] x [heads, D]^T -> [B, N, heads]
        importance = torch.einsum('bnd,hd->bnh', x, self.random_proj)
        importance = importance.transpose(1, 2)  # [B, heads, N]
        weights = F.softmax(importance * self.scale, dim=-1)

        # Values
        v = self.v_proj(x)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # All positions get same weighted sum
        weights_exp = weights.unsqueeze(2)
        out = torch.matmul(weights_exp, v)  # [B, heads, 1, head_dim]
        out = out.expand(-1, -1, N, -1)

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.o_proj(out), weights


class PositionOnlyAttention(nn.Module):
    """
    Weight based on position only, not content.

    Tests whether content-dependence matters.
    """
    def __init__(self, dim, num_heads, max_len=1024):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Learned positional weights
        self.pos_weights = nn.Parameter(torch.randn(num_heads, max_len) * 0.02)

        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape

        # Position-only weights
        weights = F.softmax(self.pos_weights[:, :N], dim=-1)  # [heads, N]
        weights = weights.unsqueeze(0).expand(B, -1, -1)  # [B, heads, N]

        # Values
        v = self.v_proj(x)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        weights_exp = weights.unsqueeze(2)
        out = torch.matmul(weights_exp, v)
        out = out.expand(-1, -1, N, -1)

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.o_proj(out), weights


class FullAttention(nn.Module):
    """Standard Q·K attention for comparison."""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = F.softmax(scores, dim=-1)

        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.o_proj(out), weights


class SimpleTransformer(nn.Module):
    """Minimal transformer for testing attention variants."""
    def __init__(self, vocab_size, dim, num_layers, num_heads, attention_type='full'):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(1024, dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if attention_type == 'full':
                attn = FullAttention(dim, num_heads)
            elif attention_type == 'content_hash':
                attn = ContentHashAttention(dim, num_heads)
            elif attention_type == 'random_proj':
                attn = RandomProjectionAttention(dim, num_heads)
            elif attention_type == 'position_only':
                attn = PositionOnlyAttention(dim, num_heads)
            else:
                raise ValueError(f"Unknown attention type: {attention_type}")

            self.layers.append(nn.ModuleDict({
                'attn': attn,
                'norm1': nn.LayerNorm(dim),
                'ffn': nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim)
                ),
                'norm2': nn.LayerNorm(dim)
            }))

        self.final_norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids, labels=None):
        B, N = input_ids.shape
        pos = torch.arange(N, device=input_ids.device).unsqueeze(0)

        x = self.embed(input_ids) + self.pos_embed(pos)

        for layer in self.layers:
            # Attention
            normed = layer['norm1'](x)
            attn_out, _ = layer['attn'](normed)
            x = x + attn_out

            # FFN
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

        # Eval
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
    print("UNIQUE WEIGHTS TEST")
    print("=" * 70)
    print("\nHypothesis: Attention just needs unique weights, not similarity")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_loader, val_loader = get_wikitext_data(tokenizer)

    results = {}

    attention_types = [
        ('full', 'Full Q·K attention'),
        ('content_hash', 'Content hash (K only, no Q)'),
        ('random_proj', 'Random projection'),
        ('position_only', 'Position only (no content)'),
    ]

    for attn_type, name in attention_types:
        print(f"\n{'='*50}")
        print(f"{name}")
        print('='*50)

        torch.manual_seed(42)
        torch.cuda.empty_cache()

        model = SimpleTransformer(
            vocab_size=tokenizer.vocab_size,
            dim=256,
            num_layers=4,
            num_heads=4,
            attention_type=attn_type
        ).to(device)

        results[name] = train_and_eval(model, train_loader, val_loader, device, epochs=5)
        del model

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for name, ppl in results.items():
        print(f"  {name:35s}: {ppl:.1f} PPL")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    full_ppl = results['Full Q·K attention']
    content_ppl = results['Content hash (K only, no Q)']

    if content_ppl < full_ppl * 1.1:
        print("\nContent hash performs similarly to full attention!")
        print("→ The Q (query) may not be necessary")
        print("→ Unique content-dependent weights are sufficient")
    else:
        print(f"\nFull attention still better ({full_ppl:.1f} vs {content_ppl:.1f})")
        print("→ The pairwise Q·K computation adds value")


if __name__ == "__main__":
    main()
