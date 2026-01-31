#!/usr/bin/env python3
"""
The Recursive Golden Spiral: Infinite Context via Geometric Accumulation

Instead of windowed attention (look at prev/next), we spiral the entire
context into a single vector:

    engram_n = engram_{n-1} * phi^{-1} + T_n

This creates a logarithmic spiral in hidden space:
- Recent tokens at the outer edge (high weight)
- Deep context compressed into the center (preserved but decayed)

The order is baked into geometry: token i gets weight phi^{-(n-i)}
Swapping two tokens moves the centrum to a completely different coordinate.
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


class RecursiveGoldenSpiral(nn.Module):
    """
    The pure recursive Golden Engram.

    Every token is rotated/scaled by phi^{-1} as it enters the stream.
    The entire context window becomes a logarithmic spiral.

    Memory: O(1) - only stores one vector (the current spiral state)
    Compute: O(N) - one addition per token
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.phi = PHI
        self.phi_inv = 1 / PHI
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.t_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch, seq, dim = x.shape
        t = self.t_proj(x)

        # Recursive accumulation: spiral the context
        engram = torch.zeros(batch, dim, device=x.device, dtype=x.dtype)
        outputs = []

        for i in range(seq):
            # Rotate existing context by phi^{-1}, add new token
            engram = (engram * self.phi_inv) + t[:, i, :]
            outputs.append(engram.clone())

        r = torch.stack(outputs, dim=1)
        return self.out_proj(r)


class RecursiveGoldenSpiralFast(nn.Module):
    """
    Vectorized version of the recursive spiral.

    Instead of a Python loop, we compute the weights phi^{-(n-i)}
    for all positions and do a single matrix multiply.
    """
    def __init__(self, embed_dim, num_heads, max_len=1024):
        super().__init__()
        self.phi = PHI
        self.phi_inv = 1 / PHI
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_len = max_len

        self.t_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Precompute causal weights: position j contributes phi^{-(i-j)} to position i
        # This is a lower triangular matrix
        weights = torch.zeros(max_len, max_len)
        for i in range(max_len):
            for j in range(i + 1):
                weights[i, j] = self.phi_inv ** (i - j)
        self.register_buffer('causal_weights', weights)

    def forward(self, x):
        batch, seq, dim = x.shape
        t = self.t_proj(x)  # [B, N, D]

        # Get causal weights for this sequence length
        weights = self.causal_weights[:seq, :seq]  # [N, N], lower triangular

        # Weighted sum: output[i] = sum_j weights[i,j] * t[j]
        # [B, N, D] = [N, N] @ [B, N, D] (broadcast over batch and dim)
        r = torch.einsum('ij,bjd->bid', weights, t)

        return self.out_proj(r)


class BidirectionalGoldenSpiral(nn.Module):
    """
    Bidirectional spiral: forward AND backward accumulation.

    Combines:
    - Forward spiral: engram_fwd[i] = sum_{j<=i} phi^{-(i-j)} * T_j
    - Backward spiral: engram_bwd[i] = sum_{j>=i} phi^{-(j-i)} * T_j

    Output = forward + backward (each position sees full context)
    """
    def __init__(self, embed_dim, num_heads, max_len=1024):
        super().__init__()
        self.phi = PHI
        self.phi_inv = 1 / PHI
        self.embed_dim = embed_dim

        self.t_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Forward weights (lower triangular)
        fwd_weights = torch.zeros(max_len, max_len)
        for i in range(max_len):
            for j in range(i + 1):
                fwd_weights[i, j] = self.phi_inv ** (i - j)

        # Backward weights (upper triangular)
        bwd_weights = torch.zeros(max_len, max_len)
        for i in range(max_len):
            for j in range(i, max_len):
                bwd_weights[i, j] = self.phi_inv ** (j - i)

        self.register_buffer('fwd_weights', fwd_weights)
        self.register_buffer('bwd_weights', bwd_weights)

    def forward(self, x):
        batch, seq, dim = x.shape
        t = self.t_proj(x)

        fwd_w = self.fwd_weights[:seq, :seq]
        bwd_w = self.bwd_weights[:seq, :seq]

        # Forward and backward spirals
        fwd = torch.einsum('ij,bjd->bid', fwd_w, t)
        bwd = torch.einsum('ij,bjd->bid', bwd_w, t)

        # Combine (subtract self to avoid double-counting current token)
        r = fwd + bwd - t

        return self.out_proj(r)


class StandardAttention(nn.Module):
    """Standard attention for comparison."""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, D = x.shape

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.o_proj(out)


class WindowedGoldenEngram(nn.Module):
    """Original windowed version for comparison."""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.phi = PHI
        self.phi_inv = 1 / PHI

        self.t_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch, seq, dim = x.shape
        t = self.t_proj(x)

        t_prev = torch.roll(t, shifts=1, dims=1)
        t_prev[:, 0, :] = 0

        t_next = torch.roll(t, shifts=-1, dims=1)
        t_next[:, -1, :] = 0

        r = t + self.phi * t_prev + self.phi_inv * t_next

        return self.out_proj(r)


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, num_heads, attention_type='standard'):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(1024, dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if attention_type == 'standard':
                attn = StandardAttention(dim, num_heads)
            elif attention_type == 'windowed':
                attn = WindowedGoldenEngram(dim, num_heads)
            elif attention_type == 'recursive':
                attn = RecursiveGoldenSpiral(dim, num_heads)
            elif attention_type == 'recursive_fast':
                attn = RecursiveGoldenSpiralFast(dim, num_heads)
            elif attention_type == 'bidirectional_spiral':
                attn = BidirectionalGoldenSpiral(dim, num_heads)
            else:
                raise ValueError(f"Unknown: {attention_type}")

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
    print("RECURSIVE GOLDEN SPIRAL: INFINITE CONTEXT VIA GEOMETRIC ACCUMULATION")
    print("=" * 70)
    print(f"\nphi = {PHI:.6f}")
    print(f"phi^-1 = {1/PHI:.6f}")
    print("\nThe entire context window becomes a logarithmic spiral.")
    print("Recent tokens at outer edge, deep context compressed into center.")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_loader, val_loader = get_wikitext_data(tokenizer)

    results = {}

    attention_types = [
        ('standard', 'Standard QKV Attention'),
        ('windowed', 'Windowed Golden (prev + next)'),
        ('recursive_fast', 'Recursive Spiral (causal)'),
        ('bidirectional_spiral', 'Bidirectional Spiral (full context)'),
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
    print("RESULTS: WINDOWED vs SPIRAL")
    print("=" * 70)
    for name, ppl in results.items():
        print(f"  {name:40s}: {ppl:.1f} PPL")

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    windowed = results['Windowed Golden (prev + next)']
    spiral = results['Bidirectional Spiral (full context)']

    if spiral < windowed:
        print(f"\nSpiral BEATS windowed! ({spiral:.1f} vs {windowed:.1f})")
        print("The infinite compressed context helps.")
    else:
        print(f"\nWindowed still wins ({windowed:.1f} vs {spiral:.1f})")
        print("Local context dominates for this task/scale.")


if __name__ == "__main__":
    main()
