#!/usr/bin/env python3
"""
Multi-Position Phi Powers Test

r = T_0 + φ*T_{-1} + φ²*T_{-2} + φ³*T_{-3} + ...
     + φ^{-1}*T_{+1} + φ^{-2}*T_{+2} + ...

Each position gets a unique power of phi.
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


class PhiPowersEngram(nn.Module):
    """
    Multi-position Golden Engram with phi powers.

    r = T_0 + φ*T_{-1} + φ²*T_{-2} + ... + φ^{-1}*T_{+1} + φ^{-2}*T_{+2} + ...
    """
    def __init__(self, embed_dim, num_heads, past_window=4, future_window=4):
        super().__init__()
        self.phi = PHI
        self.embed_dim = embed_dim
        self.past_window = past_window
        self.future_window = future_window

        self.t_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Precompute phi powers
        # Past: φ^1, φ^2, φ^3, ...
        # Future: φ^{-1}, φ^{-2}, φ^{-3}, ...
        self.past_powers = [self.phi ** (i+1) for i in range(past_window)]
        self.future_powers = [(1/self.phi) ** (i+1) for i in range(future_window)]

    def forward(self, x):
        batch, seq, dim = x.shape
        t = self.t_proj(x)

        # Start with current position (weight = 1)
        r = t.clone()

        # Add past positions with φ^i weights
        for i, power in enumerate(self.past_powers):
            shift = i + 1
            t_shifted = torch.roll(t, shifts=shift, dims=1)
            t_shifted[:, :shift, :] = 0  # Zero out wrapped positions
            r = r + power * t_shifted

        # Add future positions with φ^{-i} weights
        for i, power in enumerate(self.future_powers):
            shift = -(i + 1)
            t_shifted = torch.roll(t, shifts=shift, dims=1)
            t_shifted[:, shift:, :] = 0  # Zero out wrapped positions
            r = r + power * t_shifted

        return self.out_proj(r)


class BidirectionalPureGolden(nn.Module):
    """Original bidirectional (prev + next only) for comparison."""
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


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, num_heads, attention_type='standard',
                 past_window=4, future_window=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(1024, dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if attention_type == 'standard':
                attn = StandardAttention(dim, num_heads)
            elif attention_type == 'bidirectional':
                attn = BidirectionalPureGolden(dim, num_heads)
            elif attention_type == 'phi_powers':
                attn = PhiPowersEngram(dim, num_heads, past_window, future_window)
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


def count_attn_params(model):
    attn_params = 0
    for layer in model.layers:
        attn_params += sum(p.numel() for p in layer['attn'].parameters())
    return attn_params


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n" + "=" * 70)
    print("PHI POWERS TEST: r = T + φT_{-1} + φ²T_{-2} + ... + φ^{-1}T_{+1} + ...")
    print("=" * 70)
    print(f"\nφ = {PHI:.6f}")
    print(f"φ² = {PHI**2:.6f}")
    print(f"φ³ = {PHI**3:.6f}")
    print(f"φ^{{-1}} = {1/PHI:.6f}")
    print(f"φ^{{-2}} = {(1/PHI)**2:.6f}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_loader, val_loader = get_wikitext_data(tokenizer)

    results = {}
    attn_params = {}

    configs = [
        ('standard', 'Standard QKV Attention', 0, 0),
        ('bidirectional', 'Bidirectional (±1 only)', 0, 0),
        ('phi_powers', 'Phi Powers (±2)', 2, 2),
        ('phi_powers', 'Phi Powers (±4)', 4, 4),
        ('phi_powers', 'Phi Powers (±8)', 8, 8),
    ]

    for attn_type, name, past_w, future_w in configs:
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
            attention_type=attn_type,
            past_window=past_w,
            future_window=future_w
        ).to(device)

        attn_params[name] = count_attn_params(model)
        print(f"Attention params: {attn_params[name]:,}")

        if attn_type == 'phi_powers':
            print(f"Past window: {past_w}, Future window: {future_w}")
            powers_past = [PHI ** (i+1) for i in range(past_w)]
            powers_future = [(1/PHI) ** (i+1) for i in range(future_w)]
            print(f"Past weights: {[f'{p:.3f}' for p in powers_past]}")
            print(f"Future weights: {[f'{p:.3f}' for p in powers_future]}")

        results[name] = train_and_eval(model, train_loader, val_loader, device, epochs=5)
        del model

    print("\n" + "=" * 70)
    print("RESULTS: PHI POWERS")
    print("=" * 70)
    print(f"\n{'Method':<35} {'PPL':>8} {'Attn Params':>15}")
    print("-" * 60)

    for name, ppl in results.items():
        print(f"{name:<35} {ppl:>8.1f} {attn_params[name]:>15,}")

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    baseline = results['Bidirectional (±1 only)']
    for name, ppl in results.items():
        if name != 'Bidirectional (±1 only)':
            diff = (ppl - baseline) / baseline * 100
            print(f"{name}: {diff:+.1f}% vs bidirectional")


if __name__ == "__main__":
    main()
