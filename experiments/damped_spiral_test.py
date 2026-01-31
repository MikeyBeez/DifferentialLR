#!/usr/bin/env python3
"""
Damped Golden Spiral: Convergent Geometric Series

Forward:  engram_fwd[i] = T[i] + φ^{-1} * engram_fwd[i-1]
Backward: engram_bwd[i] = T[i] + φ^{-1} * engram_bwd[i+1]
Combined: output[i] = engram_fwd[i] + engram_bwd[i] - T[i]

The weights form a convergent series: 1 + φ^{-1} + φ^{-2} + ... = φ ≈ 2.618
No explosion possible. Infinite context in O(1) memory per direction.

This is essentially Mamba with A = φ^{-1} * I (fixed golden decay).
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


class DampedGoldenSpiral(nn.Module):
    """
    Bidirectional Damped Spiral with O(N) compute, O(1) memory state.

    Forward pass:  accumulate past with φ^{-1} decay
    Backward pass: accumulate future with φ^{-1} decay
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.phi_inv = 1 / PHI
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.t_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch, seq, dim = x.shape
        t = self.t_proj(x).view(batch, seq, self.num_heads, self.head_dim)

        # Forward pass: leaky integrator from past
        fwd_outputs = []
        engram_fwd = torch.zeros(batch, self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)
        for i in range(seq):
            engram_fwd = t[:, i] + self.phi_inv * engram_fwd
            fwd_outputs.append(engram_fwd.clone())
        fwd = torch.stack(fwd_outputs, dim=1)

        # Backward pass: leaky integrator from future
        bwd_outputs = []
        engram_bwd = torch.zeros(batch, self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)
        for i in range(seq - 1, -1, -1):
            engram_bwd = t[:, i] + self.phi_inv * engram_bwd
            bwd_outputs.append(engram_bwd.clone())
        bwd_outputs.reverse()
        bwd = torch.stack(bwd_outputs, dim=1)

        # Combine (subtract T to avoid double-counting current token)
        r = fwd + bwd - t

        output = r.reshape(batch, seq, dim)
        return self.out_proj(output)


class CausalDampedSpiral(nn.Module):
    """Causal-only damped spiral (for comparison)."""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.phi_inv = 1 / PHI
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.t_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch, seq, dim = x.shape
        t = self.t_proj(x).view(batch, seq, self.num_heads, self.head_dim)

        outputs = []
        engram = torch.zeros(batch, self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)
        for i in range(seq):
            engram = t[:, i] + self.phi_inv * engram
            outputs.append(engram.clone())

        r = torch.stack(outputs, dim=1)
        output = r.reshape(batch, seq, dim)
        return self.out_proj(output)


class BidirectionalPureGolden(nn.Module):
    """Simple ±1 bidirectional (the 1.3 PPL winner)."""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.phi = PHI
        self.phi_inv = 1 / PHI
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.t_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch, seq, dim = x.shape
        t = self.t_proj(x).view(batch, seq, self.num_heads, self.head_dim)

        t_prev = torch.roll(t, shifts=1, dims=1)
        t_prev[:, 0, :, :] = 0

        t_next = torch.roll(t, shifts=-1, dims=1)
        t_next[:, -1, :, :] = 0

        r = t + self.phi * t_prev + self.phi_inv * t_next

        output = r.reshape(batch, seq, dim)
        return self.out_proj(output)


class StandardAttention(nn.Module):
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
    def __init__(self, vocab_size, dim, num_layers, num_heads, attention_type='standard'):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(1024, dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if attention_type == 'standard':
                attn = StandardAttention(dim, num_heads)
            elif attention_type == 'bidirectional':
                attn = BidirectionalPureGolden(dim, num_heads)
            elif attention_type == 'causal_damped':
                attn = CausalDampedSpiral(dim, num_heads)
            elif attention_type == 'bidirectional_damped':
                attn = DampedGoldenSpiral(dim, num_heads)
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
    print("DAMPED GOLDEN SPIRAL: Convergent Geometric Series")
    print("=" * 70)
    print(f"\nφ^{{-1}} = {1/PHI:.6f}")
    print(f"Sum of series (1 + φ^-1 + φ^-2 + ...) = φ = {PHI:.6f}")
    print("\nThe weights NEVER explode. Infinite context, bounded magnitude.")
    print("This is Mamba with A = φ^{-1} * I (fixed golden decay).")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_loader, val_loader = get_wikitext_data(tokenizer)

    results = {}
    attn_params = {}

    configs = [
        ('standard', 'Standard QKV Attention'),
        ('bidirectional', 'Bidirectional ±1 (φ, φ^{-1})'),
        ('causal_damped', 'Causal Damped Spiral'),
        ('bidirectional_damped', 'Bidirectional Damped Spiral'),
    ]

    for attn_type, name in configs:
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

        attn_params[name] = count_attn_params(model)
        print(f"Attention params: {attn_params[name]:,}")

        results[name] = train_and_eval(model, train_loader, val_loader, device, epochs=5)
        del model

    print("\n" + "=" * 70)
    print("RESULTS: DAMPED SPIRAL")
    print("=" * 70)
    print(f"\n{'Method':<40} {'PPL':>8} {'Attn Params':>15} {'Memory':>10}")
    print("-" * 75)

    memory = {
        'Standard QKV Attention': 'O(N²)',
        'Bidirectional ±1 (φ, φ^{-1})': 'O(N)',
        'Causal Damped Spiral': 'O(1)',
        'Bidirectional Damped Spiral': 'O(1)',
    }

    for name, ppl in results.items():
        print(f"{name:<40} {ppl:>8.1f} {attn_params[name]:>15,} {memory[name]:>10}")

    print("\n" + "=" * 70)
    print("ANALYSIS: Golden SSM vs Attention")
    print("=" * 70)

    std = results['Standard QKV Attention']
    bidir = results['Bidirectional ±1 (φ, φ^{-1})']
    damped = results['Bidirectional Damped Spiral']

    print(f"\nStandard Attention:     {std:.1f} PPL (baseline)")
    print(f"Simple Bidirectional:   {bidir:.1f} PPL ({(bidir-std)/std*100:+.1f}%)")
    print(f"Damped Spiral:          {damped:.1f} PPL ({(damped-std)/std*100:+.1f}%)")

    if damped < bidir:
        print(f"\nDamped Spiral WINS! Infinite context beats local.")
    else:
        print(f"\nSimple ±1 still wins. Local context dominates.")


if __name__ == "__main__":
    main()
