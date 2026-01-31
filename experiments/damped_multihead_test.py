#!/usr/bin/env python3
"""
Multi-Resolution Damped Golden Spiral

Each head uses a different decay rate:
- Head 0: φ^{-1} decay (slow decay, long memory)
- Head 1: φ^{-2} decay (faster decay)
- Head 2: φ^{-3} decay (even faster)
- Head 3: φ^{-4} decay (shortest memory)

This gives multi-resolution temporal analysis - some heads focus
on recent context, others on deep history.
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
    """
    Bidirectional Damped Spiral with different decay rates per head.

    Head i uses decay rate φ^{-(i+1)}:
    - Head 0: 0.618 (slow decay)
    - Head 1: 0.382 (medium decay)
    - Head 2: 0.236 (fast decay)
    - Head 3: 0.146 (very fast decay)
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.t_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Different decay rates per head
        # phi_inv^1, phi_inv^2, phi_inv^3, phi_inv^4
        decay_rates = torch.tensor([(1/PHI) ** (i+1) for i in range(num_heads)])
        self.register_buffer('decay_rates', decay_rates)

        print(f"  Decay rates per head: {[f'{d:.4f}' for d in decay_rates.tolist()]}")

    def forward(self, x):
        batch, seq, dim = x.shape
        t = self.t_proj(x).view(batch, seq, self.num_heads, self.head_dim)

        # Forward pass: leaky integrator with per-head decay
        fwd_outputs = []
        engram_fwd = torch.zeros(batch, self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)
        for i in range(seq):
            # Each head has its own decay rate
            # decay_rates is [num_heads], need to broadcast to [batch, num_heads, head_dim]
            decay = self.decay_rates.view(1, self.num_heads, 1)
            engram_fwd = t[:, i] + decay * engram_fwd
            fwd_outputs.append(engram_fwd.clone())
        fwd = torch.stack(fwd_outputs, dim=1)

        # Backward pass: same per-head decay rates
        bwd_outputs = []
        engram_bwd = torch.zeros(batch, self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)
        for i in range(seq - 1, -1, -1):
            decay = self.decay_rates.view(1, self.num_heads, 1)
            engram_bwd = t[:, i] + decay * engram_bwd
            bwd_outputs.append(engram_bwd.clone())
        bwd_outputs.reverse()
        bwd = torch.stack(bwd_outputs, dim=1)

        # Combine (subtract T to avoid double-counting)
        r = fwd + bwd - t

        output = r.reshape(batch, seq, dim)
        return self.out_proj(output)


class UniformDampedSpiral(nn.Module):
    """Uniform decay (same φ^{-1} for all heads) for comparison."""
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

        fwd_outputs = []
        engram_fwd = torch.zeros(batch, self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)
        for i in range(seq):
            engram_fwd = t[:, i] + self.phi_inv * engram_fwd
            fwd_outputs.append(engram_fwd.clone())
        fwd = torch.stack(fwd_outputs, dim=1)

        bwd_outputs = []
        engram_bwd = torch.zeros(batch, self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)
        for i in range(seq - 1, -1, -1):
            engram_bwd = t[:, i] + self.phi_inv * engram_bwd
            bwd_outputs.append(engram_bwd.clone())
        bwd_outputs.reverse()
        bwd = torch.stack(bwd_outputs, dim=1)

        r = fwd + bwd - t
        output = r.reshape(batch, seq, dim)
        return self.out_proj(output)


class MultiResolutionBidirectional(nn.Module):
    """Multi-resolution ±1 bidirectional (the 1.3 PPL winner) for comparison."""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.phi = PHI
        self.phi_inv = 1 / PHI
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.t_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Multi-resolution: different φ powers for each head
        phi_past = torch.tensor([self.phi ** (i + 1) for i in range(num_heads)])
        phi_future = torch.tensor([self.phi_inv ** (i + 1) for i in range(num_heads)])
        self.register_buffer('phi_past', phi_past.view(1, 1, num_heads, 1))
        self.register_buffer('phi_future', phi_future.view(1, 1, num_heads, 1))

    def forward(self, x):
        batch, seq, dim = x.shape
        t = self.t_proj(x).view(batch, seq, self.num_heads, self.head_dim)

        t_prev = torch.roll(t, shifts=1, dims=1)
        t_prev[:, 0, :, :] = 0

        t_next = torch.roll(t, shifts=-1, dims=1)
        t_next[:, -1, :, :] = 0

        r = t + self.phi_past * t_prev + self.phi_future * t_next

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
            elif attention_type == 'multires_bidir':
                attn = MultiResolutionBidirectional(dim, num_heads)
            elif attention_type == 'uniform_spiral':
                attn = UniformDampedSpiral(dim, num_heads)
            elif attention_type == 'multires_spiral':
                attn = MultiResolutionDampedSpiral(dim, num_heads)
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
    print("MULTI-RESOLUTION DAMPED SPIRAL")
    print("=" * 70)
    print(f"\nφ = {PHI:.6f}")
    print("\nEach head uses a different decay rate:")
    print("  Head 0: φ^{-1} = 0.618 (slow decay, long memory)")
    print("  Head 1: φ^{-2} = 0.382 (medium decay)")
    print("  Head 2: φ^{-3} = 0.236 (fast decay)")
    print("  Head 3: φ^{-4} = 0.146 (very fast, local focus)")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_loader, val_loader = get_wikitext_data(tokenizer)

    results = {}
    attn_params = {}

    configs = [
        ('standard', 'Standard QKV Attention'),
        ('multires_bidir', 'Multi-Res Bidirectional ±1'),
        ('uniform_spiral', 'Uniform Damped Spiral'),
        ('multires_spiral', 'Multi-Res Damped Spiral'),
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
    print("RESULTS: MULTI-RESOLUTION COMPARISON")
    print("=" * 70)
    print(f"\n{'Method':<35} {'PPL':>8} {'Memory':>10} {'Context':>15}")
    print("-" * 70)

    memory = {
        'Standard QKV Attention': 'O(N²)',
        'Multi-Res Bidirectional ±1': 'O(N)',
        'Uniform Damped Spiral': 'O(1)',
        'Multi-Res Damped Spiral': 'O(1)',
    }

    context = {
        'Standard QKV Attention': 'Full (learned)',
        'Multi-Res Bidirectional ±1': '±1 only',
        'Uniform Damped Spiral': 'Infinite',
        'Multi-Res Damped Spiral': 'Infinite',
    }

    for name, ppl in results.items():
        print(f"{name:<35} {ppl:>8.1f} {memory[name]:>10} {context[name]:>15}")

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    std = results['Standard QKV Attention']
    bidir = results['Multi-Res Bidirectional ±1']
    uniform = results['Uniform Damped Spiral']
    multires = results['Multi-Res Damped Spiral']

    print(f"\nStandard Attention:      {std:.1f} PPL (baseline)")
    print(f"Multi-Res Bidir ±1:      {bidir:.1f} PPL ({(bidir-std)/std*100:+.1f}%)")
    print(f"Uniform Spiral:          {uniform:.1f} PPL ({(uniform-std)/std*100:+.1f}%)")
    print(f"Multi-Res Spiral:        {multires:.1f} PPL ({(multires-std)/std*100:+.1f}%)")

    if multires < uniform:
        improvement = (uniform - multires) / uniform * 100
        print(f"\nMulti-resolution HELPS! {improvement:.1f}% improvement over uniform.")

    if multires < bidir:
        print(f"\nSpiral BEATS ±1 bidirectional!")
        print("Infinite context with O(1) memory wins.")


if __name__ == "__main__":
    main()
