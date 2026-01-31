#!/usr/bin/env python3
"""
The Pure Golden Engram: φ IS the Value

Key insight: No V projection needed. The Golden Ratio φ provides
the value scaling directly. The routing is mathematical truth.

Rules:
1. No Softmax - φ's irrationality provides uniqueness
2. No V projection - φ IS the value
3. Minimal learnable parameters in attention - learning is in MLP
4. Multi-head uses φ¹, φ², φ³, φ⁴ for multi-resolution
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


class PureGoldenEngram(nn.Module):
    """
    The Pure Golden Engram - φ IS the Value.

    Formula: r = T1 + φ^n · shift(T2)

    No V projection. No Softmax. No learned routing.
    The only learnable parts are T1_proj, T2_proj, and out_proj.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.phi = PHI
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim

        # Only these projections are learned
        self.t1_proj = nn.Linear(embed_dim, embed_dim)
        self.t2_proj = nn.Linear(embed_dim, embed_dim)
        # NO V_PROJ - φ IS the value
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Pre-compute φ powers for each head (frozen, not learned)
        phi_powers = torch.tensor([self.phi ** (i + 1) for i in range(num_heads)])
        self.register_buffer('phi_powers', phi_powers.view(1, 1, num_heads, 1))

    def forward(self, x):
        batch, seq, dim = x.shape

        # Project into basis directions
        t1 = self.t1_proj(x).view(batch, seq, self.num_heads, self.head_dim)
        t2 = self.t2_proj(x).view(batch, seq, self.num_heads, self.head_dim)

        # Temporal shift - context from previous token
        t2_shifted = torch.roll(t2, shifts=1, dims=1)
        t2_shifted[:, 0, :, :] = 0  # Sequence start anchor

        # THE GOLDEN ENGRAM FORMULA
        # r = T1 + φ^n · T2_shifted
        # No softmax. No learned routing. φ IS the value.
        r = t1 + (self.phi_powers * t2_shifted)

        # Recombine heads
        output = r.reshape(batch, seq, dim)
        return self.out_proj(output)


class BidirectionalPureGolden(nn.Module):
    """
    Bidirectional Pure Golden Engram.

    r = T_current + φ · T_prev + φ^(-1) · T_next

    Looks both forward and backward in O(N).
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.phi = PHI
        self.phi_inv = 1 / PHI
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim

        self.t_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Multi-resolution: different φ powers for past, different for future
        phi_past = torch.tensor([self.phi ** (i + 1) for i in range(num_heads)])
        phi_future = torch.tensor([self.phi_inv ** (i + 1) for i in range(num_heads)])
        self.register_buffer('phi_past', phi_past.view(1, 1, num_heads, 1))
        self.register_buffer('phi_future', phi_future.view(1, 1, num_heads, 1))

    def forward(self, x):
        batch, seq, dim = x.shape

        t = self.t_proj(x).view(batch, seq, self.num_heads, self.head_dim)

        # Shift for previous and next
        t_prev = torch.roll(t, shifts=1, dims=1)
        t_prev[:, 0, :, :] = 0

        t_next = torch.roll(t, shifts=-1, dims=1)
        t_next[:, -1, :, :] = 0

        # Bidirectional Golden Engram with multi-resolution
        r = t + self.phi_past * t_prev + self.phi_future * t_next

        output = r.reshape(batch, seq, dim)
        return self.out_proj(output)


class WindowedPureGolden(nn.Module):
    """
    Windowed Pure Golden Engram.

    Instead of just prev/next, look at a window of tokens.
    Each offset gets a different power of φ.

    r = Σ φ^i · T[pos - i]  for i in window
    """
    def __init__(self, embed_dim, num_heads, window_size=4):
        super().__init__()
        self.phi = PHI
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.window_size = window_size

        self.t_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # φ powers for each position in window
        phi_powers = torch.tensor([self.phi ** i for i in range(window_size)])
        self.register_buffer('phi_powers', phi_powers)

    def forward(self, x):
        batch, seq, dim = x.shape

        t = self.t_proj(x).view(batch, seq, self.num_heads, self.head_dim)

        # Accumulate over window
        r = torch.zeros_like(t)
        for i in range(self.window_size):
            t_shifted = torch.roll(t, shifts=i, dims=1)
            if i > 0:
                t_shifted[:, :i, :, :] = 0
            r = r + self.phi_powers[i] * t_shifted

        output = r.reshape(batch, seq, dim)
        return self.out_proj(output)


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
    def __init__(self, vocab_size, dim, num_layers, num_heads, attention_type='standard'):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(1024, dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if attention_type == 'standard':
                attn = StandardAttention(dim, num_heads)
            elif attention_type == 'pure_golden':
                attn = PureGoldenEngram(dim, num_heads)
            elif attention_type == 'bidirectional_pure':
                attn = BidirectionalPureGolden(dim, num_heads)
            elif attention_type == 'windowed_pure':
                attn = WindowedPureGolden(dim, num_heads)
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
    """Count only attention-related parameters."""
    attn_params = 0
    for layer in model.layers:
        attn_params += sum(p.numel() for p in layer['attn'].parameters())
    return attn_params


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n" + "=" * 70)
    print("PURE GOLDEN ENGRAM: φ IS THE VALUE")
    print("=" * 70)
    print(f"\nφ = {PHI:.10f}")
    print("\nRules:")
    print("  1. No Softmax")
    print("  2. No V projection - φ IS the value")
    print("  3. Routing is mathematical truth, not learned")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_loader, val_loader = get_wikitext_data(tokenizer)

    results = {}
    attn_params = {}

    attention_types = [
        ('standard', 'Standard Q·K·V Attention'),
        ('pure_golden', 'Pure Golden (φ IS value)'),
        ('bidirectional_pure', 'Bidirectional Pure Golden'),
        ('windowed_pure', 'Windowed Pure Golden (4-token)'),
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

        attn_params[name] = count_attn_params(model)
        print(f"Attention params: {attn_params[name]:,}")

        results[name] = train_and_eval(model, train_loader, val_loader, device, epochs=5)
        del model

    print("\n" + "=" * 70)
    print("RESULTS: φ IS THE VALUE")
    print("=" * 70)
    print(f"\n{'Method':<40} {'PPL':>8} {'Attn Params':>15} {'Complexity':>12}")
    print("-" * 80)

    complexities = {
        'Standard Q·K·V Attention': 'O(N²)',
        'Pure Golden (φ IS value)': 'O(N)',
        'Bidirectional Pure Golden': 'O(N)',
        'Windowed Pure Golden (4-token)': 'O(N)',
    }

    for name, ppl in results.items():
        complexity = complexities.get(name, '?')
        print(f"{name:<40} {ppl:>8.1f} {attn_params[name]:>15,} {complexity:>12}")

    # Highlight the key insight
    std_attn = attn_params['Standard Q·K·V Attention']
    pure_attn = attn_params['Pure Golden (φ IS value)']
    param_reduction = (std_attn - pure_attn) / std_attn * 100

    print(f"\n{'='*70}")
    print("KEY INSIGHT")
    print('='*70)
    print(f"\nAttention parameter reduction: {param_reduction:.1f}%")
    print(f"  Standard: {std_attn:,} params (Q, K, V projections)")
    print(f"  Pure Golden: {pure_attn:,} params (T1, T2 only - NO V)")


if __name__ == "__main__":
    main()
