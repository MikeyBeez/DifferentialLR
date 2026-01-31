#!/usr/bin/env python3
"""
The Golden Engram: Geometric Routing in Semi-Deterministic Neural Networks

This implements the O(N) Golden Ratio attention that replaces Q·K similarity
with deterministic geometric binding: address = T1 + φ·T2

Multi-head uses different powers of φ (φ¹, φ², φ³, φ⁴) to create a
multi-resolution geometric lattice.

Key insight: The routing is mathematical truth, not learned behavior.
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

# The Golden Ratio
PHI = (1 + math.sqrt(5)) / 2  # ≈ 1.618033988749895


class GoldenEngramAttention(nn.Module):
    """
    O(N) Golden Ratio Attention - No similarity search.

    Instead of Q @ K.T (O(N²)), we compute:
        address = T1 + φ · shift(T2)

    This is direct geometric addressing, not search.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.phi = PHI
        self.embed_dim = embed_dim

        # Project to create semantic directions
        self.t1_proj = nn.Linear(embed_dim, embed_dim)
        self.t2_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x: [Batch, Seq_Len, Embed_Dim]

        # Generate semantic directions
        t1 = self.t1_proj(x)
        t2 = self.t2_proj(x)

        # Shift t2 to create next-token relationship
        t2_shifted = torch.roll(t2, shifts=1, dims=1)
        t2_shifted[:, 0, :] = 0  # Zero out first token's "previous"

        # The Golden Engram: T1 + φ·T2
        # This replaces the entire attention matrix
        engram_address = t1 + (self.phi * t2_shifted)

        # Project back to semantic space
        return self.out_proj(engram_address)


class MultiHeadGoldenEngram(nn.Module):
    """
    Multi-head Golden Engram with different powers of φ per head.

    Head 1: φ¹ = 1.618...
    Head 2: φ² = 2.618...
    Head 3: φ³ = 4.236...
    Head 4: φ⁴ = 6.854...

    This creates a multi-resolution geometric lattice.
    Different heads "see" different geometric relationships.
    """
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Compute powers of φ for each head
        self.phi_powers = [PHI ** (i + 1) for i in range(num_heads)]

        # Projections
        self.t1_proj = nn.Linear(embed_dim, embed_dim)
        self.t2_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, D = x.shape

        # Project
        t1 = self.t1_proj(x).view(B, N, self.num_heads, self.head_dim)
        t2 = self.t2_proj(x).view(B, N, self.num_heads, self.head_dim)

        # Shift t2
        t2_shifted = torch.roll(t2, shifts=1, dims=1)
        t2_shifted[:, 0, :, :] = 0

        # Apply different φ powers to each head
        outputs = []
        for h in range(self.num_heads):
            phi_h = self.phi_powers[h]
            engram = t1[:, :, h, :] + phi_h * t2_shifted[:, :, h, :]
            outputs.append(engram)

        # Concatenate heads
        out = torch.stack(outputs, dim=2)  # [B, N, H, D]
        out = out.view(B, N, D)

        return self.out_proj(out)


class BidirectionalGoldenEngram(nn.Module):
    """
    Bidirectional Golden Engram - looks forward AND backward.

    address = T_current + φ·T_prev + φ⁻¹·T_next

    This captures both past and future context in O(N).
    """
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.phi = PHI
        self.phi_inv = 1 / PHI  # ≈ 0.618

        self.t_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, D = x.shape

        t = self.t_proj(x).view(B, N, self.num_heads, self.head_dim)

        # Shift for previous and next
        t_prev = torch.roll(t, shifts=1, dims=1)
        t_prev[:, 0, :, :] = 0

        t_next = torch.roll(t, shifts=-1, dims=1)
        t_next[:, -1, :, :] = 0

        # Bidirectional engram
        engram = t + self.phi * t_prev + self.phi_inv * t_next

        out = engram.view(B, N, D)
        return self.out_proj(out)


class GoldenEngramWithMemory(nn.Module):
    """
    Golden Engram with accumulated memory state.

    Combines local (shift-based) and global (accumulated) context.
    Similar to how Mamba accumulates state.
    """
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.phi = PHI

        self.t1_proj = nn.Linear(embed_dim, embed_dim)
        self.t2_proj = nn.Linear(embed_dim, embed_dim)
        self.gate_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Decay factor for memory accumulation
        self.decay = nn.Parameter(torch.tensor(0.9))

    def forward(self, x):
        B, N, D = x.shape

        t1 = self.t1_proj(x)
        t2 = self.t2_proj(x)
        gate = torch.sigmoid(self.gate_proj(x))

        # Local engram (shift-based)
        t2_shifted = torch.roll(t2, shifts=1, dims=1)
        t2_shifted[:, 0, :] = 0
        local_engram = t1 + self.phi * t2_shifted

        # Global memory (accumulated with decay)
        memory = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        global_context = []

        for i in range(N):
            memory = self.decay * memory + (1 - self.decay) * t2[:, i, :]
            global_context.append(memory.clone())

        global_context = torch.stack(global_context, dim=1)  # [B, N, D]

        # Mix local and global
        out = gate * local_engram + (1 - gate) * global_context

        return self.out_proj(out)


class StandardAttention(nn.Module):
    """Standard attention for comparison."""
    def __init__(self, embed_dim, num_heads=4):
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
    """Minimal transformer for testing."""
    def __init__(self, vocab_size, dim, num_layers, num_heads, attention_type='standard'):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(1024, dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if attention_type == 'standard':
                attn = StandardAttention(dim, num_heads)
            elif attention_type == 'golden_engram':
                attn = GoldenEngramAttention(dim)
            elif attention_type == 'multihead_golden':
                attn = MultiHeadGoldenEngram(dim, num_heads)
            elif attention_type == 'bidirectional_golden':
                attn = BidirectionalGoldenEngram(dim, num_heads)
            elif attention_type == 'golden_memory':
                attn = GoldenEngramWithMemory(dim, num_heads)
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


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n" + "=" * 70)
    print("THE GOLDEN ENGRAM TEST")
    print("=" * 70)
    print(f"\nφ = {PHI:.10f}")
    print(f"φ² = {PHI**2:.10f}")
    print(f"φ³ = {PHI**3:.10f}")
    print(f"φ⁴ = {PHI**4:.10f}")
    print("\nHypothesis: Geometric routing is mathematical truth, not learned behavior")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_loader, val_loader = get_wikitext_data(tokenizer)

    results = {}
    params = {}

    attention_types = [
        ('standard', 'Standard Q·K Attention (O(N²))'),
        ('golden_engram', 'Golden Engram (O(N), single head)'),
        ('multihead_golden', 'Multi-Head Golden (φ¹,φ²,φ³,φ⁴)'),
        ('bidirectional_golden', 'Bidirectional Golden (past + future)'),
        ('golden_memory', 'Golden + Memory (local + global)'),
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

        params[name] = count_params(model)
        print(f"Parameters: {params[name]:,}")

        results[name] = train_and_eval(model, train_loader, val_loader, device, epochs=5)
        del model

    print("\n" + "=" * 70)
    print("RESULTS: THE GOLDEN ENGRAM")
    print("=" * 70)
    print(f"\n{'Method':<45} {'PPL':>8} {'Params':>12} {'Complexity':>12}")
    print("-" * 80)

    complexities = {
        'Standard Q·K Attention (O(N²))': 'O(N²)',
        'Golden Engram (O(N), single head)': 'O(N)',
        'Multi-Head Golden (φ¹,φ²,φ³,φ⁴)': 'O(N)',
        'Bidirectional Golden (past + future)': 'O(N)',
        'Golden + Memory (local + global)': 'O(N)',
    }

    for name, ppl in results.items():
        complexity = complexities.get(name, '?')
        print(f"{name:<45} {ppl:>8.1f} {params[name]:>12,} {complexity:>12}")

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    standard_ppl = results['Standard Q·K Attention (O(N²))']
    best_golden = min(v for k, v in results.items() if 'Golden' in k)
    best_golden_name = [k for k, v in results.items() if v == best_golden and 'Golden' in k][0]

    print(f"\nStandard Attention: {standard_ppl:.1f} PPL (O(N²))")
    print(f"Best Golden Engram: {best_golden:.1f} PPL (O(N))")

    if best_golden < standard_ppl * 2:
        print(f"\n✓ Golden Engram is VIABLE!")
        print(f"  The 'search' was just overhead - geometric routing works.")
        print(f"  Memory scales O(N) instead of O(N²)")
    else:
        print(f"\n✗ Golden Engram needs more work")
        print(f"  The pairwise interaction in Q·K provides signal we're losing")


if __name__ == "__main__":
    main()
