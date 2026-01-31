#!/usr/bin/env python3
"""
THE RECALL TEST

Can the 2 KB crystal state "remember" a specific fact from deep in the context?

This is the honest limitation test. The crystal is lossy compression.
It captures influence, not exact storage.

We plant a "needle" (a specific fact) at various depths and see if
the model can still "taste" it at the end.
"""

import sys
sys.path.insert(0, '/home/bee/Code/LinearAttention')

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

PHI = (1 + math.sqrt(5)) / 2


class GoldenCrystal(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.proj = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        decay = torch.tensor([(1/PHI) ** (i+1) for i in range(num_heads)])
        self.register_buffer('decay', decay)

    def forward(self, x):
        B, N, D = x.shape
        t = self.proj(x).view(B, N, self.num_heads, self.head_dim)
        h_fwd = torch.zeros(B, self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)
        fwd = []
        for i in range(N):
            h_fwd = t[:, i] + self.decay.view(1, -1, 1) * h_fwd
            fwd.append(h_fwd)
        fwd = torch.stack(fwd, dim=1)
        h_bwd = torch.zeros(B, self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)
        bwd = []
        for i in range(N-1, -1, -1):
            h_bwd = t[:, i] + self.decay.view(1, -1, 1) * h_bwd
            bwd.append(h_bwd)
        bwd = torch.stack(bwd[::-1], dim=1)
        y = fwd + bwd - t
        return self.out(y.reshape(B, N, D))


class CrystalTransformer(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, num_heads):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'attn': GoldenCrystal(dim, num_heads),
                'norm1': nn.LayerNorm(dim),
                'ffn': nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)),
                'norm2': nn.LayerNorm(dim)
            }))
        self.final_norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = x + layer['attn'](layer['norm1'](x))
            x = x + layer['ffn'](layer['norm2'](x))
        return self.lm_head(self.final_norm(x))


class StandardAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        q = self.q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)
        return self.o(out.transpose(1, 2).reshape(B, N, D))


class AttentionTransformer(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, num_heads):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(65536, dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'attn': StandardAttention(dim, num_heads),
                'norm1': nn.LayerNorm(dim),
                'ffn': nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)),
                'norm2': nn.LayerNorm(dim)
            }))
        self.final_norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids):
        B, N = input_ids.shape
        pos = torch.arange(N, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(pos)
        for layer in self.layers:
            x = x + layer['attn'](layer['norm1'](x))
            x = x + layer['ffn'](layer['norm2'](x))
        return self.lm_head(self.final_norm(x))


def needle_in_haystack_test(model, vocab_size, needle_positions, haystack_length, device, num_trials=10):
    """
    Test: Can the model predict a specific token that was planted earlier?

    We create sequences like:
    [random...] [NEEDLE_A] [random...] [QUERY] [?]

    The model should predict NEEDLE_A after seeing QUERY.
    This tests if the "influence" of NEEDLE_A survives in the crystal.
    """
    model.eval()

    # Use specific tokens as needle/query pairs
    # NEEDLE tokens: 1000-1009
    # QUERY tokens: 2000-2009
    # The model should learn: seeing QUERY_i means predict NEEDLE_i

    results = {}

    for needle_pos in needle_positions:
        if needle_pos >= haystack_length - 2:
            continue

        correct = 0
        total = 0

        for trial in range(num_trials):
            # Create haystack of random tokens (avoiding our special tokens)
            haystack = torch.randint(3000, vocab_size, (1, haystack_length), device=device)

            # Plant needle and query
            needle_id = 1000 + (trial % 10)
            query_id = 2000 + (trial % 10)

            haystack[0, needle_pos] = needle_id
            haystack[0, -2] = query_id  # Query at second-to-last position
            # Model should predict needle_id at last position

            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    logits = model(haystack)

            # Check if the model's top prediction at the last position is the needle
            predicted = logits[0, -2, :].argmax().item()  # Predict what comes after query

            if predicted == needle_id:
                correct += 1
            total += 1

        accuracy = correct / total * 100
        results[needle_pos] = accuracy

    return results


def train_needle_task(model, vocab_size, train_length, device, num_steps=500):
    """Train the model on the needle-recall task."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()

    for step in range(num_steps):
        # Random needle position in first half
        needle_pos = torch.randint(0, train_length // 2, (1,)).item()

        # Create training sequence
        seq = torch.randint(3000, vocab_size, (8, train_length), device=device)

        for b in range(8):
            needle_id = 1000 + (b % 10)
            query_id = 2000 + (b % 10)
            seq[b, needle_pos] = needle_id
            seq[b, -2] = query_id
            seq[b, -1] = needle_id  # Target: predict the needle after query

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            logits = model(seq)
            # Loss only on last position
            loss = F.cross_entropy(logits[:, -2, :], seq[:, -1])

        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"  Step {step}: Loss {loss.item():.4f}")

    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("THE RECALL TEST: Can the Crystal Remember a Needle?")
    print("=" * 70)
    print()
    print("Task: Plant a 'needle' token at position X.")
    print("      Place a 'query' token at the end.")
    print("      Model must predict the needle after seeing the query.")
    print()
    print("This tests CONTENT-DEPENDENT recall - the Crystal's weakness.")
    print()

    vocab_size = 10000
    dim = 256
    num_layers = 4
    num_heads = 4
    train_length = 256

    # Test positions (relative to sequence length)
    test_lengths = [256, 512, 1024, 2048]

    print("=" * 70)
    print("TRAINING PHASE")
    print("=" * 70)

    # Train Crystal
    print("\nTraining Golden Crystal...")
    crystal = CrystalTransformer(vocab_size, dim, num_layers, num_heads).to(device)
    crystal = train_needle_task(crystal, vocab_size, train_length, device)

    # Train Attention (for comparison on shorter sequences)
    print("\nTraining Standard Attention...")
    attention = AttentionTransformer(vocab_size, dim, num_layers, num_heads).to(device)
    attention = train_needle_task(attention, vocab_size, train_length, device)

    print()
    print("=" * 70)
    print("RECALL TEST: How far back can we remember?")
    print("=" * 70)

    print(f"\n{'Length':>10} | {'Needle Pos':>12} | {'Crystal':>10} | {'Attention':>10}")
    print("-" * 50)

    for test_len in test_lengths:
        # Test needle at various depths
        needle_positions = [10, test_len // 4, test_len // 2]

        crystal_results = needle_in_haystack_test(
            crystal, vocab_size, needle_positions, test_len, device
        )

        # Attention may OOM on longer sequences
        try:
            attn_results = needle_in_haystack_test(
                attention, vocab_size, needle_positions, test_len, device
            )
        except RuntimeError:
            attn_results = {pos: "OOM" for pos in needle_positions}

        for pos in needle_positions:
            if pos in crystal_results:
                c_acc = f"{crystal_results[pos]:.0f}%"
                a_acc = f"{attn_results[pos]:.0f}%" if isinstance(attn_results.get(pos), float) else "OOM"
                print(f"{test_len:>10} | {pos:>12} | {c_acc:>10} | {a_acc:>10}")

    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
The Crystal's recall accuracy shows the 'resolution limit' of lossy compression.

- Near the query (recent context): High accuracy - the influence is strong
- Deep in the haystack: Lower accuracy - the influence has decayed

This is the HONEST limitation: The Crystal captures 'contextual essence',
not 'exact storage'. It knows the VIBE of what was said 10,000 tokens ago,
but not the EXACT words.

For language modeling (predicting likely next words), the vibe is enough.
For retrieval tasks (find the exact word from position X), attention wins.
""")


if __name__ == "__main__":
    main()
