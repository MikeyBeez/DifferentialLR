#!/usr/bin/env python3
"""
Where Is The Magic?

The model solves paired recall (query != key) with 95%+ accuracy,
but layer 1 shows NO manufactured similarity. Where does it happen?

Hypotheses:
1. Position-based: "Always attend to position 0"
2. Deeper layers: Layer 2+ does the content matching
3. Value/FFN routing: The V projection or FFN does the work

This experiment probes each layer to find where the magic happens.

Usage:
    python experiments/where_is_the_magic_test.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')


@dataclass
class Config:
    vocab_size: int = 256
    dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    head_dim: int = 32


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.dim = config.dim

        self.W_q = nn.Linear(config.dim, config.dim)
        self.W_k = nn.Linear(config.dim, config.dim)
        self.W_v = nn.Linear(config.dim, config.dim)
        self.W_o = nn.Linear(config.dim, config.dim)

        self.scale = self.head_dim ** -0.5
        self.last_attn_weights = None

    def forward(self, x, return_attn=False):
        B, T, C = x.shape

        q = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        self.last_attn_weights = attn.detach()

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.W_o(out)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.dim)
        self.attn = Attention(config)
        self.norm2 = nn.LayerNorm(config.dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.dim, config.dim * 4),
            nn.GELU(),
            nn.Linear(config.dim * 4, config.dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.dim)
        self.output = nn.Linear(config.dim, config.vocab_size)

    def forward(self, x):
        h = self.embedding(x)
        for block in self.blocks:
            h = block(h)
        return self.output(self.norm(h))

    def get_attention_patterns(self, x):
        """Run forward and return attention weights from each layer."""
        h = self.embedding(x)
        patterns = []
        for block in self.blocks:
            h = block(h)
            patterns.append(block.attn.last_attn_weights)
        return patterns


def generate_paired_batch(batch_size, seq_len, vocab_size, device):
    KEY_START, KEY_END = 10, 60
    QUERY_START, QUERY_END = 60, 110
    VAL_START, VAL_END = 110, 210
    NOISE_START = 210

    seqs, keys, queries, vals = [], [], [], []

    for _ in range(batch_size):
        seq = torch.zeros(seq_len, dtype=torch.long, device=device)
        pair_idx = torch.randint(0, 50, (1,), device=device).item()

        key = KEY_START + pair_idx
        query = QUERY_START + pair_idx
        val = torch.randint(VAL_START, VAL_END, (1,), device=device).item()

        seq[0] = key
        seq[1] = val
        seq[2:-1] = torch.randint(NOISE_START, vocab_size, (seq_len - 3,), device=device)
        seq[-1] = query

        seqs.append(seq)
        keys.append(key)
        queries.append(query)
        vals.append(val)

    return (torch.stack(seqs),
            torch.tensor(keys, device=device),
            torch.tensor(queries, device=device),
            torch.tensor(vals, device=device))


def train_model(config, seq_len=64, epochs=20):
    print("Training model...")

    model = Model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        model.train()
        total_correct, total = 0, 0

        for _ in range(100):
            seqs, _, _, vals = generate_paired_batch(32, seq_len, config.vocab_size, device)

            optimizer.zero_grad()
            logits = model(seqs)
            loss = F.cross_entropy(logits[:, -1], vals)
            loss.backward()
            optimizer.step()

            total_correct += (logits[:, -1].argmax(-1) == vals).sum().item()
            total += 32

        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: {total_correct/total:.1%}")

        if total_correct / total > 0.95:
            print(f"  Solved at epoch {epoch}")
            break

    return model


@torch.no_grad()
def analyze_attention_patterns(model, config, seq_len=64, num_batches=20):
    """Analyze where the model attends at the query position."""
    print("\n" + "="*60)
    print("ATTENTION PATTERN ANALYSIS")
    print("="*60)
    print("Where does position -1 (query) attend?")

    model.eval()

    # Collect attention to position 0 (key) and position 1 (value)
    attn_to_key = {i: [] for i in range(config.num_layers)}
    attn_to_val = {i: [] for i in range(config.num_layers)}
    attn_to_noise = {i: [] for i in range(config.num_layers)}

    for _ in range(num_batches):
        seqs, _, _, _ = generate_paired_batch(32, seq_len, config.vocab_size, device)

        patterns = model.get_attention_patterns(seqs)

        for layer_idx, attn in enumerate(patterns):
            # attn shape: (B, num_heads, T, T)
            # Get attention FROM position -1 (query)
            query_attn = attn[:, :, -1, :]  # (B, num_heads, T)

            # Average over heads
            query_attn = query_attn.mean(dim=1)  # (B, T)

            # Attention to position 0 (key), 1 (value), and average of noise
            attn_to_key[layer_idx].append(query_attn[:, 0].cpu())
            attn_to_val[layer_idx].append(query_attn[:, 1].cpu())
            attn_to_noise[layer_idx].append(query_attn[:, 2:-1].mean(dim=1).cpu())

    print(f"\n{'Layer':<8} {'Attn to Key':<15} {'Attn to Val':<15} {'Attn to Noise':<15}")
    print("-" * 55)

    for layer in range(config.num_layers):
        key_attn = torch.cat(attn_to_key[layer]).mean().item()
        val_attn = torch.cat(attn_to_val[layer]).mean().item()
        noise_attn = torch.cat(attn_to_noise[layer]).mean().item()

        marker = " ← KEY FOUND" if key_attn > 0.3 else ""
        marker = " ← VAL FOUND" if val_attn > 0.3 else marker

        print(f"{layer:<8} {key_attn:<15.3f} {val_attn:<15.3f} {noise_attn:<15.3f}{marker}")

    return attn_to_key, attn_to_val, attn_to_noise


@torch.no_grad()
def test_position_vs_content(model, config, seq_len=64):
    """Test if the model uses position or content."""
    print("\n" + "="*60)
    print("POSITION vs CONTENT TEST")
    print("="*60)

    model.eval()

    # Normal task
    seqs, keys, queries, vals = generate_paired_batch(100, seq_len, config.vocab_size, device)
    logits = model(seqs)
    normal_acc = (logits[:, -1].argmax(-1) == vals).float().mean().item()
    print(f"\nNormal accuracy: {normal_acc:.1%}")

    # Swap key and noise positions
    # If model uses position, it will fail
    # If model uses content, it should still work
    seqs_swapped = seqs.clone()
    for i in range(len(seqs_swapped)):
        # Swap position 0 (key) with position 10 (noise)
        seqs_swapped[i, 0], seqs_swapped[i, 10] = seqs_swapped[i, 10].item(), seqs_swapped[i, 0].item()

    logits_swapped = model(seqs_swapped)
    swapped_acc = (logits_swapped[:, -1].argmax(-1) == vals).float().mean().item()
    print(f"Key moved to position 10: {swapped_acc:.1%}")

    if swapped_acc < 0.2:
        print("\n→ Model uses POSITION (failed when key moved)")
    elif swapped_acc > 0.8:
        print("\n→ Model uses CONTENT (still works when key moved)")
    else:
        print("\n→ Model uses BOTH position and content")


@torch.no_grad()
def analyze_what_position_1_carries(model, config, seq_len=64):
    """
    The model might be doing:
    1. Query → attend to position 0 (key) → but key alone doesn't have value info
    2. Actually: Query → attend to position 1 (value) directly?

    Or multi-hop:
    1. Layer 1: Position 1 attends to position 0, enriches with key info
    2. Layer 2+: Query attends to position 1 (now has both key and value)
    """
    print("\n" + "="*60)
    print("INFORMATION FLOW ANALYSIS")
    print("="*60)

    model.eval()

    seqs, _, _, vals = generate_paired_batch(32, seq_len, config.vocab_size, device)
    patterns = model.get_attention_patterns(seqs)

    print("\nAttention FROM position 1 (value) TO position 0 (key):")
    print("-" * 40)

    for layer_idx, attn in enumerate(patterns):
        # Attention from position 1 to position 0
        val_to_key = attn[:, :, 1, 0].mean().item()
        print(f"Layer {layer_idx}: {val_to_key:.3f}")

    print("\nInterpretation:")
    print("If position 1 attends strongly to position 0 in early layers,")
    print("it's enriching the value representation with key information.")
    print("Then later layers can do: query → position 1 (which now 'knows' the key)")


def main():
    print("="*60)
    print("WHERE IS THE MAGIC?")
    print("Finding how the model solves paired recall")
    print("="*60)

    config = Config()
    seq_len = 64

    model = train_model(config, seq_len)

    # Analyze
    analyze_attention_patterns(model, config, seq_len)
    test_position_vs_content(model, config, seq_len)
    analyze_what_position_1_carries(model, config, seq_len)

    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("""
The model likely solves this via POSITION, not manufactured similarity.

For the "it"/"cat" problem in real language, the model needs:
1. Pre-training that establishes "pronouns refer to nouns"
2. The projection matrices learn during pre-training
3. By inference time, Q("it") and K("cat") are already aligned

Our toy task doesn't force content-based matching because
the key is always at position 0. The model takes the shortcut.
""")


if __name__ == "__main__":
    main()
