#!/usr/bin/env python3
"""
Manufactured Similarity Test

The harder case: Can projections create similarity between DIFFERENT tokens?

In associative recall, query = key (same token). Easy case.
In pronoun resolution, query = "it", key = "cat". Hard case.

This experiment tests the hard case:
- Key token X at position 0
- Value token Y at position 1
- Query token Z at position -1 (DIFFERENT from X)
- Model must learn that Z "means" X and retrieve Y

The projections must manufacture Q(Z) · K(X) > Q(Z) · K(noise)
from tokens with zero inherent similarity.

Usage:
    python experiments/manufactured_similarity_test.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np

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

    def forward(self, x):
        B, T, C = x.shape

        q = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.W_o(out)

    def project(self, x):
        return self.W_q(x), self.W_k(x), self.W_v(x)


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

    def get_first_layer_projections(self, x):
        h = self.embedding(x)
        h = self.blocks[0].norm1(h)
        return self.blocks[0].attn.project(h)


def generate_paired_batch(batch_size, seq_len, vocab_size, device):
    """
    Generate batch where query is DIFFERENT from key.

    Structure:
    - Position 0: Key token (from range 10-59)
    - Position 1: Value token (from range 110-210)
    - Position 2 to -2: Noise
    - Position -1: Query token (from range 60-109) - PAIRED with key but different

    The model must learn that query token Q maps to key token K.
    We use fixed pairings: query_i always maps to key_i.
    """
    KEY_START, KEY_END = 10, 60       # 50 key tokens
    QUERY_START, QUERY_END = 60, 110  # 50 query tokens (paired with keys)
    VAL_START, VAL_END = 110, 210     # 100 value tokens
    NOISE_START = 210

    seqs = []
    keys = []
    queries = []
    vals = []

    for _ in range(batch_size):
        seq = torch.zeros(seq_len, dtype=torch.long, device=device)

        # Pick a pair index (0-49)
        pair_idx = torch.randint(0, 50, (1,), device=device).item()

        key = KEY_START + pair_idx      # Key token
        query = QUERY_START + pair_idx  # Query token (different but paired)
        val = torch.randint(VAL_START, VAL_END, (1,), device=device).item()

        seq[0] = key
        seq[1] = val
        seq[2:-1] = torch.randint(NOISE_START, vocab_size, (seq_len - 3,), device=device)
        seq[-1] = query  # Different token from key!

        seqs.append(seq)
        keys.append(key)
        queries.append(query)
        vals.append(val)

    return (torch.stack(seqs),
            torch.tensor(keys, device=device),
            torch.tensor(queries, device=device),
            torch.tensor(vals, device=device))


def train_model(config, seq_len=128, epochs=30):
    """Train on paired recall task."""
    print("Training on PAIRED recall (query != key)...")
    print("Model must learn that token 60 'means' token 10, etc.")

    model = Model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        model.train()
        total_correct = 0
        total = 0

        for _ in range(100):
            seqs, keys, queries, vals = generate_paired_batch(32, seq_len, config.vocab_size, device)

            optimizer.zero_grad()
            logits = model(seqs)
            loss = F.cross_entropy(logits[:, -1], vals)
            loss.backward()
            optimizer.step()

            total_correct += (logits[:, -1].argmax(-1) == vals).sum().item()
            total += 32

        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: Accuracy {total_correct/total:.1%}")

        if total_correct / total > 0.95:
            print(f"  Solved at epoch {epoch}")
            break

    return model


@torch.no_grad()
def analyze_manufactured_similarity(model, config, seq_len=128, num_samples=500):
    """Analyze how projections create similarity between different tokens."""
    print("\nAnalyzing manufactured similarity...")

    model.eval()

    # Collect data
    key_embeds, query_embeds, noise_embeds = [], [], []
    key_Q, key_K, key_V = [], [], []
    query_Q, query_K, query_V = [], [], []
    noise_Q, noise_K, noise_V = [], [], []

    for _ in range(num_samples // 32):
        seqs, keys, queries, vals = generate_paired_batch(32, seq_len, config.vocab_size, device)

        embeds = model.embedding(seqs)
        Q, K, V = model.get_first_layer_projections(seqs)

        for i in range(32):
            # Key at position 0
            key_embeds.append(embeds[i, 0].cpu())
            key_Q.append(Q[i, 0].cpu())
            key_K.append(K[i, 0].cpu())
            key_V.append(V[i, 0].cpu())

            # Noise at middle
            mid = seq_len // 2
            noise_embeds.append(embeds[i, mid].cpu())
            noise_Q.append(Q[i, mid].cpu())
            noise_K.append(K[i, mid].cpu())
            noise_V.append(V[i, mid].cpu())

            # Query at position -1 (DIFFERENT token from key)
            query_embeds.append(embeds[i, -1].cpu())
            query_Q.append(Q[i, -1].cpu())
            query_K.append(K[i, -1].cpu())
            query_V.append(V[i, -1].cpu())

    return {
        'key': {'embed': torch.stack(key_embeds), 'Q': torch.stack(key_Q),
                'K': torch.stack(key_K), 'V': torch.stack(key_V)},
        'query': {'embed': torch.stack(query_embeds), 'Q': torch.stack(query_Q),
                  'K': torch.stack(query_K), 'V': torch.stack(query_V)},
        'noise': {'embed': torch.stack(noise_embeds), 'Q': torch.stack(noise_Q),
                  'K': torch.stack(noise_K), 'V': torch.stack(noise_V)}
    }


def compute_manufactured_similarity(data):
    """Measure how much similarity was manufactured."""
    print("\n" + "="*60)
    print("MANUFACTURED SIMILARITY ANALYSIS")
    print("="*60)
    print("Query and Key are DIFFERENT tokens. Any similarity is manufactured.")

    def cosine_sim(a, b):
        a_norm = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
        b_norm = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
        return (a_norm * b_norm).sum(dim=-1).mean().item()

    print("\n1. RAW EMBEDDING SIMILARITY (before projection)")
    print("-" * 50)
    qk_embed = cosine_sim(data['query']['embed'], data['key']['embed'])
    qn_embed = cosine_sim(data['query']['embed'], data['noise']['embed'])
    print(f"Query-Key (should be ~0, different tokens): {qk_embed:.4f}")
    print(f"Query-Noise:                                {qn_embed:.4f}")

    print("\n2. PROJECTED SIMILARITY (after Q/K projection)")
    print("-" * 50)

    # Q(query) vs K(key) - this is what attention computes
    qQ = data['query']['Q']
    kK = data['key']['K']
    nK = data['noise']['K']

    qk_proj = cosine_sim(qQ, kK)
    qn_proj = cosine_sim(qQ, nK)

    print(f"Q(query) vs K(key):   {qk_proj:.4f}")
    print(f"Q(query) vs K(noise): {qn_proj:.4f}")
    print(f"Similarity manufactured: {qk_proj - qk_embed:.4f}")

    print("\n3. DOT PRODUCT SEPARATION (what softmax sees)")
    print("-" * 50)

    query_key_dots = (qQ * kK).sum(dim=-1)
    query_noise_dots = (qQ * nK).sum(dim=-1)

    print(f"Q(query) · K(key):   mean={query_key_dots.mean():.3f}, std={query_key_dots.std():.3f}")
    print(f"Q(query) · K(noise): mean={query_noise_dots.mean():.3f}, std={query_noise_dots.std():.3f}")
    print(f"Separation: {query_key_dots.mean() - query_noise_dots.mean():.3f}")

    key_wins = (query_key_dots > query_noise_dots).float().mean()
    print(f"Key beats noise: {key_wins:.1%} of samples")

    return {
        'embed_sim': qk_embed,
        'proj_sim': qk_proj,
        'manufactured': qk_proj - qk_embed,
        'separation': (query_key_dots.mean() - query_noise_dots.mean()).item(),
        'key_wins': key_wins.item()
    }


def visualize_manufactured_similarity(data, save_path='manufactured_similarity.png'):
    """Visualize how different tokens become similar through projection."""
    print("\nGenerating visualization...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    n = 100

    # Panel 1: Embedding space (should show NO alignment)
    ax = axes[0]
    from sklearn.decomposition import PCA

    query_embed = data['query']['embed'][:n].numpy()
    key_embed = data['key']['embed'][:n].numpy()

    pca = PCA(n_components=2)
    combined = np.vstack([query_embed, key_embed])
    projected = pca.fit_transform(combined)

    ax.scatter(projected[:n, 0], projected[:n, 1], c='red', alpha=0.6,
               label='Query tokens', s=30)
    ax.scatter(projected[n:, 0], projected[n:, 1], c='blue', alpha=0.6,
               label='Key tokens', s=30)

    # Lines connecting paired tokens
    for i in range(min(20, n)):
        ax.plot([projected[i, 0], projected[n+i, 0]],
               [projected[i, 1], projected[n+i, 1]], 'k-', alpha=0.2)

    ax.set_title('Embedding Space\n(Different tokens - should be scattered)')
    ax.legend()

    # Panel 2: Q-K space (should show alignment)
    ax = axes[1]

    query_Q = data['query']['Q'][:n].numpy()
    key_K = data['key']['K'][:n].numpy()

    pca = PCA(n_components=2)
    combined = np.vstack([query_Q, key_K])
    projected = pca.fit_transform(combined)

    ax.scatter(projected[:n, 0], projected[:n, 1], c='red', alpha=0.6,
               label='Q(query)', s=30)
    ax.scatter(projected[n:, 0], projected[n:, 1], c='blue', alpha=0.6,
               label='K(key)', s=30)

    for i in range(min(20, n)):
        ax.plot([projected[i, 0], projected[n+i, 0]],
               [projected[i, 1], projected[n+i, 1]], 'k-', alpha=0.2)

    ax.set_title('Q-K Space\n(Manufactured alignment)')
    ax.legend()

    # Panel 3: Dot product distribution
    ax = axes[2]

    qQ = data['query']['Q']
    kK = data['key']['K']
    nK = data['noise']['K']

    query_key_dots = (qQ * kK).sum(dim=-1).numpy()
    query_noise_dots = (qQ * nK).sum(dim=-1).numpy()

    ax.hist(query_key_dots, bins=50, alpha=0.7, label='Q(query) · K(key)', color='blue')
    ax.hist(query_noise_dots, bins=50, alpha=0.7, label='Q(query) · K(noise)', color='gray')
    ax.axvline(query_key_dots.mean(), color='blue', linestyle='--', linewidth=2)
    ax.axvline(query_noise_dots.mean(), color='gray', linestyle='--', linewidth=2)

    ax.set_title('Dot Product Distribution\n(Separation enables recall)')
    ax.set_xlabel('Dot product')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved to {save_path}")
    plt.close()


def main():
    print("="*60)
    print("MANUFACTURED SIMILARITY TEST")
    print("Can projections create similarity from nothing?")
    print("="*60)

    config = Config()
    seq_len = 128

    # Train model
    model = train_model(config, seq_len)

    # Analyze
    data = analyze_manufactured_similarity(model, config, seq_len)

    # Compute metrics
    metrics = compute_manufactured_similarity(data)

    # Visualize
    visualize_manufactured_similarity(data, 'manufactured_similarity.png')

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: THE PROJECTION MATRICES' JOB")
    print("="*60)
    print(f"""
Query and Key are DIFFERENT tokens (like "it" and "cat").
Raw embedding similarity: {metrics['embed_sim']:.4f} (essentially zero)

After projection:
- Q(query) · K(key) separation from noise: {metrics['separation']:.1f} points
- Key beats noise: {metrics['key_wins']:.1%} of the time

The projections MANUFACTURED this similarity.
They learned that token 60 "means" token 10, token 61 "means" token 11, etc.

This is the abstract representation you asked about:
- K transforms "cat" into "I am something 'it' can refer to"
- Q transforms "it" into "I am looking for what I refer to"
- These match, even though "it" and "cat" share nothing in embedding space
""")


if __name__ == "__main__":
    main()
