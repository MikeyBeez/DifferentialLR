#!/usr/bin/env python3
"""
Projection Geometry Visualization

What do Q, K, V projections actually learn for associative recall?

The mystery:
- "it" and "cat" have no inherent similarity
- Yet Q("it") must match K("cat") for coreference
- K transforms tokens into "findable" representations (loses identity)
- V transforms tokens into "retrievable" representations (preserves identity)

This experiment visualizes the geometric structure that emerges:
1. Where do keys, values, and noise land in Q-space vs K-space?
2. How does the query token relate to the key token in projected space?
3. What does V preserve that K destroys?

Usage:
    python experiments/projection_geometry_test.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')


@dataclass
class Config:
    vocab_size: int = 256
    dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    head_dim: int = 32  # dim // num_heads


class Attention(nn.Module):
    """Standard attention with accessible Q, K, V projections."""

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.dim = config.dim

        # Separate projections for analysis
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
        """Get Q, K, V projections without attention."""
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        return q, k, v


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

    def get_embeddings(self, tokens):
        """Get raw embeddings for tokens."""
        return self.embedding(tokens)

    def get_first_layer_projections(self, x):
        """Get Q, K, V from first attention layer."""
        h = self.embedding(x)
        h = self.blocks[0].norm1(h)
        return self.blocks[0].attn.project(h)


def generate_batch(batch_size, seq_len, vocab_size, device):
    """Generate associative recall batch, returning structured info."""
    KEY_START, KEY_END = 10, 110
    VAL_START, VAL_END = 110, 210
    NOISE_START = 210

    seqs = []
    keys = []
    vals = []

    for _ in range(batch_size):
        seq = torch.zeros(seq_len, dtype=torch.long, device=device)
        key = torch.randint(KEY_START, KEY_END, (1,), device=device)
        val = torch.randint(VAL_START, VAL_END, (1,), device=device)

        seq[0] = key
        seq[1] = val
        seq[2:-1] = torch.randint(NOISE_START, vocab_size, (seq_len - 3,), device=device)
        seq[-1] = key  # Query

        seqs.append(seq)
        keys.append(key)
        vals.append(val)

    return torch.stack(seqs), torch.stack(keys).squeeze(), torch.stack(vals).squeeze()


def train_model(config, seq_len=128, epochs=20):
    """Train model on associative recall."""
    print("Training model on associative recall...")

    model = Model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        model.train()
        total_correct = 0
        total = 0

        for _ in range(100):
            seqs, keys, vals = generate_batch(32, seq_len, config.vocab_size, device)

            optimizer.zero_grad()
            logits = model(seqs)
            loss = F.cross_entropy(logits[:, -1], vals)
            loss.backward()
            optimizer.step()

            total_correct += (logits[:, -1].argmax(-1) == vals).sum().item()
            total += 32

        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: Accuracy {total_correct/total:.1%}")

        if total_correct / total > 0.99:
            print(f"  Solved at epoch {epoch}")
            break

    return model


@torch.no_grad()
def analyze_projections(model, config, seq_len=128, num_samples=500):
    """Analyze what Q, K, V projections learn."""
    print("\nAnalyzing projection geometry...")

    model.eval()

    # Collect projections for different token types
    key_embeds = []      # Raw embeddings of key tokens
    val_embeds = []      # Raw embeddings of value tokens
    noise_embeds = []    # Raw embeddings of noise tokens
    query_embeds = []    # Raw embeddings of query tokens (same as key, different position)

    key_Q, key_K, key_V = [], [], []
    val_Q, val_K, val_V = [], [], []
    noise_Q, noise_K, noise_V = [], [], []
    query_Q, query_K, query_V = [], [], []

    for _ in range(num_samples // 32):
        seqs, keys, vals = generate_batch(32, seq_len, config.vocab_size, device)

        # Get raw embeddings
        embeds = model.get_embeddings(seqs)

        # Get projections from first layer
        Q, K, V = model.get_first_layer_projections(seqs)

        for i in range(32):
            # Key position (0)
            key_embeds.append(embeds[i, 0].cpu())
            key_Q.append(Q[i, 0].cpu())
            key_K.append(K[i, 0].cpu())
            key_V.append(V[i, 0].cpu())

            # Value position (1)
            val_embeds.append(embeds[i, 1].cpu())
            val_Q.append(Q[i, 1].cpu())
            val_K.append(K[i, 1].cpu())
            val_V.append(V[i, 1].cpu())

            # Noise position (middle)
            mid = seq_len // 2
            noise_embeds.append(embeds[i, mid].cpu())
            noise_Q.append(Q[i, mid].cpu())
            noise_K.append(K[i, mid].cpu())
            noise_V.append(V[i, mid].cpu())

            # Query position (last) - same token as key, different position
            query_embeds.append(embeds[i, -1].cpu())
            query_Q.append(Q[i, -1].cpu())
            query_K.append(K[i, -1].cpu())
            query_V.append(V[i, -1].cpu())

    return {
        'key': {
            'embed': torch.stack(key_embeds),
            'Q': torch.stack(key_Q),
            'K': torch.stack(key_K),
            'V': torch.stack(key_V)
        },
        'val': {
            'embed': torch.stack(val_embeds),
            'Q': torch.stack(val_Q),
            'K': torch.stack(val_K),
            'V': torch.stack(val_V)
        },
        'noise': {
            'embed': torch.stack(noise_embeds),
            'Q': torch.stack(noise_Q),
            'K': torch.stack(noise_K),
            'V': torch.stack(noise_V)
        },
        'query': {
            'embed': torch.stack(query_embeds),
            'Q': torch.stack(query_Q),
            'K': torch.stack(query_K),
            'V': torch.stack(query_V)
        }
    }


def compute_similarities(data):
    """Compute similarity metrics between token types in different spaces."""
    print("\n" + "="*60)
    print("SIMILARITY ANALYSIS")
    print("="*60)

    def cosine_sim(a, b):
        a_norm = a / a.norm(dim=-1, keepdim=True)
        b_norm = b / b.norm(dim=-1, keepdim=True)
        return (a_norm * b_norm).sum(dim=-1).mean().item()

    def avg_norm(x):
        return x.norm(dim=-1).mean().item()

    spaces = ['embed', 'Q', 'K', 'V']

    print("\n1. QUERY-KEY ALIGNMENT (should be HIGH for recall)")
    print("-" * 50)
    print(f"{'Space':<10} {'Query-Key Sim':<15} {'Query-Noise Sim':<15} {'Ratio':<10}")
    print("-" * 50)

    for space in spaces:
        qk_sim = cosine_sim(data['query'][space], data['key'][space])
        qn_sim = cosine_sim(data['query'][space], data['noise'][space])
        ratio = qk_sim / (qn_sim + 1e-8)
        print(f"{space:<10} {qk_sim:<15.3f} {qn_sim:<15.3f} {ratio:<10.2f}")

    print("\n2. K vs V: FINDABILITY vs CONTENT")
    print("-" * 50)
    print("K should transform for matching, V should preserve identity")
    print(f"{'Comparison':<25} {'K-space sim':<15} {'V-space sim':<15}")
    print("-" * 50)

    # Key-Value distinction
    kv_K = cosine_sim(data['key']['K'], data['val']['K'])
    kv_V = cosine_sim(data['key']['V'], data['val']['V'])
    print(f"{'Key vs Value':<25} {kv_K:<15.3f} {kv_V:<15.3f}")

    # Key-Noise distinction
    kn_K = cosine_sim(data['key']['K'], data['noise']['K'])
    kn_V = cosine_sim(data['key']['V'], data['noise']['V'])
    print(f"{'Key vs Noise':<25} {kn_K:<15.3f} {kn_V:<15.3f}")

    # Query-Key matching
    qk_K = cosine_sim(data['query']['K'], data['key']['K'])
    qk_V = cosine_sim(data['query']['V'], data['key']['V'])
    print(f"{'Query vs Key':<25} {qk_K:<15.3f} {qk_V:<15.3f}")

    print("\n3. NORM ANALYSIS")
    print("-" * 50)
    print(f"{'Token Type':<10} {'Embed':<12} {'Q':<12} {'K':<12} {'V':<12}")
    print("-" * 50)

    for ttype in ['key', 'val', 'noise', 'query']:
        norms = [avg_norm(data[ttype][s]) for s in spaces]
        print(f"{ttype:<10} {norms[0]:<12.3f} {norms[1]:<12.3f} {norms[2]:<12.3f} {norms[3]:<12.3f}")

    print("\n4. THE MAGIC: Q(query) · K(key) vs Q(query) · K(noise)")
    print("-" * 50)

    # Actual dot products (what attention computes)
    qQ = data['query']['Q']
    kK = data['key']['K']
    nK = data['noise']['K']

    # Dot product scores
    query_key_dots = (qQ * kK).sum(dim=-1)
    query_noise_dots = (qQ * nK).sum(dim=-1)

    print(f"Q(query) · K(key):   mean={query_key_dots.mean():.3f}, std={query_key_dots.std():.3f}")
    print(f"Q(query) · K(noise): mean={query_noise_dots.mean():.3f}, std={query_noise_dots.std():.3f}")
    print(f"Separation: {(query_key_dots.mean() - query_noise_dots.mean()):.3f}")

    # What fraction of time does key beat noise?
    key_wins = (query_key_dots > query_noise_dots).float().mean()
    print(f"Key beats noise: {key_wins:.1%} of samples")


def visualize_spaces(data, save_path='projection_geometry.png'):
    """Visualize token distributions in different spaces."""
    print("\nGenerating visualizations...")

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    spaces = ['embed', 'Q', 'K', 'V']
    colors = {'key': 'blue', 'val': 'green', 'noise': 'gray', 'query': 'red'}

    n_samples = 200  # Limit for visualization

    for col, space in enumerate(spaces):
        # Combine all token types for PCA
        all_data = torch.cat([
            data['key'][space][:n_samples],
            data['val'][space][:n_samples],
            data['noise'][space][:n_samples],
            data['query'][space][:n_samples]
        ]).numpy()

        # PCA
        pca = PCA(n_components=2)
        projected = pca.fit_transform(all_data)

        ax = axes[0, col]
        for i, ttype in enumerate(['key', 'val', 'noise', 'query']):
            start = i * n_samples
            end = (i + 1) * n_samples
            ax.scatter(projected[start:end, 0], projected[start:end, 1],
                      c=colors[ttype], alpha=0.5, label=ttype, s=10)

        ax.set_title(f'{space}-space (PCA)')
        ax.legend()
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')

        # t-SNE for bottom row
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        projected_tsne = tsne.fit_transform(all_data)

        ax = axes[1, col]
        for i, ttype in enumerate(['key', 'val', 'noise', 'query']):
            start = i * n_samples
            end = (i + 1) * n_samples
            ax.scatter(projected_tsne[start:end, 0], projected_tsne[start:end, 1],
                      c=colors[ttype], alpha=0.5, label=ttype, s=10)

        ax.set_title(f'{space}-space (t-SNE)')
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved visualization to {save_path}")
    plt.close()


def visualize_query_key_alignment(data, save_path='query_key_alignment.png'):
    """Visualize how query and key tokens align in K-space."""
    print("\nVisualizing query-key alignment...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    n = 100

    # In embedding space
    ax = axes[0]
    query_embed = data['query']['embed'][:n].numpy()
    key_embed = data['key']['embed'][:n].numpy()

    pca = PCA(n_components=2)
    combined = np.vstack([query_embed, key_embed])
    projected = pca.fit_transform(combined)

    ax.scatter(projected[:n, 0], projected[:n, 1], c='red', alpha=0.6, label='Query (pos -1)', s=30)
    ax.scatter(projected[n:, 0], projected[n:, 1], c='blue', alpha=0.6, label='Key (pos 0)', s=30)

    # Draw lines connecting same-token pairs
    for i in range(min(20, n)):
        ax.plot([projected[i, 0], projected[n+i, 0]],
               [projected[i, 1], projected[n+i, 1]], 'k-', alpha=0.2)

    ax.set_title('Embedding Space\n(Same token, different position)')
    ax.legend()

    # In K-space (what attention actually uses for keys)
    ax = axes[1]
    query_Q = data['query']['Q'][:n].numpy()  # Query uses Q projection
    key_K = data['key']['K'][:n].numpy()       # Key uses K projection

    pca = PCA(n_components=2)
    combined = np.vstack([query_Q, key_K])
    projected = pca.fit_transform(combined)

    ax.scatter(projected[:n, 0], projected[:n, 1], c='red', alpha=0.6, label='Q(query)', s=30)
    ax.scatter(projected[n:, 0], projected[n:, 1], c='blue', alpha=0.6, label='K(key)', s=30)

    for i in range(min(20, n)):
        ax.plot([projected[i, 0], projected[n+i, 0]],
               [projected[i, 1], projected[n+i, 1]], 'k-', alpha=0.2)

    ax.set_title('Q-K Space\n(What attention computes)')
    ax.legend()

    # Dot product distribution
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

    ax.set_title('Dot Product Distribution\n(Attention scores before softmax)')
    ax.set_xlabel('Dot product')
    ax.set_ylabel('Count')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved alignment visualization to {save_path}")
    plt.close()


def analyze_v_preservation(data):
    """Analyze whether V preserves token identity better than K."""
    print("\n" + "="*60)
    print("V PRESERVATION ANALYSIS")
    print("="*60)
    print("Does V preserve token identity while K destroys it for matching?")

    # For same tokens at different positions (query = key token)
    # V should give similar representations
    # K should give representations optimized for matching, not identity

    query_V = data['query']['V']
    key_V = data['key']['V']
    query_K = data['query']['K']
    key_K = data['key']['K']
    query_embed = data['query']['embed']
    key_embed = data['key']['embed']

    # These are the SAME tokens at different positions
    # Embedding should be identical (same token)
    embed_sim = F.cosine_similarity(query_embed, key_embed, dim=-1).mean()

    # V should preserve this similarity (same content)
    v_sim = F.cosine_similarity(query_V, key_V, dim=-1).mean()

    # K might transform differently for matching purposes
    k_sim = F.cosine_similarity(query_K, key_K, dim=-1).mean()

    print(f"\nSame token at pos 0 vs pos -1:")
    print(f"  Embedding similarity: {embed_sim:.3f} (should be 1.0)")
    print(f"  V similarity:         {v_sim:.3f}")
    print(f"  K similarity:         {k_sim:.3f}")

    # Now compare to DIFFERENT tokens
    val_V = data['val']['V']
    val_K = data['val']['K']

    v_diff = F.cosine_similarity(query_V, val_V, dim=-1).mean()
    k_diff = F.cosine_similarity(query_K, val_K, dim=-1).mean()

    print(f"\nQuery token vs Value token (different tokens):")
    print(f"  V similarity: {v_diff:.3f}")
    print(f"  K similarity: {k_diff:.3f}")

    print(f"\nInterpretation:")
    if v_sim > k_sim:
        print(f"  V preserves same-token identity better than K ({v_sim:.3f} > {k_sim:.3f})")
    else:
        print(f"  Surprisingly, K preserves identity as well as V")

    if v_sim - v_diff > k_sim - k_diff:
        print(f"  V better distinguishes same vs different tokens")
    else:
        print(f"  K distinguishes tokens as well as V")


def main():
    print("="*60)
    print("PROJECTION GEOMETRY VISUALIZATION")
    print("What do Q, K, V actually learn for recall?")
    print("="*60)

    config = Config()
    seq_len = 128

    # Train model
    model = train_model(config, seq_len)

    # Analyze projections
    data = analyze_projections(model, config, seq_len)

    # Compute similarities
    compute_similarities(data)

    # Analyze V preservation
    analyze_v_preservation(data)

    # Visualize
    visualize_spaces(data, 'projection_geometry.png')
    visualize_query_key_alignment(data, 'query_key_alignment.png')

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
The projections learn to:

1. K-SPACE: Transform tokens into "findable" representations
   - Query and Key tokens should align despite being at different positions
   - Noise tokens should be orthogonal/distant

2. V-SPACE: Preserve token identity for retrieval
   - Same tokens at different positions should stay similar
   - Different tokens should stay different

3. THE MAGIC: Q and K create an artificial similarity
   - Raw embeddings of "it" and "cat" are unrelated
   - Q("it") and K("cat") are aligned (for pronoun resolution)
   - V("cat") still means "cat" (for retrieval)

The visualization shows how this geometric structure emerges from training.
""")


if __name__ == "__main__":
    main()
