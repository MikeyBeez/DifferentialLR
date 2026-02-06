#!/usr/bin/env python3
"""
Associative Recall Test

Tests whether O(N) causal convolution can retrieve specific tokens from
long distances, or if it only works for "gist" tasks like language modeling.

Task: Key: <value> ... [N random tokens] ... Question: What was the Key?

This is the standard synthetic benchmark used by Mamba, Hyena, and other
linear attention papers to prove long-range retrieval capability.

If conv fails here, it means:
- Conv works for language modeling (predicting likely next tokens)
- Conv fails for precise retrieval (finding specific past tokens)
- The O(N) advantage is real but limited to certain task types
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from dataclasses import dataclass
from typing import Tuple


@dataclass
class TestConfig:
    vocab_size: int = 256  # Simple vocab for synthetic task
    dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.0  # No dropout for this test
    kernel_size: int = 64


# --- Attention Variants ---

class StandardAttention(nn.Module):
    """Standard O(N²) attention - should ace this test."""
    def __init__(self, config: TestConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.dim // config.num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(config.dim, config.dim * 3)
        self.out = nn.Linear(config.dim, config.dim)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class CausalConvAttention(nn.Module):
    """O(N) causal convolution - the model we're testing."""
    def __init__(self, config: TestConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.dim // config.num_heads
        self.kernel_size = config.kernel_size
        self.dim = config.dim

        self.v_proj = nn.Linear(config.dim, config.dim)
        self.out = nn.Linear(config.dim, config.dim)

        self.kernel_logits = nn.Parameter(torch.zeros(config.num_heads, config.kernel_size))

    def forward(self, x):
        B, T, C = x.shape

        v = self.v_proj(x).transpose(1, 2)
        v_padded = F.pad(v, (self.kernel_size - 1, 0))

        kernel = F.softmax(self.kernel_logits, dim=-1)
        kernel_expanded = kernel.unsqueeze(1).expand(-1, self.head_dim, -1)
        kernel_expanded = kernel_expanded.reshape(self.dim, 1, self.kernel_size)

        out = F.conv1d(v_padded, kernel_expanded, groups=self.dim)
        out = out.transpose(1, 2)

        return self.out(out)


class ExtendedConvAttention(nn.Module):
    """O(N) causal convolution with LARGER kernel for long-range."""
    def __init__(self, config: TestConfig, kernel_size: int = 512):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.dim // config.num_heads
        self.kernel_size = kernel_size
        self.dim = config.dim

        self.v_proj = nn.Linear(config.dim, config.dim)
        self.out = nn.Linear(config.dim, config.dim)

        self.kernel_logits = nn.Parameter(torch.zeros(config.num_heads, kernel_size))

    def forward(self, x):
        B, T, C = x.shape

        v = self.v_proj(x).transpose(1, 2)
        v_padded = F.pad(v, (self.kernel_size - 1, 0))

        kernel = F.softmax(self.kernel_logits, dim=-1)
        kernel_expanded = kernel.unsqueeze(1).expand(-1, self.head_dim, -1)
        kernel_expanded = kernel_expanded.reshape(self.dim, 1, self.kernel_size)

        out = F.conv1d(v_padded, kernel_expanded, groups=self.dim)
        out = out.transpose(1, 2)

        return self.out(out)


# --- Model ---

class TransformerBlock(nn.Module):
    def __init__(self, config: TestConfig, attention_class):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.dim)
        self.attn = attention_class(config)
        self.norm2 = nn.LayerNorm(config.dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.dim, config.dim * 4),
            nn.GELU(),
            nn.Linear(config.dim * 4, config.dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class AssociativeRecallModel(nn.Module):
    def __init__(self, config: TestConfig, attention_class):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(config, attention_class)
            for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


# --- Data Generation ---

def generate_associative_recall_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    num_pairs: int = 1,
    device: torch.device = torch.device('cpu')
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate associative recall task data.

    Format: KEY1 VAL1 KEY2 VAL2 ... [noise] ... QUERY_KEY
    Target: The value associated with QUERY_KEY

    We use special tokens:
    - 0-9: reserved (padding, special markers)
    - 10-109: keys (100 possible keys)
    - 110-209: values (100 possible values)
    - 210+: noise tokens
    """
    KEY_START = 10
    KEY_END = 110
    VAL_START = 110
    VAL_END = 210
    NOISE_START = 210

    sequences = []
    targets = []
    query_positions = []

    for _ in range(batch_size):
        seq = torch.zeros(seq_len, dtype=torch.long, device=device)

        # Generate key-value pairs
        keys = torch.randint(KEY_START, KEY_END, (num_pairs,), device=device)
        vals = torch.randint(VAL_START, VAL_END, (num_pairs,), device=device)

        # Place pairs at the beginning
        for i, (k, v) in enumerate(zip(keys, vals)):
            seq[i * 2] = k
            seq[i * 2 + 1] = v

        pair_end = num_pairs * 2

        # Fill middle with noise
        noise_len = seq_len - pair_end - 1
        seq[pair_end:pair_end + noise_len] = torch.randint(
            NOISE_START, vocab_size, (noise_len,), device=device
        )

        # Query: repeat one of the keys at the end
        query_idx = torch.randint(0, num_pairs, (1,)).item()
        seq[-1] = keys[query_idx]

        # Target: the corresponding value
        target = vals[query_idx]

        sequences.append(seq)
        targets.append(target)
        query_positions.append(seq_len - 1)

    return (
        torch.stack(sequences),
        torch.tensor(targets, device=device),
        torch.tensor(query_positions, device=device)
    )


# --- Training ---

def train_epoch(model, config, seq_len, num_batches=100, batch_size=32, device='cuda'):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    total_loss = 0
    total_correct = 0
    total_samples = 0

    for _ in range(num_batches):
        seqs, targets, query_pos = generate_associative_recall_batch(
            batch_size, seq_len, config.vocab_size, num_pairs=1, device=device
        )

        optimizer.zero_grad()
        logits = model(seqs)

        # Get logits at query position (last position predicts the answer)
        # We want the prediction AFTER seeing the query key
        query_logits = logits[torch.arange(batch_size), query_pos]

        loss = F.cross_entropy(query_logits, targets)
        loss.backward()
        optimizer.step()

        # Accuracy
        preds = query_logits.argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size
        total_loss += loss.item()

    return total_loss / num_batches, total_correct / total_samples


@torch.no_grad()
def evaluate(model, config, seq_len, num_batches=50, batch_size=32, device='cuda'):
    model.eval()

    total_correct = 0
    total_samples = 0

    for _ in range(num_batches):
        seqs, targets, query_pos = generate_associative_recall_batch(
            batch_size, seq_len, config.vocab_size, num_pairs=1, device=device
        )

        logits = model(seqs)
        query_logits = logits[torch.arange(batch_size), query_pos]

        preds = query_logits.argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size

    return total_correct / total_samples


def test_model(name, model, config, seq_lengths, device, epochs=10):
    """Test a model across different sequence lengths."""
    print(f"\n{'='*60}", flush=True)
    print(f"{name}", flush=True)
    print(f"{'='*60}", flush=True)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}", flush=True)

    results = {}

    for seq_len in seq_lengths:
        print(f"\n--- Sequence Length: {seq_len} ---", flush=True)

        # Reset model
        for module in model.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

        # Reinitialize
        model_fresh = type(model)(config, type(model.blocks[0].attn)).to(device)

        best_acc = 0
        for epoch in range(1, epochs + 1):
            loss, train_acc = train_epoch(model_fresh, config, seq_len, device=device)
            val_acc = evaluate(model_fresh, config, seq_len, device=device)

            if val_acc > best_acc:
                best_acc = val_acc

            if epoch % 2 == 0 or epoch == epochs:
                print(f"Epoch {epoch}: Loss {loss:.4f}, Train Acc {train_acc:.1%}, Val Acc {val_acc:.1%}", flush=True)

            # Early stopping if solved
            if val_acc > 0.99:
                print(f"Solved at epoch {epoch}!", flush=True)
                break

        results[seq_len] = best_acc
        print(f"Best accuracy: {best_acc:.1%}", flush=True)

        del model_fresh
        torch.cuda.empty_cache()

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Associative Recall Test", flush=True)
    print(f"Testing if O(N) convolution can retrieve specific tokens from long range", flush=True)

    config = TestConfig()

    # Test at increasing distances
    # The key-value pair is at positions 0-1
    # The query is at the last position
    # Distance = seq_len - 2
    seq_lengths = [32, 64, 128, 256, 512, 1024]

    all_results = {}

    # 1. Standard attention (should work perfectly)
    model = AssociativeRecallModel(config, StandardAttention).to(device)
    all_results['Standard O(N²)'] = test_model(
        "Standard Attention O(N²)", model, config, seq_lengths, device
    )
    del model
    torch.cuda.empty_cache()

    # 2. Causal conv with kernel=64 (our default)
    model = AssociativeRecallModel(config, CausalConvAttention).to(device)
    all_results['Conv K=64'] = test_model(
        "Causal Conv O(N) [kernel=64]", model, config, seq_lengths, device
    )
    del model
    torch.cuda.empty_cache()

    # 3. Causal conv with kernel=256 (extended)
    class ConvK256(CausalConvAttention):
        def __init__(self, config):
            super().__init__(config)
            self.kernel_size = 256
            self.kernel_logits = nn.Parameter(torch.zeros(config.num_heads, 256))

    model = AssociativeRecallModel(config, ConvK256).to(device)
    all_results['Conv K=256'] = test_model(
        "Causal Conv O(N) [kernel=256]", model, config, seq_lengths, device
    )
    del model
    torch.cuda.empty_cache()

    # 4. Causal conv with kernel=512 (large)
    class ConvK512(CausalConvAttention):
        def __init__(self, config):
            super().__init__(config)
            self.kernel_size = 512
            self.kernel_logits = nn.Parameter(torch.zeros(config.num_heads, 512))

    model = AssociativeRecallModel(config, ConvK512).to(device)
    all_results['Conv K=512'] = test_model(
        "Causal Conv O(N) [kernel=512]", model, config, seq_lengths, device
    )
    del model
    torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}", flush=True)
    print("ASSOCIATIVE RECALL RESULTS", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"\nAccuracy by sequence length (key at position 0, query at end):", flush=True)
    print(f"\n{'Distance':<10}", end="", flush=True)
    for name in all_results.keys():
        print(f"{name:<15}", end="", flush=True)
    print(flush=True)
    print("-" * 70, flush=True)

    for seq_len in seq_lengths:
        distance = seq_len - 2
        print(f"{distance:<10}", end="", flush=True)
        for name, results in all_results.items():
            acc = results.get(seq_len, 0)
            status = "✓" if acc > 0.95 else "✗" if acc < 0.5 else "~"
            print(f"{acc:>6.1%} {status:<7}", end="", flush=True)
        print(flush=True)

    print(f"\n{'='*70}", flush=True)

    # Analysis
    print("\nANALYSIS:", flush=True)

    std_results = all_results['Standard O(N²)']
    conv64_results = all_results['Conv K=64']
    conv512_results = all_results['Conv K=512']

    # Find where conv breaks down
    conv64_fails = [sl for sl, acc in conv64_results.items() if acc < 0.9]
    conv512_fails = [sl for sl, acc in conv512_results.items() if acc < 0.9]

    if not conv64_fails:
        print("Conv K=64: Solves ALL distances tested!", flush=True)
    else:
        print(f"Conv K=64: Fails at distance {min(conv64_fails) - 2}+", flush=True)

    if not conv512_fails:
        print("Conv K=512: Solves ALL distances tested!", flush=True)
    else:
        print(f"Conv K=512: Fails at distance {min(conv512_fails) - 2}+", flush=True)

    # Conclusion
    print("\nCONCLUSION:", flush=True)

    max_tested = max(seq_lengths) - 2
    conv64_max_solved = max([sl for sl, acc in conv64_results.items() if acc > 0.9], default=0) - 2
    conv512_max_solved = max([sl for sl, acc in conv512_results.items() if acc > 0.9], default=0) - 2

    if conv64_max_solved >= max_tested:
        print(f"POSITIVE: Conv K=64 can retrieve tokens from {max_tested}+ positions away.", flush=True)
        print("The O(N) architecture has TRUE long-range capability.", flush=True)
    elif conv512_max_solved >= max_tested:
        print(f"MIXED: Conv needs larger kernel for long range.", flush=True)
        print(f"K=64 works up to ~{conv64_max_solved}, K=512 works up to {max_tested}+", flush=True)
    else:
        print(f"NEGATIVE: Conv fails associative recall beyond ~{max(conv64_max_solved, conv512_max_solved)} tokens.", flush=True)
        print("Conv works for language modeling but NOT for precise long-range retrieval.", flush=True)
        print("This is a fundamental limitation vs attention.", flush=True)


if __name__ == "__main__":
    main()
