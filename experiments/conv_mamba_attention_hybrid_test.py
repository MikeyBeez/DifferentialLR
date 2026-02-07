#!/usr/bin/env python3
"""
Conv + Mamba + Attention Hybrid Test

Hypothesis: Can we get fast inference AND needle-in-haystack retrieval?

Architecture:
- Conv layers: O(1) per token, local patterns
- Mamba layers: O(1) per token, builds compressed state/memory
- Attention layer: O(N) per token, precise retrieval (but only 1 layer!)

If this works, we get ~8x inference speedup with full retrieval capability.

Usage:
    python3 experiments/conv_mamba_attention_hybrid_test.py
"""

import sys
sys.path.insert(0, '/home/bee/Code/LinearAttention')

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from dataclasses import dataclass
from typing import List, Optional

# Use optimized Mamba from repo
from src.chunked_attention import SelectiveSSM

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")


@dataclass
class Config:
    vocab_size: int = 256
    dim: int = 128
    num_heads: int = 4
    dropout: float = 0.0

    # Layer configuration: list of 'conv', 'mamba', 'attention'
    layer_types: List[str] = None

    # Conv settings
    kernel_size: int = 64

    # Mamba settings
    d_state: int = 64  # Larger state for better recall
    d_conv: int = 4
    expand: int = 2

    def __post_init__(self):
        if self.layer_types is None:
            # Default: 6 conv + 2 mamba + 1 attention
            self.layer_types = [
                'conv', 'conv',      # Local processing
                'mamba',             # Build state
                'conv', 'conv',      # More local
                'mamba',             # Refine state
                'conv',              # Local
                'attention'          # Precise retrieval (last layer)
            ]


class CausalConv(nn.Module):
    """O(1) causal convolution layer."""

    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.kernel_size = config.kernel_size

        self.v_proj = nn.Linear(config.dim, config.dim)
        self.out = nn.Linear(config.dim, config.dim)

        # Learned kernel per head
        self.kernel_logits = nn.Parameter(
            torch.zeros(config.num_heads, config.kernel_size)
        )
        self.head_dim = config.dim // config.num_heads
        self.num_heads = config.num_heads

    def forward(self, x):
        B, T, C = x.shape

        v = self.v_proj(x).transpose(1, 2)  # (B, C, T)
        v_padded = F.pad(v, (self.kernel_size - 1, 0))

        kernel = F.softmax(self.kernel_logits, dim=-1)
        kernel_expanded = kernel.unsqueeze(1).expand(-1, self.head_dim, -1)
        kernel_expanded = kernel_expanded.reshape(self.dim, 1, self.kernel_size)

        out = F.conv1d(v_padded, kernel_expanded, groups=self.dim)
        out = out.transpose(1, 2)

        return self.out(out)


class MambaWrapper(nn.Module):
    """
    Wrapper around optimized SelectiveSSM from repo.
    O(1) per token during inference (just state update).
    """

    def __init__(self, config: Config):
        super().__init__()
        self.ssm = SelectiveSSM(
            d_model=config.dim,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
        )

    def forward(self, x):
        return self.ssm(x)


class CausalAttention(nn.Module):
    """Standard O(N²) causal attention."""

    def __init__(self, config: Config):
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


class HybridBlock(nn.Module):
    """Transformer block with configurable attention type."""

    def __init__(self, config: Config, layer_type: str):
        super().__init__()
        self.layer_type = layer_type

        if layer_type == 'conv':
            self.mixer = CausalConv(config)
        elif layer_type == 'mamba':
            self.mixer = MambaWrapper(config)
        elif layer_type == 'attention':
            self.mixer = CausalAttention(config)
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

        self.norm1 = nn.LayerNorm(config.dim)
        self.norm2 = nn.LayerNorm(config.dim)

        self.ffn = nn.Sequential(
            nn.Linear(config.dim, config.dim * 4),
            nn.GELU(),
            nn.Linear(config.dim * 4, config.dim)
        )

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class HybridModel(nn.Module):
    """Hybrid model with configurable layer types."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([
            HybridBlock(config, layer_type)
            for layer_type in config.layer_types
        ])
        self.norm = nn.LayerNorm(config.dim)
        self.output = nn.Linear(config.dim, config.vocab_size)

    def forward(self, x):
        h = self.embedding(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        return self.output(h)

    def get_layer_summary(self):
        return ' → '.join(self.config.layer_types)


def generate_recall_batch(batch_size, seq_len, vocab_size, device='cuda'):
    """
    Associative recall task:
    - Key at position 0
    - Value at position 1
    - Noise in positions 2 to seq_len-2
    - Query (same key) at position seq_len-1
    - Target: predict the value
    """
    KEY_START, KEY_END = 10, 110
    VAL_START, VAL_END = 110, 210
    NOISE_START = 210

    sequences = []
    targets = []

    for _ in range(batch_size):
        seq = torch.zeros(seq_len, dtype=torch.long, device=device)
        key = torch.randint(KEY_START, KEY_END, (1,), device=device)
        val = torch.randint(VAL_START, VAL_END, (1,), device=device)

        seq[0] = key
        seq[1] = val
        seq[2:-1] = torch.randint(NOISE_START, vocab_size, (seq_len - 3,), device=device)
        seq[-1] = key  # Query

        sequences.append(seq)
        targets.append(val)

    return torch.stack(sequences), torch.tensor(targets, device=device).squeeze()


def train_epoch(model, optimizer, seq_len, num_batches=100, batch_size=32, device='cuda'):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    vocab_size = model.config.vocab_size

    for _ in range(num_batches):
        seq, targets = generate_recall_batch(batch_size, seq_len, vocab_size, device)

        optimizer.zero_grad()
        logits = model(seq)

        loss = F.cross_entropy(logits[:, -1], targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits[:, -1].argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size

    return total_loss / num_batches, total_correct / total_samples


@torch.no_grad()
def evaluate(model, seq_len, num_batches=50, batch_size=32, device='cuda'):
    model.eval()
    total_correct = 0
    total_samples = 0
    vocab_size = model.config.vocab_size

    for _ in range(num_batches):
        seq, targets = generate_recall_batch(batch_size, seq_len, vocab_size, device)
        logits = model(seq)
        preds = logits[:, -1].argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size

    return total_correct / total_samples


@torch.no_grad()
def measure_inference_time(model, seq_len, num_tokens=100, device='cuda'):
    """Measure time to generate tokens (simulated)."""
    model.eval()

    # Warm up
    x = torch.randint(0, model.config.vocab_size, (1, seq_len), device=device)
    for _ in range(5):
        _ = model(x)

    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(num_tokens):
        _ = model(x)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / num_tokens * 1000  # ms per forward pass


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_architecture(name, layer_types, seq_lengths, device, epochs=30):
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")

    config = Config(layer_types=layer_types)

    print(f"Architecture: {' → '.join(layer_types)}")
    print(f"  Conv layers: {layer_types.count('conv')}")
    print(f"  Mamba layers: {layer_types.count('mamba')}")
    print(f"  Attention layers: {layer_types.count('attention')}")

    results = {}

    for seq_len in seq_lengths:
        print(f"\n--- Seq Length: {seq_len} (distance: {seq_len - 2}) ---")

        model = HybridModel(config).to(device)

        if seq_len == seq_lengths[0]:
            print(f"Parameters: {count_params(model):,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        best_acc = 0
        for epoch in range(1, epochs + 1):
            loss, train_acc = train_epoch(model, optimizer, seq_len, device=device)
            val_acc = evaluate(model, seq_len, device=device)

            if val_acc > best_acc:
                best_acc = val_acc

            if epoch % 5 == 0 or val_acc >= 0.99:
                print(f"Epoch {epoch}: Loss {loss:.3f}, Train {train_acc:.1%}, Val {val_acc:.1%}")

            if val_acc >= 0.99:
                print("SOLVED!")
                break

        # Measure inference time
        inf_time = measure_inference_time(model, seq_len, device=device)
        print(f"Inference: {inf_time:.2f} ms/forward")

        results[seq_len] = {'accuracy': best_acc, 'inference_ms': inf_time}

        del model, optimizer
        torch.cuda.empty_cache()

    return results


def main():
    print("=" * 70)
    print("CONV + MAMBA + ATTENTION HYBRID TEST")
    print("Can we get fast inference AND needle-in-haystack retrieval?")
    print("=" * 70)

    seq_lengths = [64, 128, 256, 512]

    all_results = {}

    # Baseline: Pure attention (8 layers)
    all_results['Pure Attention'] = test_architecture(
        "Pure Attention (8 layers) - Baseline",
        ['attention'] * 8,
        seq_lengths,
        device
    )

    # Pure conv (should fail at long range)
    all_results['Pure Conv'] = test_architecture(
        "Pure Conv (8 layers) - Should fail at recall",
        ['conv'] * 8,
        seq_lengths,
        device
    )

    # Hybrid: 6 conv + 1 mamba + 1 attention
    all_results['6C+1M+1A'] = test_architecture(
        "Hybrid: 6 Conv + 1 Mamba + 1 Attention",
        ['conv', 'conv', 'conv', 'mamba', 'conv', 'conv', 'conv', 'attention'],
        seq_lengths,
        device
    )

    # Hybrid: 5 conv + 2 mamba + 1 attention (user's suggestion)
    all_results['5C+2M+1A'] = test_architecture(
        "Hybrid: 5 Conv + 2 Mamba + 1 Attention",
        ['conv', 'conv', 'mamba', 'conv', 'conv', 'mamba', 'conv', 'attention'],
        seq_lengths,
        device
    )

    # Hybrid: 4 conv + 2 mamba + 2 attention
    all_results['4C+2M+2A'] = test_architecture(
        "Hybrid: 4 Conv + 2 Mamba + 2 Attention",
        ['conv', 'conv', 'mamba', 'conv', 'mamba', 'conv', 'attention', 'attention'],
        seq_lengths,
        device
    )

    # Mamba-heavy: 4 conv + 3 mamba + 1 attention
    all_results['4C+3M+1A'] = test_architecture(
        "Mamba-heavy: 4 Conv + 3 Mamba + 1 Attention",
        ['conv', 'mamba', 'conv', 'mamba', 'conv', 'mamba', 'conv', 'attention'],
        seq_lengths,
        device
    )

    # Summary
    print("\n" + "=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)

    # Accuracy table
    print(f"\n{'Architecture':<20}", end="")
    for seq_len in seq_lengths:
        print(f"{'Dist '+str(seq_len-2):<12}", end="")
    print("Avg Acc")
    print("-" * 90)

    for name, results in all_results.items():
        print(f"{name:<20}", end="")
        accs = []
        for seq_len in seq_lengths:
            acc = results[seq_len]['accuracy']
            accs.append(acc)
            print(f"{acc:<12.1%}", end="")
        print(f"{sum(accs)/len(accs):.1%}")

    # Inference time table
    print(f"\n{'Architecture':<20}", end="")
    for seq_len in seq_lengths:
        print(f"{'Len '+str(seq_len):<12}", end="")
    print("Avg Time")
    print("-" * 90)

    for name, results in all_results.items():
        print(f"{name:<20}", end="")
        times = []
        for seq_len in seq_lengths:
            t = results[seq_len]['inference_ms']
            times.append(t)
            print(f"{t:<12.2f}", end="")
        print(f"{sum(times)/len(times):.2f} ms")

    # Analysis
    print("\n" + "=" * 90)
    print("ANALYSIS")
    print("=" * 90)

    baseline_acc = sum(r['accuracy'] for r in all_results['Pure Attention'].values()) / len(seq_lengths)
    baseline_time = sum(r['inference_ms'] for r in all_results['Pure Attention'].values()) / len(seq_lengths)

    print(f"\nBaseline (Pure Attention): {baseline_acc:.1%} accuracy, {baseline_time:.2f} ms")

    for name, results in all_results.items():
        if name == 'Pure Attention':
            continue

        avg_acc = sum(r['accuracy'] for r in results.values()) / len(seq_lengths)
        avg_time = sum(r['inference_ms'] for r in results.values()) / len(seq_lengths)

        speedup = baseline_time / avg_time if avg_time > 0 else 0
        acc_diff = avg_acc - baseline_acc

        print(f"\n{name}:")
        print(f"  Accuracy: {avg_acc:.1%} ({'+' if acc_diff >= 0 else ''}{acc_diff*100:.1f}%)")
        print(f"  Inference: {avg_time:.2f} ms ({speedup:.1f}x {'faster' if speedup > 1 else 'slower'})")

        if avg_acc >= 0.95 and speedup > 1.5:
            print(f"  ✓ GOOD: Fast AND accurate!")
        elif avg_acc >= 0.95:
            print(f"  ~ Accurate but not faster")
        elif speedup > 1.5:
            print(f"  ✗ Fast but accuracy dropped")
        else:
            print(f"  ✗ Neither fast nor accurate")


if __name__ == "__main__":
    main()
