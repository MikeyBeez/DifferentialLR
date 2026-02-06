#!/usr/bin/env python3
"""
Dilated Convolution Associative Recall Test

Tests whether stacked dilated convolutions can achieve long-range retrieval
that plain causal convolution cannot.

Dilated convolution with dilation rates (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
gives a receptive field of 1024 tokens with only 10 layers, all O(N).

This is the WaveNet trick - exponentially growing receptive field with
linear compute.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import List


@dataclass
class TestConfig:
    vocab_size: int = 256
    dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.0
    kernel_size: int = 64


# --- Attention Variants ---

class StandardAttention(nn.Module):
    """Standard O(N²) attention - baseline."""
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
    """Plain O(N) causal convolution - for comparison."""
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


class DilatedCausalConv(nn.Module):
    """
    Single dilated causal convolution layer.

    With dilation d and kernel size k, sees positions:
    [t - d*(k-1), t - d*(k-2), ..., t - d, t]
    """
    def __init__(self, dim: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation  # Causal padding

        self.conv = nn.Conv1d(dim, dim, kernel_size, dilation=dilation, groups=dim)
        self.gate_conv = nn.Conv1d(dim, dim, kernel_size, dilation=dilation, groups=dim)

    def forward(self, x):
        # x: (B, C, T)
        x_padded = F.pad(x, (self.padding, 0))

        # Gated activation (WaveNet style)
        out = torch.tanh(self.conv(x_padded)) * torch.sigmoid(self.gate_conv(x_padded))
        return out


class DilatedConvStack(nn.Module):
    """
    Stack of dilated convolutions with exponentially increasing dilation.

    Dilations: 1, 2, 4, 8, 16, 32, ...
    Receptive field = sum of (kernel_size - 1) * dilation for each layer + 1

    With kernel_size=2 and 10 layers: RF = 1 + 2 + 4 + ... + 512 = 1023
    """
    def __init__(self, dim: int, num_layers: int = 10, kernel_size: int = 2):
        super().__init__()

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            dilation = 2 ** i
            self.layers.append(DilatedCausalConv(dim, kernel_size, dilation))
            self.norms.append(nn.LayerNorm(dim))

        # Calculate receptive field
        self.receptive_field = sum((kernel_size - 1) * (2 ** i) for i in range(num_layers)) + 1

    def forward(self, x):
        # x: (B, T, C)
        x = x.transpose(1, 2)  # (B, C, T)

        for conv, norm in zip(self.layers, self.norms):
            residual = x
            x = conv(x)
            x = x + residual  # Residual connection
            x = norm(x.transpose(1, 2)).transpose(1, 2)  # LayerNorm

        return x.transpose(1, 2)  # (B, T, C)


class DilatedConvAttention(nn.Module):
    """
    O(N) attention replacement using dilated convolution stack.
    Exponential receptive field growth with linear compute.
    """
    def __init__(self, config: TestConfig, num_dilated_layers: int = 10):
        super().__init__()
        self.v_proj = nn.Linear(config.dim, config.dim)
        self.dilated_stack = DilatedConvStack(config.dim, num_dilated_layers, kernel_size=2)
        self.out = nn.Linear(config.dim, config.dim)

    def forward(self, x):
        v = self.v_proj(x)
        out = self.dilated_stack(v)
        return self.out(out)


class DilatedConvAttentionV2(nn.Module):
    """
    Version 2: Dilated conv with larger kernel for better information flow.
    """
    def __init__(self, config: TestConfig, num_dilated_layers: int = 8):
        super().__init__()
        self.v_proj = nn.Linear(config.dim, config.dim)
        self.dilated_stack = DilatedConvStack(config.dim, num_dilated_layers, kernel_size=3)
        self.out = nn.Linear(config.dim, config.dim)
        self.receptive_field = self.dilated_stack.receptive_field

    def forward(self, x):
        v = self.v_proj(x)
        out = self.dilated_stack(v)
        return self.out(out)


class MultiScaleDilatedConv(nn.Module):
    """
    Multiple parallel dilated conv stacks at different scales,
    then combined. Inspired by Inception architecture.
    """
    def __init__(self, config: TestConfig):
        super().__init__()
        dim = config.dim

        self.v_proj = nn.Linear(dim, dim)

        # Multiple scales
        self.scale1 = DilatedConvStack(dim // 4, num_layers=6, kernel_size=2)   # RF=63
        self.scale2 = DilatedConvStack(dim // 4, num_layers=8, kernel_size=2)   # RF=255
        self.scale3 = DilatedConvStack(dim // 4, num_layers=10, kernel_size=2)  # RF=1023
        self.scale4 = DilatedConvStack(dim // 4, num_layers=12, kernel_size=2)  # RF=4095

        self.proj_out = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, C = x.shape
        v = self.v_proj(x)

        # Split channels
        v1, v2, v3, v4 = v.chunk(4, dim=-1)

        # Process at different scales
        o1 = self.scale1(v1)
        o2 = self.scale2(v2)
        o3 = self.scale3(v3)
        o4 = self.scale4(v4)

        # Combine
        out = torch.cat([o1, o2, o3, o4], dim=-1)
        out = self.proj_out(out)

        return self.out(out)


# --- Model ---

class TransformerBlock(nn.Module):
    def __init__(self, config: TestConfig, attention_class, **attn_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.dim)
        self.attn = attention_class(config, **attn_kwargs)
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
    def __init__(self, config: TestConfig, attention_class, **attn_kwargs):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(config, attention_class, **attn_kwargs)
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

def generate_associative_recall_batch(batch_size, seq_len, vocab_size, device):
    """Generate Key-Value associative recall task."""
    KEY_START, KEY_END = 10, 110
    VAL_START, VAL_END = 110, 210
    NOISE_START = 210

    sequences = []
    targets = []

    for _ in range(batch_size):
        seq = torch.zeros(seq_len, dtype=torch.long, device=device)

        # Key-value pair at start
        key = torch.randint(KEY_START, KEY_END, (1,), device=device)
        val = torch.randint(VAL_START, VAL_END, (1,), device=device)
        seq[0] = key
        seq[1] = val

        # Noise in middle
        seq[2:-1] = torch.randint(NOISE_START, vocab_size, (seq_len - 3,), device=device)

        # Query at end
        seq[-1] = key

        sequences.append(seq)
        targets.append(val)

    return torch.stack(sequences), torch.tensor(targets, device=device).squeeze()


# --- Training ---

def train_epoch(model, config, seq_len, num_batches=100, batch_size=32, device='cuda'):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    total_correct = 0
    total_samples = 0

    for _ in range(num_batches):
        seqs, targets = generate_associative_recall_batch(
            batch_size, seq_len, config.vocab_size, device
        )

        optimizer.zero_grad()
        logits = model(seqs)
        query_logits = logits[:, -1]  # Prediction after query

        loss = F.cross_entropy(query_logits, targets)
        loss.backward()
        optimizer.step()

        preds = query_logits.argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size

    return total_correct / total_samples


@torch.no_grad()
def evaluate(model, config, seq_len, num_batches=50, batch_size=32, device='cuda'):
    model.eval()

    total_correct = 0
    total_samples = 0

    for _ in range(num_batches):
        seqs, targets = generate_associative_recall_batch(
            batch_size, seq_len, config.vocab_size, device
        )

        logits = model(seqs)
        query_logits = logits[:, -1]

        preds = query_logits.argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size

    return total_correct / total_samples


def test_model(name, attention_class, config, seq_lengths, device, epochs=15, **attn_kwargs):
    """Test a model across different sequence lengths."""
    print(f"\n{'='*60}", flush=True)
    print(f"{name}", flush=True)
    print(f"{'='*60}", flush=True)

    results = {}

    for seq_len in seq_lengths:
        print(f"\n--- Sequence Length: {seq_len} (distance: {seq_len-2}) ---", flush=True)

        model = AssociativeRecallModel(config, attention_class, **attn_kwargs).to(device)
        params = sum(p.numel() for p in model.parameters())
        if seq_len == seq_lengths[0]:
            print(f"Parameters: {params:,}", flush=True)

        best_acc = 0
        for epoch in range(1, epochs + 1):
            train_acc = train_epoch(model, config, seq_len, device=device)
            val_acc = evaluate(model, config, seq_len, device=device)

            if val_acc > best_acc:
                best_acc = val_acc

            if epoch % 3 == 0 or epoch == epochs:
                print(f"Epoch {epoch}: Train {train_acc:.1%}, Val {val_acc:.1%}", flush=True)

            if val_acc > 0.99:
                print(f"Solved at epoch {epoch}!", flush=True)
                break

        results[seq_len] = best_acc
        print(f"Best: {best_acc:.1%}", flush=True)

        del model
        torch.cuda.empty_cache()

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Dilated Convolution Associative Recall Test", flush=True)
    print(f"Testing if exponential receptive field enables long-range retrieval", flush=True)

    config = TestConfig()
    seq_lengths = [32, 64, 128, 256, 512, 1024]

    all_results = {}

    # 1. Standard attention (baseline)
    all_results['Standard'] = test_model(
        "Standard Attention O(N²)", StandardAttention, config, seq_lengths, device
    )

    # 2. Plain causal conv K=64 (known to fail)
    all_results['Conv K=64'] = test_model(
        "Plain Causal Conv K=64", CausalConvAttention, config, seq_lengths, device
    )

    # 3. Dilated conv (10 layers, RF=1023)
    all_results['Dilated-10'] = test_model(
        "Dilated Conv (10 layers, RF=1023)", DilatedConvAttention, config, seq_lengths, device,
        num_dilated_layers=10
    )

    # 4. Dilated conv V2 (8 layers, kernel=3, RF larger)
    all_results['Dilated-V2'] = test_model(
        "Dilated Conv V2 (8 layers, k=3)", DilatedConvAttentionV2, config, seq_lengths, device,
        num_dilated_layers=8
    )

    # 5. Multi-scale dilated conv
    all_results['MultiScale'] = test_model(
        "Multi-Scale Dilated Conv", MultiScaleDilatedConv, config, seq_lengths, device
    )

    # Summary
    print(f"\n{'='*70}", flush=True)
    print("DILATED CONVOLUTION RESULTS", flush=True)
    print(f"{'='*70}", flush=True)

    print(f"\n{'Distance':<10}", end="", flush=True)
    for name in all_results.keys():
        print(f"{name:<14}", end="", flush=True)
    print(flush=True)
    print("-" * 80, flush=True)

    for seq_len in seq_lengths:
        distance = seq_len - 2
        print(f"{distance:<10}", end="", flush=True)
        for name, results in all_results.items():
            acc = results.get(seq_len, 0)
            status = "✓" if acc > 0.95 else "✗" if acc < 0.5 else "~"
            print(f"{acc:>5.0%} {status:<7}", end="", flush=True)
        print(flush=True)

    print(f"\n{'='*70}", flush=True)

    # Analysis
    print("\nANALYSIS:", flush=True)

    for name, results in all_results.items():
        if name == 'Standard':
            continue
        max_solved = max([sl for sl, acc in results.items() if acc > 0.9], default=0)
        if max_solved > 0:
            print(f"{name}: Solves up to distance {max_solved - 2}", flush=True)
        else:
            print(f"{name}: Fails at all distances", flush=True)

    # Conclusion
    dilated_results = all_results.get('Dilated-10', {})
    dilated_max = max([sl for sl, acc in dilated_results.items() if acc > 0.9], default=0)

    print("\nCONCLUSION:", flush=True)
    if dilated_max >= 512:
        print("POSITIVE: Dilated convolution enables long-range retrieval!", flush=True)
        print("Exponential receptive field growth is the key.", flush=True)
    elif dilated_max > 64:
        print("PARTIAL: Dilated conv extends range but has limits.", flush=True)
    else:
        print("NEGATIVE: Dilated conv still fails at long range.", flush=True)
        print("The problem may be information compression, not receptive field.", flush=True)


if __name__ == "__main__":
    main()
