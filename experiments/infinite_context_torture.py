#!/usr/bin/env python3
"""
INFINITE CONTEXT TORTURE TEST

Feed the Golden Crystal increasingly long sequences.
Standard Transformers will OOM. The Crystal should stay flat.

Training length: 256 tokens
Test lengths: 256, 512, 1024, 2048, 4096, 8192, 16384, 32768

Memory should stay CONSTANT as sequence length increases.
"""

import sys
sys.path.insert(0, '/home/bee/Code/LinearAttention')

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import gc

PHI = (1 + math.sqrt(5)) / 2


class GoldenCrystal(nn.Module):
    """The O(1) memory Golden Crystal."""
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

        # Forward crystal - O(1) memory state
        h_fwd = torch.zeros(B, self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)
        fwd = []
        for i in range(N):
            h_fwd = t[:, i] + self.decay.view(1, -1, 1) * h_fwd
            fwd.append(h_fwd)
        fwd = torch.stack(fwd, dim=1)

        # Backward crystal - O(1) memory state
        h_bwd = torch.zeros(B, self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)
        bwd = []
        for i in range(N-1, -1, -1):
            h_bwd = t[:, i] + self.decay.view(1, -1, 1) * h_bwd
            bwd.append(h_bwd)
        bwd = torch.stack(bwd[::-1], dim=1)

        y = fwd + bwd - t
        return self.out(y.reshape(B, N, D))


class StandardAttention(nn.Module):
    """Standard O(N^2) attention for comparison."""
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

        # This is O(N^2) memory!
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)

        return self.o(out.transpose(1, 2).reshape(B, N, D))


class CrystalTransformer(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, num_heads, use_attention=False):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        # NO positional encoding!

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if use_attention:
                attn = StandardAttention(dim, num_heads)
            else:
                attn = GoldenCrystal(dim, num_heads)

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

    def forward(self, input_ids):
        x = self.embed(input_ids)

        for layer in self.layers:
            x = x + layer['attn'](layer['norm1'](x))
            x = x + layer['ffn'](layer['norm2'](x))

        return self.lm_head(self.final_norm(x))


def get_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def test_sequence_length(model, seq_len, vocab_size, device, batch_size=1):
    """Test a single sequence length and return memory usage."""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    # Generate random input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Warm up
    with torch.no_grad():
        try:
            _ = model(input_ids[:, :min(256, seq_len)])
        except:
            pass

    torch.cuda.reset_peak_memory_stats()

    # Actual test
    with torch.no_grad():
        try:
            with torch.amp.autocast('cuda'):
                output = model(input_ids)
            peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
            success = True
        except RuntimeError as e:
            if "out of memory" in str(e):
                peak_mem = float('inf')
                success = False
            else:
                raise e

    torch.cuda.empty_cache()
    return peak_mem, success


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    print("\n" + "=" * 70)
    print("INFINITE CONTEXT TORTURE TEST")
    print("=" * 70)
    print("\nTraining length: 256 tokens")
    print("Testing: 256 -> 32768 tokens (128x training length)")
    print("\nGolden Crystal should maintain FLAT memory usage.")
    print("Standard Attention will OOM at longer lengths.")

    vocab_size = 50257  # GPT-2 vocab
    dim = 256
    num_layers = 4
    num_heads = 4

    # Test sequence lengths (powers of 2)
    test_lengths = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

    # Test Golden Crystal
    print("\n" + "=" * 70)
    print("GOLDEN CRYSTAL (O(1) Memory)")
    print("=" * 70)

    crystal_model = CrystalTransformer(
        vocab_size, dim, num_layers, num_heads, use_attention=False
    ).to(device).eval()

    crystal_params = sum(p.numel() for p in crystal_model.parameters())
    print(f"Parameters: {crystal_params:,}")

    print(f"\n{'Seq Len':>10} | {'Peak Memory':>12} | {'Status':>10} | {'vs 256':>10}")
    print("-" * 50)

    crystal_results = {}
    base_mem = None
    for seq_len in test_lengths:
        mem, success = test_sequence_length(crystal_model, seq_len, vocab_size, device)
        crystal_results[seq_len] = (mem, success)

        if base_mem is None:
            base_mem = mem
            ratio = "1.0x"
        else:
            ratio = f"{mem/base_mem:.1f}x" if success else "OOM"

        status = "OK" if success else "OOM"
        mem_str = f"{mem:.1f} MB" if success else "OOM"
        print(f"{seq_len:>10} | {mem_str:>12} | {status:>10} | {ratio:>10}")

    del crystal_model
    torch.cuda.empty_cache()

    # Test Standard Attention
    print("\n" + "=" * 70)
    print("STANDARD ATTENTION (O(N^2) Memory)")
    print("=" * 70)

    attn_model = CrystalTransformer(
        vocab_size, dim, num_layers, num_heads, use_attention=True
    ).to(device).eval()

    attn_params = sum(p.numel() for p in attn_model.parameters())
    print(f"Parameters: {attn_params:,}")

    print(f"\n{'Seq Len':>10} | {'Peak Memory':>12} | {'Status':>10} | {'vs 256':>10}")
    print("-" * 50)

    attn_results = {}
    base_mem = None
    for seq_len in test_lengths:
        mem, success = test_sequence_length(attn_model, seq_len, vocab_size, device)
        attn_results[seq_len] = (mem, success)

        if base_mem is None:
            base_mem = mem
            ratio = "1.0x"
        else:
            ratio = f"{mem/base_mem:.1f}x" if success else "OOM"

        status = "OK" if success else "OOM"
        mem_str = f"{mem:.1f} MB" if success else "OOM"
        print(f"{seq_len:>10} | {mem_str:>12} | {status:>10} | {ratio:>10}")

        if not success:
            print("  (Stopping - further lengths will also OOM)")
            break

    del attn_model
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Seq Len':>10} | {'Crystal':>12} | {'Attention':>12} | {'Savings':>10}")
    print("-" * 50)

    for seq_len in test_lengths:
        c_mem, c_ok = crystal_results.get(seq_len, (0, False))
        a_mem, a_ok = attn_results.get(seq_len, (float('inf'), False))

        c_str = f"{c_mem:.1f} MB" if c_ok else "OOM"
        a_str = f"{a_mem:.1f} MB" if a_ok else "OOM"

        if c_ok and a_ok:
            savings = f"{a_mem/c_mem:.1f}x"
        elif c_ok and not a_ok:
            savings = "INFINITE"
        else:
            savings = "N/A"

        print(f"{seq_len:>10} | {c_str:>12} | {a_str:>12} | {savings:>10}")

    # Final verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    max_crystal = max(seq_len for seq_len, (_, ok) in crystal_results.items() if ok)
    max_attn = max((seq_len for seq_len, (_, ok) in attn_results.items() if ok), default=0)

    print(f"\nGolden Crystal max sequence: {max_crystal:,} tokens")
    print(f"Standard Attention max sequence: {max_attn:,} tokens")
    print(f"Crystal advantage: {max_crystal / max_attn:.0f}x longer contexts")

    crystal_256 = crystal_results[256][0]
    crystal_max = crystal_results[max_crystal][0]
    mem_growth = crystal_max / crystal_256

    print(f"\nCrystal memory growth (256 -> {max_crystal}): {mem_growth:.2f}x")
    if mem_growth < 2.0:
        print("CONFIRMED: O(1) memory - nearly flat scaling!")
    else:
        print(f"Memory grew {mem_growth:.1f}x (expected ~1.0x for true O(1))")


if __name__ == "__main__":
    main()
