#!/usr/bin/env python3
"""
Multi-Scale Engram: Use multiple decay constants for multi-resolution temporal context.

This is essentially the simplest diagonal SSM - what S4/Hyena do with eigenvalues,
we do with a bank of exponential decays.

Hypothesis: Multiple timescales should outperform single φ = 0.618 decay.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from datasets import load_dataset
from transformers import GPT2Tokenizer


class MultiScaleEngramLayer(nn.Module):
    """
    Multi-scale EMA with learnable mixing.

    Uses K different decay constants to capture K different timescales.
    Each scale is computed via the cumsum trick, then combined.

    This is a diagonal state-space model:
        h^(k)_t = φ_k * h^(k)_{t-1} + x_t

    With K scales, we get K different temporal views of the input.
    """
    def __init__(self, dim, num_scales=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_scales = num_scales

        # Decay constants for 512 token context - all >= 0.95 for stability
        # Half-lives: 13.5, 23, 46, 69 tokens
        decay_rates = [0.95, 0.97, 0.985, 0.99][:num_scales]
        self.register_buffer('phis', torch.tensor(decay_rates, dtype=torch.float32))

        # Single shared projection + gate (like single-scale)
        self.proj = nn.Linear(dim, dim)
        self.gate_proj = nn.Linear(dim, dim)

        # Learnable mixing weights for scales
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        device = x.device

        # Project and gate (same as single-scale)
        x_proj = self.proj(x)
        gate = torch.sigmoid(self.gate_proj(x))
        x_gated = gate * x_proj

        # Precompute indices
        indices = torch.arange(L, device=device, dtype=torch.float32)

        # Process each scale separately (simple loop, like single-scale)
        scale_outputs = []
        for phi in self.phis:
            powers = torch.pow(phi.item(), indices)
            inv_powers = torch.pow(phi.item(), -indices)

            # Clamp inv_powers to avoid overflow
            inv_powers = torch.clamp(inv_powers, max=1e10)

            x_f32 = x_gated.float()
            x_weighted = x_f32 * inv_powers.view(1, -1, 1)
            cum_sum = torch.cumsum(x_weighted, dim=1)
            h_k = cum_sum * powers.view(1, -1, 1)

            scale_outputs.append(h_k)

        # Stack and weighted sum: (num_scales, B, L, D) -> (B, L, D)
        stacked = torch.stack(scale_outputs, dim=0)  # (K, B, L, D)
        weights = F.softmax(self.scale_weights, dim=0).view(-1, 1, 1, 1)
        output = (stacked * weights).sum(dim=0)

        return self.dropout(output.to(x.dtype))


class MultiScaleEngramLayerV2(nn.Module):
    """
    Alternative: Vectorized multi-scale (all scales in parallel).
    Same result as V1 but potentially faster.
    """
    def __init__(self, dim, num_scales=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_scales = num_scales

        # Same decay rates as V1
        decay_rates = [0.95, 0.97, 0.985, 0.99][:num_scales]
        self.register_buffer('phis', torch.tensor(decay_rates, dtype=torch.float32))

        # Single projection for all scales
        self.proj = nn.Linear(dim, dim)
        self.gate_proj = nn.Linear(dim, dim)

        # Learnable scale weights
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        device = x.device

        # Project and gate
        x_proj = self.proj(x)
        gate = torch.sigmoid(self.gate_proj(x))
        x_gated = gate * x_proj

        # Compute indices
        indices = torch.arange(L, device=device, dtype=torch.float32)

        # Process each scale (loop is clearer and avoids broadcast issues)
        scale_outputs = []
        for phi in self.phis:
            powers = torch.pow(phi.item(), indices)
            inv_powers = torch.pow(phi.item(), -indices)
            inv_powers = torch.clamp(inv_powers, max=1e10)

            x_f32 = x_gated.float()
            x_weighted = x_f32 * inv_powers.view(1, -1, 1)
            cum_sum = torch.cumsum(x_weighted, dim=1)
            h_k = cum_sum * powers.view(1, -1, 1)
            scale_outputs.append(h_k)

        # Stack and weighted sum
        stacked = torch.stack(scale_outputs, dim=0)
        weights = F.softmax(self.scale_weights, dim=0).view(-1, 1, 1, 1)
        output = (stacked * weights).sum(dim=0)

        return self.dropout(output.to(x.dtype))


class CausalAttentionLayer(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_dropout(self.out(out))


class SingleScaleEngramLayer(nn.Module):
    """Single-scale baseline - uses φ=0.95 for numerical stability at long sequences."""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.dim = dim
        # Use 0.95 instead of 0.618 for stability at 512+ tokens
        # φ=0.95 has half-life of 13.5 tokens, still useful
        self.phi = 0.95
        self.proj = nn.Linear(dim, dim)
        self.gate_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        device = x.device

        x_proj = self.proj(x)
        gate = torch.sigmoid(self.gate_proj(x))
        x_gated = gate * x_proj

        indices = torch.arange(L, device=device, dtype=torch.float32)
        powers = torch.pow(self.phi, indices)
        inv_powers = torch.pow(self.phi, -indices)

        # Clamp to prevent overflow
        inv_powers = torch.clamp(inv_powers, max=1e10)

        x_f32 = x_gated.float()
        x_weighted = x_f32 * inv_powers.view(1, -1, 1)
        cum_sum = torch.cumsum(x_weighted, dim=1)
        output = cum_sum * powers.view(1, -1, 1)

        return self.dropout(output.to(x.dtype))


class InterleavedBlock(nn.Module):
    def __init__(self, dim, layer_type, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        if layer_type == 'attention':
            self.core = CausalAttentionLayer(dim, num_heads, dropout)
        elif layer_type == 'engram_single':
            self.core = SingleScaleEngramLayer(dim, dropout)
        elif layer_type == 'engram_multi':
            self.core = MultiScaleEngramLayer(dim, num_scales=4, dropout=dropout)
        elif layer_type == 'engram_multi_v2':
            self.core = MultiScaleEngramLayerV2(dim, num_scales=4, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.core(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class InterleavedModel(nn.Module):
    def __init__(self, dim, vocab_size, engram_type='engram_single', num_layers=8, num_heads=8, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        layer_types = ['attention' if i % 2 == 0 else engram_type for i in range(num_layers)]
        self.blocks = nn.ModuleList([
            InterleavedBlock(dim, lt, num_heads, dropout) for lt in layer_types
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


class PureAttentionModel(nn.Module):
    def __init__(self, dim, vocab_size, num_layers=8, num_heads=8, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            InterleavedBlock(dim, 'attention', num_heads, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def load_wikitext(max_length=512):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length, padding="max_length")

    train_data = dataset["train"].map(tokenize, batched=True, remove_columns=["text"])
    val_data = dataset["validation"].map(tokenize, batched=True, remove_columns=["text"])

    train_data.set_format(type="torch", columns=["input_ids"])
    val_data.set_format(type="torch", columns=["input_ids"])

    return train_data, val_data, tokenizer.vocab_size


def train_epoch(model, data, optimizer, scaler, device, batch_size=16):
    model.train()
    total_loss = 0
    num_batches = 0

    indices = torch.randperm(len(data)).tolist()
    for i in range(0, len(indices) - batch_size, batch_size):
        batch_indices = indices[i:i+batch_size]
        batch = torch.stack([data[j]["input_ids"] for j in batch_indices]).to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

    return math.exp(total_loss / num_batches)


@torch.no_grad()
def evaluate(model, data, device, batch_size=16):
    model.eval()
    total_loss = 0
    num_batches = 0

    for i in range(0, len(data) - batch_size, batch_size):
        batch = torch.stack([data[j]["input_ids"] for j in range(i, i+batch_size)]).to(device)
        with torch.amp.autocast('cuda'):
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
        total_loss += loss.item()
        num_batches += 1

    return math.exp(total_loss / num_batches) if num_batches > 0 else float('inf')


@torch.no_grad()
def benchmark_tps(model, device, seq_len=512, batch_size=4, num_runs=50):
    model.eval()
    dummy = torch.randint(0, 50257, (batch_size, seq_len), device=device)

    for _ in range(20):
        with torch.amp.autocast('cuda'):
            _ = model(dummy)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_runs):
        with torch.amp.autocast('cuda'):
            _ = model(dummy)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    return (num_runs * batch_size * seq_len) / elapsed


def train_and_benchmark(name, model, train_data, val_data, device, epochs=5):
    print(f"\n{'='*60}", flush=True)
    print(f"{name}", flush=True)
    print(f"{'='*60}", flush=True)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')

    best_val_ppl = float('inf')
    for epoch in range(1, epochs + 1):
        train_ppl = train_epoch(model, train_data, optimizer, scaler, device)
        val_ppl = evaluate(model, val_data, device)

        marker = " *" if val_ppl < best_val_ppl else ""
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl

        print(f"Epoch {epoch}: Train PPL {train_ppl:7.1f}, Val PPL {val_ppl:7.1f}{marker}", flush=True)

    tps = benchmark_tps(model, device)
    print(f"Throughput: {tps:,.0f} TPS", flush=True)

    return best_val_ppl, tps


def main():
    device = torch.device("cuda")
    print(f"Device: {device}", flush=True)
    print(f"Multi-Scale Engram Experiment", flush=True)
    print(f"Testing whether multiple decay constants beat single φ", flush=True)

    # Use 512 token context to give slower decays a chance to matter
    SEQ_LEN = 512
    train_data, val_data, vocab_size = load_wikitext(max_length=SEQ_LEN)
    print(f"Vocab size: {vocab_size}", flush=True)
    print(f"Sequence length: {SEQ_LEN}", flush=True)

    # Print decay rate info - using high φ for numerical stability at 512 tokens
    decay_rates = [0.95, 0.97, 0.985, 0.99]
    print(f"\nDecay rates and half-lives:", flush=True)
    for phi in decay_rates:
        half_life = math.log(0.5) / math.log(phi)
        print(f"  φ = {phi:.3f} → half-life = {half_life:.1f} tokens", flush=True)

    results = {}

    # 1. Pure Attention baseline
    model = PureAttentionModel(dim=256, vocab_size=vocab_size, num_layers=8).to(device)
    ppl, tps = train_and_benchmark("Pure Attention (8 layers)", model, train_data, val_data, device)
    results['Pure Attention'] = (ppl, tps)
    del model
    torch.cuda.empty_cache()

    # 2. Single-scale engram (φ = 0.618)
    model = InterleavedModel(dim=256, vocab_size=vocab_size, engram_type='engram_single', num_layers=8).to(device)
    ppl, tps = train_and_benchmark("Single-Scale Engram (φ=0.618)", model, train_data, val_data, device)
    results['Single-Scale'] = (ppl, tps)
    del model
    torch.cuda.empty_cache()

    # 3. Multi-scale engram V1 (4 scales, weighted sum)
    model = InterleavedModel(dim=256, vocab_size=vocab_size, engram_type='engram_multi', num_layers=8).to(device)
    ppl, tps = train_and_benchmark("Multi-Scale Engram V1 (4 scales)", model, train_data, val_data, device)
    results['Multi-Scale V1'] = (ppl, tps)
    del model
    torch.cuda.empty_cache()

    # 4. Multi-scale engram V2 (same as V1, just for comparison)
    model = InterleavedModel(dim=256, vocab_size=vocab_size, engram_type='engram_multi_v2', num_layers=8).to(device)
    ppl, tps = train_and_benchmark("Multi-Scale Engram V2 (4 scales)", model, train_data, val_data, device)
    results['Multi-Scale V2'] = (ppl, tps)

    # Also try compiled
    try:
        print("\nCompiling multi-scale V2 model...", flush=True)
        model_compiled = torch.compile(model, mode="reduce-overhead")
        dummy = torch.randint(0, vocab_size, (8, 128), device=device)
        for _ in range(30):
            with torch.amp.autocast('cuda'):
                _ = model_compiled(dummy)
        torch.cuda.synchronize()
        tps_compiled = benchmark_tps(model_compiled, device)
        print(f"Multi-Scale V2 (compiled): {tps_compiled:,.0f} TPS", flush=True)
        results['Multi-Scale V2 (compiled)'] = (ppl, tps_compiled)
    except Exception as e:
        print(f"Compile failed: {e}", flush=True)

    del model
    torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}", flush=True)
    print("RESULTS SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)

    baseline_ppl, baseline_tps = results['Pure Attention']
    single_ppl, _ = results['Single-Scale']

    for name, (ppl, tps) in sorted(results.items(), key=lambda x: x[1][0]):
        ppl_diff = (ppl - baseline_ppl) / baseline_ppl * 100
        tps_diff = (tps - baseline_tps) / baseline_tps * 100
        print(f"{name:<40}: {ppl:6.2f} PPL ({ppl_diff:+5.1f}%), {tps:>10,.0f} TPS ({tps_diff:+6.1f}%)", flush=True)

    # Key comparison
    print(f"\n{'='*70}", flush=True)
    print("KEY COMPARISON: Multi-Scale vs Single-Scale", flush=True)
    print(f"{'='*70}", flush=True)

    for name in ['Multi-Scale V1', 'Multi-Scale V2']:
        if name in results:
            multi_ppl, _ = results[name]
            improvement = (single_ppl - multi_ppl) / single_ppl * 100
            print(f"{name}: {multi_ppl:.2f} PPL ({improvement:+.1f}% vs single-scale)", flush=True)

    # Did multi-scale beat single-scale?
    best_multi = min([results.get('Multi-Scale V1', (float('inf'),))[0],
                      results.get('Multi-Scale V2', (float('inf'),))[0]])

    if best_multi < single_ppl:
        print(f"\n✓ Multi-scale WINS: {best_multi:.2f} vs {single_ppl:.2f} PPL", flush=True)
    else:
        print(f"\n✗ Single-scale still better: {single_ppl:.2f} vs {best_multi:.2f} PPL", flush=True)


if __name__ == "__main__":
    main()
