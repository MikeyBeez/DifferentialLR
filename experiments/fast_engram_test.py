#!/usr/bin/env python3
"""
Fast Engram: Replace sequential loop with vectorized cumsum.
Goal: 100k+ TPS while maintaining PPL improvement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from datasets import load_dataset
from transformers import GPT2Tokenizer

PHI_INV = 1 / 1.61803398875  # 0.618...


class SequentialEngramLayer(nn.Module):
    """Original sequential engram (slow but accurate baseline)."""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.relo_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        state = torch.zeros(batch_size, self.dim, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seq_len):
            current_x = x[:, t, :]
            alignment = torch.sum(current_x * state, dim=-1, keepdim=True) / math.sqrt(self.dim)
            gate = torch.sigmoid(alignment)
            new_info = self.relo_proj(current_x)
            state = (PHI_INV * state) + (gate * new_info)
            outputs.append(state.unsqueeze(1))

        return self.dropout(torch.cat(outputs, dim=1))


class FastEngramLayer(nn.Module):
    """
    Vectorized engram using cumsum trick.

    The recurrence h_t = φ * h_{t-1} + x_t can be rewritten as:
    h_t = Σ_{i=0}^{t} φ^{t-i} * x_i

    Computed via:
    1. x_weighted[i] = x[i] / φ^i
    2. cumsum[t] = Σ_{i=0}^{t} x_weighted[i]
    3. h[t] = cumsum[t] * φ^t

    This is O(N) parallel operations instead of O(N) sequential.

    Note: Removes the state-dependent gate for speed.
    Uses a learned input gate instead.
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, dim)
        self.gate_proj = nn.Linear(dim, dim)  # Learned gate (not state-dependent)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        device = x.device

        # Project input and compute gate
        x_proj = self.proj(x)  # (B, L, D)
        gate = torch.sigmoid(self.gate_proj(x))  # Input-dependent gate
        x_gated = gate * x_proj

        # Compute powers of phi: [φ^0, φ^1, φ^2, ...]
        # Use float32 for numerical stability
        indices = torch.arange(seq_len, device=device, dtype=torch.float32)
        powers = torch.pow(PHI_INV, indices)  # φ^i
        inv_powers = torch.pow(PHI_INV, -indices)  # φ^{-i}

        # Cast x to float32 for cumsum stability, then back
        x_f32 = x_gated.float()

        # Weighted cumsum trick:
        # h_t = Σ_{i=0}^{t} φ^{t-i} * x_i = φ^t * Σ_{i=0}^{t} φ^{-i} * x_i
        x_weighted = x_f32 * inv_powers.view(1, -1, 1)
        cum_sum = torch.cumsum(x_weighted, dim=1)
        output = cum_sum * powers.view(1, -1, 1)

        # Cast back to original dtype
        output = output.to(x.dtype)

        return self.dropout(output)


class FastEngramLayerV2(nn.Module):
    """
    Alternative: Use causal conv1d with decay kernel.
    More numerically stable for long sequences.
    """
    def __init__(self, dim, dropout=0.1, max_len=256):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, dim)
        self.gate_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Pre-compute decay kernel: [φ^{L-1}, φ^{L-2}, ..., φ^1, φ^0]
        # Flipped for conv1d
        kernel = torch.pow(PHI_INV, torch.arange(max_len, dtype=torch.float32))
        self.register_buffer('kernel', kernel)
        self.max_len = max_len

    def forward(self, x):
        batch_size, seq_len, dim = x.shape

        # Project and gate
        x_proj = self.proj(x)
        gate = torch.sigmoid(self.gate_proj(x))
        x_gated = gate * x_proj

        # Use cumsum trick with truncated kernel for numerical stability
        # For each position t, we compute: Σ_{i=0}^{t} φ^{t-i} * x_i
        # This is equivalent to conv with kernel [1, φ, φ^2, ...]

        # Transpose for conv1d: (B, D, L)
        x_t = x_gated.transpose(1, 2).float()

        # Get kernel for this sequence length
        kernel = self.kernel[:seq_len].flip(0).view(1, 1, -1)

        # Apply depthwise conv: each channel independently
        # Output: (B, D, L)
        output = F.conv1d(
            x_t.reshape(batch_size * dim, 1, seq_len),
            kernel,
            padding=seq_len - 1
        )[:, :, :seq_len]

        output = output.view(batch_size, dim, seq_len).transpose(1, 2)

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


class InterleavedBlock(nn.Module):
    def __init__(self, dim, layer_type, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        if layer_type == 'attention':
            self.core = CausalAttentionLayer(dim, num_heads, dropout)
        elif layer_type == 'engram_seq':
            self.core = SequentialEngramLayer(dim, dropout)
        elif layer_type == 'engram_fast':
            self.core = FastEngramLayer(dim, dropout)
        elif layer_type == 'engram_conv':
            self.core = FastEngramLayerV2(dim, dropout)
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
    def __init__(self, dim, vocab_size, engram_type='engram_seq', num_layers=8, num_heads=8, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        # A→E→A→E→A→E→A→E pattern
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


def load_wikitext():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")

    train_data = dataset["train"].map(tokenize, batched=True, remove_columns=["text"])
    val_data = dataset["validation"].map(tokenize, batched=True, remove_columns=["text"])

    train_data.set_format(type="torch", columns=["input_ids"])
    val_data.set_format(type="torch", columns=["input_ids"])

    return train_data, val_data, tokenizer.vocab_size


def train_epoch(model, data, optimizer, scaler, device, batch_size=32):
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
def evaluate(model, data, device, batch_size=32):
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
def benchmark_tps(model, device, seq_len=128, batch_size=8, num_runs=100):
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

    train_data, val_data, vocab_size = load_wikitext()
    print(f"Vocab size: {vocab_size}", flush=True)

    results = {}

    # 1. Pure Attention baseline
    model = PureAttentionModel(dim=256, vocab_size=vocab_size, num_layers=8).to(device)
    ppl, tps = train_and_benchmark("Pure Attention (8 layers)", model, train_data, val_data, device)
    results['Pure Attention'] = (ppl, tps)
    del model
    torch.cuda.empty_cache()

    # 2. Sequential Engram (original - slow)
    model = InterleavedModel(dim=256, vocab_size=vocab_size, engram_type='engram_seq', num_layers=8).to(device)
    ppl, tps = train_and_benchmark("Interleaved 4A+4E (Sequential)", model, train_data, val_data, device)
    results['Sequential Engram'] = (ppl, tps)
    del model
    torch.cuda.empty_cache()

    # 3. Fast Engram (cumsum trick)
    model = InterleavedModel(dim=256, vocab_size=vocab_size, engram_type='engram_fast', num_layers=8).to(device)
    ppl, tps = train_and_benchmark("Interleaved 4A+4E (Fast Cumsum)", model, train_data, val_data, device)
    results['Fast Engram (cumsum)'] = (ppl, tps)

    # Also try compiled
    try:
        print("\nCompiling fast engram model...", flush=True)
        model_compiled = torch.compile(model, mode="reduce-overhead")
        dummy = torch.randint(0, vocab_size, (8, 128), device=device)
        for _ in range(30):
            with torch.amp.autocast('cuda'):
                _ = model_compiled(dummy)
        torch.cuda.synchronize()
        tps_compiled = benchmark_tps(model_compiled, device)
        print(f"Fast Engram (cumsum + compiled): {tps_compiled:,.0f} TPS", flush=True)
        results['Fast Engram (compiled)'] = (ppl, tps_compiled)
    except Exception as e:
        print(f"Compile failed: {e}", flush=True)

    del model
    torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}", flush=True)
    print("RESULTS SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)

    baseline_ppl, baseline_tps = results['Pure Attention']
    for name, (ppl, tps) in sorted(results.items(), key=lambda x: x[1][0]):
        ppl_diff = (ppl - baseline_ppl) / baseline_ppl * 100
        tps_diff = (tps - baseline_tps) / baseline_tps * 100
        print(f"{name:<35}: {ppl:6.2f} PPL ({ppl_diff:+5.1f}%), {tps:>10,.0f} TPS ({tps_diff:+6.1f}%)", flush=True)

    print(f"\n{'='*70}", flush=True)

    # Check if we hit the goal
    if 'Fast Engram (compiled)' in results:
        fast_ppl, fast_tps = results['Fast Engram (compiled)']
        seq_ppl, _ = results['Sequential Engram']

        ppl_preserved = fast_ppl <= seq_ppl * 1.05  # Within 5% of sequential
        speed_goal = fast_tps >= 100000

        if speed_goal and ppl_preserved:
            print("SUCCESS: 100k+ TPS with PPL preserved!", flush=True)
        elif speed_goal:
            print(f"SPEED OK ({fast_tps:,.0f} TPS) but PPL degraded", flush=True)
        elif ppl_preserved:
            print(f"PPL OK but speed only {fast_tps:,.0f} TPS", flush=True)
        else:
            print(f"Need more work: {fast_tps:,.0f} TPS, PPL = {fast_ppl:.2f}", flush=True)


if __name__ == "__main__":
    main()
