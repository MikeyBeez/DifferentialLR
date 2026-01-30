#!/usr/bin/env python3
"""
Hybrid Architecture with Projection Bottleneck

Combines two key insights:
1. Static Function Hypothesis: Early layers need dynamic attention,
   later layers can use static MLPs
2. Projection Bottleneck: The output contains more information than needed.
   A learned projection can compress to the essential "geometry"

Architecture:
- Layers 1-4: Full softmax attention (dynamic feature extraction)
- Layers 5-8: Static MLP + Projection Bottleneck (compress to essentials)

The projection learns what information from the MLP output is actually
necessary for the task, discarding redundancy.
"""

import sys
sys.path.insert(0, '/home/bee/Code/LinearAttention')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import math
import time


class SoftmaxAttention(nn.Module):
    """Full softmax attention for early layers."""

    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        B, L, D = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(out)


class StaticMLPWithProjection(nn.Module):
    """
    Static MLP with projection bottleneck for later layers.

    The MLP learns the transformation, then the projection
    compresses to essential information only.

    Key insight: Attention output is high-dimensional but
    the essential "geometric navigation" may live in a
    lower-dimensional subspace.
    """

    def __init__(self, hidden_dim, mlp_expansion=2, bottleneck_ratio=0.5):
        super().__init__()
        mlp_dim = hidden_dim * mlp_expansion
        bottleneck_dim = int(hidden_dim * bottleneck_ratio)

        # MLP learns the transformation
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim),
        )

        # Projection bottleneck compresses to essentials
        self.down_proj = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        self.up_proj = nn.Linear(bottleneck_dim, hidden_dim, bias=False)
        self.bottleneck_norm = nn.LayerNorm(bottleneck_dim)

    def forward(self, x):
        # MLP transformation
        mlp_out = self.mlp(x)

        # Project through bottleneck
        compressed = self.down_proj(mlp_out)
        compressed = self.bottleneck_norm(compressed)
        refined = self.up_proj(compressed)

        return refined


class StaticMLPWithLearnedGate(nn.Module):
    """
    Static MLP with learned gating and projection.

    Adds data-dependent gating: the model learns when
    to use the MLP transformation vs pass through.
    """

    def __init__(self, hidden_dim, mlp_expansion=2, bottleneck_ratio=0.5):
        super().__init__()
        mlp_dim = hidden_dim * mlp_expansion
        bottleneck_dim = int(hidden_dim * bottleneck_ratio)

        # MLP transformation
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim),
        )

        # Learned gate: how much to use MLP vs skip
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

        # Projection bottleneck
        self.down_proj = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        self.up_proj = nn.Linear(bottleneck_dim, hidden_dim, bias=False)
        self.bottleneck_norm = nn.LayerNorm(bottleneck_dim)

    def forward(self, x):
        # MLP transformation
        mlp_out = self.mlp(x)

        # Learned gate
        gate = self.gate(x)  # (B, L, 1)
        gated_out = gate * mlp_out + (1 - gate) * x

        # Project through bottleneck
        compressed = self.down_proj(gated_out)
        compressed = self.bottleneck_norm(compressed)
        refined = self.up_proj(compressed)

        return refined


class StaticMLPWithMultiHeadProjection(nn.Module):
    """
    Static MLP with multi-head projection bottleneck.

    Different "heads" of the projection can capture
    different aspects of the essential geometry.
    """

    def __init__(self, hidden_dim, mlp_expansion=2, num_proj_heads=4, bottleneck_ratio=0.5):
        super().__init__()
        mlp_dim = hidden_dim * mlp_expansion
        self.num_heads = num_proj_heads
        self.head_dim = hidden_dim // num_proj_heads
        bottleneck_per_head = int(self.head_dim * bottleneck_ratio)

        # MLP transformation
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim),
        )

        # Multi-head projection
        self.down_projs = nn.ModuleList([
            nn.Linear(self.head_dim, bottleneck_per_head, bias=False)
            for _ in range(num_proj_heads)
        ])
        self.up_projs = nn.ModuleList([
            nn.Linear(bottleneck_per_head, self.head_dim, bias=False)
            for _ in range(num_proj_heads)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(bottleneck_per_head)
            for _ in range(num_proj_heads)
        ])

    def forward(self, x):
        B, L, D = x.shape

        # MLP transformation
        mlp_out = self.mlp(x)

        # Split into heads
        mlp_out = mlp_out.view(B, L, self.num_heads, self.head_dim)

        # Project each head through its bottleneck
        refined_heads = []
        for h in range(self.num_heads):
            head = mlp_out[:, :, h, :]  # (B, L, head_dim)
            compressed = self.norms[h](self.down_projs[h](head))
            refined = self.up_projs[h](compressed)
            refined_heads.append(refined)

        # Combine heads
        out = torch.stack(refined_heads, dim=2).view(B, L, D)
        return out


class TransformerBlock(nn.Module):
    """Transformer block with configurable attention/MLP type."""

    def __init__(self, hidden_dim, num_heads, intermediate_dim, dropout,
                 block_type='softmax', bottleneck_ratio=0.5):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_dim)

        if block_type == 'softmax':
            self.attention = SoftmaxAttention(hidden_dim, num_heads)
        elif block_type == 'mlp':
            self.attention = StaticMLPWithProjection(hidden_dim, bottleneck_ratio=1.0)  # No bottleneck
        elif block_type == 'mlp_proj':
            self.attention = StaticMLPWithProjection(hidden_dim, bottleneck_ratio=bottleneck_ratio)
        elif block_type == 'mlp_gate_proj':
            self.attention = StaticMLPWithLearnedGate(hidden_dim, bottleneck_ratio=bottleneck_ratio)
        elif block_type == 'mlp_multihead_proj':
            self.attention = StaticMLPWithMultiHeadProjection(hidden_dim, bottleneck_ratio=bottleneck_ratio)
        else:
            raise ValueError(f"Unknown block type: {block_type}")

        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attention(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class HybridProjectionTransformer(nn.Module):
    """
    Hybrid Transformer with projection bottleneck in later layers.

    Combines:
    1. Static Function Hypothesis (early softmax, late MLP)
    2. Projection bottleneck (compress to essential geometry)
    """

    def __init__(self, vocab_size, hidden_dim=512, num_layers=8, num_heads=8,
                 intermediate_dim=2048, max_seq_len=1024, dropout=0.1,
                 early_layers=4, late_type='mlp_proj', bottleneck_ratio=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            if i < early_layers:
                block_type = 'softmax'
            else:
                block_type = late_type

            self.blocks.append(
                TransformerBlock(hidden_dim, num_heads, intermediate_dim, dropout,
                               block_type, bottleneck_ratio)
            )

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        output = {"logits": logits}

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            output["loss"] = loss

        return output


def get_wikitext_data(tokenizer, seq_length=512, batch_size=8):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    val_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=False, padding=False)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    val_tokenized = val_dataset.map(tokenize, batched=True, remove_columns=["text"])

    all_tokens, val_tokens = [], []
    for item in tokenized:
        all_tokens.extend(item["input_ids"])
    for item in val_tokenized:
        val_tokens.extend(item["input_ids"])

    def create_sequences(tokens, seq_len):
        return torch.tensor([tokens[i:i+seq_len] for i in range(0, len(tokens)-seq_len, seq_len)])

    train_loader = DataLoader(create_sequences(all_tokens, seq_length), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(create_sequences(val_tokens, seq_length), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, scaler, device):
    model.train()
    total_loss, num_batches = 0, 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            outputs = model(batch, labels=batch)
            loss = outputs["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    total_loss, num_batches = 0, 0

    for batch in val_loader:
        batch = batch.to(device)
        with torch.amp.autocast('cuda'):
            outputs = model(batch, labels=batch)
        total_loss += outputs["loss"].item()
        num_batches += 1

    return math.exp(total_loss / num_batches)


@torch.no_grad()
def benchmark_speed(model, device, seq_len=512, batch_size=1, num_runs=50):
    model.eval()
    dummy = torch.randint(0, 1000, (batch_size, seq_len), device=device)

    # Warmup
    for _ in range(10):
        _ = model(dummy)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_runs):
        _ = model(dummy)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    return (num_runs * batch_size * seq_len) / elapsed


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n" + "="*70)
    print("HYBRID ARCHITECTURE WITH PROJECTION BOTTLENECK")
    print("="*70)
    print("\nCombining:")
    print("  1. Static Function Hypothesis (early softmax, late MLP)")
    print("  2. Projection Bottleneck (compress to essential geometry)")

    print("\nLoading data...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_loader, val_loader = get_wikitext_data(tokenizer, seq_length=512, batch_size=8)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    configs = [
        # Baselines
        ("Full Softmax", 8, 'softmax', 1.0),
        ("4 Softmax + 4 MLP (no proj)", 4, 'mlp', 1.0),

        # With projection bottleneck
        ("4 Softmax + 4 MLP+Proj (0.5x)", 4, 'mlp_proj', 0.5),
        ("4 Softmax + 4 MLP+Proj (0.25x)", 4, 'mlp_proj', 0.25),

        # With gating + projection
        ("4 Softmax + 4 MLP+Gate+Proj", 4, 'mlp_gate_proj', 0.5),

        # With multi-head projection
        ("4 Softmax + 4 MLP+MultiProj", 4, 'mlp_multihead_proj', 0.5),

        # More aggressive: 6 MLP layers
        ("2 Softmax + 6 MLP+Proj", 2, 'mlp_proj', 0.5),
    ]

    results = []

    for name, early_layers, late_type, bottleneck_ratio in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"  Early layers (softmax): {early_layers}")
        print(f"  Late layers ({late_type}): {8 - early_layers}")
        print(f"  Bottleneck ratio: {bottleneck_ratio}")
        print("="*60)

        model = HybridProjectionTransformer(
            vocab_size=tokenizer.vocab_size,
            hidden_dim=512,
            num_layers=8,
            num_heads=8,
            intermediate_dim=2048,
            early_layers=early_layers,
            late_type=late_type if early_layers < 8 else 'softmax',
            bottleneck_ratio=bottleneck_ratio,
        ).to(device)

        num_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {num_params:,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
        scaler = torch.amp.GradScaler('cuda')

        best_ppl = float('inf')
        start = time.time()

        for epoch in range(10):
            train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
            train_ppl = math.exp(train_loss)
            eval_ppl = evaluate(model, val_loader, device)

            marker = "*" if eval_ppl < best_ppl else ""
            best_ppl = min(best_ppl, eval_ppl)

            elapsed = (time.time() - start) / 60
            print(f"Epoch {epoch+1:2d} | Train PPL: {train_ppl:7.1f} | Eval PPL: {eval_ppl:7.1f} {marker} | {elapsed:.1f}m", flush=True)

        speed = benchmark_speed(model, device)

        results.append({
            'name': name,
            'ppl': best_ppl,
            'params': num_params,
            'speed': speed,
        })

        print(f"\n{name}:")
        print(f"  Best PPL: {best_ppl:.1f}")
        print(f"  Speed: {speed:.0f} tokens/sec")

    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'Configuration':<40} | {'PPL':>7} | {'Speed':>10} | {'Params':>12}")
    print("-"*75)

    baseline_ppl = results[0]['ppl']
    baseline_speed = results[0]['speed']

    for r in results:
        ppl_diff = (r['ppl'] - baseline_ppl) / baseline_ppl * 100
        speed_diff = (r['speed'] - baseline_speed) / baseline_speed * 100
        print(f"{r['name']:<40} | {r['ppl']:>7.1f} | {r['speed']:>7.0f} t/s | {r['params']:>10,}")
        if r['name'] != results[0]['name']:
            print(f"{'':40} | {ppl_diff:>+6.1f}% | {speed_diff:>+6.1f}%")

    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    # Find best quality and best speed
    best_quality = min(results, key=lambda x: x['ppl'])
    best_speed = max(results, key=lambda x: x['speed'])

    # Find best tradeoff (lowest PPL among faster-than-baseline)
    faster_configs = [r for r in results if r['speed'] > baseline_speed]
    if faster_configs:
        best_tradeoff = min(faster_configs, key=lambda x: x['ppl'])
        print(f"Best quality:  {best_quality['name']} (PPL {best_quality['ppl']:.1f})")
        print(f"Best speed:    {best_speed['name']} ({best_speed['speed']:.0f} t/s)")
        print(f"Best tradeoff: {best_tradeoff['name']} (PPL {best_tradeoff['ppl']:.1f}, {best_tradeoff['speed']:.0f} t/s)")

    print("\n" + "="*70)
    print("COMPARISON TO PREVIOUS RESULTS")
    print("="*70)
    print(f"  Linear attention (learned gate): PPL 167.4")
    print(f"  Pure softmax baseline:           PPL 173.4")
    print(f"  Previous hybrid (4+4 MLP):       PPL 174.5, 156k t/s")
    print(f"  Best from this experiment:       PPL {best_quality['ppl']:.1f}")


if __name__ == "__main__":
    main()
