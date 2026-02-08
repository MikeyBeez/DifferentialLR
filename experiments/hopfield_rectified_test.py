#!/usr/bin/env python3
"""
Hopfield + Rectified Bottleneck Test

Combines two ideas:
1. Hopfield β=2: Sharpen attention to find the right position
2. Rectified bottleneck: Sparse representations to extract clean signal

Hypothesis: Together they might close the remaining gap.

Usage:
    python experiments/hopfield_rectified_test.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")


@dataclass
class Config:
    vocab_size: int = 256
    dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.0

    # Hopfield setting
    beta: float = 2.0  # Attention sharpness

    # Rectified bottleneck settings
    use_bottleneck: bool = True
    bottleneck_dim: int = 64
    initial_threshold: float = 0.1

    # Sparsity settings
    sparsity_weight: float = 0.01
    diversity_weight: float = 0.1


class LearnedShrinkage(nn.Module):
    """Soft shrinkage with learnable per-dimension threshold."""

    def __init__(self, dim, initial_threshold=0.1):
        super().__init__()
        self.threshold = nn.Parameter(torch.full((dim,), initial_threshold))

    def forward(self, x):
        tau = F.softplus(self.threshold)
        magnitude = x.abs()
        shrunk = F.relu(magnitude - tau)
        return x.sign() * shrunk


class RectifiedBottleneck(nn.Module):
    """Bottleneck with learned sparsity."""

    def __init__(self, input_dim, bottleneck_dim, initial_threshold=0.1):
        super().__init__()
        self.down = nn.Linear(input_dim, bottleneck_dim)
        self.shrinkage = LearnedShrinkage(bottleneck_dim, initial_threshold)
        self.up = nn.Linear(bottleneck_dim, input_dim)
        self.norm = nn.LayerNorm(bottleneck_dim)

    def forward(self, x):
        h = self.norm(self.down(x))
        h_sparse = self.shrinkage(h)
        return self.up(h_sparse), h_sparse


class HopfieldAttention(nn.Module):
    """Hopfield attention with configurable β."""

    def __init__(self, config: Config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.dim // config.num_heads
        self.scale = self.head_dim ** -0.5
        self.beta = config.beta

        self.qkv = nn.Linear(config.dim, config.dim * 3)
        self.out = nn.Linear(config.dim, config.dim)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        # Hopfield attention with β scaling
        attn = (q @ k.transpose(-2, -1)) * self.scale * self.beta

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class HopfieldRectifiedBlock(nn.Module):
    """Block with Hopfield attention and optional rectified FFN."""

    def __init__(self, config: Config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.dim)
        self.attn = HopfieldAttention(config)
        self.norm2 = nn.LayerNorm(config.dim)

        self.use_bottleneck = config.use_bottleneck

        if config.use_bottleneck:
            self.ffn_up = nn.Linear(config.dim, config.dim * 4)
            self.bottleneck = RectifiedBottleneck(
                config.dim * 4, config.bottleneck_dim, config.initial_threshold
            )
            self.ffn_down = nn.Linear(config.dim * 4, config.dim)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(config.dim, config.dim * 4),
                nn.GELU(),
                nn.Linear(config.dim * 4, config.dim)
            )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))

        if self.use_bottleneck:
            h = F.gelu(self.ffn_up(self.norm2(x)))
            h, sparse_act = self.bottleneck(h)
            x = x + self.ffn_down(h)
            return x, sparse_act
        else:
            x = x + self.ffn(self.norm2(x))
            return x, None


class HopfieldRectifiedModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([HopfieldRectifiedBlock(config) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.dim)
        self.output = nn.Linear(config.dim, config.vocab_size)

    def forward(self, x):
        h = self.embedding(x)

        sparse_activations = []
        for block in self.blocks:
            h, sparse_act = block(h)
            if sparse_act is not None:
                sparse_activations.append(sparse_act)

        h = self.norm(h)
        logits = self.output(h)

        if sparse_activations:
            sparse_activations = torch.stack(sparse_activations, dim=0)
        else:
            sparse_activations = None

        return logits, sparse_activations


def compute_sparsity_loss(activations):
    if activations is None:
        return torch.tensor(0.0)
    return activations.abs().mean()


def compute_diversity_loss(activations, eps=0.01):
    if activations is None:
        return torch.tensor(0.0)

    B, T, D = activations.shape[-3:]
    flat = activations.reshape(-1, D)
    mean = flat.mean(dim=0, keepdim=True)
    centered = flat - mean
    cov = (centered.T @ centered) / (flat.shape[0] - 1)
    cov = cov + eps * torch.eye(D, device=cov.device)

    try:
        log_det = torch.logdet(cov)
        if torch.isnan(log_det) or torch.isinf(log_det):
            log_det = torch.tensor(0.0, device=cov.device)
    except:
        log_det = torch.tensor(0.0, device=cov.device)

    return log_det


def generate_recall_batch(batch_size, seq_len, vocab_size, device='cuda'):
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
        seq[-1] = key

        sequences.append(seq)
        targets.append(val)

    return torch.stack(sequences), torch.tensor(targets, device=device).squeeze()


def train_epoch(model, optimizer, seq_len, config, sparsity_weight=0.0, diversity_weight=0.0,
                num_batches=100, batch_size=32, device='cuda'):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for _ in range(num_batches):
        seq, targets = generate_recall_batch(batch_size, seq_len, config.vocab_size, device)

        optimizer.zero_grad()
        logits, sparse_activations = model(seq)

        task_loss = F.cross_entropy(logits[:, -1], targets)
        sparsity_loss = compute_sparsity_loss(sparse_activations)
        diversity_loss = compute_diversity_loss(sparse_activations)

        loss = task_loss + sparsity_weight * sparsity_loss - diversity_weight * diversity_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += task_loss.item()
        preds = logits[:, -1].argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size

    return total_loss / num_batches, total_correct / total_samples


@torch.no_grad()
def evaluate(model, seq_len, config, num_batches=50, batch_size=32, device='cuda'):
    model.eval()
    total_correct = 0
    total_samples = 0

    for _ in range(num_batches):
        seq, targets = generate_recall_batch(batch_size, seq_len, config.vocab_size, device)
        logits, _ = model(seq)
        preds = logits[:, -1].argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size

    return total_correct / total_samples


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(name, config, seq_len, device, epochs=25):
    """Train a model configuration."""
    print(f"\n{'='*60}")
    print(f"{name} (distance={seq_len-2})")
    print(f"{'='*60}")

    model = HopfieldRectifiedModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print(f"Parameters: {count_params(model):,}")
    print(f"β={config.beta}, bottleneck={config.use_bottleneck}")

    best_acc = 0
    solved_epoch = None

    for epoch in range(1, epochs + 1):
        # Use curriculum for sparsity
        if config.use_bottleneck and epoch > 10:
            sparsity_w = config.sparsity_weight * ((epoch - 10) / 15)
        else:
            sparsity_w = 0.0

        loss, train_acc = train_epoch(
            model, optimizer, seq_len, config,
            sparsity_weight=sparsity_w,
            diversity_weight=config.diversity_weight if config.use_bottleneck else 0.0,
            device=device
        )
        val_acc = evaluate(model, seq_len, config, device=device)

        if val_acc > best_acc:
            best_acc = val_acc

        if epoch % 5 == 0 or val_acc >= 0.99:
            print(f"  Epoch {epoch}: Loss {loss:.3f}, Val {val_acc:.1%}")

        if val_acc >= 0.99 and solved_epoch is None:
            solved_epoch = epoch
            print(f"  SOLVED at epoch {epoch}!")

    print(f"\nFinal: {best_acc:.1%} accuracy")

    return {'accuracy': best_acc, 'solved_epoch': solved_epoch or epochs}


def main():
    print("="*60)
    print("HOPFIELD + RECTIFIED BOTTLENECK TEST")
    print("Does combining β=2 with sparse representations help?")
    print("="*60)

    seq_lengths = [128, 256, 512]

    configs = {
        'β=1 (baseline)': Config(beta=1.0, use_bottleneck=False),
        'β=2 (Hopfield)': Config(beta=2.0, use_bottleneck=False),
        'β=2 + Rectified': Config(beta=2.0, use_bottleneck=True),
        'β=1 + Rectified': Config(beta=1.0, use_bottleneck=True),
    }

    all_results = {}

    for config_name, config in configs.items():
        all_results[config_name] = {}
        for seq_len in seq_lengths:
            result = train_model(config_name, config, seq_len, device, epochs=25)
            all_results[config_name][seq_len] = result
            torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    print(f"\n{'Config':<20}", end="")
    for seq_len in seq_lengths:
        print(f"{'Dist '+str(seq_len-2):<15}", end="")
    print()
    print("-"*65)

    for config_name, results in all_results.items():
        print(f"{config_name:<20}", end="")
        for seq_len in seq_lengths:
            r = results[seq_len]
            print(f"{r['accuracy']:.0%} (ep {r['solved_epoch']})   ", end="")
        print()

    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    # Compare at distance 510
    dist_510 = 512
    print(f"\nAt distance 510:")
    for config_name, results in all_results.items():
        r = results[dist_510]
        print(f"  {config_name}: {r['accuracy']:.1%}")

    # Best configuration
    best_config = max(all_results.keys(),
                      key=lambda c: sum(all_results[c][s]['accuracy'] for s in seq_lengths))
    print(f"\nBest overall: {best_config}")


if __name__ == "__main__":
    main()
