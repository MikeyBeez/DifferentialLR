#!/usr/bin/env python3
"""
Softsign Maturation Test

Tests refined hardening strategies that preserve dynamic range:

1. Softsign: x/(1+|x|) - polynomial saturation, larger linear range
2. Judiciary: GELU everywhere except final layer (Tanh verdict)
3. Learnable: Soft saturation with learnable hardness parameter

Hypothesis: Softsign gives "Solid State" feel without "Tanh Lobotomy"

Usage:
    python experiments/softsign_maturation_test.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")


def softsign(x):
    """Softsign: x / (1 + |x|) - gentler saturation than tanh."""
    return x / (1 + x.abs())


class LearnableSaturation(nn.Module):
    """
    Learnable interpolation between GELU and Tanh.

    output = (1 - α) * gelu(x) + α * tanh(x)

    α starts at 0 (pure GELU) and can be pushed toward 1 (pure Tanh)
    when gradients stabilize.
    """

    def __init__(self):
        super().__init__()
        # Start with pure GELU (alpha=0)
        self.alpha_logit = nn.Parameter(torch.tensor(-5.0))  # sigmoid(-5) ≈ 0.007

    def forward(self, x):
        alpha = torch.sigmoid(self.alpha_logit)
        return (1 - alpha) * F.gelu(x) + alpha * torch.tanh(x)

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_logit).item()


@dataclass
class Config:
    vocab_size: int = 256
    dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.0
    beta: float = 2.0  # Hopfield attention


class HopfieldAttention(nn.Module):
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
        attn = (q @ k.transpose(-2, -1)) * self.scale * self.beta
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class FlexibleBlock(nn.Module):
    """Block with configurable activation."""

    def __init__(self, config: Config, activation='gelu'):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.dim)
        self.attn = HopfieldAttention(config)
        self.norm2 = nn.LayerNorm(config.dim)
        self.ffn_up = nn.Linear(config.dim, config.dim * 4)
        self.ffn_down = nn.Linear(config.dim * 4, config.dim)

        self.activation_type = activation
        if activation == 'learnable':
            self.activation = LearnableSaturation()
        else:
            self.activation = None  # Use functional

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        h = self.ffn_up(self.norm2(x))

        if self.activation_type == 'gelu':
            h = F.gelu(h)
        elif self.activation_type == 'tanh':
            h = torch.tanh(h)
        elif self.activation_type == 'softsign':
            h = softsign(h)
        elif self.activation_type == 'learnable':
            h = self.activation(h)

        x = x + self.ffn_down(h)
        return x


class FlexibleModel(nn.Module):
    def __init__(self, config: Config, activation_scheme='gelu'):
        """
        activation_scheme options:
        - 'gelu': All GELU
        - 'tanh': All Tanh
        - 'softsign': All Softsign
        - 'judiciary': GELU except final layer (Tanh)
        - 'learnable': Learnable GELU-Tanh interpolation
        """
        super().__init__()
        self.config = config
        self.activation_scheme = activation_scheme
        self.embedding = nn.Embedding(config.vocab_size, config.dim)

        # Build blocks with appropriate activations
        self.blocks = nn.ModuleList()
        for i in range(config.num_layers):
            if activation_scheme == 'judiciary' and i == config.num_layers - 1:
                # Final layer uses Tanh for "verdict"
                self.blocks.append(FlexibleBlock(config, 'tanh'))
            elif activation_scheme in ['gelu', 'tanh', 'softsign', 'learnable']:
                self.blocks.append(FlexibleBlock(config, activation_scheme))
            else:
                self.blocks.append(FlexibleBlock(config, 'gelu'))

        self.norm = nn.LayerNorm(config.dim)
        self.output = nn.Linear(config.dim, config.vocab_size)

    def forward(self, x):
        h = self.embedding(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        return self.output(h)

    def get_learnable_alphas(self):
        """Get learnable saturation parameters if using learnable scheme."""
        alphas = []
        for block in self.blocks:
            if hasattr(block.activation, 'alpha'):
                alphas.append(block.activation.alpha)
        return alphas


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


def train_epoch(model, optimizer, seq_len, config, num_batches=100, batch_size=32, device='cuda'):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for _ in range(num_batches):
        seq, targets = generate_recall_batch(batch_size, seq_len, config.vocab_size, device)
        optimizer.zero_grad()
        logits = model(seq)
        loss = F.cross_entropy(logits[:, -1], targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
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
        logits = model(seq)
        preds = logits[:, -1].argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size

    return total_correct / total_samples


@torch.no_grad()
def evaluate_robustness(model, seq_len, config, noise_levels=[0.0, 0.1, 0.2, 0.5, 1.0],
                        num_batches=30, batch_size=32, device='cuda'):
    """Test robustness to embedding noise."""
    model.eval()
    results = {}

    for noise in noise_levels:
        total_correct = 0
        total_samples = 0

        for _ in range(num_batches):
            seq, targets = generate_recall_batch(batch_size, seq_len, config.vocab_size, device)

            h = model.embedding(seq)
            if noise > 0:
                h = h + noise * torch.randn_like(h)

            for block in model.blocks:
                h = block(h)
            h = model.norm(h)
            logits = model.output(h)

            preds = logits[:, -1].argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_samples += batch_size

        results[noise] = total_correct / total_samples

    return results


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_and_evaluate(name, activation_scheme, config, seq_len, device, epochs=25):
    """Train and evaluate a model configuration."""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")

    model = FlexibleModel(config, activation_scheme).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print(f"Parameters: {count_params(model):,}")
    print(f"Activation: {activation_scheme}")

    best_acc = 0
    solved_epoch = None

    for epoch in range(1, epochs + 1):
        loss, train_acc = train_epoch(model, optimizer, seq_len, config, device=device)
        val_acc = evaluate(model, seq_len, config, device=device)

        if val_acc > best_acc:
            best_acc = val_acc

        # Show learnable alpha if applicable
        alphas = model.get_learnable_alphas()
        alpha_str = f", α={alphas[0]:.3f}" if alphas else ""

        if epoch % 5 == 0 or val_acc >= 0.99:
            print(f"  Epoch {epoch}: Loss {loss:.3f}, Val {val_acc:.1%}{alpha_str}")

        if val_acc >= 0.99 and solved_epoch is None:
            solved_epoch = epoch
            print(f"  SOLVED at epoch {epoch}!")

    # Robustness test
    print(f"\nRobustness (noise → accuracy):")
    robustness = evaluate_robustness(model, seq_len, config, device=device)
    for noise, acc in robustness.items():
        print(f"  {noise}: {acc:.1%}")

    return {
        'accuracy': best_acc,
        'solved_epoch': solved_epoch or epochs,
        'robustness': robustness
    }


def main():
    print("="*60)
    print("SOFTSIGN MATURATION TEST")
    print("Finding the Goldilocks activation for noise resistance")
    print("="*60)

    config = Config()
    seq_len = 256  # Distance 254

    schemes = {
        'GELU (baseline)': 'gelu',
        'Tanh (full hardening)': 'tanh',
        'Softsign (gentle hardening)': 'softsign',
        'Judiciary (GELU + Tanh final)': 'judiciary',
        'Learnable (GELU→Tanh blend)': 'learnable',
    }

    results = {}

    for name, scheme in schemes.items():
        results[name] = train_and_evaluate(name, scheme, config, seq_len, device)
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Noise Robustness Battle")
    print("="*70)

    print(f"\n{'Activation':<30} {'Solved':<8} {'Noise 0.2':<10} {'Noise 0.5':<10} {'Noise 1.0':<10}")
    print("-"*70)

    for name, r in results.items():
        rob = r['robustness']
        print(f"{name:<30} {r['solved_epoch']:<8} {rob[0.2]:<10.1%} {rob[0.5]:<10.1%} {rob[1.0]:<10.1%}")

    # Find best at high noise
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    best_at_05 = max(results.keys(), key=lambda k: results[k]['robustness'][0.5])
    best_at_10 = max(results.keys(), key=lambda k: results[k]['robustness'][1.0])

    print(f"\nBest at noise=0.5: {best_at_05} ({results[best_at_05]['robustness'][0.5]:.1%})")
    print(f"Best at noise=1.0: {best_at_10} ({results[best_at_10]['robustness'][1.0]:.1%})")

    # Compare Softsign to GELU
    gelu_rob = results['GELU (baseline)']['robustness']
    soft_rob = results['Softsign (gentle hardening)']['robustness']

    print(f"\nSoftsign vs GELU:")
    for noise in [0.2, 0.5, 1.0]:
        diff = soft_rob[noise] - gelu_rob[noise]
        symbol = "✓" if diff > 0 else "✗" if diff < 0 else "="
        print(f"  {symbol} Noise {noise}: {soft_rob[noise]:.1%} vs {gelu_rob[noise]:.1%} ({diff*100:+.1f}pp)")


if __name__ == "__main__":
    main()
