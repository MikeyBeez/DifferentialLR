#!/usr/bin/env python3
"""
Inverted Maturation Test

Tests the hypothesis that models should START clamped and EXPAND to unbounded:

1. Inverted: Tanh → GELU → ReLU (focus first, then explore)
2. Ceiling Lift: α * tanh(x/α) with α growing (dimmer switch)

Hypothesis: Starting saturated forces focus on salient features,
then unclamping allows handling high-noise outliers.

Usage:
    python experiments/inverted_maturation_test.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")


class CeilingLiftActivation(nn.Module):
    """
    Scaled tanh that "lifts the ceiling" as training progresses.

    f(x) = α * tanh(x / α)

    When α=1: tight saturation (tanh-like)
    When α=5: wide range, nearly linear for small x
    """

    def __init__(self, initial_alpha=1.0):
        super().__init__()
        self.alpha = initial_alpha

    def forward(self, x):
        return self.alpha * torch.tanh(x / self.alpha)

    def set_alpha(self, alpha):
        self.alpha = alpha


class InvertedCurriculum:
    """
    Tracks training progress and determines activation phase.

    Inverted: Tanh → GELU → ReLU (opposite of original hypothesis)
    """

    def __init__(self, total_epochs, phase_fractions=[0.3, 0.4, 0.3]):
        self.total_epochs = total_epochs
        self.phase_fractions = phase_fractions
        self.current_phase = 0

    def update(self, epoch):
        progress = epoch / self.total_epochs
        cumulative = 0
        for i, frac in enumerate(self.phase_fractions):
            cumulative += frac
            if progress <= cumulative:
                if i != self.current_phase:
                    phases = ['Tanh (clamped)', 'GELU (expanding)', 'ReLU (unbounded)']
                    print(f"  → Phase: {phases[i]}")
                    self.current_phase = i
                return i
        return len(self.phase_fractions) - 1

    def get_activation(self):
        if self.current_phase == 0:
            return torch.tanh
        elif self.current_phase == 1:
            return F.gelu
        else:
            return F.relu


@dataclass
class Config:
    vocab_size: int = 256
    dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    beta: float = 2.0


class HopfieldAttention(nn.Module):
    def __init__(self, config):
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


class DynamicBlock(nn.Module):
    """Block with dynamic activation (curriculum or ceiling lift)."""

    def __init__(self, config, mode='gelu'):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.dim)
        self.attn = HopfieldAttention(config)
        self.norm2 = nn.LayerNorm(config.dim)
        self.ffn_up = nn.Linear(config.dim, config.dim * 4)
        self.ffn_down = nn.Linear(config.dim * 4, config.dim)

        self.mode = mode
        if mode == 'ceiling':
            self.activation = CeilingLiftActivation(initial_alpha=1.0)
        else:
            self.activation = None
            self.activation_fn = F.gelu  # Default, will be overwritten

    def set_activation_fn(self, fn):
        """For curriculum mode."""
        self.activation_fn = fn

    def set_alpha(self, alpha):
        """For ceiling lift mode."""
        if self.activation:
            self.activation.set_alpha(alpha)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        h = self.ffn_up(self.norm2(x))

        if self.mode == 'ceiling':
            h = self.activation(h)
        else:
            h = self.activation_fn(h)

        x = x + self.ffn_down(h)
        return x


class DynamicModel(nn.Module):
    def __init__(self, config, mode='gelu'):
        super().__init__()
        self.config = config
        self.mode = mode
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([DynamicBlock(config, mode) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.dim)
        self.output = nn.Linear(config.dim, config.vocab_size)

    def set_activation_fn(self, fn):
        for block in self.blocks:
            block.set_activation_fn(fn)

    def set_alpha(self, alpha):
        for block in self.blocks:
            block.set_alpha(alpha)

    def forward(self, x):
        h = self.embedding(x)
        for block in self.blocks:
            h = block(h)
        return self.output(self.norm(h))


def generate_recall_batch(batch_size, seq_len, vocab_size, device='cuda'):
    KEY_START, KEY_END = 10, 110
    VAL_START, VAL_END = 110, 210
    NOISE_START = 210
    sequences, targets = [], []
    for _ in range(batch_size):
        seq = torch.zeros(seq_len, dtype=torch.long, device=device)
        key = torch.randint(KEY_START, KEY_END, (1,), device=device)
        val = torch.randint(VAL_START, VAL_END, (1,), device=device)
        seq[0], seq[1] = key, val
        seq[2:-1] = torch.randint(NOISE_START, vocab_size, (seq_len - 3,), device=device)
        seq[-1] = key
        sequences.append(seq)
        targets.append(val)
    return torch.stack(sequences), torch.tensor(targets, device=device).squeeze()


def train_epoch(model, optimizer, seq_len, config, num_batches=100, batch_size=32, device='cuda'):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    for _ in range(num_batches):
        seq, targets = generate_recall_batch(batch_size, seq_len, config.vocab_size, device)
        optimizer.zero_grad()
        logits = model(seq)
        loss = F.cross_entropy(logits[:, -1], targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        total_correct += (logits[:, -1].argmax(-1) == targets).sum().item()
        total_samples += batch_size
    return total_loss / num_batches, total_correct / total_samples


@torch.no_grad()
def evaluate(model, seq_len, config, num_batches=50, batch_size=32, device='cuda'):
    model.eval()
    total_correct, total_samples = 0, 0
    for _ in range(num_batches):
        seq, targets = generate_recall_batch(batch_size, seq_len, config.vocab_size, device)
        logits = model(seq)
        total_correct += (logits[:, -1].argmax(-1) == targets).sum().item()
        total_samples += batch_size
    return total_correct / total_samples


@torch.no_grad()
def evaluate_robustness(model, seq_len, config, noise_levels=[0.0, 0.1, 0.2, 0.5, 1.0],
                        num_batches=30, batch_size=32, device='cuda'):
    model.eval()
    results = {}
    for noise in noise_levels:
        total_correct, total_samples = 0, 0
        for _ in range(num_batches):
            seq, targets = generate_recall_batch(batch_size, seq_len, config.vocab_size, device)
            h = model.embedding(seq)
            if noise > 0:
                h = h + noise * torch.randn_like(h)
            for block in model.blocks:
                h = block(h)
            logits = model.output(model.norm(h))
            total_correct += (logits[:, -1].argmax(-1) == targets).sum().item()
            total_samples += batch_size
        results[noise] = total_correct / total_samples
    return results


def train_inverted(config, seq_len, device, epochs=30):
    """Train with inverted maturation: Tanh → GELU → ReLU."""
    print(f"\n{'='*60}")
    print("INVERTED MATURATION: Tanh → GELU → ReLU")
    print("='*60")

    model = DynamicModel(config, mode='curriculum').to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    curriculum = InvertedCurriculum(epochs)

    best_acc, solved_epoch = 0, None

    for epoch in range(1, epochs + 1):
        curriculum.update(epoch)
        model.set_activation_fn(curriculum.get_activation())

        loss, _ = train_epoch(model, optimizer, seq_len, config, device=device)
        val_acc = evaluate(model, seq_len, config, device=device)

        if val_acc > best_acc:
            best_acc = val_acc
        if val_acc >= 0.99 and solved_epoch is None:
            solved_epoch = epoch
            print(f"  Epoch {epoch}: SOLVED! Val {val_acc:.1%}")

        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: Loss {loss:.3f}, Val {val_acc:.1%}")

    robustness = evaluate_robustness(model, seq_len, config, device=device)
    print(f"\nRobustness: " + ", ".join(f"{n}:{r:.1%}" for n, r in robustness.items()))

    return {'accuracy': best_acc, 'solved': solved_epoch or epochs, 'robustness': robustness}


def train_ceiling_lift(config, seq_len, device, epochs=30, alpha_start=1.0, alpha_end=5.0):
    """Train with ceiling lift: α grows from 1 to 5."""
    print(f"\n{'='*60}")
    print(f"CEILING LIFT: α = {alpha_start} → {alpha_end}")
    print("='*60")

    model = DynamicModel(config, mode='ceiling').to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    best_acc, solved_epoch = 0, None

    for epoch in range(1, epochs + 1):
        # Linear interpolation of alpha
        progress = epoch / epochs
        alpha = alpha_start + (alpha_end - alpha_start) * progress
        model.set_alpha(alpha)

        loss, _ = train_epoch(model, optimizer, seq_len, config, device=device)
        val_acc = evaluate(model, seq_len, config, device=device)

        if val_acc > best_acc:
            best_acc = val_acc
        if val_acc >= 0.99 and solved_epoch is None:
            solved_epoch = epoch
            print(f"  Epoch {epoch}: SOLVED! Val {val_acc:.1%}, α={alpha:.2f}")

        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: Loss {loss:.3f}, Val {val_acc:.1%}, α={alpha:.2f}")

    robustness = evaluate_robustness(model, seq_len, config, device=device)
    print(f"\nRobustness: " + ", ".join(f"{n}:{r:.1%}" for n, r in robustness.items()))

    return {'accuracy': best_acc, 'solved': solved_epoch or epochs, 'robustness': robustness}


def train_baseline(config, seq_len, device, epochs=30, activation='gelu'):
    """Train baseline with fixed activation."""
    print(f"\n{'='*60}")
    print(f"BASELINE: {activation.upper()}")
    print("='*60")

    model = DynamicModel(config, mode='fixed').to(device)

    if activation == 'gelu':
        act_fn = F.gelu
    elif activation == 'relu':
        act_fn = F.relu
    elif activation == 'tanh':
        act_fn = torch.tanh

    model.set_activation_fn(act_fn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    best_acc, solved_epoch = 0, None

    for epoch in range(1, epochs + 1):
        loss, _ = train_epoch(model, optimizer, seq_len, config, device=device)
        val_acc = evaluate(model, seq_len, config, device=device)

        if val_acc > best_acc:
            best_acc = val_acc
        if val_acc >= 0.99 and solved_epoch is None:
            solved_epoch = epoch
            print(f"  Epoch {epoch}: SOLVED! Val {val_acc:.1%}")

        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: Loss {loss:.3f}, Val {val_acc:.1%}")

    robustness = evaluate_robustness(model, seq_len, config, device=device)
    print(f"\nRobustness: " + ", ".join(f"{n}:{r:.1%}" for n, r in robustness.items()))

    return {'accuracy': best_acc, 'solved': solved_epoch or epochs, 'robustness': robustness}


def main():
    print("="*60)
    print("INVERTED MATURATION TEST")
    print("Does starting clamped and expanding to unbounded help?")
    print("="*60)

    config = Config()
    seq_len = 256

    results = {}

    # Baselines
    results['GELU (baseline)'] = train_baseline(config, seq_len, device, activation='gelu')
    results['ReLU (baseline)'] = train_baseline(config, seq_len, device, activation='relu')

    # Inverted maturation
    results['Inverted (Tanh→GELU→ReLU)'] = train_inverted(config, seq_len, device)

    # Ceiling lift variants
    results['Ceiling Lift (1→3)'] = train_ceiling_lift(config, seq_len, device, alpha_start=1.0, alpha_end=3.0)
    results['Ceiling Lift (1→5)'] = train_ceiling_lift(config, seq_len, device, alpha_start=1.0, alpha_end=5.0)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\n{'Config':<30} {'Solved':<8} {'Noise 0.5':<10} {'Noise 1.0':<10}")
    print("-"*60)
    for name, r in results.items():
        rob = r['robustness']
        print(f"{name:<30} {r['solved']:<8} {rob[0.5]:<10.1%} {rob[1.0]:<10.1%}")

    # Find best
    best_05 = max(results.keys(), key=lambda k: results[k]['robustness'][0.5])
    best_10 = max(results.keys(), key=lambda k: results[k]['robustness'][1.0])

    print(f"\nBest at noise=0.5: {best_05} ({results[best_05]['robustness'][0.5]:.1%})")
    print(f"Best at noise=1.0: {best_10} ({results[best_10]['robustness'][1.0]:.1%})")


if __name__ == "__main__":
    main()
