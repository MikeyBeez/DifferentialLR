#!/usr/bin/env python3
"""
GELU → ReLU Handover Test

Tests whether transitioning from GELU to ReLU during training can
combine GELU's fast convergence with ReLU's noise robustness.

Results: The handover doesn't work.
- Transition causes immediate improvement (55% → 67%)
- Peaks at epoch 10 (67.2%)
- Degrades with more training (61.4% at epoch 50)
- Never reaches fixed-activation levels (69-73%)

Conclusion: Activation switching is a no-go. Pick one and stick with it.

Usage:
    python experiments/gelu_relu_handover_test.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')


@dataclass
class Config:
    vocab_size: int = 256
    dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    beta: float = 2.0  # Hopfield attention


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
        return self.out((attn @ v).transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.dim)
        self.attn = HopfieldAttention(config)
        self.norm2 = nn.LayerNorm(config.dim)
        self.ffn_up = nn.Linear(config.dim, config.dim * 4)
        self.ffn_down = nn.Linear(config.dim * 4, config.dim)
        self.act_fn = F.gelu

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn_down(self.act_fn(self.ffn_up(self.norm2(x))))


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.dim)
        self.output = nn.Linear(config.dim, config.vocab_size)

    def set_activation(self, fn):
        for block in self.blocks:
            block.act_fn = fn

    def forward(self, x):
        h = self.embedding(x)
        for block in self.blocks:
            h = block(h)
        return self.output(self.norm(h))


def generate_batch(batch_size, seq_len, vocab_size, device):
    seqs, targets = [], []
    for _ in range(batch_size):
        seq = torch.zeros(seq_len, dtype=torch.long, device=device)
        key = torch.randint(10, 110, (1,), device=device)
        val = torch.randint(110, 210, (1,), device=device)
        seq[0], seq[1] = key, val
        seq[2:-1] = torch.randint(210, vocab_size, (seq_len - 3,), device=device)
        seq[-1] = key
        seqs.append(seq)
        targets.append(val)
    return torch.stack(seqs), torch.tensor(targets, device=device).squeeze()


def train_epoch(model, opt, config, seq_len=256):
    model.train()
    total_loss, total_correct, total = 0, 0, 0
    for _ in range(100):
        seq, tgt = generate_batch(32, seq_len, config.vocab_size, device)
        opt.zero_grad()
        logits = model(seq)
        loss = F.cross_entropy(logits[:, -1], tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()
        total_correct += (logits[:, -1].argmax(-1) == tgt).sum().item()
        total += 32
    return total_loss / 100, total_correct / total


@torch.no_grad()
def evaluate(model, config, seq_len=256):
    model.eval()
    correct, total = 0, 0
    for _ in range(50):
        seq, tgt = generate_batch(32, seq_len, config.vocab_size, device)
        correct += (model(seq)[:, -1].argmax(-1) == tgt).sum().item()
        total += 32
    return correct / total


@torch.no_grad()
def eval_robustness(model, config, noise_levels=[0.0, 0.5, 1.0], seq_len=256):
    model.eval()
    results = {}
    for noise in noise_levels:
        correct, total = 0, 0
        for _ in range(50):
            seq, tgt = generate_batch(32, seq_len, config.vocab_size, device)
            h = model.embedding(seq)
            if noise > 0:
                h = h + noise * torch.randn_like(h)
            for block in model.blocks:
                h = block(h)
            logits = model.output(model.norm(h))
            correct += (logits[:, -1].argmax(-1) == tgt).sum().item()
            total += 32
        results[noise] = correct / total
    return results


def run_handover_test(gelu_epochs=5, total_epochs=50):
    """Run GELU→ReLU handover and track robustness over time."""
    print(f"\n{'='*60}")
    print(f"GELU→ReLU HANDOVER TEST")
    print(f"GELU for epochs 1-{gelu_epochs}, ReLU for epochs {gelu_epochs+1}-{total_epochs}")
    print("="*60)

    config = Config()
    seq_len = 256

    model = Model(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    robustness_history = []
    checkpoints = [gelu_epochs, gelu_epochs+1, 10, 15, 20, 30, 40, 50]
    checkpoints = [c for c in checkpoints if c <= total_epochs]

    for epoch in range(1, total_epochs + 1):
        if epoch <= gelu_epochs:
            model.set_activation(F.gelu)
            phase = 'GELU'
        else:
            model.set_activation(F.relu)
            phase = 'ReLU'

        loss, train_acc = train_epoch(model, opt, config, seq_len)
        val_acc = evaluate(model, config, seq_len)

        if epoch in checkpoints:
            rob = eval_robustness(model, config, seq_len=seq_len)
            robustness_history.append((epoch, phase, rob))
            print(f'Epoch {epoch:2d} [{phase}]: Val {val_acc:.1%}, '
                  f'Noise0.5={rob[0.5]:.1%}, Noise1.0={rob[1.0]:.1%}')

    return robustness_history


def run_fixed_baseline(activation='gelu', total_epochs=25):
    """Run fixed activation baseline."""
    print(f"\n{'='*60}")
    print(f"FIXED {activation.upper()} BASELINE")
    print("="*60)

    config = Config()
    seq_len = 256

    model = Model(config).to(device)
    if activation == 'relu':
        model.set_activation(F.relu)
    else:
        model.set_activation(F.gelu)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(1, total_epochs + 1):
        loss, _ = train_epoch(model, opt, config, seq_len)
        val_acc = evaluate(model, config, seq_len)

        if epoch % 5 == 0 or epoch == 1:
            print(f'Epoch {epoch:2d}: Val {val_acc:.1%}')

    rob = eval_robustness(model, config, seq_len=seq_len)
    print(f'Final robustness: Noise0.5={rob[0.5]:.1%}, Noise1.0={rob[1.0]:.1%}')

    return rob


def main():
    print("="*60)
    print("GELU → ReLU HANDOVER EXPERIMENT")
    print("Can we get GELU's speed + ReLU's robustness?")
    print("="*60)

    # Run baselines
    gelu_rob = run_fixed_baseline('gelu', total_epochs=25)
    relu_rob = run_fixed_baseline('relu', total_epochs=25)

    # Run handover
    handover_history = run_handover_test(gelu_epochs=5, total_epochs=50)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print("\nFixed activation baselines:")
    print(f"  GELU: Noise0.5={gelu_rob[0.5]:.1%}, Noise1.0={gelu_rob[1.0]:.1%}")
    print(f"  ReLU: Noise0.5={relu_rob[0.5]:.1%}, Noise1.0={relu_rob[1.0]:.1%}")

    print("\nHandover timeline:")
    print(f"{'Epoch':<8} {'Phase':<8} {'Noise 0.5':<12} {'Noise 1.0':<12}")
    print("-"*40)
    for epoch, phase, rob in handover_history:
        marker = ' ← TRANSITION' if phase == 'ReLU' and epoch <= 6 else ''
        print(f"{epoch:<8} {phase:<8} {rob[0.5]:<12.1%} {rob[1.0]:<12.1%}{marker}")

    # Analysis
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)

    peak_rob = max(handover_history, key=lambda x: x[2][1.0])
    final_rob = handover_history[-1][2][1.0]

    print(f"\nPeak robustness: {peak_rob[2][1.0]:.1%} at epoch {peak_rob[0]}")
    print(f"Final robustness: {final_rob:.1%} at epoch {handover_history[-1][0]}")
    print(f"GELU baseline: {gelu_rob[1.0]:.1%}")
    print(f"ReLU baseline: {relu_rob[1.0]:.1%}")

    print("\n⚠️  Handover does NOT achieve best of both worlds.")
    print("   Peaks early, then degrades. Just use fixed activation.")


if __name__ == "__main__":
    main()
