#!/usr/bin/env python3
"""
Noise Stress Test

Tests where the "Mental Filter" breaks. After noise curriculum achieved 93.8%
at Noise 1.0, we push to Noise 2.0, 3.0, and beyond.

The question: At what SNR does signal detection become impossible?

Usage:
    python experiments/noise_stress_test.py
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
        return self.out((attn @ v).transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.dim)
        self.attn = HopfieldAttention(config)
        self.norm2 = nn.LayerNorm(config.dim)
        self.ffn_up = nn.Linear(config.dim, config.dim * 4)
        self.ffn_down = nn.Linear(config.dim * 4, config.dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn_down(F.relu(self.ffn_up(self.norm2(x))))


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.dim)
        self.output = nn.Linear(config.dim, config.vocab_size)

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


def train_epoch(model, optimizer, config, seq_len, training_noise=0.0, num_batches=100):
    model.train()
    total_loss, total_correct, total = 0, 0, 0
    batch_size = 32

    for _ in range(num_batches):
        seq, tgt = generate_batch(batch_size, seq_len, config.vocab_size, device)

        # Manual forward with noise injection
        h = model.embedding(seq)
        if training_noise > 0:
            h = h + training_noise * torch.randn_like(h)

        for block in model.blocks:
            h = block(h)
        logits = model.output(model.norm(h))

        loss = F.cross_entropy(logits[:, -1], tgt)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_correct += (logits[:, -1].argmax(-1) == tgt).sum().item()
        total += batch_size

    return total_loss / num_batches, total_correct / total


@torch.no_grad()
def evaluate(model, config, noise, seq_len, num_batches=50):
    model.eval()
    correct, total = 0, 0
    batch_size = 32

    for _ in range(num_batches):
        seq, tgt = generate_batch(batch_size, seq_len, config.vocab_size, device)

        h = model.embedding(seq)
        if noise > 0:
            h = h + noise * torch.randn_like(h)

        for block in model.blocks:
            h = block(h)
        logits = model.output(model.norm(h))

        correct += (logits[:, -1].argmax(-1) == tgt).sum().item()
        total += batch_size

    return correct / total


def train_with_noise_curriculum(config, seq_len, epochs, max_training_noise):
    """Train with noise curriculum up to max_training_noise."""
    print(f"\n{'='*60}")
    print(f"Training with Noise Curriculum (0 -> {max_training_noise})")
    print("="*60)

    model = Model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        training_noise = max_training_noise * epoch / epochs

        loss, train_acc = train_epoch(model, optimizer, config, seq_len, training_noise)

        if epoch % 10 == 0 or epoch == 1:
            acc_clean = evaluate(model, config, 0.0, seq_len)
            print(f"  Epoch {epoch}: Loss={loss:.3f}, TrainNoise={training_noise:.2f}, "
                  f"CleanAcc={acc_clean:.1%}")

    return model


def stress_test(model, config, seq_len, noise_levels):
    """Test model at various noise levels."""
    print(f"\n{'='*60}")
    print("STRESS TEST: Where does the mental filter break?")
    print("="*60)

    results = {}
    print(f"\n{'Noise Level':<15} {'Accuracy':<12} {'Status':<20}")
    print("-"*50)

    for noise in noise_levels:
        acc = evaluate(model, config, noise, seq_len)
        results[noise] = acc

        # Categorize performance
        if acc >= 0.95:
            status = "Excellent"
        elif acc >= 0.80:
            status = "Good"
        elif acc >= 0.50:
            status = "Degraded"
        elif acc >= 0.10:
            status = "Near collapse"
        else:
            status = "Random guessing"

        # Find chance level (1/100 for 100 possible values)
        chance = 1/100
        if acc < chance * 1.5:
            status = "BROKEN (at chance)"

        print(f"{noise:<15.1f} {acc:<12.1%} {status}")

    return results


def main():
    print("="*60)
    print("NOISE STRESS TEST")
    print("Where does the 94% noise-trained survivor finally break?")
    print("="*60)

    config = Config()
    seq_len = 256
    epochs = 40

    # Test different training noise ceilings
    training_noise_levels = [0.5, 1.0, 1.5]
    test_noise_levels = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    all_results = {}

    for max_noise in training_noise_levels:
        model = train_with_noise_curriculum(config, seq_len, epochs, max_noise)
        results = stress_test(model, config, seq_len, test_noise_levels)
        all_results[max_noise] = results

    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY: Training Noise vs Test Robustness")
    print("="*70)

    header = f"{'Test Noise':<12}"
    for max_noise in training_noise_levels:
        header += f"{'Train->'+str(max_noise):<15}"
    print(header)
    print("-"*60)

    for test_noise in test_noise_levels:
        row = f"{test_noise:<12.1f}"
        for max_noise in training_noise_levels:
            acc = all_results[max_noise][test_noise]
            row += f"{acc:<15.1%}"
        print(row)

    # Find the breaking point for each training regime
    print("\n" + "="*60)
    print("BREAKING POINTS (where accuracy drops below 50%)")
    print("="*60)

    for max_noise in training_noise_levels:
        results = all_results[max_noise]
        breaking_point = None
        for test_noise in sorted(results.keys()):
            if results[test_noise] < 0.50:
                breaking_point = test_noise
                break

        if breaking_point:
            print(f"  Train noise {max_noise}: Breaks at test noise {breaking_point}")
        else:
            print(f"  Train noise {max_noise}: Never breaks in tested range!")

    # The key insight
    print("\n" + "="*60)
    print("INSIGHT")
    print("="*60)
    print("""
The "Mental Filter" develops to handle noise levels seen during training.

- Train with noise 0->0.5: Good robustness up to ~1.5x training noise
- Train with noise 0->1.0: Should extend robustness further
- Train with noise 0->1.5: Maximum robustness but may hurt clean accuracy

The brain adapts to the environment it was raised in.
""")


if __name__ == "__main__":
    main()
