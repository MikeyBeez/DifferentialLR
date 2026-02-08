#!/usr/bin/env python3
"""
Conviction Pump Test

Implements a "Conviction Pump" mechanism to help ReLU neurons commit
to signals rather than hedging. Based on autopsy findings:
- Neurons aren't dead, they're cautious
- Mean activation drops from 0.724 to 0.696 on failures
- We want to push neurons to "shout or shut up"

Mechanisms:
1. Conviction Loss: Penalize activations in the hedging zone (0 < x < τ)
2. Noisy Training: Train with noise injection so model learns to handle it
3. Weight Pulsing: Periodic LR spikes to break out of cautious equilibria
4. Contrast Scaling: Amplify weights that show decisive patterns

Goal: Move Noise 1.0 robustness from 68% to 80%

Usage:
    python experiments/conviction_pump_test.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')


@dataclass
class Config:
    vocab_size: int = 256
    dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    beta: float = 2.0  # Hopfield attention

    # Conviction Pump settings
    conviction_threshold: float = 0.3  # Activations below this are "hedging"
    conviction_weight: float = 0.1  # Weight for conviction loss
    noise_curriculum: bool = True  # Gradually increase training noise
    pulse_interval: int = 10  # Epochs between LR pulses
    pulse_magnitude: float = 3.0  # LR multiplier during pulse


class ConvictionLoss(nn.Module):
    """
    Penalizes activations in the "hedging zone" (0 < x < threshold).
    Pushes neurons to either shut up (0) or shout (>threshold).

    The loss is higher for activations closer to 0 (more hedging).
    """

    def __init__(self, threshold=0.3, mode='linear'):
        super().__init__()
        self.threshold = threshold
        self.mode = mode

    def forward(self, activations):
        # Flatten all activations
        flat = activations.view(-1)

        # Find hedging activations: 0 < x < threshold
        hedging_mask = (flat > 0) & (flat < self.threshold)
        hedging = flat[hedging_mask]

        if len(hedging) == 0:
            return torch.tensor(0.0, device=activations.device)

        if self.mode == 'linear':
            # Linear penalty: closer to 0 = higher penalty
            penalty = (self.threshold - hedging).mean()
        elif self.mode == 'quadratic':
            # Quadratic: stronger penalty for very low activations
            penalty = ((self.threshold - hedging) ** 2).mean()
        elif self.mode == 'log':
            # Log: infinite penalty as activation approaches 0
            penalty = -torch.log(hedging / self.threshold + 1e-8).mean()

        return penalty


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


class ConvictionBlock(nn.Module):
    """Block that tracks activations for conviction loss."""

    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.dim)
        self.attn = HopfieldAttention(config)
        self.norm2 = nn.LayerNorm(config.dim)
        self.ffn_up = nn.Linear(config.dim, config.dim * 4)
        self.ffn_down = nn.Linear(config.dim * 4, config.dim)

        self.last_activations = None

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        h = self.ffn_up(self.norm2(x))
        h = F.relu(h)
        self.last_activations = h  # Store for conviction loss
        return x + self.ffn_down(h)


class ConvictionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([ConvictionBlock(config) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.dim)
        self.output = nn.Linear(config.dim, config.vocab_size)

        self.conviction_loss_fn = ConvictionLoss(
            threshold=config.conviction_threshold,
            mode='linear'
        )

    def forward(self, x, return_conviction_loss=False):
        h = self.embedding(x)
        for block in self.blocks:
            h = block(h)
        logits = self.output(self.norm(h))

        if return_conviction_loss:
            # Compute conviction loss across all layers
            total_conviction = 0
            for block in self.blocks:
                if block.last_activations is not None:
                    total_conviction += self.conviction_loss_fn(block.last_activations)
            return logits, total_conviction / len(self.blocks)

        return logits

    def get_activation_stats(self):
        """Get statistics about current activations."""
        stats = []
        for i, block in enumerate(self.blocks):
            if block.last_activations is not None:
                acts = block.last_activations.detach()
                hedging = ((acts > 0) & (acts < self.config.conviction_threshold)).float().mean()
                mean_act = acts[acts > 0].mean() if (acts > 0).any() else 0
                stats.append({
                    'layer': i,
                    'hedging_frac': hedging.item(),
                    'mean_activation': mean_act.item() if isinstance(mean_act, torch.Tensor) else mean_act
                })
        return stats


class PulseLRScheduler:
    """
    Learning rate scheduler with periodic "pulses" to break plateaus.

    Normal LR with periodic spikes that help escape local minima.
    """

    def __init__(self, optimizer, base_lr, pulse_interval, pulse_magnitude, pulse_duration=1):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.pulse_interval = pulse_interval
        self.pulse_magnitude = pulse_magnitude
        self.pulse_duration = pulse_duration
        self.epoch = 0

    def step(self):
        self.epoch += 1

        # Check if we're in a pulse
        in_pulse = (self.epoch % self.pulse_interval) < self.pulse_duration

        if in_pulse:
            lr = self.base_lr * self.pulse_magnitude
        else:
            lr = self.base_lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr, in_pulse


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


def train_epoch(model, optimizer, config, seq_len=256, training_noise=0.0,
                use_conviction_loss=True, num_batches=100, batch_size=32):
    model.train()
    total_task_loss = 0
    total_conviction_loss = 0
    total_correct = 0
    total_samples = 0

    for _ in range(num_batches):
        seq, tgt = generate_batch(batch_size, seq_len, config.vocab_size, device)

        # Inject noise during training (noise curriculum)
        h = model.embedding(seq)
        if training_noise > 0:
            h = h + training_noise * torch.randn_like(h)

        # Forward through blocks manually to use noisy embeddings
        for block in model.blocks:
            h = block(h)
        logits = model.output(model.norm(h))

        # Task loss
        task_loss = F.cross_entropy(logits[:, -1], tgt)

        # Conviction loss
        if use_conviction_loss:
            conviction_loss = 0
            for block in model.blocks:
                if block.last_activations is not None:
                    conviction_loss += model.conviction_loss_fn(block.last_activations)
            conviction_loss = conviction_loss / len(model.blocks)
            loss = task_loss + config.conviction_weight * conviction_loss
            total_conviction_loss += conviction_loss.item()
        else:
            loss = task_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_task_loss += task_loss.item()
        total_correct += (logits[:, -1].argmax(-1) == tgt).sum().item()
        total_samples += batch_size

    return {
        'task_loss': total_task_loss / num_batches,
        'conviction_loss': total_conviction_loss / num_batches if use_conviction_loss else 0,
        'accuracy': total_correct / total_samples
    }


@torch.no_grad()
def evaluate(model, config, noise=0.0, seq_len=256, num_batches=50, batch_size=32):
    model.eval()
    total_correct = 0
    total_samples = 0

    for _ in range(num_batches):
        seq, tgt = generate_batch(batch_size, seq_len, config.vocab_size, device)

        h = model.embedding(seq)
        if noise > 0:
            h = h + noise * torch.randn_like(h)

        for block in model.blocks:
            h = block(h)
        logits = model.output(model.norm(h))

        total_correct += (logits[:, -1].argmax(-1) == tgt).sum().item()
        total_samples += batch_size

    return total_correct / total_samples


@torch.no_grad()
def get_activation_profile(model, config, noise=0.0, seq_len=256, num_batches=20):
    """Get detailed activation statistics."""
    model.eval()

    all_activations = []

    for _ in range(num_batches):
        seq, _ = generate_batch(32, seq_len, config.vocab_size, device)

        h = model.embedding(seq)
        if noise > 0:
            h = h + noise * torch.randn_like(h)

        for block in model.blocks:
            h = block(h)

        # Collect final layer activations
        final_acts = model.blocks[-1].last_activations[:, -1, :]
        all_activations.append(final_acts)

    acts = torch.cat(all_activations, dim=0)

    return {
        'mean': acts[acts > 0].mean().item(),
        'sparsity': (acts == 0).float().mean().item(),
        'hedging_frac': ((acts > 0) & (acts < config.conviction_threshold)).float().mean().item(),
        'confident_frac': (acts >= config.conviction_threshold).float().mean().item()
    }


def train_with_conviction_pump(config, seq_len=256, epochs=30, use_conviction=True,
                                use_noise_curriculum=True, use_pulse=True):
    """Train with all Conviction Pump mechanisms."""

    name = []
    if use_conviction:
        name.append("Conviction")
    if use_noise_curriculum:
        name.append("NoiseCurr")
    if use_pulse:
        name.append("Pulse")
    name = "+".join(name) if name else "Baseline"

    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print("="*60)

    model = ConvictionModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    if use_pulse:
        scheduler = PulseLRScheduler(
            optimizer,
            base_lr=1e-3,
            pulse_interval=config.pulse_interval,
            pulse_magnitude=config.pulse_magnitude
        )

    best_noise_1 = 0

    for epoch in range(1, epochs + 1):
        # Noise curriculum: start clean, gradually add noise
        if use_noise_curriculum:
            training_noise = min(0.5, epoch / epochs)  # 0 → 0.5 over training
        else:
            training_noise = 0.0

        # Pulse LR
        if use_pulse:
            lr, in_pulse = scheduler.step()
            pulse_marker = " [PULSE]" if in_pulse else ""
        else:
            pulse_marker = ""

        metrics = train_epoch(
            model, optimizer, config, seq_len,
            training_noise=training_noise,
            use_conviction_loss=use_conviction
        )

        # Evaluate
        acc_clean = evaluate(model, config, noise=0.0, seq_len=seq_len)
        acc_05 = evaluate(model, config, noise=0.5, seq_len=seq_len)
        acc_10 = evaluate(model, config, noise=1.0, seq_len=seq_len)

        if acc_10 > best_noise_1:
            best_noise_1 = acc_10

        if epoch % 5 == 0 or epoch == 1:
            conv_str = f", Conv={metrics['conviction_loss']:.3f}" if use_conviction else ""
            print(f"  Epoch {epoch}: Loss={metrics['task_loss']:.3f}{conv_str}, "
                  f"Clean={acc_clean:.1%}, N0.5={acc_05:.1%}, N1.0={acc_10:.1%}{pulse_marker}")

    # Final profile
    profile = get_activation_profile(model, config, noise=1.0, seq_len=seq_len)

    print(f"\nFinal activation profile at noise=1.0:")
    print(f"  Mean activation: {profile['mean']:.3f}")
    print(f"  Sparsity: {profile['sparsity']:.1%}")
    print(f"  Hedging (<{config.conviction_threshold}): {profile['hedging_frac']:.1%}")
    print(f"  Confident (≥{config.conviction_threshold}): {profile['confident_frac']:.1%}")

    return {
        'name': name,
        'best_noise_1': best_noise_1,
        'final_noise_1': acc_10,
        'profile': profile
    }


def main():
    print("="*60)
    print("CONVICTION PUMP TEST")
    print("Can we push Noise 1.0 robustness from 68% to 80%?")
    print("="*60)

    config = Config()
    seq_len = 256
    epochs = 40

    results = {}

    # Baseline: Pure ReLU, no tricks
    results['baseline'] = train_with_conviction_pump(
        config, seq_len, epochs,
        use_conviction=False, use_noise_curriculum=False, use_pulse=False
    )

    # Just conviction loss
    results['conviction'] = train_with_conviction_pump(
        config, seq_len, epochs,
        use_conviction=True, use_noise_curriculum=False, use_pulse=False
    )

    # Just noise curriculum
    results['noise_curr'] = train_with_conviction_pump(
        config, seq_len, epochs,
        use_conviction=False, use_noise_curriculum=True, use_pulse=False
    )

    # Full Conviction Pump
    results['full_pump'] = train_with_conviction_pump(
        config, seq_len, epochs,
        use_conviction=True, use_noise_curriculum=True, use_pulse=True
    )

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\n{'Config':<30} {'Best N1.0':<12} {'Final N1.0':<12} {'Hedging%':<12}")
    print("-"*70)

    for key, r in results.items():
        print(f"{r['name']:<30} {r['best_noise_1']:<12.1%} {r['final_noise_1']:<12.1%} "
              f"{r['profile']['hedging_frac']*100:<12.1f}")

    # Did we hit 80%?
    best_result = max(results.values(), key=lambda x: x['best_noise_1'])

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    print(f"\nBest result: {best_result['name']} with {best_result['best_noise_1']:.1%} at Noise 1.0")

    if best_result['best_noise_1'] >= 0.80:
        print("\n✓ SUCCESS! Hit 80% target!")
    elif best_result['best_noise_1'] >= 0.75:
        print("\n~ PROGRESS! Got above 75%, approaching target.")
    else:
        print("\n✗ Did not reach 80% target. Need different approach.")

    # Analyze what worked
    baseline_score = results['baseline']['best_noise_1']
    for key, r in results.items():
        if key != 'baseline':
            improvement = (r['best_noise_1'] - baseline_score) * 100
            if improvement > 0:
                print(f"  {r['name']}: +{improvement:.1f}pp over baseline")


if __name__ == "__main__":
    main()
