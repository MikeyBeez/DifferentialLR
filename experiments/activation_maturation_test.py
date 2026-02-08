#!/usr/bin/env python3
"""
Activation Maturation Test

Tests whether transitioning activation functions during training improves
robustness and final accuracy:

1. Exploration (ReLU): High plasticity, jagged manifold
2. Stabilization (GELU): Smooth gradients, attractor basins form
3. Hardening (Tanh): Saturating function locks weights in place

The transition is triggered by gradient norm stabilization.

Usage:
    python experiments/activation_maturation_test.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from enum import Enum

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")


class ActivationPhase(Enum):
    EXPLORATION = "relu"
    STABILIZATION = "gelu"
    HARDENING = "tanh"


@dataclass
class Config:
    vocab_size: int = 256
    dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.0
    beta: float = 2.0  # Hopfield attention

    # Maturation settings
    use_maturation: bool = True
    grad_ema_decay: float = 0.99  # EMA decay for gradient norm tracking
    stabilization_threshold: float = 0.1  # Gradient norm ratio to trigger GELU
    hardening_threshold: float = 0.05  # Gradient norm ratio to trigger Tanh


class ActivationCurriculum:
    """
    Tracks gradient norms and determines current activation phase.

    Transitions:
    - EXPLORATION (ReLU): Initial phase, high gradient variance
    - STABILIZATION (GELU): Gradients settling, switch when grad_norm < initial * stabilization_threshold
    - HARDENING (Tanh): Final phase, lock in weights
    """

    def __init__(self, config: Config):
        self.config = config
        self.grad_ema = None
        self.initial_grad_norm = None
        self.current_phase = ActivationPhase.EXPLORATION
        self.phase_history = []

    def update(self, grad_norm):
        """Update gradient tracking and potentially transition phase."""
        if self.initial_grad_norm is None:
            self.initial_grad_norm = grad_norm
            self.grad_ema = grad_norm
        else:
            self.grad_ema = (self.config.grad_ema_decay * self.grad_ema +
                           (1 - self.config.grad_ema_decay) * grad_norm)

        # Check for phase transitions
        if self.initial_grad_norm > 0:
            ratio = self.grad_ema / self.initial_grad_norm

            if self.current_phase == ActivationPhase.EXPLORATION:
                if ratio < self.config.stabilization_threshold:
                    self.current_phase = ActivationPhase.STABILIZATION
                    print(f"  → Phase transition: EXPLORATION → STABILIZATION (ratio={ratio:.4f})")

            elif self.current_phase == ActivationPhase.STABILIZATION:
                if ratio < self.config.hardening_threshold:
                    self.current_phase = ActivationPhase.HARDENING
                    print(f"  → Phase transition: STABILIZATION → HARDENING (ratio={ratio:.4f})")

        self.phase_history.append(self.current_phase)
        return self.current_phase

    def get_activation(self):
        """Return the activation function for current phase."""
        if self.current_phase == ActivationPhase.EXPLORATION:
            return F.relu
        elif self.current_phase == ActivationPhase.STABILIZATION:
            return F.gelu
        else:  # HARDENING
            return torch.tanh


class DynamicActivation(nn.Module):
    """
    Activation that changes based on training phase.
    """

    def __init__(self, curriculum: ActivationCurriculum):
        super().__init__()
        self.curriculum = curriculum

    def forward(self, x):
        activation_fn = self.curriculum.get_activation()
        return activation_fn(x)


class HopfieldAttention(nn.Module):
    """Hopfield attention with β scaling."""

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


class MaturationBlock(nn.Module):
    """Transformer block with dynamic activation."""

    def __init__(self, config: Config, curriculum: ActivationCurriculum = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.dim)
        self.attn = HopfieldAttention(config)
        self.norm2 = nn.LayerNorm(config.dim)

        self.ffn_up = nn.Linear(config.dim, config.dim * 4)
        self.ffn_down = nn.Linear(config.dim * 4, config.dim)

        if config.use_maturation and curriculum is not None:
            self.activation = DynamicActivation(curriculum)
        else:
            self.activation = nn.GELU()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        h = self.ffn_up(self.norm2(x))
        h = self.activation(h)
        x = x + self.ffn_down(h)
        return x


class MaturationModel(nn.Module):
    def __init__(self, config: Config, curriculum: ActivationCurriculum = None):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([
            MaturationBlock(config, curriculum) for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(config.dim)
        self.output = nn.Linear(config.dim, config.vocab_size)

    def forward(self, x):
        h = self.embedding(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        return self.output(h)


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


def compute_gradient_norm(model):
    """Compute total gradient norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def train_epoch(model, optimizer, seq_len, config, curriculum=None,
                num_batches=100, batch_size=32, device='cuda'):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    total_grad_norm = 0

    for _ in range(num_batches):
        seq, targets = generate_recall_batch(batch_size, seq_len, config.vocab_size, device)

        optimizer.zero_grad()
        logits = model(seq)
        loss = F.cross_entropy(logits[:, -1], targets)

        loss.backward()

        # Track gradient norm
        grad_norm = compute_gradient_norm(model)
        total_grad_norm += grad_norm

        # Update curriculum if using maturation
        if curriculum is not None:
            curriculum.update(grad_norm)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = logits[:, -1].argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size

    return {
        'loss': total_loss / num_batches,
        'accuracy': total_correct / total_samples,
        'grad_norm': total_grad_norm / num_batches
    }


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
def evaluate_robustness(model, seq_len, config, noise_levels=[0.0, 0.1, 0.2, 0.5],
                        num_batches=20, batch_size=32, device='cuda'):
    """
    Test robustness to input noise.

    Adds noise to embeddings and measures accuracy degradation.
    Hardened models should be more robust.
    """
    model.eval()
    results = {}

    for noise in noise_levels:
        total_correct = 0
        total_samples = 0

        for _ in range(num_batches):
            seq, targets = generate_recall_batch(batch_size, seq_len, config.vocab_size, device)

            # Get embeddings and add noise
            with torch.no_grad():
                h = model.embedding(seq)
                if noise > 0:
                    h = h + noise * torch.randn_like(h)

                # Forward through rest of model
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


def train_model(name, config, seq_len, device, epochs=30, use_maturation=True):
    """Train a model with or without activation maturation."""
    print(f"\n{'='*60}")
    print(f"{name} (distance={seq_len-2})")
    print(f"{'='*60}")

    config = Config(use_maturation=use_maturation)

    if use_maturation:
        curriculum = ActivationCurriculum(config)
    else:
        curriculum = None

    model = MaturationModel(config, curriculum).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print(f"Parameters: {count_params(model):,}")
    print(f"Maturation: {use_maturation}")

    best_acc = 0
    solved_epoch = None

    for epoch in range(1, epochs + 1):
        metrics = train_epoch(model, optimizer, seq_len, config, curriculum, device=device)
        val_acc = evaluate(model, seq_len, config, device=device)

        if val_acc > best_acc:
            best_acc = val_acc

        phase = curriculum.current_phase.value if curriculum else "gelu"

        if epoch % 5 == 0 or val_acc >= 0.99:
            print(f"  Epoch {epoch}: Loss {metrics['loss']:.3f}, Val {val_acc:.1%}, "
                  f"GradNorm {metrics['grad_norm']:.4f}, Phase: {phase}")

        if val_acc >= 0.99 and solved_epoch is None:
            solved_epoch = epoch
            print(f"  SOLVED at epoch {epoch}!")

    # Test robustness
    print(f"\nRobustness test (noise → accuracy):")
    robustness = evaluate_robustness(model, seq_len, config, device=device)
    for noise, acc in robustness.items():
        print(f"  Noise {noise}: {acc:.1%}")

    return {
        'accuracy': best_acc,
        'solved_epoch': solved_epoch or epochs,
        'robustness': robustness,
        'phase_history': curriculum.phase_history if curriculum else None
    }


def main():
    print("="*60)
    print("ACTIVATION MATURATION TEST")
    print("Does ReLU→GELU→Tanh improve robustness?")
    print("="*60)

    seq_len = 256  # Distance 254

    # Train with and without maturation
    results = {}

    print("\n" + "="*60)
    print("FIXED ACTIVATION (GELU baseline)")
    print("="*60)
    results['fixed_gelu'] = train_model(
        "Fixed GELU", Config(), seq_len, device,
        epochs=30, use_maturation=False
    )

    print("\n" + "="*60)
    print("ACTIVATION MATURATION (ReLU→GELU→Tanh)")
    print("="*60)
    results['maturation'] = train_model(
        "Maturation", Config(), seq_len, device,
        epochs=30, use_maturation=True
    )

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print(f"\n{'Config':<20} {'Accuracy':<12} {'Solved':<10} {'Noise 0.1':<12} {'Noise 0.2':<12}")
    print("-"*70)

    for name, r in results.items():
        rob = r['robustness']
        print(f"{name:<20} {r['accuracy']:<12.1%} {r['solved_epoch']:<10} "
              f"{rob[0.1]:<12.1%} {rob[0.2]:<12.1%}")

    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)

    fixed_rob = results['fixed_gelu']['robustness']
    mat_rob = results['maturation']['robustness']

    print(f"\nRobustness comparison (accuracy at noise=0.2):")
    print(f"  Fixed GELU:  {fixed_rob[0.2]:.1%}")
    print(f"  Maturation:  {mat_rob[0.2]:.1%}")

    if mat_rob[0.2] > fixed_rob[0.2]:
        improvement = (mat_rob[0.2] - fixed_rob[0.2]) * 100
        print(f"\n✓ Maturation improves robustness by {improvement:.1f} percentage points")
    else:
        print(f"\n✗ Maturation does not improve robustness")

    # Phase transition analysis
    if results['maturation']['phase_history']:
        history = results['maturation']['phase_history']
        phases = [p.value for p in history]
        print(f"\nPhase transitions:")
        print(f"  Total steps: {len(phases)}")
        print(f"  ReLU steps: {phases.count('relu')}")
        print(f"  GELU steps: {phases.count('gelu')}")
        print(f"  Tanh steps: {phases.count('tanh')}")


if __name__ == "__main__":
    main()
