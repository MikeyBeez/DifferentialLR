#!/usr/bin/env python3
"""
Rectified Bottleneck Test

Implements a learned activation function inspired by Rectified JEPA (Assran et al., 2024).
Goal: Close the ~3% recall gap by forcing sparse, decisive representations.

Key ideas:
1. Learnable threshold (soft shrinkage operator)
2. L1 penalty to tax non-zero activations (sparsity pressure)
3. Log-determinant term to prevent collapse (anti-lazy insurance)
4. Curriculum learning: task first, then sparsity pressure

Usage:
    python experiments/rectified_bottleneck_test.py
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

    # Rectified bottleneck settings
    bottleneck_dim: int = 64  # Compressed representation
    initial_threshold: float = 0.1  # Starting threshold for shrinkage

    # Sparsity settings
    sparsity_weight: float = 0.01  # λ₁ for L1 penalty
    diversity_weight: float = 0.1  # λ₂ for log-det anti-collapse


class LearnedShrinkage(nn.Module):
    """
    Soft shrinkage operator with learnable threshold.

    shrink(x, τ) = sign(x) * max(|x| - τ, 0)

    This is differentiable and learns which activations to keep.
    """

    def __init__(self, dim, initial_threshold=0.1):
        super().__init__()
        # Per-dimension learnable threshold
        self.threshold = nn.Parameter(torch.full((dim,), initial_threshold))

    def forward(self, x):
        # Soft shrinkage: push small values to exactly zero
        # threshold must be positive
        tau = F.softplus(self.threshold)

        # shrink(x) = sign(x) * max(|x| - τ, 0)
        magnitude = x.abs()
        shrunk_magnitude = F.relu(magnitude - tau)
        return x.sign() * shrunk_magnitude


class RectifiedBottleneck(nn.Module):
    """
    Bottleneck layer with learned sparsity.

    Compresses to lower dimension, applies learned shrinkage,
    then expands back. Forces the model to find sparse, salient features.
    """

    def __init__(self, input_dim, bottleneck_dim, initial_threshold=0.1):
        super().__init__()

        # Compress
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)

        # Learned shrinkage (the key innovation)
        self.shrinkage = LearnedShrinkage(bottleneck_dim, initial_threshold)

        # Expand back
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)

        # Layer norm for stability
        self.norm = nn.LayerNorm(bottleneck_dim)

    def forward(self, x):
        # Compress to bottleneck
        h = self.down_proj(x)
        h = self.norm(h)

        # Apply learned shrinkage (this creates sparsity)
        h_sparse = self.shrinkage(h)

        # Expand back to original dimension
        out = self.up_proj(h_sparse)

        return out, h_sparse  # Return sparse activations for loss computation


class CausalAttention(nn.Module):
    """Standard causal attention."""

    def __init__(self, config: Config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.dim // config.num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(config.dim, config.dim * 3)
        self.out = nn.Linear(config.dim, config.dim)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class RectifiedBlock(nn.Module):
    """
    Transformer block with rectified bottleneck in the FFN.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.dim)
        self.attn = CausalAttention(config)
        self.norm2 = nn.LayerNorm(config.dim)

        # Replace standard FFN with rectified bottleneck
        self.ffn_up = nn.Linear(config.dim, config.dim * 4)
        self.bottleneck = RectifiedBottleneck(
            config.dim * 4,
            config.bottleneck_dim,
            config.initial_threshold
        )
        self.ffn_down = nn.Linear(config.dim * 4, config.dim)

    def forward(self, x):
        # Attention
        x = x + self.attn(self.norm1(x))

        # FFN with rectified bottleneck
        h = self.norm2(x)
        h = F.gelu(self.ffn_up(h))
        h, sparse_activations = self.bottleneck(h)
        h = self.ffn_down(h)
        x = x + h

        return x, sparse_activations


class RectifiedModel(nn.Module):
    """
    Transformer with rectified bottleneck for sparse representations.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([RectifiedBlock(config) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.dim)
        self.output = nn.Linear(config.dim, config.vocab_size)

    def forward(self, x):
        h = self.embedding(x)

        all_sparse_activations = []
        for block in self.blocks:
            h, sparse_act = block(h)
            all_sparse_activations.append(sparse_act)

        h = self.norm(h)
        logits = self.output(h)

        # Stack all sparse activations for loss computation
        sparse_activations = torch.stack(all_sparse_activations, dim=0)  # [layers, B, T, bottleneck_dim]

        return logits, sparse_activations


def compute_sparsity_loss(activations):
    """
    L1 penalty on activations to encourage sparsity.
    """
    return activations.abs().mean()


def compute_diversity_loss(activations, eps=0.01):
    """
    Log-determinant of covariance to prevent collapse.

    If all neurons die, covariance becomes singular and log-det → -inf.
    This term encourages diverse, non-redundant representations.
    """
    # Reshape to [batch * seq, features]
    B, T, D = activations.shape[-3:]
    flat = activations.reshape(-1, D)  # [N, D]

    # Compute covariance
    mean = flat.mean(dim=0, keepdim=True)
    centered = flat - mean
    cov = (centered.T @ centered) / (flat.shape[0] - 1)

    # Add regularization for numerical stability
    cov = cov + eps * torch.eye(D, device=cov.device)

    # Log-determinant (higher = more diverse)
    try:
        log_det = torch.logdet(cov)
        if torch.isnan(log_det) or torch.isinf(log_det):
            log_det = torch.tensor(0.0, device=cov.device)
    except:
        log_det = torch.tensor(0.0, device=cov.device)

    return log_det


def compute_sparsity_stats(activations):
    """Compute sparsity statistics for monitoring."""
    # What fraction of activations are exactly zero?
    zero_frac = (activations == 0).float().mean().item()

    # What fraction are "nearly zero" (< 0.01)?
    near_zero_frac = (activations.abs() < 0.01).float().mean().item()

    # Average magnitude of non-zero activations
    nonzero_mask = activations != 0
    if nonzero_mask.any():
        avg_magnitude = activations[nonzero_mask].abs().mean().item()
    else:
        avg_magnitude = 0.0

    return zero_frac, near_zero_frac, avg_magnitude


def generate_recall_batch(batch_size, seq_len, vocab_size, device='cuda'):
    """Generate associative recall task."""
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
    """Train for one epoch with sparsity regularization."""
    model.train()
    total_loss = 0
    total_task_loss = 0
    total_sparsity_loss = 0
    total_diversity_loss = 0
    total_correct = 0
    total_samples = 0
    total_zero_frac = 0

    for _ in range(num_batches):
        seq, targets = generate_recall_batch(batch_size, seq_len, config.vocab_size, device)

        optimizer.zero_grad()
        logits, sparse_activations = model(seq)

        # Task loss
        task_loss = F.cross_entropy(logits[:, -1], targets)

        # Sparsity loss (L1)
        sparsity_loss = compute_sparsity_loss(sparse_activations)

        # Diversity loss (log-det, we want to maximize this so subtract)
        diversity_loss = compute_diversity_loss(sparse_activations)

        # Combined loss
        loss = task_loss + sparsity_weight * sparsity_loss - diversity_weight * diversity_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        total_task_loss += task_loss.item()
        total_sparsity_loss += sparsity_loss.item()
        total_diversity_loss += diversity_loss.item()

        preds = logits[:, -1].argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size

        zero_frac, _, _ = compute_sparsity_stats(sparse_activations)
        total_zero_frac += zero_frac

    return {
        'loss': total_loss / num_batches,
        'task_loss': total_task_loss / num_batches,
        'sparsity_loss': total_sparsity_loss / num_batches,
        'diversity_loss': total_diversity_loss / num_batches,
        'accuracy': total_correct / total_samples,
        'zero_frac': total_zero_frac / num_batches
    }


@torch.no_grad()
def evaluate(model, seq_len, config, num_batches=50, batch_size=32, device='cuda'):
    """Evaluate model."""
    model.eval()
    total_correct = 0
    total_samples = 0
    total_zero_frac = 0

    for _ in range(num_batches):
        seq, targets = generate_recall_batch(batch_size, seq_len, config.vocab_size, device)
        logits, sparse_activations = model(seq)

        preds = logits[:, -1].argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size

        zero_frac, _, _ = compute_sparsity_stats(sparse_activations)
        total_zero_frac += zero_frac

    return total_correct / total_samples, total_zero_frac / num_batches


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_with_curriculum(config, seq_len, device,
                          phase1_epochs=15, phase2_epochs=15,
                          max_sparsity_weight=0.05, diversity_weight=0.1):
    """
    Two-phase curriculum:
    1. Learn the task (no sparsity pressure)
    2. Add sparsity pressure while maintaining accuracy
    """
    print(f"\n{'='*70}")
    print(f"Rectified Bottleneck (distance={seq_len-2})")
    print(f"{'='*70}")

    model = RectifiedModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print(f"Parameters: {count_params(model):,}")
    print(f"Bottleneck dim: {config.bottleneck_dim}")

    # Phase 1: Learn the task
    print(f"\nPhase 1: Learning task (no sparsity pressure)")
    best_acc = 0
    phase1_solved = False

    for epoch in range(1, phase1_epochs + 1):
        metrics = train_epoch(
            model, optimizer, seq_len, config,
            sparsity_weight=0.0, diversity_weight=diversity_weight,
            device=device
        )
        val_acc, val_zero_frac = evaluate(model, seq_len, config, device=device)

        if val_acc > best_acc:
            best_acc = val_acc

        if epoch % 5 == 0 or val_acc >= 0.99:
            print(f"  Epoch {epoch}: Loss {metrics['task_loss']:.3f}, "
                  f"Val {val_acc:.1%}, Zero {val_zero_frac:.1%}")

        if val_acc >= 0.99:
            phase1_solved = True
            print(f"  SOLVED at epoch {epoch}!")
            break

    if not phase1_solved:
        print(f"  Phase 1 did not solve (best: {best_acc:.1%})")

    # Phase 2: Add sparsity pressure
    print(f"\nPhase 2: Adding sparsity pressure (λ→{max_sparsity_weight})")
    phase2_best_acc = best_acc
    phase2_best_zero = val_zero_frac

    for epoch in range(1, phase2_epochs + 1):
        # Gradually increase sparsity pressure
        sparsity_weight = max_sparsity_weight * (epoch / phase2_epochs)

        metrics = train_epoch(
            model, optimizer, seq_len, config,
            sparsity_weight=sparsity_weight, diversity_weight=diversity_weight,
            device=device
        )
        val_acc, val_zero_frac = evaluate(model, seq_len, config, device=device)

        if val_acc >= 0.95:
            if val_acc > phase2_best_acc or val_zero_frac > phase2_best_zero:
                phase2_best_acc = val_acc
                phase2_best_zero = val_zero_frac

        if epoch % 5 == 0 or epoch == phase2_epochs:
            print(f"  Epoch {epoch}: λ={sparsity_weight:.3f}, "
                  f"Val {val_acc:.1%}, Zero {val_zero_frac:.1%}, "
                  f"Sparsity Loss {metrics['sparsity_loss']:.3f}")

    print(f"\nFinal: {phase2_best_acc:.1%} accuracy, {phase2_best_zero:.1%} zero activations")

    # Analyze learned thresholds
    print(f"\nLearned thresholds:")
    for i, block in enumerate(model.blocks):
        tau = F.softplus(block.bottleneck.shrinkage.threshold)
        print(f"  Block {i}: min={tau.min().item():.3f}, max={tau.max().item():.3f}, "
              f"mean={tau.mean().item():.3f}")

    return {
        'accuracy': phase2_best_acc,
        'zero_frac': phase2_best_zero,
        'model': model
    }


def main():
    print("="*70)
    print("RECTIFIED BOTTLENECK TEST")
    print("Can learned sparsity close the 3% recall gap?")
    print("="*70)

    config = Config()

    # Test at different distances
    seq_lengths = [128, 256, 512]

    results = {}

    for seq_len in seq_lengths:
        result = train_with_curriculum(
            config, seq_len, device,
            phase1_epochs=20, phase2_epochs=15,
            max_sparsity_weight=0.05, diversity_weight=0.1
        )
        results[seq_len] = result
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    print(f"\n{'Distance':<12} {'Accuracy':<12} {'Zero Frac':<12}")
    print("-"*40)
    for seq_len, r in results.items():
        print(f"{seq_len-2:<12} {r['accuracy']:<12.1%} {r['zero_frac']:<12.1%}")

    # Compare to baseline
    print("\n" + "="*70)
    print("COMPARISON TO BASELINE")
    print("="*70)
    print("""
Previous results (standard attention):
  Distance 126: 100% accuracy
  Distance 254: 100% accuracy
  Distance 510: 94-100% accuracy (depending on β)

Rectified bottleneck hypothesis:
  If sparsity helps, we should see:
  1. Equal or better accuracy
  2. Higher zero activation fraction (sparser representations)
  3. Learned thresholds that vary by dimension (selective sparsity)
""")


if __name__ == "__main__":
    main()
