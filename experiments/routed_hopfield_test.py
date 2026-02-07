#!/usr/bin/env python3
"""
Routed Hopfield Attention Test

Combines two findings:
1. Routed attention: Route between cheap conv and expensive attention
2. Hopfield β=2: Sharper attention improves recall

Question: Does β=2 help routed attention?
- Faster convergence?
- Less attention needed?
- Better accuracy at long range?

Usage:
    python3 experiments/routed_hopfield_test.py
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
    kernel_size: int = 64

    # Router settings
    attention_cost: float = 0.0  # Start with no penalty (phase 1)
    temperature: float = 1.0

    # Hopfield setting
    beta: float = 1.0  # Attention sharpness


class CausalConv(nn.Module):
    """O(N) causal convolution - cheap path."""

    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.kernel_size = config.kernel_size

        self.v_proj = nn.Linear(config.dim, config.dim)
        self.out = nn.Linear(config.dim, config.dim)

        self.kernel_logits = nn.Parameter(
            torch.zeros(config.num_heads, config.kernel_size)
        )
        self.head_dim = config.dim // config.num_heads
        self.num_heads = config.num_heads

    def forward(self, x):
        B, T, C = x.shape

        v = self.v_proj(x).transpose(1, 2)
        v_padded = F.pad(v, (self.kernel_size - 1, 0))

        kernel = F.softmax(self.kernel_logits, dim=-1)
        kernel_expanded = kernel.unsqueeze(1).expand(-1, self.head_dim, -1)
        kernel_expanded = kernel_expanded.reshape(self.dim, 1, self.kernel_size)

        out = F.conv1d(v_padded, kernel_expanded, groups=self.dim)
        out = out.transpose(1, 2)

        return self.out(out)


class HopfieldAttention(nn.Module):
    """O(N²) Hopfield attention with configurable beta."""

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

        # Hopfield attention with beta
        attn = (q @ k.transpose(-2, -1)) * self.scale * self.beta

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class Router(nn.Module):
    """Per-position router: conv or attention?"""

    def __init__(self, config: Config):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(config.dim, config.dim // 4),
            nn.GELU(),
            nn.Linear(config.dim // 4, 2)
        )
        self.temperature = config.temperature

    def forward(self, x):
        logits = self.router(x)
        weights = F.gumbel_softmax(logits, tau=self.temperature, hard=True)
        return weights, logits


class RoutedHopfieldLayer(nn.Module):
    """Routes between conv and Hopfield attention."""

    def __init__(self, config: Config):
        super().__init__()
        self.conv = CausalConv(config)
        self.attn = HopfieldAttention(config)
        self.router = Router(config)

    def forward(self, x):
        weights, logits = self.router(x)

        conv_out = self.conv(x)
        attn_out = self.attn(x)

        out = weights[..., 0:1] * conv_out + weights[..., 1:2] * attn_out

        attn_frac = weights[..., 1].mean().item()
        return out, attn_frac


class RoutedBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.mixer = RoutedHopfieldLayer(config)
        self.norm1 = nn.LayerNorm(config.dim)
        self.norm2 = nn.LayerNorm(config.dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.dim, config.dim * 4),
            nn.GELU(),
            nn.Linear(config.dim * 4, config.dim)
        )

    def forward(self, x):
        normed = self.norm1(x)
        mixed, attn_frac = self.mixer(normed)
        x = x + mixed
        x = x + self.ffn(self.norm2(x))
        return x, attn_frac


class RoutedHopfieldModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([RoutedBlock(config) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.dim)
        self.output = nn.Linear(config.dim, config.vocab_size)

    def forward(self, x):
        h = self.embedding(x)
        total_attn = 0
        for block in self.blocks:
            h, attn_frac = block(h)
            total_attn += attn_frac
        h = self.norm(h)
        avg_attn = total_attn / len(self.blocks)
        return self.output(h), avg_attn


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


def train_epoch(model, optimizer, seq_len, attention_cost=0.0,
                num_batches=100, batch_size=32, device='cuda'):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    total_attn = 0

    for _ in range(num_batches):
        seq, targets = generate_recall_batch(batch_size, seq_len, model.config.vocab_size, device)

        optimizer.zero_grad()
        logits, attn_frac = model(seq)

        task_loss = F.cross_entropy(logits[:, -1], targets)
        cost_loss = attention_cost * attn_frac
        loss = task_loss + cost_loss

        loss.backward()
        optimizer.step()

        total_loss += task_loss.item()
        total_attn += attn_frac
        preds = logits[:, -1].argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size

    return total_loss / num_batches, total_correct / total_samples, total_attn / num_batches


@torch.no_grad()
def evaluate(model, seq_len, num_batches=50, batch_size=32, device='cuda'):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_attn = 0

    for _ in range(num_batches):
        seq, targets = generate_recall_batch(batch_size, seq_len, model.config.vocab_size, device)
        logits, attn_frac = model(seq)
        preds = logits[:, -1].argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size
        total_attn += attn_frac

    return total_correct / total_samples, total_attn / num_batches


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_with_curriculum(name, beta, seq_len, device, phase1_epochs=20, phase2_epochs=15):
    """Two-phase curriculum: learn task, then minimize attention."""
    print(f"\n{'='*60}")
    print(f"{name} (β={beta}, distance={seq_len-2})")
    print(f"{'='*60}")

    config = Config(beta=beta)
    model = RoutedHopfieldModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print(f"Parameters: {count_params(model):,}")

    # Phase 1: Learn the task (no attention penalty)
    print("\nPhase 1: Learning task (λ=0)")
    best_acc = 0
    phase1_solved_epoch = None

    for epoch in range(1, phase1_epochs + 1):
        loss, train_acc, attn_usage = train_epoch(
            model, optimizer, seq_len, attention_cost=0.0, device=device
        )
        val_acc, val_attn = evaluate(model, seq_len, device=device)

        if val_acc > best_acc:
            best_acc = val_acc

        if epoch % 5 == 0 or val_acc >= 0.99:
            print(f"  Epoch {epoch}: Loss {loss:.3f}, Val {val_acc:.1%}, Attn {val_attn:.1%}")

        if val_acc >= 0.99 and phase1_solved_epoch is None:
            phase1_solved_epoch = epoch
            print(f"  SOLVED at epoch {epoch}!")
            break

    if phase1_solved_epoch is None:
        print(f"  Phase 1 did not solve (best: {best_acc:.1%})")
        return {
            'accuracy': best_acc,
            'attn_usage': val_attn,
            'phase1_epochs': phase1_epochs,
            'phase2_epochs': 0
        }

    # Phase 2: Minimize attention while maintaining accuracy
    print("\nPhase 2: Minimizing attention (λ→0.5)")
    phase2_best_acc = best_acc
    phase2_best_attn = val_attn

    for epoch in range(1, phase2_epochs + 1):
        # Gradually increase attention cost
        lambda_val = 0.5 * (epoch / phase2_epochs)

        loss, train_acc, attn_usage = train_epoch(
            model, optimizer, seq_len, attention_cost=lambda_val, device=device
        )
        val_acc, val_attn = evaluate(model, seq_len, device=device)

        if val_acc >= 0.95 and val_attn < phase2_best_attn:
            phase2_best_attn = val_attn
            phase2_best_acc = val_acc

        if epoch % 5 == 0 or epoch == phase2_epochs:
            print(f"  Epoch {epoch}: λ={lambda_val:.2f}, Val {val_acc:.1%}, Attn {val_attn:.1%}")

    print(f"\nFinal: {phase2_best_acc:.1%} accuracy, {phase2_best_attn:.1%} attention")

    return {
        'accuracy': phase2_best_acc,
        'attn_usage': phase2_best_attn,
        'phase1_epochs': phase1_solved_epoch,
        'phase2_epochs': phase2_epochs
    }


def main():
    print("=" * 60)
    print("ROUTED HOPFIELD ATTENTION TEST")
    print("Does β=2 help routed attention?")
    print("=" * 60)

    seq_lengths = [128, 256, 512]
    betas = [1.0, 2.0, 4.0]

    all_results = {}

    for beta in betas:
        all_results[f'β={beta}'] = {}
        for seq_len in seq_lengths:
            result = train_with_curriculum(
                f"Routed Hopfield",
                beta=beta,
                seq_len=seq_len,
                device=device
            )
            all_results[f'β={beta}'][seq_len] = result
            torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    print(f"\n{'Config':<15}", end="")
    for seq_len in seq_lengths:
        print(f"{'Dist '+str(seq_len-2):<20}", end="")
    print()
    print("-" * 80)

    for beta_name, results in all_results.items():
        print(f"{beta_name:<15}", end="")
        for seq_len in seq_lengths:
            r = results[seq_len]
            print(f"{r['accuracy']:.0%} ({r['attn_usage']:.0%} attn)     ", end="")
        print()

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    for seq_len in seq_lengths:
        print(f"\nDistance {seq_len - 2}:")
        for beta_name, results in all_results.items():
            r = results[seq_len]
            print(f"  {beta_name}: {r['accuracy']:.1%} acc, {r['attn_usage']:.1%} attention, "
                  f"solved in {r['phase1_epochs']} epochs")

    # Best configuration
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    beta1_attn = sum(all_results['β=1.0'][s]['attn_usage'] for s in seq_lengths) / len(seq_lengths)
    beta2_attn = sum(all_results['β=2.0'][s]['attn_usage'] for s in seq_lengths) / len(seq_lengths)

    print(f"\nAverage attention usage:")
    print(f"  β=1.0: {beta1_attn:.1%}")
    print(f"  β=2.0: {beta2_attn:.1%}")

    if beta2_attn < beta1_attn:
        print(f"\n=> β=2 uses LESS attention ({(beta1_attn - beta2_attn)*100:.1f}% less)")
        print("   Sharper attention = more efficient routing!")
    else:
        print(f"\n=> β=2 doesn't reduce attention usage")


if __name__ == "__main__":
    main()
