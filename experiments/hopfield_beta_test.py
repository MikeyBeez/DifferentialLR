#!/usr/bin/env python3
"""
Hopfield Attention with Learnable Beta

Modern Hopfield networks show that attention IS associative memory retrieval.
The sharpness parameter β controls retrieval precision:
- β = 1: Standard softmax (soft average)
- β > 1: Sharper retrieval (more precise lookup)
- β >> 1: Approaches argmax (exact match)

Hypothesis: Learning β might help the model find optimal sharpness for recall.

Usage:
    python3 experiments/hopfield_beta_test.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")


class HopfieldAttention(nn.Module):
    """
    Attention with configurable/learnable inverse temperature (beta).

    Standard attention: softmax(QK^T / sqrt(d))
    Hopfield attention: softmax(β * QK^T / sqrt(d))

    Higher β = sharper retrieval = better for exact recall
    """

    def __init__(self, dim, num_heads=4, beta_init=1.0, learn_beta=False, per_head_beta=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.learn_beta = learn_beta
        self.per_head_beta = per_head_beta

        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)

        # Beta parameter
        if per_head_beta:
            # Each head learns its own beta
            beta_shape = (num_heads, 1, 1)
        else:
            # Single beta for all heads
            beta_shape = (1,)

        if learn_beta:
            # Learnable beta (in log space for stability)
            self.log_beta = nn.Parameter(torch.full(beta_shape, math.log(beta_init)))
        else:
            # Fixed beta
            self.register_buffer('log_beta', torch.full(beta_shape, math.log(beta_init)))

    @property
    def beta(self):
        return torch.exp(self.log_beta)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        # Hopfield attention with beta
        attn = (q @ k.transpose(-2, -1)) * self.scale * self.beta

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, beta_init=1.0, learn_beta=False, per_head_beta=False):
        super().__init__()
        self.attn = HopfieldAttention(dim, num_heads, beta_init, learn_beta, per_head_beta)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class HopfieldModel(nn.Module):
    def __init__(self, vocab_size, dim, num_layers=4, num_heads=4,
                 beta_init=1.0, learn_beta=False, per_head_beta=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.beta_init = beta_init
        self.learn_beta = learn_beta

        self.embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, beta_init, learn_beta, per_head_beta)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.embedding(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        return self.output(h)

    def get_betas(self):
        """Get current beta values for all layers."""
        return [block.attn.beta.detach().cpu().numpy().flatten() for block in self.blocks]


def generate_recall_batch(batch_size, seq_len, vocab_size, device='cuda'):
    """Associative recall: key at pos 0, value at pos 1, query at end."""
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
        seq[-1] = key  # Query

        sequences.append(seq)
        targets.append(val)

    return torch.stack(sequences), torch.tensor(targets, device=device).squeeze()


def train_epoch(model, optimizer, seq_len, num_batches=100, batch_size=32, device='cuda'):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for _ in range(num_batches):
        seq, targets = generate_recall_batch(batch_size, seq_len, model.vocab_size, device)

        optimizer.zero_grad()
        logits = model(seq)

        loss = F.cross_entropy(logits[:, -1], targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits[:, -1].argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size

    return total_loss / num_batches, total_correct / total_samples


@torch.no_grad()
def evaluate(model, seq_len, num_batches=50, batch_size=32, device='cuda'):
    model.eval()
    total_correct = 0
    total_samples = 0

    for _ in range(num_batches):
        seq, targets = generate_recall_batch(batch_size, seq_len, model.vocab_size, device)
        logits = model(seq)
        preds = logits[:, -1].argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size

    return total_correct / total_samples


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_config(name, beta_init, learn_beta, per_head_beta, seq_lengths, device, epochs=30):
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    print(f"  beta_init={beta_init}, learn_beta={learn_beta}, per_head_beta={per_head_beta}")

    results = {}

    for seq_len in seq_lengths:
        print(f"\n--- Seq Length: {seq_len} (distance: {seq_len - 2}) ---")

        model = HopfieldModel(
            vocab_size=256,
            dim=128,
            num_layers=4,
            num_heads=4,
            beta_init=beta_init,
            learn_beta=learn_beta,
            per_head_beta=per_head_beta
        ).to(device)

        if seq_len == seq_lengths[0]:
            print(f"Parameters: {count_params(model):,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        best_acc = 0
        for epoch in range(1, epochs + 1):
            loss, train_acc = train_epoch(model, optimizer, seq_len, device=device)
            val_acc = evaluate(model, seq_len, device=device)

            if val_acc > best_acc:
                best_acc = val_acc

            if epoch % 5 == 0 or val_acc >= 0.99:
                betas = model.get_betas()
                beta_str = ", ".join([f"L{i}:{b[0]:.2f}" for i, b in enumerate(betas)])
                print(f"Epoch {epoch}: Loss {loss:.3f}, Train {train_acc:.1%}, Val {val_acc:.1%} | β: [{beta_str}]")

            if val_acc >= 0.99:
                print("SOLVED!")
                break

        # Final betas
        betas = model.get_betas()
        results[seq_len] = {
            'accuracy': best_acc,
            'final_betas': betas
        }

        del model, optimizer
        torch.cuda.empty_cache()

    return results


def main():
    print("=" * 70)
    print("HOPFIELD ATTENTION: LEARNABLE BETA TEST")
    print("Can we learn optimal sharpness for associative recall?")
    print("=" * 70)

    seq_lengths = [64, 128, 256, 512]

    all_results = {}

    # Fixed beta = 1 (standard attention)
    all_results['β=1 (standard)'] = test_config(
        "Standard Attention (β=1, fixed)",
        beta_init=1.0, learn_beta=False, per_head_beta=False,
        seq_lengths=seq_lengths, device=device
    )

    # Fixed beta = 2 (sharper)
    all_results['β=2 (fixed)'] = test_config(
        "Sharper Attention (β=2, fixed)",
        beta_init=2.0, learn_beta=False, per_head_beta=False,
        seq_lengths=seq_lengths, device=device
    )

    # Fixed beta = 4 (very sharp)
    all_results['β=4 (fixed)'] = test_config(
        "Very Sharp Attention (β=4, fixed)",
        beta_init=4.0, learn_beta=False, per_head_beta=False,
        seq_lengths=seq_lengths, device=device
    )

    # Fixed beta = 8 (extremely sharp)
    all_results['β=8 (fixed)'] = test_config(
        "Extremely Sharp Attention (β=8, fixed)",
        beta_init=8.0, learn_beta=False, per_head_beta=False,
        seq_lengths=seq_lengths, device=device
    )

    # Learnable beta (starting at 1)
    all_results['β=1→learned'] = test_config(
        "Learnable Beta (init=1)",
        beta_init=1.0, learn_beta=True, per_head_beta=False,
        seq_lengths=seq_lengths, device=device
    )

    # Learnable beta (starting at 2)
    all_results['β=2→learned'] = test_config(
        "Learnable Beta (init=2)",
        beta_init=2.0, learn_beta=True, per_head_beta=False,
        seq_lengths=seq_lengths, device=device
    )

    # Per-head learnable beta
    all_results['per-head β'] = test_config(
        "Per-Head Learnable Beta (init=1)",
        beta_init=1.0, learn_beta=True, per_head_beta=True,
        seq_lengths=seq_lengths, device=device
    )

    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    print(f"\n{'Config':<20}", end="")
    for seq_len in seq_lengths:
        print(f"{'Dist '+str(seq_len-2):<12}", end="")
    print("Avg Acc")
    print("-" * 80)

    for name, results in all_results.items():
        print(f"{name:<20}", end="")
        accs = []
        for seq_len in seq_lengths:
            acc = results[seq_len]['accuracy']
            accs.append(acc)
            print(f"{acc:<12.1%}", end="")
        print(f"{sum(accs)/len(accs):.1%}")

    # Beta analysis for learnable configs
    print("\n" + "=" * 80)
    print("LEARNED BETA VALUES")
    print("=" * 80)

    for name in ['β=1→learned', 'β=2→learned', 'per-head β']:
        if name in all_results:
            print(f"\n{name}:")
            for seq_len in seq_lengths:
                betas = all_results[name][seq_len]['final_betas']
                beta_str = ", ".join([f"L{i}:{b[0]:.2f}" for i, b in enumerate(betas)])
                print(f"  Dist {seq_len-2}: [{beta_str}]")

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    baseline_avg = sum(r['accuracy'] for r in all_results['β=1 (standard)'].values()) / len(seq_lengths)

    print(f"\nBaseline (β=1): {baseline_avg:.1%}")

    for name, results in all_results.items():
        if name == 'β=1 (standard)':
            continue
        avg = sum(r['accuracy'] for r in results.values()) / len(seq_lengths)
        diff = avg - baseline_avg
        print(f"{name}: {avg:.1%} ({'+' if diff >= 0 else ''}{diff*100:.1f}%)")

    # Best config
    best_name = max(all_results, key=lambda n: sum(r['accuracy'] for r in all_results[n].values()))
    best_avg = sum(r['accuracy'] for r in all_results[best_name].values()) / len(seq_lengths)
    print(f"\nBest config: {best_name} ({best_avg:.1%})")

    if 'learned' in best_name or 'per-head' in best_name:
        print("\n=> Learnable beta helps! The model learns task-appropriate sharpness.")
    else:
        print(f"\n=> Fixed β={best_name.split('=')[1].split()[0]} is best. Simple solution wins.")


if __name__ == "__main__":
    main()
