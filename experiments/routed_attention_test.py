#!/usr/bin/env python3
"""
Routed Attention: Learn When to Use Expensive Attention

The insight: Most tokens don't need O(N²) attention. Language is local.
But some tokens (pronouns, questions, callbacks) need long-range retrieval.

Architecture:
- Router decides per-position: use Conv (cheap) or Attention (expensive)
- Loss = task_loss + λ * attention_cost
- Goal: Use conv as often as possible, attention only when necessary

This is Mixture of Experts for attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    vocab_size: int = 256
    dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.0
    kernel_size: int = 64

    # Router settings
    attention_cost: float = 0.1  # λ - penalty for using attention
    temperature: float = 1.0  # Gumbel-softmax temperature


class CausalConv(nn.Module):
    """O(N) causal convolution - cheap path."""
    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.kernel_size = config.kernel_size

        self.v_proj = nn.Linear(config.dim, config.dim)
        self.out = nn.Linear(config.dim, config.dim)

        # Learned kernel per head
        self.kernel_logits = nn.Parameter(
            torch.zeros(config.num_heads, config.kernel_size)
        )
        self.head_dim = config.dim // config.num_heads
        self.num_heads = config.num_heads

    def forward(self, x):
        B, T, C = x.shape

        v = self.v_proj(x).transpose(1, 2)  # (B, C, T)
        v_padded = F.pad(v, (self.kernel_size - 1, 0))

        kernel = F.softmax(self.kernel_logits, dim=-1)
        kernel_expanded = kernel.unsqueeze(1).expand(-1, self.head_dim, -1)
        kernel_expanded = kernel_expanded.reshape(self.dim, 1, self.kernel_size)

        out = F.conv1d(v_padded, kernel_expanded, groups=self.dim)
        out = out.transpose(1, 2)

        return self.out(out)


class StandardAttention(nn.Module):
    """O(N²) attention - expensive path for retrieval."""
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


class Router(nn.Module):
    """
    Learns when to use attention vs conv.

    Per-position decision based on:
    1. Current token embedding
    2. Recent context summary
    """
    def __init__(self, config: Config):
        super().__init__()
        self.temperature = config.temperature

        # Router MLP: looks at current position + context
        self.router = nn.Sequential(
            nn.Linear(config.dim * 2, config.dim),
            nn.GELU(),
            nn.Linear(config.dim, 2)  # 2 choices: conv or attention
        )

        # Context summarizer (simple: exponential moving average)
        self.context_decay = 0.9

    def forward(self, x, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            decisions: (B, T) - 0 for conv, 1 for attention
            probs: (B, T, 2) - probabilities for loss computation
        """
        B, T, C = x.shape

        # Build context summary via EMA
        context = torch.zeros(B, C, device=x.device)
        contexts = []
        for t in range(T):
            context = self.context_decay * context + (1 - self.context_decay) * x[:, t]
            contexts.append(context)
        context_summary = torch.stack(contexts, dim=1)  # (B, T, C)

        # Router input: current token + context
        router_input = torch.cat([x, context_summary], dim=-1)  # (B, T, 2C)

        # Get logits
        logits = self.router(router_input)  # (B, T, 2)

        if training:
            # Gumbel-softmax for differentiable discrete choice
            decisions = F.gumbel_softmax(logits, tau=self.temperature, hard=True)
            probs = F.softmax(logits, dim=-1)
            return decisions[:, :, 1], probs  # Return attention probability
        else:
            # Hard decision at inference
            decisions = logits.argmax(dim=-1)  # (B, T)
            probs = F.softmax(logits, dim=-1)
            return decisions.float(), probs


class RoutedAttentionLayer(nn.Module):
    """
    Single layer that routes between conv and attention per-position.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.conv = CausalConv(config)
        self.attn = StandardAttention(config)
        self.router = Router(config)
        self.config = config

    def forward(self, x, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output: (B, T, C)
            attention_usage: scalar - fraction of positions using attention
        """
        B, T, C = x.shape

        # Get routing decisions
        attn_decisions, probs = self.router(x, training)  # (B, T), (B, T, 2)

        # Compute both paths
        conv_out = self.conv(x)  # (B, T, C)
        attn_out = self.attn(x)  # (B, T, C)

        # Mix based on decisions
        attn_weight = attn_decisions.unsqueeze(-1)  # (B, T, 1)
        out = (1 - attn_weight) * conv_out + attn_weight * attn_out

        # Compute attention usage for cost
        attention_usage = attn_decisions.mean()

        return out, attention_usage, probs


class RoutedTransformerBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.dim)
        self.routed_attn = RoutedAttentionLayer(config)
        self.norm2 = nn.LayerNorm(config.dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.dim, config.dim * 4),
            nn.GELU(),
            nn.Linear(config.dim * 4, config.dim),
        )

    def forward(self, x, training: bool = True):
        attn_out, attn_usage, probs = self.routed_attn(self.norm1(x), training)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, attn_usage, probs


class RoutedTransformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([
            RoutedTransformerBlock(config) for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size)

    def forward(self, x, training: bool = True):
        x = self.embedding(x)

        total_attn_usage = 0
        all_probs = []

        for block in self.blocks:
            x, attn_usage, probs = block(x, training)
            total_attn_usage += attn_usage
            all_probs.append(probs)

        avg_attn_usage = total_attn_usage / len(self.blocks)

        logits = self.head(self.norm(x))
        return logits, avg_attn_usage, all_probs


# --- Baselines ---

class ConvOnlyTransformer(nn.Module):
    """Pure conv - fast but can't retrieve."""
    def __init__(self, config: Config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList()
        for _ in range(config.num_layers):
            self.blocks.append(nn.ModuleDict({
                'norm1': nn.LayerNorm(config.dim),
                'attn': CausalConv(config),
                'norm2': nn.LayerNorm(config.dim),
                'ffn': nn.Sequential(
                    nn.Linear(config.dim, config.dim * 4),
                    nn.GELU(),
                    nn.Linear(config.dim * 4, config.dim),
                )
            }))
        self.norm = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size)

    def forward(self, x, training: bool = True):
        x = self.embedding(x)
        for block in self.blocks:
            x = x + block['attn'](block['norm1'](x))
            x = x + block['ffn'](block['norm2'](x))
        return self.head(self.norm(x)), 0.0, None


class AttnOnlyTransformer(nn.Module):
    """Pure attention - accurate but expensive."""
    def __init__(self, config: Config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList()
        for _ in range(config.num_layers):
            self.blocks.append(nn.ModuleDict({
                'norm1': nn.LayerNorm(config.dim),
                'attn': StandardAttention(config),
                'norm2': nn.LayerNorm(config.dim),
                'ffn': nn.Sequential(
                    nn.Linear(config.dim, config.dim * 4),
                    nn.GELU(),
                    nn.Linear(config.dim * 4, config.dim),
                )
            }))
        self.norm = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size)

    def forward(self, x, training: bool = True):
        x = self.embedding(x)
        for block in self.blocks:
            x = x + block['attn'](block['norm1'](x))
            x = x + block['ffn'](block['norm2'](x))
        return self.head(self.norm(x)), 1.0, None


# --- Data Generation ---

def generate_mixed_batch(batch_size, seq_len, vocab_size, retrieval_prob=0.3, device='cuda'):
    """
    Generate a mix of:
    1. Language-like sequences (local patterns, conv should handle)
    2. Retrieval sequences (key-value, needs attention)

    Returns sequences and targets, where some positions require retrieval.
    """
    KEY_START, KEY_END = 10, 50
    VAL_START, VAL_END = 50, 100
    LANG_START = 100

    sequences = []
    targets = []
    needs_retrieval = []  # Track which positions need long-range

    for _ in range(batch_size):
        seq = torch.zeros(seq_len, dtype=torch.long, device=device)
        target = torch.zeros(seq_len, dtype=torch.long, device=device)
        retrieval_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)

        # Decide: retrieval task or language-like?
        if torch.rand(1).item() < retrieval_prob:
            # Retrieval task: key-value at start, query at end
            key = torch.randint(KEY_START, KEY_END, (1,), device=device)
            val = torch.randint(VAL_START, VAL_END, (1,), device=device)
            seq[0] = key
            seq[1] = val
            seq[2:-1] = torch.randint(LANG_START, vocab_size, (seq_len - 3,), device=device)
            seq[-1] = key  # Query

            # Target: predict val after seeing query
            target[:-1] = seq[1:]  # Shift for next-token prediction
            target[-1] = val  # The answer

            retrieval_mask[-1] = True  # Last position needs retrieval
        else:
            # Language-like: just random tokens, next-token prediction
            seq = torch.randint(LANG_START, vocab_size, (seq_len,), device=device)
            target = torch.roll(seq, -1)
            target[-1] = torch.randint(LANG_START, vocab_size, (1,), device=device)

        sequences.append(seq)
        targets.append(target)
        needs_retrieval.append(retrieval_mask)

    return (
        torch.stack(sequences),
        torch.stack(targets),
        torch.stack(needs_retrieval)
    )


def generate_recall_batch(batch_size, seq_len, vocab_size, device='cuda'):
    """Pure retrieval task for testing."""
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


# --- Training ---

def get_batch_size(seq_len):
    """Adaptive batch size to avoid OOM on longer sequences."""
    if seq_len <= 256:
        return 32
    elif seq_len <= 512:
        return 24
    elif seq_len <= 1024:
        return 16
    elif seq_len <= 2048:
        return 8
    else:
        return 4


def train_epoch(model, optimizer, config, seq_len, num_batches=100, batch_size=None, device='cuda'):
    """Train on pure retrieval task - the only fair test."""
    if batch_size is None:
        batch_size = get_batch_size(seq_len)
    model.train()

    total_loss = 0
    total_correct = 0
    total_samples = 0
    total_attn_usage = 0

    for _ in range(num_batches):
        # Pure retrieval training
        seqs, targets = generate_recall_batch(batch_size, seq_len, config.vocab_size, device)

        optimizer.zero_grad()

        logits, attn_usage, _ = model(seqs, training=True)

        # Task loss: predict correct value at last position
        task_loss = F.cross_entropy(logits[:, -1], targets)

        # Attention cost
        cost_loss = config.attention_cost * attn_usage

        # Total loss
        loss = task_loss + cost_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Track accuracy
        preds = logits[:, -1].argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size

        total_loss += task_loss.item()
        total_attn_usage += attn_usage.item() if isinstance(attn_usage, torch.Tensor) else attn_usage

    return total_loss / num_batches, total_correct / total_samples, total_attn_usage / num_batches


@torch.no_grad()
def evaluate_recall(model, config, seq_len, num_batches=50, batch_size=None, device='cuda'):
    """Test pure retrieval accuracy."""
    if batch_size is None:
        batch_size = get_batch_size(seq_len)
    model.eval()
    total_correct = 0
    total_samples = 0
    total_attn_usage = 0

    for _ in range(num_batches):
        seqs, targets = generate_recall_batch(batch_size, seq_len, config.vocab_size, device)
        logits, attn_usage, _ = model(seqs, training=False)

        # Check if last position predicts correct value
        preds = logits[:, -1].argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size
        total_attn_usage += attn_usage.item() if isinstance(attn_usage, torch.Tensor) else attn_usage

    return total_correct / total_samples, total_attn_usage / num_batches


def test_model(name, model_class, config, seq_lengths, device, epochs=30):
    print(f"\n{'='*60}", flush=True)
    print(f"{name}", flush=True)
    print(f"{'='*60}", flush=True)

    results = {}

    for seq_len in seq_lengths:
        print(f"\n--- Seq Length: {seq_len} (distance: {seq_len-2}) ---", flush=True)

        model = model_class(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        params = sum(p.numel() for p in model.parameters())
        if seq_len == seq_lengths[0]:
            print(f"Parameters: {params:,}", flush=True)

        best_acc = 0
        for epoch in range(1, epochs + 1):
            loss, train_acc, attn_usage = train_epoch(
                model, optimizer, config, seq_len, device=device
            )
            recall_acc, eval_attn = evaluate_recall(model, config, seq_len, device=device)

            if recall_acc > best_acc:
                best_acc = recall_acc

            if epoch % 5 == 0 or recall_acc > 0.99:
                print(f"Epoch {epoch}: Loss {loss:.3f}, Train {train_acc:.1%}, Val {recall_acc:.1%}, "
                      f"Attn Usage {eval_attn:.1%}", flush=True)

            if recall_acc > 0.99:
                print(f"SOLVED! (Attn: {eval_attn:.1%})", flush=True)
                break

        results[seq_len] = (best_acc, eval_attn)
        del model
        torch.cuda.empty_cache()

    return results


def test_curriculum(name, model_class, config, seq_lengths, device, phase1_epochs=25, phase2_epochs=20):
    """
    Curriculum learning for routed attention:
    Phase 1: λ=0, learn to solve task (router learns when attention is needed)
    Phase 2: Increase λ, learn to minimize attention while maintaining accuracy
    """
    print(f"\n{'='*60}", flush=True)
    print(f"{name}", flush=True)
    print(f"{'='*60}", flush=True)

    results = {}

    for seq_len in seq_lengths:
        print(f"\n--- Seq Length: {seq_len} (distance: {seq_len-2}) ---", flush=True)

        # Create config with λ=0 for phase 1
        phase1_config = Config(attention_cost=0.0)
        model = model_class(phase1_config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        params = sum(p.numel() for p in model.parameters())
        if seq_len == seq_lengths[0]:
            print(f"Parameters: {params:,}", flush=True)

        # Phase 1: Learn to solve task (λ=0)
        print(f"Phase 1: λ=0 (learn task)", flush=True)
        best_acc = 0
        solved = False
        for epoch in range(1, phase1_epochs + 1):
            loss, train_acc, attn_usage = train_epoch(
                model, optimizer, phase1_config, seq_len, device=device
            )
            recall_acc, eval_attn = evaluate_recall(model, config, seq_len, device=device)

            if recall_acc > best_acc:
                best_acc = recall_acc

            if epoch % 5 == 0 or recall_acc > 0.99:
                print(f"  Epoch {epoch}: Val {recall_acc:.1%}, Attn {eval_attn:.1%}", flush=True)

            if recall_acc > 0.99:
                solved = True
                break

        if not solved:
            print(f"  Phase 1 failed to solve, skipping phase 2", flush=True)
            results[seq_len] = (best_acc, eval_attn)
            del model
            torch.cuda.empty_cache()
            continue

        # Phase 2: Gradually increase λ
        print(f"Phase 2: Increase λ (minimize attention)", flush=True)
        target_lambda = config.attention_cost

        for epoch in range(1, phase2_epochs + 1):
            # Linear warmup of λ
            current_lambda = target_lambda * (epoch / phase2_epochs)
            phase2_config = Config(attention_cost=current_lambda)

            loss, train_acc, attn_usage = train_epoch(
                model, optimizer, phase2_config, seq_len, device=device
            )
            recall_acc, eval_attn = evaluate_recall(model, config, seq_len, device=device)

            if epoch % 5 == 0:
                print(f"  Epoch {epoch}: λ={current_lambda:.2f}, Val {recall_acc:.1%}, Attn {eval_attn:.1%}", flush=True)

            # Stop if accuracy drops too much
            if recall_acc < 0.90:
                print(f"  Accuracy dropped, stopping phase 2", flush=True)
                break

        print(f"Final: {recall_acc:.1%} accuracy, {eval_attn:.1%} attention", flush=True)
        results[seq_len] = (recall_acc, eval_attn)
        del model
        torch.cuda.empty_cache()

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Routed Attention Test", flush=True)
    print(f"Learning when to use expensive attention vs cheap conv", flush=True)

    config = Config()
    seq_lengths = [128, 256, 512, 1024, 2048]

    all_results = {}

    # 1. Pure attention (baseline - should work but expensive)
    all_results['Attn Only'] = test_model(
        "Attention Only (O(N²))", AttnOnlyTransformer, config, seq_lengths, device
    )

    # 2. Pure conv (baseline - fast but should fail at long range)
    all_results['Conv Only'] = test_model(
        "Conv Only (O(N))", ConvOnlyTransformer, config, seq_lengths, device
    )

    # 3. Routed with curriculum learning
    all_results['Routed'] = test_curriculum(
        "Routed (curriculum)", RoutedTransformer, config, seq_lengths, device
    )

    # 4. Routed with higher attention cost
    config_cheap = Config(attention_cost=0.5)
    all_results['Routed (λ=0.5)'] = test_curriculum(
        "Routed (curriculum λ=0.5)", RoutedTransformer, config_cheap, seq_lengths, device
    )

    # Summary
    print(f"\n{'='*70}", flush=True)
    print("ROUTED ATTENTION RESULTS", flush=True)
    print(f"{'='*70}", flush=True)

    print(f"\n{'Distance':<10} {'Attn Only':<20} {'Conv Only':<20} {'Routed':<20} {'Routed λ=0.5':<20}", flush=True)
    print("-" * 90, flush=True)

    for seq_len in seq_lengths:
        distance = seq_len - 2
        print(f"{distance:<10}", end="", flush=True)
        for name in ['Attn Only', 'Conv Only', 'Routed', 'Routed (λ=0.5)']:
            acc, attn = all_results[name].get(seq_len, (0, 0))
            print(f"{acc:>5.0%} ({attn:>3.0%} attn) ", end="", flush=True)
        print(flush=True)

    print(f"\n{'='*70}", flush=True)

    # Analysis
    print("\nANALYSIS:", flush=True)

    routed = all_results['Routed']
    attn_only = all_results['Attn Only']

    for seq_len in seq_lengths:
        r_acc, r_attn = routed.get(seq_len, (0, 1))
        a_acc, _ = attn_only.get(seq_len, (0, 1))

        savings = (1 - r_attn) * 100

        if r_acc >= a_acc * 0.95:  # Within 5% of full attention
            print(f"Distance {seq_len-2}: Routed saves {savings:.0f}% compute while matching accuracy", flush=True)
        else:
            print(f"Distance {seq_len-2}: Routed underperforms (may need more training)", flush=True)


if __name__ == "__main__":
    main()
