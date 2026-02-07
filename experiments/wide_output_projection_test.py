"""
Wide Output Projection Test

Hypothesis: W_O (the output projection in attention) may be a bottleneck for
associative recall. If attention aggregates information correctly but W_O
compresses it too much, recall suffers.

Test: Vary the width of W_O and measure recall accuracy.

Variants:
1. Standard: W_O is dim x dim (256 x 256)
2. Wide: W_O is (dim * 2) x dim, with intermediate projection
3. Very Wide: W_O is (dim * 4) x dim
4. Bottleneck: W_O is (dim // 2) x dim (control - should be worse)

Usage:
    python3 experiments/wide_output_projection_test.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")


class StandardAttention(nn.Module):
    """Standard attention with normal W_O."""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)  # Standard: dim -> dim

    def forward(self, x):
        B, L, D = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


class WideOutputAttention(nn.Module):
    """Attention with wider intermediate representation before output projection."""

    def __init__(self, dim, num_heads=8, expansion=2):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.expansion = expansion
        self.wide_dim = dim * expansion

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, self.wide_dim)  # Project V to wider space

        # Output projection: wide -> dim
        self.out_proj = nn.Linear(self.wide_dim, dim)

    def forward(self, x):
        B, L, D = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # V projects to wider space
        wide_head_dim = self.wide_dim // self.num_heads
        v = self.v_proj(x).view(B, L, self.num_heads, wide_head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, L, self.wide_dim)
        return self.out_proj(out)


class WideOutputMLPAttention(nn.Module):
    """Attention with MLP-style wide output projection."""

    def __init__(self, dim, num_heads=8, expansion=2):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.expansion = expansion
        self.wide_dim = dim * expansion

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # MLP-style output: dim -> wide -> dim
        self.out_expand = nn.Linear(dim, self.wide_dim)
        self.out_contract = nn.Linear(self.wide_dim, dim)

    def forward(self, x):
        B, L, D = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, L, D)

        # MLP-style: expand then contract
        out = self.out_expand(out)
        out = F.gelu(out)
        out = self.out_contract(out)
        return out


class BottleneckAttention(nn.Module):
    """Attention with narrower output projection (control - should be worse)."""

    def __init__(self, dim, num_heads=8, bottleneck=2):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.narrow_dim = dim // bottleneck

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, self.narrow_dim)

        self.out_proj = nn.Linear(self.narrow_dim, dim)

    def forward(self, x):
        B, L, D = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        narrow_head_dim = self.narrow_dim // self.num_heads
        v = self.v_proj(x).view(B, L, self.num_heads, narrow_head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, L, self.narrow_dim)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Single transformer block with attention + FFN."""

    def __init__(self, dim, attention_class, **attn_kwargs):
        super().__init__()
        self.attention = attention_class(dim, **attn_kwargs)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class RecallModel(nn.Module):
    """Model for associative recall task - 4 layers like the working version."""

    def __init__(self, vocab_size, dim, attention_class, num_layers=4, **attn_kwargs):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, attention_class, **attn_kwargs)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.output(h)


def generate_recall_batch(batch_size, seq_len, vocab_size, device='cuda'):
    """
    Generate recall task matching the working version:
    - Key at position 0
    - Value at position 1
    - Noise at positions 2 to seq_len-2
    - Query (same key) at position seq_len-1
    - Target: predict the value
    """
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
        seq[-1] = key  # Query: repeat the key
        sequences.append(seq)
        targets.append(val)

    return torch.stack(sequences), torch.tensor(targets, device=device).squeeze()


def train_epoch(model, optimizer, seq_len, num_batches=100, batch_size=32, device='cuda'):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    vocab_size = model.output.out_features

    for _ in range(num_batches):
        seq, targets = generate_recall_batch(batch_size, seq_len, vocab_size, device)

        optimizer.zero_grad()
        logits = model(seq)

        # Loss only on last position
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
    vocab_size = model.output.out_features

    for _ in range(num_batches):
        seq, targets = generate_recall_batch(batch_size, seq_len, vocab_size, device)
        logits = model(seq)
        preds = logits[:, -1].argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size

    return total_correct / total_samples


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model(name, attention_class, seq_lengths, device, epochs=30, **attn_kwargs):
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")

    vocab_size = 256  # Match working model
    dim = 128  # Match working model

    results = {}

    for seq_len in seq_lengths:
        print(f"\n--- Seq Length: {seq_len} (distance: {seq_len - 2}) ---")

        model = RecallModel(vocab_size, dim, attention_class, **attn_kwargs).to(device)

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
                print(f"Epoch {epoch}: Loss {loss:.3f}, Train {train_acc:.1%}, Val {val_acc:.1%}")

            if val_acc >= 0.99:
                print(f"SOLVED!")
                break

        results[seq_len] = best_acc
        del model, optimizer
        torch.cuda.empty_cache()

    return results


def main():
    print("=" * 60)
    print("Wide Output Projection Test")
    print("Does increasing W_O capacity improve associative recall?")
    print("=" * 60)

    seq_lengths = [64, 128, 256, 512]

    all_results = {}

    # Standard attention (baseline)
    all_results['Standard (1x)'] = test_model(
        "Standard Attention (W_O: dim -> dim)",
        StandardAttention,
        seq_lengths,
        device,
        num_heads=8
    )

    # Wide V projection (2x)
    all_results['Wide V (2x)'] = test_model(
        "Wide V Projection (V: dim -> 2*dim, W_O: 2*dim -> dim)",
        WideOutputAttention,
        seq_lengths,
        device,
        num_heads=8,
        expansion=2
    )

    # Wide V projection (4x)
    all_results['Wide V (4x)'] = test_model(
        "Wide V Projection (V: dim -> 4*dim, W_O: 4*dim -> dim)",
        WideOutputAttention,
        seq_lengths,
        device,
        num_heads=8,
        expansion=4
    )

    # MLP-style output (2x)
    all_results['MLP Output (2x)'] = test_model(
        "MLP Output (attn -> expand 2x -> GELU -> contract)",
        WideOutputMLPAttention,
        seq_lengths,
        device,
        num_heads=8,
        expansion=2
    )

    # Bottleneck (control)
    all_results['Bottleneck (0.5x)'] = test_model(
        "Bottleneck (V: dim -> dim/2, W_O: dim/2 -> dim)",
        BottleneckAttention,
        seq_lengths,
        device,
        num_heads=8,
        bottleneck=2
    )

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Distance':<12}", end="")
    for name in all_results:
        print(f"{name:<18}", end="")
    print()
    print("-" * 100)

    for seq_len in seq_lengths:
        distance = seq_len - 2
        print(f"{distance:<12}", end="")
        for name, results in all_results.items():
            acc = results.get(seq_len, 0)
            print(f"{acc:<18.1%}", end="")
        print()

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Check if wider helps
    baseline_avg = sum(all_results['Standard (1x)'].values()) / len(seq_lengths)
    wide2_avg = sum(all_results['Wide V (2x)'].values()) / len(seq_lengths)
    wide4_avg = sum(all_results['Wide V (4x)'].values()) / len(seq_lengths)

    print(f"\nAverage accuracy across distances:")
    print(f"  Standard (1x):    {baseline_avg:.1%}")
    print(f"  Wide V (2x):      {wide2_avg:.1%} ({'+' if wide2_avg > baseline_avg else ''}{(wide2_avg - baseline_avg)*100:.1f}%)")
    print(f"  Wide V (4x):      {wide4_avg:.1%} ({'+' if wide4_avg > baseline_avg else ''}{(wide4_avg - baseline_avg)*100:.1f}%)")

    if wide2_avg > baseline_avg + 0.02:
        print("\n=> Wider W_O HELPS! The output projection may be a bottleneck.")
    elif wide2_avg < baseline_avg - 0.02:
        print("\n=> Wider W_O HURTS. More capacity doesn't help recall.")
    else:
        print("\n=> Wider W_O has minimal effect. Bottleneck is elsewhere.")


if __name__ == "__main__":
    main()
