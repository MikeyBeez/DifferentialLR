#!/usr/bin/env python3
"""
Combined Approach: Linear Attention Early + MLP Late

Testing the hypothesis that:
- Early layers: Learned-gate linear attention (efficient, PPL 167.4)
- Later layers: Static MLPs (fast, works on structured representations)

This could give us efficiency throughout while maintaining quality.
"""

import sys
sys.path.insert(0, '/home/bee/Code/LinearAttention')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import math
import time


class SoftmaxAttention(nn.Module):
    """Full softmax attention."""

    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        B, L, D = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(out)


class LearnedGateLinearAttention(nn.Module):
    """
    Linear attention with learned gating - the best linear variant (PPL 167.4).

    Uses chunked processing with:
    - ELU+1 feature map
    - Learned data-dependent gate for state updates
    """

    def __init__(self, hidden_dim, num_heads, chunk_size=64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Learned gate for state updates
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, num_heads),
            nn.Sigmoid(),
        )

        self.eps = 1e-6

    def feature_map(self, x):
        return F.elu(x) + 1

    def forward(self, x):
        B, L, D = x.shape
        H = self.num_heads
        d = self.head_dim
        C = self.chunk_size

        q = self.q_proj(x).view(B, L, H, d)
        k = self.k_proj(x).view(B, L, H, d)
        v = self.v_proj(x).view(B, L, H, d)

        # Apply feature map
        q = self.feature_map(q)
        k = self.feature_map(k)

        # Compute gates
        gates = self.gate(x)  # (B, L, H)

        # Pad to multiple of chunk_size
        num_chunks = (L + C - 1) // C
        pad_len = num_chunks * C - L

        if pad_len > 0:
            q = F.pad(q, (0, 0, 0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, 0, 0, pad_len))
            gates = F.pad(gates, (0, 0, 0, pad_len))

        L_padded = q.shape[1]

        # Reshape to chunks
        q = q.view(B, num_chunks, C, H, d)
        k = k.view(B, num_chunks, C, H, d)
        v = v.view(B, num_chunks, C, H, d)
        gates = gates.view(B, num_chunks, C, H)

        outputs = []
        kv_state = torch.zeros(B, H, d, d, device=x.device)
        k_state = torch.zeros(B, H, d, device=x.device)

        for i in range(num_chunks):
            q_c = q[:, i].transpose(1, 2)  # (B, H, C, d)
            k_c = k[:, i].transpose(1, 2)
            v_c = v[:, i].transpose(1, 2)
            g_c = gates[:, i]  # (B, C, H)

            # Within-chunk linear attention (causal)
            within_scores = torch.matmul(q_c, k_c.transpose(-2, -1))
            causal_mask = torch.triu(torch.ones(C, C, device=x.device), diagonal=1).bool()
            within_scores = within_scores.masked_fill(causal_mask, 0)
            within_norm = within_scores.sum(dim=-1, keepdim=True) + self.eps
            within_out = torch.matmul(within_scores / within_norm, v_c)

            # Cross-chunk: use accumulated state
            cross_out = torch.einsum('bhcd,bhde->bhce', q_c, kv_state)
            cross_norm = torch.einsum('bhcd,bhd->bhc', q_c, k_state).unsqueeze(-1) + self.eps
            cross_out = cross_out / cross_norm

            # Combine within and cross
            out = within_out + cross_out
            outputs.append(out.transpose(1, 2))  # (B, C, H, d)

            # Update state with learned gating
            chunk_gate = g_c.mean(dim=1)  # (B, H)
            k_mean = k_c.mean(dim=2)  # (B, H, d)
            v_mean = v_c.mean(dim=2)

            gate_expanded = chunk_gate.unsqueeze(-1).unsqueeze(-1)
            kv_state = gate_expanded * kv_state + torch.einsum('bhd,bhe->bhde', k_mean, v_mean)
            k_state = chunk_gate.unsqueeze(-1) * k_state + k_mean

        out = torch.cat(outputs, dim=1)  # (B, L_padded, H, d)

        # Remove padding
        if pad_len > 0:
            out = out[:, :L]

        out = out.reshape(B, L, D)
        return self.o_proj(out)


class StaticMLP(nn.Module):
    """Static MLP for later layers."""

    def __init__(self, hidden_dim, expansion_factor=2):
        super().__init__()
        intermediate = hidden_dim * expansion_factor

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, intermediate),
            nn.GELU(),
            nn.Linear(intermediate, hidden_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class StaticMLPWithProjection(nn.Module):
    """Static MLP with projection bottleneck."""

    def __init__(self, hidden_dim, expansion_factor=2, bottleneck_ratio=0.5):
        super().__init__()
        intermediate = hidden_dim * expansion_factor
        bottleneck = int(hidden_dim * bottleneck_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, intermediate),
            nn.GELU(),
            nn.Linear(intermediate, hidden_dim),
        )

        self.down_proj = nn.Linear(hidden_dim, bottleneck, bias=False)
        self.up_proj = nn.Linear(bottleneck, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(bottleneck)

    def forward(self, x):
        mlp_out = self.mlp(x)
        compressed = self.norm(self.down_proj(mlp_out))
        return self.up_proj(compressed)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, intermediate_dim, dropout, attn_type='softmax'):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_dim)

        if attn_type == 'softmax':
            self.attention = SoftmaxAttention(hidden_dim, num_heads)
        elif attn_type == 'linear_gate':
            self.attention = LearnedGateLinearAttention(hidden_dim, num_heads)
        elif attn_type == 'mlp':
            self.attention = StaticMLP(hidden_dim)
        elif attn_type == 'mlp_proj':
            self.attention = StaticMLPWithProjection(hidden_dim)
        else:
            raise ValueError(f"Unknown attention type: {attn_type}")

        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attention(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class HybridTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim=512, num_layers=8, num_heads=8,
                 intermediate_dim=2048, max_seq_len=1024, dropout=0.1,
                 early_type='softmax', late_type='mlp', early_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            if i < early_layers:
                attn_type = early_type
            else:
                attn_type = late_type
            self.blocks.append(
                TransformerBlock(hidden_dim, num_heads, intermediate_dim, dropout, attn_type)
            )

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        output = {"logits": logits}

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            output["loss"] = loss

        return output


def get_wikitext_data(tokenizer, seq_length=512, batch_size=8):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    val_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=False, padding=False)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    val_tokenized = val_dataset.map(tokenize, batched=True, remove_columns=["text"])

    all_tokens, val_tokens = [], []
    for item in tokenized:
        all_tokens.extend(item["input_ids"])
    for item in val_tokenized:
        val_tokens.extend(item["input_ids"])

    def create_sequences(tokens, seq_len):
        return torch.tensor([tokens[i:i+seq_len] for i in range(0, len(tokens)-seq_len, seq_len)])

    train_loader = DataLoader(create_sequences(all_tokens, seq_length), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(create_sequences(val_tokens, seq_length), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, scaler, device):
    model.train()
    total_loss, num_batches = 0, 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            outputs = model(batch, labels=batch)
            loss = outputs["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    total_loss, num_batches = 0, 0

    for batch in val_loader:
        batch = batch.to(device)
        with torch.amp.autocast('cuda'):
            outputs = model(batch, labels=batch)
        total_loss += outputs["loss"].item()
        num_batches += 1

    return math.exp(total_loss / num_batches)


@torch.no_grad()
def benchmark_speed(model, device, seq_len=512, batch_size=1, num_runs=50):
    model.eval()
    dummy = torch.randint(0, 1000, (batch_size, seq_len), device=device)

    for _ in range(10):
        _ = model(dummy)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_runs):
        _ = model(dummy)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    return (num_runs * batch_size * seq_len) / elapsed


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n" + "="*70)
    print("LINEAR ATTENTION EARLY + MLP LATE")
    print("="*70)
    print("\nTesting combinations of attention types across layers")

    print("\nLoading data...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_loader, val_loader = get_wikitext_data(tokenizer, seq_length=512, batch_size=8)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    configs = [
        # Baselines
        ("Full Softmax (baseline)", 'softmax', 'softmax', 8),
        ("Full Linear+Gate", 'linear_gate', 'linear_gate', 8),

        # Softmax early + MLP late (previous best)
        ("4 Softmax + 4 MLP", 'softmax', 'mlp', 4),

        # Linear early + MLP late (new test)
        ("4 Linear+Gate + 4 MLP", 'linear_gate', 'mlp', 4),
        ("6 Linear+Gate + 2 MLP", 'linear_gate', 'mlp', 6),
        ("2 Linear+Gate + 6 MLP", 'linear_gate', 'mlp', 2),

        # Linear early + MLP+Proj late
        ("4 Linear+Gate + 4 MLP+Proj", 'linear_gate', 'mlp_proj', 4),

        # Mixed: some softmax, some linear, some MLP
        ("2 Softmax + 2 Linear + 4 MLP", 'softmax', 'mlp', 2),  # Will manually adjust
    ]

    results = []

    for name, early_type, late_type, early_layers in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"  Early layers ({early_type}): {early_layers}")
        print(f"  Late layers ({late_type}): {8 - early_layers}")
        print("="*60)

        model = HybridTransformer(
            vocab_size=tokenizer.vocab_size,
            hidden_dim=512,
            num_layers=8,
            num_heads=8,
            intermediate_dim=2048,
            early_type=early_type,
            late_type=late_type,
            early_layers=early_layers,
        ).to(device)

        num_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {num_params:,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
        scaler = torch.amp.GradScaler('cuda')

        best_ppl = float('inf')
        start = time.time()

        for epoch in range(10):
            train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
            train_ppl = math.exp(train_loss)
            eval_ppl = evaluate(model, val_loader, device)

            marker = "*" if eval_ppl < best_ppl else ""
            best_ppl = min(best_ppl, eval_ppl)

            elapsed = (time.time() - start) / 60
            print(f"Epoch {epoch+1:2d} | Train PPL: {train_ppl:7.1f} | Eval PPL: {eval_ppl:7.1f} {marker} | {elapsed:.1f}m", flush=True)

        speed = benchmark_speed(model, device)

        results.append({
            'name': name,
            'ppl': best_ppl,
            'params': num_params,
            'speed': speed,
        })

        print(f"\n{name}:")
        print(f"  Best PPL: {best_ppl:.1f}")
        print(f"  Speed: {speed:.0f} tokens/sec")

    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'Configuration':<35} | {'PPL':>7} | {'Speed':>10} | {'Params':>10}")
    print("-"*70)

    baseline_ppl = results[0]['ppl']
    baseline_speed = results[0]['speed']

    for r in results:
        ppl_diff = (r['ppl'] - baseline_ppl) / baseline_ppl * 100
        speed_diff = (r['speed'] - baseline_speed) / baseline_speed * 100
        print(f"{r['name']:<35} | {r['ppl']:>7.1f} | {r['speed']:>7.0f} t/s | {r['params']:>10,}")
        if r['name'] != results[0]['name']:
            print(f"{'':35} | {ppl_diff:>+6.1f}% | {speed_diff:>+6.1f}%")

    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    # Find interesting results
    softmax_mlp = next((r for r in results if '4 Softmax + 4 MLP' in r['name']), None)
    linear_mlp = next((r for r in results if '4 Linear+Gate + 4 MLP' == r['name']), None)

    if softmax_mlp and linear_mlp:
        print(f"\nSoftmax early vs Linear early (both + 4 MLP late):")
        print(f"  4 Softmax + 4 MLP:     PPL {softmax_mlp['ppl']:.1f}, {softmax_mlp['speed']:.0f} t/s")
        print(f"  4 Linear+Gate + 4 MLP: PPL {linear_mlp['ppl']:.1f}, {linear_mlp['speed']:.0f} t/s")

    print("\n" + "="*70)
    print("COMPARISON TO PREVIOUS BEST RESULTS")
    print("="*70)
    print(f"  Linear attention (learned gate, all 8): PPL 167.4")
    print(f"  Full Softmax baseline:                  PPL 170.4")
    print(f"  Hybrid 4 softmax + 4 MLP:               PPL 174.5")


if __name__ == "__main__":
    main()
