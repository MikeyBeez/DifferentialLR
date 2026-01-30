#!/usr/bin/env python3
"""
Static Function Hypothesis: Hybrid Transformer

Based on the insight that:
- Early layers need DYNAMIC attention for feature extraction on unstructured input
- Later layers perform PREDICTABLE operations on already-structured representations

Architecture:
- Layers 1-N/2: Full softmax attention (dynamic feature extraction)
- Layers N/2+1-N: Static MLP OR linear attention (predictable transformations)

This should give us:
- Quality of softmax (from early layers doing the heavy lifting)
- Speed of linear/MLP (from later layers being simple)
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
    """Full softmax attention for early layers."""

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


class StaticMLP(nn.Module):
    """
    Static MLP replacement for attention in later layers.

    Per the Static Function Hypothesis: later layers perform predictable
    transformations on already-structured representations. A simple MLP
    can learn this static function.
    """

    def __init__(self, hidden_dim, expansion_factor=2):
        super().__init__()
        intermediate = hidden_dim * expansion_factor

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, intermediate),
            nn.GELU(),
            nn.Linear(intermediate, hidden_dim),
        )

    def forward(self, x):
        # Per-token transformation (no cross-token interaction needed
        # because representations are already structured)
        return self.mlp(x)


class LinearAttentionWithGate(nn.Module):
    """
    Linear attention with learned gating for later layers.

    An alternative to static MLP: maintains some cross-token interaction
    but with O(L) complexity instead of O(L²).
    """

    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim

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

        q = self.q_proj(x).view(B, L, H, d)
        k = self.k_proj(x).view(B, L, H, d)
        v = self.v_proj(x).view(B, L, H, d)

        # Apply feature map
        q = self.feature_map(q)
        k = self.feature_map(k)

        # Compute gates
        gates = self.gate(x)  # (B, L, H)

        # Efficient chunked linear attention
        chunk_size = 64
        num_chunks = (L + chunk_size - 1) // chunk_size

        # Pad if necessary
        pad_len = num_chunks * chunk_size - L
        if pad_len > 0:
            q = F.pad(q, (0, 0, 0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, 0, 0, pad_len))
            gates = F.pad(gates, (0, 0, 0, pad_len))

        L_padded = q.shape[1]

        # Reshape to chunks
        q = q.view(B, num_chunks, chunk_size, H, d)
        k = k.view(B, num_chunks, chunk_size, H, d)
        v = v.view(B, num_chunks, chunk_size, H, d)
        gates = gates.view(B, num_chunks, chunk_size, H)

        outputs = []
        kv_state = torch.zeros(B, H, d, d, device=x.device)
        k_state = torch.zeros(B, H, d, device=x.device)

        for i in range(num_chunks):
            q_c = q[:, i]  # (B, chunk_size, H, d)
            k_c = k[:, i]
            v_c = v[:, i]
            g_c = gates[:, i]  # (B, chunk_size, H)

            # Within-chunk: simple linear attention
            q_c_t = q_c.transpose(1, 2)  # (B, H, chunk_size, d)
            k_c_t = k_c.transpose(1, 2)
            v_c_t = v_c.transpose(1, 2)

            # Within-chunk attention (causal)
            within_scores = torch.matmul(q_c_t, k_c_t.transpose(-2, -1))
            causal_mask = torch.triu(torch.ones(chunk_size, chunk_size, device=x.device), diagonal=1).bool()
            within_scores = within_scores.masked_fill(causal_mask, 0)
            within_norm = within_scores.sum(dim=-1, keepdim=True) + self.eps
            within_out = torch.matmul(within_scores / within_norm, v_c_t)

            # Cross-chunk: use accumulated state
            cross_out = torch.einsum('bhcd,bhde->bhce', q_c_t, kv_state)
            cross_norm = torch.einsum('bhcd,bhd->bhc', q_c_t, k_state).unsqueeze(-1) + self.eps
            cross_out = cross_out / cross_norm

            # Combine
            out = within_out + 0.5 * cross_out
            outputs.append(out.transpose(1, 2))

            # Update state with gating
            chunk_gate = g_c.mean(dim=1)  # (B, H)
            k_mean = k_c.mean(dim=1)  # (B, H, d)
            v_mean = v_c.mean(dim=1)

            gate_expanded = chunk_gate.unsqueeze(-1).unsqueeze(-1)
            kv_state = 0.9 * kv_state + gate_expanded * torch.einsum('bhd,bhe->bhde', k_mean, v_mean)
            k_state = 0.9 * k_state + chunk_gate.unsqueeze(-1) * k_mean

        out = torch.cat(outputs, dim=1)  # (B, L_padded, H, d)

        # Remove padding
        if pad_len > 0:
            out = out[:, :L]

        out = out.reshape(B, L if pad_len == 0 else L_padded - pad_len, D)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    """Transformer block with configurable attention type."""

    def __init__(self, hidden_dim, num_heads, intermediate_dim, dropout, attn_type='softmax'):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_dim)

        if attn_type == 'softmax':
            self.attention = SoftmaxAttention(hidden_dim, num_heads)
        elif attn_type == 'linear':
            self.attention = LinearAttentionWithGate(hidden_dim, num_heads)
        elif attn_type == 'mlp':
            self.attention = StaticMLP(hidden_dim)
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


class StaticFunctionHybridTransformer(nn.Module):
    """
    Hybrid Transformer implementing the Static Function Hypothesis.

    Early layers: Full softmax attention (dynamic feature extraction)
    Later layers: Static MLP or linear attention (predictable transformations)
    """

    def __init__(self, vocab_size, hidden_dim=512, num_layers=8, num_heads=8,
                 intermediate_dim=2048, max_seq_len=1024, dropout=0.1,
                 early_layers=4, late_type='mlp'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Build layers according to Static Function Hypothesis
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            if i < early_layers:
                # Early layers: full softmax attention
                attn_type = 'softmax'
            else:
                # Later layers: static function (MLP or linear)
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
    """Benchmark inference speed."""
    model.eval()

    # Warmup
    dummy = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    for _ in range(10):
        _ = model(dummy)

    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(num_runs):
        _ = model(dummy)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    tokens_per_sec = (num_runs * batch_size * seq_len) / elapsed
    return tokens_per_sec


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n" + "="*70)
    print("STATIC FUNCTION HYPOTHESIS: HYBRID TRANSFORMER")
    print("="*70)
    print("\nHypothesis: Early layers need dynamic attention, later layers")
    print("            can use static functions on structured representations.")

    print("\nLoading data...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_loader, val_loader = get_wikitext_data(tokenizer, seq_length=512, batch_size=8)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    configs = [
        ("Full Softmax (baseline)", 8, 'softmax'),      # All 8 layers softmax
        ("Hybrid: 4 softmax + 4 MLP", 4, 'mlp'),        # 4 softmax, 4 MLP
        ("Hybrid: 4 softmax + 4 linear", 4, 'linear'),  # 4 softmax, 4 linear
        ("Hybrid: 2 softmax + 6 MLP", 2, 'mlp'),        # 2 softmax, 6 MLP
        ("Hybrid: 6 softmax + 2 MLP", 6, 'mlp'),        # 6 softmax, 2 MLP
    ]

    results = []

    for name, early_layers, late_type in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"  Early layers (softmax): {early_layers}")
        print(f"  Late layers ({late_type}): {8 - early_layers}")
        print("="*60)

        model = StaticFunctionHybridTransformer(
            vocab_size=tokenizer.vocab_size,
            hidden_dim=512,
            num_layers=8,
            num_heads=8,
            intermediate_dim=2048,
            early_layers=early_layers,
            late_type=late_type if early_layers < 8 else 'softmax',
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

        # Benchmark speed
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
    print(f"{'Configuration':<35} | {'PPL':>8} | {'Speed':>12} | {'Params':>12}")
    print("-"*70)

    baseline_ppl = results[0]['ppl']
    baseline_speed = results[0]['speed']

    for r in results:
        ppl_diff = (r['ppl'] - baseline_ppl) / baseline_ppl * 100
        speed_diff = (r['speed'] - baseline_speed) / baseline_speed * 100
        print(f"{r['name']:<35} | {r['ppl']:>8.1f} | {r['speed']:>8.0f} t/s | {r['params']:>10,}")
        if r['name'] != results[0]['name']:
            print(f"{'':35} | {ppl_diff:>+7.1f}% | {speed_diff:>+7.1f}%")

    print("\n" + "="*70)
    print("COMPARISON TO PREVIOUS RESULTS")
    print("="*70)
    print(f"  Linear attention (learned gate, all layers): PPL 167.4")
    print(f"  Pure softmax baseline:                       PPL 173.4")
    print(f"  Best hybrid from this experiment:            PPL {min(r['ppl'] for r in results):.1f}")


if __name__ == "__main__":
    main()
