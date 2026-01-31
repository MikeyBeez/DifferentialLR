#!/usr/bin/env python3
"""
Golden Ratio Attention: Using φ as the deterministic binding constant.

Hypothesis: If attention is geometric binding (not similarity search),
we can use the golden ratio φ ≈ 1.618 as a fixed scaling factor.

φ is the "most irrational" number - it creates the least collisions
when summing scaled vectors because it's maximally incommensurable.

address = Q + φ·K  (instead of Q·K similarity)
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

# The Golden Ratio
PHI = (1 + math.sqrt(5)) / 2  # ≈ 1.618033988749895


class StandardAttention(nn.Module):
    """Standard Q·K softmax attention (baseline)."""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.o_proj(out)


class GoldenRatioAttention(nn.Module):
    """
    Attention using golden ratio binding.

    Instead of: score = Q · K^T (dot product similarity)
    We use:     address = Q + φ·K (golden ratio binding)

    The "attention weight" comes from how well V aligns with this address.
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        # Address-to-score projection (learns to interpret the binding)
        self.score_proj = nn.Linear(self.head_dim, 1)

    def forward(self, x):
        B, N, D = x.shape

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Golden ratio binding: address[i,j] = q[i] + φ·k[j]
        # Shape: [B, H, N_q, N_k, head_dim]
        q_expanded = q.unsqueeze(3)  # [B, H, N, 1, D]
        k_expanded = k.unsqueeze(2)  # [B, H, 1, N, D]
        addresses = q_expanded + PHI * k_expanded  # [B, H, N, N, D]

        # Convert addresses to scores
        scores = self.score_proj(addresses).squeeze(-1)  # [B, H, N, N]
        scores = scores * self.scale

        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.o_proj(out)


class GoldenRatioAttentionV2(nn.Module):
    """
    Golden ratio binding with dot-product retrieval.

    address = Q + φ·K
    score = address · V  (how well does V match this address?)
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Golden ratio binding
        q_expanded = q.unsqueeze(3)  # [B, H, N, 1, D]
        k_expanded = k.unsqueeze(2)  # [B, H, 1, N, D]
        addresses = q_expanded + PHI * k_expanded  # [B, H, N, N, D]

        # Score = address · V (retrieval by alignment)
        v_expanded = v.unsqueeze(2)  # [B, H, 1, N, D]
        scores = (addresses * v_expanded).sum(dim=-1)  # [B, H, N, N]
        scores = scores * self.scale

        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.o_proj(out)


class GoldenRatioLinear(nn.Module):
    """
    O(N) Golden ratio attention using state accumulation.

    Instead of computing N² addresses, accumulate K·V into state,
    then query with Q.

    state = Σ (φ^i · k_i) ⊗ v_i  (position-weighted binding)
    output = q · state
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Position weights using powers of φ (normalized to prevent explosion)
        positions = torch.arange(N, device=x.device, dtype=x.dtype)
        # Use φ^(-i) to keep weights bounded, or use modular arithmetic
        pos_weights = (PHI ** (-positions / N)).view(1, 1, N, 1)  # [1, 1, N, 1]

        # Weight keys by position
        k_weighted = k * pos_weights  # [B, H, N, D]

        # Accumulate K^T @ V (the "binding state")
        # This is O(N) - no N² matrix
        kv_state = torch.einsum('bhnd,bhnv->bhdv', k_weighted, v)  # [B, H, D, D]

        # Query the state
        out = torch.einsum('bhnd,bhdv->bhnv', q, kv_state)  # [B, H, N, D]

        # Normalize
        k_sum = k_weighted.sum(dim=2, keepdim=True)  # [B, H, 1, D]
        normalizer = (q * k_sum).sum(dim=-1, keepdim=True).clamp(min=1e-6)
        out = out / normalizer

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.o_proj(out)


class GoldenShiftAttention(nn.Module):
    """
    Combine golden ratio scaling with circular shift.

    address = Q + φ·roll(K, shift)

    The shift preserves order (AB ≠ BA), φ prevents collisions.
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.shift = self.head_dim // 4  # Shift by 1/4 of head dimension

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Shift K (preserves order information)
        k_shifted = torch.roll(k, shifts=self.shift, dims=-1)

        # Golden ratio binding with shifted K
        q_expanded = q.unsqueeze(3)
        k_expanded = k_shifted.unsqueeze(2)
        addresses = q_expanded + PHI * k_expanded

        # Score via learned projection
        # Use magnitude of address as score (simpler than another projection)
        scores = addresses.norm(dim=-1) * self.scale

        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.o_proj(out)


class CircularGoldenAttention(nn.Module):
    """
    FFT-based Circular Correlation Attention (from Gemini/HRR theory).

    Uses Fast Fourier Transform to perform binding in O(N log N).
    No N² attention matrix is ever created.

    This is "Holographic Reduced Representations" applied to attention.
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Transform to frequency domain
        q_freq = torch.fft.rfft(q, dim=-1)
        k_freq = torch.fft.rfft(k, dim=-1)

        # Circular correlation: Q ⊛ K = IFFT(FFT(Q) * conj(FFT(K)))
        # Scale by φ for deterministic uniqueness
        binding_freq = q_freq * torch.conj(k_freq) * PHI

        # Transform back to spatial domain
        # This gives us "addresses" - one per position
        addresses = torch.fft.irfft(binding_freq, n=self.head_dim, dim=-1)

        # Gate V using the addresses (no softmax needed!)
        gate = torch.sigmoid(addresses)
        out = gate * v

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.o_proj(out)


class CircularGoldenAttentionV2(nn.Module):
    """
    FFT Circular Attention with cross-position binding.

    The original CircularGoldenAttention binds Q[i] with K[i] (same position).
    This version binds each Q[i] with the accumulated K state (all positions).
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Accumulate K into a "memory state" (sum across positions)
        # This is the HRR "bundling" operation
        k_state = k.sum(dim=2, keepdim=True)  # [B, H, 1, D]
        v_state = v.sum(dim=2, keepdim=True)  # [B, H, 1, D]

        # FFT binding: each Q binds with the accumulated K state
        q_freq = torch.fft.rfft(q, dim=-1)
        k_state_freq = torch.fft.rfft(k_state, dim=-1)

        # Circular correlation with state
        binding_freq = q_freq * torch.conj(k_state_freq) * PHI
        addresses = torch.fft.irfft(binding_freq, n=self.head_dim, dim=-1)

        # Use addresses to weight retrieval from V state
        # Expand v_state to match positions
        gate = torch.sigmoid(addresses)

        # Mix position-specific V with global V state
        out = gate * v + (1 - gate) * v_state.expand(-1, -1, N, -1)

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.o_proj(out)


class SimpleTransformer(nn.Module):
    """Minimal transformer for testing attention variants."""
    def __init__(self, vocab_size, dim, num_layers, num_heads, attention_type='standard'):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(1024, dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if attention_type == 'standard':
                attn = StandardAttention(dim, num_heads)
            elif attention_type == 'golden_ratio':
                attn = GoldenRatioAttention(dim, num_heads)
            elif attention_type == 'golden_ratio_v2':
                attn = GoldenRatioAttentionV2(dim, num_heads)
            elif attention_type == 'golden_linear':
                attn = GoldenRatioLinear(dim, num_heads)
            elif attention_type == 'golden_shift':
                attn = GoldenShiftAttention(dim, num_heads)
            elif attention_type == 'circular_golden':
                attn = CircularGoldenAttention(dim, num_heads)
            elif attention_type == 'circular_golden_v2':
                attn = CircularGoldenAttentionV2(dim, num_heads)
            else:
                raise ValueError(f"Unknown attention type: {attention_type}")

            self.layers.append(nn.ModuleDict({
                'attn': attn,
                'norm1': nn.LayerNorm(dim),
                'ffn': nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim)
                ),
                'norm2': nn.LayerNorm(dim)
            }))

        self.final_norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids, labels=None):
        B, N = input_ids.shape
        pos = torch.arange(N, device=input_ids.device).unsqueeze(0)

        x = self.embed(input_ids) + self.pos_embed(pos)

        for layer in self.layers:
            normed = layer['norm1'](x)
            attn_out = layer['attn'](normed)
            x = x + attn_out

            normed = layer['norm2'](x)
            x = x + layer['ffn'](normed)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        result = {'logits': logits}
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            result['loss'] = loss

        return result


def get_wikitext_data(tokenizer, seq_length=256, batch_size=16):
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

    return (DataLoader(create_sequences(all_tokens, seq_length), batch_size=batch_size, shuffle=True),
            DataLoader(create_sequences(val_tokens, seq_length), batch_size=batch_size, shuffle=False))


def train_and_eval(model, train_loader, val_loader, device, epochs=5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out = model(batch, labels=batch)
                loss = out['loss']
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                with torch.amp.autocast('cuda'):
                    out = model(batch, labels=batch)
                val_loss += out['loss'].item()

        val_ppl = math.exp(val_loss / len(val_loader))
        print(f"  Epoch {epoch+1}: Train {math.exp(total_loss/len(train_loader)):.1f}, Val {val_ppl:.1f}")

    return val_ppl


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n" + "=" * 70)
    print("GOLDEN RATIO ATTENTION TEST")
    print("=" * 70)
    print(f"\nUsing φ = {PHI:.10f} as binding constant")
    print("Hypothesis: φ·K binding can replace Q·K similarity")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_loader, val_loader = get_wikitext_data(tokenizer)

    results = {}

    attention_types = [
        ('standard', 'Standard Q·K Attention'),
        ('golden_ratio', 'Golden Ratio (Q + φ·K → learned score)'),
        ('golden_ratio_v2', 'Golden Ratio V2 (Q + φ·K → dot V)'),
        ('golden_linear', 'Golden Linear (O(N) state accumulation)'),
        ('golden_shift', 'Golden Shift (Q + φ·roll(K))'),
        ('circular_golden', 'Circular FFT (O(N) HRR binding)'),
        ('circular_golden_v2', 'Circular FFT V2 (state accumulation)'),
    ]

    for attn_type, name in attention_types:
        print(f"\n{'='*50}")
        print(f"{name}")
        print('='*50)

        torch.manual_seed(42)
        torch.cuda.empty_cache()

        model = SimpleTransformer(
            vocab_size=tokenizer.vocab_size,
            dim=256,
            num_layers=4,
            num_heads=4,
            attention_type=attn_type
        ).to(device)

        results[name] = train_and_eval(model, train_loader, val_loader, device, epochs=5)
        del model

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for name, ppl in results.items():
        print(f"  {name:45s}: {ppl:.1f} PPL")

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    standard_ppl = results['Standard Q·K Attention']
    golden_ppl = results['Golden Ratio (Q + φ·K → learned score)']

    if golden_ppl < standard_ppl * 1.5:
        print(f"\nGolden ratio binding is competitive!")
        print(f"  Standard: {standard_ppl:.1f} PPL")
        print(f"  Golden:   {golden_ppl:.1f} PPL")
        print(f"\n→ The 'similarity' in Q·K is not essential")
        print(f"→ Deterministic binding with φ can work")
    else:
        print(f"\nStandard attention still wins:")
        print(f"  Standard: {standard_ppl:.1f} PPL")
        print(f"  Golden:   {golden_ppl:.1f} PPL")
        print(f"\n→ Q·K dot product provides something φ-binding doesn't")


if __name__ == "__main__":
    main()
