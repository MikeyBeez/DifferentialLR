#!/usr/bin/env python3
"""
Test: Attention as Associative Memory Bank

Hypothesis: If Q·K is a deterministic binding operation (not similarity search),
we should be able to replace softmax-weighted retrieval with direct address lookup.

The "address" is the binding of Q and K vectors.
The "memory" is the Value vectors.

Standard attention: softmax(Q·K^T) · V  (weighted search)
Direct binding:     f(Q, K) → address → retrieve V  (direct lookup)
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

        # Standard: softmax(Q·K^T) · V
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.o_proj(out)


class DirectBindingAttention(nn.Module):
    """
    Direct Address Retrieval via Binding.

    Instead of: softmax(Q·K^T) · V
    We compute: address = bind(Q, K), then retrieve from address

    The "binding" creates a deterministic coordinate.
    We retrieve by projecting V into that coordinate system.
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        # Binding weights (learnable but small range)
        self.bind_scale = nn.Parameter(torch.tensor(0.5))

        # Address-to-retrieval projection
        self.retrieval_proj = nn.Linear(self.head_dim, self.head_dim)

    def forward(self, x):
        B, N, D = x.shape

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Binding: Each query position binds with each key position
        # address[i,j] = q[i] + bind_scale * k[j]
        # Shape: [B, heads, N_q, N_k, head_dim]
        q_expanded = q.unsqueeze(3)  # [B, H, N, 1, D]
        k_expanded = k.unsqueeze(2)  # [B, H, 1, N, D]
        addresses = q_expanded + self.bind_scale * k_expanded  # [B, H, N, N, D]

        # Project addresses to retrieval space
        addresses = self.retrieval_proj(addresses)  # [B, H, N, N, D]

        # Retrieve: dot product between address and value
        # This checks "does this address match this value's location?"
        v_expanded = v.unsqueeze(2)  # [B, H, 1, N, D]

        # Retrieval score = how well does address[i,j] match v[j]?
        retrieval_scores = (addresses * v_expanded).sum(dim=-1)  # [B, H, N, N]

        # Normalize to get weights (still need this for combination)
        weights = F.softmax(retrieval_scores * (self.head_dim ** -0.5), dim=-1)

        # Weighted combination of values
        out = torch.matmul(weights, v)

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.o_proj(out)


class HardAddressAttention(nn.Module):
    """
    Hard Address Lookup (no softmax at all).

    The binding creates an address. We use that address directly
    to index into a learned memory bank.

    This is closest to the "Associative Memory Bank" concept.
    """
    def __init__(self, dim, num_heads, num_slots=256):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_slots = num_slots

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        # Memory bank: learned slots that addresses map to
        self.memory_keys = nn.Parameter(torch.randn(num_heads, num_slots, self.head_dim) * 0.02)
        self.memory_values = nn.Parameter(torch.randn(num_heads, num_slots, self.head_dim) * 0.02)

        # Binding projection
        self.bind_proj = nn.Linear(self.head_dim * 2, self.head_dim)

    def forward(self, x):
        B, N, D = x.shape

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # For each position, create a binding address from Q and context summary
        # Context summary = mean of K (simplified)
        context = k.mean(dim=2, keepdim=True).expand(-1, -1, N, -1)  # [B, H, N, D]

        # Bind Q with context to get address
        binding_input = torch.cat([q, context], dim=-1)  # [B, H, N, 2D]
        address = self.bind_proj(binding_input)  # [B, H, N, D]

        # Look up in memory bank: find closest slot
        # address: [B, H, N, D], memory_keys: [H, slots, D]
        address_expanded = address.unsqueeze(3)  # [B, H, N, 1, D]
        keys_expanded = self.memory_keys.unsqueeze(0).unsqueeze(2)  # [1, H, 1, slots, D]

        # Distance to each memory slot
        similarity = (address_expanded * keys_expanded).sum(dim=-1)  # [B, H, N, slots]

        # Soft lookup (could be hard with argmax, but that's not differentiable)
        weights = F.softmax(similarity * (self.head_dim ** -0.5), dim=-1)  # [B, H, N, slots]

        # Retrieve from memory
        memory_out = torch.matmul(weights, self.memory_values)  # [B, H, N, D]

        # Also add the original value signal (residual path)
        # The memory provides "global patterns", V provides "local content"
        out = memory_out + v.mean(dim=2, keepdim=True).expand(-1, -1, N, -1)

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.o_proj(out)


class LinearBindingAttention(nn.Module):
    """
    Linear-time binding attention.

    If binding is deterministic, we don't need O(N²) comparisons.
    We can compute a global "binding state" and query it directly.

    Similar to linear attention but with explicit binding semantics.
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        # Feature map for linearization (elu + 1 for positivity)
        self.feature_dim = self.head_dim

    def feature_map(self, x):
        # ELU + 1 ensures positivity (required for linear attention)
        return F.elu(x) + 1

    def forward(self, x):
        B, N, D = x.shape

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply feature map (makes Q, K positive)
        q = self.feature_map(q)
        k = self.feature_map(k)

        # Linear attention: Q @ (K^T @ V) instead of (Q @ K^T) @ V
        # This is O(N) instead of O(N²)

        # Binding interpretation: K^T @ V creates a "memory state"
        # Each K[i] "binds" with its V[i] and accumulates into state
        kv = torch.einsum('bhnd,bhnv->bhdv', k, v)  # [B, H, D, D]

        # Q "queries" this bound state
        # The address (Q) retrieves from the bound memory
        out = torch.einsum('bhnd,bhdv->bhnv', q, kv)  # [B, H, N, D]

        # Normalize (equivalent to denominator in linear attention)
        k_sum = k.sum(dim=2, keepdim=True)  # [B, H, 1, D]
        normalizer = torch.einsum('bhnd,bhkd->bhn', q, k_sum).unsqueeze(-1).clamp(min=1e-6)
        out = out / normalizer

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
            elif attention_type == 'direct_binding':
                attn = DirectBindingAttention(dim, num_heads)
            elif attention_type == 'hard_address':
                attn = HardAddressAttention(dim, num_heads)
            elif attention_type == 'linear_binding':
                attn = LinearBindingAttention(dim, num_heads)
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
            # Attention
            normed = layer['norm1'](x)
            attn_out = layer['attn'](normed)
            x = x + attn_out

            # FFN
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

        # Eval
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
    print("ASSOCIATIVE MEMORY BANK TEST")
    print("=" * 70)
    print("\nHypothesis: If attention is deterministic binding (not similarity),")
    print("we can replace softmax-weighted search with direct address retrieval.")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_loader, val_loader = get_wikitext_data(tokenizer)

    results = {}

    attention_types = [
        ('standard', 'Standard Q·K Softmax'),
        ('direct_binding', 'Direct Binding (address + retrieval)'),
        ('hard_address', 'Hard Address (memory bank)'),
        ('linear_binding', 'Linear Binding (O(N) complexity)'),
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
        print(f"  {name:40s}: {ppl:.1f} PPL")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    standard_ppl = results['Standard Q·K Softmax']
    binding_ppl = results['Direct Binding (address + retrieval)']
    linear_ppl = results['Linear Binding (O(N) complexity)']

    if binding_ppl < standard_ppl * 1.2:
        print("\nDirect binding approaches standard attention!")
        print("→ Attention CAN be viewed as deterministic address lookup")
        print("→ The 'similarity' interpretation is just a convenient proxy")

    if linear_ppl < standard_ppl * 1.5:
        print("\nLinear binding is competitive!")
        print("→ O(N²) computation may not be necessary")
        print("→ Binding can be accumulated into a state (like Mamba)")


if __name__ == "__main__":
    main()
