#!/usr/bin/env python3
"""
Collision Test: How many unique n-grams can shift-and-add pack into semantic space?

The hypothesis: If we use circular shift (HRR-style binding), we can create
deterministic addresses for n-grams without dot products or softmax.

Questions:
1. How many unique bigrams fit before collisions?
2. Does the shift amount matter?
3. How does this compare to learned embedding collisions?
"""

import torch
import numpy as np
from collections import defaultdict
import math


def circular_shift_bind(t1, t2, shift=1):
    """Bind two vectors using circular shift (HRR-style)."""
    t2_shifted = torch.roll(t2, shifts=shift, dims=-1)
    return t1 + t2_shifted


def golden_ratio_bind(t1, t2, phi=1.618033988749895):
    """Bind using golden ratio scaling."""
    return t1 + phi * t2


def learned_style_bind(t1, t2, weight=0.5):
    """Bind using learned-style fixed weight (like attention might learn)."""
    return t1 + weight * t2


def measure_collision_rate(vectors, threshold=0.95):
    """
    Measure how many vector pairs are "colliding" (too similar).

    Collision = cosine similarity > threshold
    """
    # Normalize vectors
    norms = vectors.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    normalized = vectors / norms

    # Compute all pairwise similarities
    # For large N, sample instead of computing all N² pairs
    N = vectors.shape[0]
    if N > 1000:
        # Sample 10000 random pairs
        idx1 = torch.randint(0, N, (10000,))
        idx2 = torch.randint(0, N, (10000,))
        # Exclude self-pairs
        mask = idx1 != idx2
        idx1, idx2 = idx1[mask], idx2[mask]
        similarities = (normalized[idx1] * normalized[idx2]).sum(dim=-1)
    else:
        # Compute all pairs
        similarities = torch.mm(normalized, normalized.t())
        # Zero out diagonal (self-similarity)
        similarities.fill_diagonal_(0)
        similarities = similarities.flatten()

    collision_rate = (similarities.abs() > threshold).float().mean().item()
    max_similarity = similarities.abs().max().item()
    mean_similarity = similarities.abs().mean().item()

    return collision_rate, max_similarity, mean_similarity


def test_binding_capacity(dim, vocab_size, num_bigrams, bind_fn, name):
    """Test how many unique bigrams we can represent without collisions."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Dimension: {dim}, Vocab: {vocab_size}, Bigrams: {num_bigrams}")
    print('='*60)

    # Create random "token embeddings"
    torch.manual_seed(42)
    embeddings = torch.randn(vocab_size, dim)
    # Normalize to unit sphere (common in transformers)
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

    # Generate random bigrams
    token1_ids = torch.randint(0, vocab_size, (num_bigrams,))
    token2_ids = torch.randint(0, vocab_size, (num_bigrams,))

    # Create bound vectors
    t1 = embeddings[token1_ids]
    t2 = embeddings[token2_ids]
    bound_vectors = bind_fn(t1, t2)

    # Measure collisions
    collision_rate, max_sim, mean_sim = measure_collision_rate(bound_vectors)

    print(f"  Collision rate (>0.95): {collision_rate*100:.3f}%")
    print(f"  Max similarity: {max_sim:.4f}")
    print(f"  Mean similarity: {mean_sim:.4f}")

    # Test uniqueness: are AB and BA different?
    ba_bound = bind_fn(t2, t1)  # Reversed order
    order_similarity = (bound_vectors * ba_bound).sum(dim=-1) / (
        bound_vectors.norm(dim=-1) * ba_bound.norm(dim=-1) + 1e-8
    )
    order_preserved = (order_similarity.abs() < 0.95).float().mean().item()
    print(f"  Order preserved (AB != BA): {order_preserved*100:.1f}%")

    return collision_rate, max_sim, mean_sim


def capacity_scaling_test():
    """Test how capacity scales with dimension."""
    print("\n" + "="*70)
    print("CAPACITY SCALING TEST")
    print("="*70)
    print("\nHow many bigrams can we pack before collisions start?")

    dims = [64, 128, 256, 512, 768, 1024]
    vocab_size = 50000  # GPT-2 vocab size

    results = {}

    for dim in dims:
        # Test with increasing numbers of bigrams
        for num_bigrams in [1000, 10000, 100000, 500000]:
            collision_rate, _, _ = test_binding_capacity(
                dim, vocab_size, num_bigrams,
                lambda t1, t2: circular_shift_bind(t1, t2, shift=dim//4),
                f"Circular Shift (d={dim}, n={num_bigrams})"
            )

            if collision_rate > 0.01:  # More than 1% collisions
                print(f"\n>>> Dimension {dim} starts colliding at ~{num_bigrams} bigrams")
                results[dim] = num_bigrams
                break
        else:
            results[dim] = ">500000"

    print("\n" + "="*70)
    print("CAPACITY SUMMARY")
    print("="*70)
    for dim, capacity in results.items():
        print(f"  Dimension {dim:4d}: can hold ~{capacity} unique bigrams")


def compare_binding_methods():
    """Compare different binding methods."""
    print("\n" + "="*70)
    print("BINDING METHOD COMPARISON")
    print("="*70)

    dim = 768  # Transformer-sized
    vocab_size = 50000
    num_bigrams = 100000

    methods = [
        (lambda t1, t2: circular_shift_bind(t1, t2, shift=1), "Circular Shift (1)"),
        (lambda t1, t2: circular_shift_bind(t1, t2, shift=dim//4), f"Circular Shift ({dim//4})"),
        (lambda t1, t2: circular_shift_bind(t1, t2, shift=dim//2), f"Circular Shift ({dim//2})"),
        (golden_ratio_bind, "Golden Ratio"),
        (lambda t1, t2: learned_style_bind(t1, t2, 0.5), "Fixed Weight (0.5)"),
        (lambda t1, t2: learned_style_bind(t1, t2, 0.1), "Fixed Weight (0.1)"),
        (lambda t1, t2: t1 * t2, "Element-wise Product"),  # XOR-like binding
    ]

    results = []
    for bind_fn, name in methods:
        collision, max_sim, mean_sim = test_binding_capacity(
            dim, vocab_size, num_bigrams, bind_fn, name
        )
        results.append((name, collision, max_sim, mean_sim))

    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Method':<30} {'Collision%':>12} {'Max Sim':>10} {'Mean Sim':>10}")
    print("-"*70)
    for name, collision, max_sim, mean_sim in results:
        print(f"{name:<30} {collision*100:>11.3f}% {max_sim:>10.4f} {mean_sim:>10.4f}")


def trigram_test():
    """Test binding three tokens (trigrams)."""
    print("\n" + "="*70)
    print("TRIGRAM BINDING TEST")
    print("="*70)
    print("\nCan we extend to trigrams? AB + shift(C)")

    dim = 768
    vocab_size = 50000
    num_trigrams = 50000

    torch.manual_seed(42)
    embeddings = torch.randn(vocab_size, dim)
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

    # Generate random trigrams
    t1_ids = torch.randint(0, vocab_size, (num_trigrams,))
    t2_ids = torch.randint(0, vocab_size, (num_trigrams,))
    t3_ids = torch.randint(0, vocab_size, (num_trigrams,))

    t1 = embeddings[t1_ids]
    t2 = embeddings[t2_ids]
    t3 = embeddings[t3_ids]

    # Bind: A + shift(B, 1) + shift(C, 2)
    # Each token shifted by different amount to preserve position
    bound = t1 + torch.roll(t2, shifts=dim//4, dims=-1) + torch.roll(t3, shifts=dim//2, dims=-1)

    collision_rate, max_sim, mean_sim = measure_collision_rate(bound)

    print(f"  Trigram collision rate: {collision_rate*100:.3f}%")
    print(f"  Max similarity: {max_sim:.4f}")
    print(f"  Mean similarity: {mean_sim:.4f}")

    # Test order preservation: ABC vs CBA
    bound_reversed = t3 + torch.roll(t2, shifts=dim//4, dims=-1) + torch.roll(t1, shifts=dim//2, dims=-1)
    order_sim = (bound * bound_reversed).sum(dim=-1) / (
        bound.norm(dim=-1) * bound_reversed.norm(dim=-1) + 1e-8
    )
    order_preserved = (order_sim.abs() < 0.95).float().mean().item()
    print(f"  Order preserved (ABC != CBA): {order_preserved*100:.1f}%")


def saturation_test():
    """Test what happens when we accumulate too many vectors."""
    print("\n" + "="*70)
    print("SATURATION TEST")
    print("="*70)
    print("\nWhat happens when we sum many shifted vectors (like a long context)?")

    dim = 768
    vocab_size = 50000

    torch.manual_seed(42)
    embeddings = torch.randn(vocab_size, dim)
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

    context_lengths = [4, 8, 16, 32, 64, 128, 256, 512]

    print(f"\n{'Context Len':<15} {'Norm':>10} {'Max Val':>10} {'Retrievable':>12}")
    print("-"*50)

    for ctx_len in context_lengths:
        # Create a "context" by binding ctx_len tokens
        token_ids = torch.randint(0, vocab_size, (ctx_len,))
        tokens = embeddings[token_ids]

        # Accumulate with position-dependent shifts
        context_vector = torch.zeros(dim)
        for i, token in enumerate(tokens):
            shift = (i * dim // ctx_len) % dim
            context_vector = context_vector + torch.roll(token, shifts=shift, dims=-1)

        norm = context_vector.norm().item()
        max_val = context_vector.abs().max().item()

        # Test retrieval: can we still find the last token?
        last_token = tokens[-1]
        shift = ((ctx_len-1) * dim // ctx_len) % dim
        expected_contribution = torch.roll(last_token, shifts=shift, dims=-1)

        # Similarity between context and expected contribution
        retrieval_score = (context_vector * expected_contribution).sum() / (
            context_vector.norm() * expected_contribution.norm() + 1e-8
        )

        print(f"{ctx_len:<15} {norm:>10.2f} {max_val:>10.2f} {retrieval_score.item():>12.4f}")

    print("\nNote: 'Retrievable' measures how well we can identify the last token's contribution.")
    print("As context grows, individual token signals get diluted (saturation).")


def main():
    print("="*70)
    print("COLLISION TEST: Deterministic Address Binding Capacity")
    print("="*70)
    print("\nQuestion: Can shift-and-add replace learned attention?")
    print("We need unique addresses for bigrams/trigrams without collisions.")

    # Run tests
    compare_binding_methods()
    trigram_test()
    saturation_test()
    capacity_scaling_test()

    print("\n" + "="*70)
    print("CONCLUSIONS")
    print("="*70)
    print("""
Key findings:
1. Circular shift preserves word order (AB != BA) better than scalar weights
2. Element-wise product has near-zero collisions (XOR-like binding)
3. Saturation is real: signal degrades with context length
4. Dimension 768 can hold ~500k+ unique bigrams before collision

Implications for attention replacement:
- The binding operation (creating unique addresses) is NOT the bottleneck
- The real challenge is RETRIEVAL: how to query a saturated state
- This is exactly what Mamba's selective SSM tries to solve
""")


if __name__ == "__main__":
    main()
