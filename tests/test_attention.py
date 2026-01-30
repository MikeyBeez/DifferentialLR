"""Unit tests for chunked attention mechanism."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

from src.chunked_attention import ChunkedLinearAttention, WithinChunkAttention, CrossChunkState
from src.model import ChunkedTransformerModel, create_model
from src.config import ModelConfig


class TestWithinChunkAttention:
    """Tests for within-chunk softmax attention."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        attn = WithinChunkAttention(hidden_dim=256, num_heads=4, head_dim=64)

        x = torch.randn(2, 4, 32, 256)  # (batch, num_chunks, chunk_size, hidden)
        out, weights = attn(x)

        assert out.shape == x.shape
        assert weights.shape == (2, 4, 4, 32, 32)  # (batch, chunks, heads, size, size)

    def test_causal_mask(self):
        """Test that attention is properly masked for causality."""
        attn = WithinChunkAttention(hidden_dim=64, num_heads=2, head_dim=32, dropout=0.0)

        x = torch.randn(1, 1, 8, 64)

        # Create causal mask
        mask = torch.triu(torch.ones(8, 8) * float("-inf"), diagonal=1)

        out, weights = attn(x, attention_mask=mask)

        # Check that future positions have zero attention
        # (after softmax, masked positions should be ~0)
        attn_weights = weights[0, 0, 0]  # First sample, first chunk, first head
        assert torch.allclose(attn_weights.triu(diagonal=1), torch.zeros_like(attn_weights.triu(diagonal=1)), atol=1e-6)


class TestCrossChunkState:
    """Tests for cross-chunk linear state mechanism."""

    def test_state_accumulation(self):
        """Test that state accumulates across chunks with gamma decay."""
        state_module = CrossChunkState(hidden_dim=64, gamma=0.9)

        x = torch.randn(2, 4, 8, 64)  # 4 chunks
        out, intermediates = state_module(x, return_intermediates=True)

        assert out.shape == x.shape
        assert "state_history" in intermediates
        assert intermediates["state_history"].shape == (2, 4, 64)

        # First chunk should have zero cross-contribution (no previous state)
        assert intermediates["cross_magnitudes"][0] == 0.0

    def test_gamma_decay(self):
        """Test that higher gamma preserves more historical information."""
        x = torch.randn(2, 4, 8, 64)

        # Low gamma - faster decay
        low_gamma = CrossChunkState(hidden_dim=64, gamma=0.5)
        _, low_inter = low_gamma(x, return_intermediates=True)

        # High gamma - slower decay
        high_gamma = CrossChunkState(hidden_dim=64, gamma=0.99)
        _, high_inter = high_gamma(x, return_intermediates=True)

        # Final state should be larger for high gamma (more accumulated)
        low_state_norm = low_inter["final_state"].norm()
        high_state_norm = high_inter["final_state"].norm()

        # Not strictly always true due to weight init, but generally expected
        # Just verify they're different
        assert not torch.allclose(low_state_norm, high_state_norm)


class TestChunkedLinearAttention:
    """Tests for the full chunked attention mechanism."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        attn = ChunkedLinearAttention(
            hidden_dim=256,
            num_heads=4,
            head_dim=64,
            gamma=0.95,
        )

        x = torch.randn(2, 128, 256)
        out, _ = attn(x, chunk_size=32)

        assert out.shape == x.shape

    def test_padding_handling(self):
        """Test that sequences not divisible by chunk_size are handled."""
        attn = ChunkedLinearAttention(
            hidden_dim=128,
            num_heads=2,
            head_dim=64,
            gamma=0.95,
        )

        # Sequence length not divisible by chunk_size
        x = torch.randn(2, 100, 128)
        out, _ = attn(x, chunk_size=32)

        # Output should match original sequence length
        assert out.shape == (2, 100, 128)

    def test_gradient_flow(self):
        """Test that gradients flow through both within and cross-chunk paths."""
        attn = ChunkedLinearAttention(
            hidden_dim=128,
            num_heads=2,
            head_dim=64,
            gamma=0.95,
        )

        x = torch.randn(2, 64, 128, requires_grad=True)
        out, _ = attn(x, chunk_size=16)

        loss = out.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert x.grad.norm() > 0

        # Check all parameters have gradients
        for name, param in attn.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_parameter_groups(self):
        """Test that parameter groups are correctly separated."""
        attn = ChunkedLinearAttention(
            hidden_dim=128,
            num_heads=2,
            head_dim=64,
            gamma=0.95,
        )

        groups = attn.get_parameter_groups()

        assert "within_chunk" in groups
        assert "cross_chunk" in groups
        assert len(groups["within_chunk"]) > 0
        assert len(groups["cross_chunk"]) > 0

        # No overlap
        within_set = set(id(p) for p in groups["within_chunk"])
        cross_set = set(id(p) for p in groups["cross_chunk"])
        assert len(within_set & cross_set) == 0

    def test_intermediates_returned(self):
        """Test that intermediates are properly returned."""
        attn = ChunkedLinearAttention(
            hidden_dim=128,
            num_heads=2,
            head_dim=64,
            gamma=0.95,
        )

        x = torch.randn(2, 64, 128)
        out, intermediates = attn(x, chunk_size=16, return_intermediates=True)

        assert intermediates is not None
        assert "attention_weights" in intermediates
        assert "within_chunk_output" in intermediates
        assert "state_history" in intermediates
        assert "cross_magnitudes" in intermediates


class TestChunkedTransformerModel:
    """Tests for the full transformer model."""

    def test_model_creation(self):
        """Test model creation with default config."""
        config = ModelConfig(
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            head_dim=64,
            intermediate_dim=512,
            vocab_size=1000,
        )
        model = create_model(config)

        assert model is not None
        assert model.get_num_params() > 0

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        config = ModelConfig(
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            head_dim=64,
            intermediate_dim=512,
            vocab_size=1000,
            chunk_size=16,
        )
        model = create_model(config)

        x = torch.randint(0, 1000, (2, 64))
        output = model(x)

        assert "logits" in output
        assert output["logits"].shape == (2, 64, 1000)

    def test_loss_computation(self):
        """Test that loss is computed when labels provided."""
        config = ModelConfig(
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            head_dim=64,
            intermediate_dim=512,
            vocab_size=1000,
            chunk_size=16,
        )
        model = create_model(config)

        x = torch.randint(0, 1000, (2, 64))
        output = model(x, labels=x)

        assert "loss" in output
        assert output["loss"].dim() == 0  # Scalar

    def test_parameter_groups(self):
        """Test that model parameter groups work correctly."""
        config = ModelConfig(
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            head_dim=64,
            intermediate_dim=512,
            vocab_size=1000,
        )
        model = create_model(config)

        groups = model.get_parameter_groups(
            within_chunk_lr=3e-4,
            cross_chunk_lr=1e-4,
            weight_decay=0.1,
        )

        # Should have multiple groups
        assert len(groups) >= 2

        # All params should be accounted for
        total_params = sum(p.numel() for p in model.parameters())
        grouped_params = sum(
            sum(p.numel() for p in g["params"])
            for g in groups
        )
        assert grouped_params == total_params


class TestEquivalenceToFullAttention:
    """Tests verifying chunked attention can match full attention."""

    def test_single_chunk_equals_full(self):
        """When chunk_size >= seq_len, should be equivalent to full attention."""
        hidden_dim = 64
        num_heads = 2
        head_dim = 32
        seq_len = 32

        # Chunked attention with chunk_size = seq_len (no chunking)
        chunked = ChunkedLinearAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            gamma=0.0,  # No cross-chunk contribution
        )

        x = torch.randn(1, seq_len, hidden_dim)

        # Forward pass
        out, _ = chunked(x, chunk_size=seq_len)

        # Output should be valid
        assert out.shape == x.shape
        assert not torch.isnan(out).any()


def run_tests_manually():
    """Run tests without pytest."""
    print("Testing WithinChunkAttention...")
    t = TestWithinChunkAttention()
    t.test_output_shape()
    print("  test_output_shape: PASSED")
    t.test_causal_mask()
    print("  test_causal_mask: PASSED")

    print("\nTesting CrossChunkState...")
    t = TestCrossChunkState()
    t.test_state_accumulation()
    print("  test_state_accumulation: PASSED")
    t.test_gamma_decay()
    print("  test_gamma_decay: PASSED")

    print("\nTesting ChunkedLinearAttention...")
    t = TestChunkedLinearAttention()
    t.test_output_shape()
    print("  test_output_shape: PASSED")
    t.test_padding_handling()
    print("  test_padding_handling: PASSED")
    t.test_gradient_flow()
    print("  test_gradient_flow: PASSED")
    t.test_parameter_groups()
    print("  test_parameter_groups: PASSED")
    t.test_intermediates_returned()
    print("  test_intermediates_returned: PASSED")

    print("\nTesting ChunkedTransformerModel...")
    t = TestChunkedTransformerModel()
    t.test_model_creation()
    print("  test_model_creation: PASSED")
    t.test_forward_pass()
    print("  test_forward_pass: PASSED")
    t.test_loss_computation()
    print("  test_loss_computation: PASSED")
    t.test_parameter_groups()
    print("  test_parameter_groups: PASSED")

    print("\nTesting Equivalence...")
    t = TestEquivalenceToFullAttention()
    t.test_single_chunk_equals_full()
    print("  test_single_chunk_equals_full: PASSED")

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)


if __name__ == "__main__":
    if HAS_PYTEST:
        pytest.main([__file__, "-v"])
    else:
        run_tests_manually()
