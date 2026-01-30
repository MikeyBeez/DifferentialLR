"""Full transformer model using chunked linear attention with hybrid support."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple

from .config import ModelConfig
from .chunked_attention import ChunkedLinearAttentionLayer, FullSoftmaxAttentionLayer, MambaLayer


class ChunkedTransformerBlock(nn.Module):
    """Single transformer block with chunked linear attention, full softmax, or Mamba."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        intermediate_dim: int,
        decay: float = 0.99,
        feature_type: str = "elu",
        dropout: float = 0.1,
        use_decay: bool = True,
        use_learned_gate: bool = False,
        use_softmax: bool = False,  # If True, use full softmax instead of chunked linear
        use_mamba: bool = False,    # If True, use Mamba SSM layer
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_use_ffn: bool = False,
    ):
        super().__init__()
        self.use_softmax = use_softmax
        self.use_mamba = use_mamba

        if use_mamba:
            # Mamba layer (SSM-based)
            self.layer = MambaLayer(
                hidden_dim=hidden_dim,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                intermediate_dim=intermediate_dim if mamba_use_ffn else None,
                dropout=dropout,
                use_ffn=mamba_use_ffn,
            )
        elif use_softmax:
            # Full softmax attention layer (checkpoint layer in hybrid arch)
            self.layer = FullSoftmaxAttentionLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                intermediate_dim=intermediate_dim,
                dropout=dropout,
            )
        else:
            # Chunked linear attention layer
            self.layer = ChunkedLinearAttentionLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                intermediate_dim=intermediate_dim,
                decay=decay,
                feature_type=feature_type,
                dropout=dropout,
                use_decay=use_decay,
                use_learned_gate=use_learned_gate,
            )

    def forward(
        self,
        x: torch.Tensor,
        chunk_size: int,
        return_intermediates: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        return self.layer(x, chunk_size, return_intermediates)

    def get_parameter_groups(self) -> Dict[str, list]:
        return self.layer.get_parameter_groups()


class ChunkedTransformerModel(nn.Module):
    """
    Full transformer model with chunked linear attention.

    Architecture (TinyLlama-style ~125M params):
    - Token embeddings + positional embeddings
    - N transformer blocks with chunked attention
    - Final LayerNorm
    - Language model head
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.hidden_dim)

        # Dropout for embeddings
        self.embed_dropout = nn.Dropout(config.dropout)

        # Transformer blocks (with optional hybrid architecture)
        softmax_layers = set(getattr(config, 'softmax_layers', []))
        mamba_layers = set(getattr(config, 'mamba_layers', []))
        use_learned_gate = getattr(config, 'use_learned_gate', False)

        # Mamba config
        mamba_d_state = getattr(config, 'mamba_d_state', 16)
        mamba_d_conv = getattr(config, 'mamba_d_conv', 4)
        mamba_expand = getattr(config, 'mamba_expand', 2)
        mamba_use_ffn = getattr(config, 'mamba_use_ffn', False)

        self.blocks = nn.ModuleList([
            ChunkedTransformerBlock(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                intermediate_dim=config.intermediate_dim,
                decay=config.decay,
                feature_type=config.feature_type,
                dropout=config.dropout,
                use_decay=config.use_decay,
                use_learned_gate=use_learned_gate,
                use_softmax=(i in softmax_layers),
                use_mamba=(i in mamba_layers),
                mamba_d_state=mamba_d_state,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand,
                mamba_use_ffn=mamba_use_ffn,
            )
            for i in range(config.num_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.hidden_dim)

        # Language model head (tied with token embeddings)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Storage for layer intermediates
        self._layer_intermediates = []

    def _init_weights(self, module: nn.Module):
        """Initialize weights with small normal distribution."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        return_intermediates: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass.

        Args:
            input_ids: (batch, seq_len) token IDs
            labels: Optional (batch, seq_len) labels for LM loss
            chunk_size: Chunk size for attention (uses config default if None)
            return_intermediates: Whether to return attention intermediates

        Returns:
            Dict with 'logits', optionally 'loss' and 'intermediates'
        """
        B, L = input_ids.shape
        chunk_size = chunk_size or self.config.chunk_size

        # Get embeddings
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embed_dropout(x)

        # Clear intermediates storage
        self._layer_intermediates = []

        # Pass through transformer blocks
        for block in self.blocks:
            x, intermediates = block(x, chunk_size, return_intermediates)
            if return_intermediates and intermediates is not None:
                self._layer_intermediates.append(intermediates)

        # Final norm and LM head
        x = self.final_norm(x)
        logits = self.lm_head(x)

        output = {"logits": logits}

        # Compute loss if labels provided
        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            output["loss"] = loss

        if return_intermediates:
            output["intermediates"] = self._layer_intermediates

        return output

    def get_parameter_groups(
        self,
        within_chunk_lr: float,
        cross_chunk_lr: float,
        weight_decay: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """
        Get parameter groups for optimizer with differential learning rates.

        Args:
            within_chunk_lr: LR for within-chunk components
            cross_chunk_lr: LR for cross-chunk components
            weight_decay: Weight decay coefficient

        Returns:
            List of parameter group dicts for optimizer
        """
        within_params = []
        cross_params = []

        # Embedding parameters (no weight decay)
        embed_params = list(self.token_embedding.parameters()) + \
                       list(self.position_embedding.parameters())

        # Collect from blocks
        for block in self.blocks:
            groups = block.get_parameter_groups()
            within_params.extend(groups["within_chunk"])
            cross_params.extend(groups["cross_chunk"])

        # Final norm goes to within_chunk
        within_params.extend(list(self.final_norm.parameters()))

        # Note: lm_head is tied to token_embedding

        # Separate weight decay and no weight decay params
        within_decay = [p for p in within_params if p.dim() >= 2]
        within_no_decay = [p for p in within_params if p.dim() < 2]
        cross_decay = [p for p in cross_params if p.dim() >= 2]
        cross_no_decay = [p for p in cross_params if p.dim() < 2]

        return [
            {"params": embed_params, "lr": within_chunk_lr, "weight_decay": 0.0, "name": "embeddings"},
            {"params": within_decay, "lr": within_chunk_lr, "weight_decay": weight_decay, "name": "within_chunk"},
            {"params": within_no_decay, "lr": within_chunk_lr, "weight_decay": 0.0, "name": "within_chunk_no_decay"},
            {"params": cross_decay, "lr": cross_chunk_lr, "weight_decay": weight_decay, "name": "cross_chunk"},
            {"params": cross_no_decay, "lr": cross_chunk_lr, "weight_decay": 0.0, "name": "cross_chunk_no_decay"},
        ]

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Get number of parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.position_embedding.weight.numel()
        return n_params

    def get_layer_intermediates(self) -> List[Dict[str, Any]]:
        """Get intermediates from last forward pass."""
        return self._layer_intermediates


def create_model(config: ModelConfig) -> ChunkedTransformerModel:
    """Factory function to create model from config."""
    model = ChunkedTransformerModel(config)
    return model
