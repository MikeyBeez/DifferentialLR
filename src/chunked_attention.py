"""Core chunked linear attention mechanism with proper KV state accumulation.

This module implements a hybrid attention mechanism:
- Within each chunk: standard softmax attention O(C²)
- Across chunks: linear attention with KV state accumulation O(L/C * d²)

The cross-chunk mechanism properly accumulates φ(K)^T @ V states and queries
them with φ(Q), following established linear attention formulations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any

# Try to import triton for fast selective scan
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def selective_scan_kernel(
        x_ptr, delta_ptr, A_ptr, B_ptr, C_ptr, out_ptr,
        batch, seq_len, d_inner, d_state,
        stride_xb, stride_xl, stride_xd,
        stride_db, stride_dl, stride_dd,
        stride_ad, stride_as,
        stride_bb, stride_bl, stride_bs,
        stride_cb, stride_cl, stride_cs,
        stride_ob, stride_ol, stride_od,
        BLOCK_D: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        """Triton kernel for selective scan - processes one (batch, d_block) slice."""
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)

        d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d_offs < d_inner
        s_offs = tl.arange(0, BLOCK_S)
        s_mask = s_offs < d_state

        h = tl.zeros((BLOCK_D, BLOCK_S), dtype=tl.float32)
        A = tl.load(A_ptr + d_offs[:, None] * stride_ad + s_offs[None, :] * stride_as,
                    mask=d_mask[:, None] & s_mask[None, :], other=0.0)

        for t in range(seq_len):
            x = tl.load(x_ptr + pid_b * stride_xb + t * stride_xl + d_offs * stride_xd,
                        mask=d_mask, other=0.0)
            delta = tl.load(delta_ptr + pid_b * stride_db + t * stride_dl + d_offs * stride_dd,
                            mask=d_mask, other=0.0)
            B = tl.load(B_ptr + pid_b * stride_bb + t * stride_bl + s_offs * stride_bs,
                        mask=s_mask, other=0.0)
            C = tl.load(C_ptr + pid_b * stride_cb + t * stride_cl + s_offs * stride_cs,
                        mask=s_mask, other=0.0)

            dA = tl.exp(delta[:, None] * A)
            dB_x = delta[:, None] * B[None, :] * x[:, None]
            h = dA * h + dB_x
            y = tl.sum(h * C[None, :], axis=1)

            tl.store(out_ptr + pid_b * stride_ob + t * stride_ol + d_offs * stride_od,
                     y, mask=d_mask)

    def selective_scan_triton(x, delta, A, B, C):
        """Fast triton-based selective scan."""
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        out = torch.empty_like(x)

        BLOCK_D = min(64, triton.next_power_of_2(d_inner))
        BLOCK_S = min(32, triton.next_power_of_2(d_state))
        grid = (batch, triton.cdiv(d_inner, BLOCK_D))

        selective_scan_kernel[grid](
            x, delta, A, B, C, out,
            batch, seq_len, d_inner, d_state,
            x.stride(0), x.stride(1), x.stride(2),
            delta.stride(0), delta.stride(1), delta.stride(2),
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1), B.stride(2),
            C.stride(0), C.stride(1), C.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_D=BLOCK_D, BLOCK_S=BLOCK_S,
        )
        return out


class FeatureMap(nn.Module):
    """Feature map for linear attention.

    Transforms Q and K before computing attention to enable
    the associative property: (QK^T)V = Q(K^T V)

    Options:
    - elu: ELU(x) + 1 (always positive, smooth)
    - relu: ReLU(x) + eps (simple, but can be sparse)
    - softmax: Softmax over head dimension (normalized)
    - identity: No transformation (for ablation)
    - l2: L2 normalization (stable, bounded dot products)
    - power2: x^2 (Taylor expansion approximation of exp)
    - power3: x^3 (sharper than power2)
    """

    def __init__(self, head_dim: int, feature_type: str = "elu", eps: float = 1e-6):
        super().__init__()
        self.head_dim = head_dim
        self.feature_type = feature_type
        self.eps = eps
        # Scale for power maps to prevent explosion
        self.power_scale = head_dim ** -0.25

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feature map.

        Args:
            x: (..., head_dim)

        Returns:
            φ(x): (..., head_dim) with non-negative values
        """
        if self.feature_type == "elu":
            return F.elu(x) + 1
        elif self.feature_type == "relu":
            return F.relu(x) + self.eps
        elif self.feature_type == "softmax":
            return F.softmax(x, dim=-1)
        elif self.feature_type == "identity":
            return x
        elif self.feature_type == "l2":
            # L2 normalization with ReLU to ensure positive values
            # This maintains bounded magnitudes while keeping positivity
            x_pos = F.relu(x) + self.eps  # Ensure positive
            normalized = x_pos / (torch.norm(x_pos, p=2, dim=-1, keepdim=True) + self.eps)
            return normalized * (self.head_dim ** 0.5)
        elif self.feature_type == "power2":
            # x^2 - approximates Taylor expansion of exp
            # Make positive first, then square
            x_pos = F.relu(x) + self.eps
            return (x_pos * self.power_scale) ** 2
        elif self.feature_type == "power3":
            # x^3 - sharper attention than power2
            x_pos = F.relu(x) + self.eps
            return (x_pos * self.power_scale) ** 3
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")


class WithinChunkAttention(nn.Module):
    """Standard multi-head softmax attention applied within each chunk."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_qkv: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """
        Args:
            x: (batch, num_chunks, chunk_size, hidden_dim)
            attention_mask: Optional causal mask
            return_qkv: Whether to return Q, K, V for cross-chunk use

        Returns:
            output: (batch, num_chunks, chunk_size, hidden_dim)
            attention_weights: (batch, num_chunks, num_heads, chunk_size, chunk_size)
            qkv: Optional tuple of (Q, K, V) each (batch, num_chunks, chunk_size, num_heads, head_dim)
        """
        B, num_chunks, chunk_size, D = x.shape

        # Reshape for parallel attention across all chunks
        x_flat = x.view(B * num_chunks, chunk_size, D)

        # Project to Q, K, V
        q = self.q_proj(x_flat).view(B * num_chunks, chunk_size, self.num_heads, self.head_dim)
        k = self.k_proj(x_flat).view(B * num_chunks, chunk_size, self.num_heads, self.head_dim)
        v = self.v_proj(x_flat).view(B * num_chunks, chunk_size, self.num_heads, self.head_dim)

        # Transpose to (batch*chunks, heads, seq, head_dim)
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * self.scale

        # Apply causal mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, v_t)

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B * num_chunks, chunk_size, -1)
        out = self.out_proj(out)

        # Reshape to chunks format
        out = out.view(B, num_chunks, chunk_size, D)
        attn_weights = attn_weights.view(B, num_chunks, self.num_heads, chunk_size, chunk_size)

        # Optionally return QKV for cross-chunk attention
        qkv = None
        if return_qkv:
            q = q.view(B, num_chunks, chunk_size, self.num_heads, self.head_dim)
            k = k.view(B, num_chunks, chunk_size, self.num_heads, self.head_dim)
            v = v.view(B, num_chunks, chunk_size, self.num_heads, self.head_dim)
            qkv = (q, k, v)

        return out, attn_weights, qkv


class CrossChunkLinearAttention(nn.Module):
    """Linear attention for cross-chunk information flow.

    Implements proper KV state accumulation with optional learned gating:
        g_i = sigmoid(gate_proj(x_i))  # Data-dependent forget gate
        S_i = g_i * S_{i-1} + φ(K_i)^T @ V_i
        cross_output_i = φ(Q_i) @ S_{i-1}

    This allows each chunk to attend to information from all previous chunks
    with O(d²) state size per head, rather than O(L) for full attention.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        feature_type: str = "elu",
        decay: float = 0.99,
        use_decay: bool = True,
        use_learned_gate: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.decay = decay
        self.use_decay = use_decay
        self.use_learned_gate = use_learned_gate

        # Feature map for linear attention
        self.feature_map = FeatureMap(head_dim, feature_type)

        # Separate projections for cross-chunk (can differ from within-chunk)
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)

        # Learned forget gate (data-dependent decay)
        if use_learned_gate:
            self.forget_gate = nn.Sequential(
                nn.Linear(hidden_dim, num_heads),
                nn.Sigmoid(),
            )
        else:
            self.forget_gate = None

        # Gating: blend within-chunk and cross-chunk outputs
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        # LayerNorm for stability
        self.cross_ln = nn.LayerNorm(hidden_dim)

        # Scale factor for KV state (prevents explosion)
        self.scale = head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        within_chunk_output: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Compute cross-chunk linear attention and combine with within-chunk output.

        Args:
            x: Original input (batch, num_chunks, chunk_size, hidden_dim)
            within_chunk_output: Output from within-chunk attention
            return_intermediates: Whether to return diagnostic info

        Returns:
            output: Combined output (batch, num_chunks, chunk_size, hidden_dim)
            intermediates: Optional diagnostic dict
        """
        B, num_chunks, chunk_size, D = x.shape
        device = x.device
        dtype = x.dtype

        # Project to Q, K, V for cross-chunk attention
        x_flat = x.view(B * num_chunks * chunk_size, D)

        q = self.q_proj(x_flat).view(B, num_chunks, chunk_size, self.num_heads, self.head_dim)
        k = self.k_proj(x_flat).view(B, num_chunks, chunk_size, self.num_heads, self.head_dim)
        v = self.v_proj(x_flat).view(B, num_chunks, chunk_size, self.num_heads, self.head_dim)

        # Apply feature map to Q and K
        q_phi = self.feature_map(q)  # (B, num_chunks, chunk_size, num_heads, head_dim)
        k_phi = self.feature_map(k)

        # Initialize KV state: (B, num_heads, head_dim, head_dim)
        kv_state = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim,
                               device=device, dtype=dtype)

        # Also track key normalizer for numerical stability
        k_state = torch.zeros(B, self.num_heads, self.head_dim, device=device, dtype=dtype)

        cross_outputs = []
        state_norms = [] if return_intermediates else None

        for i in range(num_chunks):
            # Get Q, K, V for this chunk
            q_i = q_phi[:, i]  # (B, chunk_size, num_heads, head_dim)
            k_i = k_phi[:, i]  # (B, chunk_size, num_heads, head_dim)
            v_i = v[:, i]      # (B, chunk_size, num_heads, head_dim)

            # Reshape for batch matmul: (B, num_heads, chunk_size, head_dim)
            q_i_t = q_i.permute(0, 2, 1, 3)

            # Compute cross-chunk output using PREVIOUS state
            # For first chunk, state is zero so output is zero (no prior context)
            if i == 0:
                cross_out = torch.zeros_like(q_i_t)
            else:
                # Query the state: (B, num_heads, chunk_size, head_dim)
                cross_out = torch.matmul(q_i_t, kv_state)

                # Normalize by accumulated key norm for stability
                # z = sum of φ(k) over previous chunks
                normalizer = torch.matmul(q_i_t, k_state.unsqueeze(-1)).squeeze(-1)
                normalizer = normalizer.clamp(min=1e-6)
                cross_out = cross_out / normalizer.unsqueeze(-1)

            # Reshape back: (B, chunk_size, num_heads, head_dim)
            cross_out = cross_out.permute(0, 2, 1, 3)
            cross_outputs.append(cross_out)

            if return_intermediates:
                state_norms.append(kv_state.norm().item())

            # Update state with this chunk's KV outer product
            # S = decay * S + sum_t(φ(k_t)^T @ v_t)
            k_i_t = k_i.permute(0, 2, 1, 3)  # (B, heads, chunk_size, d)
            v_i_t = v_i.permute(0, 2, 1, 3)

            # Compute chunk's KV contribution: (B, heads, d, d)
            # Scale by 1/sqrt(d) for numerical stability
            chunk_kv = torch.einsum('bhcd,bhce->bhde', k_i_t, v_i_t) * self.scale

            # Update key normalizer
            chunk_k_sum = k_i_t.sum(dim=2)  # (B, heads, d)

            # Compute decay factor (static or learned)
            if self.use_learned_gate and self.forget_gate is not None:
                # Data-dependent forget gate based on chunk content
                chunk_mean = x[:, i].mean(dim=1)  # (B, D) - mean over chunk
                gate_value = self.forget_gate(chunk_mean)  # (B, num_heads)
                # For kv_state: need (B, H, 1, 1) to broadcast with (B, H, d, d)
                gate_kv = gate_value.unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
                # For k_state: need (B, H, 1) to broadcast with (B, H, d)
                gate_k = gate_value.unsqueeze(-1)  # (B, H, 1)

                kv_state = gate_kv * kv_state + chunk_kv
                k_state = gate_k * k_state + chunk_k_sum
            elif self.use_decay:
                kv_state = self.decay * kv_state + chunk_kv
                k_state = self.decay * k_state + chunk_k_sum
            else:
                kv_state = kv_state + chunk_kv
                k_state = k_state + chunk_k_sum

        # Stack cross outputs: (B, num_chunks, chunk_size, num_heads, head_dim)
        cross_out_all = torch.stack(cross_outputs, dim=1)

        # Reshape and project: (B, num_chunks, chunk_size, hidden_dim)
        cross_out_all = cross_out_all.reshape(B, num_chunks, chunk_size, -1)
        cross_out_all = self.out_proj(cross_out_all)

        # Apply layer norm for stability
        cross_out_all = self.cross_ln(cross_out_all)

        # Gated combination of within-chunk and cross-chunk
        gate_input = torch.cat([within_chunk_output, cross_out_all], dim=-1)
        gate = self.gate(gate_input)
        output = within_chunk_output + gate * cross_out_all

        intermediates = None
        if return_intermediates:
            intermediates = {
                "state_norms": state_norms,
                "final_kv_state": kv_state.detach(),
                "cross_output_norm": cross_out_all.norm(dim=-1).mean().item(),
                "gate_mean": gate.mean().item(),
            }

        return output, intermediates


class ChunkedLinearAttention(nn.Module):
    """
    Chunked Linear Attention with proper KV state accumulation.

    Key features:
    - Within each chunk: O(C²) softmax attention for fine-grained local modeling
    - Across chunks: O(d²) linear attention state for long-range dependencies
    - Feature maps (φ) enable efficient state accumulation
    - Optional decay or learned gating for recency bias
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        decay: float = 0.99,
        feature_type: str = "elu",
        dropout: float = 0.1,
        use_decay: bool = True,
        use_learned_gate: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.decay = decay

        # Within-chunk attention (standard softmax MHA)
        self.within_chunk_attn = WithinChunkAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )

        # Cross-chunk linear attention with KV state
        self.cross_chunk_attn = CrossChunkLinearAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            feature_type=feature_type,
            decay=decay,
            use_decay=use_decay,
            use_learned_gate=use_learned_gate,
        )

        self._last_intermediates = None

    def forward(
        self,
        x: torch.Tensor,
        chunk_size: int,
        attention_mask: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Forward pass with chunked attention.

        Args:
            x: Input tensor (batch, seq_len, hidden_dim)
            chunk_size: Size of each chunk
            attention_mask: Optional attention mask
            return_intermediates: Whether to return diagnostic intermediates

        Returns:
            output: (batch, seq_len, hidden_dim)
            intermediates: Optional dict with attention weights, state history, etc.
        """
        B, L, D = x.shape
        original_len = L

        # Pad sequence if necessary
        if L % chunk_size != 0:
            pad_len = chunk_size - (L % chunk_size)
            x = F.pad(x, (0, 0, 0, pad_len))
            L = x.shape[1]

        num_chunks = L // chunk_size

        # Reshape to chunks: (B, num_chunks, chunk_size, D)
        x_chunks = x.view(B, num_chunks, chunk_size, D)

        # Create causal mask for within-chunk attention
        causal_mask = self._create_causal_mask(chunk_size, x.device, x.dtype)

        # Within-chunk attention (parallel across chunks)
        within_out, attn_weights, _ = self.within_chunk_attn(x_chunks, causal_mask)

        # Cross-chunk linear attention with KV state accumulation
        output, cross_intermediates = self.cross_chunk_attn(
            x_chunks, within_out, return_intermediates=return_intermediates
        )

        # Reshape back to sequence
        output = output.view(B, L, D)

        # Trim padding if added
        if L != original_len:
            output = output[:, :original_len, :]

        intermediates = None
        if return_intermediates:
            intermediates = {
                "attention_weights": attn_weights,
                "within_chunk_output": within_out,
                **(cross_intermediates or {}),
            }
            self._last_intermediates = intermediates

        return output, intermediates

    def _create_causal_mask(
        self,
        size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.triu(
            torch.ones(size, size, device=device, dtype=dtype) * float("-inf"),
            diagonal=1,
        )
        return mask

    def get_parameter_groups(self) -> Dict[str, list]:
        """Get parameter groups for differential learning rates."""
        within_params = list(self.within_chunk_attn.parameters())
        cross_params = list(self.cross_chunk_attn.parameters())

        return {
            "within_chunk": within_params,
            "cross_chunk": cross_params,
        }

    def get_last_intermediates(self) -> Optional[Dict[str, Any]]:
        """Get intermediates from last forward pass."""
        return self._last_intermediates


class ChunkedLinearAttentionLayer(nn.Module):
    """
    Full attention layer with pre-norm, residual connection, and FFN.
    """

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
    ):
        super().__init__()

        # Attention sublayer
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = ChunkedLinearAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            decay=decay,
            feature_type=feature_type,
            dropout=dropout,
            use_decay=use_decay,
            use_learned_gate=use_learned_gate,
        )

        # FFN sublayer
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        chunk_size: int,
        return_intermediates: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        # Attention with residual
        normed = self.attn_norm(x)
        attn_out, intermediates = self.attn(
            normed, chunk_size, return_intermediates=return_intermediates
        )
        x = x + attn_out

        # FFN with residual
        x = x + self.ffn(self.ffn_norm(x))

        return x, intermediates

    def get_parameter_groups(self) -> Dict[str, list]:
        """Get parameter groups for differential LR."""
        attn_groups = self.attn.get_parameter_groups()

        # Add FFN params to within_chunk group (local operations)
        ffn_params = list(self.ffn.parameters()) + list(self.ffn_norm.parameters())
        attn_groups["within_chunk"].extend(ffn_params)
        attn_groups["within_chunk"].extend(list(self.attn_norm.parameters()))

        return attn_groups


class FullSoftmaxAttention(nn.Module):
    """Standard causal multi-head softmax attention (no chunking).

    Used in hybrid architectures where some layers need full attention
    to "re-focus" information that gets smeared by linear attention.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            attention_mask: Optional causal mask

        Returns:
            output: (batch, seq_len, hidden_dim)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        B, L, D = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Create causal mask if not provided
        if attention_mask is None:
            attention_mask = torch.triu(
                torch.ones(L, L, device=x.device, dtype=x.dtype) * float("-inf"),
                diagonal=1,
            )

        attn_scores = attn_scores + attention_mask

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        out = self.out_proj(out)

        return out, attn_weights


class FullSoftmaxAttentionLayer(nn.Module):
    """
    Full softmax attention layer with pre-norm, residual, and FFN.

    Used as "checkpoint" layers in hybrid architectures to re-focus
    information that gets diffused by linear attention layers.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        intermediate_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Attention sublayer
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = FullSoftmaxAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )

        # FFN sublayer
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        chunk_size: int = None,  # Ignored, for API compatibility
        return_intermediates: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        # Attention with residual
        normed = self.attn_norm(x)
        attn_out, attn_weights = self.attn(normed)
        x = x + attn_out

        # FFN with residual
        x = x + self.ffn(self.ffn_norm(x))

        intermediates = None
        if return_intermediates:
            intermediates = {
                "attention_weights": attn_weights,
                "layer_type": "softmax",
            }

        return x, intermediates

    def get_parameter_groups(self) -> Dict[str, list]:
        """Get parameter groups - all params go to within_chunk for softmax layers."""
        all_params = list(self.parameters())
        return {
            "within_chunk": all_params,
            "cross_chunk": [],  # No cross-chunk params in softmax layers
        }


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (S6) - the core of Mamba.

    Unlike standard SSM with fixed A, B, C matrices, S6 makes these
    input-dependent (selective), allowing the model to focus on relevant
    inputs and filter out irrelevant ones.

    State equation: h(t) = A * h(t-1) + B * x(t)
    Output equation: y(t) = C * h(t) + D * x(t)

    Where A, B, C, delta are computed from the input.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        bias: bool = False,
        conv_bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)

        # dt_rank: rank of delta projection, set to ceil(d_model / 16)
        self.dt_rank = math.ceil(d_model / 16)

        # Input projection: d_model -> 2 * d_inner (for x and z paths)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # 1D convolution for local context (applied to x path)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,  # Depthwise convolution
            bias=conv_bias,
        )

        # Projections for selective parameters B, C, delta
        # x_proj: d_inner -> dt_rank + d_state * 2
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)

        # dt_proj: dt_rank -> d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize dt_proj bias for stability
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_min)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # Inverse of softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # A parameter - initialized with special structure
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))  # Store log for stability

        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection: d_inner -> d_model
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Input projection and split into x and z (gate) paths
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_path, z = xz.chunk(2, dim=-1)  # Each (B, L, d_inner)

        # 1D convolution on x_path for local context
        # Conv1d expects (B, C, L), so transpose
        x_path = x_path.transpose(1, 2)  # (B, d_inner, L)
        x_path = self.conv1d(x_path)[:, :, :seq_len]  # Trim to original length
        x_path = x_path.transpose(1, 2)  # (B, L, d_inner)

        # Apply SiLU activation
        x_path = F.silu(x_path)

        # Compute selective parameters B, C, delta from x_path
        x_dbl = self.x_proj(x_path)  # (B, L, dt_rank + 2*d_state)

        # Split into delta, B, C
        delta, B, C = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)

        # Project delta: (B, L, dt_rank) -> (B, L, d_inner)
        delta = self.dt_proj(delta)
        delta = F.softplus(delta)  # Ensure positive

        # Get A from log-space
        A = -torch.exp(self.A_log)  # (d_inner, d_state), negative for stability

        # Run selective scan
        y = self.selective_scan(x_path, delta, A, B, C)

        # Skip connection
        y = y + x_path * self.D

        # Apply gate (z path) with SiLU
        y = y * F.silu(z)

        # Output projection
        output = self.out_proj(y)

        return output

    def selective_scan(
        self,
        x: torch.Tensor,      # (B, L, d_inner)
        delta: torch.Tensor,  # (B, L, d_inner) - discretization step
        A: torch.Tensor,      # (d_inner, d_state) - state transition
        B: torch.Tensor,      # (B, L, d_state) - input-to-state
        C: torch.Tensor,      # (B, L, d_state) - state-to-output
    ) -> torch.Tensor:
        """
        Selective scan - uses Triton kernel if available, else PyTorch fallback.
        """
        # Use fast triton kernel if available
        if HAS_TRITON and x.is_cuda:
            return selective_scan_triton(x, delta, A, B, C)

        # PyTorch fallback
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        device, dtype = x.device, x.dtype

        # Compute discretized A: (B, L, d_inner, d_state)
        dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))

        # Compute discretized B*x: (B, L, d_inner, d_state)
        dB_x = delta.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)

        # Process in chunks for memory efficiency with cross-chunk state propagation
        chunk_size = 64 if batch <= 2 else 16
        num_chunks = (seq_len + chunk_size - 1) // chunk_size

        outputs = []
        h = torch.zeros(batch, d_inner, d_state, device=device, dtype=dtype)

        for c in range(num_chunks):
            start = c * chunk_size
            end = min(start + chunk_size, seq_len)
            chunk_len = end - start

            # Get chunk data
            dA_chunk = dA[:, start:end]      # (B, chunk_len, d_inner, d_state)
            dB_x_chunk = dB_x[:, start:end]  # (B, chunk_len, d_inner, d_state)
            C_chunk = C[:, start:end]        # (B, chunk_len, d_state)

            # Compute cumulative product of gates within chunk
            # cum_dA[t] = prod(dA[0:t]) for computing h[t] from h[-1]
            log_dA = torch.log(dA_chunk.clamp(min=1e-10))
            cum_log_dA = torch.cumsum(log_dA, dim=1)
            cum_dA = torch.exp(cum_log_dA)  # (B, chunk_len, d_inner, d_state)

            # Contribution from initial state h
            # h_contrib[t] = cum_dA[t] * h_init
            h_contrib = cum_dA * h.unsqueeze(1)  # (B, chunk_len, d_inner, d_state)

            # Contribution from inputs within chunk
            # For each t, sum over s < t of: prod(dA[s+1:t]) * dB_x[s]
            # = sum_s cum_dA[t] / cum_dA[s] * dB_x[s] (when s < t)

            # Compute input contributions using causal sum
            # Scale inputs: dB_x_scaled[s] = dB_x[s] / cum_dA[s]
            dB_x_scaled = dB_x_chunk / cum_dA.clamp(min=1e-10)

            # Cumsum gives sum of scaled inputs up to each position
            cum_dB_x_scaled = torch.cumsum(dB_x_scaled, dim=1)

            # Multiply back by cum_dA to get actual contributions
            # But we need causal: input at t doesn't contribute to h[t], only h[t+1:]
            # Shift: h[t] = cum_dA[t] * (cum_sum[t-1])
            # Use: cum_sum[t] - dB_x_scaled[t] = cum_sum[t-1]
            input_contrib = cum_dA * (cum_dB_x_scaled - dB_x_scaled)

            # Total hidden state
            h_all = h_contrib + input_contrib  # (B, chunk_len, d_inner, d_state)

            # Add current timestep input (the dB_x[t] term in h[t])
            h_all = h_all + dB_x_chunk

            # Output: y[t] = C[t] @ h[t]
            y_chunk = (h_all * C_chunk.unsqueeze(2)).sum(dim=-1)  # (B, chunk_len, d_inner)
            outputs.append(y_chunk)

            # Update h for next chunk: h = h_all[-1]
            h = h_all[:, -1]  # (B, d_inner, d_state)

        return torch.cat(outputs, dim=1)


class MambaLayer(nn.Module):
    """
    Full Mamba layer with pre-norm, residual connection, and optional FFN.

    Designed to be a drop-in replacement for attention layers in the
    hybrid architecture. Follows the Static Function Hypothesis: later
    layers can use efficient SSM instead of attention.
    """

    def __init__(
        self,
        hidden_dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        intermediate_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_ffn: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_ffn = use_ffn

        # Mamba sublayer
        self.mamba_norm = nn.LayerNorm(hidden_dim)
        self.mamba = SelectiveSSM(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.mamba_dropout = nn.Dropout(dropout)

        # Optional FFN sublayer
        if use_ffn and intermediate_dim is not None:
            self.ffn_norm = nn.LayerNorm(hidden_dim)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, intermediate_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(intermediate_dim, hidden_dim),
                nn.Dropout(dropout),
            )
        else:
            self.ffn_norm = None
            self.ffn = None

    def forward(
        self,
        x: torch.Tensor,
        chunk_size: int = None,  # Ignored, for API compatibility
        return_intermediates: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Forward pass.

        Args:
            x: (batch, seq_len, hidden_dim)
            chunk_size: Ignored (Mamba processes full sequence)
            return_intermediates: Whether to return diagnostic info

        Returns:
            output: (batch, seq_len, hidden_dim)
            intermediates: Optional dict with diagnostic info
        """
        # Mamba block with residual
        normed = self.mamba_norm(x)
        mamba_out = self.mamba(normed)
        mamba_out = self.mamba_dropout(mamba_out)
        x = x + mamba_out

        # Optional FFN block with residual
        if self.ffn is not None:
            x = x + self.ffn(self.ffn_norm(x))

        intermediates = None
        if return_intermediates:
            intermediates = {
                "layer_type": "mamba",
                "mamba_output_norm": mamba_out.norm(dim=-1).mean().item(),
            }

        return x, intermediates

    def get_parameter_groups(self) -> Dict[str, list]:
        """
        Get parameter groups for differential learning rates.

        State-related params (A_log, dt_proj, D) go to cross_chunk (long-range).
        Other params go to within_chunk (local processing).
        """
        state_params = []
        other_params = []

        for name, param in self.named_parameters():
            if 'A_log' in name or 'dt_proj' in name or '.D' in name:
                state_params.append(param)
            else:
                other_params.append(param)

        return {
            "within_chunk": other_params,
            "cross_chunk": state_params,
        }
