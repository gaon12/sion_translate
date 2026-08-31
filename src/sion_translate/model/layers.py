# PyTorch attention and RoPE overloads do not retain cache tuple shapes.
# pyright: reportCallIssue=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false

from __future__ import annotations


import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    """Apply RMS normalization without LayerNorm's mean subtraction.

    Each hidden vector is divided by its root-mean-square magnitude before a
    learned scale is applied. The normalization is always computed in float32
    for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        # AMP keeps master parameters in FP32. Multiplying the normalized BF16
        # activation by the FP32 weight without this cast promotes the whole
        # residual stream back to FP32, defeating most of BF16's memory benefit.
        return normalized.to(dtype=x.dtype) * self.weight.to(dtype=x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Represent relative positions with rotary positional embeddings (RoPE).

    RoPE rotates query and key vectors by a position-dependent angle instead of
    adding a separate position embedding. Precomputed cosine and sine tables are
    non-persistent buffers, so checkpoints do not store them and each process
    rebuilds them.
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE head dimension must be even")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        cos, sin = self._build_cache()
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        # Python attributes survive ``meta -> to_empty(device)`` materialization.
        # The tensor storage does not, so this marker lets the first real
        # forward rebuild non-persistent caches instead of reading garbage.
        self._cache_device = str(cos.device)

    def _build_cache(
        self, device: torch.device | str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(
                    0,
                    self.head_dim,
                    2,
                    dtype=torch.float32,
                    device=device,
                )
                / self.head_dim
            )
        )
        positions = torch.arange(
            self.max_seq_len,
            dtype=torch.float32,
            device=device,
        )
        frequencies = torch.outer(positions, inv_freq)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        return embedding.cos(), embedding.sin()

    def reset_parameters(self) -> None:
        """Rebuild derived RoPE buffers after meta-device materialization.

        ``to_empty`` allocates storage for non-persistent buffers but cannot
        reconstruct their values. FSDP's meta initialization therefore calls
        the model initializer, which in turn calls this method.
        """

        device = self.cos.device
        self.cos, self.sin = self._build_cache(device)
        self._cache_device = str(device)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cache_device != str(q.device) or self.cos.device != q.device:
            self.cos, self.sin = self._build_cache(q.device)
            self._cache_device = str(q.device)
        seq_len = q.shape[-2]
        if offset + seq_len > self.cos.shape[0]:
            raise ValueError(
                f"Sequence length {offset + seq_len} exceeds configured RoPE length {self.cos.shape[0]}"
            )
        cos = self.cos[offset : offset + seq_len].to(dtype=q.dtype)[None, None]
        sin = self.sin[offset : offset + seq_len].to(dtype=q.dtype)[None, None]
        return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


def _head_rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype)


class GQAAttention(nn.Module):
    """Grouped-Query Attention.

    Queries use ``num_heads`` heads, while keys and values use the smaller
    ``num_kv_heads`` count and are shared across query groups. This reduces KV
    memory and computation while retaining most of the quality of full
    multi-head attention.

    - ``qk_norm=True`` applies RMS normalization per query/key head to keep
      attention scores bounded and improve training stability.
    - RoPE is used only for self-attention. Decoder-to-encoder cross-attention
      does not rotate positions.
    """

    rope: RotaryEmbedding | None

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        *,
        dropout: float,
        qk_norm: bool,
        norm_eps: float,
        rope: RotaryEmbedding | None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.qk_norm = qk_norm
        self.norm_eps = norm_eps
        # RoPE is registered once on the root model and shared by all layers.
        # Bypass Module.__setattr__ here to avoid registering the same buffer-owning
        # module under every FSDP unit.
        self.__dict__["rope"] = rope
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=False)

    def _shape(self, x: torch.Tensor, heads: int) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, heads, self.head_dim).transpose(1, 2)

    def project_key_value(
        self,
        key_value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project reusable key/value states without computing a query."""
        key = self._shape(self.k_proj(key_value_states), self.num_kv_heads)
        value = self._shape(self.v_proj(key_value_states), self.num_kv_heads)
        if self.qk_norm:
            key = _head_rms_norm(key, self.norm_eps)
        return key, value

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        key_value_states: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_offset: int = 0,
        use_cache: bool = False,
    ):
        """Compute attention and optionally return ``(output, (key, value))``.

        The inference KV cache follows two policies:

        - Self-attention computes only the new token's key/value tensors and
          appends them to ``past_key_value``. ``position_offset`` supplies the
          number of tokens already generated so RoPE uses the correct position.
        - Cross-attention reuses the encoder key/value tensors computed at the
          first step because encoder output does not change during generation.
        """
        is_cross_attention = key_value_states is not None
        q = self._shape(self.q_proj(hidden_states), self.num_heads)
        if self.qk_norm:
            q = _head_rms_norm(q, self.norm_eps)

        if is_cross_attention and past_key_value is not None:
            # A cross-attention cache hit avoids recomputing encoder keys and values.
            k, v = past_key_value
        else:
            source = key_value_states if key_value_states is not None else hidden_states
            k, v = self.project_key_value(source)
            if self.rope is not None and not is_cross_attention:
                q, k = self.rope(q, k, offset=position_offset)
            if not is_cross_attention and past_key_value is not None:
                # Append the new token after cached self-attention keys and values.
                k = torch.cat((past_key_value[0], k), dim=-2)
                v = torch.cat((past_key_value[1], v), dim=-2)
        present_key_value = (k, v) if use_cache else None

        attention_mask = None
        if key_padding_mask is not None:
            expected_shape = (q.shape[0], k.shape[-2])
            if tuple(key_padding_mask.shape) != expected_shape:
                raise ValueError(
                    "key_padding_mask shape must match the attention batch and key length: "
                    f"expected {expected_shape}, got {tuple(key_padding_mask.shape)}"
                )
            attention_mask = key_padding_mask[:, None, None, :].to(
                device=q.device,
                dtype=torch.bool,
            )

        # Some CUDA SDPA backends reject an explicit mask together with
        # ``is_causal=True``. Merge both constraints into one boolean mask in
        # that case. Cached chunks need a lower-right causal alignment: query
        # row zero follows every cached key and may also attend to its own key.
        self_cache = past_key_value if not is_cross_attention else None
        has_self_cache = self_cache is not None
        if is_causal and q.shape[-2] == 1:
            # A single cached query is the final key, so every key is in its
            # causal past. For the one-token non-cached case this is equivalent
            # to the ordinary 1x1 causal mask.
            is_causal = False
        elif is_causal and (attention_mask is not None or has_self_cache):
            causal_diagonal = self_cache[0].shape[-2] if self_cache is not None else 0
            causal_mask = torch.ones(
                (q.shape[-2], k.shape[-2]),
                dtype=torch.bool,
                device=q.device,
            ).tril(diagonal=causal_diagonal)
            if attention_mask is None:
                attention_mask = causal_mask[None, None, :, :]
            else:
                attention_mask = attention_mask & causal_mask[None, None, :, :]
            is_causal = False
        enable_gqa = self.num_heads != self.num_kv_heads
        if enable_gqa and not q.is_cuda:
            repeats = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
            enable_gqa = False
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
            enable_gqa=enable_gqa,
        )
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(hidden_states.shape[0], hidden_states.shape[1], self.d_model)
        )
        output = self.out_proj(output)
        if use_cache:
            return output, present_key_value
        return output


class SwiGLU(nn.Module):
    """Use a gated ``gate * up`` SwiGLU feed-forward transformation.

    This modern replacement for a plain ReLU FFN generally provides better
    quality at a comparable parameter count.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float,
        *,
        gate_beta: float | None = None,
        up_beta: float | None = None,
    ):
        super().__init__()
        if (gate_beta is None) != (up_beta is None):
            raise ValueError("gate_beta and up_beta must be configured together")
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.gate_beta = gate_beta
        self.up_beta = up_beta

    def gated_activations(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        gate_beta = getattr(self, "gate_beta", None)
        up_beta = getattr(self, "up_beta", None)
        if gate_beta is None or up_beta is None:
            return F.silu(gate) * up
        # SiTU(z) = sigmoid(z) * beta_1*tanh(z/beta_1), while the
        # up branch is beta_2*tanh(z/beta_2). With 4/25 the product is
        # smoothly bounded by 100 but matches SwiGLU to first order at zero.
        capped_gate = torch.sigmoid(gate) * gate_beta * torch.tanh(gate / gate_beta)
        capped_up = up_beta * torch.tanh(up / up_beta)
        return capped_gate * capped_up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.dropout(self.gated_activations(x)))


class EncoderLayer(nn.Module):
    """Run pre-RMSNorm self-attention followed by pre-RMSNorm SwiGLU.

    Placing normalization before each residual branch keeps deep stacks stable.
    The final attention-mask multiplication keeps hidden states at padding
    positions equal to zero.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        d_ff: int,
        *,
        dropout: float,
        qk_norm: bool,
        norm_eps: float,
        rope: RotaryEmbedding,
        ffn_gate_beta: float | None = None,
        ffn_up_beta: float | None = None,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(d_model, norm_eps)
        self.self_attn = GQAAttention(
            d_model,
            num_heads,
            num_kv_heads,
            dropout=dropout,
            qk_norm=qk_norm,
            norm_eps=norm_eps,
            rope=rope,
        )
        self.ffn_norm = RMSNorm(d_model, norm_eps)
        self.ffn = SwiGLU(
            d_model,
            d_ff,
            dropout,
            gate_beta=ffn_gate_beta,
            up_beta=ffn_up_beta,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.attn_norm(x), key_padding_mask=attention_mask))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x * attention_mask.unsqueeze(-1).to(x.dtype)


class DecoderLayer(nn.Module):
    """Run causal self-attention, source cross-attention, and SwiGLU.

    - Self-attention uses a causal mask so a token cannot see future targets.
    - Cross-attention uses encoder output as keys and values, while
      ``source_mask`` hides padded source positions.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        d_ff: int,
        *,
        dropout: float,
        qk_norm: bool,
        norm_eps: float,
        rope: RotaryEmbedding,
        ffn_gate_beta: float | None = None,
        ffn_up_beta: float | None = None,
    ):
        super().__init__()
        self.self_norm = RMSNorm(d_model, norm_eps)
        self.self_attn = GQAAttention(
            d_model,
            num_heads,
            num_kv_heads,
            dropout=dropout,
            qk_norm=qk_norm,
            norm_eps=norm_eps,
            rope=rope,
        )
        self.cross_norm = RMSNorm(d_model, norm_eps)
        self.cross_attn = GQAAttention(
            d_model,
            num_heads,
            num_kv_heads,
            dropout=dropout,
            qk_norm=qk_norm,
            norm_eps=norm_eps,
            rope=None,
        )
        self.ffn_norm = RMSNorm(d_model, norm_eps)
        self.ffn = SwiGLU(
            d_model,
            d_ff,
            dropout,
            gate_beta=ffn_gate_beta,
            up_beta=ffn_up_beta,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_states: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.self_norm(x), is_causal=True))
        x = x + self.dropout(
            self.cross_attn(
                self.cross_norm(x),
                key_value_states=encoder_states,
                key_padding_mask=source_mask,
            )
        )
        return x + self.dropout(self.ffn(self.ffn_norm(x)))

    def project_cross_key_value(
        self,
        encoder_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project reusable cross-attention state through this layer's FSDP unit.

        Generation used to call ``cross_attn.project_key_value`` directly. When
        this decoder layer is an FSDP2 child shard, that bypasses the layer's
        pre-forward all-gather hook and leaves the projection weights as
        DTensors. Exposing the operation on the owning layer lets the
        distributed setup register it as a custom FSDP forward method.
        """

        return self.cross_attn.project_key_value(encoder_states)

    def forward_step(
        self,
        x: torch.Tensor,
        encoder_states: torch.Tensor,
        source_mask: torch.Tensor,
        *,
        self_kv: tuple[torch.Tensor, torch.Tensor] | None,
        cross_kv: tuple[torch.Tensor, torch.Tensor] | None,
        position_offset: int,
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        """Decode one new token with the inference-only KV cache.

        The result matches ``forward``, but previously processed attention is
        not recomputed. This reduces per-token work from quadratic to linear in
        the current sequence length. The return value contains the output,
        updated self-attention keys/values, and cross-attention keys/values.
        """
        attn_out, self_kv = self.self_attn(
            self.self_norm(x),
            past_key_value=self_kv,
            position_offset=position_offset,
            use_cache=True,
        )
        x = x + attn_out
        cross_out, cross_kv = self.cross_attn(
            self.cross_norm(x),
            key_value_states=encoder_states,
            key_padding_mask=source_mask,
            past_key_value=cross_kv,
            use_cache=True,
        )
        x = x + cross_out
        x = x + self.ffn(self.ffn_norm(x))
        if self_kv is None or cross_kv is None:
            raise RuntimeError("decoder attention cache was not initialized")
        return x, self_kv, cross_kv
