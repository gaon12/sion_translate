from __future__ import annotations


import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    """RMS 정규화. LayerNorm 에서 평균 빼기를 생략한 가벼운 버전입니다.

    각 위치의 hidden vector 를 자기 크기(RMS)로 나눠 스케일을 일정하게
    맞춘 뒤, 학습 가능한 가중치를 곱합니다. 수치 안정성을 위해 계산은
    항상 float32 로 수행합니다.
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
    """RoPE(회전 위치 인코딩). 위치 정보를 별도 임베딩으로 더하는 대신,
    query/key 벡터를 위치에 비례한 각도로 '회전'시켜 상대 위치를 표현합니다.

    cos/sin 표는 미리 계산해 버퍼로 보관합니다 (persistent=False 이므로
    체크포인트에는 저장되지 않고 매번 다시 계산됩니다).
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

    query 는 ``num_heads`` 개를 쓰지만 key/value 는 더 적은 ``num_kv_heads``
    개만 만들어 여러 query head 가 공유합니다. 품질은 거의 유지하면서
    KV 메모리와 연산을 줄이는 기법입니다.

    - ``qk_norm=True`` 면 query/key 를 head 단위로 RMS 정규화해
      attention 점수 폭주를 막습니다 (학습 안정성 향상).
    - self-attention 일 때만 RoPE 를 적용하고, cross-attention
      (decoder→encoder)에는 위치 회전을 적용하지 않습니다.
    """

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
        """attention 계산. ``use_cache=True`` 면 (출력, (k, v)) 튜플을 반환합니다.

        KV cache (추론 가속용):
        - self-attention: 새 토큰의 k/v 만 계산해 ``past_key_value`` 뒤에
          이어 붙입니다. ``position_offset`` 은 지금까지 생성된 토큰 수로,
          RoPE 회전 각도를 올바른 위치에 맞추는 데 씁니다.
        - cross-attention: encoder 출력은 생성 내내 변하지 않으므로
          첫 스텝에 계산한 k/v 를 ``past_key_value`` 로 그대로 재사용합니다.
        """
        is_cross_attention = key_value_states is not None
        q = self._shape(self.q_proj(hidden_states), self.num_heads)
        if self.qk_norm:
            q = _head_rms_norm(q, self.norm_eps)

        if is_cross_attention and past_key_value is not None:
            # cross-attention 캐시 적중: encoder 쪽 k/v 재계산 생략
            k, v = past_key_value
        else:
            source = key_value_states if is_cross_attention else hidden_states
            k = self._shape(self.k_proj(source), self.num_kv_heads)
            v = self._shape(self.v_proj(source), self.num_kv_heads)
            if self.qk_norm:
                k = _head_rms_norm(k, self.norm_eps)
            if self.rope is not None and not is_cross_attention:
                q, k = self.rope(q, k, offset=position_offset)
            if not is_cross_attention and past_key_value is not None:
                # self-attention 캐시: 과거 토큰의 k/v 뒤에 새 토큰을 이어 붙임
                k = torch.cat((past_key_value[0], k), dim=-2)
                v = torch.cat((past_key_value[1], v), dim=-2)
        present_key_value = (k, v) if use_cache else None

        attention_mask = None
        if key_padding_mask is not None:
            attention_mask = key_padding_mask[:, None, None, :].to(torch.bool)
        # 토큰을 1개씩 생성하는 캐시 디코딩에서는 query 가 항상 마지막 위치이므로
        # causal mask 가 필요 없습니다 (SDPA 의 is_causal 은 query/key 길이가
        # 다르면 마스크를 잘못 정렬하므로 여기서 반드시 꺼야 합니다).
        if is_causal and q.shape[-2] == 1:
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
    """SwiGLU feed-forward. 일반 FFN(ReLU) 대신 gate × up 곱 구조를 써서
    같은 파라미터 수로 더 좋은 성능을 내는 현대 표준 구성입니다."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.dropout(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class EncoderLayer(nn.Module):
    """인코더 한 층: pre-RMSNorm → self-attention → pre-RMSNorm → SwiGLU.

    잔차 연결(residual) 앞에 norm 을 두는 pre-norm 구조라 깊게 쌓아도
    학습이 안정적입니다. 마지막에 attention_mask 를 곱해 패딩 위치의
    hidden 을 0 으로 유지합니다.
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
        self.ffn = SwiGLU(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.attn_norm(x), key_padding_mask=attention_mask))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x * attention_mask.unsqueeze(-1).to(x.dtype)


class DecoderLayer(nn.Module):
    """디코더 한 층: causal self-attention → cross-attention(원문 참조) → SwiGLU.

    - self-attention 은 미래 토큰을 보지 못하도록 causal mask 를 사용합니다.
    - cross-attention 은 인코더 출력(원문)을 key/value 로 사용하며,
      원문 패딩 위치는 ``source_mask`` 로 가립니다.
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
        self.ffn = SwiGLU(d_model, d_ff, dropout)
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
        """KV cache 를 사용한 '새 토큰 1개' 디코딩 스텝 (추론 전용).

        forward 와 계산 결과는 같지만, 이미 처리한 토큰의 attention 을
        다시 계산하지 않아 토큰당 비용이 O(전체 길이²) → O(전체 길이)로
        줄어듭니다. 반환값: (출력, 갱신된 self k/v, cross k/v)
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
        return x, self_kv, cross_kv
