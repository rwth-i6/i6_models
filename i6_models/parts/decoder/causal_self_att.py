__all__ = ["CausalSelfAttentionV1Config", "CausalSelfAttentionV1State", "CausalSelfAttentionV1"]

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, TypedDict

from torch import nn, Tensor
import torch.nn.functional as F
import torch

from i6_models.config import ModelConfiguration
from i6_models.parts.dropout import BroadcastDropout
from i6_models.parts.decoder.util import ModuleWithState, make_kv_attn_mask


@dataclass
class CausalSelfAttentionV1Config(ModelConfiguration):
    """
    Attributes:
        att_dropout: dropout applied to attention weights
        att_dropout_broadcast_axes: On which axes attention weight dropout is broadcast.
            Currently the implementation does not support broadcasting.
        dropout: dropout applied to the output of the attention module
        dropout_broadcast_axes: On which axes output dropout is broadcast.
        model_dim: dimension of the model
        key_dim_total: total dimension of the key, across all heads
        value_dim_total: total dimension of the value, across all heads
        num_heads: number of attention heads
        with_bias: whether to use bias in the linear layers
    """

    att_dropout: float
    att_dropout_broadcast_axes: Optional[Literal["B", "T", "BT"]]
    dropout: float
    dropout_broadcast_axes: Optional[Literal["B", "T", "BT"]]
    model_dim: int
    key_dim_total: int
    value_dim_total: int
    num_heads: int
    with_bias: bool

    def __post_init__(self):
        super().__post_init__()

        assert self.att_dropout >= 0
        assert self.att_dropout_broadcast_axes is None, "torch.sdpa does not support dropout broadcast customization"
        assert self.dropout >= 0
        assert self.model_dim > 0
        assert self.num_heads > 0
        assert self.model_dim % self.num_heads == 0
        assert self.key_dim_total % self.num_heads == 0
        assert self.value_dim_total % self.num_heads == 0


class CausalSelfAttentionV1State(TypedDict):
    """Recurrent state of the causal self attention module."""

    k_accum: Tensor
    """key accumulated over time axis"""
    v_accum: Tensor
    """value accumulated over time axis"""
    lens_accum: Tensor
    """accumulated seq lengths"""


class CausalSelfAttentionV1(nn.Module, ModuleWithState[CausalSelfAttentionV1State]):
    """Causal self attention module that is steppable over time for optimized decoding."""

    def __init__(self, cfg: CausalSelfAttentionV1Config):
        super().__init__()

        self.att_dropout = cfg.att_dropout
        self.key_dim_total = cfg.key_dim_total
        self.value_dim_total = cfg.value_dim_total
        self.num_heads = cfg.num_heads

        self.norm = nn.LayerNorm(cfg.model_dim)
        self.qkv = nn.Linear(
            cfg.model_dim,
            2 * cfg.key_dim_total + cfg.value_dim_total,
            bias=cfg.with_bias,
        )
        self.out_proj = nn.Linear(cfg.value_dim_total, cfg.model_dim, bias=cfg.with_bias)
        self.dropout = BroadcastDropout(cfg.dropout, cfg.dropout_broadcast_axes)

    def get_initial_state(self) -> CausalSelfAttentionV1State:
        return {
            "k_accum": torch.zeros((0,)),
            "v_accum": torch.zeros((0,)),
            "lens_accum": torch.zeros((0,)),
        }

    def transform_encoder_output(
        self,
        encoder_output: Tensor,
        encoder_output_lens: Tensor,
        state: CausalSelfAttentionV1State,
    ) -> CausalSelfAttentionV1State:
        return state

    def _step_state(
        self, state: CausalSelfAttentionV1State, k: Tensor, v: Tensor, lens: Tensor
    ) -> Tuple[CausalSelfAttentionV1State, Tensor, Tensor, Tensor]:
        assert k.shape[1] == v.shape[1]

        k_accum = state["k_accum"]
        v_accum = state["v_accum"]
        lens_accum = state["lens_accum"]
        if k_accum.numel() == 0 and v_accum.numel() == 0 and lens_accum.numel() == 0:  # initial state
            pass
        elif k.shape[-2] == 1:  # single time step -> accum
            # expand potential beam dimensions to widening beam
            k_accum = k_accum.expand(*k.shape[:-2], *k_accum.shape[-2:])
            v_accum = v_accum.expand(*v.shape[:-2], *v_accum.shape[-2:])
            lens_accum = lens_accum.expand(*lens.shape)
            k = torch.cat([k_accum, k], dim=-2)  # B... T F
            v = torch.cat([v_accum, v], dim=-2)  # B... T F
            lens = torch.stack([lens_accum, lens], dim=-1).sum(dim=-1)  # B...
        else:
            raise NotImplementedError(
                f"Cannot step {self.__class__.__name__} more than one step at a time "
                f"after it has been stepped, but trying to go {k.shape[1]} steps."
            )

        return {**state, "k_accum": k, "v_accum": v, "lens_accum": lens}, k, v, lens

    def forward(
        self, x: Tensor, x_lens: Tensor, state: CausalSelfAttentionV1State
    ) -> Tuple[Tensor, CausalSelfAttentionV1State]:
        """
        Apply causal multi-head self attention.

        :param x: input tensor of shape (B..., T, F)
        :param x_lens: lengths of data in x, shape (B...,)
        :param state: recurrent state
        """

        # E: dim of attention query/key/value

        x = self.norm(x)  # B... T F
        qkv = self.qkv(x)  # B... T ~3HE
        q, k, v = torch.tensor_split(qkv, (self.key_dim_total, 2 * self.key_dim_total), dim=-1)  # B... T HE
        new_state, k, v, x_lens_accum = self._step_state(state, k, v, x_lens)  # k, v: B T HE, x_lens: B...

        q = torch.unflatten(q, -1, (self.num_heads, -1)).transpose(-3, -2)  # B... H T E
        k = torch.unflatten(k, -1, (self.num_heads, -1)).transpose(-3, -2)  # B... H T E
        v = torch.unflatten(v, -1, (self.num_heads, -1)).transpose(-3, -2)  # B... H T E

        causal_mask = nn.Transformer.generate_square_subsequent_mask(k.shape[-2], device=k.device)  # T T

        # If we are stepping the decoder we need just the subset of causal mask for the current step.
        t_dim = q.shape[-2]
        l_dim = k.shape[-2]
        if l_dim > t_dim:
            assert t_dim == 1, f"stepping decoder, but got query time dim {t_dim} (should be 1)"
            causal_mask = causal_mask[l_dim - 1, :]  # B... 1 (heads) 1 (q) T (k/v)

        kv_mask = make_kv_attn_mask(k, x_lens_accum)  # B... 1 (heads) 1 (q) T (k/v)

        att_out = F.scaled_dot_product_attention(
            q,
            key=k,
            value=v,
            attn_mask=causal_mask + kv_mask,
            dropout_p=self.att_dropout if self.training else 0.0,
        )  # B H T E
        out = att_out.transpose(-2, -3).flatten(-2)  # B... T E
        out = self.out_proj(out)  # B... T F
        out = self.dropout(out)

        return out, new_state
