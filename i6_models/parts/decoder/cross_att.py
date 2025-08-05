__all__ = ["CrossAttentionV1Config", "CrossAttentionV1State", "CrossAttentionV1"]

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, TypedDict

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from i6_models.config import ModelConfiguration
from i6_models.parts.dropout import BroadcastDropout
from i6_models.parts.decoder.util import ModuleWithState, make_kv_attn_mask


@dataclass
class CrossAttentionV1Config(ModelConfiguration):
    """
    Attributes:
        att_dropout: dropout applied to attention weights
        att_dropout_broadcast_axes: On which axes attention weight dropout is broadcast.
            Currently the implementation does not support broadcasting.
        dropout: dropout applied to the output of the attention module
        dropout_broadcast_axes: On which axes output dropout is broadcast.
        encoder_dim: dimension of the encoder output
        model_dim: dimension of the decoder model
        key_dim_total: total dimension of the key, across all heads
        value_dim_total: total dimension of the value, across all heads
        num_heads: number of attention heads
        with_bias: whether to use bias in the linear layers
    """

    att_dropout: float
    att_dropout_broadcast_axes: Optional[Literal["B", "T", "BT"]]
    dropout: float
    dropout_broadcast_axes: Optional[Literal["B", "T", "BT"]]
    encoder_dim: int
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
        assert self.encoder_dim > 0
        assert self.model_dim > 0
        assert self.num_heads > 0
        assert self.encoder_dim % self.num_heads == 0
        assert self.model_dim % self.num_heads == 0
        assert self.key_dim_total % self.num_heads == 0
        assert self.value_dim_total % self.num_heads == 0


class CrossAttentionV1State(TypedDict):
    """Recurrent state of the cross attention module."""

    k: Tensor
    """pre-computed key for the entire encoder output"""
    v: Tensor
    """pre-computed value for the entire encoder output"""
    mask: Tensor
    """attention mask to remove padding"""


class CrossAttentionV1(nn.Module, ModuleWithState[CrossAttentionV1State]):
    """Standard cross attention."""

    def __init__(self, cfg: CrossAttentionV1Config):
        super().__init__()

        self.att_dropout = cfg.att_dropout
        self.key_dim_total = cfg.key_dim_total
        self.value_dim_total = cfg.value_dim_total
        self.num_heads = cfg.num_heads

        self.norm = nn.LayerNorm(cfg.model_dim)
        self.q = nn.Linear(cfg.model_dim, cfg.key_dim_total, bias=cfg.with_bias)
        self.kv = nn.Linear(cfg.encoder_dim, cfg.key_dim_total + cfg.value_dim_total, bias=cfg.with_bias)
        self.out_proj = nn.Linear(cfg.value_dim_total, cfg.model_dim, bias=cfg.with_bias)
        self.dropout = BroadcastDropout(cfg.dropout, cfg.dropout_broadcast_axes)

    def get_initial_state(self) -> CrossAttentionV1State:
        return {
            "k": torch.zeros((0,)),
            "v": torch.zeros((0,)),
            "mask": torch.zeros((0,)),
        }

    def transform_encoder_output(
        self,
        encoder_output: Tensor,
        encoder_output_lens: Tensor,
        state: CrossAttentionV1State,
    ) -> CrossAttentionV1State:
        """Transform encoder output to key and value for cross attention."""

        # E: attention key/value dim
        kv = self.kv(encoder_output)  # B... T 2E
        k, v = torch.tensor_split(kv, (self.key_dim_total,), dim=-1)  # B... T E

        mask = make_kv_attn_mask(encoder_output, encoder_output_lens)  # B... 1 1 T

        k = torch.unflatten(k, -1, (self.num_heads, -1)).transpose(-3, -2)  # B... H T E
        v = torch.unflatten(v, -1, (self.num_heads, -1)).transpose(-3, -2)  # B... H T E

        return {**state, "k": k, "v": v, "mask": mask}

    def forward(self, x: Tensor, x_lens: Tensor, state: CrossAttentionV1State) -> Tuple[Tensor, CrossAttentionV1State]:
        """
        Apply cross attention.

        :param x: input of shape (B..., T, F)
        :param x_lens: unused
        :param state: recurrent state of the cross attention module
        """

        # Ev: attention value dim
        # L: length of query (i.e. length of token sequence)

        x = self.norm(x)
        q = self.q(x)  # B... L Ev
        q = torch.unflatten(q, -1, (self.num_heads, -1)).transpose(-3, -2)  # B... H L Ev

        att_out = F.scaled_dot_product_attention(
            q,
            key=state["k"],
            value=state["v"],
            attn_mask=state["mask"],
            dropout_p=self.att_dropout if self.training else 0.0,
        )  # B... H L E
        out = att_out.transpose(-3, -2).flatten(-2)  # B... L E
        out = self.out_proj(out)  # B... L F
        out = self.dropout(out)

        return out, state
