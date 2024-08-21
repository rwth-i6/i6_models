from __future__ import annotations


__all__ = ["ConformerMHSARelPosV1", "ConformerMHSARelPosV1Config"]

from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from i6_models.config import ModelConfiguration
from i6_models.util import compat


@dataclass
class ConformerMHSARelPosV1Config(ModelConfiguration):
    """
    Attributes:
        input_dim: input dim and total dimension for query/key and value projections, should be divisible by `num_att_heads`
        num_att_heads: number of attention heads
        att_weights_dropout: attention weights dropout
        learnable_pos_emb: whether to use learnable relative positional embeddings instead of fixed sinusoidal ones
        rel_pos_clip: maximal relative postion for embedding
        with_pos_bias: whether to add additional position bias terms to the attention scores
        pos_emb_dropout: dropout for the positional embeddings
        dropout: multi-headed self attention output dropout
        dropout_broadcast_axes: string of axes to which dropout is broadcast, e.g. "T" for broadcasting to the time axis
                                setting to None to disable broadcasting
    """

    input_dim: int
    num_att_heads: int
    att_weights_dropout: float
    dropout: float
    learnable_pos_emb: bool = True
    rel_pos_clip: Optional[int] = None
    with_pos_bias: bool = False
    pos_emb_dropout: float = 0.0
    dropout_broadcast_axes: Optional[str] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.input_dim % self.num_att_heads == 0, "input_dim must be divisible by num_att_heads"
        assert self.dropout_broadcast_axes is None or self.dropout_broadcast_axes in [
            "B",
            "T",
            "BT",
        ], "invalid value, supported are None, 'B', 'T' and 'BT'"


class ConformerMHSARelPosV1(nn.Module):
    """
    Conformer multi-headed self-attention module supporting
        - relative positional encoding proposed by Shaw et al. (cf. https://arxiv.org/abs/1803.02155) by setting `learnable_pos_emb` to True and `with_pos_bias` to False
        - and Transformer-XL style relative PE by Dai et al. (cf. https://arxiv.org/abs/1901.02860) by setting `learnable_pos_emb` to False and `with_pos_bias` to True

    """

    def __init__(self, cfg: ConformerMHSARelPosV1Config):

        super().__init__()

        self.layernorm = nn.LayerNorm(cfg.input_dim)

        self.embed_dim = cfg.input_dim
        self.num_heads = cfg.num_att_heads
        self.embed_dim_per_head = self.embed_dim // self.num_heads

        self.learnable_pos_emb = cfg.learnable_pos_emb
        self.rel_pos_clip = cfg.rel_pos_clip
        self.with_pos_bias = cfg.with_pos_bias
        self.pos_emb_dropout = nn.Dropout(cfg.pos_emb_dropout)

        assert not self.learnable_pos_emb or self.rel_pos_clip

        self.att_weights_dropout = nn.Dropout(cfg.att_weights_dropout)

        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"

        # projection matrices
        self.q_proj_weight = nn.parameter.Parameter(torch.empty((self.embed_dim, self.embed_dim)))
        self.k_proj_weight = nn.parameter.Parameter(torch.empty((self.embed_dim, self.embed_dim)))
        self.v_proj_weight = nn.parameter.Parameter(torch.empty((self.embed_dim, self.embed_dim)))

        self.in_proj_bias = nn.parameter.Parameter(torch.empty(3 * self.embed_dim))

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        self.register_parameter("rel_pos_embeddings", None)
        self.register_parameter("pos_bias_u", None)
        self.register_parameter("pos_bias_v", None)

        if self.learnable_pos_emb:
            self.rel_pos_embeddings = nn.parameter.Parameter(
                torch.empty(self.rel_pos_clip * 2 + 1, self.embed_dim // self.num_heads)
            )
        if self.with_pos_bias:
            self.pos_bias_u = nn.parameter.Parameter(torch.empty(self.num_heads, self.embed_dim_per_head))
            self.pos_bias_v = nn.parameter.Parameter(torch.empty(self.num_heads, self.embed_dim_per_head))

        self.dropout = nn.Dropout1d(cfg.dropout) if cfg.dropout_broadcast_axes else nn.Dropout(cfg.dropout)
        self.dropout_broadcast_axes = cfg.dropout_broadcast_axes

        self._reset_parameters()  # initialize parameters

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj_weight)
        nn.init.xavier_uniform_(self.k_proj_weight)
        nn.init.xavier_uniform_(self.v_proj_weight)

        # TODO: choose kind of initialization
        if self.learnable_pos_emb:
            nn.init.normal_(self.rel_pos_embeddings)
        if self.with_pos_bias:
            nn.init.constant_(self.pos_bias_u, 0.0)
            nn.init.constant_(self.pos_bias_v, 0.0)
        nn.init.constant_(self.in_proj_bias, 0.0)

    def forward(self, input_tensor: torch.Tensor, sequence_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply layer norm and multi-head self attention and dropout

        :param input_tensor: Input to the self attention of shape (B, T, F)
        :param sequence_mask: bool mask of shape (B, T), True signals within sequence, False outside, will be inverted to match the torch.nn.MultiheadAttention module
        which will be applied/added to dot product, used to mask padded key positions out
        """
        output_tensor = self.layernorm(input_tensor)  # [B,T,F]

        time_dim_size = output_tensor.shape[1]
        batch_dim_size = output_tensor.shape[0]

        # attention mask
        inv_sequence_mask = compat.logical_not(sequence_mask)  # [B, T]
        mask = (
            torch.zeros_like(inv_sequence_mask, dtype=input_tensor.dtype)
            .masked_fill(inv_sequence_mask, float("-inf"))
            .view(batch_dim_size, 1, 1, time_dim_size)
            .expand(-1, self.num_heads, -1, -1)
        )  # [B, #heads, 1, T']

        # query, key and value sequences
        bias_k, bias_q, bias_v = self.in_proj_bias.chunk(3)

        query_seq = F.linear(output_tensor, self.q_proj_weight, bias_q)  # [B, T, #heads * F']
        key_seq = F.linear(output_tensor, self.k_proj_weight, bias_k)
        value_seq = F.linear(output_tensor, self.v_proj_weight, bias_v)

        q = query_seq.view(batch_dim_size, -1, self.num_heads, self.embed_dim_per_head)  # [B, T, #heads, F']
        k = key_seq.view(batch_dim_size, -1, self.num_heads, self.embed_dim_per_head)  # [B, T', #heads, F']

        if self.learnable_pos_emb:
            pos_seq_q = torch.arange(time_dim_size, device=input_tensor.device)
            pos_seq_k = torch.arange(time_dim_size, device=input_tensor.device)

            distance_mat = pos_seq_k[None, :] - pos_seq_q[:, None]
            distance_mat_clipped = torch.clamp(distance_mat, -self.rel_pos_clip, self.rel_pos_clip)

            final_mat = distance_mat_clipped + self.rel_pos_clip

            rel_pos_embeddings = self.rel_pos_embeddings[final_mat]  # [T, T', F']
        else:
            rel_pos_embeddings = self._sinusoidal_pe(
                torch.arange(time_dim_size - 1, -time_dim_size, -1, device=input_tensor.device, dtype=torch.float32),
                self.embed_dim_per_head,
            ).expand(
                time_dim_size, 2 * time_dim_size - 1, self.embed_dim_per_head
            )  # [T, T+T'-1, F']

        # dropout relative positional embeddings
        rel_pos_embeddings = self.pos_emb_dropout(rel_pos_embeddings)

        q_with_bias_u = q + self.pos_bias_u if self.with_pos_bias else q  # [B, T, #heads, F']
        q_with_bias_v = q + self.pos_bias_v if self.with_pos_bias else q

        # attention matrix a and c
        attn_ac = torch.einsum("bihf, bjhf -> bhij", q_with_bias_u, k)  # [B, #heads, T, T']

        # attention matrix b and d
        attn_bd = torch.einsum("bihf, ijf -> bhij", q_with_bias_v, rel_pos_embeddings)

        if not self.learnable_pos_emb:
            attn_bd = self._rel_shift_bhij(attn_bd, k_len=time_dim_size)

        attn = attn_ac + attn_bd + mask  # [B, #heads, T, T']
        attn_scaled = attn * (math.sqrt(1.0 / float(self.embed_dim_per_head)))  # [B, #heads, T, T']

        # softmax and dropout
        attn_output_weights = self.att_weights_dropout(F.softmax(attn_scaled, dim=-1))  # [B, #heads, T, T']

        # sequence of weighted sums over value sequence
        v = value_seq.view(batch_dim_size, -1, self.num_heads, self.embed_dim_per_head)  # [B, T, H, F']
        attn_output = (
            torch.einsum("bhij, bjhf -> bhif", attn_output_weights, v)
            .transpose(1, 2)
            .contiguous()
            .view(batch_dim_size, -1, self.embed_dim)
        )

        output_tensor = self.out_proj(attn_output)

        if self.dropout_broadcast_axes is None:
            output_tensor = self.dropout(output_tensor)
        elif self.dropout_broadcast_axes == "T":
            output_tensor = self.dropout(output_tensor.transpose(1, 2)).transpose(1, 2)
        elif self.dropout_broadcast_axes == "B":
            output_tensor = self.dropout(output_tensor.permute(1, 2, 0)).permute(2, 0, 1)
        elif self.dropout_broadcast_axes == "BT":
            batch_dim_size = output_tensor.shape[0]
            feature_dim_size = output_tensor.shape[-1]

            output_tensor = (
                self.output(output_tensor.reshape(-1, feature_dim_size).transpose(0, 1))
                .transpose(0, 1)
                .reshape(batch_dim_size, -1, feature_dim_size)
            )

        return output_tensor  # [B,T,F]

    @staticmethod
    def _rel_shift_bhij(x, k_len=None):
        """
        :param x: input tensor of shape (B, H, T, L) to apply left shift
        :k_len: length of the key squence
        """
        x_shape = x.shape

        x = torch.nn.functional.pad(x, (1, 0))  # [B, H, T, L+1]
        x = x.reshape(x_shape[0], x_shape[1], x_shape[3] + 1, x_shape[2])  # [B, H, L+1, T]
        x = x[:, :, 1:]  # [B, H, L, T]
        x = x.reshape(x_shape)  # [B, H, T, L]]

        return x[:, :, :, :k_len] if k_len else x  # [B, H, T, T']

    @staticmethod
    def _sinusoidal_pe(pos_seq: torch.Tensor, embed_dim: int):
        """
        :param pos_seq: 1-D position sequence for which to compute embeddings
        :param embed_dim: embedding dimension
        """
        inv_freq = 1 / (10000 ** (torch.arange(0.0, embed_dim, 2.0, device=pos_seq.device) / embed_dim))

        sinusoid_input = torch.outer(pos_seq, inv_freq)
        pos_emb = torch.cat([sinusoid_input.sin(), sinusoid_input.cos()], dim=-1)  # [num. positions, embed_dim]

        return pos_emb
