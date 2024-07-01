from __future__ import annotations

__all__ = ["ConformerMHSARelPosV1", "ConformerMHSARelPosV1Config"]

import math
from dataclasses import dataclass
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
        rel_pos_clip: maximal relative postion for embedding
        att_weights_dropout: attention weights dropout
        dropout: multi-headed self attention output dropout
        broadcast_dropout: whether to broadcast dropout on the feature axis to time axis
    """

    input_dim: int
    num_att_heads: int
    rel_pos_clip: int
    att_weights_dropout: float
    dropout: float
    broadcast_dropout: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.input_dim % self.num_att_heads == 0, "input_dim must be divisible by num_att_heads"


class ConformerMHSARelPosV1(nn.Module):
    """
    Conformer multi-headed self-attention module with relative positional encoding proposed by Shaw et al. (cf. https://arxiv.org/abs/1803.02155)
    """

    def __init__(self, cfg: ConformerMHSARelPosV1Config):

        super().__init__()

        self.layernorm = nn.LayerNorm(cfg.input_dim)

        self.embed_dim = cfg.input_dim
        self.num_heads = cfg.num_att_heads
        self.embed_dim_per_head = self.embed_dim // self.num_heads

        self.rel_pos_clip = cfg.rel_pos_clip

        self.att_weights_dropout = cfg.att_weights_dropout

        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"

        # projection matrices
        self.q_proj_weight = nn.parameter.Parameter(torch.empty((self.embed_dim, self.embed_dim)))
        self.k_proj_weight = nn.parameter.Parameter(torch.empty((self.embed_dim, self.embed_dim)))
        self.v_proj_weight = nn.parameter.Parameter(torch.empty((self.embed_dim, self.embed_dim)))

        self.in_proj_bias = nn.parameter.Parameter(torch.empty(3 * self.embed_dim))

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        if self.rel_pos_clip:
            self.rel_pos_embeddings = nn.parameter.Parameter(
                torch.empty(self.rel_pos_clip * 2 + 1, self.embed_dim // self.num_heads)
            )
        else:
            self.register_parameter("rel_pos_embeddings", None)

        self.dropout = cfg.dropout
        self.broadcast_dropout = cfg.broadcast_dropout

        self._reset_parameters()  # initialize parameters

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj_weight)
        nn.init.xavier_uniform_(self.k_proj_weight)
        nn.init.xavier_uniform_(self.v_proj_weight)

        if self.rel_pos_clip:
            nn.init.normal_(self.rel_pos_embeddings)

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

        q1 = query_seq.view(batch_dim_size, -1, self.num_heads, self.embed_dim_per_head).transpose(
            1, 2
        )  # [B, #heads, T, F']
        k_t = key_seq.view(batch_dim_size, -1, self.num_heads, self.embed_dim_per_head).permute(
            0, 2, 3, 1
        )  # [B, #heads, F', T']
        # attention between query and key sequences
        attn1 = torch.matmul(q1, k_t)  # [B, #heads, T, T']i

        if self.rel_pos_clip:
            q2 = (
                query_seq.transpose(0, 1)
                .contiguous()
                .view(time_dim_size, batch_dim_size * self.num_heads, self.embed_dim_per_head)
            )  # [T, B*#heads, F']

            range_vec_q = torch.arange(time_dim_size, device=input_tensor.device)
            range_vec_k = torch.arange(time_dim_size, device=input_tensor.device)

            distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
            distance_mat_clipped = torch.clamp(distance_mat, -self.rel_pos_clip, self.rel_pos_clip)

            final_mat = distance_mat_clipped + self.rel_pos_clip
            # relative positional embeddings
            rel_pos_embeddings = self.rel_pos_embeddings[final_mat]  # [T, T', F']

            # attention between query sequence and relative positional embeddings
            attn2 = torch.matmul(q2, rel_pos_embeddings.transpose(1, 2)).transpose(0, 1)  # [B*#heads, T, T']
            attn2 = attn2.contiguous().view(
                batch_dim_size, self.num_heads, time_dim_size, time_dim_size
            )  # [B, #heads, T, T']

            attn = (attn1 + attn2 + mask) * (math.sqrt(1.0 / float(self.embed_dim_per_head)))  # [B, #heads, T, T']
        else:
            attn = (attn1 + mask) * (math.sqrt(1.0 / float(self.embed_dim_per_head)))  # [B, #heads, T, T']

        # softmax and dropout
        attn_output_weights = F.dropout(
            F.softmax(attn, dim=-1), p=self.att_weights_dropout, training=self.training
        )  # [B, #heads, T, T']

        # sequence of weighted sums over value sequence
        v = value_seq.view(batch_dim_size, -1, self.num_heads, self.embed_dim_per_head).transpose(
            1, 2
        )  # [B, #heads, T', F']
        attn_output = (
            torch.matmul(attn_output_weights, v).transpose(1, 2).contiguous().view(batch_dim_size, -1, self.embed_dim)
        )  # [B, T, F]

        output_tensor = self.out_proj(attn_output)

        output_tensor = F.dropout(output_tensor, p=self.dropout, training=self.training)  # [B,T,F]

        if self.broadcast_dropout:
            output_tensor = F.dropout1d(
                output_tensor.transpose(1, 2), p=self.dropout, training=self.training
            ).transpose(1, 2)
        else:
            output_tensor = F.dropout(output_tensor, p=self.dropout, training=self.training)  # [B,T,F]

        return output_tensor
