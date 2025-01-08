from __future__ import annotations

__all__ = ["ConformerMHSAWithGateV1", "ConformerMHSAV1Config"]

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch.nn import init

from i6_models.config import ModelConfiguration


@dataclass
class ConformerMHSAV1Config(ModelConfiguration):
    """
    Attributes:
        input_dim: input dim and total dimension for query/key and value projections, should be divisible by `num_att_heads`
        num_att_heads: number of attention heads
        att_weights_dropout: attention weights dropout
        dropout: multi-headed self attention output dropout
    """

    input_dim: int
    num_att_heads: int
    att_weights_dropout: float
    dropout: float

    def __post_init__(self) -> None:
        super().__post_init__()
        assert (
                self.input_dim % self.num_att_heads == 0
        ), "input_dim must be divisible by num_att_heads"


class ConformerMHSAWithGateV1(torch.nn.Module):
    """
    Conformer multi-headed self-attention module
    """

    def __init__(self, cfg: ConformerMHSAV1Config):
        super().__init__()

        self.layernorm = torch.nn.LayerNorm(cfg.input_dim)
        self.mhsa = MultiheadAttention(
            cfg.input_dim,
            cfg.num_att_heads,
            dropout=cfg.att_weights_dropout,
        )
        self.dropout = cfg.dropout

    def forward(
            self, input_tensor: torch.Tensor, sequence_mask: torch.Tensor, head_gates: torch.Tensor = None, hard_prune: bool = False
    ) -> torch.Tensor:
        """
        Apply layer norm and multi-head self attention and dropout

        :param input_tensor: Input to the self attention of shape (B, T, F)
        :param sequence_mask: bool mask of shape (B, T), True signals within sequence, False outside, will be inverted to match the torch.nn.MultiheadAttention module
        which will be applied/added to dot product, used to mask padded key positions out
        """

        sequence_mask = sequence_mask.unsqueeze(-2)
        len_max = sequence_mask.size(-1)
        att_mask_tmp = sequence_mask.transpose(1, 2).repeat(1, 1, len_max)
        att_mask = sequence_mask.repeat(1, len_max, 1) & att_mask_tmp

        output_tensor = self.layernorm(input_tensor)  # [B,T,F]

        output_tensor = self.mhsa(
            output_tensor,
            output_tensor,
            output_tensor,
            mask=att_mask,
            head_gates=head_gates,
            hard_prune=hard_prune
        )  # [B,T,F]
        output_tensor = torch.nn.functional.dropout(
            output_tensor, p=self.dropout, training=self.training
        )  # [B,T,F]

        return output_tensor


class MultiheadAttention(torch.nn.Module):
    """Multi-Head Attention layer.

    Args:
        num_att_heads (int): The number of heads.
        input_dim (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, input_dim, num_att_heads, dropout):
        """Construct an MultiHeadedAttention object."""
        super(MultiheadAttention, self).__init__()
        assert input_dim % num_att_heads == 0
        # We assume d_v always equals d_k
        self.d_k = input_dim // num_att_heads
        self.h = num_att_heads

        self.linear_q_weight = torch.nn.parameter.Parameter(torch.empty((input_dim, input_dim)))
        self.linear_q_bias = torch.nn.parameter.Parameter(torch.empty(input_dim))
        self.linear_k_weight = torch.nn.parameter.Parameter(torch.empty((input_dim, input_dim)))
        self.linear_k_bias = torch.nn.parameter.Parameter(torch.empty(input_dim))
        self.linear_v_weight = torch.nn.parameter.Parameter(torch.empty((input_dim, input_dim)))
        self.linear_v_bias = torch.nn.parameter.Parameter(torch.empty(input_dim))
        self.linear_out_weight = torch.nn.parameter.Parameter(torch.empty((input_dim, input_dim)))
        self.linear_out_bias = torch.nn.parameter.Parameter(torch.empty(input_dim))

        self.attn = None
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """ same initialization as default pytorch linear layer"""
        for weight, bias in [(self.linear_q_weight, self.linear_q_bias), (self.linear_k_weight, self.linear_k_bias),
                             (self.linear_v_weight, self.linear_v_bias),
                             (self.linear_out_weight, self.linear_out_bias)]:
            init.kaiming_uniform_(weight, a=math.sqrt(5))
            if bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(bias, -bound, bound)

    def forward_qkv(self, query, key, value, head_gates=None, hard_prune: bool = False):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        if head_gates is None:
            q = F.linear(query, self.linear_q_weight, self.linear_q_bias).view(n_batch, -1, self.h, self.d_k)
            k = F.linear(key, self.linear_k_weight, self.linear_k_bias).view(n_batch, -1, self.h, self.d_k)
            v = F.linear(value, self.linear_v_weight, self.linear_v_bias).view(n_batch, -1, self.h, self.d_k)
        else:
            input_dim = self.linear_q_weight.size()[1]
            q_weights = torch.reshape(self.linear_q_weight, (-1, self.d_k, input_dim))  # (h, d_k, F)
            q_bias = torch.reshape(self.linear_q_bias, (-1, self.d_k))  # (h, d_k)
            k_weights = torch.reshape(self.linear_k_weight, (-1, self.d_k, input_dim))  # (h, d_k, F)
            k_bias = torch.reshape(self.linear_k_bias, (-1, self.d_k))  # (h, d_k)
            v_weights = torch.reshape(self.linear_v_weight, (-1, self.d_k, input_dim))  # (h, d_k, F)
            v_bias = torch.reshape(self.linear_v_bias, (-1, self.d_k))  # (h, d_k)
            if not hard_prune and self.training:
                q_weights = torch.einsum('hij,h->hij', q_weights, head_gates)  # (h, d_k, h*d_k)
                q_bias = torch.einsum('hi,h->hi', q_bias, head_gates)  # (h, d_k)
                k_weights = torch.einsum('hij,h->hij', k_weights, head_gates)  # (h, d_k, h*d_k)
                k_bias = torch.einsum('hi,h->hi', k_bias, head_gates)  # (h, d_k)
                v_weights = torch.einsum('hij,h->hij', v_weights, head_gates)  # (h, d_k, h*d_k)
                v_bias = torch.einsum('hi,h->hi', v_bias, head_gates)  # (h, d_k)
            else:
                q_weights = q_weights[head_gates.bool()]  # (h', d_k, h*d_k)
                q_bias = q_bias[head_gates.bool()]  # (h', d_k)
                k_weights = k_weights[head_gates.bool()]  # (h', d_k, h*d_k)
                k_bias = k_bias[head_gates.bool()]  # (h', d_k)
                v_weights = v_weights[head_gates.bool()]  # (h', d_k, h*d_k)
                v_bias = v_bias[head_gates.bool()]  # (h', d_k)

            if hard_prune:
                num_heads = torch.count_nonzero(head_gates)
            else:
                num_heads = self.h
            q_weights = torch.reshape(q_weights, (-1, input_dim))  # (h'*d_k, h*d_k)
            q_bias = torch.reshape(q_bias, (num_heads * self.d_k,))  # (h'*d_k, h*d_k)
            k_weights = torch.reshape(k_weights, (-1, input_dim))  # (h'*d_k, h*d_k)
            k_bias = torch.reshape(k_bias, (num_heads * self.d_k,))  # (h'*d_k, h*d_k)
            v_weights = torch.reshape(v_weights, (-1, input_dim))  # (h'*d_k, h*d_k)
            v_bias = torch.reshape(v_bias, (num_heads * self.d_k,))  # (h'*d_k, h*d_k)

            q = F.linear(query, q_weights, q_bias).view(n_batch, -1, num_heads, self.d_k)
            k = F.linear(key, k_weights, k_bias).view(n_batch, -1, num_heads, self.d_k)
            v = F.linear(value, v_weights, v_bias).view(n_batch, -1, num_heads, self.d_k)

        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self, value, scores, mask, head_gates=None, hard_prune: bool = False):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask, min_value)

            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = torch.nn.functional.dropout(
            self.attn, p=self.dropout, training=self.training
        )
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)

        num_heads = self.h
        out_weights = self.linear_out_weight
        # ===============================================
        if head_gates is not None:
            input_dim = self.linear_out_weight.size()[1]
            out_weights = torch.reshape(self.linear_out_weight, (input_dim, self.d_k, -1))  # (F, d_k, h')

            if not hard_prune and self.training:
                x = torch.einsum('bhtd,h->bhtd', x, head_gates)
                out_weights = torch.einsum('ijH,h->ijH', out_weights, head_gates)   # (F, d_k, h')
            else:
                num_heads = torch.count_nonzero(head_gates)
                if not x.size()[1] == num_heads:
                    x = x[:, head_gates.bool()]
                out_weights = out_weights[:, :, head_gates.bool()]  # (F, d_k, h')

            out_weights = torch.reshape(out_weights, (input_dim, -1))
        # ===============================================

        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, num_heads * self.d_k)
        )  # (batch, time1, d_model)

        return F.linear(x, out_weights, self.linear_out_bias)  # (batch, time1, d_model)

    def forward(self, query, key, value, mask, head_gates=None, hard_prune: bool = False):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value, head_gates, hard_prune)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask, head_gates, hard_prune)
