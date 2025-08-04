from __future__ import annotations

__all__ = ["ConformerMHSAWithGateV1", "ConformerMHSAV1Config"]

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch.nn import init

from i6_models.config import ModelConfiguration
from i6_models.util import compat


@dataclass
class ConformerMHSAV1Config(ModelConfiguration):
    """
    Attributes:
        input_dim: input dim and total dimension for query/key and value projections, should be divisible by `num_att_heads`
        num_att_heads: number of attention heads
        att_head_dim: the dimension for each attention head
        att_weights_dropout: attention weights dropout
        dropout: multi-headed self attention output dropout
    """

    input_dim: int
    num_att_heads: int
    att_head_dim: int
    att_weights_dropout: float
    dropout: float

    def __post_init__(self) -> None:
        super().__post_init__()
        # assert (
        #         self.input_dim % self.num_att_heads == 0
        # ), "input_dim must be divisible by num_att_heads"


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
            cfg.att_head_dim,
            dropout=cfg.att_weights_dropout,
        )
        self.dropout = cfg.dropout

    def forward(
        self,
        input_tensor: torch.Tensor,
        sequence_mask: torch.Tensor,
        head_gates: torch.Tensor = None,
        hard_prune: bool = False,
        adjust_dropout: bool = False,
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
        # inv_attn_mask = (
        #     compat.logical_not(att_mask) if att_mask is not None else None
        # )
        # print("inv_attn_mask", inv_attn_mask)

        if self.mhsa.linear_out_weight.size()[1] == 0:
            return input_tensor

        dropout_p = self.dropout
        attn_dropout_p = self.mhsa.dropout
        if adjust_dropout:
            adjust_scale = self.mhsa.h / (self.mhsa.input_dim // self.mhsa.d_k)
            dropout_p *= adjust_scale
            attn_dropout_p *= adjust_scale

        output_tensor = self.layernorm(input_tensor)  # [B,T,F]

        output_tensor = self.mhsa(
            output_tensor,
            output_tensor,
            output_tensor,
            mask=att_mask,
            attn_dropout=attn_dropout_p,
            head_gates=head_gates,
            hard_prune=hard_prune,
        )  # [B,T,F]
        output_tensor = torch.nn.functional.dropout(
            output_tensor, p=dropout_p, training=self.training
        )  # [B,T,F]

        return output_tensor


class MultiheadAttention(torch.nn.Module):
    """Multi-Head Attention layer.

    Args:
        num_att_heads (int): The number of heads.
        input_dim (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, input_dim, num_att_heads, att_head_dim, dropout):
        """Construct an MultiHeadedAttention object."""
        super(MultiheadAttention, self).__init__()
        # assert input_dim % num_att_heads == 0
        # We assume d_v always equals d_k
        self.d_k = att_head_dim
        self.h = num_att_heads
        self.input_dim = input_dim

        self.linear_q_weight = torch.nn.parameter.Parameter(
            torch.empty((self.h * self.d_k, input_dim))
        )
        self.linear_q_bias = torch.nn.parameter.Parameter(
            torch.empty(self.h * self.d_k)
        )
        self.linear_k_weight = torch.nn.parameter.Parameter(
            torch.empty((self.h * self.d_k, input_dim))
        )
        self.linear_k_bias = torch.nn.parameter.Parameter(
            torch.empty(self.h * self.d_k)
        )
        self.linear_v_weight = torch.nn.parameter.Parameter(
            torch.empty((self.h * self.d_k, input_dim))
        )
        self.linear_v_bias = torch.nn.parameter.Parameter(
            torch.empty(self.h * self.d_k)
        )
        self.linear_out_weight = torch.nn.parameter.Parameter(
            torch.empty((input_dim, self.h * self.d_k))
        )
        self.linear_out_bias = torch.nn.parameter.Parameter(torch.empty(input_dim))

        self.attn = None
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """same initialization as default pytorch linear layer"""
        for weight, bias in [
            (self.linear_q_weight, self.linear_q_bias),
            (self.linear_k_weight, self.linear_k_bias),
            (self.linear_v_weight, self.linear_v_bias),
            (self.linear_out_weight, self.linear_out_bias),
        ]:
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
            q = F.linear(query, self.linear_q_weight, self.linear_q_bias).view(
                n_batch, -1, self.h, self.d_k
            )
            k = F.linear(key, self.linear_k_weight, self.linear_k_bias).view(
                n_batch, -1, self.h, self.d_k
            )
            v = F.linear(value, self.linear_v_weight, self.linear_v_bias).view(
                n_batch, -1, self.h, self.d_k
            )
        else:
            input_dim = self.linear_q_weight.size()[1]
            q_weights = torch.reshape(
                self.linear_q_weight, (-1, self.d_k, input_dim)
            )  # (h, d_k, F)
            q_bias = torch.reshape(self.linear_q_bias, (-1, self.d_k))  # (h, d_k)
            k_weights = torch.reshape(
                self.linear_k_weight, (-1, self.d_k, input_dim)
            )  # (h, d_k, F)
            k_bias = torch.reshape(self.linear_k_bias, (-1, self.d_k))  # (h, d_k)
            v_weights = torch.reshape(
                self.linear_v_weight, (-1, self.d_k, input_dim)
            )  # (h, d_k, F)
            v_bias = torch.reshape(self.linear_v_bias, (-1, self.d_k))  # (h, d_k)
            if not hard_prune and self.training:
                q_weights = torch.einsum(
                    "hij,h->hij", q_weights, head_gates
                )  # (h, d_k, h*d_k)
                q_bias = torch.einsum("hi,h->hi", q_bias, head_gates)  # (h, d_k)
                k_weights = torch.einsum(
                    "hij,h->hij", k_weights, head_gates
                )  # (h, d_k, h*d_k)
                k_bias = torch.einsum("hi,h->hi", k_bias, head_gates)  # (h, d_k)
                v_weights = torch.einsum(
                    "hij,h->hij", v_weights, head_gates
                )  # (h, d_k, h*d_k)
                v_bias = torch.einsum("hi,h->hi", v_bias, head_gates)  # (h, d_k)
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

            q = F.linear(query, q_weights, q_bias).view(
                n_batch, -1, num_heads, self.d_k
            )
            k = F.linear(key, k_weights, k_bias).view(n_batch, -1, num_heads, self.d_k)
            v = F.linear(value, v_weights, v_bias).view(
                n_batch, -1, num_heads, self.d_k
            )

        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(
        self, value, scores, mask, att_drop, head_gates=None, hard_prune: bool = False
    ):
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
            self.attn, p=att_drop, training=self.training
        )
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)

        num_heads = self.h
        out_weights = self.linear_out_weight
        # ===============================================
        if head_gates is not None:
            out_weights = torch.reshape(
                self.linear_out_weight, (self.input_dim, self.d_k, -1)
            )  # (F, d_k, h')

            if not hard_prune and self.training:
                x = torch.einsum("bhtd,h->bhtd", x, head_gates)
                out_weights = torch.einsum(
                    "ijh,h->ijh", out_weights, head_gates
                )  # (F, d_k, h')
            else:
                num_heads = torch.count_nonzero(head_gates)
                if not x.size()[1] == num_heads:
                    x = x[:, head_gates.bool()]
                out_weights = out_weights[:, :, head_gates.bool()]  # (F, d_k, h')

            out_weights = torch.reshape(out_weights, (self.input_dim, -1))
        # ===============================================

        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, num_heads * self.d_k)
        )  # (batch, time1, d_model)

        return F.linear(x, out_weights, self.linear_out_bias)  # (batch, time1, d_model)

    def forward(
        self,
        query,
        key,
        value,
        mask,
        attn_dropout,
        head_gates=None,
        hard_prune: bool = False,
    ):
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
        return self.forward_attention(
            v, scores, mask, attn_dropout, head_gates, hard_prune
        )

    # queue, key, value
    @staticmethod
    def reshape_and_filter(weight, bias, head_gates, h, d_k, input_dim):
        reshaped_weight = weight.reshape(h, d_k, input_dim)[head_gates.bool(), :, :]
        filtered_bias = bias.reshape(h, d_k)[head_gates.bool(), :]
        return reshaped_weight, filtered_bias

    def adapt_module(self, new_num_att_heads: int, head_gates: torch.tensor):
        if not (new_num_att_heads == self.h and torch.all(head_gates)):
            if new_num_att_heads >= len(head_gates):
                new_heads_gates = torch.cat(
                    (
                        head_gates,
                        torch.zeros(new_num_att_heads - self.h, dtype=torch.bool),
                    )
                )
            else:
                if new_num_att_heads > torch.sum(head_gates):
                    new_heads_gates = torch.tensor(
                        [True] * torch.sum(head_gates)
                        + [False] * (new_num_att_heads - torch.sum(head_gates))
                    )
                else:
                    new_heads_gates = torch.tensor([True] * new_num_att_heads)

            # Process the linear weights and biases for query, key, and value
            temp_linear_q_weight, temp_linear_q_bias = self.reshape_and_filter(
                self.linear_q_weight,
                self.linear_q_bias,
                head_gates,
                self.h,
                self.d_k,
                self.input_dim,
            )
            temp_linear_k_weight, temp_linear_k_bias = self.reshape_and_filter(
                self.linear_k_weight,
                self.linear_k_bias,
                head_gates,
                self.h,
                self.d_k,
                self.input_dim,
            )
            temp_linear_v_weight, temp_linear_v_bias = self.reshape_and_filter(
                self.linear_v_weight,
                self.linear_v_bias,
                head_gates,
                self.h,
                self.d_k,
                self.input_dim,
            )

            # out linear projection
            temp_linear_out_weight = torch.reshape(
                self.linear_out_weight, (self.input_dim, self.h, self.d_k)
            )
            temp_linear_out_weight = temp_linear_out_weight[:, head_gates.bool(), :]

            with torch.no_grad():
                self.linear_q_weight = torch.nn.parameter.Parameter(
                    torch.empty((new_num_att_heads * self.d_k, self.input_dim))
                )
                self.linear_q_bias = torch.nn.parameter.Parameter(
                    torch.empty(new_num_att_heads * self.d_k)
                )
                self.linear_k_weight = torch.nn.parameter.Parameter(
                    torch.empty((new_num_att_heads * self.d_k, self.input_dim))
                )
                self.linear_k_bias = torch.nn.parameter.Parameter(
                    torch.empty(new_num_att_heads * self.d_k)
                )
                self.linear_v_weight = torch.nn.parameter.Parameter(
                    torch.empty((new_num_att_heads * self.d_k, self.input_dim))
                )
                self.linear_v_bias = torch.nn.parameter.Parameter(
                    torch.empty(new_num_att_heads * self.d_k)
                )
                self.linear_out_weight = torch.nn.parameter.Parameter(
                    torch.empty((self.input_dim, new_num_att_heads * self.d_k))
                )

                self.reset_parameters()

                if new_num_att_heads > 0:
                    # queue, key ,value
                    linear_q_weight = torch.reshape(
                        self.linear_q_weight,
                        (new_num_att_heads, self.d_k, self.input_dim),
                    ).clone()
                    linear_q_weight[new_heads_gates] = temp_linear_q_weight.to(
                        linear_q_weight.device
                    )
                    self.linear_q_weight = torch.nn.parameter.Parameter(
                        torch.reshape(linear_q_weight, (-1, self.input_dim))
                    )

                    linear_q_bias = torch.reshape(
                        self.linear_q_bias, (new_num_att_heads, self.d_k)
                    ).clone()
                    linear_q_bias[new_heads_gates] = temp_linear_q_bias.to(
                        linear_q_bias.device
                    )
                    self.linear_q_bias = torch.nn.parameter.Parameter(
                        torch.flatten(linear_q_bias)
                    )

                    linear_k_weight = torch.reshape(
                        self.linear_k_weight,
                        (new_num_att_heads, self.d_k, self.input_dim),
                    ).clone()
                    linear_k_weight[new_heads_gates] = temp_linear_k_weight.to(
                        linear_k_weight.device
                    )
                    self.linear_k_weight = torch.nn.parameter.Parameter(
                        torch.reshape(linear_k_weight, (-1, self.input_dim))
                    )

                    linear_k_bias = torch.reshape(
                        self.linear_k_bias, (new_num_att_heads, self.d_k)
                    ).clone()
                    linear_k_bias[new_heads_gates] = temp_linear_k_bias.to(
                        linear_k_bias.device
                    )
                    self.linear_k_bias = torch.nn.parameter.Parameter(
                        torch.flatten(linear_k_bias)
                    )

                    linear_v_weight = torch.reshape(
                        self.linear_v_weight,
                        (new_num_att_heads, self.d_k, self.input_dim),
                    ).clone()
                    linear_v_weight[new_heads_gates] = temp_linear_v_weight.to(
                        linear_v_weight.device
                    )
                    self.linear_v_weight = torch.nn.parameter.Parameter(
                        torch.reshape(linear_v_weight, (-1, self.input_dim))
                    )

                    linear_v_bias = torch.reshape(
                        self.linear_v_bias, (new_num_att_heads, self.d_k)
                    ).clone()
                    linear_v_bias[new_heads_gates] = temp_linear_v_bias.to(
                        linear_v_bias.device
                    )
                    self.linear_v_bias = torch.nn.parameter.Parameter(
                        torch.flatten(linear_v_bias)
                    )

                    linear_out_weight = torch.reshape(
                        self.linear_out_weight,
                        (self.input_dim, new_num_att_heads, self.d_k),
                    ).clone()
                    linear_out_weight[:, new_heads_gates] = temp_linear_out_weight.to(
                        linear_out_weight.device
                    )
                    self.linear_out_weight = torch.nn.parameter.Parameter(
                        torch.reshape(linear_out_weight, (self.input_dim, -1))
                    )

                self.h = new_num_att_heads

    def double_and_prune_params(
        self, new_num_att_heads: int, selected_indices: list, weight_noise: float = 0
    ):
        if new_num_att_heads > 0:
            # Process the linear weights and biases for query, key, and value
            temp_linear_q_weight = torch.reshape(
                self.linear_q_weight, (self.h, self.d_k, self.input_dim)
            )
            temp_linear_q_bias = torch.reshape(self.linear_q_bias, (self.h, self.d_k))

            temp_linear_k_weight = torch.reshape(
                self.linear_k_weight, (self.h, self.d_k, self.input_dim)
            )
            temp_linear_k_bias = torch.reshape(self.linear_k_bias, (self.h, self.d_k))

            temp_linear_v_weight = torch.reshape(
                self.linear_v_weight, (self.h, self.d_k, self.input_dim)
            )
            temp_linear_v_bias = torch.reshape(self.linear_v_bias, (self.h, self.d_k))

            # out linear projection
            temp_linear_out_weight = torch.reshape(
                self.linear_out_weight, (self.input_dim, self.h, self.d_k)
            )

        with torch.no_grad():
            self.linear_q_weight = torch.nn.parameter.Parameter(
                torch.empty((new_num_att_heads * self.d_k, self.input_dim))
            )
            self.linear_q_bias = torch.nn.parameter.Parameter(
                torch.empty(new_num_att_heads * self.d_k)
            )
            self.linear_k_weight = torch.nn.parameter.Parameter(
                torch.empty((new_num_att_heads * self.d_k, self.input_dim))
            )
            self.linear_k_bias = torch.nn.parameter.Parameter(
                torch.empty(new_num_att_heads * self.d_k)
            )
            self.linear_v_weight = torch.nn.parameter.Parameter(
                torch.empty((new_num_att_heads * self.d_k, self.input_dim))
            )
            self.linear_v_bias = torch.nn.parameter.Parameter(
                torch.empty(new_num_att_heads * self.d_k)
            )
            self.linear_out_weight = torch.nn.parameter.Parameter(
                torch.empty((self.input_dim, new_num_att_heads * self.d_k))
            )

            self.reset_parameters()

            if new_num_att_heads > 0:
                linear_q_weight = temp_linear_q_weight[selected_indices, :, :]
                self.linear_q_weight.data = torch.reshape(
                    linear_q_weight, (-1, self.input_dim)
                )
                linear_q_bias = temp_linear_q_bias[selected_indices, :]
                self.linear_q_bias.data = torch.flatten(linear_q_bias)

                linear_k_weight = temp_linear_k_weight[selected_indices, :, :]
                self.linear_k_weight.data = torch.reshape(
                    linear_k_weight, (-1, self.input_dim)
                )
                linear_k_bias = temp_linear_k_bias[selected_indices, :]
                self.linear_k_bias.data = torch.flatten(linear_k_bias)

                linear_v_weight = temp_linear_v_weight[selected_indices, :, :]
                self.linear_v_weight.data = torch.reshape(
                    linear_v_weight, (-1, self.input_dim)
                )
                linear_v_bias = temp_linear_v_bias[selected_indices, :]
                self.linear_v_bias.data = torch.flatten(linear_v_bias)

                linear_out_weight = temp_linear_out_weight[:, selected_indices, :]
                self.linear_out_weight.data = torch.reshape(
                    linear_out_weight, (self.input_dim, -1)
                )

                if weight_noise > 0:
                    self.linear_q_weight.data += (weight_noise**0.5) * torch.randn(
                        self.linear_q_weight.data.shape
                    ).to("cuda")
                    self.linear_q_bias.data += (weight_noise**0.5) * torch.randn(
                        self.linear_q_bias.data.shape
                    ).to("cuda")
                    self.linear_k_weight.data += (weight_noise**0.5) * torch.randn(
                        self.linear_k_weight.data.shape
                    ).to("cuda")
                    self.linear_k_bias.data += (weight_noise**0.5) * torch.randn(
                        self.linear_k_bias.data.shape
                    ).to("cuda")
                    self.linear_v_weight.data += (weight_noise**0.5) * torch.randn(
                        self.linear_v_weight.data.shape
                    ).to("cuda")
                    self.linear_v_bias.data += (weight_noise**0.5) * torch.randn(
                        self.linear_v_bias.data.shape
                    ).to("cuda")
                    self.linear_out_weight.data += (weight_noise**0.5) * torch.randn(
                        self.linear_out_weight.data.shape
                    ).to("cuda")

            self.h = new_num_att_heads
