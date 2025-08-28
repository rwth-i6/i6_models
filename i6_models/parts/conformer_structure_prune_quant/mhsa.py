from __future__ import annotations

__all__ = ["ConformerMHSAWithGateV1", "ConformerMHSAV1Config"]

from dataclasses import dataclass
import math
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch.nn import init

from i6_models.config import ModelConfiguration
from .weight_quantizer import WeightQuantizer, ActivationQuantizer


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
    weight_quant_dtype: torch.dtype
    weight_quant_method: str
    activation_quant_dtype: torch.dtype
    activation_quant_method: str
    dot_quant_dtype: torch.dtype
    dot_quant_method: str
    Av_quant_dtype: torch.dtype
    Av_quant_method: str
    bit_prec_W_q: Union[int, float]
    bit_prec_W_k: Union[int, float]
    bit_prec_W_v: Union[int, float]
    bit_prec_dot: Union[int, float]
    bit_prec_A_v: Union[int, float]
    bit_prec_W_o: Union[int, float]
    activation_bit_prec: Union[int, float]
    moving_average: Optional[float]  # Moving average for input quantization
    dropout: float

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.input_dim % self.num_att_heads == 0, "input_dim must be divisible by num_att_heads"


class ConformerMHSAWithGateV1(torch.nn.Module):
    """
    Conformer multi-headed self-attention module
    """

    def __init__(self, cfg: ConformerMHSAV1Config):
        super().__init__()

        self.layernorm = torch.nn.LayerNorm(cfg.input_dim)

        assert cfg.input_dim %cfg.num_att_heads == 0
        # We assume d_v always equals d_k
        self.d_k = cfg.input_dim // cfg.num_att_heads
        self.h = cfg.num_att_heads

        self.bit_prec_dot = cfg.bit_prec_dot
        self.bit_prec_Av = cfg.bit_prec_A_v
        self.weight_quant_dtype = cfg.weight_quant_dtype
        self.weight_quant_method = cfg.weight_quant_method
        self.activation_quant_dtype = cfg.activation_quant_dtype
        self.activation_quant_method = cfg.activation_quant_method
        self.dot_quant_dtype = cfg.dot_quant_dtype
        self.dot_quant_method = cfg.dot_quant_method
        self.Av_quant_dtype = cfg.Av_quant_dtype
        self.Av_quant_method = cfg.Av_quant_method

        self.linear_q_weight = torch.nn.parameter.Parameter(torch.empty((cfg.input_dim, cfg.input_dim)))
        self.linear_q_bias = torch.nn.parameter.Parameter(torch.empty(cfg.input_dim))
        self.linear_k_weight = torch.nn.parameter.Parameter(torch.empty((cfg.input_dim, cfg.input_dim)))
        self.linear_k_bias = torch.nn.parameter.Parameter(torch.empty(cfg.input_dim))
        self.linear_v_weight = torch.nn.parameter.Parameter(torch.empty((cfg.input_dim, cfg.input_dim)))
        self.linear_v_bias = torch.nn.parameter.Parameter(torch.empty(cfg.input_dim))
        self.linear_out_weight = torch.nn.parameter.Parameter(torch.empty((cfg.input_dim, cfg.input_dim)))
        self.linear_out_bias = torch.nn.parameter.Parameter(torch.empty(cfg.input_dim))

        self.input_quant = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.moving_average,
            observer_only_in_train=cfg.observer_only_in_train,
        )

        self.linear_q_weight_quantizer = WeightQuantizer(
            bit_precision=cfg.weight_bit_prec,
            dtype=cfg.weight_quant_dtype,
            method=cfg.weight_quant_method,
            observer_only_in_train=cfg.observer_only_in_train,
        )

        self.linear_k_weight_quantizer = WeightQuantizer(
            bit_precision=cfg.weight_bit_prec,
            dtype=cfg.weight_quant_dtype,
            method=cfg.weight_quant_method,
            observer_only_in_train=cfg.observer_only_in_train,
        )

        self.linear_v_weight_quantizer = WeightQuantizer(
            bit_precision=cfg.weight_bit_prec,
            dtype=cfg.weight_quant_dtype,
            method=cfg.weight_quant_method,
            observer_only_in_train=cfg.observer_only_in_train,
        )

        self.linear_out_weight_quantizer = WeightQuantizer(
            bit_precision=cfg.weight_bit_prec,
            dtype=cfg.weight_quant_dtype,
            method=cfg.weight_quant_method,
            observer_only_in_train=cfg.observer_only_in_train,
        )

        self.out_proj_in_quant = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.moving_average,
            observer_only_in_train=cfg.observer_only_in_train,
        )

        self.out_proj_out_quant = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.moving_average,
            observer_only_in_train=cfg.observer_only_in_train,
        )

        self.q_quantizer = ActivationQuantizer(
            self.bit_prec_dot,
            self.dot_quant_dtype,
            self.dot_quant_method,
            channel_axis=None if self.dot_quant_method == "per_tensor" else 3,
            moving_avrg=cfg.moving_average,
            observer_only_in_train=cfg.observer_only_in_train,
        )
        self.k_quantizer = ActivationQuantizer(
            self.bit_prec_dot,
            self.dot_quant_dtype,
            self.dot_quant_method,
            channel_axis=None if self.dot_quant_method == "per_tensor" else 2,
            moving_avrg=cfg.moving_average,
            observer_only_in_train=cfg.observer_only_in_train,
        )

        self.a_quantizer = WeightQuantizer(
            self.bit_prec_Av,
            self.Av_quant_dtype,
            self.Av_quant_method,
            observer_only_in_train=cfg.observer_only_in_train,
        )
        self.v_quantizer = ActivationQuantizer(
            self.bit_prec_Av,
            self.Av_quant_dtype,
            self.Av_quant_method,
            moving_avrg=cfg.moving_average,
            channel_axis=None if self.dot_quant_method == "per_tensor" else NotImplementedError,
            observer_only_in_train=cfg.observer_only_in_train,
        )

        self.attn = None
        self.reset_parameters()

        self.dropout = cfg.dropout
    
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

    def forward(
        self,
        input_tensor: torch.Tensor,
        sequence_mask: torch.Tensor,
        head_gates: torch.Tensor = None,
        hard_prune: bool = False,
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

        tensor = self.layernorm(input_tensor)  # [B,T,F]
        tensor = self.input_quant(tensor)

        query = key = value = tensor

        n_batch = query.size(0)
        if head_gates is None:
            q = F.linear(query, self.linear_q_weight_quantizer(self.linear_q_weight), self.linear_q_bias).view(n_batch, -1, self.h, self.d_k)
            k = F.linear(key, self.linear_k_weight_quantizer(self.linear_k_weight), self.linear_k_bias).view(n_batch, -1, self.h, self.d_k)
            v = F.linear(value, self.linear_v_weight_quantizer(self.linear_v_weight), self.linear_v_bias).view(n_batch, -1, self.h, self.d_k)
        else:
            input_dim = self.linear_q_weight.size()[1]
            q_weights = torch.reshape(self.linear_q_weight, (-1, self.d_k, input_dim))  # (h, d_k, F)
            q_bias = torch.reshape(self.linear_q_bias, (-1, self.d_k))  # (h, d_k)
            k_weights = torch.reshape(self.linear_k_weight, (-1, self.d_k, input_dim))  # (h, d_k, F)
            k_bias = torch.reshape(self.linear_k_bias, (-1, self.d_k))  # (h, d_k)
            v_weights = torch.reshape(self.linear_v_weight, (-1, self.d_k, input_dim))  # (h, d_k, F)
            v_bias = torch.reshape(self.linear_v_bias, (-1, self.d_k))  # (h, d_k)
            if not hard_prune and self.training:
                q_weights = torch.einsum("hij,h->hij", q_weights, head_gates)  # (h, d_k, h*d_k)
                q_bias = torch.einsum("hi,h->hi", q_bias, head_gates)  # (h, d_k)
                k_weights = torch.einsum("hij,h->hij", k_weights, head_gates)  # (h, d_k, h*d_k)
                k_bias = torch.einsum("hi,h->hi", k_bias, head_gates)  # (h, d_k)
                v_weights = torch.einsum("hij,h->hij", v_weights, head_gates)  # (h, d_k, h*d_k)
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

            q = F.linear(query, self.linear_q_weight_quantizer(q_weights), q_bias).view(n_batch, -1, num_heads, self.d_k)
            k = F.linear(key, self.linear_k_weight_quantizer(k_weights), k_bias).view(n_batch, -1, num_heads, self.d_k)
            v = F.linear(value, self.linear_v_weight_quantizer(v_weights), v_bias).view(n_batch, -1, num_heads, self.d_k)

        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        if self.bit_prec_dot < 16:
            q = self.q_quantizer(q)
            k = self.k_quantizer(k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        n_batch = value.size(0)
        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(att_mask, min_value)

            self.attn = torch.softmax(scores, dim=-1).masked_fill(att_mask, 0.0)  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        # p_attn = torch.nn.functional.dropout(self.attn, p=self.dropout, training=self.training)

        if self.bit_prec_Av < 16:
            alpha = self.a_quantizer(alpha)
            value = self.v_quantizer(value)

        x = torch.matmul(self.attn, value)  # (batch, head, time1, d_k)

        num_heads = self.h
        out_weights = self.linear_out_weight
        # ===============================================
        if head_gates is not None:
            input_dim = self.linear_out_weight.size()[1]
            out_weights = torch.reshape(self.linear_out_weight, (input_dim, self.d_k, -1))  # (F, d_k, h')

            if not hard_prune and self.training:
                x = torch.einsum("bhtd,h->bhtd", x, head_gates)
                out_weights = torch.einsum("ijH,h->ijH", out_weights, head_gates)  # (F, d_k, h')
            else:
                num_heads = torch.count_nonzero(head_gates)
                if not x.size()[1] == num_heads:
                    x = x[:, head_gates.bool()]
                out_weights = out_weights[:, :, head_gates.bool()]  # (F, d_k, h')

            out_weights = torch.reshape(out_weights, (input_dim, -1))
        # ===============================================

        x = x.transpose(1, 2).contiguous().view(n_batch, -1, num_heads * self.d_k)  # (batch, time1, d_model)


        x = self.out_proj_in_quant(x)
        out_tensor = F.linear(x, self.linear_out_weight_quantizer(out_weights), self.linear_out_bias)  # (batch, time1, d_model)
        out_tensor = self.out_proj_out_quant(out_tensor)

        out_tensor = torch.nn.functional.dropout(out_tensor, p=self.dropout, training=self.training)  # [B,T,F]

        return out_tensor
    
