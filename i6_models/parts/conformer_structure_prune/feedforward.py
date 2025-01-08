from __future__ import annotations

__all__ = [
    "ConformerPositionwiseFeedForwardV1",
    "ConformerPositionwiseFeedForwardV1Config",
]

from dataclasses import dataclass
from typing import Callable
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from i6_models.config import ModelConfiguration


@dataclass
class ConformerPositionwiseFeedForwardV1Config(ModelConfiguration):
    """
    Attributes:
        input_dim: input dimension
        hidden_dim: hidden dimension (normally set to 4*input_dim as suggested by the paper)
        dropout: dropout probability
        activation: activation function
    """

    input_dim: int
    hidden_dim: int
    dropout: float
    activation: Callable[[torch.Tensor], torch.Tensor] = nn.functional.silu


class ConformerPositionwiseFeedForwardV1(nn.Module):
    """
    Conformer feedforward module
    """

    def __init__(self, cfg: ConformerPositionwiseFeedForwardV1Config):
        super().__init__()

        self.layer_norm = nn.LayerNorm(cfg.input_dim)
        self.linear_ff_weight = torch.nn.parameter.Parameter(torch.empty((cfg.hidden_dim, cfg.input_dim)))
        self.linear_ff_bias = torch.nn.parameter.Parameter(torch.empty(cfg.hidden_dim))
        self.activation = cfg.activation
        self.linear_out_weight = torch.nn.parameter.Parameter(torch.empty((cfg.input_dim, cfg.hidden_dim)))
        self.linear_out_bias = torch.nn.parameter.Parameter(torch.empty(cfg.input_dim))
        self.dropout = cfg.dropout
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """ same initialization as default pytorch linear layer"""
        init.kaiming_uniform_(self.linear_ff_weight, a=math.sqrt(5))
        if self.linear_ff_bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.linear_ff_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.linear_ff_bias, -bound, bound)

        init.kaiming_uniform_(self.linear_out_weight, a=math.sqrt(5))
        if self.linear_out_bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.linear_out_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.linear_out_bias, -bound, bound)

    def forward(self, tensor: torch.Tensor, channel_chunk_gates: torch.Tensor = None, hard_prune: bool=False) -> torch.Tensor:
        """
        :param tensor: shape [B,T,F], F=input_dim
        :return: shape [B,T,F], F=input_dim
        """
        input_dim, hidden_dim = self.linear_out_weight.size()
        if channel_chunk_gates is not None:
            assert channel_chunk_gates.size()[0] == hidden_dim//input_dim, "size dispatch"
        tensor = self.layer_norm(tensor)
        if channel_chunk_gates is None:
            tensor = F.linear(tensor, self.linear_ff_weight, self.linear_ff_bias)
        else:
            weights = torch.reshape(self.linear_ff_weight, (-1, input_dim, input_dim)) # (C, F, F)
            bias = torch.reshape(self.linear_ff_bias, (-1, input_dim))  # (C, F)
            if not hard_prune and self.training:
                weights = torch.einsum('cij,c->cij', weights, channel_chunk_gates)  # (C, F, F)
                bias = torch.einsum('ci,c->ci', bias, channel_chunk_gates)  # (C, F)
            else:
                weights = weights[channel_chunk_gates.bool()]
                bias = bias[channel_chunk_gates.bool()]
            weights = torch.reshape(weights, (-1, input_dim))  # (C*F, F)
            if hard_prune:
                num_chunks = torch.count_nonzero(channel_chunk_gates)
            else:
                num_chunks = hidden_dim//input_dim
            bias = torch.reshape(bias, (num_chunks*input_dim,))  # (C*F, 1)
            tensor = F.linear(tensor, weights, bias)

        tensor = self.activation(tensor)  # [B,T,F]
        tensor = nn.functional.dropout(
            tensor, p=self.dropout, training=self.training
        )  # [B,T,F]
        if channel_chunk_gates is None:
            tensor = F.linear(tensor, self.linear_out_weight, self.linear_out_bias)
        else:
            weights = torch.reshape(self.linear_out_weight, (input_dim, input_dim, -1)) # (F, F, 4)
            if not hard_prune and self.training:
                weights = torch.einsum('ijc,c->ijc', weights, channel_chunk_gates)  # (F, F, C)
                weights = torch.reshape(weights, (input_dim, -1))  # (F, C*F)
            else:
                weights = weights[:, :, channel_chunk_gates.bool()]
                weights = torch.reshape(weights, (input_dim, -1))
            tensor = F.linear(tensor, weights, self.linear_out_bias)

        tensor = nn.functional.dropout(
            tensor, p=self.dropout, training=self.training
        )  # [B,T,F]
        return tensor
