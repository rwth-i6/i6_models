from __future__ import annotations

__all__ = ["ConformerPositionwiseFeedForwardV1", "ConformerPositionwiseFeedForwardV1Config"]

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

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
        self.linear_ff = nn.Linear(in_features=cfg.input_dim, out_features=cfg.hidden_dim, bias=True)
        self.activation = cfg.activation
        self.linear_out = nn.Linear(in_features=cfg.hidden_dim, out_features=cfg.input_dim, bias=True)
        self.dropout = cfg.dropout

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        :param tensor: shape [B,T,F], F=input_dim
        :return: shape [B,T,F], F=input_dim
        """
        tensor = self.layer_norm(tensor)
        tensor = self.linear_ff(tensor)  # [B,T,F]
        tensor = self.activation(tensor)  # [B,T,F]
        tensor = nn.functional.dropout(tensor, p=self.dropout, training=self.training)  # [B,T,F]
        tensor = self.linear_out(tensor)  # [B,T,F]
        tensor = nn.functional.dropout(tensor, p=self.dropout, training=self.training)  # [B,T,F]
        return tensor
