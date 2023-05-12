from __future__ import annotations
from typing import Callable
import torch
from torch import nn


class ConformerPositionwiseFeedForwardV1(nn.Module):
    """
    Conformer feedforward module
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        activation: Callable[[torch.Tensor], torch.Tensor] = nn.functional.silu,
    ):
        """
        :param input_dim: input dimension
        :param hidden_dim: hidden dimension (normally set to 4*input_dim as suggested by the paper)
        :param dropout: dropout probability
        :param activation: activation function
        """
        super().__init__()

        self.layer_norm = nn.LayerNorm(input_dim)
        self.linear_ff = nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=True)
        self.activation = activation
        self.linear_out = nn.Linear(in_features=hidden_dim, out_features=input_dim, bias=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        :param tensor: shape [B,T,F], F=input_dim
        :return: shape [B,T,F], F=input_dim
        """
        tensor = self.layer_norm(tensor)
        tensor = self.linear_ff(tensor)  # [B,T,F]
        tensor = self.activation(tensor)  # [B,T,F]
        tensor = self.dropout(tensor)  # [B,T,F]
        tensor = self.linear_out(tensor)  # [B,T,F]
        tensor = self.dropout(tensor)  # [B,T,F]
        return tensor
