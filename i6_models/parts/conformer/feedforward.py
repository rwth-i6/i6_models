from __future__ import annotations
import torch
from torch import nn


class ConformerPositionwiseFeedForward(nn.Module):
    """
    Conformer feedforward module
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        """
        :param input_dim: input dimension
        :param hidden_dim: hidden dimension (normally set to 4*input_dim as suggested by the paper)
        :param dropout: dropout probability
        """
        super().__init__()

        self.linear_ff = nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=True)

        self.swish_activation = nn.SiLU()

        self.linear_out = nn.Linear(in_features=hidden_dim, out_features=input_dim, bias=True)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tensor: torch.Tensor):
        """
        :param torch.Tensor tensor: input tensor of shape [B,T,F]
        :return: torch.Tensor of shape [B,T,F]
        """
        out_tensor = self.linear_ff(tensor)  # [B,T,F]

        out_tensor = self.swish_activation(out_tensor)  # [B,T,F]

        out_tensor = self.linear_out(out_tensor)  # [B,T,F]

        out_tensor = self.dropout(out_tensor)  # [B,T,F]

        return tensor
