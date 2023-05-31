from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from i6_models.config import ModelConfiguration


@dataclass
class ConvolutionalGatingMLPV1Config(ModelConfiguration):
    input_dim: int
    """input dimension"""
    hidden_dim: int
    """hidden dimension (normally set to 6*input_dim as suggested by the paper)"""
    kernel_size: int
    """kernel size of the depth-wise convolution layer"""
    dropout: float
    """dropout probability"""
    activation: Callable[[torch.Tensor], torch.Tensor] = nn.functional.gelu
    """activation function"""

    def check_valid(self):
        assert self.kernel_size % 2 == 1, "ConvolutionalGatingMLPV1 only supports odd kernel sizes"

    def __post__init__(self):
        super().__post_init__()
        self.check_valid()


class ConvolutionalGatingMLPV1(nn.Module):
    """Convolutional Gating MLP (cgMLP)."""

    def __init__(self, cfg: ConvolutionalGatingMLPV1Config):
        super().__init__()

        self.layer_norm_input = nn.LayerNorm(cfg.input_dim)
        self.linear_ff = nn.Linear(in_features=cfg.input_dim, out_features=cfg.hidden_dim, bias=True)
        self.activation = cfg.activation
        self.layer_norm_csgu = nn.LayerNorm(cfg.hidden_dim // 2)
        self.depthwise_conv = nn.Conv1d(
            in_channels=cfg.hidden_dim // 2,
            out_channels=cfg.hidden_dim // 2,
            kernel_size=cfg.kernel_size,
            padding=(cfg.kernel_size - 1) // 2,
            groups=cfg.hidden_dim // 2,
        )
        self.linear_out = nn.Linear(in_features=cfg.hidden_dim // 2, out_features=cfg.input_dim, bias=True)
        self.dropout = cfg.dropout

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        :param tensor: shape [B,T,F], F=input_dim
        :return: shape [B,T,F], F=input_dim
        """
        tensor = self.layer_norm_input(tensor)  # (B,T,F)
        tensor = self.linear_ff(tensor)  # (B,T,6*F)
        tensor = self.activation(tensor)

        # convolutional spatial gating unit (csgu)
        tensor_r, tensor_g = tensor.chunk(2, dim=-1)  # (B,T,3*F), (B,T,3*F)
        tensor_g = self.layer_norm_csgu(tensor_g)
        # conv layers expect shape [B,F,T] so we have to transpose here
        tensor_g = tensor_g.transpose(1, 2)  # [B,3*F,T]
        tensor_g = self.depthwise_conv(tensor_g)
        tensor_g = tensor_g.transpose(1, 2)  # [B,T,3*F]
        tensor = tensor_r * tensor_g
        tensor = nn.functional.dropout(tensor, p=self.dropout, training=self.training)

        tensor = self.linear_out(tensor)  # [B,T,F]
        tensor = nn.functional.dropout(tensor, p=self.dropout, training=self.training)
        return tensor
