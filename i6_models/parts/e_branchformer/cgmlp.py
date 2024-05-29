from __future__ import annotations

__all__ = ["ConvolutionalGatingMLPV1Config", "ConvolutionalGatingMLPV1"]

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from i6_models.config import ModelConfiguration


@dataclass
class ConvolutionalGatingMLPV1Config(ModelConfiguration):
    """
    Attributes:
        input_dim: input dimension
        hidden_dim: hidden dimension (normally set to 6*input_dim as suggested by the paper)
        kernel_size: kernel size of the depthwise convolution layer
        dropout: dropout probability
        activation: activation function
    """

    input_dim: int
    hidden_dim: int
    kernel_size: int
    dropout: float
    activation: Callable[[torch.Tensor], torch.Tensor] = nn.functional.gelu

    def check_valid(self):
        assert self.kernel_size % 2 == 1, "ConvolutionalGatingMLPV1 only supports odd kernel sizes"
        assert self.hidden_dim % 2 == 0, "ConvolutionalGatingMLPV1 only supports even hidden_dim"

    def __post__init__(self):
        super().__post_init__()
        self.check_valid()


class ConvolutionalGatingMLPV1(nn.Module):
    """Convolutional Gating MLP (cgMLP)."""

    def __init__(self, model_cfg: ConvolutionalGatingMLPV1Config):
        super().__init__()

        self.layer_norm_input = nn.LayerNorm(model_cfg.input_dim)
        self.linear_ff = nn.Linear(in_features=model_cfg.input_dim, out_features=model_cfg.hidden_dim, bias=True)
        self.activation = model_cfg.activation
        self.layer_norm_csgu = nn.LayerNorm(model_cfg.hidden_dim // 2)
        self.depthwise_conv = nn.Conv1d(
            in_channels=model_cfg.hidden_dim // 2,
            out_channels=model_cfg.hidden_dim // 2,
            kernel_size=model_cfg.kernel_size,
            padding=(model_cfg.kernel_size - 1) // 2,
            groups=model_cfg.hidden_dim // 2,
        )
        self.linear_out = nn.Linear(in_features=model_cfg.hidden_dim // 2, out_features=model_cfg.input_dim, bias=True)
        self.dropout = model_cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: shape [B, T, F], F=input_dim
        :return: shape [B, T, F], F=input_dim
        """
        x = self.layer_norm_input(x)  # [B, T, F]
        x = self.linear_ff(x)  # [B, T, F']
        x = self.activation(x)

        # convolutional spatial gating unit (csgu)
        x_1, x_2 = x.chunk(2, dim=-1)  # [B, T, F'//2], [B, T, F'//2]
        x_2 = self.layer_norm_csgu(x_2)
        # conv layers expect shape [B, F, T] so we have to transpose here
        x_2 = x_2.transpose(1, 2)  # [B, F'//2, T]
        x_2 = self.depthwise_conv(x_2)  # [B, F'//2, T]
        x_2 = x_2.transpose(1, 2)  # [B, T, F'//2]
        x = x_1 * x_2  # [B, T, F'//2]
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        x = self.linear_out(x)  # [B, T, F]
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return x
