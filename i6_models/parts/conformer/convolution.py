from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from i6_models.config import ModelConfiguration
from typing import Callable, Union


@dataclass
class ConformerConvolutionV1Config(ModelConfiguration):
    channels: int
    """number of channels for conv layers"""
    kernel_size: int
    """kernel size of conv layers"""
    dropout: float
    """dropout probability"""
    activation: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]
    """activation function applied after norm"""
    norm: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]
    """normalization layer with input of shape [N,C,T]"""


class ConformerConvolutionV1(nn.Module):
    """
    Conformer convolution module.
    see also: https://github.com/espnet/espnet/blob/713e784c0815ebba2053131307db5f00af5159ea/espnet/nets/pytorch_backend/conformer/convolution.py#L13
    """

    def __init__(self, model_cfg: ConformerConvolutionV1Config):
        """
        :param model_cfg: model configuration for this module
        """
        super().__init__()

        self.pointwise_conv1 = nn.Linear(in_features=model_cfg.channels, out_features=2 * model_cfg.channels)
        self.depthwise_conv = nn.Conv1d(
            in_channels=model_cfg.channels,
            out_channels=model_cfg.channels,
            kernel_size=model_cfg.kernel_size,
            padding="same",
            groups=model_cfg.channels,
        )
        self.pointwise_conv2 = nn.Linear(in_features=model_cfg.channels, out_features=model_cfg.channels)
        self.layer_norm = nn.LayerNorm(model_cfg.channels)
        self.norm = model_cfg.norm
        self.dropout = nn.Dropout(model_cfg.dropout)
        self.activation = model_cfg.activation

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        :param tensor: input tensor of shape [B,T,F]
        :return: torch.Tensor of shape [B,T,F]
        """
        tensor = self.layer_norm(tensor)
        tensor = self.pointwise_conv1(tensor)  # [B,T,2F]
        tensor = nn.functional.glu(tensor, dim=-1)  # [B,T,F]

        # conv layers expect shape [B,F,T] so we have to transpose here
        tensor = tensor.transpose(1, 2)  # [B,F,T]
        tensor = self.depthwise_conv(tensor)

        tensor = self.norm(tensor)
        tensor = tensor.transpose(1, 2)  # transpose back to [B,T,F]

        tensor = self.activation(tensor)
        tensor = self.pointwise_conv2(tensor)

        return self.dropout(tensor)
