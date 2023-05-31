from __future__ import annotations

__all__ = []

from dataclasses import dataclass
from typing import Callable, Tuple, Union

import torch
from torch import nn

from i6_models.config import ModelConfiguration


@dataclass()
class ConformerVGGFrontendV1Config(ModelConfiguration):
    channels: int
    """number of channels for conv layers"""
    conv_kernel_size: int
    """kernel size of conv layers"""
    pool_1_kernel_size: int
    """kernel size of first pooling layer"""
    pool_1_strides: int
    """strides of first pooling layer"""
    pool_2_kernel_size: int
    """kernel size of second pooling layer"""
    pool_2_strides: int
    """strides of second pooling layer"""
    activation: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]
    """activation function applied after norm"""


class ConformerVGGFrontendV1(nn.Module):
    def __init__(self, model_cfg: ConformerVGGFrontendV1Config):
        """
        :param model_cfg: model configuration for this module
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=model_cfg.channels,
            out_channels=model_cfg.channels,
            kernel_size=model_cfg.conv_kernel_size,
            padding="same",
        )
        self.conv2 = nn.Conv2d(
            in_channels=model_cfg.channels,
            out_channels=model_cfg.channels,
            kernel_size=model_cfg.conv_kernel_size,
            padding="same",
        )
        self.pooling1 = nn.MaxPool2d(
            kernel_size=model_cfg.pool_1_kernel_size,
            stride=model_cfg.pool_1_strides,
            padding=1,
        )
        self.conv3 = nn.Conv2d(
            in_channels=model_cfg.channels,
            out_channels=model_cfg.channels,
            kernel_size=model_cfg.conv_kernel_size,
            padding="same",
        )
        self.conv4 = nn.Conv2d(
            in_channels=model_cfg.channels,
            out_channels=model_cfg.channels,
            kernel_size=model_cfg.conv_kernel_size,
            padding="same",
        )
        self.pooling2 = nn.MaxPool2d(
            kernel_size=model_cfg.pool_2_kernel_size,
            stride=model_cfg.pool_2_strides,
            padding=1,
        )
        self.activation = model_cfg.activation

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        T might be reduced to T' depending on ...

        :param tensor: input tensor of shape [B,T,F]
        :return: torch.Tensor of shape [B,T',F]
        """
        # conv layers expect shape [B,F,T] so we have to transpose here
        tensor = tensor.transpose(1, 2)  # [B,F,T]
        tensor = self.conv1(tensor)
        tensor = self.conv2(tensor)
        tensor = tensor.transpose(1, 2)  # transpose back to [B,T,F]

        tensor = self.activation(tensor)
        tensor = self.pooling1(tensor)

        # conv layers expect shape [B,F,T] so we have to transpose here
        tensor = tensor.transpose(1, 2)  # [B,F,T]
        tensor = tensor.transpose(1, 2)  # transpose back to [B,T,F]
        tensor = self.conv3(tensor)
        tensor = self.conv4(tensor)
        tensor = self.activation(tensor)
        tensor = self.pooling2(tensor)

        return tensor


@dataclass()
class ConformerVGGFrontendV2Config(ModelConfiguration):
    channels: int
    """number of channels for conv layers"""
    conv_kernel_size: int
    """kernel size of conv layers"""
    pool_kernel_size: int
    """kernel size of pooling layer"""
    pool_strides: int
    """strides of pooling layer"""
    activation: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]
    """activation function applied after norm"""


class ConformerVGGFrontendV2(nn.Module):
    def __init__(self, model_cfg: ConformerVGGFrontendV2Config):
        """
        :param model_cfg: model configuration for this module
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=model_cfg.channels,
            out_channels=model_cfg.channels,
            kernel_size=model_cfg.conv_kernel_size,
            padding="same",
        )
        self.conv2 = nn.Conv2d(
            in_channels=model_cfg.channels,
            out_channels=model_cfg.channels,
            kernel_size=model_cfg.conv_kernel_size,
            padding="same",
        )
        self.pooling = nn.MaxPool2d(
            kernel_size=model_cfg.pool_kernel_size,
            stride=model_cfg.pool_strides,
            padding=1,
        )
        self.conv3 = nn.Conv2d(
            in_channels=model_cfg.channels,
            out_channels=model_cfg.channels,
            kernel_size=model_cfg.conv_kernel_size,
            padding="same",
        )
        self.conv4 = nn.Conv2d(
            in_channels=model_cfg.channels,
            out_channels=model_cfg.channels,
            kernel_size=model_cfg.conv_kernel_size,
            padding="same",
        )
        self.activation = model_cfg.activation

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        T might be reduced to T' depending on ...

        :param tensor: input tensor of shape [B,T,F]
        :return: torch.Tensor of shape [B,T',F]
        """
        # conv layers expect shape [B,F,T] so we have to transpose here
        tensor = tensor.transpose(1, 2)  # [B,F,T]
        tensor = self.conv1(tensor)
        tensor = tensor.transpose(1, 2)  # transpose back to [B,T,F]

        tensor = self.activation(tensor)
        tensor = self.pooling(tensor)

        tensor = tensor.transpose(1, 2)  # [B,F,T]
        tensor = self.conv2(tensor)
        tensor = tensor.transpose(1, 2)  # transpose back to [B,T,F]

        tensor = self.activation(tensor)

        tensor = tensor.transpose(1, 2)  # [B,F,T]
        tensor = self.conv3(tensor)
        tensor = tensor.transpose(1, 2)  # transpose back to [B,T,F]

        tensor = self.activation(tensor)

        tensor = tensor.transpose(1, 2)  # [B,F,T]
        tensor = self.conv4(tensor)
        tensor = tensor.transpose(1, 2)  # transpose back to [B,T,F]

        return tensor
