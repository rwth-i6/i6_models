from __future__ import annotations

__all__ = []

from dataclasses import dataclass
from typing import Callable, Tuple, Union

import torch
from torch import nn

from i6_models.config import ModelConfiguration


@dataclass()
class ConformerVGGFrontendV1Config(ModelConfiguration):
    """
    Attributes:
        channels: number of channels for conv layers
        conv_kernel_size: kernel size of conv layers
        pool_1_kernel_size: kernel size of first pooling layer
        pool_1_strides: strides of first pooling layer
        pool_2_kernel_size: kernel size of second pooling layer
        pool_2_strides: strides of second pooling layer
        activation: activation function applied after norm
    """

    channels: int
    conv_kernel_size: int
    pool_1_kernel_size: int
    pool_1_strides: int
    pool_2_kernel_size: int
    pool_2_strides: int
    activation: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]

    def check_valid(self):
        assert self.conv_kernel_size % 2 == 1, "ConformerVGGFrontendV1 only supports odd kernel sizes"
        assert self.pool_1_kernel_size % 2 == 1, "ConformerVGGFrontendV1 only supports odd kernel sizes"
        assert self.pool_2_kernel_size % 2 == 1, "ConformerVGGFrontendV1 only supports odd kernel sizes"

    def __post__init__(self):
        super().__post_init__()
        self.check_valid()


class ConformerVGGFrontendV1(nn.Module):
    """
    Convolutional Front-End

    Uses explicit padding for ONNX exportability, see:
    https://github.com/pytorch/pytorch/issues/68880
    """

    def __init__(self, model_cfg: ConformerVGGFrontendV1Config):
        """
        :param model_cfg: model configuration for this module
        """
        super().__init__()

        model_cfg.check_valid()

        conv_padding = (model_cfg.conv_kernel_size - 1) // 2
        pool_1_padding = (model_cfg.pool_1_kernel_size - 1) // 2
        pool_2_padding = (model_cfg.pool_2_kernel_size - 1) // 2

        self.conv1 = nn.Conv2d(
            in_channels=model_cfg.channels,
            out_channels=model_cfg.channels,
            kernel_size=model_cfg.conv_kernel_size,
            padding=conv_padding,
        )
        self.conv2 = nn.Conv2d(
            in_channels=model_cfg.channels,
            out_channels=model_cfg.channels,
            kernel_size=model_cfg.conv_kernel_size,
            padding=conv_padding,
        )
        self.pooling1 = nn.MaxPool2d(
            kernel_size=model_cfg.pool_1_kernel_size,
            stride=model_cfg.pool_1_strides,
            padding=pool_1_padding,
        )
        self.conv3 = nn.Conv2d(
            in_channels=model_cfg.channels,
            out_channels=model_cfg.channels,
            kernel_size=model_cfg.conv_kernel_size,
            padding=conv_padding,
        )
        self.conv4 = nn.Conv2d(
            in_channels=model_cfg.channels,
            out_channels=model_cfg.channels,
            kernel_size=model_cfg.conv_kernel_size,
            padding=conv_padding,
        )
        self.pooling2 = nn.MaxPool2d(
            kernel_size=model_cfg.pool_2_kernel_size,
            stride=model_cfg.pool_2_strides,
            padding=pool_2_padding,
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
    """
    Attributes:
        channels: number of channels for conv layers
        conv_kernel_size: kernel size of conv layers
        pool_kernel_size: kernel size of first pooling layer
        pool_strides: strides of first pooling layer
        activation: activation function applied after norm
    """

    channels: int
    conv_kernel_size: int
    pool_kernel_size: int
    pool_strides: int
    activation: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]

    def check_valid(self):
        assert self.conv_kernel_size % 2 == 1, "ConformerVGGFrontendV2 only supports odd kernel sizes"
        assert self.pool_kernel_size % 2 == 1, "ConformerVGGFrontendV2 only supports odd kernel sizes"

    def __post__init__(self):
        super().__post_init__()
        self.check_valid()


class ConformerVGGFrontendV2(nn.Module):
    """
    Convolutional Front-End

    Uses explicit padding for ONNX exportability, see:
    https://github.com/pytorch/pytorch/issues/68880
    """

    def __init__(self, model_cfg: ConformerVGGFrontendV2Config):
        """
        :param model_cfg: model configuration for this module
        """
        super().__init__()

        model_cfg.check_valid()

        conv_padding = (model_cfg.conv_kernel_size - 1) // 2
        pool_padding = (model_cfg.pool_kernel_size - 1) // 2

        self.conv1 = nn.Conv2d(
            in_channels=model_cfg.channels,
            out_channels=model_cfg.channels,
            kernel_size=model_cfg.conv_kernel_size,
            padding=conv_padding,
        )
        self.conv2 = nn.Conv2d(
            in_channels=model_cfg.channels,
            out_channels=model_cfg.channels,
            kernel_size=model_cfg.conv_kernel_size,
            padding=conv_padding,
        )
        self.pooling = nn.MaxPool2d(
            kernel_size=model_cfg.pool_kernel_size,
            stride=model_cfg.pool_strides,
            padding=pool_padding,
        )
        self.conv3 = nn.Conv2d(
            in_channels=model_cfg.channels,
            out_channels=model_cfg.channels,
            kernel_size=model_cfg.conv_kernel_size,
            padding=conv_padding,
        )
        self.conv4 = nn.Conv2d(
            in_channels=model_cfg.channels,
            out_channels=model_cfg.channels,
            kernel_size=model_cfg.conv_kernel_size,
            padding=conv_padding,
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
