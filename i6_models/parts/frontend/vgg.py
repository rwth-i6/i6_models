from __future__ import annotations

__all__ = [
    "VGG4LayerActFrontendV1",
    "VGG4LayerActFrontendV1Config",
    "VGG4LayerPoolFrontendV1",
    "VGG4LayerPoolFrontendV1Config",
]

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn

from i6_models.config import ModelConfiguration


@dataclass
class VGG4LayerActFrontendV1Config(ModelConfiguration):
    """
    Attributes:
        in_features: feature dimension of input
        conv1_channels: number of channels for first conv layers
        conv2_channels: number of channels for second conv layers
        conv3_channels: number of channels for third conv layers
        conv4_channels: number of channels for fourth dconv layers
        conv_kernel_size: kernel size of conv layers
        pool1_kernel_size: kernel size of first pooling layer
        pool1_strides: strides of first pooling layer
        pool2_kernel_size: kernel size of second pooling layer
        pool2_strides: strides of second pooling layer
        activation: activation function at the end
    """

    in_features: int
    conv1_channels: int
    conv2_channels: int
    conv3_channels: int
    conv4_channels: int
    conv_kernel_size: Union[int, Tuple[int, ...]]
    pool1_kernel_size: Union[int, Tuple[int, ...]]
    pool1_strides: Optional[Union[int, Tuple[int, ...]]]
    pool2_kernel_size: Union[int, Tuple[int, ...]]
    pool2_strides: Optional[Union[int, Tuple[int, ...]]]
    activation: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]

    def check_valid(self):
        if isinstance(self.conv_kernel_size, int):
            assert self.conv_kernel_size % 2 == 1, "ConformerVGGFrontendV1 only supports odd kernel sizes"
        if isinstance(self.pool1_kernel_size, int):
            assert self.pool1_kernel_size % 2 == 1, "ConformerVGGFrontendV1 only supports odd kernel sizes"
        if isinstance(self.pool2_kernel_size, int):
            assert self.pool2_kernel_size % 2 == 1, "ConformerVGGFrontendV1 only supports odd kernel sizes"

    def __post__init__(self):
        super().__post_init__()
        self.check_valid()


class VGG4LayerActFrontendV1(nn.Module):
    """
    Convolutional Front-End

    Uses explicit padding for ONNX exportability, see:
    https://github.com/pytorch/pytorch/issues/68880
    """

    def __init__(self, model_cfg: VGG4LayerActFrontendV1Config):
        """
        :param model_cfg: model configuration for this module
        """
        super().__init__()

        model_cfg.check_valid()

        conv_padding = _get_padding(model_cfg.conv_kernel_size)
        pool1_padding = _get_padding(model_cfg.pool1_kernel_size)
        pool2_padding = _get_padding(model_cfg.pool2_kernel_size)

        self.conv1 = nn.Conv2d(
            in_channels=model_cfg.in_features,
            out_channels=model_cfg.conv1_channels,
            kernel_size=model_cfg.conv_kernel_size,
            padding=conv_padding,
        )
        self.conv2 = nn.Conv2d(
            in_channels=model_cfg.conv1_channels,
            out_channels=model_cfg.conv2_channels,
            kernel_size=model_cfg.conv_kernel_size,
            padding=conv_padding,
        )
        self.pool1 = nn.MaxPool2d(
            kernel_size=model_cfg.pool1_kernel_size,
            stride=model_cfg.pool1_strides,
            padding=pool1_padding,
        )
        self.conv3 = nn.Conv2d(
            in_channels=model_cfg.conv2_channels,
            out_channels=model_cfg.conv3_channels,
            kernel_size=model_cfg.conv_kernel_size,
            padding=conv_padding,
        )
        self.conv4 = nn.Conv2d(
            in_channels=model_cfg.conv3_channels,
            out_channels=model_cfg.conv4_channels,
            kernel_size=model_cfg.conv_kernel_size,
            padding=conv_padding,
        )
        self.pool2 = nn.MaxPool2d(
            kernel_size=model_cfg.pool2_kernel_size,
            stride=model_cfg.pool2_strides,
            padding=pool2_padding,
        )
        self.activation = model_cfg.activation

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        T might be reduced to T' depending on ...

        :param tensor: input tensor of shape [B,T,F]
        :return: torch.Tensor of shape [B,T',F]
        """
        # conv 2d layers expect shape [B,F,T,C] so we have to transpose here
        tensor = torch.transpose(tensor, 1, 2)  # [B,F,T]
        # and add a dim
        tensor = tensor[:, None, :, :]  # [B,C=1,F,T]
        tensor = self.conv1(tensor)
        tensor = self.conv2(tensor)
        tensor = torch.transpose(tensor, 1, 2)  # transpose back to [B,T,F,C]
        tensor = torch.squeeze(tensor)  # [B,T,F]

        tensor = self.activation(tensor)
        tensor = self.pool1(tensor)

        # conv 2d layers expect shape [B,F,T,C] so we have to transpose here
        tensor = torch.transpose(tensor, 1, 2)  # [B,F,T]
        tensor = tensor[:, :, :, None]  # [B,F,T,C]
        tensor = self.conv3(tensor)
        tensor = self.conv4(tensor)
        tensor = torch.transpose(tensor, 1, 2)  # transpose back to [B,T,F,C]
        tensor = torch.squeeze(tensor)  # [B,T,F]

        tensor = self.activation(tensor)
        tensor = self.pool2(tensor)

        return tensor


@dataclass
class VGG4LayerPoolFrontendV1Config(ModelConfiguration):
    """
    Attributes:
        channels: number of channels for conv layers
        conv_kernel_size: kernel size of conv layers
        pool_kernel_size: kernel size of first pooling layer
        pool_strides: strides of first pooling layer
        activation: activation function at the end
    """

    features: int
    channels: int
    conv_kernel_size: Union[int, Tuple[int, ...]]
    pool_kernel_size: Union[int, Tuple[int, ...]]
    pool_strides: Optional[Union[int, Tuple[int, ...]]]
    activation: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]

    def check_valid(self):
        if isinstance(self.conv_kernel_size, int):
            assert self.conv_kernel_size % 2 == 1, "ConformerVGGFrontendV2 only supports odd kernel sizes"
        if isinstance(self.pool_kernel_size, int):
            assert self.pool_kernel_size % 2 == 1, "ConformerVGGFrontendV2 only supports odd kernel sizes"

    def __post__init__(self):
        super().__post_init__()
        self.check_valid()


class VGG4LayerPoolFrontendV1(nn.Module):
    """
    Convolutional Front-End

    Uses explicit padding for ONNX exportability, see:
    https://github.com/pytorch/pytorch/issues/68880
    """

    def __init__(self, model_cfg: VGG4LayerPoolFrontendV1Config):
        """
        :param model_cfg: model configuration for this module
        """
        super().__init__()

        model_cfg.check_valid()

        conv_padding = _get_padding(model_cfg.conv_kernel_size)
        pool_padding = _get_padding(model_cfg.pool_kernel_size)

        self.conv1 = nn.Conv2d(
            in_channels=model_cfg.features,
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
        self.pool = nn.MaxPool2d(
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
        tensor = torch.transpose(tensor, 1, 2)  # [B,F,T]
        tensor = tensor[:, :, :, None]  # [B,F,T,C]
        tensor = self.conv1(tensor)
        tensor = torch.transpose(tensor, 1, 2)  # transpose back to [B,T,F,C]
        tensor = torch.squeeze(tensor)  # [B,T,F]

        tensor = self.activation(tensor)
        tensor = self.pool(tensor)

        tensor = torch.transpose(tensor, 1, 2)  # [B,F,T]
        tensor = tensor[:, :, :, None]  # [B,F,T,C]
        tensor = self.conv2(tensor)
        tensor = torch.transpose(tensor, 1, 2)  # transpose back to [B,T,F,C]
        tensor = torch.squeeze(tensor)  # [B,T,F]

        tensor = self.activation(tensor)

        tensor = torch.transpose(tensor, 1, 2)  # [B,F,T]
        tensor = tensor[:, :, :, None]  # [B,F,T,C]
        tensor = self.conv3(tensor)
        tensor = torch.transpose(tensor, 1, 2)  # transpose back to [B,T,F,C]
        tensor = torch.squeeze(tensor)  # [B,T,F]

        tensor = self.activation(tensor)

        tensor = torch.transpose(tensor, 1, 2)  # [B,F,T]
        tensor = tensor[:, :, :, None]  # [B,F,T,C]
        tensor = self.conv4(tensor)
        tensor = torch.transpose(tensor, 1, 2)  # transpose back to [B,T,F,C]
        tensor = torch.squeeze(tensor)  # [B,T,F]

        return tensor


def _get_padding(input_size: Union[int, Tuple[int, ...]]) -> int:
    if isinstance(input_size, int):
        out = (input_size - 1) // 2
    elif isinstance(input_size, tuple):
        out = min(input_size) // 2
    else:
        raise TypeError(f"unexpected size type {type(input_size)}")
        
    return out
