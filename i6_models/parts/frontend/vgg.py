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


IntTupleInt = Union[int, Tuple[int, int]]


@dataclass
class VGG4LayerActFrontendV1Config(ModelConfiguration):
    """
    Attributes:
        conv1_channels: number of channels for first conv layers
        conv2_channels: number of channels for second conv layers
        conv3_channels: number of channels for third conv layers
        conv4_channels: number of channels for fourth dconv layers
        conv_kernel_size: kernel size of conv layers
        conv_padding: padding for the convolution
        pool1_kernel_size: kernel size of first pooling layer
        pool1_stride: stride of first pooling layer
        pool1_padding: padding for first pooling layer
        pool2_kernel_size: kernel size of second pooling layer
        pool2_stride: stride of second pooling layer
        pool2_padding: padding for second pooling layer
        activation: activation function at the end
    """

    conv1_channels: int
    conv2_channels: int
    conv3_channels: int
    conv4_channels: int
    conv_kernel_size: IntTupleInt
    conv_padding: Optional[IntTupleInt]
    pool1_kernel_size: IntTupleInt
    pool1_stride: Optional[IntTupleInt]
    pool1_padding: Optional[IntTupleInt]
    pool2_kernel_size: IntTupleInt
    pool2_stride: Optional[IntTupleInt]
    pool2_padding: Optional[IntTupleInt]
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

    The frond-end utilizes convolutional and pooling layers, as well as activation functions
    to transform a feature vector, typically Log-Mel or Gammatone for audio, into an intermediate
    representation.

    Structure of the front-end:
      - Conv
      - Conv
      - Activation
      - Pool
      - Conv
      - Conv
      - Activation
      - Pool

    Uses explicit padding for ONNX exportability, see:
    https://github.com/pytorch/pytorch/issues/68880
    """

    def __init__(self, model_cfg: VGG4LayerActFrontendV1Config):
        """
        :param model_cfg: model configuration for this module
        """
        super().__init__()

        model_cfg.check_valid()

        conv_padding = (
            model_cfg.conv_padding if model_cfg.conv_padding is not None else _get_padding(model_cfg.conv_kernel_size)
        )
        pool1_padding = model_cfg.pool1_padding if model_cfg.pool1_padding is not None else 0
        pool2_padding = model_cfg.pool2_padding if model_cfg.pool2_padding is not None else 0

        pool1_stride = model_cfg.pool1_stride
        self.time1_red = pool1_stride[0] if isinstance(pool1_stride, tuple) else pool1_stride
        pool2_stride = model_cfg.pool2_stride
        self.time2_red = pool2_stride[0] if isinstance(pool2_stride, tuple) else pool2_stride

        self.conv1 = nn.Conv2d(
            in_channels=1,
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
            stride=model_cfg.pool1_stride,
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
            stride=model_cfg.pool2_stride,
            padding=pool2_padding,
        )
        self.activation = model_cfg.activation

    def forward(self, tensor: torch.Tensor, sequence_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        T might be reduced to T' or T'' depending on stride of the layers

        :param tensor: input tensor of shape [B,T,F]
        :param sequence_mask: the sequence mask for the tensor
        :return: torch.Tensor of shape [B,T",F'] and the shape of the sequence mask
        """
        # and add a dim
        tensor = tensor[:, None, :, :]  # [B,C=1,T,F]

        tensor = self.conv1(tensor)
        tensor = self.conv2(tensor)

        tensor = self.activation(tensor)
        tensor = self.pool1(tensor)  # [B,C,T',F]
        sequence_mask = sequence_mask // self.time1_red

        tensor = self.conv3(tensor)
        tensor = self.conv4(tensor)

        tensor = self.activation(tensor)
        tensor = self.pool2(tensor)  # [B,C,T",F]
        sequence_mask = sequence_mask // self.time2_red

        tensor = torch.transpose(tensor, 1, 2)  # transpose to [B,T",C,F]
        tensor = torch.flatten(tensor, start_dim=2, end_dim=-1)  # [B,T",C*F]

        return tensor, sequence_mask


@dataclass
class VGG4LayerPoolFrontendV1Config(ModelConfiguration):
    """
    Attributes:
        conv1_channels: number of channels for first conv layers
        conv2_channels: number of channels for second conv layers
        conv3_channels: number of channels for third conv layers
        conv4_channels: number of channels for fourth conv layers
        conv_padding: padding for the convolution
        conv_kernel_size: kernel size of conv layers
        pool_kernel_size: kernel size of pooling layer
        pool_stride: stride of pooling layer
        pool_padding: padding for pooling layer
        activation: activation function at the end
    """

    conv1_channels: int
    conv2_channels: int
    conv3_channels: int
    conv4_channels: int
    conv_kernel_size: IntTupleInt
    conv_padding: Optional[IntTupleInt]
    pool_kernel_size: IntTupleInt
    pool_stride: Optional[IntTupleInt]
    pool_padding: Optional[IntTupleInt]
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

    The frond-end utilizes convolutional and pooling layers, as well as activation functions
    to transform a feature vector, typically Log-Mel or Gammatone for audio, into an intermediate
    representation.

    Structure of the front-end:
      - Conv
      - Activation
      - Pool
      - Conv
      - Activation
      - Conv
      - Activation
      - Conv

    Uses explicit padding for ONNX exportability, see:
    https://github.com/pytorch/pytorch/issues/68880
    """

    def __init__(self, model_cfg: VGG4LayerPoolFrontendV1Config):
        """
        :param model_cfg: model configuration for this module
        """
        super().__init__()

        model_cfg.check_valid()

        conv_padding = (
            model_cfg.conv_padding if model_cfg.conv_padding is not None else _get_padding(model_cfg.conv_kernel_size)
        )
        pool_padding = model_cfg.pool_padding if model_cfg.pool_padding is not None else 0

        pool_stride = model_cfg.pool_stride
        self.time_red = pool_stride[0] if isinstance(pool_stride, tuple) else pool_stride

        self.conv1 = nn.Conv2d(
            in_channels=1,
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
        self.pool = nn.MaxPool2d(
            kernel_size=model_cfg.pool_kernel_size,
            stride=model_cfg.pool_stride,
            padding=pool_padding,
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
        self.activation = model_cfg.activation

    def forward(self, tensor: torch.Tensor, sequence_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        T might be reduced to T' depending on the stride of the layers

        :param tensor: input tensor of shape [B,T,F]
        :param sequence_mask: masking tensor of shape [B,T], contains length information of the sequences
        :return: torch.Tensor of shape [B,T',F']
        """
        tensor = tensor[:, None, :, :]  # [B,C=1,T,F]

        tensor = self.conv1(tensor)
        tensor = self.activation(tensor)
        tensor = self.pool(tensor)  # [B,C,T',F]
        sequence_mask = sequence_mask // self.time_red

        tensor = self.conv2(tensor)
        tensor = self.activation(tensor)

        tensor = self.conv3(tensor)
        tensor = self.activation(tensor)

        tensor = self.conv4(tensor)

        tensor = torch.transpose(tensor, 1, 2)  # transpose to [B,T',C,F]
        tensor = torch.flatten(tensor, start_dim=2, end_dim=-1)  # [B,T',C*F]

        return tensor, sequence_mask


def _get_padding(input_size: Union[int, Tuple[int, ...]]) -> Union[int, Tuple[int, ...]]:
    """
    get padding in order to not reduce the time dimension
    :param input_size:
    :return:
    """
    if isinstance(input_size, int):
        return (input_size - 1) // 2
    elif isinstance(input_size, tuple):
        return tuple((s - 1) // 2 for s in input_size)
    else:
        raise TypeError(f"unexpected size type {type(input_size)}")


def _reduce_sequence_length(sequence_mask: torch.Tensor, reduction_factor: int) -> torch.Tensor:
    """
    reduce the sequence length stored in the sequence_mask by a certain factor

    :param sequence_mask: stored as [B,T] either as floats or bools
    :param reduction_factor:
    :return:
    """
    pass