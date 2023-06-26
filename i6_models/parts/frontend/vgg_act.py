from __future__ import annotations

__all__ = [
    "VGG4LayerActFrontendV1",
    "VGG4LayerActFrontendV1Config",
]

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn

from i6_models.config import ModelConfiguration

from .common import IntTupleIntType, _get_padding, _mask_pool, _get_int_tuple_int


@dataclass
class VGG4LayerActFrontendV1Config(ModelConfiguration):
    """
    Attributes:
        conv1_channels: number of channels for first conv layer
        conv2_channels: number of channels for second conv layer
        conv3_channels: number of channels for third conv layer
        conv4_channels: number of channels for fourth conv layer
        conv_kernel_size: kernel size of conv layers
        conv_padding: padding for the convolution
        pool1_kernel_size: kernel size of first pooling layer
        pool1_stride: stride of first pooling layer
        pool1_padding: padding for first pooling layer
        pool2_kernel_size: kernel size of second pooling layer
        pool2_stride: stride of second pooling layer
        pool2_padding: padding for second pooling layer
        activation: activation function at the end
        linear_input_dim: input size of the final linear layer
        linear_output_dim: output size of the final linear layer
    """

    conv1_channels: int
    conv2_channels: int
    conv3_channels: int
    conv4_channels: int
    conv_kernel_size: IntTupleIntType
    conv_padding: Optional[IntTupleIntType]
    pool1_kernel_size: IntTupleIntType
    pool1_stride: Optional[IntTupleIntType]
    pool1_padding: Optional[IntTupleIntType]
    pool2_kernel_size: IntTupleIntType
    pool2_stride: Optional[IntTupleIntType]
    pool2_padding: Optional[IntTupleIntType]
    activation: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]
    linear_input_dim: Optional[int]
    linear_output_dim: Optional[int]

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

        self.include_linear_layer = True if model_cfg.linear_output_dim is not None else False

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
        if self.include_linear_layer:
            self.linear = nn.Linear(
                in_features=model_cfg.linear_input_dim,
                out_features=model_cfg.linear_output_dim,
                bias=True,
            )

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
        tensor = self.pool1(tensor)  # [B,C,T',F']
        sequence_mask = sequence_mask.float()
        sequence_mask = _mask_pool(
            sequence_mask,
            _get_int_tuple_int(self.pool1.kernel_size, 0),
            _get_int_tuple_int(self.pool1.stride, 0),
            _get_int_tuple_int(self.pool1.padding, 0),
        )

        tensor = self.conv3(tensor)
        tensor = self.conv4(tensor)

        tensor = self.activation(tensor)
        tensor = self.pool2(tensor)  # [B,C,T",F"]
        sequence_mask = _mask_pool(
            sequence_mask,
            _get_int_tuple_int(self.pool2.kernel_size, 0),
            _get_int_tuple_int(self.pool2.stride, 0),
            _get_int_tuple_int(self.pool2.padding, 0),
        )
        sequence_mask = sequence_mask.bool()

        tensor = torch.transpose(tensor, 1, 2)  # transpose to [B,T",C,F"]
        tensor = torch.flatten(tensor, start_dim=2, end_dim=-1)  # [B,T",C*F"]

        if self.include_linear_layer:
            tensor = self.linear(tensor)

        return tensor, sequence_mask
