from __future__ import annotations

__all__ = [
    "VGG4LayerPoolFrontendV1",
    "VGG4LayerPoolFrontendV1Config",
]

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn

from i6_models.config import ModelConfiguration

from .common import IntTupleIntType, _get_padding, _mask_pool, _get_int_tuple_int


@dataclass
class VGG4LayerPoolFrontendV1Config(ModelConfiguration):
    """
    Attributes:
        conv1_channels: number of channels for first conv layer
        pool_kernel_size: kernel size of pooling layer
        pool_padding: padding for pooling layer
        conv2_channels: number of channels for second conv layer
        conv2_stride: stride param for second conv layer
        conv3_channels: number of channels for third conv layer
        conv3_stride: stride param for third conv layer
        conv4_channels: number of channels for fourth layer
        conv_kernel_size: kernel size of conv layers
        conv_padding: padding for the convolution
        activation: activation function at the end
        linear_input_dim: input size of the final linear layer
        linear_output_dim: output size of the final linear layer
    """

    conv1_channels: int
    pool_kernel_size: IntTupleIntType
    pool_padding: Optional[IntTupleIntType]
    conv2_channels: int
    conv2_stride: IntTupleIntType
    conv3_channels: int
    conv3_stride: IntTupleIntType
    conv4_channels: int
    conv_kernel_size: IntTupleIntType
    conv_padding: Optional[IntTupleIntType]
    activation: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]
    linear_input_dim: Optional[int]
    linear_output_dim: Optional[int]

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

        self.include_linear_layer = True if model_cfg.linear_output_dim is not None else False

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=model_cfg.conv1_channels,
            kernel_size=model_cfg.conv_kernel_size,
            padding=conv_padding,
        )
        self.pool = nn.MaxPool2d(
            kernel_size=model_cfg.pool_kernel_size,
            padding=pool_padding,
        )
        self.conv2 = nn.Conv2d(
            in_channels=model_cfg.conv1_channels,
            out_channels=model_cfg.conv2_channels,
            kernel_size=model_cfg.conv_kernel_size,
            stride=model_cfg.conv2_stride,
            padding=conv_padding,
        )
        self.conv3 = nn.Conv2d(
            in_channels=model_cfg.conv2_channels,
            out_channels=model_cfg.conv3_channels,
            kernel_size=model_cfg.conv_kernel_size,
            stride=model_cfg.conv3_stride,
            padding=conv_padding,
        )
        self.conv4 = nn.Conv2d(
            in_channels=model_cfg.conv3_channels,
            out_channels=model_cfg.conv4_channels,
            kernel_size=model_cfg.conv_kernel_size,
            padding=conv_padding,
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
        T might be reduced to T'' depending on the stride of the layers

        :param tensor: input tensor of shape [B,T,F]
        :param sequence_mask: masking tensor of shape [B,T], contains length information of the sequences
        :return: torch.Tensor of shape [B,T',F']
        """
        tensor = tensor[:, None, :, :]  # [B,C=1,T,F]

        tensor = self.conv1(tensor)
        #tensor = self.activation(tensor)
        #tensor = self.pool(tensor)  # [B,C,T,F']

        #tensor = self.conv2(tensor)  # [B,C,T',F']
        sequence_mask = sequence_mask.float()
        print(sequence_mask)
        print(self.conv2.kernel_size[0], self.conv2.stride[0], self.conv2.padding[0])
        sequence_mask = _mask_pool(
            sequence_mask, self.conv2.kernel_size[0], self.conv2.stride[0], self.conv2.padding[0]
        )
        print(sequence_mask)
        #tensor = self.activation(tensor)

        #tensor = self.conv3(tensor)  # [B,C,T",F']
        #sequence_mask = _mask_pool(
        #    sequence_mask, self.conv3.kernel_size[0], self.conv3.stride[0], self.conv3.padding[0]
        #)
        #tensor = self.activation(tensor)

        #tensor = self.conv4(tensor)

        sequence_mask = sequence_mask.bool()

        tensor = torch.transpose(tensor, 1, 2)  # transpose to [B,T",C,F']
        tensor = torch.flatten(tensor, start_dim=2, end_dim=-1)  # [B,T",C*F']

        if self.include_linear_layer:
            tensor = self.linear(tensor)

        return tensor, sequence_mask
