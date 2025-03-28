from __future__ import annotations

__all__ = [
    "ConformerConvolutionV1",
    "ConformerConvolutionV1Config",
    "ConformerConvolutionV2",
    "ConformerConvolutionV2Config",
    "ConformerConvolutionV3",
]

from dataclasses import dataclass
from copy import deepcopy
from inspect import signature
from typing import Callable, Union, Optional, Literal

import torch
from torch import nn
from i6_models.config import ModelConfiguration
from i6_models.parts.dropout import BroadcastDropout


@dataclass
class ConformerConvolutionV1Config(ModelConfiguration):
    """
    Attributes:
        channels: number of channels for conv layers
        kernel_size: kernel size of conv layers
        dropout: dropout probability
        activation: activation function applied after normalization
        norm: normalization layer with input of shape [N,C,T]
    """

    channels: int
    kernel_size: int
    dropout: float
    activation: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]
    norm: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]

    def check_valid(self):
        assert self.kernel_size % 2 == 1, "ConformerConvolutionV1 only supports odd kernel sizes"

    def __post_init__(self):
        super().__post_init__()
        self.check_valid()


class ConformerConvolutionV1(nn.Module):
    """
    Conformer convolution module.
    see also: https://github.com/espnet/espnet/blob/713e784c0815ebba2053131307db5f00af5159ea/espnet/nets/pytorch_backend/conformer/convolution.py#L13

    Uses explicit padding for ONNX exportability, see:
    https://github.com/pytorch/pytorch/issues/68880
    """

    def __init__(self, model_cfg: ConformerConvolutionV1Config):
        """
        :param model_cfg: model configuration for this module
        """
        super().__init__()
        model_cfg.check_valid()
        self.pointwise_conv1 = nn.Linear(in_features=model_cfg.channels, out_features=2 * model_cfg.channels)
        self.depthwise_conv = nn.Conv1d(
            in_channels=model_cfg.channels,
            out_channels=model_cfg.channels,
            kernel_size=model_cfg.kernel_size,
            padding=(model_cfg.kernel_size - 1) // 2,
            groups=model_cfg.channels,
        )
        self.pointwise_conv2 = nn.Linear(in_features=model_cfg.channels, out_features=model_cfg.channels)
        self.layer_norm = nn.LayerNorm(model_cfg.channels)
        self.norm = deepcopy(model_cfg.norm)
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


@dataclass
class ConformerConvolutionV2Config(ConformerConvolutionV1Config):
    """
    New attribute:
        dropout_broadcast_axes: string of axes to which dropout is broadcast, e.g. "T" for broadcasting to the time axis
                                setting to None to disable broadcasting
    """

    dropout_broadcast_axes: Optional[Literal["B", "T", "BT"]]

    def check_valid(self):
        assert self.kernel_size % 2 == 1, "ConformerConvolutionV1 only supports odd kernel sizes"

        assert self.dropout_broadcast_axes in [
            None,
            "B",
            "T",
            "BT",
        ], "invalid value, supported are None, 'B', 'T' and 'BT'"


class ConformerConvolutionV2(ConformerConvolutionV1):
    """
    Augments ConformerMHSAV1 with dropout broadcasting
    """

    def __init__(self, model_cfg: ConformerConvolutionV2Config):
        """
        :param model_cfg: model configuration for this module
        """
        super().__init__(model_cfg)

        self.dropout = BroadcastDropout(model_cfg.dropout, dropout_broadcast_axes=model_cfg.dropout_broadcast_axes)

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

        tensor = self.dropout(tensor)

        return tensor


class ConformerConvolutionV3(ConformerConvolutionV2):
    """
    Augments ConformerConvolutionV3 with support for normalization schemes that
    require access to the sequence mask.
    """

    def __init__(self, model_cfg: ConformerConvolutionV2Config):
        """
        :param model_cfg: model configuration for this module
        """
        super().__init__(model_cfg)

        self.pass_mask_to_norm = len(signature(self.norm.forward).parameters) > 1

    def forward(self, tensor: torch.Tensor, tensor_mask: torch.Tensor) -> torch.Tensor:
        """
        :param tensor: input tensor of shape [B,T,F]
        :param tensor_mask: sequence mask for data inside input, shape [B,T]
        :return: torch.Tensor of shape [B,T,F]
        """
        tensor = self.layer_norm(tensor)
        tensor = self.pointwise_conv1(tensor)  # [B,T,2F]
        tensor = nn.functional.glu(tensor, dim=-1)  # [B,T,F]

        # conv layers expect shape [B,F,T] so we have to transpose here
        tensor = tensor.transpose(1, 2)  # [B,F,T]
        tensor = self.depthwise_conv(tensor)

        tensor = self.norm(tensor, tensor_mask) if self.pass_mask_to_norm else self.norm(tensor)
        tensor = tensor.transpose(1, 2)  # transpose back to [B,T,F]

        tensor = self.activation(tensor)
        tensor = self.pointwise_conv2(tensor)

        tensor = self.dropout(tensor)

        return tensor
