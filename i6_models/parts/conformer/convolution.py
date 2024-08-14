from __future__ import annotations

__all__ = ["ConformerConvolutionV1", "ConformerConvolutionV1Config", "ConformerConvolutionV2", "ConformerConvolutionV2Config"]

from dataclasses import dataclass
from copy import deepcopy

import torch
from torch import nn
from i6_models.config import ModelConfiguration
from typing import Callable, Union, Optional


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
    Allows even kernel size
    """

    dropout_broadcast_axes: Optional[str] = None

    def check_valid(self):
        assert self.kernel_size % 2 == 1, "ConformerConvolutionV1 only supports odd kernel sizes"

        assert self.dropout_broadcast_axes is None or self.dropout_broadcast_axes in [
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

        self.dropout_broadcast_axes = model_cfg.dropout_broadcast_axes
        self.dropout = (
            nn.Dropout1d(model_cfg.dropout) if model_cfg.dropout_broadcast_axes else nn.Dropout(model_cfg.dropout)
        )

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

        if self.dropout_broadcast_axes is None:
            tensor = self.dropout(tensor)
        elif self.dropout_broadcast_axes == "T":
            tensor = self.dropout(tensor.transpose(1, 2)).transpose(1, 2)
        elif self.dropout_broadcast_axes == "B":
            tensor = self.dropout(tensor.permute(1, 2, 0)).permute(2, 0, 1)
        elif self.dropout_broadcast_axes == "BT":
            batch_dim_size = tensor.shape[0]
            feature_dim_size = tensor.shape[-1]

            tensor = (
                self.dropout(tensor.reshape(-1, feature_dim_size).transpose(0, 1))
                .transpose(0, 1)
                .reshape(batch_dim_size, -1, feature_dim_size)
            )

        return tensor
