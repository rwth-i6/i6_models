__all__ = [
    "WindowConvolutionFrontendV1Config",
    "WindowConvolutionFrontendV1",
]

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from i6_models.config import ModelConfiguration

from .common import mask_pool, apply_same_padding


@dataclass
class WindowConvolutionFrontendV1Config(ModelConfiguration):
    """
    Attributes:
        input_dim: number of input features to module
        output_dim: output dimension
        dropout: dropout after linear layer
        kernel_size: number of feature frames to convolve
        stride: skip (stride - 1) feature frames; stride > 1 implies subsampling
        activation: activation function applied after linear computation
    """

    input_dim: int
    output_dim: int
    dropout: float
    kernel_size: int
    stride: int
    activation: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]

    def __post_init__(self):
        super().__post_init__()
        assert self.stride >= 1, "Choose an integer >= 1 for stride"
        assert 0.0 <= self.dropout <= 1.0, "Dropout value must be a probability"


class WindowConvolutionFrontendV1(nn.Module):
    """
    Simple feed-forward front-end that computes over a window
    of input features. Choosing a stride > 1 allows for subsampling
    of the features.
    """

    def __init__(self, cfg: WindowConvolutionFrontendV1Config):
        """
        :param cfg: model configuration for this module
        """
        super().__init__()
        self.conv = torch.nn.Conv1d(
            in_channels=cfg.input_dim,
            out_channels=cfg.output_dim,
            kernel_size=cfg.kernel_size,
            stride=cfg.stride,
            padding=0,
            bias=True,
        )
        self.activation = cfg.activation
        self.pad = lambda x: apply_same_padding(x, cfg.kernel_size)
        self.dropout = torch.nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, /, sequence_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        T might be reduced to T' on stride

        :param x: input tensor of shape [B,T,F]
        :param sequence_mask: the sequence mask for the tensor
        :return: torch.Tensor of shape [B,T',F'] and the shape of the sequence mask
        """
        # torch 1d convolution is over last dim but we want time conv
        x = x.transpose(1, 2)  # [B, F, T]
        x = self.pad(x)
        x = self.conv(x).transpose(1, 2)  # [B, T', F']

        # change masking according to stride value
        sequence_mask = mask_pool(
            sequence_mask,
            kernel_size=1,
            stride=self.conv.stride[0],
            padding=0,  # done manually
        )
        x = self.activation(x)
        x = self.dropout(x)

        return x, sequence_mask
