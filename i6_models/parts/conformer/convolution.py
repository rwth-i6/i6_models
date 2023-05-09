from __future__ import annotations
import torch
from torch import nn


class ConformerConvolutionV1(nn.Module):
    """
    Conformer convolution module.
    see also: https://github.com/espnet/espnet/blob/713e784c0815ebba2053131307db5f00af5159ea/espnet/nets/pytorch_backend/conformer/convolution.py#L13
    """

    def __init__(self, channels: int, kernel_size: int, dropout: float = 0.1, activation: nn.Module = nn.SiLU()):
        """
        :param channels: number of channels for conv layers
        :param kernel_size: kernel size of conv layers
        :param dropout: dropout probability
        :param activation: activation function applied after batch norm
        """
        super().__init__()

        self.pointwise_conv1 = nn.Conv1d(
            in_channels=channels,
            out_channels=2 * channels,
            kernel_size=1,
        )
        self.depthwise_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding="same",
            groups=channels,
        )
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
        )
        self.layer_norm = nn.LayerNorm(channels)
        self.batch_norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        :param tensor: input tensor of shape [B,T,F]
        :return: torch.Tensor of shape [B,T,F]
        """
        tensor = self.layer_norm(tensor)

        # conv layers expect shape [B,F,T] so we have to transpose here
        tensor = tensor.transpose(1, 2)  # [B,F,T]

        tensor = self.pointwise_conv1(tensor)  # [B,2F,T]
        tensor = nn.functional.glu(tensor, dim=1)  # [B,F,T]

        tensor = self.depthwise_conv(tensor)
        tensor = self.batch_norm(tensor)
        tensor = self.activation(tensor)

        tensor = self.pointwise_conv2(tensor)

        return self.dropout(tensor.transpose(1, 2))
