__all__ = ["LayerNormNC"]
import torch
import torch.nn as nn


class LayerNormNC(nn.LayerNorm):
    """
    LayerNorm that accepts [N,C,*] tensors and normalizes over C (channels) dimension.
    see here: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    """

    def __init__(self, channels: int):
        """
        :param channels: number of channels for normalization
        """
        super().__init__(channels)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        :param tensor: input tensor with shape [N,C,*]
        :return: normalized tensor with shape [N,C,*]
        """
        return super().forward(tensor.transpose(1, -1)).transpose(1, -1)
