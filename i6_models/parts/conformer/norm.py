import torch
import torch.nn as nn


class LayerNorm(nn.LayerNorm):
    """
    LayerNorm that accepts [N,C,*] tensors and normalizes over C (channels) dimension.
    see here: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    """

    def __init__(self, features: int):
        """
        :param channels: number of channels for normalization
        """
        super().__init__(features)
        self.layer_norm = nn.LayerNorm(features)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        :param tensor: input tensor with shape [N,C,*]
        :return: normalized tensor with shape [N,C,*]
        """
        tensor = tensor.transpose(1, -1)  # swap C to last dim
        tensor = self.layer_norm(tensor)
        tensor = tensor.transpose(-1, 1)  # transpose back
        return tensor
