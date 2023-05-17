import torch
import torch.nn as nn
from typing import Union


class LayerNorm(nn.Module):
    """
    Generic LayerNorm that accepts any input shape
    see here: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    """

    def __init__(self, normalized_shape: Union[int, list[int], torch.Size]):
        """
        :param normalized_shape: shape for normalization
        """
        super().__init__()

        self.normalized_shape = normalized_shape
        self.layer_norm = nn.LayerNorm(normalized_shape)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        :param tensor: input tensor of any shape
        """
        shape = tensor.shape
        tensor = tensor.reshape(-1, self.normalized_shape)
        tensor = self.layer_norm(tensor)
        tensor = tensor.reshape(shape)
        return tensor
