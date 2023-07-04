import torch
import torch.nn as nn
from typing import Optional, Protocol, Tuple


class BaseFrontendInterfaceV1(Protocol):
    def forward(self, tensor: torch.Tensor, sequence_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        :param tensor: input tensor of shape [B,T,F]
        :param sequence_mask: masking tensor of shape [B,T], bool tensor, True == alive, False == dead
        :return: torch.Tensor of shape [B,T',F']
        """
        raise NotImplementedError


class FrontendInterfaceV1(BaseFrontendInterfaceV1, nn.Module):
    def forward(self, tensor: torch.Tensor, sequence_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        :param tensor: input tensor of shape [B,T,F]
        :param sequence_mask: masking tensor of shape [B,T], bool tensor, True == alive, False == dead
        :return: torch.Tensor of shape [B,T',F']
        """
        return tensor, sequence_mask
