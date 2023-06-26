import torch
import torch.nn as nn
from abc import abstractmethod
from typing import Protocol, Tuple


class BaseFrontendInterfaceV1(Protocol):
    def forward(self, tensor: torch.Tensor, sequence_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param tensor: input tensor of shape [B,T,F]
        :param sequence_mask: masking tensor of shape [B,T], contains length information of the sequences
        :return: torch.Tensor of shape [B,T',F']
        """
        raise NotImplementedError


class FrontendInterfaceV1(BaseFrontendInterfaceV1, nn.Module):
    def forward(self, tensor: torch.Tensor, sequence_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param tensor: input tensor of shape [B,T,F]
        :param sequence_mask: masking tensor of shape [B,T], contains length information of the sequences
        :return: torch.Tensor of shape [B,T',F']
        """
        return tensor, sequence_mask
