from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from i6_models.config import ModelConfiguration


@dataclass
class MergerV1Config(ModelConfiguration):
    """
    The merge module to merge the outputs of local extractor and global extractor
    """

    input_dim: int
    """input dimension"""
    kernel_size: int
    """kernel size of the depth-wise convolution layer"""
    dropout: float
    """dropout probability"""


class MergerV1(nn.Module):
    def __init__(self, cfg: MergerV1Config):
        """
        Here we take the best variant from the E-branchformer paper, refer to
        https://arxiv.org/abs/2210.00077 for more merge module variants
        """
        super().__init__()

        self.depthwise_conv = nn.Conv1d(
            in_channels=cfg.input_dim * 2,
            out_channels=cfg.input_dim * 2,
            kernel_size=cfg.kernel_size,
            padding=(cfg.kernel_size - 1) // 2,
            groups=cfg.input_dim * 2 // 2,
        )
        self.linear_ff = nn.Linear(in_features=2 * cfg.input_dim, out_features=cfg.input_dim, bias=True)

    def forward(self, tensor_local: torch.Tensor, tensor_global: torch.Tensor) -> torch.Tensor:
        tensor_concat = torch.cat([tensor_local, tensor_global], dim=-1)  # (B,T,2F)
        # conv layers expect shape [B,F,T] so we have to transpose here
        tensor = tensor_concat.transpose(1, 2)  # (B,2F,T)
        tensor = self.depthwise_conv(tensor)
        tensor = tensor.transpose(1, 2)  # (B,T,2F)
        tensor = tensor + tensor_concat
        tensor = self.linear_ff(tensor)
        tensor = nn.functional.dropout(tensor, p=self.dropout, training=self.training)

        return tensor
