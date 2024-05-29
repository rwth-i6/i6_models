from __future__ import annotations

__all__ = ["MergerV1Config", "MergerV1"]

from dataclasses import dataclass

import torch
from torch import nn

from i6_models.config import ModelConfiguration


@dataclass
class MergerV1Config(ModelConfiguration):
    """
    Attributes:
        input_dim: input dimension
        kernel_size: kernel size of the depthwise convolution layer
        dropout: dropout probability
    """

    input_dim: int
    kernel_size: int
    dropout: float

    def check_valid(self):
        assert self.kernel_size % 2 == 1, "MergerV1 only supports odd kernel sizes"

    def __post__init__(self):
        super().__post_init__()
        self.check_valid()


class MergerV1(nn.Module):
    def __init__(self, model_cfg: MergerV1Config):
        """
        The merge module to merge the outputs of local extractor and global extractor
        Here we take the best variant from the E-branchformer paper (Fig. 3c), refer to
        https://arxiv.org/abs/2210.00077 for more merge module variants
        """
        super().__init__()

        self.depthwise_conv = nn.Conv1d(
            in_channels=model_cfg.input_dim * 2,
            out_channels=model_cfg.input_dim * 2,
            kernel_size=model_cfg.kernel_size,
            padding=(model_cfg.kernel_size - 1) // 2,
            groups=model_cfg.input_dim * 2,
        )
        self.linear_ff = nn.Linear(in_features=2 * model_cfg.input_dim, out_features=model_cfg.input_dim, bias=True)
        self.dropout = model_cfg.dropout

    def forward(self, x_1: torch.Tensor, x_2: torch.Tensor) -> torch.Tensor:
        x_concat = torch.cat([x_1, x_2], dim=-1)  # [B, T, 2F]
        # conv layers expect shape [B, F, T] so we have to transpose here
        x = x_concat.transpose(1, 2)  # [B, 2F, T]
        x = self.depthwise_conv(x)
        x = x.transpose(1, 2)  # [B, T, 2F]
        x = x + x_concat
        x = self.linear_ff(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return x
