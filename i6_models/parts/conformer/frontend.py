from __future__ import annotations

__all__ = ["ConformerFrontendV1", "ConformerFrontendV1Config"]
from typing import Optional
from dataclasses import dataclass
import torch
from torch import nn

from i6_models.config import ModelConfiguration


@dataclass
class ConformerFrontendV1Config(ModelConfiguration):
    """
    :param feature_dim: Feature dimension of the input data
    :param hidden_dim: Hidden dimension used in the model internally
    :param dropout: Dropout value after linear transformation
    :param conv_stride: Stride of down-sampling cov
    :param conv_kernel: Kernel size of down-sampling conv
    :param conv_padding: Padding factor of down-sampling conv
    :param spec_aug_cfg: TODO
    """

    feature_dim: int
    hidden_dim: int
    dropout: float
    conv_stride: int
    conv_kernel: int
    conv_padding: int

    # nested configurations
    spec_aug_cfg: Optional[ModelConfiguration]  # TODO


class ConformerFrontendV1(nn.Module):
    """
    Frontend part of the standard conformer doing down-sampling
    """

    def __init__(self, cfg: ConformerFrontendV1Config):
        """
        :param cfg: conformer frontend configuration
        """
        super().__init__()
        if cfg.spec_aug_cfg:
            # TODO: This still needs to be implemented
            raise NotImplementedError
        else:
            self.spec_aug = None
        self.subsampling = nn.Conv1d(
            in_channels=cfg.feature_dim,
            out_channels=cfg.feature_dim,
            kernel_size=cfg.conv_kernel,
            stride=cfg.conv_stride,
            padding=cfg.conv_padding,
        )
        self.linear = nn.Linear(in_features=cfg.feature_dim, out_features=cfg.hidden_dim)
        self.dropout = nn.Dropout(p=cfg.dropout)

    def forward(self, data_tensor: torch.Tensor):
        """
        :param data_tensor: input tensor of shape [B, T, F]
        :return: torch.Tensor of shape [B, T', F']

        F: input feature dim, F': internal and output feature dim
        T: data time dim, T': down-sampled time dim
        """
        x = data_tensor
        if self.spec_aug:
            x = self.spec_aug(x)
        x = x.transpose(1, 2)  # [B, F, T]
        x = self.subsampling(x)  # [B, F, T']
        x = x.transpose(2, 1)  # [B, T', F]
        x = self.linear(x)  # [B, T', F']
        x = self.dropout(x)  # [B, T', F']

        return x
