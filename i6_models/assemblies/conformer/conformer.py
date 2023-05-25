from __future__ import annotations

__all__ = ["ConformerBlockV1Config", "ConformerEncoderV1Config", "ConformerBlockV1", "ConformerEncoderV1"]

import torch
from torch import nn
from dataclasses import dataclass

from i6_models.config import ModelConfiguration
from i6_models.parts.conformer import (
    ConformerConvolutionV1,
    ConformerConvolutionV1Config,
    ConformerFrontendV1,
    ConformerFrontendV1Config,
    ConformerMHSAV1,
    ConformerMHSAV1Config,
    ConformerPositionwiseFeedForwardV1,
    ConformerPositionwiseFeedForwardV1Config,
)


@dataclass
class ConformerBlockV1Config(ModelConfiguration):
    """
    :param ff_cfg: Configuration for ConformerPositionwiseFeedForwardV1
    :param mhsa_cfg: Configuration for ConformerMHSAV1
    :param conv_cfg: Configuration for ConformerConvolutionV1
    """

    # nested configurations
    ff_cfg: ConformerPositionwiseFeedForwardV1Config
    mhsa_cfg: ConformerMHSAV1Config
    conv_cfg: ConformerConvolutionV1Config


class ConformerBlockV1(nn.Module):
    """
    Conformer block module
    """

    def __init__(self, cfg: ConformerBlockV1Config):
        """
        :param cfg: conformer block configuration with subunits for the different conformer parts
        """
        super().__init__()
        self.ff_1 = ConformerPositionwiseFeedForwardV1(cfg=cfg.ff_cfg)
        self.mhsa = ConformerMHSAV1(cfg=cfg.mhsa_cfg)
        self.conv = ConformerConvolutionV1(model_cfg=cfg.conv_cfg)
        self.ff_2 = ConformerPositionwiseFeedForwardV1(cfg=cfg.ff_cfg)
        self.final_layer_norm = torch.nn.LayerNorm(cfg.ff_cfg.input_dim)

    def forward(self, tensor: torch.Tensor, sequence_mask: torch.Tensor):
        """
        :param tensor: input tensor of shape [B, T, F]
        :param sequence_mask: mask tensor where 0 defines positions within the sequence and 1 outside, shape: [B, T]
        :return: torch.Tensor of shape [B, T, F]
        """
        residual = tensor  #  [B, T, F]
        x = self.ff_1(residual)  #  [B, T, F]
        residual = 0.5 * x + residual  #  [B, T, F]
        x = self.mhsa(residual, sequence_mask)  #  [B, T, F]
        residual = x + residual  # [B, T, F]
        x = self.conv(residual)  #  [B, T, F]
        residual = x + residual  # [B, T, F]
        x = self.ff_2(residual)  #  [B, T, F]
        x = 0.5 * x + residual  #  [B, T, F]
        x = self.final_layer_norm(x)  #  [B, T, F]
        return x


@dataclass
class ConformerEncoderV1Config(ModelConfiguration):
    """
    :param num_layers: Number of conformer layers in the conformer encoder
    :param front_cfg: Configuration for ConformerFrontendV1
    :param block_cfg: Configuration for ConformerBlockV1
    """

    num_layers: int

    # nested configurations
    front_cfg: ConformerFrontendV1Config
    block_cfg: ConformerBlockV1Config


class ConformerEncoderV1(nn.Module):
    """
    Implementation of the convolution-augmented Transformer (short Conformer), as in the original publication.
    The model consists of a frontend and a stack of N conformer blocks.
    C.f. https://arxiv.org/pdf/2005.08100.pdf
    """

    def __init__(self, cfg: ConformerEncoderV1Config):
        """
        :param cfg: conformer encoder configuration with subunits for frontend and conformer blocks
        """
        super().__init__()

        self.frontend = ConformerFrontendV1(cfg=cfg.front_cfg)
        self.module_list = torch.nn.ModuleList([ConformerBlockV1(cfg.block_cfg) for _ in range(cfg.num_layers)])

    def forward(self, data_tensor: torch.Tensor, sequence_mask: torch.Tensor):
        """
        :param data_tensor: input tensor of shape [B, T', F]
        :param sequence_mask: mask tensor where 0 defines positions within the sequence and 1 outside, shape: [B, T']
        :return: torch.Tensor of shape [B, T, F']

        F: input feature dim, F': internal and output feature dim
        T': data time dim, T: down-sampled time dim (internal time dim)
        """
        x = self.frontend(data_tensor)  # [B, T, F']
        for module in self.module_list:
            x = module(x, sequence_mask)  # [B, T, F']

        return x
