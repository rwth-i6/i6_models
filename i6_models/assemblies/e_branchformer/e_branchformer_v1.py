from __future__ import annotations

__all__ = [
    "EbranchformerBlockV1Config",
    "EbranchformerBlockV1",
    "EbranchformerEncoderV1Config",
    "EbranchformerEncoderV1",
]

import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple

from i6_models.config import ModelConfiguration, ModuleFactoryV1
from i6_models.parts.conformer import (
    ConformerMHSAV1 as MHSAV1,
    ConformerMHSAV1Config as MHSAV1Config,
    ConformerPositionwiseFeedForwardV1 as PositionwiseFeedForwardV1,
    ConformerPositionwiseFeedForwardV1Config as PositionwiseFeedForwardV1Config,
)
from i6_models.parts.e_branchformer import (
    ConvolutionalGatingMLPV1Config,
    ConvolutionalGatingMLPV1,
    MergerV1Config,
    MergerV1,
)


@dataclass
class EbranchformerBlockV1Config(ModelConfiguration):
    """
    Attributes:
        ff_cfg: Configuration for PositionwiseFeedForwardV1 module
        mhsa_cfg: Configuration for MHSAV1 module
        cgmlp_cfg: Configuration for ConvolutionalGatingMLPV1 module
        merger_cfg: Configuration for MergerV1 module
    """

    ff_cfg: PositionwiseFeedForwardV1Config
    mhsa_cfg: MHSAV1Config
    cgmlp_cfg: ConvolutionalGatingMLPV1Config
    merger_cfg: MergerV1Config


class EbranchformerBlockV1(nn.Module):
    """
    Ebranchformer block module
    """

    def __init__(self, cfg: EbranchformerBlockV1Config):
        """
        :param cfg: e-branchformer block configuration with subunits for the different e-branchformer parts
        """
        super().__init__()
        self.ff_1 = PositionwiseFeedForwardV1(cfg=cfg.ff_cfg)
        self.mhsa = MHSAV1(cfg=cfg.mhsa_cfg)
        self.cgmlp = ConvolutionalGatingMLPV1(model_cfg=cfg.cgmlp_cfg)
        self.merger = MergerV1(model_cfg=cfg.merger_cfg)
        self.ff_2 = PositionwiseFeedForwardV1(cfg=cfg.ff_cfg)
        self.final_layer_norm = torch.nn.LayerNorm(cfg.ff_cfg.input_dim)

    def forward(self, x: torch.Tensor, /, sequence_mask: torch.Tensor) -> torch.Tensor:
        """
        :param tensor: input tensor of shape [B, T, F]
        :param sequence_mask: mask tensor where 0 defines positions within the sequence and 1 outside, shape: [B, T]
        :return: torch.Tensor of shape [B, T, F]
        """
        x = 0.5 * self.ff1(x) + x  #  [B, T, F]
        x_1 = self.mhsa(x, sequence_mask)  #  [B, T, F]
        x_2 = self.cgmlp(x)  #  [B, T, F]
        x = self.merger(x_1, x_2) + x  #  [B, T, F]
        x = 0.5 * self.ff2(x) + x  #  [B, T, F]
        x = self.final_layer_norm(x)  # [B, T, F]
        return x


class EbranchformerEncoderV1Config(ModelConfiguration):
    """
    Attributes:
        num_layers: Number of e-branchformer layers in the e-branchformer encoder
        frontend: A pair of ConformerFrontend and corresponding config
        block_cfg: Configuration for EbranchformerBlockV1
    """

    num_layers: int

    # nested configurations
    frontend: ModuleFactoryV1
    block_cfg: EbranchformerBlockV1Config


class EbranchformerEncoderV1(nn.Module):
    """
    Implementation of the Branchformer with Enhanced merging (short e-branchformer), as in the original publication.
    The model consists of a frontend and a stack of N e-branchformer blocks.
    C.f. https://arxiv.org/pdf/2210.00077.pdf
    """

    def __init__(self, cfg: EbranchformerEncoderV1Config):
        """
        :param cfg: e-branchformer encoder configuration with subunits for frontend and e-branchformer blocks
        """
        super().__init__()

        self.frontend = cfg.frontend()
        self.module_list = torch.nn.ModuleList([EbranchformerBlockV1(cfg.block_cfg) for _ in range(cfg.num_layers)])

    def forward(self, data_tensor: torch.Tensor, sequence_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param data_tensor: input tensor of shape [B, T', F']
        :param sequence_mask: mask tensor where 1 defines positions within the sequence and 0 outside, shape: [B, T']
        :return: (output, out_seq_mask)
            where output is torch.Tensor of shape [B, T, F],
            out_seq_mask is a torch.Tensor of shape [B, T]

        F': input feature dim, F: internal and output feature dim
        T': data time dim, T: down-sampled time dim (internal time dim)
        """
        x, sequence_mask = self.frontend(data_tensor, sequence_mask)  # [B, T, F]
        for module in self.module_list:
            x = module(x, sequence_mask)  # [B, T, F]

        return x, sequence_mask
