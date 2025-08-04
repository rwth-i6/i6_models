from __future__ import annotations

__all__ = [
    "EbranchformerBlockV1Config",
    "EbranchformerBlockV1",
    "EbranchformerEncoderV1Config",
    "EbranchformerEncoderV1",
]

import numpy as np
import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple, List

from i6_models.config import ModelConfiguration, ModuleFactoryV1
from i6_models.parts.dynamic_adaptable_conformer import (
    ConformerMHSAWithGateV1,
    ConformerMHSAV1Config,
    ConformerPositionwiseFeedForwardV1,
    ConformerPositionwiseFeedForwardV1Config,
)
from i6_models.parts.dynamic_adaptable_e_branchformer import (
    ConvolutionalGatingMLPV1Config,
    ConvolutionalGatingMLPV1,
    MergerV1Config,
    MergerV1,
)

EPSILON = np.finfo(np.float32).tiny


@dataclass
class EbranchformerBlockV1Config(ModelConfiguration):
    """
    Attributes:
        ff1_cfg: Configuration for the 1st ConformerPositionwiseFeedForwardV1
        ff2_cfg: Configuration for the 2nd ConformerPositionwiseFeedForwardV1
        mhsa_cfg: Configuration for MHSAV1 module
        cgmlp_cfg: Configuration for ConvolutionalGatingMLPV1 module
        merger_cfg: Configuration for MergerV1 module
        adjust_dropout: whether adjust the dropout based on hidden dimension
    """

    ff1_cfg: ConformerPositionwiseFeedForwardV1Config
    ff2_cfg: ConformerPositionwiseFeedForwardV1Config
    mhsa_cfg: ConformerMHSAV1Config
    cgmlp_cfg: ConvolutionalGatingMLPV1Config
    merger_cfg: MergerV1Config
    adjust_dropout: bool = False


class EbranchformerBlockV1(nn.Module):
    """
    Ebranchformer block module
    """

    def __init__(self, cfg: EbranchformerBlockV1Config):
        """
        :param cfg: e-branchformer block configuration with subunits for the different e-branchformer parts
        """
        super().__init__()
        self.ff_1 = ConformerPositionwiseFeedForwardV1(cfg=cfg.ff1_cfg)
        self.mhsa = ConformerMHSAWithGateV1(cfg=cfg.mhsa_cfg)
        self.cgmlp = ConvolutionalGatingMLPV1(model_cfg=cfg.cgmlp_cfg)
        self.merger = MergerV1(model_cfg=cfg.merger_cfg)
        self.ff_2 = ConformerPositionwiseFeedForwardV1(cfg=cfg.ff2_cfg)
        self.adjust_dropout = cfg.adjust_dropout
        self.final_layer_norm = torch.nn.LayerNorm(cfg.ff1_cfg.input_dim)

    def forward(self, x: torch.Tensor, /, sequence_mask: torch.Tensor) -> torch.Tensor:
        """
        :param tensor: input tensor of shape [B, T, F]
        :param sequence_mask: mask tensor where 0 defines positions within the sequence and 1 outside, shape: [B, T]
        :return: torch.Tensor of shape [B, T, F]
        """
        x = 0.5 * self.ff_1(x, adjust_dropout=self.adjust_dropout) + x  #  [B, T, F]
        x_1 = self.mhsa(
            x, sequence_mask, adjust_dropout=self.adjust_dropout
        )  #  [B, T, F]
        x_2 = self.cgmlp(x, adjust_dropout=self.adjust_dropout)  #  [B, T, F]
        x = self.merger(x_1, x_2) + x  #  [B, T, F]
        x = 0.5 * self.ff_2(x, adjust_dropout=self.adjust_dropout) + x  #  [B, T, F]
        x = self.final_layer_norm(x)  # [B, T, F]
        return x


@dataclass
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
    block_cfgs: List[EbranchformerBlockV1Config]


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
        block_list = []
        for block_cfg in cfg.block_cfgs:
            block_list.append(EbranchformerBlockV1(block_cfg))
        self.module_list = torch.nn.ModuleList(block_list)

    def forward(
        self,
        data_tensor: torch.Tensor,
        /,
        sequence_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
