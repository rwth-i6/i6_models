from __future__ import annotations

__all__ = ["ConformerBlockV2Config", "ConformerEncoderV2Config", "ConformerBlockV2", "ConformerEncoderV2"]

import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple, List, Dict

from i6_models.config import ModelConfiguration, ModuleFactoryV1
from i6_models.parts.conformer import (
    ConformerConvolutionV1,
    ConformerConvolutionV1Config,
    ConformerMHSAV1,
    ConformerMHSAV1Config,
    ConformerPositionwiseFeedForwardV1,
    ConformerPositionwiseFeedForwardV1Config,
)


@dataclass
class ConformerBlockV2Config(ModelConfiguration):
    """
    Attributes:
        ff_cfg: Configuration for ConformerPositionwiseFeedForwardV1
        mhsa_cfg: Configuration for ConformerMHSAV1
        conv_cfg: Configuration for ConformerConvolutionV1
        swap_mhsa_conv: swap the execution order of MHSA and Conv module
    """

    # nested configurations
    ff_cfg: ConformerPositionwiseFeedForwardV1Config
    mhsa_cfg: ConformerMHSAV1Config
    conv_cfg: ConformerConvolutionV1Config
    swap_mhsa_conv: bool = True


class ConformerBlockV2(nn.Module):
    """
    Conformer block module
    """

    def __init__(self, cfg: ConformerBlockV2Config):
        """
        :param cfg: conformer block configuration with subunits for the different conformer parts
        """
        super().__init__()
        self.ff1 = ConformerPositionwiseFeedForwardV1(cfg=cfg.ff_cfg)
        self.mhsa = ConformerMHSAV1(cfg=cfg.mhsa_cfg)
        self.conv = ConformerConvolutionV1(model_cfg=cfg.conv_cfg)
        self.ff2 = ConformerPositionwiseFeedForwardV1(cfg=cfg.ff_cfg)
        self.final_layer_norm = torch.nn.LayerNorm(cfg.ff_cfg.input_dim)
        self.swap_mhsa_conv = cfg.swap_mhsa_conv

    def forward(self, x: torch.Tensor, /, sequence_mask: torch.Tensor) -> torch.Tensor:
        """
        :param x: input tensor of shape [B, T, F]
        :param sequence_mask: mask tensor where 1 defines positions within the sequence and 0 outside, shape: [B, T]
        :return: torch.Tensor of shape [B, T, F]
        """
        x = 0.5 * self.ff1(x) + x  #  [B, T, F]
        if not self.swap_mhsa_conv:
            x = self.mhsa(x, sequence_mask) + x  #  [B, T, F]
            x = self.conv(x) + x  #  [B, T, F]
        else:
            x = self.conv(x) + x  # [B, T, F]
            x = self.mhsa(x, sequence_mask) + x  # [B, T, F]
        x = 0.5 * self.ff2(x) + x  #  [B, T, F]
        x = self.final_layer_norm(x)  #  [B, T, F]
        return x


@dataclass
class ConformerEncoderV2Config(ModelConfiguration):
    """
    Attributes:
        num_layers: Number of conformer layers in the conformer encoder
        loss_scales: Dictionary defining the loss layers and corresponding loss scales e.g. {6: 0.3, 12:1}
        frontend: A pair of ConformerFrontend and corresponding config
        block_cfg: Configuration for ConformerBlockV2
    """

    num_layers: int
    loss_scales: Dict[int, float]

    # nested configurations
    frontend: ModuleFactoryV1
    block_cfg: ConformerBlockV2Config


class ConformerEncoderV2(nn.Module):
    """
    Modification compared to ConformerEncoderV1:
    1. enable adding auxiliary losses with different loss scales
        (C.f. https://arxiv.org/abs/2102.03216 for more details of auxiliary loss)
    2. enable swapping the MHSA and Conv module
        (C.f. https://arxiv.org/abs/2011.10798 for intuition behind swapping)
    """

    def __init__(self, cfg: ConformerEncoderV2Config):
        """
        :param cfg: conformer encoder configuration with subunits for frontend and conformer blocks
        """
        super().__init__()

        self.frontend = cfg.frontend()
        self.module_list = torch.nn.ModuleList([ConformerBlockV2(cfg.block_cfg) for _ in range(cfg.num_layers)])
        self.output_layers = list(cfg.loss_scales.keys())
        assert cfg.num_layers in self.output_layers, "The final layer must be included in loss layers"

    def forward(
        self, data_tensor: torch.Tensor, sequence_mask: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        :param data_tensor: input tensor of shape [B, T', F]
        :param sequence_mask: mask tensor where 1 defines positions within the sequence and 0 outside, shape: [B, T']
        :return: (output, out_seq_mask)
            where output is torch.Tensor of shape [B, T, F'],
            out_seq_mask is a torch.Tensor of shape [B, T]

        F: input feature dim, F': internal and output feature dim
        T': data time dim, T: down-sampled time dim (internal time dim)
        """
        x, sequence_mask = self.frontend(data_tensor, sequence_mask)  # [B, T, F']

        outputs = []
        for i in range(len(self.module_list)):
            x = self.module_list[i](x, sequence_mask)  # [B, T, F']
            if i + 1 in self.output_layers:
                outputs.append(x)

        return outputs, sequence_mask
