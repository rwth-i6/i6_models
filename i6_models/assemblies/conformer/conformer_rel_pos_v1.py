from __future__ import annotations

__all__ = [
    "ConformerRelPosBlockV1Config",
    "ConformerRelPosEncoderV1Config",
    "ConformerRelPosBlockV1",
    "ConformerRelPosEncoderV1",
]

import torch
from torch import nn
from dataclasses import dataclass, field
from typing import List, Optional

from i6_models.config import ModelConfiguration, ModuleFactoryV1
from i6_models.parts.conformer import (
    ConformerConvolutionV2Config,
    ConformerConvolutionV3,
    ConformerMHSARelPosV1,
    ConformerMHSARelPosV1Config,
    ConformerPositionwiseFeedForwardV2,
    ConformerPositionwiseFeedForwardV2Config,
)
from i6_models.assemblies.conformer import ConformerEncoderV2


@dataclass
class ConformerRelPosBlockV1Config(ModelConfiguration):
    """
    Attributes:
        ff_cfg: Configuration for ConformerPositionwiseFeedForwardV2
        mhsa_cfg: Configuration for ConformerMHSARelPosV1
        conv_cfg: Configuration for ConformerConvolutionV2
        modules: List of modules to use for ConformerRelPosBlockV1,
            "ff" for feed forward module, "mhsa" for multi-head self attention module, "conv" for conv module
        scales: List of scales to apply to the module outputs before the residual connection
    """

    # nested configurations
    ff_cfg: ConformerPositionwiseFeedForwardV2Config
    mhsa_cfg: ConformerMHSARelPosV1Config
    conv_cfg: ConformerConvolutionV2Config
    modules: List[str] = field(default_factory=lambda: ["ff", "mhsa", "conv", "ff"])
    scales: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.0, 0.5])

    def __post__init__(self):
        super().__post_init__()
        assert len(self.modules) == len(self.scales), "modules and scales must have same length"
        for module_name in self.modules:
            assert module_name in ["ff", "mhsa", "conv"], "module not supported"


class ConformerRelPosBlockV1(nn.Module):
    """
    Conformer block module, modifications compared to ConformerBlockV1:
    - uses ConfomerMHSARelPosV1 as MHSA module
    - enable constructing the block with self-defined module_list as ConformerBlockV2
    """

    def __init__(self, cfg: ConformerRelPosBlockV1Config):
        """
        :param cfg: conformer block configuration with subunits for the different conformer parts
        """
        super().__init__()

        modules = []
        for module_name in cfg.modules:
            if module_name == "ff":
                modules.append(ConformerPositionwiseFeedForwardV2(cfg=cfg.ff_cfg))
            elif module_name == "mhsa":
                modules.append(ConformerMHSARelPosV1(cfg=cfg.mhsa_cfg))
            elif module_name == "conv":
                modules.append(ConformerConvolutionV3(model_cfg=cfg.conv_cfg))
            else:
                raise NotImplementedError

        self.module_list = nn.ModuleList(modules)
        self.scales = cfg.scales
        self.final_layer_norm = torch.nn.LayerNorm(cfg.ff_cfg.input_dim)

    def forward(self, x: torch.Tensor, /, sequence_mask: torch.Tensor) -> torch.Tensor:
        """
        :param x: input tensor of shape [B, T, F]
        :param sequence_mask: mask tensor where 1 defines positions within the sequence and 0 outside, shape: [B, T]
        :return: torch.Tensor of shape [B, T, F]
        """
        for scale, module in zip(self.scales, self.module_list):
            if isinstance(module, (ConformerMHSARelPosV1, ConformerConvolutionV3)):
                x = scale * module(x, sequence_mask) + x
            else:
                x = scale * module(x) + x
        x = self.final_layer_norm(x)  #  [B, T, F]
        return x


@dataclass
class ConformerRelPosEncoderV1Config(ModelConfiguration):
    """
    Attributes:
        num_layers: Number of conformer layers in the conformer encoder
        frontend: A pair of ConformerFrontend and corresponding config
        block_cfg: Configuration for ConformerRelPosBlockV1
    """

    num_layers: int

    # nested configurations
    frontend: Optional[ModuleFactoryV1]
    block_cfg: ConformerRelPosBlockV1Config


class ConformerRelPosEncoderV1(ConformerEncoderV2):
    """
    Modifications compared to ConformerEncoderV2:
    - supports Shaw's relative positional encoding using learnable position embeddings
      and Transformer-XL style relative PE using fixed sinusoidal or learnable position embeddings
    """

    def __init__(self, cfg: ConformerRelPosEncoderV1Config):
        """
        :param cfg: conformer encoder configuration with subunits for frontend and conformer blocks
        """
        super().__init__(cfg)

        if cfg.frontend is not None:
            self.frontend = cfg.frontend()
        else:
            self.frontend = nn.Identity()
        self.module_list = torch.nn.ModuleList([ConformerRelPosBlockV1(cfg.block_cfg) for _ in range(cfg.num_layers)])
