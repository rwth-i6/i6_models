from __future__ import annotations

__all__ = ["ConformerBlockV2Config", "ConformerEncoderV2Config", "ConformerBlockV2", "ConformerEncoderV2"]

import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple, List, Optional

from i6_models.config import ModelConfiguration, ModuleFactoryV1
from i6_models.assemblies.conformer import (
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
        modules: List of modules to use for ConformerBlockV2,
                       "ff" for feed forward module, "mhsa" for multi-head self attention module, "conv" for conv module
        scales: List of scales to apply to the module outputs before the residual connection
    """

    # nested configurations
    ff_cfg: ConformerPositionwiseFeedForwardV1Config
    mhsa_cfg: ConformerMHSAV1Config
    conv_cfg: ConformerConvolutionV1Config
    modules: List[str] = ["ff", "mhsa", "conv", "ff"]
    scales: List[float] = [0.5, 1.0, 1.0, 0.5]

    def __post__init__(self):
        super().__post_init__()
        assert len(self.modules) == len(self.scales), "modules and scales must have same length"
        for module_name in self.modules:
            assert module_name in ["ff", "mhsa", "conv"], "module not supported"


class ConformerBlockV2(nn.Module):
    """
    Conformer block module
    """

    def __init__(self, cfg: ConformerBlockV2Config):
        """
        :param cfg: conformer block configuration with subunits for the different conformer parts
        """
        super().__init__()

        modules = []
        for module_name in cfg.modules:
            if module_name == "ff":
                modules.append(ConformerPositionwiseFeedForwardV1(cfg=cfg.ff_cfg))
            elif module_name == "mhsa":
                modules.append(ConformerMHSAV1(cfg=cfg.mhsa_cfg))
            elif module_name == "conv":
                modules.append(ConformerConvolutionV1(cfg=cfg.conv_cfg))
            else:
                raise NotImplementedError

        self.modules = nn.ModuleList(modules)
        self.scales = cfg.scales
        self.final_layer_norm = torch.nn.LayerNorm(cfg.ff_cfg.input_dim)

    def forward(self, x: torch.Tensor, /, sequence_mask: torch.Tensor) -> torch.Tensor:
        """
        :param x: input tensor of shape [B, T, F]
        :param sequence_mask: mask tensor where 0 defines positions within the sequence and 1 outside, shape: [B, T]
        :return: torch.Tensor of shape [B, T, F]
        """
        for scale, module in zip(self.scales, self.modules):
            if isinstance(module, ConformerMHSAV1):
                x = scale * module(x, sequence_mask) + x
            else:
                x = scale * module(x)

        x = self.final_layer_norm(x)  #  [B, T, F]
        return x


@dataclass
class ConformerEncoderV2Config(ModelConfiguration):
    """
    Attributes:
        num_layers: Number of conformer layers in the conformer encoder
        frontend: A pair of ConformerFrontend and corresponding config
        block_cfg: Configuration for ConformerBlockV2
    """

    num_layers: int

    # nested configurations
    frontend: ModuleFactoryV1
    block_cfg: ConformerBlockV2Config


class ConformerEncoderV2(nn.Module):
    """
    Implementation of the convolution-augmented Transformer (short Conformer), as in the original publication.
    The model consists of a frontend and a stack of N conformer blocks.
    C.f. https://arxiv.org/pdf/2005.08100.pdf
    Each conformer block is composed of fixed number of feed forward modules, multi-head self attention modules and conv modules.
    """

    def __init__(self, cfg: ConformerEncoderV2Config):
        """
        :param cfg: conformer encoder configuration with subunits for frontend and conformer blocks
        """
        super().__init__()

        self.frontend = cfg.frontend()
        self.module_list = torch.nn.ModuleList([ConformerBlockV2(cfg.block_cfg) for _ in range(cfg.num_layers)])
        self.output_layers = cfg.loss_layers

    def forward(
        self, data_tensor: torch.Tensor, /, sequence_mask: torch.Tensor, return_layers: Optional[List[int]] = None
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        :param data_tensor: input tensor of shape [B, T', F]
        :param sequence_mask: mask tensor where 1 defines positions within the sequence and 0 outside, shape: [B, T']
        :param return_layers: list of layer indices specifying which layers to return
        :return: (outputs, out_seq_mask)
            where outputs is a list of torch.Tensor of shape [B, T, F']
            for each of the layers in output_layers,
            out_seq_mask is a torch.Tensor of shape [B, T]

        F: input feature dim, F': internal and output feature dim
        T': data time dim, T: down-sampled time dim (internal time dim)
        """

        if return_layers is None:
            return_layers = [len(self.module_list)]
        x, sequence_mask = self.frontend(data_tensor, sequence_mask)  # [B, T, F']

        outputs = []
        assert max(return_layers) <= len(self.module_list)
        for i in range(max(return_layers)):
            x = self.module_list[i](x, sequence_mask)  # [B, T, F']
            if i + 1 in return_layers:
                outputs.append(x)

        return outputs, sequence_mask
