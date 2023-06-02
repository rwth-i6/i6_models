
import torch
from torch import nn
from dataclasses import dataclass

from i6_models.config import ModelConfiguration, SubassemblyWithOptions
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
    MergerV1
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
        :param cfg: e_branchformer block configuration with subunits for the different e_branchformer parts
        """
        super().__init__()
        self.ff_1 = PositionwiseFeedForwardV1(cfg=cfg.ff_cfg)
        self.mhsa = MHSAV1(cfg=cfg.mhsa_cfg)
        self.cgmlp = ConvolutionalGatingMLPV1(cfg=cfg.cgmlp_cfg)
        self.merger = MergerV1(cfg=cfg.merger_cfg)
        self.ff_2 = PositionwiseFeedForwardV1(cfg=cfg.ff_cfg)
        self.final_layer_norm = torch.nn.LayerNorm(cfg.ff_cfg.input_dim)

    def forward(self, tensor: torch.Tensor, sequence_mask: torch.Tensor) -> torch.Tensor:
        """
        :param tensor: input tensor of shape [B,T,F]
        :param sequence_mask: mask tensor where 0 defines positions within the sequence and 1 outside, shape: [B, T]
        :return: torch.Tensor of shape [B, T, F]
        """
        residual = tensor  # [B, T, F]
        tensor = self.ff_1(residual)
        residual = 0.5*tensor + residual

        tensor_global = self.mhsa(residual, sequence_mask)
        tensor_local = self.cgmlp(residual)
        merger_tensor = self.merger(tensor_global, tensor_local)
        residual = merger_tensor + residual

        tensor = self.ff_2(residual)
        residual = 0.5*residual + tensor

        tensor = self.final_layer_norm(residual)
        return tensor


class EbranchformerEncoderV1Config(ModelConfiguration):
    """
    Attributes:
        num_layers: Number of e-branchformer layers in the e-branchformer encoder
        front_cfg: Configuration for FrontendV1
        block_cfg: Configuration for EbranchformerBlockV1
    """

    num_layers: int

    # nested configurations
    frontend: SubassemblyWithOptions
    block_cfg: EbranchformerBlockV1Config


class EbranchformerrEncoderV1(nn.Module):
    """
    Implementation of the Branchformer with Enhanced merging (short Conformer), as in the original publication.
    The model consists of a frontend and a stack of N e-branchformer blocks.
    C.f. https://arxiv.org/pdf/2210.00077.pdf
    """

    def __init__(self, cfg: EbranchformerEncoderV1Config):
        """
        :param cfg: conformer encoder configuration with subunits for frontend and conformer blocks
        """
        super().__init__()

        self.frontend = cfg.frontend.construct()
        self.module_list = torch.nn.ModuleList([EbranchformerBlockV1(cfg.block_cfg) for _ in range(cfg.num_layers)])

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