from __future__ import annotations
import torch
from torch import nn

# TODO probably used for SpecAugment
from torchaudio.transforms import TimeMasking, TimeStretch, FrequencyMasking
from dataclasses import dataclass

from i6_models.config import ModelConfiguration

from i6_models.parts.conformer.convolution import ConformerConvolutionV1
from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV1
from i6_models.parts.conformer.mhsa import ConformerMHSAV1


@dataclass
class ConformerPositionwiseFeedForwardV1Config(ModelConfiguration):
    dim: int


@dataclass
class ConformerMHSAV1Config(ModelConfiguration):
    pass


@dataclass
class ConformerConvolutionV1Config(ModelConfiguration):
    pass


@dataclass
class ConformerBlockV1Config(ModelConfiguration):
    # nested configurations
    ff_cfg: ConformerPositionwiseFeedForwardV1  # Configuration for ConformerPositionwiseFeedForwardV1
    mhsa_cfg: ConformerMHSAV1Config  # Configuration for ConformerMHSAV1
    conv_cfg: ConformerConvolutionV1Config  # Configuration for ConformerConvolutionV1


@dataclass
class ConformerFrontendV1Config(ModelConfiguration):
    feature_dim: int  # Feature dimension of the input data
    hidden_dim: int  # Hidden dimension used in the model internally
    dropout: float  # Dropout value after linear transformation

    # TODO: Maybe put this in own config?
    conv_stride: int  # Stride of down-sampling cov
    conv_kernel: int  # Kernel size of down-sampling conv
    conv_padding: int  # Padding factor of down-sampling conv

    spec_aug_cfg: ModelConfiguration  # TODO


@dataclass
class ConformerEncoderV1Config(ModelConfiguration):
    # hyperparameters
    num_layers: int  # Number of conformer layers in the conformer encoder

    # nested configurations
    front_cfg: ConformerFrontendV1Config  # Configuration for ConformerFrontendV1
    block_cfg: ConformerBlockV1Config  # Configuration for ConformerBlockV1


class ConformerBlockV1(nn.Module):
    """
    Conformer block module
    """

    def __init__(self, cfg: ConformerBlockV1Config):
        """
        :param cfg: conformer block configuration with subunits for the different conformer parts
        """
        super().__init__()
        self.ff_1 = ConformerPositionwiseFeedForwardV1(cfg=cfg.ff_config)
        self.mhsa = ConformerMHSAV1(cfg=cfg.mhsa_config)
        self.conv = ConformerConvolutionV1(cfg=cfg.conv_config)
        self.ff_2 = ConformerPositionwiseFeedForwardV1(cfg=cfg.ff_config)
        self.final_layer_norm = torch.nn.LayerNorm(cfg.ff_config.dim)

    def forward(self, tensor: torch.Tensor):
        """
        :param tensor: input tensor of shape [B, T, F]
        :return: torch.Tensor of shape [B, T, F]
        """
        residual = tensor  #  [B, T, F]
        x = self.ff_1(residual)  #  [B, T, F]
        residual = 0.5 * x + residual  #  [B, T, F]
        x = self.mhsa(residual)  #  [B, T, F]
        residual = x + residual  # [B, T, F]
        x = self.conv(residual)  #  [B, T, F]
        residual = x + residual  # [B, T, F]
        x = self.ff_2(residual)  #  [B, T, F]
        x = 0.5 * x + residual  #  [B, T, F]
        x = self.final_layer_norm(x)  #  [B, T, F]
        return x


class ConformerFrontendV1(nn.Module):
    """
    Frontend part of the standard conformer doing down-sampling
    """

    def __init__(self, cfg: ConformerFrontendV1Config):
        """
        :param cfg: conformer frontend configuration
        """
        super().__init__()
        self.spec_aug = ...  # TODO
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

        F: feature dim after feature extraction, F': internal model feature dim
        T: data time dim, T': down-sampled time dim
        """
        # TODO: SpecAugment
        x = data_tensor.transpose(1, 2)  # [B, F, T]
        x = self.subsampling(x)  # [B, F, T']
        x = x.transpose(2, 1)  # [B, T', F]
        x = self.linear(x)  # [B, T', F']
        x = self.dropout(x)  # [B, T', F']

        return x


class ConformerEncoderV1(nn.Module):
    def __init__(self, cfg: ConformerEncoderV1Config):
        """
        :param cfg: conformer encoder configuration with subunits for frontend and conformer blocks
        """
        super().__init__()

        self.frontend = ConformerFrontendV1(cfg=cfg.front_cfg)
        self.block_stack = torch.nn.Sequential(*[ConformerBlockV1(cfg.block_cfg) for _ in range(cfg.num_layers)])

    def forward(self, data_tensor: torch.Tensor):
        """
        :param data_tensor: input tensor of shape [B, T, F]
        :return: torch.Tensor of shape [B, T', F']

        F: feature dim after feature extraction, F': internal model feature dim
        T: data time dim, T': down-sampled time dim
        """
        x = self.frontend(data_tensor)  # [B, T', F']
        x = self.block_stack(x)  # [B, T', F']
        return x
