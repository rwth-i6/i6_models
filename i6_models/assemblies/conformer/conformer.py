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
class ConformerFeedForwardV1Config(ModelConfiguration):
    dim: int


@dataclass
class ConformerMHSAV1Config(ModelConfiguration):
    pass


@dataclass
class ConformerConvV1Config(ModelConfiguration):
    pass


@dataclass
class ConformerBlockV1Config(ModelConfiguration):
    # hyperparameters
    norm_dim: int

    # nested configurations
    ff_cfg: ConformerFeedForwardV1Config  # Configuration for ConformerPositionwiseFeedForwardV1
    mhsa_cfg: ConformerMHSAV1Config  # Configuration for ConformerMHSAV1
    conv_cfg: ConformerConvV1Config  # Configuration for ConformerConvolutionV1


@dataclass
class ConformerFrontendV1Config(ModelConfiguration):
    feature_dim: int
    hidden_dim: int
    dropout: float

    # TODO: Maybe put this in own config?
    conv_stride: int
    conv_kernel: int
    conv_stride: int
    conv_padding: int

    spec_aug_cfg: ModelConfiguration  # TODO


@dataclass
class ConformerV1Config(ModelConfiguration):
    # hyperparameters
    num_layers: int

    # nested configurations
    front_cfg: ConformerFrontendV1Config
    block_cfg: ConformerBlockV1Config


class ConformerBlockV1(nn.Module):
    """
    Conformer block module
    """

    def __init__(self, cfg: ConformerBlockV1Config):
        """
        :param cfg: Conformer Block Configuration with subunits for the different Conformer parts
        """
        super().__init__()

        self.ff_config = cfg.ff_cfg
        self.mhsa_config = cfg.mhsa_cfg
        self.conv_config = cfg.conv_cfg

        self.ff_1 = ConformerPositionwiseFeedForwardV1(cfg=self.ff_config)
        self.mhsa = ConformerMHSAV1(cfg=self.mhsa_config)
        self.conv = ConformerConvolutionV1(cfg=self.conv_config)
        self.ff_2 = ConformerPositionwiseFeedForwardV1(cfg=self.ff_config)
        self.final_layer_norm = torch.nn.LayerNorm(self.ff_config.dim)

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
        x = 0.5 * x + residual #  [B, T, F]
        x = self.final_layer_norm(x)  #  [B, T, F]
        return x


class ConformerFrontendV1(nn.Module):
    """
    Frontend part of the standard conformer doing down-sampling
    """

    def __init__(self, cfg: ConformerFrontendV1Config):
        """
        :param cfg: Conformer Frontend Configuration
        """
        super().__init__()

        self.feature_dim = cfg.feature_dim
        self.hidden_dim = cfg.hidden_dim
        self.dropout_value = cfg.dropout
        self.conv_stride = cfg.conv_stride
        self.conv_kernel = cfg.conv_kernel
        self.conv_padding = cfg.conv_padding

        self.spec_aug = ...  # TODO
        self.subsampling = nn.Conv1d(
            in_channels=self.feature_dim,
            out_channels=self.feature_dim,
            kernel_size=self.conv_kernel,
            stride=self.conv_stride,
            padding=self.conv_padding,
        )
        self.linear = nn.Linear(in_features=self.feature_dim, out_features=self.hidden_dim)
        self.dropout = nn.Dropout(p=self.dropout_value)

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
    def __init__(self, cfg: ConformerV1Config):
        """
        :param cfg: Conformer Encoder Configuration with subunits for Frontend and Conformer blocks
        """
        super().__init__()

        self.num_layers = cfg.num_layers

        self.front_cfg = cfg.front_cfg
        self.block_cfg = cfg.block_cfg

        self.frontend = ConformerFrontendV1(cfg=self.front_cfg)
        self.block_stack = torch.nn.Sequential(*[ConformerBlockV1(self.block_cfg) for _ in range(self.num_layers)])

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
