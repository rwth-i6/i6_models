__all__ = ["FeedForwardEncoderV1Config", "FeedForwardEncoderV1"]

from typing import Tuple
from dataclasses import dataclass
import torch
from torch import nn

from i6_models.parts.ffnn import FeedForwardLayerV1, FeedForwardLayerV1Config
from i6_models.config import ModelConfiguration, ModuleFactoryV1


@dataclass
class FeedForwardEncoderV1Config(ModelConfiguration):
    """
    Attributes:
        num_layers: number of feed-forward layers
        frontend: module factory for the frontend
        layer_cfg: configuration object for each feed-forward layer
    """

    num_layers: int
    frontend: ModuleFactoryV1
    layer_cfg: FeedForwardLayerV1Config


class FeedForwardEncoderV1(nn.Module):
    """
    Simple feed-forward encoder.
    Subsampling can be achieved by setting stride > 1 in the frontend config.
    """

    def __init__(self, cfg: FeedForwardEncoderV1Config):
        super().__init__()
        self.frontend = cfg.frontend()
        self.module_list = nn.ModuleList([FeedForwardLayerV1(cfg.layer_cfg) for _ in range(cfg.num_layers)])

    def forward(self, data_tensor: torch.Tensor, sequence_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, sequence_mask = self.frontend(data_tensor, sequence_mask)  # [B, T, F']
        for module in self.module_list:
            x, sequence_mask = module(x, sequence_mask)  # [B, T, F']

        return x, sequence_mask
