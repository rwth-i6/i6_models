__all__ = [
    "FeedForwardLayerV1Config",
    "FeedForwardLayerV1",
    "FeedForwardBlockV1Config",
    "FeedForwardBlockV1",
]

from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Tuple, Union, List

import torch
from torch import nn
import torch.nn.functional as F

from i6_models.config import ModelConfiguration


@dataclass
class FeedForwardLayerV1Config(ModelConfiguration):
    """
    Attributes:
        input_dim: input feature dimension
        output_dim: output feature dimension
        dropout: dropout probability
        activation: activation function applied after linear computation
    """

    input_dim: int
    output_dim: int
    dropout: float
    activation: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]

    def __post_init__(self):
        super().__post_init__()
        assert 0.0 <= self.dropout <= 1.0, "Dropout value must be a probability"


class FeedForwardLayerV1(nn.Module):
    """
    Simple feed-forward layer module consisting of:
        - linear
        - activation
        - dropout
    """

    def __init__(self, cfg: FeedForwardLayerV1Config):
        super().__init__()
        self.linear_ff = nn.Linear(in_features=cfg.input_dim, out_features=cfg.output_dim, bias=True)
        self.activation = cfg.activation
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self, tensor: torch.Tensor, sequence_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param tensor: shape [B,T,F], F=input_dim
        :param sequence_mask: shape [B,T]
        :return: shape [B,T,F'], F'=output_dim
        """
        tensor = self.linear_ff(tensor)  # [B,T,F]
        tensor = self.activation(tensor)  # [B,T,F]
        tensor = self.dropout(tensor)  # [B,T,F]
        return tensor, sequence_mask


@dataclass
class FeedForwardBlockV1Config(ModelConfiguration):
    """
    Configuration for the FeedForwardBlockV1 module.

    Attributes:
        input_dim: Input feature dimension.
        layer_sizes: List of hidden layer sizes.  The length of this list
                     determines the number of layers.
        dropouts: Dropout probability for each layer.
        layer_activations: List of activation function applied after each linear layer.
                           None represents no activation.
                           Must have the same length as layer_sizes.
        use_layer_norm: Whether to use Layer Normalization.
    """

    input_dim: int
    layer_sizes: List[int]
    dropouts: List[float]
    layer_activations: List[Optional[Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]]]
    use_layer_norm: bool = True

    def __post_init__(self):
        super().__post_init__()
        assert all(0.0 <= dropout <= 1.0 for dropout in self.dropouts), "Dropout values must be probabilities"
        assert len(self.layer_sizes) > 0, "layer_sizes must not be empty"
        assert len(self.layer_sizes) == len(self.layer_activations)
        assert len(self.layer_sizes) == len(self.dropouts)


class FeedForwardBlockV1(nn.Module):
    """
    A multi-layer feed-forward network block with optional Layer Normalization.
    """

    def __init__(self, cfg: FeedForwardBlockV1Config):
        super().__init__()
        self.cfg = cfg
        network_layers: List[nn.Module] = []
        prev_size = cfg.input_dim

        for i, layer_size in enumerate(cfg.layer_sizes):
            if cfg.use_layer_norm:
                network_layers.append(nn.LayerNorm(prev_size))
            network_layers.append(nn.Linear(prev_size, layer_size))
            prev_size = layer_size
            if cfg.layer_activations[i] is not None:
                network_layers.append(cfg.layer_activations[i])
            network_layers.append(nn.Dropout(cfg.dropouts[i]))

        self.output_dim = cfg.layer_sizes[-1]
        self.network = nn.Sequential(*network_layers)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feed-forward block.

        :param tensor: Input tensor of shape [B, T, F], where F is input_dim.
        :return: Output tensor of shape [B, T, output_dim].
        """
        return self.network(tensor)
