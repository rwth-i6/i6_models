__all__ = [
    "FeedForwardConfig",
    "FeedForwardModel"
]

from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F

from i6_models.config import ModelConfiguration

@dataclass
class FeedForwardLayerV1Config(ModelConfiguration):
    """
    Attributes:
        in_features: input feature dimension
        hidden_dim: output feature dimension
        dropout: dropout probability
        activation: activation function applied after linear computation
    """
    input_dim: int
    hidden_dim: int
    dropout: float
    activation: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]

    def __post_init__(self):
        super().__post_init__()
        assert 0.0 <= dropout <= 1.0, "Dropout value must be a probability"


class FeedForwardLayerV1(torch.nn.Module):
    """
    Simple feed-forward layer module consisting of:
        - linear
        - activation
        - dropout
    """

    def __init__(self, cfg: FeedForwardLayerV1Config):
        super().__init__()
        self.linear_ff = torch.nn.Linear(in_features=cfg.input_dim, out_features=cfg.hidden_dim, bias=True)
        self.activation = cfg.activation
        self.dropout = torch.nn.Dropout(cfg.dropout)

    def forward(self, tensor: torch.Tensor, sequence_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param tensor: shape [B,T,F], F=input_dim
        :param sequence_mask: shape [B,T]
        :return: shape [B,T,F'], F=input_dim
        """
        tensor = self.linear_ff(tensor)  # [B,T,F]
        tensor = self.activation(tensor)  # [B,T,F]
        tensor = self.dropout(tensor)  # [B,T,F]
        return tensor, sequence_mask
