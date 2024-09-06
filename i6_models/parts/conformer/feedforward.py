from __future__ import annotations

__all__ = [
    "ConformerPositionwiseFeedForwardV1",
    "ConformerPositionwiseFeedForwardV1Config",
    "ConformerPositionwiseFeedForwardV2",
    "ConformerPositionwiseFeedForwardV2Config",
]

from dataclasses import dataclass
from typing import Callable, Optional, Literal

import torch
from torch import nn

from i6_models.config import ModelConfiguration
from i6_models.parts.dropout import BroadcastDropout


@dataclass
class ConformerPositionwiseFeedForwardV1Config(ModelConfiguration):
    """
    Attributes:
        input_dim: input dimension
        hidden_dim: hidden dimension (normally set to 4*input_dim as suggested by the paper)
        dropout: dropout probability
        activation: activation function
    """

    input_dim: int
    hidden_dim: int
    dropout: float
    activation: Callable[[torch.Tensor], torch.Tensor] = nn.functional.silu


class ConformerPositionwiseFeedForwardV1(nn.Module):
    """
    Conformer feedforward module
    """

    def __init__(self, cfg: ConformerPositionwiseFeedForwardV1Config):
        super().__init__()

        self.layer_norm = nn.LayerNorm(cfg.input_dim)
        self.linear_ff = nn.Linear(in_features=cfg.input_dim, out_features=cfg.hidden_dim, bias=True)
        self.activation = cfg.activation
        self.linear_out = nn.Linear(in_features=cfg.hidden_dim, out_features=cfg.input_dim, bias=True)
        self.dropout = cfg.dropout

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        :param tensor: shape [B,T,F], F=input_dim
        :return: shape [B,T,F], F=input_dim
        """
        tensor = self.layer_norm(tensor)
        tensor = self.linear_ff(tensor)  # [B,T,F]
        tensor = self.activation(tensor)  # [B,T,F]
        tensor = nn.functional.dropout(tensor, p=self.dropout, training=self.training)  # [B,T,F]
        tensor = self.linear_out(tensor)  # [B,T,F]
        tensor = nn.functional.dropout(tensor, p=self.dropout, training=self.training)  # [B,T,F]
        return tensor


@dataclass
class ConformerPositionwiseFeedForwardV2Config(ModelConfiguration):
    """
    New attribute:
        dropout_broadcast_axes: string of axes to which dropout is broadcast, e.g. "T" for broadcasting to the time axis
                                setting to None to disable broadcasting
    Default value for `activation` removed
    """

    input_dim: int
    hidden_dim: int
    dropout: float
    activation: Callable[[torch.Tensor], torch.Tensor]
    dropout_broadcast_axes: Optional[Literal["B", "T", "BT"]]

    def check_valid(self):
        assert self.dropout_broadcast_axes in [
            None,
            "B",
            "T",
            "BT",
        ], "invalid value, supported are None, 'B', 'T' and 'BT'"

    def __post__init__(self):
        super().__post_init__()
        self.check_valid()


class ConformerPositionwiseFeedForwardV2(ConformerPositionwiseFeedForwardV1):
    """
    Augments ConformerPositionwiseFeedForwardV1 with dropout broadcasting
    """

    def __init__(self, cfg: ConformerPositionwiseFeedForwardV2Config):
        super().__init__(cfg)

        self.dropout = BroadcastDropout(cfg.dropout, dropout_broadcast_axes=cfg.dropout_broadcast_axes)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        :param tensor: shape [B,T,F], F=input_dim
        :return: shape [B,T,F], F=input_dim
        """
        tensor = self.layer_norm(tensor)
        tensor = self.linear_ff(tensor)  # [B,T,F]
        tensor = self.activation(tensor)  # [B,T,F]

        tensor = self.dropout(tensor)  # [B,T,F]
        tensor = self.linear_out(tensor)  # [B,T,F]
        tensor = self.dropout(tensor)  # [B,T,F]

        return tensor
