from __future__ import annotations

__all__ = [
    "ConformerPositionwiseFeedForwardV1",
    "ConformerPositionwiseFeedForwardV1Config",
    "ConformerPositionwiseFeedForwardV2",
    "ConformerPositionwiseFeedForwardV2Config",
]

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import nn

from i6_models.config import ModelConfiguration


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
class ConformerPositionwiseFeedForwardV2Config(ConformerPositionwiseFeedForwardV1Config):
    """
    New attribute:
        dropout_broadcast_axes: string of axes to which dropout is broadcast, e.g. "T" for broadcasting to the time axis
                                setting to None to disable broadcasting
    """

    dropout_broadcast_axes: Optional[str] = None

    def check_valid(self):
        assert self.dropout_broadcast_axes is None or self.dropout_broadcast_axes in [
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

        self.dropout = nn.Dropout1d(cfg.dropout) if cfg.dropout_broadcast_axes else nn.Dropout(cfg.dropout)
        self.dropout_broadcast_axes = cfg.dropout_broadcast_axes

    def _broadcast_dropout(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.dropout_broadcast_axes is None:
            tensor = self.dropout(tensor)
        elif self.dropout_broadcast_axes == "T":
            tensor = self.dropout(tensor.transpose(1, 2)).transpose(1, 2)
        elif self.dropout_broadcast_axes == "B":
            tensor = self.dropout(tensor.permute(1, 2, 0)).permute(2, 0, 1)
        elif self.dropout_broadcast_axes == "BT":
            batch_dim_size = tensor.shape[0]
            feature_dim_size = tensor.shape[-1]

            tensor = (
                self.dropout(tensor.reshape(-1, feature_dim_size).transpose(0, 1))
                .transpose(0, 1)
                .reshape(batch_dim_size, -1, feature_dim_size)
            )
        return tensor

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        :param tensor: shape [B,T,F], F=input_dim
        :return: shape [B,T,F], F=input_dim
        """
        tensor = self.layer_norm(tensor)
        tensor = self.linear_ff(tensor)  # [B,T,F]
        tensor = self.activation(tensor)  # [B,T,F]

        tensor = self._broadcast_dropout(tensor)  # [B,T,F]
        tensor = self.linear_out(tensor)  # [B,T,F]
        tensor = self._broadcast_dropout(tensor)  # [B,T,F]

        return tensor
