from __future__ import annotations

__all__ = ["ConformerMHSAV1", "ConformerMHSAV1Config", "ConformerMHSAV2", "ConformerMHSAV2Config"]
from dataclasses import dataclass
import torch

from i6_models.config import ModelConfiguration
from i6_models.util import compat

from typing import Optional


@dataclass
class ConformerMHSAV1Config(ModelConfiguration):
    """
    Attributes:
        input_dim: input dim and total dimension for query/key and value projections, should be divisible by `num_att_heads`
        num_att_heads: number of attention heads
        att_weights_dropout: attention weights dropout
        dropout: multi-headed self attention output dropout
    """

    input_dim: int
    num_att_heads: int
    att_weights_dropout: float
    dropout: float

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.input_dim % self.num_att_heads == 0, "input_dim must be divisible by num_att_heads"


class ConformerMHSAV1(torch.nn.Module):
    """
    Conformer multi-headed self-attention module
    """

    def __init__(self, cfg: ConformerMHSAV1Config):

        super().__init__()

        self.layernorm = torch.nn.LayerNorm(cfg.input_dim)
        self.mhsa = torch.nn.MultiheadAttention(
            cfg.input_dim, cfg.num_att_heads, dropout=cfg.att_weights_dropout, batch_first=True
        )
        self.dropout = cfg.dropout

    def forward(self, input_tensor: torch.Tensor, sequence_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply layer norm and multi-head self attention and dropout

        :param input_tensor: Input to the self attention of shape (B, T, F)
        :param sequence_mask: bool mask of shape (B, T), True signals within sequence, False outside, will be inverted to match the torch.nn.MultiheadAttention module
        which will be applied/added to dot product, used to mask padded key positions out
        """
        inv_sequence_mask = compat.logical_not(sequence_mask)
        output_tensor = self.layernorm(input_tensor)  # [B,T,F]

        output_tensor, _ = self.mhsa(
            output_tensor, output_tensor, output_tensor, key_padding_mask=inv_sequence_mask, need_weights=False
        )  # [B,T,F]
        output_tensor = torch.nn.functional.dropout(output_tensor, p=self.dropout, training=self.training)  # [B,T,F]

        return output_tensor


@dataclass
class ConformerMHSAV2Config(ConformerMHSAV1Config):
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


class ConformerMHSAV2(ConformerMHSAV1):
    """
    Augments ConformerMHSAV1 with dropout broadcasting
    """

    def __init__(self, cfg: ConformerMHSAV2Config):

        super().__init__(cfg)

        self.dropout = torch.nn.Dropout1d(cfg.dropout) if cfg.dropout_broadcast_axes else torch.nn.Dropout(cfg.dropout)
        self.dropout_broadcast_axes = cfg.dropout_broadcast_axes

    def forward(self, input_tensor: torch.Tensor, sequence_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply layer norm and multi-head self attention and dropout

        :param input_tensor: Input to the self attention of shape (B, T, F)
        :param sequence_mask: Bool mask of shape (B, T), True signals within sequence, False outside, will be inverted to match the torch.nn.MultiheadAttention module
                              which will be applied/added to dot product, used to mask padded key positions out
        """
        inv_sequence_mask = compat.logical_not(sequence_mask)
        output_tensor = self.layernorm(input_tensor)  # [B,T,F]

        output_tensor, _ = self.mhsa(
            output_tensor, output_tensor, output_tensor, key_padding_mask=inv_sequence_mask, need_weights=False
        )  # [B,T,F]

        if self.dropout_broadcast_axes is None:
            output_tensor = self.dropout(output_tensor)
        elif self.dropout_broadcast_axes == "T":
            output_tensor = self.dropout(output_tensor.transpose(1, 2)).transpose(1, 2)
        elif self.dropout_broadcast_axes == "B":
            output_tensor = self.dropout(output_tensor.permute(1, 2, 0)).permute(2, 0, 1)
        elif self.dropout_broadcast_axes == "BT":
            batch_dim_size = output_tensor.shape[0]
            feature_dim_size = output_tensor.shape[-1]

            output_tensor = (
                self.dropout(output_tensor.reshape(-1, feature_dim_size).transpose(0, 1))
                .transpose(0, 1)
                .reshape(batch_dim_size, -1, feature_dim_size)
            )

        return output_tensor  # [B,T,F]
