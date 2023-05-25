from __future__ import annotations

__all__ = ["ConformerMHSAV1", "ConformerMHSAV1Config"]
from dataclasses import dataclass
from typing import Optional
import torch

from i6_models.config import ModelConfiguration


@dataclass
class ConformerMHSAV1Config(ModelConfiguration):
    input_dim: int
    """input dim and total dimension for query/key and value projections, should be divisible by `num_att_heads`"""
    num_att_heads: int
    """number of attention heads"""
    att_weights_dropout: float
    """attention weights dropout"""
    dropout: float
    """multi-headed self attention output dropout"""

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

    def forward(self, input_tensor: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply layer norm and multi-head self attention and dropout
        :param Optional[torch.Tensor] key_padding_mask: could be a binary or float mask of shape (B, T)
        which will be applied/added to dot product, used to mask padded key positions out
        """

        output_tensor = self.layernorm(input_tensor)  # [B,T,F]

        output_tensor, _ = self.mhsa(
            output_tensor, output_tensor, output_tensor, key_padding_mask=key_padding_mask, need_weights=False
        )  # [B,T,F]
        output_tensor = torch.nn.functional.dropout(output_tensor, p=self.dropout, training=self.training)  # [B,T,F]

        return output_tensor
