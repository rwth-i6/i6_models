from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable
import torch

from i6_models.config import ModelConfiguration


@dataclass
class ConformerMHSAV1Config(ModelConfiguration):
    input_dim: int
    """input dim and total dimension for query/key and value projections, should be dividable by `num_att_heads`"""
    num_att_heads: int
    """number of attention heads"""
    att_weights_dropout: float
    """attention weights dropout"""
    dropout: float
    """multi-headed self attention output dropout"""


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
        self.dropout = torch.nn.Dropout(cfg.dropout)

    def forward(self, input_tensor: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):

        # layer norm, Multi-head self attention with dropout
        output_tensor = self.layernorm(input_tensor)  # [B,T,F]
        output_tensor, _ = self.mhsa(
            output_tensor, output_tensor, output_tensor, key_padding_mask=key_padding_mask
        )  # [B,T,F]
        output_tensor = self.dropout(output_tensor)  # [B,T,F]

        return output_tensor
