from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable
import torch

from i6_models.config import ModelConfiguration


@dataclass
class ConformerMHSAV1Config(ModelConfiguration):
    input_dim: int  # input dim and total dimension for query/key and value projections, should be dividable by `num_att_heads`
    num_att_heads: int  # number of attention heads
    key_padding_mask: Optional[
        torch.Tensor
    ] = None  # binary or float mask deciding if a key position should ignored or not, of shape (B, T_k)
    att_weights_dropout: float = 0.1  # attention weights dropout
    dropout: float = 0.1  # multi-headed self attention output dropout


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
        self.key_padding_mask = cfg.key_padding_mask
        self.dropout = torch.nn.Dropout(cfg.dropout)

    def forward(self, input: torch.Tensor):

        # layer norm, Multi-head self attention with dropout and residual connection
        output = self.layernorm(input)  # [B,T,F]
        output, _ = self.mhsa(output, output, output, key_padding_mask=self.key_padding_mask)  # [B,T,F]
        output = self.dropout(output)  # [B,T,F]

        return output
