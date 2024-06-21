__all__ = ["MultiheadSelfAttentionV1", "MultiheadSelfAttentionV1Config"]

import math
from dataclasses import dataclass
import torch

from i6_models.config import ModelConfiguration
from i6_models.util import compat


@dataclass
class MultiheadSelfAttentionV1Config(ModelConfiguration):
    """
    Attributes:
        input_dim: input dim and total dimension for query/key and value projections, should be divisible by `num_att_heads`
        num_att_heads: number of attention heads
        att_weights_dropout: attention weights dropout
        dropout: attention weight dropout probability
    """

    input_dim: int
    num_att_heads: int
    att_weights_dropout: float
    dropout: float

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.input_dim % self.num_att_heads == 0, "input_dim must be divisible by num_att_heads"


class MultiheadSelfAttentionV1(torch.nn.Module):
    """
    Native Multihead Self Attention implementation based on 'Attention Is All You Need'
    """

    def __init__(self, cfg: MultiheadSelfAttentionV1Config):
        super().__init__()
        self.cfg = cfg
        self.num_att_heads = cfg.num_att_heads
        self.input_dim = cfg.input_dim
        self.dim_heads = self.input_dim // self.num_att_heads
        self.layernorm = torch.nn.LayerNorm(cfg.input_dim)

        self.out_proj = torch.nn.Linear(in_features=cfg.input_dim, out_features=cfg.input_dim, bias=True)
        self.in_proj = torch.nn.Linear(in_features=cfg.input_dim, out_features=3 * cfg.input_dim, bias=True)

        self.norm = math.sqrt(float(self.input_dim / self.num_att_heads))
        self.softmax = torch.nn.Softmax(-1)
        self.dropout = torch.nn.Dropout(cfg.att_weights_dropout)

    def forward(self, qkv: torch.Tensor, key_padding_mask: torch.Tensor):
        """
        Computes the forward pass of the MultiheadSelfAttentionV1 module.
        Attributes:

            qkv (torch.Tensor): The input tensor of shape (B, T, F).
            key_padding_mask (torch.Tensor): The key padding mask tensor of shape (batch_dim, num_tokens).
        """

        batch_dim, num_tokens, embed_dim = qkv.shape
        x = self.in_proj(qkv)

        hidden_dim = qkv.size(-1)
        query, key, value = x.unflatten(-1, (3, hidden_dim)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()

        query = query.view(batch_dim, -1, self.num_att_heads, self.dim_heads)  # [B, T, D//H, D']
        key = key.view(batch_dim, -1, self.num_att_heads, self.dim_heads)  # [B, T, D//H, D']
        value = value.view(batch_dim, -1, self.num_att_heads, self.dim_heads)  # [B, T, D//H, D']

        query = torch.transpose(query, 1, 2)  # [B, D//H, T, D']
        key = torch.transpose(key, 1, 2)  # [B, D//H, T, D']
        value = torch.transpose(value, 1, 2)  # [B, D//H, T, D']

        key = torch.transpose(key, -2, -1)  # [B, D//H, D', T]

        dot = torch.matmul(query, key)  # [B, D//H, T, T]
        dot = dot / self.norm

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(batch_dim, 1, 1, key_padding_mask.size(1))
            dot = dot.masked_fill(key_padding_mask, -float("inf"))

        alpha = self.softmax(dot)  # [B, D//H, T, T]
        alpha = self.dropout(alpha)

        att_out = torch.matmul(alpha, value)  # [B, D//H, T, D']
        att_out = torch.transpose(att_out, 1, 2)  # [B, T, D//H, D']
        att_out = att_out.reshape(batch_dim, -1, self.input_dim)  # [B, T, D]
        att_out = self.out_proj(att_out)

        return att_out, alpha
