__all__ = ["MultiheadAttentionV1", "MultiheadAttentionV1Config"]

import math

import torch
from torch import nn
from i6_models.config import ModelConfiguration
from i6_models.util import compat
from dataclasses import dataclass

@dataclass
class MultiheadAttentionV1Config(ModelConfiguration):
    input_dim: int
    num_att_heads: int
    att_weights_dropout: float
    dropout: float

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.input_dim % self.num_att_heads == 0, "input_dim must be divisible by num_att_heads"

class MultiheadAttentionV1(nn.Module):
    def __init__(
            self,
            cfg: MultiheadAttentionV1Config
    ):
        super().__init__()
        self.cfg = cfg
        self.num_att_heads = cfg.num_att_heads
        self.input_dim = cfg.input_dim
        self.dim_heads = self.input_dim // self.num_att_heads
        self.layernorm = torch.nn.LayerNorm(cfg.input_dim)

        self.out_proj = torch.nn.Linear(in_features=cfg.input_dim, out_features=cfg.input_dim, bias=True)
        self.in_proj = torch.nn.Linear(in_features=cfg.input_dim, out_features=3 * cfg.input_dim, bias=True)
    
        self.norm = math.sqrt(self.input_dim)
        self.softmax = nn.Softmax(-1)
        self.dropout = nn.Dropout(cfg.att_weights_dropout)

    def forward(self,  query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, key_padding_mask: torch.Tensor, need_weights: bool):

        inv_sequence_mask = compat.logical_not(key_padding_mask)
        assert query is value is key, "only supports self attention for now"

        batch_dim = query.shape[0]

        x = self.in_proj(query)
        hidden_dim = query.size(-1)
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
        if inv_sequence_mask is not None:
            inv_sequence_mask = inv_sequence_mask.view(batch_dim, 1, 1, inv_sequence_mask.size(1))
            dot = dot.masked_fill(inv_sequence_mask, -float('inf'))
        alpha = self.softmax(dot)
        alpha = self.dropout(alpha)

        att_out = torch.matmul(alpha, value)  # [B, D//H, T, D']
        att_out = torch.transpose(att_out, 1, 2)  # [B, D//H, T, D']
        att_out = att_out.reshape(batch_dim, -1, self.input_dim)  # [B, T, D]
        att_out = self.out_proj(att_out)

        return att_out, alpha
