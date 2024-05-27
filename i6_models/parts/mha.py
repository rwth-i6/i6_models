__all__ = ["MultiheadAttentionV1", "MultiheadAttentionV1Config", "MHARef"]

import math
from dataclasses import dataclass
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
    def __init__(self, cfg: MultiheadAttentionV1Config):
        super().__init__()
        self.cfg = cfg
        self.num_att_heads = cfg.num_att_heads
        self.input_dim = cfg.input_dim
        self.dim_heads = self.input_dim // self.num_att_heads
        self.layernorm = torch.nn.LayerNorm(cfg.input_dim)

        self.out_proj = nn.Linear(in_features=cfg.input_dim, out_features=cfg.input_dim, bias=True)
        self.in_proj = nn.Linear(in_features=cfg.input_dim, out_features=3 * cfg.input_dim, bias=True)

        self.norm = math.sqrt(self.input_dim)
        self.softmax = nn.Softmax(-1)
        self.dropout = nn.Dropout(cfg.att_weights_dropout)


    def forward(
            self,  
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            key_padding_mask: torch.Tensor):

        if key_padding_mask is not None:
            inv_sequence_mask = compat.logical_not(key_padding_mask)
        else:
            inv_sequence_mask = None
        assert query is value is key, "only supports self attention for now"

        batch_dim , num_tokens, embed_dim = query.shape
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

        alpha = self.softmax(dot)# [B, D//H, T, T]
        alpha = self.dropout(alpha)

        att_out = torch.matmul(alpha, value)  # [B, D//H, T, D']
        att_out = torch.transpose(att_out, 1, 2)  # [B, T, D//H, D']
        att_out = att_out.reshape(batch_dim, -1, self.input_dim)  # [B, T, D]
        att_out = self.out_proj(att_out)

        return att_out, alpha


class MHARef(nn.Module):
    def __init__(self, d_in, d_out, num_heads, dropout=0.0, qkv_bias=False):
        super().__init__()

        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.qkv = torch.nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = torch.nn.Linear(d_in, d_out)
        self.dropout = torch.nn.Dropout(dropout)

        self.register_buffer(
            "mask", torch.triu(torch.ones(d_in, d_in), diagonal=1)
        )

    def forward(self, x, y, z, key_padding_mask):
        batch_size, num_tokens, embed_dim = x.shape

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        #print(f'ref pre :{x[0]}')
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_head, num_tokens, head_dim)
        queries, keys, values = qkv.unbind(0)

        # (b, num_heads, num_tokens, head_dim) --> (b, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(-2, -1)
        attn_scores = attn_scores.masked_fill(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**-0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # (b, num_heads, num_tokens, num_tokens) --> (b, num_heads, num_tokens, head_dim)
        context_vec = attn_weights @ values

        # (b, num_heads, num_tokens, head_dim) --> (b, num_tokens, num_heads, head_dim)
        context_vec = context_vec.transpose(1, 2)

        # (b, num_tokens, num_heads, head_dim) --> (b, num_tokens, embed_dim)
        context_vec = context_vec.contiguous().view(batch_size, num_tokens, embed_dim)
        context_vec = self.proj(context_vec)

        return context_vec, attn_weights
