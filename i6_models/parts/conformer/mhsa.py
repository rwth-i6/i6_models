import torch


class ConformerMHSAV1(torch.nn.Module):
    """
    Conformer multi-headed self-attention module without relative positional embedding
    :param embed_dim: model dimension, `embed_dim // num_att_heads` becomes the key and value projection dimensions
    :param num_att_heads: number of attention heads
    :param att_weights_dropout: attention weights dropout
    :param dropout: multi-headed self attention output dropout
    """

    def __init__(
        self,
        embed_dim: int,
        num_att_heads: int,
        att_weights_dropout: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layernorm = torch.nn.LayerNorm(embed_dim)
        self.mhsa = torch.nn.MultiheadAttention(embed_dim, num_att_heads, dropout=att_weights_dropout)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, input: torch.Tensor):

        residual = input # [B,T,F]

        # layer norm, Multi-head self attention with dropout and residual connection
        output = self.layernorm(input) # [B,T,F]
        output, _ = self.mhsa(output) # [B,T,F]
        output = self.dropout(output) # [B,T,F]
        output = output + residual # [B,T,F]

        return output
