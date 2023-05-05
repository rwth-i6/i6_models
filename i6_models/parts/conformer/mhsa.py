import torch

class ConformerMHSAV1(nn.Module):
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

        self.embed_dim = embed_dim
        self.num_heads = num_att_heads
        self.att_weights_dropout = att_weights_dropout
        self.dropout = dropout

        self.mhsa = torch.nn.MultiheadAttention(embed_dim, num_att_heads, dropout=att_weights_dropout)


    def forward(self, input: torch.Tensor):

        residual = input

        # layer norm, Multi-head self attention with dropout and residual connection
        output = torch.nn.LayerNorm(self.embed_dim)
        output, _ = self.mhsa(output)
        output = torch.nn.Dropout(self.dropout)
        output = output + residual

        return output


