from torch import nn


class ConformerFeedForwardv1(nn.Module):
    """
    Conformer feedforward module
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        """
        :param int input_dim: input dimension
        :param int hidden_dim: hidden dimension (normally set to 4*input_dim as suggested by the paper)
        :param float dropout: dropout probability
        """
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=input_dim)

        self.linear_layer_1 = nn.Linear(
            in_features=input_dim, out_features=hidden_dim, bias=True
        )

        self.swish_activation = nn.SiLU()

        self.linear_layer_2 = nn.Linear(
            in_features=hidden_dim, out_features=input_dim, bias=True
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tensor):
        """
        :param torch.Tensor tensor: input tensor of shape [B,T,F]
        :return: torch.Tensor of shape [B,T,F]
        """
        out_tensor = self.layer_norm(tensor)  # [B,T,F]

        out_tensor = self.linear_layer_1(out_tensor)  # [B,T,F]
        out_tensor = self.swish_activation(out_tensor)  # [B,T,F]

        out_tensor = self.linear_layer_2(out_tensor)  # [B,T,F]

        out_tensor = self.dropout(out_tensor)  # [B,T,F]

        return tensor
