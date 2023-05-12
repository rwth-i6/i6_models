from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV1
import torch
from torch import nn
from itertools import product

def test_ConformerPositionwiseFeedForwardV1():
    def get_output_shape(input_shape, input_dim, hidden_dim, dropout, activation):
        x = torch.randn(input_shape)
        conf_ffn_part = ConformerPositionwiseFeedForwardV1(
            input_dim, hidden_dim, dropout, activation
        )
        y = conf_ffn_part(x)
        return y.shape

    for input_dim, hidden_dim, dropout, activation in product(
        [10, 20], [100, 200], [0.1, 0.3], [nn.functional.silu, nn.functional.relu]
    ):
        input_shape = (10, 100, input_dim)
        assert get_output_shape(input_shape, input_dim, hidden_dim, dropout, activation) == input_shape

test_ConformerPositionwiseFeedForwardV1()
