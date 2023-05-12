from itertools import product

import torch
from torch import nn

from i6_models.parts.conformer.feedforward import (
    ConformerPositionwiseFeedForwardV1,
    ConformerPositionwiseFeedForwardV1Config,
)


def test_ConformerPositionwiseFeedForwardV1():
    def get_output_shape(input_shape, input_dim, hidden_dim, dropout, activation):
        x = torch.randn(input_shape)
        cfg = ConformerPositionwiseFeedForwardV1Config(input_dim, hidden_dim, dropout, activation)
        conf_ffn_part = ConformerPositionwiseFeedForwardV1(cfg)
        y = conf_ffn_part(x)
        return y.shape

    for input_dim, hidden_dim, dropout, activation in product(
        [10, 20], [100, 200], [0.1, 0.3], [nn.functional.silu, nn.functional.relu]
    ):
        input_shape = (10, 100, input_dim)
        assert get_output_shape(input_shape, input_dim, hidden_dim, dropout, activation) == input_shape
