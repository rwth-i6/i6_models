import torch
from torch import nn
from itertools import product

from i6_models.parts.conformer.convolution import ConformerConvolutionV1
from i6_models.parts.conformer.feedforward import (
    ConformerPositionwiseFeedForwardV1,
    ConformerPositionwiseFeedForwardV1Config,
)

def test_conformer_convolution():
    def get_output_shape(batch, time, features, kernel_size=31, dropout=0.1):
        x = torch.randn(batch, time, features)
        conformer_conv_part = ConformerConvolutionV1(channels=features, kernel_size=kernel_size, dropout=dropout)
        y = conformer_conv_part(x)
        return y.shape

    assert get_output_shape(1, 50, 100) == (1, 50, 100)  # test with batch size 1
    assert get_output_shape(10, 50, 250) == (10, 50, 250)
    assert get_output_shape(10, 1, 50) == (10, 1, 50)  # time dim 1
    assert get_output_shape(10, 10, 20, dropout=0.0) == (10, 10, 20)  # dropout 0
    assert get_output_shape(10, 10, 20, kernel_size=3) == (10, 10, 20)  # odd kernel size
    assert get_output_shape(10, 10, 20, kernel_size=32) == (10, 10, 20)  # even kernel size


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
