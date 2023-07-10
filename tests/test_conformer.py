from __future__ import annotations
from itertools import product

import torch
from torch import nn

from i6_models.parts.conformer.convolution import ConformerConvolutionV1, ConformerConvolutionV1Config
from i6_models.parts.conformer.feedforward import (
    ConformerPositionwiseFeedForwardV1,
    ConformerPositionwiseFeedForwardV1Config,
)
from i6_models.parts.conformer.mhsa import ConformerMHSAV1Config, ConformerMHSAV1
from i6_models.parts.conformer.norm import LayerNormNC


def test_conformer_convolution_output_shape():
    def get_output_shape(batch, time, features, norm=None, kernel_size=31, dropout=0.1, activation=nn.functional.silu):
        x = torch.randn(batch, time, features)
        if norm is None:
            norm = nn.BatchNorm1d(features)
        cfg = ConformerConvolutionV1Config(
            channels=features, kernel_size=kernel_size, dropout=dropout, activation=activation, norm=norm
        )
        conformer_conv_part = ConformerConvolutionV1(cfg)
        y = conformer_conv_part(x)
        return y.shape

    assert get_output_shape(10, 50, 250) == (10, 50, 250)
    assert get_output_shape(10, 50, 250, activation=nn.functional.relu) == (10, 50, 250)  # different activation
    assert get_output_shape(10, 50, 250, norm=LayerNormNC(250)) == (10, 50, 250)  # different norm
    assert get_output_shape(1, 50, 100) == (1, 50, 100)  # test with batch size 1
    assert get_output_shape(10, 1, 50) == (10, 1, 50)  # time dim 1
    assert get_output_shape(10, 10, 20, dropout=0.0) == (10, 10, 20)  # dropout 0
    assert get_output_shape(10, 10, 20, kernel_size=3) == (10, 10, 20)  # odd kernel size


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


def test_ConformerMHSAV1():
    def get_output_shape(input_shape, cfg, **kwargs):

        input = torch.randn(input_shape)
        output = ConformerMHSAV1(cfg)(input, **kwargs)

        return list(output.shape)

    # with key padding mask
    input_shape = [4, 15, 32]  # B,T,F
    cfg = ConformerMHSAV1Config(32, 8, 0.2, 0.3)
    assert get_output_shape(input_shape, cfg, sequence_mask=(torch.randint(0, 2, input_shape[:2]) > 0)) == [4, 15, 32]


def test_layer_norm_nc():
    torch.manual_seed(42)

    def get_output(shape, norm):
        x = torch.randn(shape)
        out = norm(x)
        return out

    # test with different shape
    torch_ln = get_output([10, 50, 250], nn.LayerNorm(250))
    custom_ln = get_output([10, 250, 50], LayerNormNC(250))
    torch.allclose(torch_ln, custom_ln.transpose(1, 2))

    torch_ln = get_output([10, 8, 23], nn.LayerNorm(23))
    custom_ln = get_output([10, 23, 8], LayerNormNC(23))
    torch.allclose(torch_ln, custom_ln.transpose(1, 2))
