from i6_models.parts.conformer.convolution import ConformerConvolutionV1, ConformerConvolutionV1Config
import torch
import torch.nn as nn


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
    assert get_output_shape(10, 50, 250, norm=nn.LayerNorm(250)) == (10, 50, 250)  # different norm
    assert get_output_shape(1, 50, 100) == (1, 50, 100)  # test with batch size 1
    assert get_output_shape(10, 1, 50) == (10, 1, 50)  # time dim 1
    assert get_output_shape(10, 10, 20, dropout=0.0) == (10, 10, 20)  # dropout 0
    assert get_output_shape(10, 10, 20, kernel_size=3) == (10, 10, 20)  # odd kernel size
    assert get_output_shape(10, 10, 20, kernel_size=32) == (10, 10, 20)  # even kernel size
