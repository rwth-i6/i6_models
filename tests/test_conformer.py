from i6_models.parts.conformer.convolution import ConformerConvolutionV1
import torch


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
