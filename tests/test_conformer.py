from i6_models.parts.conformer.convolution import ConformerConvolutionV1
import torch


def test_conformer_convolution():
    def get_output_shape(batch, time, features):
        x = torch.randn(batch, time, features)
        conformer_conv_part = ConformerConvolutionV1(channels=features, kernel_size=31, dropout=0.1)
        y = conformer_conv_part(x)
        return y.shape

    assert get_output_shape(1, 50, 100) == (1, 50, 100)  # test with batch size 1
    assert get_output_shape(10, 50, 250) == (10, 50, 250)
    assert get_output_shape(10, 1, 50) == (10, 1, 50)  # time dim 1
