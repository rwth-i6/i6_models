from itertools import product

import torch
from torch import nn

from i6_models.parts.e_branchformer.cgmlp import ConvolutionalGatingMLPV1Config, ConvolutionalGatingMLPV1


def test_ConvolutionalGatingMLPV1():
    def get_output_shape(input_shape, hidden_dim, kernel_size, dropout, activation):
        input_dim = input_shape[-1]
        cfg = ConvolutionalGatingMLPV1Config(input_dim, hidden_dim, kernel_size, dropout, activation)
        e_branchformer_cgmlp_part = ConvolutionalGatingMLPV1(cfg)
        x = torch.randn(input_shape)
        y = e_branchformer_cgmlp_part(x)
        return y.shape

    for input_shape, hidden_dim, kernel_size, dropout, activation in product(
        [(100, 5, 20), (200, 30, 10)], [120, 60], [9, 15], [0.1, 0.3], [nn.functional.gelu, nn.functional.relu]
    ):
        assert get_output_shape(input_shape, hidden_dim, kernel_size, dropout, activation) == input_shape
