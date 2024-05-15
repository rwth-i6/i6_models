from itertools import product

import torch
from torch import nn

from i6_models.parts.frontend.common import apply_same_padding, get_same_padding


def test_output_shape():
    # test for even and odd dim
    last_dim = 101
    pre_last_dim = 100

    iff = lambda x, y: x and y or not x and not y  # x <=> y
    strided_dim = lambda d, s: (d - 1) // s + 1  # expected out dimension for strided conv

    # `get_same_padding` seems to work for some stride > 1
    for kernel in product(range(1, 21), repeat=2):
        conv = nn.Conv2d(1, 1, kernel_size=kernel, stride=(1, 1), padding=get_same_padding(kernel))

        x = torch.randn(1, 1, pre_last_dim, last_dim)

        out = conv(x)

        # we expect `get_same_padding` to only cover odd kernel sizes
        assert all(
            iff(out_dim == in_dim, k % 2 == 1) for in_dim, out_dim, k in zip(x.shape[2:], out.shape[2:], kernel)
        ), f"Failed for {x.shape=}, {out.shape=}, {kernel=} and stride=1"

    for kernel, stride in product(product(range(1, 21), repeat=2), range(1, 7)):
        conv = nn.Conv2d(1, 1, kernel_size=kernel, stride=(1, stride))

        x = torch.randn(1, 1, pre_last_dim, last_dim)
        x_padded = apply_same_padding(x, kernel)

        out = conv(x_padded)

        # correct out dimensions for all possible kernel sizes and strides
        assert all(
            out_dim == strided_dim(in_dim, s)
            for in_dim, out_dim, k, s in zip(x.shape[2:], out.shape[2:], kernel, (1, stride))
        ), f"Failed for {x.shape=}, {out.shape=}, {kernel=} and {stride=}"
