from itertools import product

import torch
from torch import nn
from torch.nn import functional as F

from i6_models.parts.frontend.window_convolution import WindowConvolutionFrontendV1Config, WindowConvolutionFrontendV1


def test_output_shape():
    in_features = 80
    out_features = 2048
    dropout = 0.1
    max_seq_lens = 100

    for window_size, stride in product(range(1, 22), range(1, 5)):
        frontend = WindowConvolutionFrontendV1(
            WindowConvolutionFrontendV1Config(
                input_dim=80,
                output_dim=out_features,
                kernel_size=window_size,
                dropout=dropout,
                stride=stride,
                activation=F.relu,
            )
        )

        feat_len = torch.arange(start=1, end=max_seq_lens + 1)
        mask = torch.less(torch.arange(max_seq_lens)[None, :], feat_len[:, None])

        features = torch.empty((max_seq_lens, max_seq_lens, in_features))

        out, out_mask = frontend(features, mask)

        expected_out_len = (feat_len - 1) // stride + 1
        expected_shape = (max_seq_lens, expected_out_len[-1], out_features)
        assert out.shape == expected_shape, f"Output with shape {out.shape} not as expected {expected_shape}"
        for i in range(expected_out_len[-1] - 1):
            # check if masks are correct
            assert out_mask[i, expected_out_len[i] - 1] and not out_mask[i, expected_out_len[i]], (
                f"Failed for {i=}, {stride=}, {window_size=}, {out_mask[i]=}, {out_mask[i].shape=}"
            )
