from itertools import product

import torch
from torch import nn
from torch.nn import functional as F

import sys

sys.path.insert(0, "/home/dmann/setups/2024-05-06--test-ffnn-fullsum/recipe/i6_models")

from i6_models.parts.frontend.window_ffnn import WindowFeedForwardFrontendV1Config, WindowFeedForwardFrontendV1


def test_output_shape():
    in_features = 80
    out_features = 2048
    dropout = 0.1
    max_seq_lens = 100

    # skip even window sizes for now
    for window_size, stride in product(range(1, 22, 2), range(1, 5)):
        frontend = WindowFeedForwardFrontendV1(
            WindowFeedForwardFrontendV1Config(
                in_features=80,
                out_features=out_features,
                window_size=window_size,
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
            assert (
                out_mask[i, expected_out_len[i] - 1] and not out_mask[i, expected_out_len[i]]
            ), f"Failed for {i=}, {stride=}, {window_size=}, {out_mask[i]=}, {out_mask[i].shape=}"
