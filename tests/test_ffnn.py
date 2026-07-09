from itertools import product

import torch
from torch import nn
from torch.nn import functional as F

from i6_models.assemblies.ffnn import (
    FeedForwardEncoderV1,
    FeedForwardEncoderV1Config,
)

from i6_models.parts.frontend.window_convolution import WindowConvolutionFrontendV1Config, WindowConvolutionFrontendV1

from i6_models.config import ModelConfiguration, ModuleFactoryV1
from i6_models.parts.ffnn import FeedForwardLayerV1, FeedForwardLayerV1Config


def test_output_shape():
    input_dim = 80
    output_dim = 2048
    dropout = 0.1
    max_seq_lens = 100

    for window_size, stride in product(range(1, 22), range(1, 5)):
        frontend = ModuleFactoryV1(
            WindowConvolutionFrontendV1,
            WindowConvolutionFrontendV1Config(
                input_dim=80,
                output_dim=output_dim,
                kernel_size=window_size,
                dropout=dropout,
                stride=stride,
                activation=F.relu,
            ),
        )

        layer_cfg = FeedForwardLayerV1Config(
            input_dim=2048,
            output_dim=2048,
            dropout=0.1,
            activation=F.relu,
        )

        encoder_cfg = FeedForwardEncoderV1Config(num_layers=6, layer_cfg=layer_cfg, frontend=frontend)

        encoder = FeedForwardEncoderV1(encoder_cfg)

        feat_len = torch.arange(start=1, end=max_seq_lens + 1)
        mask = torch.less(torch.arange(max_seq_lens)[None, :], feat_len[:, None])

        features = torch.empty((max_seq_lens, max_seq_lens, input_dim))

        out, out_mask = encoder(features, mask)

        expected_out_len = (feat_len - 1) // stride + 1
        expected_shape = (max_seq_lens, expected_out_len[-1], output_dim)
        assert out.shape == expected_shape, f"Output with shape {out.shape} not as expected {expected_shape}"
        for i in range(expected_out_len[-1] - 1):
            # check if masks are correct
            assert out_mask[i, expected_out_len[i] - 1] and not out_mask[i, expected_out_len[i]], (
                f"Failed for {i=}, {stride=}, {window_size=}, {out_mask[i]=}, {out_mask[i].shape=}"
            )
