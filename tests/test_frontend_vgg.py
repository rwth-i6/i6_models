from __future__ import annotations

import torch
from torch import nn

from i6_models.parts.frontend import (
    VGG4LayerActFrontendV1,
    VGG4LayerActFrontendV1Config,
)


def test_conformer_vgg_layer_act_frontend_v1():
    torch.manual_seed(42)

    def get_output_shape(
        batch,
        time,
        time_padding,
        features,
        conv1_channels,
        conv2_channels,
        conv3_channels,
        conv4_channels,
        pool1_red,
        pool2_red,
        in_dim,
        out_dim,
    ):
        data_input = torch.randn(batch, time, features)
        data_input = torch.cat((data_input, torch.zeros(batch, time_padding, features)), dim=1)
        data_input = torch.cat((data_input, torch.randn(batch, time + time_padding, features)), dim=0)

        sequence_mask = torch.ones(batch, time)
        sequence_mask = torch.cat((sequence_mask, torch.zeros(batch, time_padding)), dim=1)
        sequence_mask = torch.cat((sequence_mask, torch.ones(batch, time + time_padding)), dim=0)
        sequence_mask = sequence_mask.bool()

        cfg = VGG4LayerActFrontendV1Config(
            conv1_channels=conv1_channels,
            conv2_channels=conv2_channels,
            conv3_channels=conv3_channels,
            conv4_channels=conv4_channels,
            conv_kernel_size=(3, 3),
            conv_padding=None,
            pool1_kernel_size=pool1_red,
            pool1_stride=pool1_red,
            pool1_padding=None,
            pool2_kernel_size=pool2_red,
            pool2_stride=pool2_red,
            pool2_padding=None,
            activation=nn.functional.relu,
            linear_input_dim=in_dim,
            linear_output_dim=out_dim,
        )

        output, sequence_mask = VGG4LayerActFrontendV1(cfg)(data_input, sequence_mask)

        return output.shape, sequence_mask

    for test_inputs, test_outputs, mask_outputs in [
        [
            (10, 50, 50, 50, 1, 1, 1, 1, (1, 1), (1, 1), 50, 50),
            [20, 100, 50],
            torch.Tensor(10 * [50 * [True] + 50 * [False]] + 10 * [100 * [True]]),
        ],
        [
            (10, 50, 50, 50, 1, 1, 1, 1, (2, 1), (1, 1), 50, 50),
            [20, 50, 50],
            torch.Tensor(10 * [25 * [True] + 25 * [False]] + 10 * [50 * [True]]),
        ],
        [
            (10, 50, 50, 50, 1, 1, 1, 1, (1, 1), (2, 1), 50, 50),
            [20, 50, 50],
            torch.Tensor(10 * [25 * [True] + 25 * [False]] + 10 * [50 * [True]]),
        ],
        [
            (10, 50, 50, 50, 1, 1, 1, 1, (2, 1), (2, 1), 50, 50),
            [20, 25, 50],
            torch.Tensor(10 * [13 * [True] + 12 * [False]] + 10 * [25 * [True]]),
        ],
        [
            (10, 50, 50, 50, 2, 2, 2, 2, (1, 1), (1, 1), 100, 100),
            [20, 100, 100],
            torch.Tensor(10 * [50 * [True] + 50 * [False]] + 10 * [100 * [True]]),
        ],
        [
            (10, 50, 50, 50, 3, 3, 3, 3, (1, 1), (1, 1), 150, 150),
            [20, 100, 150],
            torch.Tensor(10 * [50 * [True] + 50 * [False]] + 10 * [100 * [True]]),
        ],
        [
            (10, 50, 50, 50, 4, 4, 4, 4, (1, 1), (1, 1), 200, 200),
            [20, 100, 200],
            torch.Tensor(10 * [50 * [True] + 50 * [False]] + 10 * [100 * [True]]),
        ],
        [
            (10, 50, 50, 50, 32, 32, 64, 64, (1, 1), (1, 1), 3200, 3200),
            [20, 100, 3200],
            torch.Tensor(10 * [50 * [True] + 50 * [False]] + 10 * [100 * [True]]),
        ],
        [
            (10, 50, 50, 50, 32, 32, 64, 64, (1, 1), (1, 1), 3200, 50),
            [20, 100, 50],
            torch.Tensor(10 * [50 * [True] + 50 * [False]] + 10 * [100 * [True]]),
        ],
        [
            (10, 50, 50, 60, 4, 4, 4, 4, (1, 2), (1, 2), None, None),
            [20, 100, 60],
            torch.Tensor(10 * [50 * [True] + 50 * [False]] + 10 * [100 * [True]]),
        ],
        [
            (10, 50, 50, 50, 3, 3, 3, 3, (2, 3), (1, 1), None, None),
            [20, 50, 48],
            torch.Tensor(10 * [25 * [True] + 25 * [False]] + 10 * [50 * [True]]),
        ],
        [
            (10, 50, 50, 70, 4, 4, 4, 4, (1, 2), (1, 2), None, None),
            [20, 100, 68],
            torch.Tensor(10 * [50 * [True] + 50 * [False]] + 10 * [100 * [True]]),
        ],
        [
            (10, 50, 50, 50, 32, 32, 64, 64, (4, 3), (1, 3), None, None),
            [20, 25, 320],
            torch.Tensor(10 * [13 * [True] + 12 * [False]] + 10 * [25 * [True]]),
        ],
    ]:
        shape, seq_mask = get_output_shape(*test_inputs)
        assert list(shape) == test_outputs
        assert torch.equal(seq_mask, mask_outputs), (seq_mask.shape, mask_outputs.shape)
