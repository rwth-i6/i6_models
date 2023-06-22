from __future__ import annotations
from itertools import product

import torch
from torch import nn

from i6_models.parts.conformer.convolution import ConformerConvolutionV1, ConformerConvolutionV1Config
from i6_models.parts.conformer.feedforward import (
    ConformerPositionwiseFeedForwardV1,
    ConformerPositionwiseFeedForwardV1Config,
)
from i6_models.parts.frontend.vgg import (
    VGG4LayerActFrontendV1,
    VGG4LayerActFrontendV1Config,
    VGG4LayerPoolFrontendV1,
    VGG4LayerPoolFrontendV1Config,
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

    # without key padding mask
    input_shape = [3, 10, 20]  # B,T,F
    cfg = ConformerMHSAV1Config(20, 4, 0.1, 0.1)
    assert get_output_shape(input_shape, cfg) == [3, 10, 20]

    # with key padding mask
    input_shape = [4, 15, 32]  # B,T,F
    cfg = ConformerMHSAV1Config(32, 8, 0.2, 0.3)
    assert get_output_shape(input_shape, cfg, key_padding_mask=torch.randint(0, 2, input_shape[:2]) > 0) == [4, 15, 32]


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


def test_conformer_vgg_layer_pool_frontend_v1():
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
        conv2_stride,
        conv3_stride,
        pool_kernel_size,
        in_dim,
        out_dim,
    ):
        data_input = torch.randn(batch, time, features)
        data_input = torch.cat((data_input, torch.zeros(batch, time_padding, features)), dim=1)
        data_input = torch.cat((data_input, torch.randn(batch, time + time_padding, features)), dim=0)

        sequence_mask = torch.ones(batch, time)
        sequence_mask = torch.cat((sequence_mask, torch.zeros(batch, time_padding)), dim=1)
        sequence_mask = torch.cat((sequence_mask, torch.ones(batch, time + time_padding)), dim=0)

        cfg = VGG4LayerPoolFrontendV1Config(
            conv1_channels=conv1_channels,
            conv2_channels=conv2_channels,
            conv3_channels=conv3_channels,
            conv4_channels=conv4_channels,
            conv2_stride=conv2_stride,
            conv3_stride=conv3_stride,
            conv_kernel_size=(3, 3),
            conv_padding=None,
            pool_kernel_size=pool_kernel_size,
            pool_padding=None,
            activation=nn.functional.relu,
            linear_input_dim=in_dim,
            linear_output_dim=out_dim,
        )

        output, sequence_mask = VGG4LayerPoolFrontendV1(cfg)(data_input, sequence_mask)

        return output.shape, sequence_mask

    for test_inputs, test_outputs, mask_outputs in [
        [
            {
                "batch": 2,
                "time": 5,
                "time_padding": 5,
                "features": 40,
                "conv1_channels": 32,
                "conv2_channels": 32,
                "conv3_channels": 64,
                "conv4_channels": 64,
                "conv2_stride": (1, 1),
                "conv3_stride": (1, 1),
                "pool_kernel_size": (1, 32),
                "in_dim": None,
                "out_dim": None,
            },
            [4, 10, 64],
            torch.Tensor(
                [
                    5 * [True] + 5 * [False],
                    5 * [True] + 5 * [False],
                    10 * [True],
                    10 * [True],
                ]
            ),
        ],
        """
        [
            (10, 50, 50, 40, 32, 32, 64, 64, (1, 1), (1, 1), None, None),
            [20, 100, 2560],
            torch.Tensor(
                10
                * [
                    50 * [False] + 50 * [True],
                ]
                + 10
                * [
                    100 * [False],
                ]
            ),
        ],
        [
            (10, 50, 50, 40, 32, 32, 64, 64, (1, 1), (2, 1), None, None),
            [20, 50, 2560],
            torch.Tensor(
                10
                * [
                    25 * [False] + 25 * [True],
                ]
                + 10
                * [
                    50 * [False],
                ]
            ),
        ],
        [
            (10, 50, 50, 40, 32, 32, 64, 64, (1, 1), (3, 1), None, None),
            [20, 33, 2560],
            torch.Tensor(
                10
                * [
                    16 * [False] + 17 * [True],
                ]
                + 10
                * [
                    33 * [False],
                ]
            ),
        ],
        [
            (10, 50, 50, 40, 32, 32, 64, 64, (1, 1), (3, 1), 2560, 100),
            [20, 33, 100],
            torch.Tensor(
                10
                * [
                    16 * [False] + 17 * [True],
                ]
                + 10
                * [
                    33 * [False],
                ]
            ),
        ],
        [
            (10, 50, 50, 40, 32, 32, 64, 64, (1, 1), (3, 2), None, None),
            [20, 33, 1280],
            torch.Tensor(
                10
                * [
                    16 * [False] + 17 * [True],
                ]
                + 10
                * [
                    33 * [False],
                ]
            ),
        ],
        [
            (10, 50, 50, 40, 32, 32, 64, 64, (1, 1), (3, 4), None, None),
            [20, 33, 640],
            torch.Tensor(
                10
                * [
                    16 * [False] + 17 * [True],
                ]
                + 10
                * [
                    33 * [False],
                ]
            ),
        ],
        [
            (10, 50, 50, 40, 32, 32, 64, 64, (2, 1), (1, 4), None, None),
            [20, 50, 640],
            torch.Tensor(
                10
                * [
                    25 * [False] + 25 * [True],
                ]
                + 10
                * [
                    50 * [False],
                ]
            ),
        ],
        """,
    ]:
        shape, seq_mask = get_output_shape(**test_inputs)
        assert list(shape) == test_outputs
        assert torch.equal(seq_mask, mask_outputs), (seq_mask.shape, mask_outputs.shape)
