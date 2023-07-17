from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn

from i6_models.parts.frontend.common import IntTupleIntType
from i6_models.parts.frontend import (
    VGG4LayerActFrontendV1,
    VGG4LayerActFrontendV1Config,
)

# TODO rewrite tests
# TODO seq mask as input


@dataclass
class VGG4LayerActTestParams:
    batch: int
    time: int
    features: int
    conv1_channels: int
    conv2_channels: int
    conv3_channels: int
    conv4_channels: int
    pool1_kernel_size: IntTupleIntType
    pool1_stride: Optional[IntTupleIntType]
    pool2_kernel_size: IntTupleIntType
    pool2_stride: Optional[IntTupleIntType]
    out_features: Optional[int]
    output_shape: List[int]
    in_sequence_mask: torch.Tensor
    out_sequence_mask: torch.Tensor


def test_conformer_vgg_layer_act_frontend_v1():
    torch.manual_seed(42)

    def get_output_shape(test_parameters: VGG4LayerActTestParams):
        data_input = torch.randn(
            test_parameters.batch,
            test_parameters.time,
            test_parameters.features,
        )

        cfg = VGG4LayerActFrontendV1Config(
            in_features=1,
            conv1_channels=test_parameters.conv1_channels,
            conv2_channels=test_parameters.conv2_channels,
            conv3_channels=test_parameters.conv3_channels,
            conv4_channels=test_parameters.conv4_channels,
            conv_kernel_size=(3, 3),
            conv_padding=None,
            pool1_kernel_size=test_parameters.pool1_kernel_size,
            pool1_stride=test_parameters.pool1_stride,
            pool1_padding=None,
            pool2_kernel_size=test_parameters.pool2_kernel_size,
            pool2_stride=test_parameters.pool2_stride,
            pool2_padding=None,
            activation=nn.functional.relu,
            out_features=test_parameters.out_features,
        )

        output, sequence_mask = VGG4LayerActFrontendV1(cfg)(
            data_input,
            test_parameters.in_sequence_mask,
        )

        return output.shape, sequence_mask

    for idx, test_params in enumerate(
        [
            VGG4LayerActTestParams(
                batch=10,  # int
                time=50,  # int
                features=50,  # int
                conv1_channels=1,  # int
                conv2_channels=1,  # int
                conv3_channels=1,  # int
                conv4_channels=1,  # int
                pool1_kernel_size=(1, 1),  # IntTupleIntType
                pool1_stride=(1, 1),  # Optional[IntTupleIntType]
                pool2_kernel_size=(1, 1),  # IntTupleIntType
                pool2_stride=(1, 1),  # Optional[IntTupleIntType]
                out_features=50,  # int
                output_shape=[10, 50, 50],  # Tuple[int, int, int]
                in_sequence_mask=torch.Tensor(10 * [25 * [True] + 25 * [False]]),  # torch.Tensor
                out_sequence_mask=torch.Tensor(10 * [25 * [True] + 25 * [False]]),  # torch.Tensor
            ),
            VGG4LayerActTestParams(
                batch=10,  # int
                time=50,  # int
                features=50,  # int
                conv1_channels=1,  # int
                conv2_channels=1,  # int
                conv3_channels=1,  # int
                conv4_channels=1,  # int
                pool1_kernel_size=(2, 1),  # IntTupleIntType
                pool1_stride=(2, 1),  # Optional[IntTupleIntType]
                pool2_kernel_size=(1, 1),  # IntTupleIntType
                pool2_stride=(1, 1),  # Optional[IntTupleIntType]
                out_features=50,  # int
                output_shape=[10, 25, 50],  # Tuple[int, int, int]
                in_sequence_mask=torch.Tensor(10 * [25 * [True] + 25 * [False]]),  # torch.Tensor
                out_sequence_mask=torch.Tensor(10 * [13 * [True] + 12 * [False]]),  # torch.Tensor
            ),
            VGG4LayerActTestParams(
                batch=10,  # int
                time=50,  # int
                features=50,  # int
                conv1_channels=1,  # int
                conv2_channels=1,  # int
                conv3_channels=1,  # int
                conv4_channels=1,  # int
                pool1_kernel_size=(1, 1),  # IntTupleIntType
                pool1_stride=(1, 1),  # Optional[IntTupleIntType]
                pool2_kernel_size=(2, 1),  # IntTupleIntType
                pool2_stride=(2, 1),  # Optional[IntTupleIntType]
                out_features=50,  # int
                output_shape=[10, 25, 50],  # Tuple[int, int, int]
                in_sequence_mask=torch.Tensor(10 * [25 * [True] + 25 * [False]]),  # torch.Tensor
                out_sequence_mask=torch.Tensor(10 * [13 * [True] + 12 * [False]]),  # torch.Tensor
            ),
            VGG4LayerActTestParams(
                batch=10,  # int
                time=50,  # int
                features=50,  # int
                conv1_channels=1,  # int
                conv2_channels=1,  # int
                conv3_channels=1,  # int
                conv4_channels=1,  # int
                pool1_kernel_size=(2, 1),  # IntTupleIntType
                pool1_stride=(2, 1),  # Optional[IntTupleIntType]
                pool2_kernel_size=(2, 1),  # IntTupleIntType
                pool2_stride=(2, 1),  # Optional[IntTupleIntType]
                out_features=50,  # int
                output_shape=[10, 12, 50],  # Tuple[int, int, int]
                in_sequence_mask=torch.Tensor(10 * [25 * [True] + 25 * [False]]),  # torch.Tensor
                out_sequence_mask=torch.Tensor(10 * [7 * [True] + 5 * [False]]),  # torch.Tensor
            ),
            VGG4LayerActTestParams(
                batch=10,  # int
                time=50,  # int
                features=50,  # int
                conv1_channels=2,  # int
                conv2_channels=2,  # int
                conv3_channels=2,  # int
                conv4_channels=2,  # int
                pool1_kernel_size=(1, 1),  # IntTupleIntType
                pool1_stride=(1, 1),  # Optional[IntTupleIntType]
                pool2_kernel_size=(1, 1),  # IntTupleIntType
                pool2_stride=(1, 1),  # Optional[IntTupleIntType]
                out_features=100,  # int
                output_shape=[10, 50, 100],  # Tuple[int, int, int]
                in_sequence_mask=torch.Tensor(10 * [25 * [True] + 25 * [False]]),  # torch.Tensor
                out_sequence_mask=torch.Tensor(10 * [25 * [True] + 25 * [False]]),  # torch.Tensor
            ),
            VGG4LayerActTestParams(
                batch=10,  # int
                time=50,  # int
                features=50,  # int
                conv1_channels=3,  # int
                conv2_channels=3,  # int
                conv3_channels=3,  # int
                conv4_channels=3,  # int
                pool1_kernel_size=(1, 1),  # IntTupleIntType
                pool1_stride=(1, 1),  # Optional[IntTupleIntType]
                pool2_kernel_size=(1, 1),  # IntTupleIntType
                pool2_stride=(1, 1),  # Optional[IntTupleIntType]
                out_features=150,  # int
                output_shape=[10, 50, 150],  # Tuple[int, int, int]
                in_sequence_mask=torch.Tensor(10 * [25 * [True] + 25 * [False]]),  # torch.Tensor
                out_sequence_mask=torch.Tensor(10 * [25 * [True] + 25 * [False]]),  # torch.Tensor
            ),
            VGG4LayerActTestParams(
                batch=10,  # int
                time=50,  # int
                features=50,  # int
                conv1_channels=4,  # int
                conv2_channels=4,  # int
                conv3_channels=4,  # int
                conv4_channels=4,  # int
                pool1_kernel_size=(1, 1),  # IntTupleIntType
                pool1_stride=(1, 1),  # Optional[IntTupleIntType]
                pool2_kernel_size=(1, 1),  # IntTupleIntType
                pool2_stride=(1, 1),  # Optional[IntTupleIntType]
                out_features=200,  # int
                output_shape=[10, 50, 200],  # Tuple[int, int, int]
                in_sequence_mask=torch.Tensor(10 * [25 * [True] + 25 * [False]]),  # torch.Tensor
                out_sequence_mask=torch.Tensor(10 * [25 * [True] + 25 * [False]]),  # torch.Tensor
            ),
            VGG4LayerActTestParams(
                batch=10,  # int
                time=50,  # int
                features=50,  # int
                conv1_channels=32,  # int
                conv2_channels=32,  # int
                conv3_channels=64,  # int
                conv4_channels=64,  # int
                pool1_kernel_size=(1, 1),  # IntTupleIntType
                pool1_stride=(1, 1),  # Optional[IntTupleIntType]
                pool2_kernel_size=(1, 1),  # IntTupleIntType
                pool2_stride=(1, 1),  # Optional[IntTupleIntType]
                out_features=3200,  # int
                output_shape=[10, 50, 3200],  # Tuple[int, int, int]
                in_sequence_mask=torch.Tensor(10 * [25 * [True] + 25 * [False]]),  # torch.Tensor
                out_sequence_mask=torch.Tensor(10 * [25 * [True] + 25 * [False]]),  # torch.Tensor
            ),
            VGG4LayerActTestParams(
                batch=10,  # int
                time=50,  # int
                features=50,  # int
                conv1_channels=32,  # int
                conv2_channels=32,  # int
                conv3_channels=64,  # int
                conv4_channels=64,  # int
                pool1_kernel_size=(1, 1),  # IntTupleIntType
                pool1_stride=(1, 1),  # Optional[IntTupleIntType]
                pool2_kernel_size=(1, 1),  # IntTupleIntType
                pool2_stride=(1, 1),  # Optional[IntTupleIntType]
                out_features=50,  # int
                output_shape=[10, 50, 50],  # Tuple[int, int, int]
                in_sequence_mask=torch.Tensor(10 * [25 * [True] + 25 * [False]]),  # torch.Tensor
                out_sequence_mask=torch.Tensor(10 * [25 * [True] + 25 * [False]]),  # torch.Tensor
            ),
            VGG4LayerActTestParams(
                batch=10,  # int
                time=50,  # int
                features=60,  # int
                conv1_channels=4,  # int
                conv2_channels=4,  # int
                conv3_channels=4,  # int
                conv4_channels=4,  # int
                pool1_kernel_size=(1, 2),  # IntTupleIntType
                pool1_stride=(1, 2),  # Optional[IntTupleIntType]
                pool2_kernel_size=(1, 2),  # IntTupleIntType
                pool2_stride=(1, 2),  # Optional[IntTupleIntType]
                out_features=None,  # int
                output_shape=[10, 50, 60],  # Tuple[int, int, int]
                in_sequence_mask=torch.Tensor(10 * [25 * [True] + 25 * [False]]),  # torch.Tensor
                out_sequence_mask=torch.Tensor(10 * [25 * [True] + 25 * [False]]),  # torch.Tensor
            ),
            VGG4LayerActTestParams(
                batch=10,  # int
                time=50,  # int
                features=50,  # int
                conv1_channels=3,  # int
                conv2_channels=3,  # int
                conv3_channels=3,  # int
                conv4_channels=3,  # int
                pool1_kernel_size=(2, 3),  # IntTupleIntType
                pool1_stride=(2, 3),  # Optional[IntTupleIntType]
                pool2_kernel_size=(1, 2),  # IntTupleIntType
                pool2_stride=(1, 2),  # Optional[IntTupleIntType]
                out_features=None,  # int
                output_shape=[10, 25, 24],  # Tuple[int, int, int]
                in_sequence_mask=torch.Tensor(10 * [25 * [True] + 25 * [False]]),  # torch.Tensor
                out_sequence_mask=torch.Tensor(10 * [13 * [True] + 12 * [False]]),  # torch.Tensor
            ),
            VGG4LayerActTestParams(
                batch=10,  # int
                time=50,  # int
                features=70,  # int
                conv1_channels=4,  # int
                conv2_channels=4,  # int
                conv3_channels=4,  # int
                conv4_channels=4,  # int
                pool1_kernel_size=(1, 2),  # IntTupleIntType
                pool1_stride=(1, 2),  # Optional[IntTupleIntType]
                pool2_kernel_size=(1, 2),  # IntTupleIntType
                pool2_stride=(1, 2),  # Optional[IntTupleIntType]
                out_features=None,  # int
                output_shape=[10, 50, 68],  # Tuple[int, int, int]
                in_sequence_mask=torch.Tensor(10 * [25 * [True] + 25 * [False]]),  # torch.Tensor
                out_sequence_mask=torch.Tensor(10 * [25 * [True] + 25 * [False]]),  # torch.Tensor
            ),
            VGG4LayerActTestParams(
                batch=10,  # int
                time=50,  # int
                features=50,  # int
                conv1_channels=32,  # int
                conv2_channels=32,  # int
                conv3_channels=64,  # int
                conv4_channels=64,  # int
                pool1_kernel_size=(4, 3),  # IntTupleIntType
                pool1_stride=(4, 3),  # Optional[IntTupleIntType]
                pool2_kernel_size=(1, 3),  # IntTupleIntType
                pool2_stride=(1, 3),  # Optional[IntTupleIntType]
                out_features=None,  # int
                output_shape=[10, 12, 320],  # Tuple[int, int, int]
                in_sequence_mask=torch.Tensor(10 * [25 * [True] + 25 * [False]]),  # torch.Tensor
                out_sequence_mask=torch.Tensor(10 * [7 * [True] + 5 * [False]]),  # torch.Tensor
            ),
        ]
    ):
        shape, seq_mask = get_output_shape(test_params)
        assert list(shape) == test_params.output_shape, (type(shape), type(test_params.output_shape))
        assert torch.equal(seq_mask, test_params.out_sequence_mask), (
            seq_mask.shape,
            test_params.out_sequence_mask.shape,
        )