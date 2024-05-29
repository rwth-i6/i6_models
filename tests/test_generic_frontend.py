from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Sequence, Union, Callable, List

import torch
from torch import nn

from i6_models.parts.frontend.generic_frontend import FrontendLayerType, GenericFrontendV1, GenericFrontendV1Config


@dataclass
class GenericFrontendV1TestParams:
    batch: int
    time: int
    in_features: int
    layer_ordering: Sequence[FrontendLayerType]
    conv_kernel_sizes: Optional[Sequence[Tuple[int, int]]]
    conv_strides: Optional[Sequence[Tuple[int, int]]]
    conv_paddings: Optional[Sequence[Tuple[int, int]]]
    conv_out_dims: Optional[Sequence[int]]
    pool_kernel_sizes: Optional[Sequence[Tuple[int, int]]]
    pool_strides: Optional[Sequence[Tuple[int, int]]]
    pool_paddings: Optional[Sequence[Tuple[int, int]]]
    activations: Optional[Sequence[Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]]]
    out_features: int
    output_shape: List[int]
    in_sequence_mask: torch.Tensor
    out_sequence_mask: torch.Tensor


def test_generic_frontend_v1():
    torch.manual_seed(42)

    def get_output_shape(test_parameters: GenericFrontendV1TestParams):
        data_input = torch.randn(
            test_parameters.batch,
            test_parameters.time,
            test_parameters.in_features,
        )

        cfg = GenericFrontendV1Config(
            in_features=test_parameters.in_features,
            layer_ordering=test_parameters.layer_ordering,
            conv_kernel_sizes=test_parameters.conv_kernel_sizes,
            conv_strides=test_parameters.conv_strides,
            conv_paddings=test_parameters.conv_paddings,
            conv_out_dims=test_parameters.conv_out_dims,
            pool_kernel_sizes=test_parameters.pool_kernel_sizes,
            pool_strides=test_parameters.pool_strides,
            pool_paddings=test_parameters.pool_paddings,
            activations=test_parameters.activations,
            out_features=test_parameters.out_features,
        )

        output, sequence_mask = GenericFrontendV1(cfg)(
            data_input,
            test_parameters.in_sequence_mask,
        )

        return output.shape, sequence_mask

    for idx, test_params in enumerate(
        [
            GenericFrontendV1TestParams(
                batch=10,
                time=50,
                in_features=50,
                layer_ordering=[FrontendLayerType.Conv2d, FrontendLayerType.Conv2d],
                conv_kernel_sizes=[(3, 3), (5, 5)],
                conv_strides=[(1, 1), (1, 1)],
                conv_paddings=None,
                conv_out_dims=[32, 32],
                pool_kernel_sizes=None,
                pool_strides=None,
                pool_paddings=None,
                activations=None,
                out_features=384,
                output_shape=[10, 50, 384],
                in_sequence_mask=torch.Tensor(10 * [25 * [True] + 25 * [False]]).bool(),
                out_sequence_mask=torch.Tensor(10 * [25 * [True] + 25 * [False]]).bool(),
            ),
            GenericFrontendV1TestParams(
                batch=10,
                time=50,
                in_features=50,
                layer_ordering=[
                    FrontendLayerType.Conv2d,
                    FrontendLayerType.Pool2d,
                    FrontendLayerType.Conv2d,
                    FrontendLayerType.Activation,
                ],
                conv_kernel_sizes=[(3, 3), (5, 5)],
                conv_strides=[(1, 1), (1, 1)],
                conv_paddings=None,
                conv_out_dims=[32, 32],
                pool_kernel_sizes=[(3, 3)],
                pool_strides=[(2, 2)],
                pool_paddings=None,
                activations=[nn.SiLU()],
                out_features=384,
                output_shape=[10, 25, 384],
                in_sequence_mask=torch.Tensor(10 * [50 * [True] + 0 * [False]]).bool(),
                out_sequence_mask=torch.Tensor(10 * [25 * [True] + 0 * [False]]).bool(),
            ),
            GenericFrontendV1TestParams(
                batch=10,
                time=50,
                in_features=50,
                layer_ordering=[
                    FrontendLayerType.Conv2d,
                    FrontendLayerType.Pool2d,
                    FrontendLayerType.Activation,
                    FrontendLayerType.Conv2d,
                    FrontendLayerType.Activation,
                ],
                conv_kernel_sizes=[(3, 3), (3, 5)],
                conv_strides=[(1, 1), (2, 1)],
                conv_paddings=None,
                conv_out_dims=[32, 32],
                pool_kernel_sizes=[(3, 3)],
                pool_strides=[(2, 2)],
                pool_paddings=None,
                activations=[nn.SiLU(), nn.SiLU()],
                out_features=384,
                output_shape=[10, 13, 384],
                in_sequence_mask=torch.Tensor(10 * [50 * [True] + 0 * [False]]).bool(),
                out_sequence_mask=torch.Tensor(10 * [13 * [True] + 0 * [False]]).bool(),
            ),
            GenericFrontendV1TestParams(
                batch=10,
                time=50,
                in_features=50,
                layer_ordering=[
                    FrontendLayerType.Conv2d,
                    FrontendLayerType.Pool2d,
                    FrontendLayerType.Activation,
                    FrontendLayerType.Conv2d,
                    FrontendLayerType.Pool2d,
                    FrontendLayerType.Activation,
                ],
                conv_kernel_sizes=[(3, 3), (3, 5)],
                conv_strides=[(1, 1), (2, 2)],
                conv_paddings=None,
                conv_out_dims=[32, 32],
                pool_kernel_sizes=[(3, 3), (5, 5)],
                pool_strides=[(1, 2), (2, 3)],
                pool_paddings=None,
                activations=[nn.SiLU(), nn.SiLU()],
                out_features=384,
                output_shape=[10, 13, 384],
                in_sequence_mask=torch.Tensor(10 * [50 * [True] + 0 * [False]]).bool(),
                out_sequence_mask=torch.Tensor(10 * [13 * [True] + 0 * [False]]).bool(),
            ),
        ]
    ):
        shape, seq_mask = get_output_shape(test_params)
        assert list(shape) == test_params.output_shape, (shape, test_params.output_shape)
        assert torch.equal(seq_mask, test_params.out_sequence_mask), (
            seq_mask.shape,
            test_params.out_sequence_mask.shape,
        )
