from __future__ import annotations

__all__ = [
    "FrontendLayerType",
    "GenericFrontendV1Config",
    "GenericFrontendV1",
]

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Optional, Tuple, Union, Sequence

import torch
from torch import nn

from i6_models.config import ModelConfiguration

from i6_models.parts.frontend.common import get_same_padding, mask_pool, calculate_output_dim


class FrontendLayerType(Enum):
    Conv2d = auto()
    Pool2d = auto()
    Activation = auto()


@dataclass
class GenericFrontendV1Config(ModelConfiguration):
    """
    Attributes:
        in_features: number of input features to module
        layer_ordering: the ordering of the front end layer sequences, the ordering element must be selected from FrontendLayerType
            e.g. the ordering of VGG4LayerActFrontendV1 would be [FrontendLayerType.Conv2d, FrontendLayerType.Activation,
            FrontendLayerType.Pool2d, FrontendLayerType.Conv2d, FrontendLayerType.Conv2d, FrontendLayerType.Activation,
            FrontendLayerType.Pool2d]
        conv_kernel_sizes: kernel sizes for each conv layer
        conv_strides: stride sizes for each conv layer
        conv_paddings: paddings sizes for each conv layer
        conv_out_dims: number of out channels for each conv layer
        pool_kernel_sizes: kernel sizes for each pool layer
        pool_strides: stride sizes for each pool layer
        pool_paddings: padding sizes for each pool layer
        activations: activation functions
        out_features: output size of the final linear layer
    """

    in_features: int
    layer_ordering: Sequence[FrontendLayerType]
    conv_kernel_sizes: Optional[Sequence[Union[int, Tuple[int, int]]]]
    conv_strides: Optional[Sequence[Union[int, Tuple[int, int]]]]
    conv_paddings: Optional[Sequence[Union[int, Tuple[int, int]]]]
    conv_out_dims: Optional[Sequence[Union[int, Tuple[int, int]]]]
    pool_kernel_sizes: Optional[Sequence[Union[int, Tuple[int, int]]]]
    pool_strides: Optional[Sequence[Union[int, Tuple[int, int]]]]
    pool_paddings: Optional[Sequence[Union[int, Tuple[int, int]]]]
    activations: Optional[Sequence[Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]]]
    out_features: int

    def check_valid(self):
        num_convs = 0 if self.conv_kernel_sizes is None else len(self.conv_kernel_sizes)
        num_pools = 0 if self.pool_kernel_sizes is None else len(self.pool_kernel_sizes)
        num_activations = 0 if self.activations is None else len(self.activations)

        assert num_convs == self.layer_ordering.count(
            FrontendLayerType.Conv2d
        ), "Number of convolution layers mismatch!"
        assert num_activations == self.layer_ordering.count(
            FrontendLayerType.Activation
        ), "Number of activation layers mismatch!"
        assert num_pools == self.layer_ordering.count(FrontendLayerType.Pool2d), "Number of pooling layers mismatch!"

        if self.conv_strides is not None:
            assert len(self.conv_strides) == num_convs, "Please specify stride for each convolution layer!"
        if self.conv_paddings is not None:
            assert len(self.conv_paddings) == num_convs, "Please specify padding for each convolution layer!"
        if num_convs != 0:
            assert (
                len(self.conv_out_dims) == num_convs
            ), "Please specify the number of channels for each convolution layer!"

        if self.pool_strides is not None:
            assert len(self.pool_strides) == num_pools, "Please specify stride for each pooling layer!"
        if self.conv_paddings is not None:
            assert len(self.pool_paddings) == num_pools, "Please specify padding for each pooling layer!"

        assert len(self.layer_ordering) == num_convs + num_pools + num_activations, "Number of total layers mismatch!"

        for kernel_sizes in [self.conv_kernel_sizes, self.pool_kernel_sizes]:
            if kernel_sizes is not None:
                for kernel_size in kernel_sizes:
                    if isinstance(kernel_size, int):
                        assert kernel_size % 2 == 1, "ConformerVGGFrontendV1 only supports odd kernel sizes"
                    elif isinstance(kernel_size, tuple):
                        for i in range(len(kernel_size)):
                            assert kernel_size[i] % 2 == 1, "ConformerVGGFrontendV1 only supports odd kernel sizes"

    def __post__init__(self):
        super().__post_init__()
        self.check_valid()


class GenericFrontendV1(nn.Module):
    def __init__(self, model_cfg: GenericFrontendV1Config):
        """
        Generic Front-End
        can be used to generate customized frontend by combining convolutional and pooling layers, as well as activation
        functions differently

        To get the ESPnet case, for example Conv2dSubsampling6, use these options
            layer_ordering = [FrontendLayerType.Conv2d, FrontendLayerType.Conv2d]
            conv_kernel_sizes = [3, 5]
            strides = [2, 3]

        To get the i6_models VGG4LayerActFrontendV1, use the options:
            layer_ordering = [FrontendLayerType.Conv2d, FrontendLayerType.Activation, FrontendLayerType.Pool2d,
                FrontendLayerType.Conv2d, FrontendLayerType.Conv2d, FrontendLayerType.Activation, FrontendLayerType.Pool2d]
            conv_kernel_sizes = [3, 3, 3]
            conv_out_dims = [32, 34, 64]
            pool_kernel_sizes = [3, 3]
            pool_strides = [2, 2]
            activations = [torch.nn.ReLU(), torch.nn.ReLU()]
        """
        super().__init__()

        model_cfg.check_valid()

        self.cfg = model_cfg

        self.frontend_layers = nn.ModuleList([])

        conv_layer_index = 0
        pool_layer_index = 0
        activation_layer_index = 0
        last_channel_dim = 1
        last_feat_dim = model_cfg.in_features
        for layer_type in model_cfg.layer_ordering:
            if layer_type == FrontendLayerType.Conv2d:
                conv_out_dim = model_cfg.conv_out_dims[conv_layer_index]
                conv_kernel_size = model_cfg.conv_kernel_sizes[conv_layer_index]
                conv_stride = 1 if model_cfg.conv_strides is None else model_cfg.conv_strides[conv_layer_index]
                conv_padding = (
                    get_same_padding(conv_kernel_size)
                    if model_cfg.conv_paddings is None
                    else model_cfg.conv_paddings[conv_layer_index]
                )

                self.frontend_layers.append(
                    nn.Conv2d(
                        in_channels=last_channel_dim,
                        out_channels=conv_out_dim,
                        kernel_size=conv_kernel_size,
                        stride=conv_stride,
                        padding=conv_padding,
                    )
                )

                last_channel_dim = conv_out_dim
                last_feat_dim = calculate_output_dim(
                    in_dim=last_feat_dim,
                    filter_size=conv_kernel_size if isinstance(conv_kernel_size, int) else conv_kernel_size[1],
                    stride=conv_stride if isinstance(conv_stride, int) else conv_stride[1],
                    padding=conv_padding if isinstance(conv_padding, int) else conv_padding[1],
                )
                conv_layer_index += 1

            elif layer_type == FrontendLayerType.Pool2d:
                pool_stride = 1 if model_cfg.pool_strides is None else model_cfg.pool_strides[pool_layer_index]
                pool_kernel_size = model_cfg.pool_kernel_sizes[pool_layer_index]
                pool_padding = (
                    get_same_padding(pool_kernel_size)
                    if model_cfg.pool_paddings is None
                    else model_cfg.pool_paddings[pool_layer_index]
                )

                self.frontend_layers.append(
                    nn.MaxPool2d(
                        kernel_size=pool_kernel_size,
                        stride=pool_stride,
                        padding=pool_padding,
                    )
                )
                last_feat_dim = calculate_output_dim(
                    in_dim=last_feat_dim,
                    filter_size=pool_kernel_size if isinstance(pool_kernel_size, int) else pool_kernel_size[1],
                    stride=pool_stride if isinstance(pool_stride, int) else pool_stride[1],
                    padding=pool_padding if isinstance(pool_padding, int) else pool_padding[1],
                )
                pool_layer_index += 1

            elif layer_type == FrontendLayerType.Activation:
                self.frontend_layers.append(model_cfg.activations[activation_layer_index])
                activation_layer_index += 1
            else:
                raise NotImplementedError

            self.linear = nn.Linear(
                in_features=last_feat_dim * last_channel_dim,
                out_features=model_cfg.out_features,
                bias=True,
            )

    def forward(self, tensor: torch.Tensor, sequence_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert tensor.shape[-1] == self.cfg.in_features
        # and add a dim
        tensor = tensor[:, None, :, :]  # [B,C=1,T,F]

        for i in range(len(self.cfg.layer_ordering)):
            layer = self.frontend_layers[i]
            tensor = layer(tensor)

            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.MaxPool2d):
                sequence_mask = mask_pool(
                    sequence_mask,
                    kernel_size=layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0],
                    stride=layer.stride if isinstance(layer.stride, int) else layer.stride[0],
                    padding=layer.padding if isinstance(layer.padding, int) else layer.padding[0],
                )

        tensor = torch.transpose(tensor, 1, 2)  # transpose to [B,T",C,F"]
        tensor = torch.flatten(tensor, start_dim=2, end_dim=-1)  # [B,T",C*F"]

        tensor = self.linear(tensor)

        return tensor, sequence_mask