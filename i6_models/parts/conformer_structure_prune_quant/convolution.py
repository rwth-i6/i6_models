from __future__ import annotations

__all__ = ["ConformerConvolutionV1", "ConformerConvolutionV1Config"]

from dataclasses import dataclass
from copy import deepcopy
import math

import torch
from torch.nn import init
from torch import nn
from i6_models.config import ModelConfiguration
from typing import Callable, Union, Optional
import torch.nn.functional as F

from .weight_quantizer import WeightQuantizer, ActivationQuantizer



@dataclass
class ConformerConvolutionV1Config(ModelConfiguration):
    """
    Attributes:
        channels: number of channels for conv layers
        kernel_size: kernel size of conv layers
        dropout: dropout probability
        activation: activation function applied after normalization
        norm: normalization layer with input of shape [N,C,T]
    """

    channels: int
    kernel_size: int
    dropout: float
    activation: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]
    norm: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]
    weight_bit_prec: Union[int, float]
    activation_bit_prec: Union[int, float]
    weight_quant_dtype: torch.dtype
    weight_quant_method: str
    activation_quant_dtype: torch.dtype
    activation_quant_method: str
    moving_average: Optional[float]  # Moving average for input quantization
    quantize_bias: Optional[str]
    observer_only_in_train: bool

    def check_valid(self):
        assert self.kernel_size % 2 == 1, "ConformerConvolutionV1 only supports odd kernel sizes"

    def __post_init__(self):
        super().__post_init__()
        self.check_valid()


class LinearQuant(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_bit_prec: int,
        weight_quant_dtype: torch.dtype,
        weight_quant_method: str,
        bias: bool,
        quantize_bias: Optional[str],
        observer_only_in_train: bool,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features)), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.empty((out_features,)), requires_grad=True)

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

        self.weight_bit_prec = weight_bit_prec
        self.weight_quant_dtype = weight_quant_dtype
        self.weight_quant_method = weight_quant_method
        self.weight_quantizer = WeightQuantizer(
            bit_precision=self.weight_bit_prec,
            dtype=self.weight_quant_dtype,
            method=self.weight_quant_method,
            observer_only_in_train=observer_only_in_train,
        )
        self.quantize_bias = quantize_bias
        if self.quantize_bias == "weight" or self.quantize_bias == True:
            self.bias_quantizer = WeightQuantizer(
                bit_precision=self.weight_bit_prec,
                dtype=self.weight_quant_dtype,
                method=self.weight_quant_method,
                observer_only_in_train=observer_only_in_train,
            )
        elif self.quantize_bias == "act":
            assert False
            self.bias_quantizer = WeightQuantizer(
                bit_precision=self.weight_bit_prec,
                dtype=self.weight_quant_dtype,
                method=self.weight_quant_method,
            )

    def forward(self, tensor: torch.Tensor):
        if self.quantize_bias is not None:
            bias = self.bias_quantizer(self.bias)
        else:
            bias = self.bias
        lin = F.linear(tensor, self.weight_quantizer(self.weight), bias)
        return lin


class Conv1dQuant(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        weight_bit_prec: int,
        weight_quant_dtype: torch.dtype,
        weight_quant_method: str,
        bias: bool,
        quantize_bias: Optional[str],
        stride: int,
        padding: Union[str, int],
        dilation: int,
        groups: int,
        observer_only_in_train: bool,
        padding_mode: str = "zeros",  # TODO: refine this type
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        self.weight = nn.Parameter(torch.empty(in_channels, out_channels // groups, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

        self.weight_bit_prec = weight_bit_prec
        self.weight_quant_dtype = weight_quant_dtype
        self.weight_quant_method = weight_quant_method
        self.weight_quantizer = WeightQuantizer(
            bit_precision=self.weight_bit_prec,
            dtype=self.weight_quant_dtype,
            method=self.weight_quant_method,
            observer_only_in_train=observer_only_in_train,
        )
        self.quantize_bias = quantize_bias
        if self.quantize_bias == "weight" or self.quantize_bias == True:
            self.bias_quantizer = WeightQuantizer(
                bit_precision=self.weight_bit_prec,
                dtype=self.weight_quant_dtype,
                method=self.weight_quant_method,
                observer_only_in_train=observer_only_in_train,
            )
        elif self.quantize_bias == "act":
            self.bias_quantizer = WeightQuantizer(
                bit_precision=self.weight_bit_prec,
                dtype=self.weight_quant_dtype,
                method=self.weight_quant_method,
                observer_only_in_train=observer_only_in_train,
            )

    def forward(self, tensor: torch.Tensor):
        if self.quantize_bias is True:
            bias = self.bias_quantizer(self.bias)
        else:
            bias = self.bias
        result = F.conv1d(
            tensor, self.weight_quantizer(self.weight), bias, self.stride, self.padding, self.dilation, self.groups
        )
        return result
    

class ConformerConvolutionV1(nn.Module):
    """
    Conformer convolution module.
    see also: https://github.com/espnet/espnet/blob/713e784c0815ebba2053131307db5f00af5159ea/espnet/nets/pytorch_backend/conformer/convolution.py#L13

    Uses explicit padding for ONNX exportability, see:
    https://github.com/pytorch/pytorch/issues/68880
    """

    def __init__(self, model_cfg: ConformerConvolutionQuantV4Config):
        """
        :param model_cfg: model configuration for this module
        """
        super().__init__()
        model_cfg.check_valid()
        self.model_cfg = model_cfg
        self.pointwise_conv1 = LinearQuant(
            in_features=model_cfg.channels,
            out_features=2 * model_cfg.channels,
            weight_bit_prec=model_cfg.weight_bit_prec,
            weight_quant_dtype=model_cfg.weight_quant_dtype,
            weight_quant_method=model_cfg.weight_quant_method,
            bias=True,
            quantize_bias=model_cfg.quantize_bias,
            observer_only_in_train=model_cfg.observer_only_in_train,
        )
        self.depthwise_conv = Conv1dQuant(
            in_channels=model_cfg.channels,
            out_channels=model_cfg.channels,
            kernel_size=model_cfg.kernel_size,
            padding=(model_cfg.kernel_size - 1) // 2,
            groups=model_cfg.channels,
            bias=True,
            stride=1,
            dilation=1,
            weight_bit_prec=model_cfg.weight_bit_prec,
            weight_quant_dtype=model_cfg.weight_quant_dtype,
            weight_quant_method=model_cfg.weight_quant_method,
            quantize_bias=model_cfg.quantize_bias,
            observer_only_in_train=model_cfg.observer_only_in_train,
        )
        self.dconv_1_in_quant = ActivationQuantizer(
            bit_precision=model_cfg.activation_bit_prec,
            dtype=model_cfg.activation_quant_dtype,
            method=model_cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=model_cfg.moving_average,
            observer_only_in_train=model_cfg.observer_only_in_train,
        )

        self.dconv_1_out_quant = ActivationQuantizer(
            bit_precision=model_cfg.activation_bit_prec,
            dtype=model_cfg.activation_quant_dtype,
            method=model_cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=model_cfg.moving_average,
            observer_only_in_train=model_cfg.observer_only_in_train,
        )

        self.pointwise_conv2 = LinearQuant(
            in_features=model_cfg.channels,
            out_features=model_cfg.channels,
            weight_bit_prec=model_cfg.weight_bit_prec,
            weight_quant_dtype=model_cfg.weight_quant_dtype,
            weight_quant_method=model_cfg.weight_quant_method,
            bias=True,
            quantize_bias=model_cfg.quantize_bias,
            observer_only_in_train=model_cfg.observer_only_in_train,
        )
        self.pconv_1_in_quant = ActivationQuantizer(
            bit_precision=model_cfg.activation_bit_prec,
            dtype=model_cfg.activation_quant_dtype,
            method=model_cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=model_cfg.moving_average,
            observer_only_in_train=model_cfg.observer_only_in_train,
        )

        self.pconv_1_out_quant = ActivationQuantizer(
            bit_precision=model_cfg.activation_bit_prec,
            dtype=model_cfg.activation_quant_dtype,
            method=model_cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=model_cfg.moving_average,
            observer_only_in_train=model_cfg.observer_only_in_train,
        )

        self.pconv_2_in_quant = ActivationQuantizer(
            bit_precision=model_cfg.activation_bit_prec,
            dtype=model_cfg.activation_quant_dtype,
            method=model_cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=model_cfg.moving_average,
            observer_only_in_train=model_cfg.observer_only_in_train,
        )

        self.pconv_2_out_quant = ActivationQuantizer(
            bit_precision=model_cfg.activation_bit_prec,
            dtype=model_cfg.activation_quant_dtype,
            method=model_cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=model_cfg.moving_average,
            observer_only_in_train=model_cfg.observer_only_in_train,
        )
        self.layer_norm = nn.LayerNorm(model_cfg.channels)
        self.norm = deepcopy(model_cfg.norm)
        self.dropout = nn.Dropout(model_cfg.dropout)
        self.activation = model_cfg.activation

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        :param tensor: input tensor of shape [B,T,F]
        :return: torch.Tensor of shape [B,T,F]
        """
        tensor = self.layer_norm(tensor)
        tensor = self.pconv_1_in_quant(tensor)
        tensor = self.pointwise_conv1(tensor)  # [B,T,2F]
        tensor = self.pconv_1_out_quant(tensor)
        tensor = nn.functional.glu(tensor, dim=-1)  # [B,T,F]

        # conv layers expect shape [B,F,T] so we have to transpose here
        tensor = tensor.transpose(1, 2)  # [B,F,T]
        tensor = self.dconv_1_in_quant(tensor)
        tensor = self.depthwise_conv(tensor)
        tensor = self.dconv_1_out_quant(tensor)

        tensor = self.norm(tensor)
        tensor = tensor.transpose(1, 2)  # transpose back to [B,T,F]

        tensor = self.activation(tensor)
        tensor = self.pconv_2_in_quant(tensor)
        tensor = self.pointwise_conv2(tensor)
        tensor = self.pconv_2_out_quant(tensor)

        return self.dropout(tensor)