from __future__ import annotations

__all__ = ["ConformerConvolutionV2", "ConformerConvolutionV2Config"]

from dataclasses import dataclass
from copy import deepcopy
import math
import torch
from torch import nn
from i6_models.config import ModelConfiguration
from typing import Callable, Union
from torch.nn import init
import torch.nn.functional as F


@dataclass
class ConformerConvolutionV2Config(ModelConfiguration):
    """
    Attributes:
        input_dim: input feature dimension
        channels: number of channels for conv layers
        kernel_size: kernel size of conv layers
        dropout: dropout probability
        activation: activation function applied after normalization
        norm: normalization layer with input of shape [N,C,T]
    """

    input_dim: int
    channels: int
    kernel_size: int
    dropout: float
    activation: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]
    norm: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]

    def check_valid(self):
        assert (
            self.kernel_size % 2 == 1
        ), "ConformerConvolutionV1 only supports odd kernel sizes"

    def __post_init__(self):
        super().__post_init__()
        self.check_valid()


class ConformerConvolutionV2(nn.Module):
    """
    Conformer convolution module.
    see also: https://github.com/espnet/espnet/blob/713e784c0815ebba2053131307db5f00af5159ea/espnet/nets/pytorch_backend/conformer/convolution.py#L13

    Uses explicit padding for ONNX exportability, see:
    https://github.com/pytorch/pytorch/issues/68880
    """

    def __init__(self, model_cfg: ConformerConvolutionV2Config):
        """
        :param model_cfg: model configuration for this module
        """
        super().__init__()
        model_cfg.check_valid()
        self.pointwise_conv1_weights = torch.nn.parameter.Parameter(
            torch.empty((2 * model_cfg.channels, model_cfg.input_dim))
        )
        self.pointwise_conv1_bias = torch.nn.parameter.Parameter(
            torch.empty(2 * model_cfg.channels)
        )
        self.depthwise_conv_weights = torch.nn.Parameter(
            torch.empty((model_cfg.channels, 1, model_cfg.kernel_size))
        )
        self.depthwise_conv_bias = torch.nn.Parameter(torch.empty(model_cfg.channels))
        self.kernel_size = model_cfg.kernel_size
        self.input_dim = model_cfg.input_dim
        self.channels = model_cfg.channels
        self.pointwise_conv2_weights = torch.nn.parameter.Parameter(
            torch.empty((model_cfg.input_dim, model_cfg.channels))
        )
        self.pointwise_conv2_bias = torch.nn.parameter.Parameter(
            torch.empty(model_cfg.input_dim)
        )
        self.layer_norm = nn.LayerNorm(model_cfg.input_dim)
        self.norm_weight = torch.nn.Parameter(torch.empty((model_cfg.channels,)))
        self.norm_bias = torch.nn.Parameter(torch.empty((model_cfg.channels,)))
        self.norm_eps = 1e-05
        self.dropout = model_cfg.dropout
        self.activation = model_cfg.activation
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.pointwise_conv1_weights, a=math.sqrt(5))
        if self.pointwise_conv1_bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.pointwise_conv1_weights)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.pointwise_conv1_bias, -bound, bound)

        init.kaiming_uniform_(self.pointwise_conv2_weights, a=math.sqrt(5))
        if self.pointwise_conv2_bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.pointwise_conv2_weights)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.pointwise_conv2_bias, -bound, bound)

        init.kaiming_uniform_(self.depthwise_conv_weights, a=math.sqrt(5))
        if self.depthwise_conv_bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.depthwise_conv_weights)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.depthwise_conv_bias, -bound, bound)

        init.ones_(self.norm_weight)
        if self.norm_bias is not None:
            init.zeros_(self.norm_bias)

    def forward(
        self,
        tensor: torch.Tensor,
        channel_chunk_gates: torch.Tensor = None,
        hard_prune: bool = False,
        adjust_dropout: bool = False,
    ) -> torch.Tensor:
        """
        :param tensor: input tensor of shape [B,T,F]
        :return: torch.Tensor of shape [B,T,F]
        """
        self.channels = self.pointwise_conv2_weights.size()[1]

        if channel_chunk_gates is not None:
            num_chunks_not_pruned = channel_chunk_gates.size()[0]
            if not hard_prune:
                num_chunks = channel_chunk_gates.size()[0]
            else:
                num_chunks = torch.count_nonzero(channel_chunk_gates)

        if self.pointwise_conv1_weights.size(0) == 0:
            return tensor

        dropout_p = self.dropout
        if adjust_dropout:
            adjust_scale = (
                self.pointwise_conv2_weights.size()[1]
                / self.pointwise_conv2_weights.size()[0]
            )
            dropout_p = self.dropout * adjust_scale

        tensor = self.layer_norm(tensor)

        if channel_chunk_gates is None:
            # print("tensor size", tensor.size())
            # print("pointwise_conv1_weights size", self.pointwise_conv1_weights.size())
            # print("pointwise_conv1_bias size", self.pointwise_conv1_bias.size())
            tensor = F.linear(
                tensor, self.pointwise_conv1_weights, self.pointwise_conv1_bias
            )
        else:
            weights = torch.reshape(
                self.pointwise_conv1_weights,
                (num_chunks_not_pruned, -1, self.input_dim),
            )
            bias = torch.reshape(
                self.pointwise_conv1_bias,
                (num_chunks_not_pruned, 2 * self.channels // num_chunks_not_pruned),
            )
            if not hard_prune and self.training:
                weights = torch.einsum(
                    "cij,c->cij", weights, channel_chunk_gates
                )  # (C, F, F)
                bias = torch.einsum("ci,c->ci", bias, channel_chunk_gates)  # (C, F)
            else:
                weights = weights[channel_chunk_gates.bool()]
                bias = bias[channel_chunk_gates.bool()]
            weights = torch.reshape(weights, (-1, self.input_dim))
            bias_dim = num_chunks / (channel_chunk_gates.size()[0]) * self.channels * 2
            bias = torch.reshape(bias, (int(bias_dim),))  # (C*F, 1)
            tensor = F.linear(tensor, weights, bias)

        tensor = nn.functional.glu(tensor, dim=-1)  # [B,T,F]

        # conv layers expect shape [B,F,T] so we have to transpose here
        tensor = tensor.transpose(1, 2)  # [B,F,T]
        if channel_chunk_gates is None:
            num_groups = (
                self.depthwise_conv_weights.size()[0]
                // self.depthwise_conv_weights.size()[1]
            )
            tensor = F.conv1d(
                tensor,
                self.depthwise_conv_weights,
                self.depthwise_conv_bias,
                1,
                (self.kernel_size - 1) // 2,
                1,
                num_groups,
            )
        else:
            weights = torch.reshape(
                self.depthwise_conv_weights,
                (num_chunks_not_pruned, -1, 1, self.kernel_size),
            )
            bias = torch.reshape(
                self.depthwise_conv_bias,
                (num_chunks_not_pruned, self.channels // num_chunks_not_pruned),
            )
            if not hard_prune and self.training:
                weights = torch.einsum(
                    "cijk,c->cijk", weights, channel_chunk_gates
                )  # (C, F, F)
                bias = torch.einsum("ci,c->ci", bias, channel_chunk_gates)  # (C, F)
            else:
                weights = weights[channel_chunk_gates.bool()]
                bias = bias[channel_chunk_gates.bool()]
            bias_dim = num_chunks / (channel_chunk_gates.size()[0]) * self.channels
            bias = torch.reshape(bias, (int(bias_dim),))  # (C*F, 1)
            weights = torch.reshape(weights, (int(bias_dim), 1, self.kernel_size))

            tensor = F.conv1d(
                tensor, weights, bias, 1, (self.kernel_size - 1) // 2, 1, int(bias_dim)
            )

        tensor = tensor.transpose(1, -1)
        if not hard_prune:
            tensor = F.layer_norm(
                tensor,
                (self.norm_weight.size(0),),
                self.norm_weight,
                self.norm_bias,
                self.norm_eps,
            )
        else:
            weights = torch.reshape(
                self.norm_weight,
                (num_chunks_not_pruned, self.channels // num_chunks_not_pruned),
            )
            bias = torch.reshape(
                self.norm_bias,
                (num_chunks_not_pruned, self.channels // num_chunks_not_pruned),
            )
            weights = weights[channel_chunk_gates.bool()]
            bias = bias[channel_chunk_gates.bool()]
            weights = torch.reshape(
                weights,
                (int(num_chunks / channel_chunk_gates.size()[0] * self.channels),),
            )
            bias_dim = num_chunks / (channel_chunk_gates.size()[0]) * self.channels
            bias = torch.reshape(bias, (int(bias_dim),))  # (C*F, 1)
            tensor = F.layer_norm(
                tensor, (int(weights.size()[0]),), weights, bias, self.norm_eps
            )

        tensor = tensor.transpose(1, -1)
        tensor = tensor.transpose(1, 2)

        tensor = self.activation(tensor)

        if channel_chunk_gates is None:
            tensor = F.linear(
                tensor, self.pointwise_conv2_weights, self.pointwise_conv2_bias
            )
        else:
            weights = torch.reshape(
                self.pointwise_conv2_weights,
                (self.input_dim, -1, num_chunks_not_pruned),
            )
            if not hard_prune and self.training:
                weights = torch.einsum(
                    "ijc,c->ijc", weights, channel_chunk_gates
                )  # (F, F, C)
            else:
                weights = weights[:, :, channel_chunk_gates.bool()]
            weights = torch.reshape(weights, (self.input_dim, -1))
            tensor = F.linear(tensor, weights, self.pointwise_conv2_bias)

        tensor = nn.functional.dropout(
            tensor, p=dropout_p, training=self.training
        )  # [B,T,F]
        return tensor

    def adapt_module(
        self, new_num_channels: int, channel_chunk_gates: torch.Tensor = None
    ):
        if len(channel_chunk_gates) > 0:
            channels_per_chunk = int(
                self.pointwise_conv2_weights.size(1) / len(channel_chunk_gates)
            )
            new_channel_chunks = int(new_num_channels // channels_per_chunk)
            prior_channel_chunks = len(channel_chunk_gates)

            if not (
                new_channel_chunks == prior_channel_chunks
                and torch.all(channel_chunk_gates)
            ):
                if new_channel_chunks >= prior_channel_chunks:
                    new_chunk_gates = torch.cat(
                        (
                            channel_chunk_gates,
                            torch.zeros(
                                new_channel_chunks - len(channel_chunk_gates),
                                dtype=torch.bool,
                            ),
                        )
                    )
                else:
                    if new_channel_chunks > torch.sum(channel_chunk_gates):
                        new_chunk_gates = torch.tensor(
                            [True] * torch.sum(channel_chunk_gates)
                            + [False]
                            * (new_channel_chunks - torch.sum(channel_chunk_gates))
                        )
                    else:
                        new_chunk_gates = torch.tensor([True] * new_channel_chunks)

                # 1st point-wise convolution
                temp_pw_conv1_weights = torch.reshape(
                    self.pointwise_conv1_weights,
                    (prior_channel_chunks, -1, self.input_dim),
                )  # (C, F, F)
                temp_pw_conv1_weights = temp_pw_conv1_weights[
                    channel_chunk_gates.bool(), :
                ]
                temp_pw_conv1_bias = torch.reshape(
                    self.pointwise_conv1_bias, (prior_channel_chunks, -1)
                )  # (C, F)
                temp_pw_conv1_bias = temp_pw_conv1_bias[channel_chunk_gates.bool(), :]

                # depth-wise convolution
                temp_dw_conv_weights = torch.reshape(
                    self.depthwise_conv_weights,
                    (prior_channel_chunks, -1, 1, self.kernel_size),
                )
                temp_dw_conv_weights = temp_dw_conv_weights[
                    channel_chunk_gates.bool(), :, :, :
                ]
                temp_dw_conv_bias = torch.reshape(
                    self.depthwise_conv_bias, (prior_channel_chunks, -1)
                )
                temp_dw_conv_bias = temp_dw_conv_bias[channel_chunk_gates.bool(), :]

                # 2nd point-wise convolution
                temp_pw_conv2_weights = torch.reshape(
                    self.pointwise_conv2_weights,
                    (
                        self.input_dim,
                        prior_channel_chunks,
                        -1,
                    ),
                )  # (C, F, F)
                temp_pw_conv2_weights = temp_pw_conv2_weights[
                    :, channel_chunk_gates.bool(), :
                ]

                with torch.no_grad():
                    self.pointwise_conv1_weights = torch.nn.parameter.Parameter(
                        torch.empty((2 * new_num_channels, self.input_dim))
                    )
                    self.pointwise_conv1_bias = torch.nn.parameter.Parameter(
                        torch.empty(2 * new_num_channels)
                    )
                    self.depthwise_conv_weights = torch.nn.parameter.Parameter(
                        torch.empty((new_num_channels, 1, self.kernel_size))
                    )
                    self.depthwise_conv_bias = torch.nn.parameter.Parameter(
                        torch.empty(new_num_channels)
                    )
                    self.pointwise_conv2_weights = torch.nn.parameter.Parameter(
                        torch.empty((self.input_dim, new_num_channels))
                    )
                    self.norm_weight = torch.nn.Parameter(
                        torch.empty((new_num_channels,))
                    )
                    self.norm_bias = torch.nn.Parameter(
                        torch.empty((new_num_channels,))
                    )

                    self.reset_parameters()

                    if new_num_channels > 0:
                        # 1st point-wise convolution
                        pointwise_conv1_weights = torch.reshape(
                            self.pointwise_conv1_weights,
                            (
                                new_channel_chunks,
                                2 * new_num_channels // new_channel_chunks,
                                self.input_dim,
                            ),
                        )
                        pointwise_conv1_weights[
                            new_chunk_gates
                        ] = temp_pw_conv1_weights.to(pointwise_conv1_weights.device)
                        self.pointwise_conv1_weights = torch.nn.parameter.Parameter(
                            torch.reshape(
                                pointwise_conv1_weights,
                                (2 * new_num_channels, self.input_dim),
                            )
                        )

                        pointwise_conv1_bias = torch.reshape(
                            self.pointwise_conv1_bias,
                            (
                                new_channel_chunks,
                                2 * new_num_channels // new_channel_chunks,
                            ),
                        )
                        pointwise_conv1_bias[new_chunk_gates] = temp_pw_conv1_bias.to(
                            pointwise_conv1_bias.device
                        )
                        self.pointwise_conv1_bias = torch.nn.parameter.Parameter(
                            torch.flatten(pointwise_conv1_bias)
                        )

                        # depth-wise convolution
                        depthwise_conv_weights = torch.reshape(
                            self.depthwise_conv_weights,
                            (new_channel_chunks, -1, 1, self.kernel_size),
                        )
                        depthwise_conv_weights[
                            new_chunk_gates
                        ] = temp_dw_conv_weights.to(depthwise_conv_weights.device)
                        self.depthwise_conv_weights = torch.nn.parameter.Parameter(
                            torch.reshape(
                                depthwise_conv_weights,
                                (new_num_channels, 1, self.kernel_size),
                            )
                        )

                        depthwise_conv_bias = torch.reshape(
                            self.depthwise_conv_bias, (new_channel_chunks, -1)
                        )
                        depthwise_conv_bias[new_chunk_gates] = temp_dw_conv_bias.to(
                            depthwise_conv_bias.device
                        )
                        self.depthwise_conv_bias = torch.nn.parameter.Parameter(
                            torch.flatten(depthwise_conv_bias)
                        )

                        # 2nd point-wise convolution
                        pointwise_conv2_weights = torch.reshape(
                            self.pointwise_conv2_weights,
                            (self.input_dim, new_channel_chunks, -1),
                        )
                        pointwise_conv2_weights[
                            :, new_chunk_gates
                        ] = temp_pw_conv2_weights.to(pointwise_conv2_weights.device)
                        self.pointwise_conv2_weights = torch.nn.parameter.Parameter(
                            torch.reshape(pointwise_conv2_weights, (self.input_dim, -1))
                        )

    def double_and_prune_params(
        self, new_num_channels: int, selected_indices: list, weight_noise: float = 0
    ):
        if len(selected_indices) > 0:
            channels_per_chunk = new_num_channels / len(selected_indices)
            prior_channel_chunks = int(
                self.pointwise_conv2_weights.size(1) // channels_per_chunk
            )

            # 1st point-wise convolution
            temp_pw_conv1_weights = torch.reshape(
                self.pointwise_conv1_weights, (prior_channel_chunks, -1, self.input_dim)
            )  # (C, F, F)
            temp_pw_conv1_bias = torch.reshape(
                self.pointwise_conv1_bias, (prior_channel_chunks, -1)
            )  # (C, F)

            # depth-wise convolution
            temp_dw_conv_weights = torch.reshape(
                self.depthwise_conv_weights,
                (prior_channel_chunks, -1, 1, self.kernel_size),
            )
            temp_dw_conv_bias = torch.reshape(
                self.depthwise_conv_bias, (prior_channel_chunks, -1)
            )

            # 2nd point-wise convolution
            temp_pw_conv2_weights = torch.reshape(
                self.pointwise_conv2_weights,
                (
                    self.input_dim,
                    prior_channel_chunks,
                    -1,
                ),
            )  # (C, F, F)

        with torch.no_grad():
            self.pointwise_conv1_weights = torch.nn.parameter.Parameter(
                torch.empty((2 * new_num_channels, self.input_dim))
            )
            self.pointwise_conv1_bias = torch.nn.parameter.Parameter(
                torch.empty(2 * new_num_channels)
            )
            self.depthwise_conv_weights = torch.nn.parameter.Parameter(
                torch.empty((new_num_channels, 1, self.kernel_size))
            )
            self.depthwise_conv_bias = torch.nn.parameter.Parameter(
                torch.empty(new_num_channels)
            )
            self.pointwise_conv2_weights = torch.nn.parameter.Parameter(
                torch.empty((self.input_dim, new_num_channels))
            )
            self.norm_weight = torch.nn.Parameter(torch.empty((new_num_channels,)))
            self.norm_bias = torch.nn.Parameter(torch.empty((new_num_channels,)))

            self.reset_parameters()

            if len(selected_indices) > 0:
                # 1st point-wise convolution
                pointwise_conv1_weights = temp_pw_conv1_weights[selected_indices, :, :]
                self.pointwise_conv1_weights = torch.nn.parameter.Parameter(
                    torch.reshape(
                        pointwise_conv1_weights, (2 * new_num_channels, self.input_dim)
                    )
                )

                pointwise_conv1_bias = temp_pw_conv1_bias[selected_indices, :]
                self.pointwise_conv1_bias = torch.nn.parameter.Parameter(
                    torch.flatten(pointwise_conv1_bias)
                )

                # depth-wise convolution
                depthwise_conv_weights = temp_dw_conv_weights[selected_indices, :, :, :]
                self.depthwise_conv_weights = torch.nn.parameter.Parameter(
                    torch.reshape(
                        depthwise_conv_weights, (new_num_channels, 1, self.kernel_size)
                    )
                )

                depthwise_conv_bias = temp_dw_conv_bias[selected_indices, :]
                self.depthwise_conv_bias = torch.nn.parameter.Parameter(
                    torch.flatten(depthwise_conv_bias)
                )

                # 2nd point-wise convolution
                pointwise_conv2_weights = temp_pw_conv2_weights[:, selected_indices, :]
                self.pointwise_conv2_weights = torch.nn.parameter.Parameter(
                    torch.reshape(pointwise_conv2_weights, (self.input_dim, -1))
                )

                if weight_noise > 0:
                    self.pointwise_conv1_weights.data += (
                        weight_noise**0.5
                    ) * torch.randn(self.pointwise_conv1_weights.data.shape).to("cuda")
                    self.pointwise_conv1_bias.data += (
                        weight_noise**0.5
                    ) * torch.randn(self.pointwise_conv1_bias.data.shape).to("cuda")
                    self.depthwise_conv_weights.data += (
                        weight_noise**0.5
                    ) * torch.randn(self.depthwise_conv_weights.data.shape).to("cuda")
                    self.depthwise_conv_bias.data += (
                        weight_noise**0.5
                    ) * torch.randn(self.depthwise_conv_bias.data.shape).to("cuda")
                    self.pointwise_conv2_weights += (weight_noise**0.5) * torch.randn(
                        self.pointwise_conv2_weights.data.shape
                    ).to("cuda")
