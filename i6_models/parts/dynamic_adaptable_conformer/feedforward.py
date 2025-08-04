from __future__ import annotations

__all__ = [
    "ConformerPositionwiseFeedForwardV1",
    "ConformerPositionwiseFeedForwardV1Config",
]

from dataclasses import dataclass
from typing import Callable
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from i6_models.config import ModelConfiguration


@dataclass
class ConformerPositionwiseFeedForwardV1Config(ModelConfiguration):
    """
    Attributes:
        input_dim: input dimension
        hidden_dim: hidden dimension (normally set to 4*input_dim as suggested by the paper)
        dropout: dropout probability
        activation: activation function
    """

    input_dim: int
    hidden_dim: int
    dropout: float
    activation: Callable[[torch.Tensor], torch.Tensor] = nn.functional.silu


class ConformerPositionwiseFeedForwardV1(nn.Module):
    """
    Conformer feedforward module
    """

    def __init__(self, cfg: ConformerPositionwiseFeedForwardV1Config):
        super().__init__()

        self.layer_norm = nn.LayerNorm(cfg.input_dim)
        self.linear_ff_weight = torch.nn.parameter.Parameter(
            torch.empty((cfg.hidden_dim, cfg.input_dim))
        )
        self.linear_ff_bias = torch.nn.parameter.Parameter(torch.empty(cfg.hidden_dim))
        self.activation = cfg.activation
        self.linear_out_weight = torch.nn.parameter.Parameter(
            torch.empty((cfg.input_dim, cfg.hidden_dim))
        )
        self.linear_out_bias = torch.nn.parameter.Parameter(torch.empty(cfg.input_dim))
        self.dropout = cfg.dropout
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """same initialization as default pytorch linear layer"""
        init.kaiming_uniform_(self.linear_ff_weight, a=math.sqrt(5))
        if self.linear_ff_bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.linear_ff_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.linear_ff_bias, -bound, bound)

        init.kaiming_uniform_(self.linear_out_weight, a=math.sqrt(5))
        if self.linear_out_bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.linear_out_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.linear_out_bias, -bound, bound)

    def forward(
        self,
        tensor: torch.Tensor,
        channel_chunk_gates: torch.Tensor = None,
        hard_prune: bool = False,
        adjust_dropout: bool = False,
    ) -> torch.Tensor:
        """
        :param tensor: shape [B,T,F], F=input_dim
        :return: shape [B,T,F], F=input_dim
        """
        input_dim, hidden_dim = self.linear_out_weight.size()
        if channel_chunk_gates is not None:
            num_chunks = len(channel_chunk_gates)
            if num_chunks == 0:
                chunk_dim = 0
            else:
                chunk_dim = hidden_dim // num_chunks

        if hidden_dim == 0:
            return tensor

        dropout_p = self.dropout
        if adjust_dropout:
            adjust_scale = hidden_dim / (input_dim * 4)
            dropout_p *= adjust_scale

        # if channel_chunk_gates is not None:
        #    assert channel_chunk_gates.size()[0] == hidden_dim // input_dim, "size dispatch"
        tensor = self.layer_norm(tensor)
        if channel_chunk_gates is None:
            tensor = F.linear(tensor, self.linear_ff_weight, self.linear_ff_bias)
        else:
            weights = torch.reshape(
                self.linear_ff_weight, (num_chunks, -1, input_dim)
            )  # (C, F, F)
            bias = torch.reshape(self.linear_ff_bias, (num_chunks, -1))  # (C, F)
            if not hard_prune and self.training:
                weights = torch.einsum(
                    "cij,c->cij", weights, channel_chunk_gates
                )  # (C, F, F)
                bias = torch.einsum("ci,c->ci", bias, channel_chunk_gates)  # (C, F)
            else:
                weights = weights[channel_chunk_gates.bool()]
                bias = bias[channel_chunk_gates.bool()]
            weights = torch.reshape(weights, (-1, input_dim))  # (C*F, F)
            if hard_prune:
                num_chunks = torch.count_nonzero(channel_chunk_gates)
            bias = torch.reshape(bias, (num_chunks * chunk_dim,))  # (C*F, 1)
            tensor = F.linear(tensor, weights, bias)

        tensor = self.activation(tensor)  # [B,T,F]
        tensor = nn.functional.dropout(
            tensor, p=dropout_p, training=self.training
        )  # [B,T,F]
        if channel_chunk_gates is None:
            tensor = F.linear(tensor, self.linear_out_weight, self.linear_out_bias)
        else:
            weights = torch.reshape(
                self.linear_out_weight, (input_dim, -1, num_chunks)
            )  # (F, F, 4)
            if not hard_prune and self.training:
                weights = torch.einsum(
                    "ijc,c->ijc", weights, channel_chunk_gates
                )  # (F, F, C)
                weights = torch.reshape(weights, (input_dim, -1))  # (F, C*F)
            else:
                weights = weights[:, :, channel_chunk_gates.bool()]
                weights = torch.reshape(weights, (input_dim, -1))
            tensor = F.linear(tensor, weights, self.linear_out_bias)

        tensor = nn.functional.dropout(
            tensor, p=dropout_p, training=self.training
        )  # [B,T,F]
        return tensor

    def adapt_module(
        self, new_hidden_dim: int, channel_chunk_gates: torch.Tensor = None
    ):
        if len(channel_chunk_gates) > 0:
            prev_hidden_dim, input_dim = self.linear_ff_weight.size()
            dim_per_chunk = prev_hidden_dim // len(channel_chunk_gates)
            new_channel_chunks = new_hidden_dim // dim_per_chunk
            prev_channel_chunks = prev_hidden_dim // dim_per_chunk

            if not (
                prev_channel_chunks == new_channel_chunks
                and torch.all(channel_chunk_gates)
            ):
                if new_channel_chunks >= prev_channel_chunks:
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

                temp_linear_ff_weight = torch.reshape(
                    self.linear_ff_weight, (-1, dim_per_chunk, input_dim)
                )  # (C, F, F)
                temp_linear_ff_weight = temp_linear_ff_weight[
                    channel_chunk_gates.bool()
                ]
                temp_linear_ff_bias = torch.reshape(
                    self.linear_ff_bias, (-1, dim_per_chunk)
                )  # (C, F)
                temp_linear_ff_bias = temp_linear_ff_bias[channel_chunk_gates.bool()]

                temp_linear_out_weight = torch.reshape(
                    self.linear_out_weight, (input_dim, dim_per_chunk, -1)
                )  # (F, F, 4)
                temp_linear_out_weight = temp_linear_out_weight[
                    :, :, channel_chunk_gates.bool()
                ]

                with torch.no_grad():
                    self.linear_ff_weight = torch.nn.parameter.Parameter(
                        torch.empty((new_hidden_dim, input_dim))
                    )
                    self.linear_ff_bias = torch.nn.parameter.Parameter(
                        torch.empty(new_hidden_dim)
                    )
                    self.linear_out_weight = torch.nn.parameter.Parameter(
                        torch.empty((input_dim, new_hidden_dim))
                    )

                    self.reset_parameters()

                    linear_ff_weight = torch.reshape(
                        self.linear_ff_weight, (-1, dim_per_chunk, input_dim)
                    )
                    linear_ff_weight[new_chunk_gates.bool()] = temp_linear_ff_weight.to(
                        linear_ff_weight.device
                    )
                    self.linear_ff_weight.data = torch.reshape(
                        self.linear_ff_weight, (-1, input_dim)
                    )

                    linear_ff_bias = torch.reshape(
                        self.linear_ff_bias, (-1, dim_per_chunk)
                    )
                    linear_ff_bias[new_chunk_gates.bool()] = temp_linear_ff_bias.to(
                        linear_ff_bias.device
                    )
                    self.linear_ff_bias.data = torch.flatten(linear_ff_bias)

                    linear_out_weight = torch.reshape(
                        self.linear_out_weight, (input_dim, dim_per_chunk, -1)
                    )
                    linear_out_weight[
                        :, :, new_chunk_gates.bool()
                    ] = temp_linear_out_weight.to(linear_out_weight.device)
                    self.linear_out_weight.data = torch.reshape(
                        linear_out_weight, (input_dim, -1)
                    )

    def double_and_prune_params(
        self, new_hidden_dim: int, selected_indices: list, weight_noise: float = 0
    ):
        prev_hidden_dim, input_dim = self.linear_ff_weight.size()

        if len(selected_indices) > 0:
            dim_per_chunk = new_hidden_dim // len(selected_indices)

            temp_linear_ff_weight = torch.reshape(
                self.linear_ff_weight, (-1, dim_per_chunk, input_dim)
            )  # (C, F, F)
            temp_linear_ff_bias = torch.reshape(
                self.linear_ff_bias, (-1, dim_per_chunk)
            )  # (C, F)
            temp_linear_out_weight = torch.reshape(
                self.linear_out_weight, (input_dim, dim_per_chunk, -1)
            )  # (F, F, 4)

        with torch.no_grad():
            self.linear_ff_weight = torch.nn.parameter.Parameter(
                torch.empty((new_hidden_dim, input_dim))
            )
            self.linear_ff_bias = torch.nn.parameter.Parameter(
                torch.empty(new_hidden_dim)
            )
            self.linear_out_weight = torch.nn.parameter.Parameter(
                torch.empty((input_dim, new_hidden_dim))
            )
            self.reset_parameters()

            if len(selected_indices) > 0:
                linear_ff_weight = temp_linear_ff_weight[selected_indices, :, :]
                self.linear_ff_weight.data = torch.reshape(
                    linear_ff_weight, (-1, input_dim)
                )

                linear_ff_bias = temp_linear_ff_bias[selected_indices, :]
                self.linear_ff_bias.data = torch.flatten(linear_ff_bias)

                linear_out_weight = temp_linear_out_weight[:, :, selected_indices]
                self.linear_out_weight.data = torch.reshape(
                    linear_out_weight, (input_dim, -1)
                )

                if weight_noise > 0:
                    self.linear_ff_weight.data += (weight_noise**0.5) * torch.randn(
                        self.linear_ff_weight.data.shape
                    ).to("cuda")
                    self.linear_ff_bias.data += (weight_noise**0.5) * torch.randn(
                        self.linear_ff_bias.data.shape
                    ).to("cuda")
                    self.linear_out_weight.data += (weight_noise**0.5) * torch.randn(
                        self.linear_out_weight.data.shape
                    ).to("cuda")
