__all__ = ["TransducerJointNetworkV1Config", "TransducerJointNetworkV1"]

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import torch
from torch import nn

from i6_models.config import ModelConfiguration
from i6_models.parts.ffnn import FeedForwardBlockV1Config, FeedForwardBlockV1


@dataclass
class TransducerJointNetworkV1Config(ModelConfiguration):
    """
    Configuration for the Transducer Joint Network.

    Attributes:
        ffnn_cfg: Configuration for the internal feed-forward network.
        num_layers: Number of FFNN prediction network layer
    """

    ffnn_cfg: FeedForwardBlockV1Config


class TransducerJointNetworkV1(nn.Module):
    def __init__(
        self,
        cfg: TransducerJointNetworkV1Config,
    ) -> None:
        super().__init__()
        self.ffnn = FeedForwardBlockV1(cfg.ffnn_cfg)
        self.output_dim = self.ffnn.output_dim

    def forward(
        self,
        source_encodings: torch.Tensor,  # [1, T, E]
        target_encodings: torch.Tensor,  # [B, S, P]
    ) -> torch.Tensor:  # [B, T, S, F]
        """
        Forward pass for recognition.
        """
        source_encodings = source_encodings.unsqueeze(2).expand(
            target_encodings.size(0), -1, target_encodings.size(1), -1
        )  # [B, T, S, E]
        target_encodings = target_encodings.unsqueeze(1).expand(-1, source_encodings.size(1), -1, -1)  # [B, T, S, P]
        joint_network_inputs = torch.cat([source_encodings, target_encodings], dim=-1)  # [B, T, S, E + P]
        output = self.ffnn(joint_network_inputs)  # [B, T, S, F]

        if not self.training:
            output = torch.log_softmax(output, dim=-1)  # [B, T, S, F]
        return output

    def forward_viterbi(
        self,
        source_encodings: torch.Tensor,  # [B, T, E]
        source_lengths: torch.Tensor,  # [B]
        target_encodings: torch.Tensor,  # [B, T, P]
        target_lengths: torch.Tensor,  # [B]
    ) -> torch.Tensor:  # [B, T, F]
        """
        Forward pass for Viterbi training.
        """
        joint_network_inputs = torch.cat([source_encodings, target_encodings], dim=-1)  # [B, T, E + P]
        output = self.ffnn(joint_network_inputs)  # [B, T, F]
        if not self.training:
            output = torch.log_softmax(output, dim=-1)  # [B, T, F]
        return output, source_lengths, target_lengths

    def forward_fullsum(
        self,
        source_encodings: torch.Tensor,  # [B, T, E]
        source_lengths: torch.Tensor,  # [B]
        target_encodings: torch.Tensor,  # [B, S+1, P]
        target_lengths: torch.Tensor,  # [B]
    ) -> torch.Tensor:  # [B, T, S+1, F]
        """
        Forward pass for fullsum training. Returns output with shape [B, T, S+1, F].
        """
        batch_outputs = []
        max_target_length = target_encodings.size(1)  # S+1
        max_source_length = source_encodings.size(1)  # T

        # Expand source_encodings
        expanded_source = source_encodings.unsqueeze(2).expand(-1, -1, max_target_length, -1)  # [B, T, S+1, E]

        # Expand target_encodings
        expanded_target = target_encodings.unsqueeze(1).expand(-1, max_source_length, -1, -1)  # [B, T, S+1, P]

        # Concatenate
        combination = torch.cat([expanded_source, expanded_target], dim=-1)  # [B, T, S+1, E + P]

        # Pass through FFNN
        output = self.ffnn(combination)  # [B, T, S+1, F]
        return output, source_lengths, target_lengths
