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

    def forward(
        self,
        source_encodings: torch.Tensor,  # [B, T, E]
        target_encodings: torch.Tensor,  # [B, S, P]
    ) -> torch.Tensor:  # [B, T, S, F]
        """
        Forward pass for recognition. Assume T = 1 and S = 1
        """
        source_encodings = source_encodings.unsqueeze(2).expand(
            -1, -1, target_encodings.size(1), -1
        )  # [B, T, S, E]
        target_encodings = target_encodings.unsqueeze(1).expand(
            -1, source_encodings.size(1), -1, -1
        )  # [B, T, S, P]
        joint_network_inputs = torch.cat(
            [source_encodings, target_encodings], dim=-1
        )  # [B, T, S, E + P]
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
        joint_network_inputs = torch.cat(
            [source_encodings, target_encodings], dim=-1
        )  # [B, T, E + P]
        output = self.ffnn(joint_network_inputs)  # [B, T, F]
        if not self.training:
            output = torch.log_softmax(output, dim=-1)  # [B, T, F]
        return output

    def forward_fullsum(
        self,
        source_encodings: torch.Tensor,  # [B, T, E]
        source_lengths: torch.Tensor,  # [B]
        target_encodings: torch.Tensor,  # [B, S+1, P]
        target_lengths: torch.Tensor,  # [B]
    ) -> torch.Tensor:  # [B, T, S+1, F]
        """
        Forward pass for fullsum training.  Returns output with shape [B, T, S+1, F].
        """
        batch_outputs = []
        for b in range(source_encodings.size(0)):
            valid_source = source_encodings[b, : source_lengths[b], :]  # [T_b, E]
            valid_target = target_encodings[b, : target_lengths[b] + 1, :]  # [S_b+1, P]

            expanded_source = valid_source.unsqueeze(1).expand(
                -1, int(target_lengths[b].item()) + 1, -1
            )  # [T_b, S_b+1, E]
            expanded_target = valid_target.unsqueeze(0).expand(
                int(source_lengths[b].item()), -1, -1
            )  # [T_b, S_b+1, P]
            combination = torch.cat(
                [expanded_source, expanded_target], dim=-1
            )  # [T_b, S_b+1, E + P]
            output = self.ffnn(combination)  # [T_b, S_b+1, F]
            batch_outputs.append(output)
        # Pad the outputs to a common shape, if necessary.  This is crucial for
        # handling variable sequence lengths within a batch.  The padding
        # ensures that the final output tensor has a consistent shape.
        padded_outputs = torch.nn.utils.rnn.pad_sequence(
            batch_outputs, batch_first=True, padding_value=0.0
        )  # [B, max_T, max_S+1, F]

        return padded_outputs
