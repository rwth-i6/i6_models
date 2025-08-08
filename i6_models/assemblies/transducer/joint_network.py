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
        joint_normalization: whether use normalized joint gradient for fullsum
    """

    ffnn_cfg: FeedForwardBlockV1Config
    joint_normalization: bool


class TransducerJointNetworkV1(nn.Module):
    def __init__(
        self,
        cfg: TransducerJointNetworkV1Config,
    ) -> None:
        super().__init__()
        self.ffnn = FeedForwardBlockV1(cfg.ffnn_cfg)
        self.joint_normalization = cfg.joint_normalization
        self.output_dim = self.ffnn.output_dim

    def forward(
        self,
        source_encodings: torch.Tensor,  # [1, T, E]
        target_encodings: torch.Tensor,  # [B, S, P]
    ) -> torch.Tensor:  # [B, T, S, F]
        """
        Forward pass for recognition.
        """
        combined_encodings = source_encodings.unsqueeze(2) + target_encodings.unsqueeze(1)
        output = self.ffnn(combined_encodings)  # [B, T, S, F]

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
        
        # additive combination
        combined_encodings = source_encodings.unsqueeze(2) + target_encodings.unsqueeze(1)

        if self.joint_normalization:
            source_lengths_safe = torch.clamp(source_lengths, min=1).float()
            target_lengths_safe = torch.clamp(target_lengths, min=1).float()
            scale_enc  = (1.0 / target_lengths_safe).view(-1, 1, 1).to(source_encodings.device)
            scale_pred = (1.0 / source_lengths_safe).view(-1, 1, 1).to(target_encodings.device)

            source_encodings.register_hook (lambda g, s=scale_enc : g * s)
            target_encodings.register_hook(lambda g, s=scale_pred: g * s) 

        # Pass through FFNN
        output = self.ffnn(combined_encodings)  # [B, T, S+1, F]
        return output, source_lengths, target_lengths
