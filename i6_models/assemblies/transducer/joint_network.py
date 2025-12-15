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
    """

    ffnn_cfg: FeedForwardBlockV1Config


class TransducerJointNetworkV1(nn.Module):
    def __init__(
        self,
        cfg: TransducerJointNetworkV1Config,
        enc_input_dim: int,
        pred_input_dim: int,
    ) -> None:
        super().__init__()
        hidden_dim = cfg.ffnn_cfg.layer_sizes[0]
        self.enc_proj = nn.Linear(enc_input_dim, hidden_dim, bias=True)
        self.pred_proj = nn.Linear(pred_input_dim, hidden_dim, bias=False) # Bias handled by enc_proj
        
        self.activation = cfg.ffnn_cfg.layer_activations[0]
        self.dropout = nn.Dropout(cfg.ffnn_cfg.dropouts[0]) if cfg.ffnn_cfg.dropouts else nn.Identity()

        # Build the rest of the network (if any)
        if len(cfg.ffnn_cfg.layer_sizes) > 1:
            remaining_cfg = FeedForwardBlockV1Config(
                input_dim=hidden_dim,
                layer_sizes=cfg.ffnn_cfg.layer_sizes[1:],
                dropouts=cfg.ffnn_cfg.dropouts[1:] if cfg.ffnn_cfg.dropouts else None,
                layer_activations=cfg.ffnn_cfg.layer_activations[1:],
                use_layer_norm=cfg.ffnn_cfg.use_layer_norm,
            )
            self.ffnn = FeedForwardBlockV1(remaining_cfg)
        else:
            self.ffnn = nn.Identity()

        self.output_dim = cfg.ffnn_cfg.layer_sizes[-1]

    def _forward_joint(self, enc: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        # Project individually then broadcast-sum
        enc_proj = self.enc_proj(enc).unsqueeze(2)   # [B, T, 1, H]
        pred_proj = self.pred_proj(pred).unsqueeze(1) # [B, 1, U, H]
        
        combined = enc_proj + pred_proj
        
        if self.activation is not None:
            combined = self.activation(combined)
        combined = self.dropout(combined)
        
        return self.ffnn(combined)

    def forward(
        self,
        source_encodings: torch.Tensor,  # [1, T, E]
        target_encodings: torch.Tensor,  # [B, S, P]
    ) -> torch.Tensor:  # [B, T, S, F]
        """
        Forward pass for recognition.
        """
        output = self._forward_joint(source_encodings, target_encodings)

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
        # For Viterbi, dimensions align (T=T), so we can sum directly without broadcasting
        enc_proj = self.enc_proj(source_encodings)
        pred_proj = self.pred_proj(target_encodings)
        combined = enc_proj + pred_proj
        
        if self.activation is not None:
            combined = self.activation(combined)
        combined = self.dropout(combined)
        
        output = self.ffnn(combined)  # [B, T, F]
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
        output = self._forward_joint(source_encodings, target_encodings)
        return output, source_lengths, target_lengths
