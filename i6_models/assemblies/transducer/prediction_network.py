__all__ = [
    "EmbeddingTransducerPredictionNetworkV1Config",
    "EmbeddingTransducerPredictionNetworkV1",
    "FfnnTransducerPredictionNetworkV1Config",
    "FfnnTransducerPredictionNetworkV1",
]

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

from i6_models.config import ModelConfiguration
from i6_models.parts.ffnn import FeedForwardBlockV1Config, FeedForwardBlockV1
from i6_models.parts.lstm import LstmBlockV1Config, LstmBlockV1


@dataclass
class EmbeddingTransducerPredictionNetworkV1Config(ModelConfiguration):
    """
    num_outputs: Number of output units (vocabulary size + blank).
    blank_id: Index of the blank token.
    context_history_size: Number of previous output tokens to consider as context
    embedding_dim: Dimension of the embedding layer.
    embedding_dropout: Dropout probability for the embedding layer.
    reduce_embedding: Whether to use a reduction mechanism for the context embedding.
    num_reduction_heads: Number of reduction heads if reduce_embedding is True.
    """

    num_outputs: int
    blank_id: int
    context_history_size: int
    embedding_dim: int
    embedding_dropout: float
    reduce_embedding: bool
    num_reduction_heads: Optional[int]

    def __post__init__(self):
        super().__post_init__()
        assert (num_reduction_heads is not None) == reduce_embedding

    @classmethod
    def from_child(cls, child_instance):
        return cls(
            child_instance.num_outputs,
            child_instance.blank_id,
            child_instance.context_history_size,
            child_instance.embedding_dim,
            child_instance.embedding_dropout,
            child_instance.reduce_embedding,
            child_instance.num_reduction_heads,
        )


class EmbeddingTransducerPredictionNetworkV1(nn.Module):
    def __init__(self, cfg: EmbeddingTransducerPredictionNetworkV1Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.blank_id = self.cfg.blank_id
        self.context_history_size = self.cfg.context_history_size
        self.embedding = nn.Embedding(
            num_embeddings=self.cfg.num_outputs,
            embedding_dim=self.cfg.embedding_dim,
            padding_idx=self.blank_id,
        )
        self.embed_dropout = nn.Dropout(self.cfg.embedding_dropout)
        self.output_dim = (
            self.cfg.embedding_dim * self.cfg.context_history_size
            if not self.cfg.reduce_embedding
            else self.cfg.embedding_dim
        )

        self.reduce_embedding = self.cfg.reduce_embedding
        self.num_reduction_heads = self.cfg.num_reduction_heads
        if self.reduce_embedding:
            self.register_buffer(
                "position_vectors",
                torch.randn(
                    self.cfg.context_history_size,
                    self.cfg.num_reduction_heads,
                    self.cfg.embedding_dim,
                ),
            )

    def _reduce_embedding(self, emb: torch.Tensor) -> torch.Tensor:
        """
        Reduces the context embedding using a weighted sum based on position vectors.
        """
        B, _, H, E = emb.shape
        emb_expanded = emb.unsqueeze(3)  # [B, 1, H, 1, E]
        pos_expanded = self.position_vectors.unsqueeze(0).unsqueeze(0)
        alpha = (emb_expanded * pos_expanded).sum(
            dim=-1, keepdim=True
        )  # [B, 1, H, K, 1]
        weighted = alpha * emb_expanded  # [B, 1, H, K, E]
        reduced = weighted.sum(dim=2).sum(dim=2)  # [B, 1, E]
        reduced *= 1.0 / (self.cfg.num_reduction_heads * self.cfg.context_history_size)
        return reduced

    def _forward_embedding(self, history: torch.Tensor) -> torch.Tensor:
        """
        Processes the input history through the embedding layer and optional reduction.
        """
        if len(history.shape) == 2:  # reshape if input shape [B, H]
            history = history.view(
                *history.shape[:-1], 1, history.shape[-1]
            )  # [B, 1, H]
        embed = self.embedding(history)  # [B, S, H, E]
        embed = self.embed_dropout(embed)
        if self.reduce_embedding:
            embed = self._reduce_embedding(embed)  # [B, S, E]
        else:
            embed = embed.flatten(start_dim=-2)  # [B, S, H*E]
        return embed

    def forward(
        self,
        history: torch.Tensor,  # [B, H]
    ) -> torch.Tensor:  # [B, 1, P]
        """
        Forward pass for recognition mode.
        """
        embed = self._forward_embedding(history)
        return embed

    def forward_fullsum(
        self,
        targets: torch.Tensor,  # [B, S]
        target_lengths: torch.Tensor,  # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # [B, S + 1, P], [B]
        """
        Forward pass for fullsum training.
        """
        non_context_padding = torch.full(
            (targets.size(0), self.cfg.context_history_size),
            fill_value=self.blank_id,
            dtype=targets.dtype,
            device=targets.device,
        )  # [B, H]
        extended_targets = torch.cat([non_context_padding, targets], dim=1)  # [B, S+H]
        history = torch.stack(
            [
                extended_targets[
                    :, self.cfg.context_history_size - 1 - i : (-i if i != 0 else None)
                ]
                for i in reversed(range(self.cfg.context_history_size))
            ],
            dim=-1,
        )  # [B, S+1, H]
        embed = self._forward_embedding(history)

        return embed, target_lengths

    def forward_viterbi(
        self,
        targets: torch.Tensor,  # [B, T]
        target_lengths: torch.Tensor,  # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # [B, T, P], [B]
        """
        Forward pass for viterbi training.
        """
        B, T = targets.shape
        history = torch.zeros(
            (B, T, self.cfg.context_history_size),
            dtype=targets.dtype,
            device=targets.device,
        )  # [B, T, H]
        recent_labels = torch.full(
            (B, self.cfg.context_history_size),
            fill_value=self.blank_id,
            dtype=targets.dtype,
            device=targets.device,
        )  # [B, H]

        for t in range(T):
            history[:, t, :] = recent_labels
            current_labels = targets[:, t]
            non_blank_positions = current_labels != self.blank_id
            recent_labels[non_blank_positions, 1:] = recent_labels[
                non_blank_positions, :-1
            ]
            recent_labels[non_blank_positions, 0] = current_labels[non_blank_positions]
        embed = self._forward_embedding(history)

        return embed, target_lengths


@dataclass
class FfnnTransducerPredictionNetworkV1Config(
    EmbeddingTransducerPredictionNetworkV1Config
):
    """
    Attributes:
        ffnn_cfg: Configuration for FFNN prediction network
    """

    ffnn_cfg: FeedForwardBlockV1Config


class FfnnTransducerPredictionNetworkV1(EmbeddingTransducerPredictionNetworkV1):
    """
    FfnnTransducerPredictionNetworkV1 with feedforward layers.
    """

    def __init__(self, cfg: FfnnTransducerPredictionNetworkV1Config):
        super().__init__(EmbeddingTransducerPredictionNetworkV1Config.from_child(cfg))
        cfg.ffnn_cfg.input_dim = self.output_dim
        self.ffnn = FeedForwardBlockV1(cfg.ffnn_cfg)
        self.output_dim = self.ffnn.output_dim

    def forward(
        self,
        history: torch.Tensor,  # [B, H]
    ) -> torch.Tensor:  # [B, 1, P]
        embed = super().forward(history)
        output = self.ffnn(embed)
        return output

    def forward_fullsum(
        self,
        targets: torch.Tensor,  # [B, S]
        target_lengths: torch.Tensor,  # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # [B, S + 1, P], [B]
        embed, _ = super().forward_fullsum(targets, target_lengths)
        output = self.ffnn(embed)
        return output, target_lengths

    def forward_viterbi(
        self,
        targets: torch.Tensor,  # [B, T]
        target_lengths: torch.Tensor,  # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # [B, T, P], [B]
        embed, _ = super().forward_viterbi(targets, target_lengths)
        output = self.ffnn(embed)
        return output, target_lengths
