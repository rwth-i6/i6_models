__all__ = [
    "DiphoneLogitsV1",
    "DiphoneProbsV1",
    "DiphoneBackendV1Config",
    "DiphoneBackendV1",
]

from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from i6_models.config import ModelConfiguration

from .util import PhonemeStateClassV1, get_center_dim, get_mlp


@dataclass
class DiphoneLogitsV1:
    """outputs of a diphone factored hybrid model"""

    embeddings_left_context: Tensor
    """the embedded left context values"""

    output_center: Tensor
    """center output"""

    output_left: Tensor
    """left output"""


class DiphoneProbsV1(DiphoneLogitsV1):
    """marker class indicating that the output tensors store probabilities and not logits"""


@dataclass
class DiphoneBackendV1Config(ModelConfiguration):
    """
    Attributes:
        context_mix_mlp_dim: inner dimension of the context mixing MLP layers
        context_mix_mlp_num_layers: how many hidden layers on the MLPs there should be
        dropout: dropout probabilty
        left_context_embedding_dim: embedding dimension of the left context
            values. Good choice is in the order of n_contexts.
        n_contexts: the number of raw phonemes/acoustic contexts
        num_hmm_states_per_phone: the number of HMM states per phoneme
        num_inputs: input dimension of the backend, must match w/ output dimension
            of main encoder (e.g. Conformer)
        phoneme_state_class: the phoneme state augmentation to apply
        activation: activation function to use in the context mixing MLP.
    """

    left_context_embedding_dim: int
    n_contexts: int
    num_hmm_states_per_phone: int
    phoneme_state_class: Union[int, PhonemeStateClassV1]

    activation: Callable[[], nn.Module]
    context_mix_mlp_dim: int
    context_mix_mlp_num_layers: int
    dropout: float
    num_inputs: int

    def __post_init__(self) -> None:
        super().__post_init__()

        assert self.context_mix_mlp_dim > 0
        assert self.context_mix_mlp_num_layers > 0
        assert self.n_contexts > 0
        assert self.num_hmm_states_per_phone > 0
        assert self.num_inputs > 0
        assert self.left_context_embedding_dim > 0
        assert 0.0 <= self.dropout <= 1.0, "dropout must be a probability"


class DiphoneBackendV1(nn.Module):
    """
    Diphone FH model backend.

    Consumes the output h(x) of a main encoder model and computes factored output
    logits/probabilities for p(c|l,x) and p(l|x).
    """

    def __init__(self, cfg: DiphoneBackendV1Config):
        super().__init__()

        self.n_contexts = cfg.n_contexts

        self.left_context_encoder = get_mlp(
            num_input=cfg.num_inputs,
            num_output=cfg.n_contexts,
            hidden_dim=cfg.context_mix_mlp_dim,
            num_layers=cfg.context_mix_mlp_num_layers,
            dropout=cfg.dropout,
            activation=cfg.activation,
        )
        self.left_context_embedding = nn.Embedding(cfg.n_contexts, cfg.left_context_embedding_dim)
        self.center_encoder = get_mlp(
            num_input=cfg.num_inputs + cfg.left_context_embedding_dim,
            num_output=get_center_dim(cfg.n_contexts, cfg.num_hmm_states_per_phone, cfg.phoneme_state_class),
            hidden_dim=cfg.context_mix_mlp_dim,
            num_layers=cfg.context_mix_mlp_num_layers,
            dropout=cfg.dropout,
            activation=cfg.activation,
        )

    def forward(
        self,
        features: Tensor,  # B, T, F
        contexts_left: Optional[Tensor] = None,  # B, T
    ) -> Union[DiphoneLogitsV1, DiphoneProbsV1]:
        """
        :param features: Main encoder output. shape B, T, F. F=num_inputs
        :param contexts_left: The left contexts used to compute p(c|l,x).
            Shape during training: B, T. Must be `None` when forwarding as the model
            will compute a flat, joint output scoring all possible contexts in forwarding
            mode.
        :return: During training: logits for p(c|l,x) and p(l|x).
            During inference: *log* probs for p(c,l|x) in one large layer.
        """

        left_logits = self.left_context_encoder(features)  # B, T, C

        if self.training:
            assert contexts_left is not None
            assert contexts_left.ndim >= 2
        else:
            assert contexts_left is None, "in eval mode, all left contexts are forwarded at the same time"
            contexts_left = torch.arange(self.n_contexts)

        # train: B, T, E
        # eval: C, E
        embedded_left_contexts = self.left_context_embedding(contexts_left)

        if self.training:
            # in training we forward exactly one context per T

            center_features = torch.cat((features, embedded_left_contexts), -1)  # B, T, F+E
            center_logits = self.center_encoder(center_features)  # B, T, C

            return DiphoneLogitsV1(
                embeddings_left_context=embedded_left_contexts, output_left=left_logits, output_center=center_logits
            )
        else:
            # here we forward every context to compute p(c, l|x) = p(c|l, x) * p(l|x)

            features = features.expand((self.n_contexts, -1, -1, -1))  # C, B, T, F
            embedded_left_contexts_ = embedded_left_contexts.reshape((self.n_contexts, 1, 1, -1)).expand(
                (-1, *features.shape[1:3], -1)
            )  # C, B, T, E
            center_features = torch.cat((features, embedded_left_contexts_), -1)  # C, B, T, F+E
            center_logits = self.center_encoder(center_features)  # C, B, T, F'
            center_probs = F.log_softmax(center_logits, -1)
            center_probs = center_probs.permute((1, 2, 3, 0))  # B, T, F', C
            left_probs = F.log_softmax(left_logits, -1)
            left_probs = left_probs.unsqueeze(-2)  # B, T, 1, C

            joint_probs = center_probs + left_probs  # B, T, F', C
            joint_probs = torch.flatten(joint_probs, start_dim=2)  # B, T, joint

            return DiphoneProbsV1(
                embeddings_left_context=embedded_left_contexts,
                output_left=left_probs.squeeze(),
                output_center=joint_probs,
            )
