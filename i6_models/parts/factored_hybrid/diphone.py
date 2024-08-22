__all__ = [
    "FactoredDiphoneBlockV1Config",
    "FactoredDiphoneBlockV1",
    "FactoredDiphoneBlockV2Config",
    "FactoredDiphoneBlockV2",
]

from dataclasses import dataclass
from typing import Callable, Tuple, Union

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from i6_models.config import ModelConfiguration

from .util import BoundaryClassV1, get_center_dim, get_mlp


@dataclass
class FactoredDiphoneBlockV1Config(ModelConfiguration):
    """
    Attributes:
        num_contexts: the number of raw phonemes/acoustic contexts
        num_hmm_states_per_phone: the number of HMM states per phoneme
        boundary_class: the phoneme state augmentation to apply

        activation: activation function to use in the context mixing MLP.
        context_mix_mlp_dim: inner dimension of the context mixing MLP layers
        context_mix_mlp_num_layers: how many hidden layers on the MLPs there should be
        left_context_embedding_dim: embedding dimension of the left context
            values. Good choice is in the order of num_contexts.

        dropout: dropout probabilty
        num_inputs: input dimension of the output block, must match w/ output dimension
            of main encoder (e.g. Conformer)
    """

    num_contexts: int
    num_hmm_states_per_phone: int
    boundary_class: Union[int, BoundaryClassV1]

    activation: Callable[[], nn.Module]
    context_mix_mlp_dim: int
    context_mix_mlp_num_layers: int
    left_context_embedding_dim: int

    dropout: float
    num_inputs: int

    def __post_init__(self):
        super().__post_init__()

        assert self.num_contexts > 0
        assert self.num_hmm_states_per_phone > 0

        assert self.context_mix_mlp_dim > 0
        assert self.context_mix_mlp_num_layers > 0
        assert self.left_context_embedding_dim > 0

        assert self.num_inputs > 0
        assert 0.0 <= self.dropout <= 1.0, "dropout must be a probability"


class FactoredDiphoneBlockV1(nn.Module):
    """
    Diphone FH model output block.

    Consumes the output h(x) of a main encoder model and computes factored or joint
    output logits/probabilities for p(c|l,h(x)) and p(l|h(x)).
    """

    def __init__(self, cfg: FactoredDiphoneBlockV1Config):
        super().__init__()

        self.boundary_class = cfg.boundary_class
        self.num_contexts = cfg.num_contexts
        self.num_hmm_states_per_phone = cfg.num_hmm_states_per_phone

        self.num_center = get_center_dim(self.num_contexts, self.num_hmm_states_per_phone, self.boundary_class)
        self.num_diphone = self.num_center * self.num_contexts

        self.left_context_encoder = get_mlp(
            num_input=cfg.num_inputs,
            num_output=cfg.num_contexts,
            hidden_dim=cfg.context_mix_mlp_dim,
            num_layers=cfg.context_mix_mlp_num_layers,
            dropout=cfg.dropout,
            activation=cfg.activation,
        )
        self.left_context_embedding = nn.Embedding(cfg.num_contexts, cfg.left_context_embedding_dim)
        self.center_encoder = get_mlp(
            num_input=cfg.num_inputs + cfg.left_context_embedding_dim,
            num_output=self.num_center,
            hidden_dim=cfg.context_mix_mlp_dim,
            num_layers=cfg.context_mix_mlp_num_layers,
            dropout=cfg.dropout,
            activation=cfg.activation,
        )

    def forward(self, *args, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        return self.forward_factored(*args, **kwargs)

    def forward_factored(
        self,
        features: Tensor,  # B, T, F
        contexts_left: Tensor,  # B, T
        **kwargs,  # kwargs because the train_step function passes contexts_center for right context training
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        :param features: Main encoder output. shape B, T, F. F=num_inputs
        :param contexts_left: The left contexts used to compute p(c|l,x), shape B, T.
            Valid values range from [0, num_contexts).
        :return: tuple of logits for p(c|l,x), p(l|x) and the embedded left context values.
        """

        logits_left = self.left_context_encoder(features)  # B, T, C
        # in training we forward exactly one context per T, so: B, T, E
        contexts_embedded_left = self.left_context_embedding(contexts_left)
        features_center = torch.cat((features, contexts_embedded_left), -1)  # B, T, F+E
        logits_center = self.center_encoder(features_center)  # B, T, C

        return logits_center, logits_left, contexts_embedded_left

    def forward_joint(self, features: Tensor) -> Tensor:
        """
        :param features: Main encoder output. shape B, T, F. F=num_inputs
        :return: log probabilities for p(c,l|x).
        """

        logits_left = self.left_context_encoder(features)  # B, T, C

        # here we forward every context to compute p(c, l|x) = p(c|l, x) * p(l|x)
        contexts_left = torch.arange(self.num_contexts, device=features.device)  # C
        contexts_embedded_left = self.left_context_embedding(contexts_left)  # C, E

        features_expanded = features.expand((self.num_contexts, -1, -1, -1))  # C, B, T, F
        contexts_embedded_left_ = contexts_embedded_left.reshape((self.num_contexts, 1, 1, -1)).expand(
            (-1, features.shape[0], features.shape[1], -1)
        )  # C, B, T, E
        features_center = torch.cat((features_expanded, contexts_embedded_left_), dim=-1)  # C, B, T, F+E
        logits_center = self.center_encoder(features_center)  # C, B, T, F'
        log_probs_center = F.log_softmax(logits_center, -1)
        log_probs_center = log_probs_center.permute((1, 2, 3, 0))  # B, T, F', C
        log_probs_left = F.log_softmax(logits_left, -1)
        log_probs_left = log_probs_left.unsqueeze(-2)  # B, T, 1, C

        joint_log_probs = log_probs_center + log_probs_left  # B, T, F', C
        joint_log_probs = joint_log_probs.reshape(
            (features.shape[0], features.shape[1], self.num_diphone)
        )  # B, T, F'*C

        return joint_log_probs


@dataclass
class FactoredDiphoneBlockV2Config(FactoredDiphoneBlockV1Config):
    """
    Attributes:
        Same attributes as parent class. In addition:

        center_state_embedding_dim: embedding dimension of the center state
            values. Good choice is in the order of num_center_states.
    """

    center_state_embedding_dim: int

    def __post_init__(self):
        super().__post_init__()

        assert self.center_state_embedding_dim > 0


class FactoredDiphoneBlockV2(FactoredDiphoneBlockV1):
    """
    Like FactoredDiphoneBlockV1, but computes an additional diphone output on the right context `p(r|c,x)`.

    This additional output is ignored when computing the joint, and only used in training.
    """

    def __init__(self, cfg: FactoredDiphoneBlockV2Config):
        super().__init__(cfg)

        self.center_state_embedding = nn.Embedding(self.num_center, cfg.center_state_embedding_dim)
        self.right_context_encoder = get_mlp(
            num_input=cfg.num_inputs + cfg.center_state_embedding_dim,
            num_output=self.num_contexts,
            hidden_dim=cfg.context_mix_mlp_dim,
            num_layers=cfg.context_mix_mlp_num_layers,
            dropout=cfg.dropout,
            activation=cfg.activation,
        )

    # update type definitions
    def forward(self, *args, **kwargs) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return super().forward(*args, **kwargs)

    def forward_factored(
        self,
        features: Tensor,  # B, T, F
        contexts_left: Tensor,  # B, T
        contexts_center: Tensor,  # B, T
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        :param features: Main encoder output. shape B, T, F. F=num_inputs
        :param contexts_left: The left contexts used to compute p(c|l,x), shape B, T.
            Valid values range from [0, num_contexts).
        :param contexts_center: The center states used to compute p(r|c,x), shape B, T.
            Given that the center state also contains the word-end class and HMM state ID, the valid values
            range from [0, num_center_states), where num_center_states >= num_contexts.
        :return: tuple of logits for p(c|l,x), p(l|x), p(r|c,x) and the embedded left context and center state values.
        """

        logits_center, logits_left, contexts_left_embedded = super().forward_factored(features, contexts_left)

        # in training we forward exactly one context per T, so: B, T, E
        center_states_embedded = self.center_state_embedding(contexts_center)
        features_right = torch.cat((features, center_states_embedded), -1)  # B, T, F+E
        logits_right = self.right_context_encoder(features_right)

        return logits_center, logits_left, logits_right, contexts_left_embedded, center_states_embedded
