__all__ = [
    "FactoredTriphoneBlockV1Config",
    "FactoredTriphoneBlockV1",
]

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn, Tensor

from .diphone import FactoredDiphoneBlockV1, FactoredDiphoneBlockV2Config
from .util import get_mlp


@dataclass
class FactoredTriphoneBlockV1Config(FactoredDiphoneBlockV2Config):
    """
    Attributes:
        Same as the FactoredDiphoneBlockV2Config.
    """


class FactoredTriphoneBlockV1(FactoredDiphoneBlockV1):
    """
    Triphone FH model output block.

    Consumes the output h(x) of a main encoder model and computes factored logits/probabilities
    for p(c|l,h(x)), p(l|h(x)) and p(r|c,l,h(x)).
    """

    def __init__(self, cfg: FactoredTriphoneBlockV1Config):
        super().__init__(cfg)

        self.center_state_embedding = nn.Embedding(self.num_center, cfg.center_state_embedding_dim)
        self.right_context_encoder = get_mlp(
            num_input=cfg.num_inputs + cfg.center_state_embedding_dim + cfg.left_context_embedding_dim,
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
        :param contexts_center: The center states used to compute p(r|c,l,x), shape B, T.
            Given that the center state also contains the word-end class and HMM state ID, the valid values
            range from [0, num_center_states), where num_center_states >= num_contexts.
        :return: tuple of logits for p(c|l,x), p(l|x), p(r|c,l,x) and the embedded left context and center state values.
        """

        logits_center, logits_left, contexts_left_embedded = super().forward_factored(features, contexts_left)

        # This logic is very similar to FactoredDiphoneBlockV2.forward, but not the same.
        # This class computes `p(r|c,l,h(x))` while FactoredDiphoneBlockV2 computes `p(r|c,h(x))`.
        center_states_embedded = self.center_state_embedding(contexts_center)  # B, T, E'
        features_right = torch.cat((features, center_states_embedded, contexts_left_embedded), -1)  # B, T, F+E'+E
        logits_right = self.right_context_encoder(features_right)  # B, T, C

        return logits_center, logits_left, logits_right, contexts_left_embedded, center_states_embedded

    def forward_joint(self, features: Tensor) -> Tensor:
        raise NotImplementedError(
            "It is computationally infeasible to forward the full triphone joint, "
            "only the diphone joint can be computed via forward_joint_diphone."
        )
