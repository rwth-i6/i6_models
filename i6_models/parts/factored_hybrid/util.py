from enum import Enum
from typing import Callable, Union

from torch import nn


class BoundaryClassV1(Enum):
    """Phoneme state class augmentation selector"""

    none = 1
    word_end = 2
    boundary = 4

    def factor(self):
        return self.value

    @classmethod
    def from_flags(cls, use_word_end_classes: bool, use_boundary_classes: bool) -> "BoundaryClassV1":
        assert not (use_word_end_classes and use_boundary_classes), "cannot use both classes"

        if use_boundary_classes:
            return cls.boundary
        elif use_word_end_classes:
            return cls.word_end
        else:
            return cls.none

    @classmethod
    def from_dense_label_info(cls, li: "i6_core.mm.context_label.DenseLabelInfo") -> "BoundaryClassV1":
        return cls.from_flags(
            use_word_end_classes=li.use_word_end_classes,
            use_boundary_classes=li.use_boundary_classes,
        )


def get_center_dim(
    n_contexts: int,
    num_hmm_states_per_phone: int,
    ph_class: Union[int, BoundaryClassV1],
) -> int:
    """
    :return: number of center phonemes given the augmentation values
    """

    factor = ph_class.factor() if isinstance(ph_class, BoundaryClassV1) else ph_class
    return n_contexts * num_hmm_states_per_phone * factor


def get_mlp(
    num_input: int,
    num_output: int,
    hidden_dim: int,
    dropout: float,
    activation: Callable[[], nn.Module],
    num_layers,
) -> nn.Module:
    """
    :return: a context-mixing MLP according to the specifications
    """

    assert num_input > 0
    assert num_output > 0
    assert num_layers > 0
    assert hidden_dim > 0
    assert 0.0 <= dropout <= 1.0

    return nn.Sequential(
        *[
            layer
            for in_dim in [
                num_input,
                *[hidden_dim for _ in range(num_layers - 1)],
            ]
            for layer in [
                nn.Linear(in_dim, hidden_dim),
                activation(),
                nn.Dropout(dropout),
            ]
        ],
        nn.Linear(hidden_dim, num_output),
    )
