__all__ = [
    "State",
    "ModuleWithState",
    "DummyState",
    "make_attn_mask",
    "make_kv_attn_mask",
]

from abc import abstractmethod
from inspect import signature
from typing import Generic, Protocol, Sequence, Tuple, TypeVar

from torch import Tensor, nn
import torch


State = TypeVar("S")


class ModuleWithState(Protocol, Generic[State]):
    """An nn.Module with steppable decoder state."""

    @abstractmethod
    def get_initial_state(self) -> State:
        """Returns initial module state."""
        raise NotImplementedError

    @abstractmethod
    def transform_encoder_output(self, enc_out: Tensor, enc_out_len: Tensor, state: State) -> State:
        """
        Given the initial state, updates it with output from the encoder.

        :param enc_out: encoder output, shape (B..., T, F)
        :param enc_out_len: length of the seqs in enc_out, shape (B...,).
        :param state: initial module state obtained from `get_initial_state()`
        :return: updated state
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, labels: Tensor, labels_len: Tensor, state: State) -> Tuple[Tensor, State]:
        """
        Given model state and previous output, computes new output and new state.

        :param labels: previous output, shape (B..., T, F)
        :param labels_len: lengths of the labels, shape (B...,)
        :param state: recurrent module state
        :return: module output and new recurrent state
        """
        raise NotImplementedError


class DummyState(nn.Module, ModuleWithState[None]):
    """
    Wrapper module to make modules that do not actually require any running state
    compatible w/ the ModuleWithState interface.
    """

    def __init__(self, mod: nn.Module):
        super().__init__()

        self.module = mod

        # decide whether to pass len/mask param
        self.num_fwd_params = len(signature(mod.forward).parameters)
        assert self.num_fwd_params in [1, 2], "can only support forwards with one or two parameters"

    def get_initial_state(self):
        return None

    def transform_encoder_output(self, enc_out, enc_out_len, state):
        return state

    def forward(self, labels: Tensor, labels_len: Tensor, state: State) -> Tuple[Tensor, State]:
        if self.num_fwd_params == 1:
            out = self.module.forward(labels)
        else:
            out = self.module.forward(labels, labels_len)
        return out, state


def make_attn_mask(features: Tensor, features_lens: Tensor, additional_dims: Sequence[int] = ()) -> Tensor:
    """
    Creates a mask compatible w/ torchs SDPA implementation.

    :param features: data tensor of shape (B..., T, F)
    :param features_lens: length tensor of Shape (B...,)
    :param additional_dims: additional broadcastable dims to introduce into the mask,
        for e.g. attention heads.
    :return: additive attention mask of shape (B..., T) with additional dims unsqueezed in
    """
    len_mask = torch.arange(features.shape[-2], device=features_lens.device) < features_lens.unsqueeze(-1)  # B... L
    len_mask_for_attn = torch.where(len_mask, 0.0, float("-inf"))  # B... L
    for dim in additional_dims:
        len_mask_for_attn = len_mask_for_attn.unsqueeze(dim)
    return len_mask_for_attn


def make_kv_attn_mask(features: Tensor, features_lens: Tensor) -> Tensor:
    """
    Creates a mask compatible w/ torchs SDPA implementation for masking out padding
    in the keys/values.

    :param features: data tensor of shape (B..., T, F)
    :param features_lens: length tensor of Shape (B...,)
    :return: additive attention mask of shape (B..., 1, 1, T)
    """
    return make_attn_mask(features, features_lens, additional_dims=(-2, -2))
