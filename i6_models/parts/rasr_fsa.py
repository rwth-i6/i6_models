from __future__ import annotations

__all__ = ["RasrFsaBuilder", "WeightedFsa"]

from functools import reduce
from typing import Iterable, NamedTuple, Tuple, Union

import numpy as np
import torch


class WeightedFsa(NamedTuple):
    """
    Convenience class that represents an FSA. It supports scaling the weights of the
    fsa by simple left-multiplication and moving the tensors to a different device.
    It can simply be passed to :func:`i6_native_ops.fbw.fbw_loss` and :func:`i6_native_ops.fast_viterbi.align_viterbi`.

    :param num_states: the total number of all states S
    :param edges: a [4, E] tensor of edges with number of edges E and where each column is an edge
        consisting of from-state, to-state, emission idx and the index of the sequence it belongs to
    :param weights: a [E,] tensor of weights for each edge scaled by the tdp_scale
    :param start_end_states: a [N, 2] tensor of start and end states for each of the N sequences
    """

    num_states: torch.IntTensor
    edges: torch.IntTensor
    weights: torch.FloatTensor
    start_end_states: torch.IntTensor

    def __mul__(self, scale: float) -> WeightedFsa:
        """Multiply the weights, i.e. the third element, with a scale."""
        return WeightedFsa(
            self.num_states,
            self.edges,
            self.weights * scale,
            self.start_end_states,
        )

    def to(self, device: Union[str, torch.device]) -> WeightedFsa:
        """Move the tensors to a given device. This wraps around the
        PyTorch `Tensor.to(device)` method."""
        return WeightedFsa._make(tensor.to(device) for tensor in self)


class RasrFsaBuilder:
    """
    Builder class that wraps around the librasr.AllophoneStateFsaBuilder,
    bringing the FSAs into the correct format for the `i6_native_ops.fbw.fbw_loss`.
    Use of this class requires a working installation of the python package `librasr`.
    Hence, the package is locally imported in case other classes are accessed from
    this module.
    This class provides an explicit implementation of the `__getstate__` and `__setstate__`
    functions, necessary for pickling as the C++-class `librasr.AllophoneStateFsaBuilder`
    is not picklable.

    :param config_path: path to the RASR fsa exporter config
    :param tdp_scale: multiply the weights by this scale
    """

    def __init__(self, config_path: str, tdp_scale: float = 1.0):
        import librasr

        self.config_path = config_path
        config = librasr.Configuration()
        config.set_from_file(self.config_path)
        self.builder = librasr.AllophoneStateFsaBuilder(config)
        self.tdp_scale = tdp_scale

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["builder"]
        return state

    def __setstate__(self, state):
        import librasr

        self.__dict__.update(state)
        config = librasr.Configuration()
        config.set_from_file(self.config_path)
        self.builder = librasr.AllophoneStateFsaBuilder(config)

    def build_single(self, seq_tag: str) -> Tuple[int, int, np.ndarray, np.ndarray]:
        """
        Build the FSA for the given sequence tag in the corpus.

        :param seq_tag: sequence tag
        :return: FSA as a tuple containing
            * number of states S
            * number of edges E
            * integer edge array of shape [E, 3] where each row is an edge
                consisting of from-state, to-state and the emission idx
            * float weight array of shape [E,]
        """
        raw_fsa = self.builder.build_by_segment_name(seq_tag)
        return raw_fsa

    def build_batch(self, seq_tags: Iterable[str]) -> WeightedFsa:
        """
        Build and concatenate the FSAs for a batch of sequence tags
        and reformat as an input to `i6_native_ops.fbw.fbw_loss`.
        Here the FSAs are concatenated to a long FSA with multiple start and
        end states corresponding to each single FSA. For the concatenation,
        the state IDs of each single FSA are incrememented and made unique in
        the batch.
        Additionally we apply an optional scale to the weights.

        :param seq_tags: an iterable object of sequence tags
        :return: a concatenated FSA
        """

        def append_fsa(a, b):
            edges = torch.from_numpy(np.int32(b[2])).reshape((3, b[1]))
            return (
                a[0] + [b[0]],  # num states
                a[1] + [b[1]],  # num edges
                torch.hstack([a[2], edges]),  # edges
                torch.cat([a[3], torch.from_numpy(b[3])]),  # weights
            )

        # concatenate all FSAs in the batch into a single one where state ids are not yet unique
        fsas = map(self.build_single, seq_tags)
        empty_fsa = ([], [], torch.empty((3, 0), dtype=torch.int32), torch.empty((0,)))
        num_states, num_edges, all_edges, all_weights = reduce(append_fsa, fsas, empty_fsa)
        num_edges = torch.tensor(num_edges, dtype=torch.int32)
        num_states = torch.tensor(num_states, dtype=torch.int32)

        # accumulate number of states for each single fsa in order to determine start and end states
        # and make states in edge tensor unique to each sequence
        cum_num_states = torch.cumsum(num_states, dim=0, dtype=torch.int32)
        state_offsets = torch.cat([torch.zeros((1,), dtype=torch.int32), cum_num_states[:-1]])
        start_end_states = torch.vstack([state_offsets, cum_num_states - 1])

        # add unique sequence ids to the edge tensor and add start states to the states
        # in order to make them unique
        edge_seq_idxs = torch.repeat_interleave(num_edges)
        all_edges[:2, :] += torch.repeat_interleave(state_offsets, num_edges)
        all_edges = torch.vstack([all_edges, edge_seq_idxs])

        out_fsa = WeightedFsa(
            cum_num_states[-1],
            all_edges,
            all_weights,
            start_end_states,
        )

        if self.tdp_scale != 1.0:
            out_fsa *= self.tdp_scale

        return out_fsa
