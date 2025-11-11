from __future__ import annotations

__all__ = ["RasrFsaBuilder", "WeightedFsa", "RasrFsaBuilderV2", "WeightedFsaV2"]

from abc import ABC, abstractmethod
from functools import reduce
from typing import Any, Iterable, NamedTuple, Union

import numpy as np
import torch


FsaTuple = tuple[int, int, np.ndarray, np.ndarray]
"""
FSA as a tuple containing
* number of states S
* number of edges E
* integer edge array of shape [E, 3] where each row is an edge
    consisting of from-state, to-state and the emission idx
* float weight array of shape [E,]

This format is how RASR outputs FSAs when retrieving an FSA by orthography/sequence tag:
https://github.com/rwth-i6/rasr/blob/2bf347fb70f1298950a4adbda39197242f78a619/src/Python/AllophoneStateFsaBuilder.cc#L60
"""


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


class WeightedFsaV2(NamedTuple):
    """
    Convenience class that represents an FSA for the fbw2 loss from i6_native_ops. It supports scaling the weights of
    the FSA by simple left-multiplication and moving the tensors to a different device.
    It can be passed to :func:`i6_native_ops.fbw2.fbw2_loss`.

    :param num_states: the number of states per sequence (shape: [N])
    :param num_edges: the number of edges per sequence (shape: [N])
    :param edges: a [3, E] tensor of edges with number of edges E and where each column is an edge
        consisting of from-state, to-state and emission idx
    :param weights: a [E,] tensor of weights for each edge scaled by the tdp_scale
    :param start_end_states: a [N, 2] tensor of start and end states for each of the N sequences
    """

    num_states: torch.IntTensor
    num_edges: torch.IntTensor
    edges: torch.IntTensor
    weights: torch.FloatTensor
    start_end_states: torch.IntTensor

    def __mul__(self, scale: float) -> WeightedFsaV2:
        """Multiply the weights, i.e. the forth element, with a scale."""
        return WeightedFsaV2(
            self.num_states,
            self.num_edges,
            self.edges,
            self.weights * scale,
            self.start_end_states,
        )

    def to(self, device: Union[str, torch.device]) -> WeightedFsaV2:
        """
        Move the tensors that can be on device to a given device.
        Some Tensors (num_states/num_edges) are expected to be in CPU memory.
        """

        return WeightedFsaV2(
            self.num_states.to("cpu"),
            self.num_edges.to("cpu"),
            self.edges.to(device),
            self.weights.to(device),
            self.start_end_states.to(device),
        )


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

    def build_single(self, seq_tag: str) -> FsaTuple:
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


class RasrFsaBatchBuilder(ABC):
    """
    Abstract base class for building an FSA, compatible with the `fbw2` op from `i6_native_ops`.

    The user must overwrite the :funcref:`build_single` method.

    The TDP scale must be explicitly called when running :funcref:`build_single`.
    For that, :funcref:`apply_tdp_scale` can be used.
    """

    def __init__(self, config_path: str, tdp_scale: float = 1.0):
        """
        :param config_path: Path to the RASR config for the FSA builder.
        :param tdp_scale: Transition penalty to apply for each edge of the resulting FSA.
            Applied in the :funcref:`build_batch` method.
        """
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

    def apply_tdp_scale(self, fsa: FsaTuple, tdp_scale: float) -> FsaTuple:
        """
        Applies the TDP scale given as parameter to an FSA represented as a tuple.

        :param fsa: FSA as a tuple containing
            * number of states S
            * number of edges E
            * integer edge array of shape [E, 3] where each row is an edge
                consisting of from-state, to-state and the emission idx
            * float weight array of shape [E,]
        :param tdp_scale: TDP scale by which the weights must be multiplied.
        :return: FSA with scaled weights corresponding to :paramref:`tdp_scale`.
        """
        return (fsa[0], fsa[1], fsa[2], fsa[3] * tdp_scale)

    @abstractmethod
    def build_single(self, single_identifier: Any) -> FsaTuple:
        """
        Builds a single FSA. The user must overwrite this function.

        Note: the user must take care of applying the TDP scale.
        For that, :funcref:`apply_tdp_scale` can be used.

        :param single_identifier: Identifier of the sequence for which an FSA must be built.
        :return: FSA as a tuple corresponding to the sequence identified by :paramref:`single_identifier`.
            The returned value contains the following fields in order:
            * number of states S
            * number of edges E
            * integer edge array of shape [E, 3] where each row is an edge
                consisting of from-state, to-state and the emission idx
            * float weight array of shape [E,]
        """
        ...

    def join_fsas(self, fsas: Iterable[FsaTuple]) -> WeightedFsaV2:
        """
        Joins a set of FSAs represented as tuples into a single :classref:`WeightedFsaV2` object.

        :param fsas: Tuples to be concatenated, represented as tuples with the following fields:
            * number of states S
            * number of edges E
            * integer edge array of shape [E, 3] where each row is an edge
                consisting of from-state, to-state and the emission idx
            * float weight array of shape [E,]
        :return: Single FSA object corresponding to the joined FSAs passed as parameter.
        """

        num_states = [f[0] for f in fsas]
        num_edges = [f[1] for f in fsas]
        start_states = np.cumsum([0] + num_states)[:-1]
        end_states = np.cumsum(num_states) - 1
        weights = np.concatenate(tuple(f[3] for f in fsas))

        edges = []
        for idx, f in enumerate(fsas):
            f_edges = f[2].reshape(3, -1).copy()
            f_edges[:2, :] += start_states[idx]
            edges.append(f_edges)

        out_fsa = WeightedFsaV2(
            torch.IntTensor(num_states).to(torch.uint32),
            torch.IntTensor(num_edges).to(torch.uint32),
            torch.IntTensor(np.concatenate(edges, axis=1)).contiguous(),
            torch.Tensor(weights),
            torch.IntTensor(np.array([start_states, end_states])),
        )

        return out_fsa

    def build_batch(self, multiple_identifiers: Iterable[Any]) -> WeightedFsaV2:
        """
        Builds and joins several FSAs into a batch.

        :param multiple_identifiers: Sequence of identifiers for which to build the batched FSA.
        :return: Single FSA object corresponding to the batched FSAs passed as parameter.
        """
        fsas = list(map(self.build_single, multiple_identifiers))

        return self.join_fsas(fsas)


class RasrFsaBuilderV2(RasrFsaBatchBuilder):
    """
    Builds an FSA by sequence tag. The implementation is compatible with the `fbw2` op from `i6_native_ops`.
    """

    def build_single(self, single_identifier: str) -> WeightedFsaV2:
        """
        Build the FSA for the given sequence tag in the corpus.

        :param single_identifier: sequence tag
        :return: FSA as a tuple containing
            * number of states S
            * number of edges E
            * integer edge array of shape [E, 3] where each row is an edge
                consisting of from-state, to-state and the emission idx
            * float weight array of shape [E,]
        """
        raw_fsa = self.builder.build_by_segment_name(single_identifier)
        return raw_fsa

