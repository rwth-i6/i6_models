from typing import Tuple, Union

import torch
from torch import nn


IntTupleIntType = Union[int, Tuple[int, int]]


def get_same_padding(input_size: Union[int, Tuple[int, ...]]) -> Union[int, Tuple[int, ...]]:
    """
    get padding in order to not reduce the time dimension

    :param input_size:
    :return:
    """
    if isinstance(input_size, int):
        return (input_size - 1) // 2
    elif isinstance(input_size, tuple):
        return tuple((s - 1) // 2 for s in input_size)
    else:
        raise TypeError(f"unexpected size type {type(input_size)}")


def mask_pool(seq_mask: torch.Tensor, kernel_size: int, stride: int, padding: int) -> torch.Tensor:
    """
    :param seq_mask: [B,T]
    :param kernel_size:
    :param stride:
    :param padding:
    :return: [B,T'] using maxpool
    """
    seq_mask = seq_mask.float()
    seq_mask = torch.unsqueeze(seq_mask, 1)  # [B,1,T]
    seq_mask = nn.functional.max_pool1d(seq_mask, kernel_size, stride, padding)  # [B,1,T']
    seq_mask = torch.squeeze(seq_mask, 1)  # [B,T']
    seq_mask = seq_mask.bool()
    return seq_mask


def get_int_tuple_int(variable: IntTupleIntType, index: int) -> int:
    return variable[index] if isinstance(variable, tuple) else variable
