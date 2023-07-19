from typing import Tuple, Union

import torch
from torch import nn
from torch.nn import functional


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


def mask_pool(seq_mask: torch.Tensor, *, kernel_size: int, stride: int, padding: int) -> torch.Tensor:
    """
    apply strides to the masking

    :param seq_mask: [B,T]
    :param kernel_size:
    :param stride:
    :param padding:
    :return: [B,T'] using maxpool
    """
    if stride == 1 and 2 * padding == kernel_size - 1:
        return seq_mask

    seq_mask = seq_mask.float()
    seq_mask = torch.unsqueeze(seq_mask, 1)  # [B,1,T]
    seq_mask = nn.functional.max_pool1d(seq_mask, kernel_size, stride, padding)  # [B,1,T']
    seq_mask = torch.squeeze(seq_mask, 1)  # [B,T']
    seq_mask = seq_mask.bool()
    return seq_mask


def calculate_output_dim(in_dim: int, *, filter_size: int, stride: int, padding: int) -> int:
    def ceildiv(a: int, b: int):
        return -(-a // b)

    return ceildiv(in_dim + 2 * padding - (filter_size - 1) * 1, stride)
