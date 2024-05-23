import torch
import torch.nn.functional as F
from torch import nn

from i6_models.parts.frontend.common import mask_pool, get_same_padding, apply_same_padding

def test_masking():
    # tensor as batch with different sequence lengths
    T = 2
    kernel_size = 3
    stride = 2
    padding = get_same_padding(kernel_size)

    # what does conv to one sequence in [B, F, T] format
    B, F = (1, 1)
    x = torch.ones((B, F, T))
    pad = lambda x: x
    conv = nn.Conv1d(
        in_channels=F,
        out_channels=F,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )
    out = conv(x)
    out_len = out.shape[-1]

    # mask for this sequence length in a batch of max length = 100
    batch_T = 100
    idx = T - 1 # sequence at this index has length T by following construction
    in_mask = torch.tensor(
        [[True] * t + [False] * (batch_T - t) for t in range(1, batch_T + 1)]
    )

    in_mask_len = len(torch.where(in_mask[idx, :])[0])
    assert in_mask_len == T

    out_mask = mask_pool(
        in_mask,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    )[idx, :]

    # we expect True for the length of the sequence and False otherwise
    mask_len = len(torch.where(out_mask)[0])
    assert out_len == mask_len, f"Actual out length of the sequence {out_len=}" \
        + f" and the length of the mask {mask_len=} are not equal where " \
        + f" {out=} and {out_mask=}."

