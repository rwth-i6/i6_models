import torch
import torch.nn as nn
from typing import Optional
import numpy as np


class RandomMask(nn.Module):
    def __init__(self, input_dim, mask_replace_val):

        if mask_replace_val == "lernable":
            self.mask_emb = nn.Parameter(torch.FloatTensor(input_dim).uniform_())
        elif mask_replace_val == 0:
            self.mask_emb = torch.zeros(input_dim)

    def forward(
        self,
        tensor: torch.tensor,
        padding_mask: Optional[torch.Tensor],
        mask_prob: float,
        mask_length: int,
        min_masks: int = 0,
    ):
        ndim_batch, ndim_time, _ = tensor.size()

        mask = torch.zeros((ndim_batch, ndim_time), dtype=torch.bool)

        mask_idcs = []
        for i in range(ndim_batch):
            if padding_mask is not None:
                seq_len = ndim_time - padding_mask[i].long().sum().item()
                assert seq_len >= 0
            else:
                seq_len = ndim_time

            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * seq_len / float(mask_length)
                + np.random.rand()
            )
            num_mask = max(min_masks, num_mask)

            min_len = mask_length
            if seq_len - min_len <= num_mask:
                min_len = seq_len - num_mask - 1
            mask_idc = np.random.choice(seq_len - min_len, num_mask, replace=False)

            mask_idc = np.asarray([mask_idc[j] + mask_length for j in range(len(mask_idc))])
        mask_idcs.append(mask_idc)

        for i, mask_idc in enumerate(mask_idcs):
            mask[i, mask_idc] = True

        tensor[mask] = self.mask_emb

        return tensor
