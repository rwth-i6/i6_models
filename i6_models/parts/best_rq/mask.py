from typing import Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

__all__ = ["RandomMask"]


class RandomMask(nn.Module):
    """
    randomly mask out consecutive frames time dimension, the masked frames can be either
    replaced with zeros or with learnable embeddings.
    simplified version from Fairseq compute_mask_indices function,
    C.f. https://github.com/facebookresearch/fairseq/blob/ecbf110e1eb43861214b05fa001eff584954f65a/fairseq/data/data_utils.py#L399
    """

    def __init__(
        self,
        input_dim: int,
        mask_replace_val: str,
        mask_percentage: float,
        mask_length: int,
    ):
        """
        :param input_dim: number of feature dimension of input
        :param mask_replace_val: the way to replace masked frames, either with zeros or lernable embeddings
        :param mask_percentage: percentage of frames to be masked out
        :param mask_length: the length of each mask span
        """
        super().__init__()

        assert mask_replace_val in ["lernable", "zero"], "not implemented yet"
        if mask_replace_val == "lernable":
            self.mask_emb = nn.Parameter(torch.FloatTensor(input_dim).uniform_())
        elif mask_replace_val == "zero":
            self.mask_emb = torch.zeros(input_dim)
        self.mask_percentage = mask_percentage
        self.mask_length = mask_length

    def forward(
        self,
        tensor: torch.tensor,
        padding_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
                self.mask_percentage * seq_len / float(self.mask_length) + np.random.rand()
            )

            min_len = self.mask_length
            if seq_len - min_len <= num_mask:
                min_len = seq_len - num_mask - 1
            mask_idc = np.random.choice(seq_len - min_len, num_mask, replace=False)

            for j in mask_idc:
                mask[i, j : j + self.mask_length] = True

        tensor[mask] = self.mask_emb.to(tensor.device)

        return tensor, torch.tensor(mask).to(tensor.device)
