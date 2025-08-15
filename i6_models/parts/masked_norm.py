__all__ = ["MaskedBatchNorm1dV1"]

from typing import Optional

import torch
from torch import nn, Tensor


def _lengths_to_mask(lengths: Tensor, max_len: Optional[int] = None, dtype=None) -> Tensor:
    """
    Converts a "lengths" tensor to its binary mask representation.
    """
    max_len = max_len or lengths.max().item()
    mask = torch.arange(max_len, device=lengths.device, dtype=lengths.dtype).expand(
        len(lengths), max_len
    ) < lengths.unsqueeze(-1)
    if dtype is not None:
        mask = mask.to(dtype=dtype, device=lengths.device)
    return mask


class MaskedBatchNorm1dV1(nn.BatchNorm1d):
    """
    1D Batch normalization that supports ignoring the padding during statistics collection.

    Same construction arguments as pytorch's `nn.BatchNorm1d`.
    """

    def forward(self, inp: Tensor, lengths_or_mask: Tensor):
        """
        Applies batch norm to `inp`, masking away the padding given by `lengths_or_mask`.

        :param inp: data to normalize, shape [B...,F,T]
        :param lengths_or_mask: seq length tensor if shape [B...,],
            or mask tensor if the shape is [B...,T].
        """

        self._check_input_dim(inp)

        exponential_average_factor = 0.0
        if self.training and self.track_running_stats and self.num_batches_tracked is not None:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / float(self.num_batches_tracked)
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        assert inp.ndim - 3 < lengths_or_mask.ndim < inp.ndim, (
            f"mask ndim ({lengths_or_mask.ndim}) should be between {inp.ndim - 3} and {inp.ndim}"
        )
        if lengths_or_mask.ndim == inp.ndim - 1:
            mask = lengths_or_mask.to(dtype=torch.float, device=inp.device)
        elif lengths_or_mask.ndim == inp.ndim - 2:
            mask = _lengths_to_mask(lengths_or_mask, max_len=inp.shape[-1], dtype=inp.dtype)
        else:
            raise ValueError(
                f"length tensor shape mismatch {lengths_or_mask.shape} wrt. input tensor shape {inp.shape}"
            )
        assert mask.ndim == 2

        n = mask.sum()
        if n == 0:
            return inp

        # we use the mask to calculate the mean
        mask = mask / n
        mask = mask.unsqueeze(-2)

        reduce_dims = list(range(inp.ndim - 2)) + [inp.ndim - 1]
        if not self.track_running_stats:
            mean = (mask * inp).sum(reduce_dims)
            var = ((mask * inp**2).sum(reduce_dims) - mean**2) * n / (n - 1)
        elif self.training:
            mean = (mask * inp).sum(reduce_dims)
            # Var(X) = E[X^2] - E[X]^2
            var = (mask * inp**2).sum(reduce_dims) - mean**2

            with torch.no_grad():
                self.running_mean = (
                    exponential_average_factor * mean + (1 - exponential_average_factor) * self.running_mean
                )
                # torch updates running statistics with unbiased var
                self.running_var = (
                    exponential_average_factor * var * n / (n - 1) + (1 - exponential_average_factor) * self.running_var
                )
        else:
            mean = self.running_mean
            var = self.running_var

        inp = (inp - mean[None, :, None]) * torch.rsqrt(var[None, :, None] + self.eps)
        if self.affine:
            inp = inp * self.weight[None, :, None] + self.bias[None, :, None]

        return inp
