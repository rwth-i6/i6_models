__all__ = ["LogUniformSampler"]


import math

import torch
from torch import nn


class LogUniformSampler(nn.Module):
    def __init__(self, num_classes: int, *, device: Optional[torch.device] = None):
        super().__init__()

        # assumes count-sorted vocabulary, descending
        self.num_classes = num_classes

        # approximately zipf distribution
        ws = torch.arange(self.num_classes, dtype=torch.get_default_dtype(), device=device)
        self._distribution = (torch.log1p(ws + 1) - torch.log1p(ws)) / torch.log1p(torch.tensor(self.num_classes))
        self._distribution.clamp_(min=1e-10)
        self._distribution /= self._distribution.sum()

        self._cat_sampler = torch.distributions.categorical.Categorical(probs=self._distribution)

    def sample(self, num_samples):
        return self._cat_sampler.sample(torch.Size([num_samples]))

    def log_prob(self, indices):
        return self._cat_sampler.log_prob(indices)
