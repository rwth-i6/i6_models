__all__ = ["LogUniformSampler"]


import math

import torch
from torch import nn


class LogUniformSampler(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # assumes count-sorted vocabulary, descending
        self.num_classes = num_classes

        # approximately zipf distribution
        self._distribution = [
            (math.log1p(w + 1) - math.log1p(w)) / math.log1p(self.num_classes) for w in range(self.num_classes)
        ]
        self._distribution = torch.tensor(self._distribution).clamp(min=1e-10)
        self._distribution /= self._distribution.sum()

        self._cat_sampler = torch.distributions.categorical.Categorical(probs=self._distribution.cuda())

    def sample(self, num_samples):
        return self._cat_sampler.sample(torch.Size([num_samples]))

    def log_prob(self, indices):
        return self._cat_sampler.log_prob(indices)
