__all__ = ["LogUniformSampler"]


import torch
from torch import nn
from typing import Optional


class LogUniformSampler(nn.Module):
    def __init__(
        self, num_classes: int, *, distribution_clamp_min: float = 1e-10, device: Optional[torch.device] = None
    ):
        """
        Samples from a log uniform distribution from classes. Sampling is performed with replacement, i.e. sampled
        indices can appear more than once in sampled set. This can be implemented with
        `torch.distributions.multinomial.Multinomial` or `torch.multinomial`.

        :param num_classes: number of classes from which the distribution is sampled. The class indices are sorted in
            descending order according to their frequency.
        :param device: device on which the distribution is sampled.
        """
        super().__init__()

        # assumes count-sorted vocabulary, descending
        self.num_classes = num_classes

        # approximately zipf distribution
        ws = torch.arange(self.num_classes, dtype=torch.get_default_dtype(), device=device)
        self._distribution = (torch.log1p(ws + 1) - torch.log1p(ws)) / torch.log1p(torch.tensor(self.num_classes))
        self._distribution.clamp_(min=distribution_clamp_min)
        self._distribution /= self._distribution.sum()

        self._cat_sampler = torch.distributions.categorical.Categorical(probs=self._distribution)

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Returns a random tensor in the size of [num_samples].

        :param num_samples: number of samples.
        :return: [num_samples]
        """
        return self._cat_sampler.sample(torch.Size([num_samples]))

    def log_prob(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Return log-probability of the given indices in the size of [B x T]

        :param indices: the ground truth target labels as indices.
        :return: [B x T]
        """
        return self._cat_sampler.log_prob(indices)
