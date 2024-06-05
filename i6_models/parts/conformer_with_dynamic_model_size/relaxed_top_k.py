__all__ = ["RelaxedTopK"]

import numpy as np
import torch

EPSILON = np.finfo(np.float32).tiny


class RelaxedTopK(torch.nn.Module):
    """
    Given scores, apply softmax k times and return either binary or relaxed k-hot vector
    refer to Top k Relaxation (https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/sampling/subsets.html) for more details
    """

    def __init__(self, k: int, hard: bool = False):
        """
        :param k: the top k elements
        :param hard: if hard return binary k-hot vector, otherewise return relaxed k-hot vector
        """
        super(RelaxedTopK, self).__init__()
        self.k = k
        self.hard = hard

    def forward(self, scores: torch.tensor) -> torch.tensor:
        """
        :param scores: the corresponding score for each element
        """
        khot = torch.zeros_like(scores)
        onehot_approx = torch.zeros_like(scores)
        for _ in range(self.k):
            khot_mask = torch.max(1.0 - onehot_approx, torch.tensor([EPSILON]).cuda())
            scores = scores + torch.log(khot_mask)
            onehot_approx = torch.nn.functional.softmax(scores)
            khot = khot + onehot_approx

        if self.hard:
            # straight through trick
            khot_hard = torch.zeros_like(khot)
            _, ind = torch.topk(khot, self.k)
            khot_hard = khot_hard.scatter_(0, ind, 1)
            res = khot_hard - khot.detach() + khot
        else:
            res = khot

        return res
