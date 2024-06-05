__all__ = ["StochasticDepth"]

import torch
import torch.fx
from torch import nn


def stochastic_depth(input: torch.tensor, p: float, mode: str, training: bool = True) -> torch.tensor:
    """
    :param input: the input tensor
    :param p: the dropout probability
    :param mode: whether to apply dropout on the whole batch (randomly zeroes the entire input) or on rows (randomly selects sequnces from the batch)
    :param training: if model is in training
    """

    if p < 0.0 or p > 1.0:
        raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
    if mode not in ["batch", "row"]:
        raise ValueError(f"mode has to be either 'batch' or 'row', but got {mode}")
    if not training or p == 0.0:
        return input

    survival_rate = 1.0 - p
    if mode == "row":
        size = [input.shape[0]] + [1] * (input.ndim - 1)
    else:
        size = [1] * input.ndim
    noise = torch.empty(size, dtype=input.dtype, device=input.device)
    noise = noise.bernoulli_(survival_rate)
    if survival_rate > 0.0:
        noise.div_(survival_rate)
    return input * noise


torch.fx.wrap("stochastic_depth")


class StochasticDepth(nn.Module):
    """
    Implements the Stochastic Depth from https://arxiv.org/abs/1603.09382
    code is based on https://pytorch.org/vision/main/_modules/torchvision/ops/stochastic_depth.html#stochastic_depth
    """

    def __init__(self, p: float, mode: str) -> None:
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, input: torch.tensor) -> torch.tensor:
        return stochastic_depth(input, self.p, self.mode, self.training)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(p={self.p}, mode={self.mode})"
        return s
