import torch
from torch.nn import Parameter

from typing import List


def _weight_drop(module: torch.nn.module, weights: List[str], dropout: float = 0.0):
    for name_w in weights:
        w = getattr(module, name_w)
        del module._parameters[name_w]
        module.register_parameter(name_w + "_raw", Parameter(w))

    def forward(*args, **kwargs):
        for name_w in weights:
            raw_w = getattr(module, name_w + "_raw")
            w = torch.nn.functional.dropout(raw_w, p=dropout, training=module.training)
            setattr(module, name_w, w)

        return module.forward(*args, **kwargs)

    setattr(module, "forward", forward)


class WeightDrop(torch.nn.Module):
    """
    Apply dropout on weights of a given nn.module
    weight dropout paper c.f. https://ieeexplore.ieee.org/document/9468799
    implementation c.f. https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/weight_drop.html
    """

    def __init__(self, module: torch.nn.module, weights: List[str], dropout: float = 0.0):
        """
        Attributes:
            module:
        """
        super(WeightDrop, self).__init__()
        _weight_drop(module, weights, dropout)
        self.forward = module.forward


class WeightDropLinear(torch.nn.Linear):
    """
    Wrapper around :class:`torch.nn.Linear` that apply dropout on the weights
    """

    def __init__(self, *args, weight_dropout: float = 0.0, **kwargs):
        """
        Attributes:
            weight_dropout: the dropout probability
        """
        super().__init__(*args, **kwargs)
        weights = ["weight"]
        _weight_drop(self, weights, weight_dropout)
