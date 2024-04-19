import torch.nn as nn
import collections


def _weight_drop(module, weights, dropout):
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


class WeightDropLinear(torch.nn.Linear):
    """
    Wrapper around :class:`torch.nn.Linear` that apply dropout on the weights
    weight dropout paper c.f. https://ieeexplore.ieee.org/document/9468799
    code c.f. https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/weight_drop.html
    """

    def __init__(self, *args, weight_dropout: float = 0.0, **kwargs):
        """
        Attributes:
            weight_dropout: the dropout probability
        """
        super().__init__(*args, **kwargs)
        weights = ["weight"]
        _weight_drop(self, weights, weight_dropout)
