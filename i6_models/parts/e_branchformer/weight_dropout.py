import torch
import torch.nn as nn

supported_modules = (nn.Linear, nn.Conv1d)


def _weight_drop(module, weights, dropout):
    """
    Helper for `WeightDrop`.
    """
    for name_w in weights:
        w = getattr(module, name_w)
        assert isinstance(w, nn.Parameter), "%s type should be nn.Parameter" % name_w
        del module._parameters[name_w]
        module.register_parameter(name_w + '_raw', nn.Parameter(w))

    original_module_forward = module.forward

    def forward(*args, **kwargs):
        for name_w in weights:
            raw_w = getattr(module, name_w + '_raw')
            w = torch.nn.functional.dropout(raw_w, p=dropout, training=module.training)
            setattr(module, name_w, w)

        return original_module_forward(*args, **kwargs)

    setattr(module, 'forward', forward)


class WeightDrop(torch.nn.Module):
    """
    A module that wraps another layer in which some weights are replaced by 0 during training
    implementation is based on https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/weight_drop.html
    """
    def __init__(self, module, weights, dropout=0):
        super(WeightDrop, self).__init__()

        assert isinstance(module, supported_modules), 'not implemented yet'
        _weight_drop(module, weights, dropout)
        self.forward = module.forward
