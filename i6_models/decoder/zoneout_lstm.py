import torch
from torch import nn

from typing import Tuple


class ZoneoutLSTMCell(nn.Module):
    """
    Wrap an LSTM cell with Zoneout regularization (https://arxiv.org/abs/1606.01305)
    """

    def __init__(self, cell: nn.RNNCellBase, zoneout_h: float, zoneout_c: float):
        """
        :param cell: LSTM cell
        :param zoneout_h: zoneout drop probability for hidden state
        :param zoneout_c: zoneout drop probability for cell state
        """
        super().__init__()
        self.cell = cell
        assert 0.0 <= zoneout_h <= 1.0 and 0.0 <= zoneout_c <= 1.0, "Zoneout drop probability must be in [0, 1]"
        self.zoneout_h = zoneout_h
        self.zoneout_c = zoneout_c

    def forward(
        self, inputs: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type="cuda", enabled=False):
            h, c = self.cell(inputs)
        prev_h, prev_c = state
        h = self._zoneout(prev_h, h, self.zoneout_h)
        c = self._zoneout(prev_c, c, self.zoneout_c)
        return h, c

    def _zoneout(self, prev_state: torch.Tensor, curr_state: torch.Tensor, factor: float):
        """
        Apply Zoneout.

        :param prev: previous state tensor
        :param curr: current state tensor
        :param factor: drop probability
        """
        if factor == 0.0:
            return curr_state
        if self.training:
            mask = curr_state.new_empty(size=curr_state.size()).bernoulli_(factor)
            return mask * prev_state + (1 - mask) * curr_state
        else:
            return factor * prev_state + (1 - factor) * curr_state
