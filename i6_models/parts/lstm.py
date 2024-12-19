__all__ = ["LstmBlockV1Config", "LstmBlockV1"]

from dataclasses import dataclass
import torch
from torch import nn
from typing import Dict, Tuple, Union

from i6_models.config import ModelConfiguration


@dataclass
class LstmBlockV1Config(ModelConfiguration):
    input_dim: int
    hidden_dim: int
    num_layers: int
    bias: bool
    dropout: float
    bidirectional: bool
    enforce_sorted: bool

    @classmethod
    def from_dict(cls, model_cfg_dict: Dict):
        model_cfg_dict = model_cfg_dict.copy()
        return cls(**model_cfg_dict)


class LstmBlockV1(nn.Module):
    def __init__(self, model_cfg: Union[LstmBlockV1Config, Dict], **kwargs):
        """
        Model definition of LSTM block. Contains single lstm stack and padding sequence in forward call.

        :param model_cfg: holds model configuration as dataclass or dict instance.
        :param kwargs:
        """
        super().__init__()

        self.cfg = LstmBlockV1Config.from_dict(model_cfg) if isinstance(model_cfg, Dict) else model_cfg

        self.dropout = self.cfg.dropout
        self.enforce_sorted = self.cgf.enforce_sorted
        self.lstm_stack = nn.LSTM(
            input_size=self.cfg.input_dim,
            hidden_size=self.cfg.hidden_dim,
            num_layers=self.cfg.num_layers,
            bias=self.cfg.bias,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=self.cfg.bidirectional,
        )

    def forward(self, x: torch.Tensor, seq_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            if seq_len.get_device() >= 0:
                seq_len = seq_len.cpu()

        lstm_packed_in = nn.utils.rnn.pack_padded_sequence(
            input=x,
            lengths=seq_len,
            enforce_sorted=self.enforce_sorted,
            batch_first=True,
        )

        lstm_out, _ = self.lstm_stack(lstm_packed_in)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out,
            padding_value=0.0,
            batch_first=True,
        )

        return lstm_out, seq_len
