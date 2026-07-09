__all__ = ["LstmBlockV1Config", "LstmBlockV1"]

from dataclasses import dataclass
import torch
from torch import nn
from typing import Any, Dict, Tuple, Union, Optional

from i6_models.config import ModelConfiguration


@dataclass
class LstmBlockV1Config(ModelConfiguration):
    """
    Attributes:
        input_dim: input dimension size
        hidden_dim: hidden dimension of one direction of LSTM
        num_layers: number of uni-directional LSTM layers, minimum 2
        bias: add a bias term to the LSTM layer
        dropout: nn.LSTM supports internal Dropout applied between each layer of LSTM (but not on input/output)
        enforce_sorted:
            True: expects that sequences are sorted by sequence length in decreasing order.
                Will not do any sorting.
                This is required for ONNX-Export, and thus the recommended setting.
            False: no expectation.
                It will internally enforce that they are sorted
                and undo the reordering at the output.

            Sorting can for example be performed independent of the ONNX export in e.g. train_step:

                audio_features_len, indices = torch.sort(audio_features_len, descending=True)
                audio_features = audio_features[indices, :, :]
                labels = labels[indices, :]
                labels_len = labels_len[indices]
    """

    input_dim: int
    hidden_dim: int
    num_layers: int
    bias: bool
    dropout: float
    enforce_sorted: bool

    @classmethod
    def from_dict(cls, model_cfg_dict: Dict[str, Any]):
        model_cfg_dict = model_cfg_dict.copy()
        return cls(**model_cfg_dict)


class LstmBlockV1(nn.Module):
    def __init__(self, model_cfg: Union[LstmBlockV1Config, Dict[str, Any]]):
        """
        Model definition of LSTM block. Contains single lstm stack and padding sequence in forward call. Including
        dropout, batch-first variant, hardcoded to use B,T,F input.

        Supports: TorchScript, ONNX-export.

        :param model_cfg: holds model configuration as dataclass or dict instance.
        """
        super().__init__()

        self.cfg = LstmBlockV1Config.from_dict(model_cfg) if isinstance(model_cfg, dict) else model_cfg

        self.dropout = self.cfg.dropout
        self.enforce_sorted = self.cfg.enforce_sorted
        self.lstm_stack = nn.LSTM(
            input_size=self.cfg.input_dim,
            hidden_size=self.cfg.hidden_dim,
            num_layers=self.cfg.num_layers,
            bias=self.cfg.bias,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=False,
        )

    def forward(self, x: torch.Tensor, seq_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: [B, T, input_dim]
        :param seq_len:[B], should be on CPU for Script/Trace mode
        :return: [B, T, hidden_dim]
        """
        lstm_packed_in = nn.utils.rnn.pack_padded_sequence(
            input=x, lengths=seq_len, enforce_sorted=self.enforce_sorted, batch_first=True
        )

        lstm_out, _ = self.lstm_stack(lstm_packed_in)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, padding_value=0.0, batch_first=True)

        return lstm_out, seq_len

    def forward_with_state(
        self,
        x: torch.Tensor,
        seq_len: torch.Tensor,
        lstm_h: Optional[torch.Tensor] = None,
        lstm_c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if lstm_h is None:
            lstm_packed_in = nn.utils.rnn.pack_padded_sequence(
                input=x, lengths=seq_len, enforce_sorted=self.enforce_sorted, batch_first=True
            )
            assert lstm_c is None
            lstm_out, (h_x, c_x) = self.lstm_stack(lstm_packed_in)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, padding_value=0.0, batch_first=True)
        else:
            assert lstm_c is not None
            lstm_out, (h_x, c_x) = self.lstm_stack(x, (lstm_h, lstm_c))

        return lstm_out, h_x, c_x
