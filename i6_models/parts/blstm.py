from dataclasses import dataclass
import torch
from torch import nn

from i6_models.config import ModelConfiguration


@dataclass
class BlstmEncoderV1Config(ModelConfiguration):
    """
    Attributes:
        num_layers: number of bi-directional LSTM layers, minimum 2
        input_dim: input dimension size
        hidden_dim: hidden dimension of one direction of LSTM, the total output size is twice of this
        dropout: nn.LSTM supports internal Dropout applied between each layer of BLSTM (but not on input/output)
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

    num_layers: int
    input_dim: int
    hidden_dim: int
    dropout: float
    enforce_sorted: bool


class BlstmEncoderV1(torch.nn.Module):
    """
    Simple multi-layer BLSTM model including dropout, batch-first variant,
    hardcoded to use B,T,F input

    supports: TorchScript, ONNX-export
    """

    def __init__(self, config: BlstmEncoderV1Config):
        """
        :param config: configuration object
        """
        super().__init__()
        self.dropout = config.dropout
        self.enforce_sorted = config.enforce_sorted
        self.blstm_stack = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            bidirectional=True,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=self.dropout,
        )

    def forward(self, x: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        """
        :param x: [B, T, input_dim]
        :param seq_len: [B], should be on CPU for Script/Trace mode
        :return [B, T, 2 * hidden_dim]
        """
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            # during graph mode we have to assume all Tensors are on the correct device,
            # otherwise move lengths to the CPU if they are on GPU
            if seq_len.get_device() >= 0:
                seq_len = seq_len.cpu()

        blstm_packed_in = nn.utils.rnn.pack_padded_sequence(
            input=x,
            lengths=seq_len,
            enforce_sorted=self.enforce_sorted,
            batch_first=True,
        )
        blstm_out, _ = self.blstm_stack(blstm_packed_in)
        blstm_out, _ = nn.utils.rnn.pad_packed_sequence(blstm_out, padding_value=0.0, batch_first=True)

        return blstm_out
