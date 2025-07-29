__all__ = [
    "LstmEncoderV1Config",
    "LstmEncoderV1",
]

from dataclasses import dataclass
import torch
from torch import nn
from typing import Any, Dict, Optional, Tuple, Union

from i6_models.config import ModelConfiguration
from i6_models.parts.lstm import LstmBlockV1Config, LstmBlockV1


@dataclass
class LstmEncoderV1Config(ModelConfiguration):
    """
    Attributes:
        input_dim: input dimension size
        embed_dim: embedding dimension
        embed_dropout: dropout layer after the embedding layer
        lstm_layers_cfg: configuration of the lstm block
        lstm_dropout: dropout layer after the lstm block
        init_args: used to initialize parameters of modules, example:
            ```
            {
                "init_args_w": {"func": "normal", "arg": {"mean": 0.0, "std": 0.1}},
                "init_args_b": {"func": "normal", "arg": {"mean": 0.0, "std": 0.1}},
            }
            ```
    """

    input_dim: int
    embed_dim: int
    embed_dropout: float
    lstm_layers_cfg: LstmBlockV1Config
    lstm_dropout: float
    init_args: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.init_args is not None:
            for _, val in self.init_args.items():
                assert "func" in val.keys()
                assert "arg" in val.keys()

    @classmethod
    def from_dict(cls, model_cfg_dict: Dict[str, Any]):
        model_cfg_dict = model_cfg_dict.copy()
        model_cfg_dict["lstm_layers_cfg"] = LstmBlockV1Config.from_dict(model_cfg_dict["lstm_layers_cfg"])
        return cls(**model_cfg_dict)


class LstmEncoderV1(nn.Module):
    def __init__(self, model_cfg: Union[LstmEncoderV1Config, Dict[str, Any]]):
        """
        Model definition of LSTM encoder. Contains embedding layer followed by single lstm stack, dropout after both.
        Padding sequence in forward call.

        :param model_cfg: holds model configuration as dataclass or dict instance.
        """
        super().__init__()

        self.cfg = LstmEncoderV1Config.from_dict(model_cfg) if isinstance(model_cfg, dict) else model_cfg

        self.embedding = nn.Embedding(self.cfg.input_dim, self.cfg.embed_dim)
        self.embed_dropout = nn.Dropout(self.cfg.embed_dropout)

        self.lstm_block = LstmBlockV1(self.cfg.lstm_layers_cfg)
        self.lstm_dropout = nn.Dropout(self.cfg.lstm_dropout)

        if self.cfg.init_args is not None:
            self._param_init(**self.cfg.init_args)

    def _param_init(self, init_args_w=None, init_args_b=None):
        for m in self.modules():
            for name, param in m.named_parameters():
                if "bias" in name:
                    if init_args_b["func"] == "normal":
                        init_func = nn.init.normal_
                    else:
                        raise NotImplementedError
                    hyp = init_args_b["arg"]
                else:
                    if init_args_w["func"] == "normal":
                        init_func = nn.init.normal_
                    else:
                        raise NotImplementedError
                    hyp = init_args_w["arg"]
                init_func(param, **hyp)

    def forward(self, x: torch.Tensor, seq_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param x: [B, T, input_dim]
        :param seq_len: [B]
        :return: [B, T, lstm_layers_cfg.hidden_dim]
        """
        embed = self.embedding(x)
        embed = self.embed_dropout(embed)

        out, _ = self.lstm_block(embed, seq_len)
        out = self.lstm_dropout(out)

        return out, seq_len

    def forward_with_state(
        self,
        x: torch.Tensor,
        seq_len: torch.Tensor,
        lstm_h: Optional[torch.Tensor] = None,
        lstm_c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embed = self.embedding(x)
        embed = self.embed_dropout(embed)

        out, lstm_h, lstm_c = self.lstm_block.forward_with_state(embed, seq_len, lstm_h=lstm_h, lstm_c=lstm_c)
        out = self.lstm_dropout(out)

        return out, lstm_h, lstm_c
