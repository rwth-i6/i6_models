from dataclasses import dataclass
from typing import Optional, List, TypedDict, NotRequired, Tuple

import torch
from torch import Tensor, nn

from i6_models.config import ModelConfiguration, ModuleFactoryV1
from i6_models.assemblies.transformer.transformer_decoder_v1 import (
    TransformerDecoderBlockV1Config,
    TransformerDecoderBlockV1State,
    TransformerDecoderBlockV1,
)
from i6_models.parts.conformer import ConformerMHSARelPosV1
from i6_models.parts.decoder import ModuleWithState
from i6_models.parts.dropout import BroadcastDropout


@dataclass
class TransformerDecoderV2Config(ModelConfiguration):
    """
    Attributes:
        block_cfg: Configuration for TransformerDecoderV1.
        input_dropout: Dropout applied to the input embedding.
        input_embedding_scale: Scale applied to the input embedding.
            Set to `None` to apply a (tuned) default.
        num_blocks: Number of transformer blocks in the decoder.
        num_output: Number of output labels/vocab dim.
        logits_bias: Whether to add a bias to the output logits.
            Usually False is a good choice.
        share_embedding: Whether to share the input and output embedding.
    """

    block_cfg: TransformerDecoderBlockV1Config
    input_dropout: float
    input_embedding_scale: Optional[float]
    num_blocks: int
    num_output: int
    logits_bias: bool
    share_embedding: bool
    positional_encoding: Optional[ModuleFactoryV1]
    output_linear_projection: bool = True


class PositionalEncodingV1State(TypedDict):
    pos: Tensor


class SinusoidalPositionalEncodingV1(nn.Module, ModuleWithState[PositionalEncodingV1State]):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: Tensor, state: PositionalEncodingV1State):
        time_dimension = inputs.shape[1]
        embedding_dimension = inputs.shape[-1]

        sinus_pe = ConformerMHSARelPosV1._sinusoidal_pe(
            torch.arange(time_dimension, device=inputs.device) + state["pos"], embedding_dimension
        )
        new_state: PositionalEncodingV1State = {"pos": state["pos"] + time_dimension}

        output = inputs + sinus_pe.unsqueeze(0)

        return output, new_state

    def get_initial_state(self) -> PositionalEncodingV1State:
        return {"pos": Tensor(0, dtype=torch.int32)}


class TransformerDecoderV2State(TypedDict):
    """Recurrent state of the transformer decoder."""

    block_state: List[TransformerDecoderBlockV1State]
    pos_state: NotRequired[PositionalEncodingV1State]


class TransformerDecoderV2(nn.Module, ModuleWithState[TransformerDecoderV2State]):
    """
    A standard transformer decoder with causal MHSA and cross attention.

    Can be driven seq-wise during training or stepwise for inference.
    """

    def __init__(self, cfg: TransformerDecoderV2Config):
        """
        :param cfg: configuration with subunits for transformer blocks
        """
        super().__init__()

        self.model_dim = cfg.block_cfg.ff_cfg.input_dim

        self.input_dropout = BroadcastDropout(cfg.input_dropout)
        self.input_embedding = nn.Embedding(cfg.num_output, self.model_dim)
        self.input_embedding_scale = (
            cfg.input_embedding_scale if cfg.input_embedding_scale is not None else self.model_dim**0.5
        )
        self.module_list = torch.nn.ModuleList(
            [TransformerDecoderBlockV1(cfg.block_cfg) for _ in range(cfg.num_blocks)]
        )
        self.out_norm = nn.LayerNorm(self.model_dim)
        self.share_embedding = cfg.share_embedding

        cfg.positional_encoding = None
        if cfg.positional_encoding is not None:
            self.positional_encoding = cfg.positional_encoding()

        self.output_linear_projection = cfg.output_linear_projection

        self.out_logits = nn.Linear(self.model_dim, cfg.num_output, bias=cfg.logits_bias)

        if cfg.share_embedding:
            self.out_logits.weight = self.input_embedding.weight
            nn.init.xavier_uniform_(self.input_embedding.weight)  # bad convergence with default init

    def get_initial_state(self) -> TransformerDecoderV2State:
        """:return: initial decoder state"""
        state: TransformerDecoderV2State = {
            "block_state": [block.get_initial_state() for block in self.module_list],
        }

        if self.positional_encoding is not None:
            state["pos_state"] = self.positional_encoding.get_initial_state()

        return state

    def transform_encoder_output(
        self,
        encoder_output: Tensor,
        encoder_output_lens: Tensor,
        state: TransformerDecoderV2State,
    ) -> TransformerDecoderV2State:
        """
        Process the given encoder output into input for the decoding process.

        :param encoder_output: encoder output tensor, (B..., T, F)
        :param encoder_output_lens: length of seqs inside encoder_output, (B...,).
        :param state: initial decoder state obtained by calling `get_initial_state()`.
        """
        new_block_state = [
            block.transform_encoder_output(encoder_output, encoder_output_lens, block_state)
            for block, block_state in zip(self.module_list, state["block_state"])
        ]
        return {**state, "block_state": new_block_state}

    def forward(
        self, labels: Tensor, labels_lens: Tensor, state: TransformerDecoderV2State
    ) -> Tuple[Tensor, TransformerDecoderV2State]:
        """
        Forwards the decoder one or multiple timesteps.

        :param labels: existing history, shape (B..., T)
        :param labels_lens: lengths of the labels in labels, shape (B...,)
        :param state: decoder state obtained by running `transform_encoder_output(enc_out, enc_out_mask, s)`, where:
            - `enc_out, enc_out_mask = forward_some_encoder(...)` and
            - `s = get_initial_state()`.
        """
        new_state: TransformerDecoderV2State = {
            **state,
        }

        x = self.input_embedding(labels) * self.input_embedding_scale

        if self.positional_encoding is not None:
            x, new_pos_state = self.positional_encoding(x, labels_lens, state["pos"])
            new_state["pos_state"] = new_pos_state

        x = self.input_dropout(x)

        output = x
        new_block_states = []
        for block, block_state in zip(self.module_list, state["block_state"]):
            output, new_block_state = block(output, labels_lens, block_state)
            new_block_states.append(new_block_state)
        new_state["block_state"] = new_block_states

        output = self.out_norm(output)

        if self.output_linear_projection:
            output = self.out_logits(output)

        return output, new_state
