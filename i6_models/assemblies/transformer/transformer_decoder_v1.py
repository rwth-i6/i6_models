from __future__ import annotations

__all__ = [
    "TransformerDecoderBlockV1Config",
    "TransformerDecoderBlockV1",
    "TransformerDecoderV1Config",
    "TransformerDecoderV1",
]

import torch
from torch import nn, Tensor

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, TypedDict, Union

from i6_models.config import ModelConfiguration
from i6_models.parts.conformer import (
    ConformerMHSARelPosV1,
    ConformerPositionwiseFeedForwardV2,
    ConformerPositionwiseFeedForwardV2Config,
)
from i6_models.parts.transformer import (
    CausalSelfAttentionV1,
    CausalSelfAttentionV1Config,
    CausalSelfAttentionV1State,
    CrossAttentionV1,
    CrossAttentionV1Config,
    CrossAttentionV1State,
    DummyState,
    ModuleWithState,
)


@dataclass
class TransformerDecoderBlockV1Config(ModelConfiguration):
    """
    Attributes:
        ff_cfg: Configuration for ConformerPositionwiseFeedForwardV1
        mhsa_cfg: Configuration for CausalSelfAttentionV1
        cross_cfg: Configuration for CrossAttentionV1
        modules: List of modules to use for ConformerBlockV2:
            - "ff" for feed forward module
            - "mhcsa" for multi-head causal self attention module
            - "conv" for conv module
        scales: List of scales to apply to the module outputs before the residual connection.
            Must have the same length as `modules`.
    """

    ff_cfg: ConformerPositionwiseFeedForwardV2Config
    mhsa_cfg: CausalSelfAttentionV1Config
    cross_cfg: CrossAttentionV1Config
    modules: List[str] = field(default_factory=lambda: ["mhcsa", "cross", "ff"])
    scales: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])

    def __post__init__(self):
        super().__post_init__()

        assert len(self.modules) == len(self.scales), "modules and scales must have same length"
        assert all(name in ["ff", "mhcsa", "cross"] for name in self.modules), "module type not supported"


class TransformerDecoderBlockV1State(TypedDict):
    """Recurrent state of a transformer block."""

    module_states: List[Union[CrossAttentionV1State, CausalSelfAttentionV1State]]


class TransformerDecoderBlockV1(nn.Module, ModuleWithState[TransformerDecoderBlockV1State]):
    """A transformer block."""

    def __init__(self, cfg: TransformerDecoderBlockV1Config):
        super().__init__()

        modules = []
        for module_name in cfg.modules:
            if module_name == "ff":
                modules.append(DummyState(ConformerPositionwiseFeedForwardV2(cfg.ff_cfg)))
            elif module_name == "mhcsa":
                modules.append(CausalSelfAttentionV1(cfg.mhsa_cfg))
            elif module_name == "cross":
                modules.append(CrossAttentionV1(cfg.cross_cfg))
            else:
                raise NotImplementedError

        self.module_list = nn.ModuleList(modules)
        self.scales = cfg.scales

    def get_initial_state(self) -> TransformerDecoderBlockV1State:
        return {"module_states": [module.get_initial_state() for module in self.module_list]}

    def transform_encoder_output(
        self,
        encoder_output: Tensor,
        encoder_output_mask: Tensor,
        state: TransformerDecoderBlockV1State,
    ) -> TransformerDecoderBlockV1State:
        new_module_state = [
            module.transform_encoder_output(encoder_output, encoder_output_mask, module_state)
            for module, module_state in zip(self.module_list, state["module_states"])
        ]
        return {**state, "module_states": new_module_state}

    def forward(
        self, labels: Tensor, labels_lens: Tensor, state: TransformerDecoderBlockV1State
    ) -> Tuple[Tensor, TransformerDecoderBlockV1State]:
        new_states = []
        for module, scale, module_state in zip(self.module_list, self.scales, state["module_states"]):
            module_output, new_mod_state = module(labels, labels_lens, module_state)
            labels = labels + module_output * scale
            new_states.append(new_mod_state)
        return labels, {**state, "module_states": new_states}


@dataclass
class TransformerDecoderV1Config(ModelConfiguration):
    """
    Attributes:
        block_cfg: Configuration for TransformerDecoderV1.
        input_dropout: Dropout applied to the input embedding.
        input_embedding_scale: Scale applied to the input embedding.
        num_blocks: Number of transformer blocks in the decoder.
        num_output: Number of output labels/vocab dim.
    """

    block_cfg: TransformerDecoderBlockV1Config
    input_dropout: float
    input_embedding_scale: Optional[float]
    num_blocks: int
    num_output: int


class TransformerDecoderV1State(TypedDict):
    """Recurrent state of the transformer decoder."""

    block_state: List[TransformerDecoderBlockV1State]
    pos: Tensor


class TransformerDecoderV1(nn.Module, ModuleWithState[TransformerDecoderV1State]):
    """
    A standard transformer decoder with causal MHSA and cross attention.

    Can be driven seq-wise during training or stepwise for inference.
    """

    def __init__(self, cfg: TransformerDecoderV1Config):
        """
        :param cfg: configuration with subunits for transformer blocks
        """
        super().__init__()

        self.model_dim = cfg.block_cfg.ff_cfg.input_dim

        self.input_dropout = nn.Dropout(cfg.input_dropout)
        self.input_embedding = nn.Embedding(cfg.num_output, cfg.block_cfg.ff_cfg.input_dim)
        self.input_embedding_scale = (
            cfg.input_embedding_scale
            if cfg.input_embedding_scale is not None
            else cfg.block_cfg.ff_cfg.input_dim**0.5
        )
        self.module_list = torch.nn.ModuleList(
            [TransformerDecoderBlockV1(cfg.block_cfg) for _ in range(cfg.num_blocks)]
        )
        self.out_norm = nn.LayerNorm(cfg.block_cfg.ff_cfg.input_dim)
        self.out_logits = nn.Linear(cfg.block_cfg.ff_cfg.input_dim, cfg.num_output)

    def get_initial_state(self) -> TransformerDecoderV1State:
        """:return: initial decoder state"""
        return {
            "block_state": [block.get_initial_state() for block in self.module_list],
            "pos": torch.tensor(0, dtype=torch.int32),
        }

    def transform_encoder_output(
        self,
        encoder_output: Tensor,
        encoder_output_mask: Tensor,
        state: TransformerDecoderV1State,
    ) -> TransformerDecoderV1State:
        """
        Process the given encoder output into input for the decoding process.

        :param encoder_output: encoder output tensor, (B, T, F)
        :param encoder_output_mask: boolean mask of valid encoder output positions,
            `True` inside valid positions, `False` outside.
        :param state: initial decoder state obtained by calling `get_initial_state()`.
        """
        new_block_state = [
            block.transform_encoder_output(encoder_output, encoder_output_mask, block_state)
            for block, block_state in zip(self.module_list, state["block_state"])
        ]
        return {**state, "block_state": new_block_state}

    def forward(
        self, labels: Tensor, labels_lens: Tensor, state: TransformerDecoderV1State
    ) -> Tuple[Tensor, TransformerDecoderV1State]:
        """
        Forwards the decoder one or multiple timesteps.

        :param labels: existing history, shape (B..., T)
        :param labels_lens: lengths of the labels in labels, shape (B...,)
        :param state: decoder state obtained by running `transform_encoder_output(enc_out, enc_out_mask, s)`, where:
            - `enc_out, enc_out_mask = forward_some_encoder(...)` and
            - `s = get_initial_state()`.
        """
        x = self.input_embedding(labels) * self.input_embedding_scale
        sinus_pe = ConformerMHSARelPosV1._sinusoidal_pe(
            torch.arange(labels.shape[-1], device=labels.device) + state["pos"], self.model_dim
        )
        x = x + sinus_pe.unsqueeze(0)
        x = self.input_dropout(x)

        output = x
        new_block_states = []
        for block, block_state in zip(self.module_list, state["block_state"]):
            output, new_block_state = block(output, labels_lens, block_state)
            new_block_states.append(new_block_state)
        new_state: TransformerDecoderV1State = {
            **state,
            "block_state": new_block_states,
            "pos": state["pos"] + labels_lens.max(),
        }

        output = self.out_norm(output)
        output_logits = self.out_logits(output)
        return output_logits, new_state
