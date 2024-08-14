from __future__ import annotations
from itertools import product

import torch
from torch import nn

from i6_models.parts.conformer.convolution import ConformerConvolutionV2, ConformerConvolutionV2Config
from i6_models.parts.conformer.feedforward import (
    ConformerPositionwiseFeedForwardV2,
    ConformerPositionwiseFeedForwardV2Config,
)
from i6_models.parts.conformer.mhsa_rel_pos import ConformerMHSARelPosV1Config, ConformerMHSARelPosV1
from i6_models.parts.conformer.norm import LayerNormNC


def test_ConformerConvolutionV2():
    def get_output_shape(
        batch,
        time,
        features,
        norm=None,
        kernel_size=31,
        dropout=0.1,
        activation=nn.functional.silu,
        dropout_broadcast_axes=None,
    ):
        x = torch.randn(batch, time, features)
        if norm is None:
            norm = nn.BatchNorm1d(features)
        cfg = ConformerConvolutionV2Config(
            channels=features,
            kernel_size=kernel_size,
            dropout=dropout,
            activation=activation,
            norm=norm,
            dropout_broadcast_axes=dropout_broadcast_axes,
        )
        conformer_conv_part = ConformerConvolutionV2(cfg)
        y = conformer_conv_part(x)
        return y.shape

    assert get_output_shape(10, 50, 250) == (10, 50, 250)
    assert get_output_shape(10, 50, 250, activation=nn.functional.relu) == (10, 50, 250)  # different activation
    assert get_output_shape(10, 50, 250, norm=LayerNormNC(250)) == (10, 50, 250)  # different norm
    assert get_output_shape(1, 50, 100) == (1, 50, 100)  # test with batch size 1
    assert get_output_shape(10, 1, 50) == (10, 1, 50)  # time dim 1
    assert get_output_shape(10, 10, 20, dropout=0.0) == (10, 10, 20)  # dropout 0
    assert get_output_shape(10, 10, 20, kernel_size=3) == (10, 10, 20)  # odd kernel size
    assert get_output_shape(5, 480, 512, dropout_broadcast_axes="T") == (5, 480, 512)  # dropout broadcast to T
    assert get_output_shape(5, 480, 512, dropout_broadcast_axes="BT") == (5, 480, 512)  # dropout broadcast to BT


def test_ConformerPositionwiseFeedForwardV2():
    def get_output_shape(input_shape, input_dim, hidden_dim, dropout, activation, dropout_broadcast_axes=None):
        x = torch.randn(input_shape)
        cfg = ConformerPositionwiseFeedForwardV2Config(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation=activation,
            dropout_broadcast_axes=dropout_broadcast_axes,
        )
        conf_ffn_part = ConformerPositionwiseFeedForwardV2(cfg)
        y = conf_ffn_part(x)
        return y.shape

    for input_dim, hidden_dim, dropout, activation, dropout_broadcast_axes in product(
        [10, 20], [100, 200], [0.1, 0.3], [nn.functional.silu, nn.functional.relu], [None, "B", "T", "BT"]
    ):
        input_shape = (10, 100, input_dim)
        assert get_output_shape(input_shape, input_dim, hidden_dim, dropout, activation) == input_shape


def test_ConformerMHSARelPosV1():
    def get_output_shape(
        input_shape,
        seq_len,
        input_dim,
        num_att_heads=8,
        att_weights_dropout=0.1,
        dropout=0.1,
        learnable_pos_emb=True,
        rel_pos_clip=16,
        with_pos_bias=False,
        pos_emb_dropout=0.0,
        dropout_broadcast_axes=None,
    ):
        assert len(input_shape) == 3 and input_shape[-1] == input_dim

        cfg = ConformerMHSARelPosV1Config(
            input_dim=input_dim,
            num_att_heads=num_att_heads,
            att_weights_dropout=att_weights_dropout,
            dropout=dropout,
            learnable_pos_emb=learnable_pos_emb,
            rel_pos_clip=rel_pos_clip,
            with_pos_bias=with_pos_bias,
            pos_emb_dropout=pos_emb_dropout,
            dropout_broadcast_axes=dropout_broadcast_axes,
        )
        conf_mhsa_rel_pos = ConformerMHSARelPosV1(cfg)
        input_tensor = torch.randn(input_shape)
        sequence_mask = torch.less(torch.arange(input_shape[1])[None, :], torch.tensor(seq_len)[:, None])

        output = conf_mhsa_rel_pos(input_tensor, sequence_mask)

        return list(output.shape)

    # with key padding mask
    input_shape = [4, 15, 32]  # B,T,F
    seq_len = [15, 12, 10, 15]

    for learnable_pos_emb, with_pos_bias, pos_emb_dropout in product([True, False], [True, False], [0.0, 0.1]):
        assert get_output_shape(
            input_shape,
            seq_len,
            32,
            learnable_pos_emb=learnable_pos_emb,
            with_pos_bias=with_pos_bias,
            pos_emb_dropout=pos_emb_dropout,
        ) == [4, 15, 32]
