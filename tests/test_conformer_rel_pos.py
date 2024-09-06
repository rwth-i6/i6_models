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
        with_bias=True,
        num_att_heads=8,
        att_weights_dropout=0.1,
        dropout=0.1,
        learnable_pos_emb=True,
        with_linear_pos=False,
        separate_pos_emb_per_head=False,
        rel_pos_clip=16,
        with_pos_bias=False,
        pos_emb_dropout=0.0,
        dropout_broadcast_axes=None,
    ):
        assert len(input_shape) == 3 and input_shape[-1] == input_dim

        cfg = ConformerMHSARelPosV1Config(
            input_dim=input_dim,
            num_att_heads=num_att_heads,
            with_bias=with_bias,
            att_weights_dropout=att_weights_dropout,
            dropout=dropout,
            learnable_pos_emb=learnable_pos_emb,
            with_linear_pos=with_linear_pos,
            separate_pos_emb_per_head=separate_pos_emb_per_head,
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

    for learnable_pos_emb, with_pos_bias, pos_emb_dropout, with_linear_pos, separate_pos_emb_per_head in product(
        [True, False], [True, False], [0.0, 0.1], [True, False], [True, False]
    ):
        assert get_output_shape(
            input_shape,
            seq_len,
            32,
            learnable_pos_emb=learnable_pos_emb,
            with_pos_bias=with_pos_bias,
            pos_emb_dropout=pos_emb_dropout,
            with_linear_pos=with_linear_pos,
            separate_pos_emb_per_head=separate_pos_emb_per_head,
        ) == [4, 15, 32]


def test_ConformerMHSARelPosV1_against_Espnet():
    from espnet2.asr_transducer.encoder.modules.attention import RelPositionMultiHeadedAttention
    from espnet2.asr_transducer.encoder.modules.positional_encoding import RelPositionalEncoding

    num_heads = 4
    embed_size = 256
    dropout_rate = 0.1
    batch_dim_size = 4
    time_dim_size = 50
    seq_len = torch.Tensor([50, 10, 20, 40])
    sequence_mask = torch.less(torch.arange(time_dim_size)[None, :], seq_len[:, None])

    espnet_mhsa_module = RelPositionMultiHeadedAttention(
        num_heads=num_heads, embed_size=embed_size, dropout_rate=dropout_rate
    )
    espnet_mhsa_module.eval()
    espnet_pos_enc_module = RelPositionalEncoding(embed_size, dropout_rate=dropout_rate)
    espnet_pos_enc_module.eval()

    cfg = ConformerMHSARelPosV1Config(
        input_dim=embed_size,
        num_att_heads=num_heads,
        with_bias=True,
        att_weights_dropout=dropout_rate,
        dropout=dropout_rate,
        learnable_pos_emb=False,
        with_linear_pos=True,
        separate_pos_emb_per_head=True,
        rel_pos_clip=None,
        with_pos_bias=True,
        pos_emb_dropout=dropout_rate,
        dropout_broadcast_axes=None,
    )
    own_mhsa_module = ConformerMHSARelPosV1(cfg)
    own_mhsa_module.eval()
    own_mhsa_module.linear_pos = espnet_mhsa_module.linear_pos
    own_mhsa_module.pos_bias_u = espnet_mhsa_module.pos_bias_u
    own_mhsa_module.pos_bias_v = espnet_mhsa_module.pos_bias_v
    own_mhsa_module.out_proj = espnet_mhsa_module.linear_out
    own_mhsa_module.qkv_proj.weight = nn.Parameter(
        torch.cat(
            [
                espnet_mhsa_module.linear_q.weight,
                espnet_mhsa_module.linear_k.weight,
                espnet_mhsa_module.linear_v.weight,
            ],
            dim=0,
        )
    )
    own_mhsa_module.qkv_proj.bias = nn.Parameter(
        torch.cat(
            [espnet_mhsa_module.linear_q.bias, espnet_mhsa_module.linear_k.bias, espnet_mhsa_module.linear_v.bias],
            dim=0,
        )
    )

    input_tensor = torch.rand((batch_dim_size, time_dim_size, embed_size))
    inv_sequence_mask = torch.logical_not(sequence_mask)

    input_tensor_layernorm = own_mhsa_module.layernorm(input_tensor)

    espnet_pos_enc = espnet_pos_enc_module(input_tensor_layernorm)
    espnet_output_tensor = espnet_mhsa_module(
        query=input_tensor_layernorm,
        key=input_tensor_layernorm,
        value=input_tensor_layernorm,
        pos_enc=espnet_pos_enc,
        mask=inv_sequence_mask,
    )

    own_output_tensor = own_mhsa_module(input_tensor, sequence_mask=sequence_mask)

    assert torch.allclose(espnet_output_tensor, own_output_tensor, rtol=1e-03, atol=1e-6)
