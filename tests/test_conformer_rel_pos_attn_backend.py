from __future__ import annotations
from itertools import product

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import pytest

from i6_models.parts.conformer.mhsa_rel_pos import ConformerMHSARelPosV1Config, ConformerMHSARelPosV1


def get_model(
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
    return ConformerMHSARelPosV1(cfg)


testdata = list(
    product(
        [True, False],
        [True, False],
        [0.0, 0.1],
        [True, False],
        [True, False],
        [
            SDPBackend.MATH,
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.CUDNN_ATTENTION,
        ],
    )
)


def backend_to_str(backend):
    for backend_, name in [
        (SDPBackend.MATH, "MATH"),
        (SDPBackend.FLASH_ATTENTION, "FLASH_ATTENTION"),
        (SDPBackend.EFFICIENT_ATTENTION, "EFFICIENT_ATTENTION"),
        (SDPBackend.CUDNN_ATTENTION, "CUDNN_ATTENTION"),
    ]:
        if backend == backend_:
            return name

    return None


@pytest.mark.parametrize(
    "learnable_pos_emb, with_pos_bias, pos_emb_dropout, with_linear_pos, separate_pos_emb_per_head, backend",
    testdata,
    ids=backend_to_str,
)
def test_fused_attn_backend(
    learnable_pos_emb, with_pos_bias, pos_emb_dropout, with_linear_pos, separate_pos_emb_per_head, backend
):
    input_shape = [4, 15, 32]  # B,T,F
    seq_len = [15, 12, 10, 15]

    input_tensor = torch.randn(input_shape)
    sequence_mask = torch.less(torch.arange(input_shape[1])[None, :], torch.tensor(seq_len)[:, None])

    model = get_model(
        input_dim=32,
        learnable_pos_emb=learnable_pos_emb,
        with_pos_bias=with_pos_bias,
        pos_emb_dropout=pos_emb_dropout,
        with_linear_pos=with_linear_pos,
        separate_pos_emb_per_head=separate_pos_emb_per_head,
    )
    with sdpa_kernel(backend):
        _outputs = model(input_tensor, sequence_mask)
