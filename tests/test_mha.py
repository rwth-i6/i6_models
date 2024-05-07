from __future__ import annotations

import torch

from i6.models.i6_models.parts.mha import MultiheadAttentionV1, MultiheadAttentionV1Config


def test_MultiheadAttentionV1():
    def get_output_shape(input_shape, cfg, key_padding_mask=None, need_weights=True):
        input_tensor = torch.randn(input_shape)
        mha = MultiheadAttentionV1(cfg)
        output, weights = mha(input_tensor, input_tensor, input_tensor, key_padding_mask, need_weights)
        return output.shape, weights.shape

    cfg = MultiheadAttentionV1Config(input_dim=32, num_att_heads=8, att_weights_dropout=0.2, dropout=0.3)
    input_shape = [4, 15, 32]  # B,T,F

    key_padding_mask = torch.randint(0, 2, (input_shape[0], input_shape[1])) > 0

    assert get_output_shape(input_shape, cfg, key_padding_mask) == (torch.Size([4, 15, 32]), torch.Size([4, 8, 15, 15]))