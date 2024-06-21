from __future__ import annotations

import torch

from i6_models.parts.mha import MultiheadSelfAttentionV1, MultiheadSelfAttentionV1Config


def test_MultiheadSelfAttentionV1():
    """
    Test the functionality of the MultiheadSelfAttentionV1 module.
    """

    def get_output_shape(input_shape, cfg, key_padding_mask=None, need_weights=True):
        input_tensor = torch.randn(input_shape)
        mha = MultiheadSelfAttentionV1(cfg)
        output, weights = mha(input_tensor, key_padding_mask)
        return output.shape, weights.shape

    cfg = MultiheadSelfAttentionV1Config(input_dim=32, num_att_heads=8, att_weights_dropout=0.2, dropout=0.3)
    input_shape = [4, 15, 32]  # B,T,F

    key_padding_mask = torch.randint(0, 2, (input_shape[0], input_shape[1])) > 0

    assert get_output_shape(input_shape, cfg, key_padding_mask) == (torch.Size([4, 15, 32]), torch.Size([4, 8, 15, 15]))


def test_ComparisonMHSAV1Torch():
    """
    Compares the output of the MultiheadSelfAttentionV1 module with the output of the torch.nn.MultiheadAttention module.
    """
    cfg = MultiheadSelfAttentionV1Config(input_dim=32, num_att_heads=8, att_weights_dropout=0, dropout=0)
    torch_mha = torch.nn.MultiheadAttention(cfg.input_dim, cfg.num_att_heads, dropout=0, batch_first=True)
    torch_mha.eval()

    mhav1 = MultiheadSelfAttentionV1(cfg)
    mhav1.eval()

    in_proj_weight = torch_mha.in_proj_weight
    in_proj_bias = torch_mha.in_proj_bias

    out_proj_weight = torch_mha.out_proj.weight
    out_proj_bias = torch_mha.out_proj.bias

    mhav1.in_proj.weight = in_proj_weight
    mhav1.in_proj.bias = in_proj_bias
    mhav1.out_proj.weight = out_proj_weight
    mhav1.out_proj.bias = out_proj_bias

    input_shape = [4, 15, 32]  # B,T,F
    input_tensor = torch.randn(input_shape)

    key_padding_mask = torch.randint(0, 2, (input_shape[0], input_shape[1])) > 0

    mhav1_out, _ = mhav1(input_tensor, key_padding_mask)
    torch_mha_out, _ = torch_mha(input_tensor, input_tensor, input_tensor, key_padding_mask)

    assert torch.allclose(mhav1_out, torch_mha_out, atol=1e-08)

test_ComparisonMHSAV1Torch()
test_MultiheadSelfAttentionV1()