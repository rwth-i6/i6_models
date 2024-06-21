from __future__ import annotations

import torch

from i6_models.parts.mhsa import MultiheadSelfAttentionV1, MultiheadSelfAttentionV1Config


def test_MultiheadSelfAttentionV1():
    """
    Test the functionality of the MultiheadSelfAttentionV1 module.
    """

    def get_output_shape(input_shape, cfg, key_padding_mask=None, need_weights=True):
        input_tensor = torch.randn(input_shape)
        mhsa = MultiheadSelfAttentionV1(cfg)
        output, weights = mhsa(input_tensor, key_padding_mask)
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
    torch_mhsa = torch.nn.MultiheadAttention(cfg.input_dim, cfg.num_att_heads, dropout=0, batch_first=True)
    torch_mhsa.eval()

    mhsav1 = MultiheadSelfAttentionV1(cfg)
    mhsav1.eval()

    in_proj_weight = torch_mhsa.in_proj_weight
    in_proj_bias = torch_mhsa.in_proj_bias

    out_proj_weight = torch_mhsa.out_proj.weight
    out_proj_bias = torch_mhsa.out_proj.bias

    mhsav1.in_proj.weight = in_proj_weight
    mhsav1.in_proj.bias = in_proj_bias
    mhsav1.out_proj.weight = out_proj_weight
    mhsav1.out_proj.bias = out_proj_bias

    input_shape = [4, 15, 32]  # B,T,F
    input_tensor = torch.randn(input_shape)

    key_padding_mask = torch.randint(0, 2, (input_shape[0], input_shape[1])) > 0

    mhsav1_out, _ = mhsav1(input_tensor, key_padding_mask)
    torch_mhsa_out, _ = torch_mhsa(input_tensor, input_tensor, input_tensor, key_padding_mask)

    assert torch.allclose(mhsav1_out, torch_mhsa_out, atol=1e-08)
