from __future__ import annotations
import torch

from i6_models.parts.conformer.mhsa import ConformerMHSAV1Config, ConformerMHSAV1


def test_ConformerMHSAV1():
    def get_output_shape(input_shape, cfg, **kwargs):

        input = torch.randn(input_shape)
        output = ConformerMHSAV1(cfg)(input, **kwargs)

        return list(output.shape)

    # default parameters
    input_shape = [3, 10, 20]  # B,T,F
    cfg = ConformerMHSAV1Config(20, 4)
    assert get_output_shape(input_shape, cfg) == [3, 10, 20]

    # all parameters
    input_shape = [4, 15, 32]  # B,T,F
    cfg = ConformerMHSAV1Config(32, 8, att_weights_dropout=0.2, dropout=0.3)
    assert get_output_shape(input_shape, cfg, key_padding_mask=torch.randint(0, 2, input_shape[:2]) > 0) == [4, 15, 32]
