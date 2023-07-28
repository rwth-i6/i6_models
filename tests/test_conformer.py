from __future__ import annotations
from itertools import product
import tempfile

import torch
from torch import nn
from torch.onnx import export as export_onnx
import onnxruntime as ort

from i6_models.parts.conformer.convolution import ConformerConvolutionV1, ConformerConvolutionV1Config
from i6_models.parts.conformer.feedforward import (
    ConformerPositionwiseFeedForwardV1,
    ConformerPositionwiseFeedForwardV1Config,
)
from i6_models.parts.conformer.mhsa import ConformerMHSAV1Config, ConformerMHSAV1
from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1, VGG4LayerActFrontendV1Config
from i6_models.assemblies.conformer.conformer_v1 import ConformerEncoderV1Config, ConformerEncoderV1
from i6_models.assemblies.conformer.conformer_v1 import ConformerBlockV1, ConformerBlockV1Config
from i6_models.config import ModuleFactoryV1
from i6_models.util.mask import tensor_mask_from_length


def test_conformer_convolution_output_shape():
    def get_output_shape(batch, time, features, norm=None, kernel_size=31, dropout=0.1, activation=nn.functional.silu):
        x = torch.randn(batch, time, features)
        if norm is None:
            norm = nn.BatchNorm1d(features)
        cfg = ConformerConvolutionV1Config(
            channels=features, kernel_size=kernel_size, dropout=dropout, activation=activation, norm=norm
        )
        conformer_conv_part = ConformerConvolutionV1(cfg)
        y = conformer_conv_part(x)
        return y.shape

    assert get_output_shape(10, 50, 250) == (10, 50, 250)
    assert get_output_shape(10, 50, 250, activation=nn.functional.relu) == (10, 50, 250)  # different activation
    assert get_output_shape(10, 50, 250, norm=LayerNormNC(250)) == (10, 50, 250)  # different norm
    assert get_output_shape(1, 50, 100) == (1, 50, 100)  # test with batch size 1
    assert get_output_shape(10, 1, 50) == (10, 1, 50)  # time dim 1
    assert get_output_shape(10, 10, 20, dropout=0.0) == (10, 10, 20)  # dropout 0
    assert get_output_shape(10, 10, 20, kernel_size=3) == (10, 10, 20)  # odd kernel size


def test_ConformerPositionwiseFeedForwardV1():
    def get_output_shape(input_shape, input_dim, hidden_dim, dropout, activation):
        x = torch.randn(input_shape)
        cfg = ConformerPositionwiseFeedForwardV1Config(input_dim, hidden_dim, dropout, activation)
        conf_ffn_part = ConformerPositionwiseFeedForwardV1(cfg)
        y = conf_ffn_part(x)
        return y.shape

    for input_dim, hidden_dim, dropout, activation in product(
        [10, 20], [100, 200], [0.1, 0.3], [nn.functional.silu, nn.functional.relu]
    ):
        input_shape = (10, 100, input_dim)
        assert get_output_shape(input_shape, input_dim, hidden_dim, dropout, activation) == input_shape


def test_ConformerMHSAV1():
    def get_output_shape(input_shape, cfg, **kwargs):

        input = torch.randn(input_shape)
        output = ConformerMHSAV1(cfg)(input, **kwargs)

        return list(output.shape)

    # with key padding mask
    input_shape = [4, 15, 32]  # B,T,F
    cfg = ConformerMHSAV1Config(32, 8, 0.2, 0.3)
    assert get_output_shape(input_shape, cfg, sequence_mask=(torch.randint(0, 2, input_shape[:2]) > 0)) == [4, 15, 32]


def test_layer_norm_nc():
    torch.manual_seed(42)

    def get_output(shape, norm):
        x = torch.randn(shape)
        out = norm(x)
        return out

    # test with different shape
    torch_ln = get_output([10, 50, 250], nn.LayerNorm(250))
    custom_ln = get_output([10, 250, 50], LayerNormNC(250))
    torch.allclose(torch_ln, custom_ln.transpose(1, 2))

    torch_ln = get_output([10, 8, 23], nn.LayerNorm(23))
    custom_ln = get_output([10, 23, 8], LayerNormNC(23))
    torch.allclose(torch_ln, custom_ln.transpose(1, 2))


def test_conformer_onnx_export():
    with torch.no_grad(), tempfile.NamedTemporaryFile() as f:
        frontend_config = VGG4LayerActFrontendV1Config(
            in_features=50,
            conv1_channels=32,
            conv2_channels=64,
            conv3_channels=64,
            conv4_channels=32,
            conv_kernel_size=(3, 3),
            conv_padding=None,  # =same
            pool1_stride=(1, 2),  # pool along the feature axis,
            pool1_kernel_size=(1, 2),
            pool1_padding=None,
            pool2_stride=(1, 2),
            pool2_kernel_size=(1, 2),
            pool2_padding=None,
            out_features=256,
            activation=nn.ReLU(),
        )
        conformer_config = ConformerEncoderV1Config(
            num_layers=8,
            frontend=ModuleFactoryV1(module_class=VGG4LayerActFrontendV1, cfg=frontend_config),
            block_cfg=ConformerBlockV1Config(
                ff_cfg=ConformerPositionwiseFeedForwardV1Config(
                    input_dim=256,
                    hidden_dim=256,
                    dropout=0.2,
                    activation=nn.functional.silu,
                ),
                mhsa_cfg=ConformerMHSAV1Config(
                    input_dim=256,
                    num_att_heads=4,
                    att_weights_dropout=0.2,
                    dropout=0.2,
                ),
                conv_cfg=ConformerConvolutionV1Config(
                    channels=256, kernel_size=9, dropout=0.2, activation=nn.functional.silu, norm=LayerNormNC(256)
                ),
            ),
        )

        class DummyConformerModel(nn.Module):
            """ """

            def __init__(self, cfg: conformer_config):
                super().__init__()
                self.model = ConformerEncoderV1(cfg=cfg)

            def forward(self, input: torch.Tensor, seq_len: torch.Tensor):
                seq_mask = tensor_mask_from_length(input, seq_len)
                logits, seq_mask = self.model(input, seq_mask)
                return logits, seq_mask

        model = DummyConformerModel(cfg=conformer_config)
        dummy_data = torch.randn(3, 30, 50)
        dummy_data_len = torch.IntTensor([30, 20, 15])
        traced_model = torch.jit.trace(model.eval(), example_inputs=(dummy_data, dummy_data_len))

        dummy_data_len_2 = torch.IntTensor([30, 15, 10])

        outputs_normal, _ = model(dummy_data, dummy_data_len)
        outputs_traced, _ = traced_model(dummy_data, dummy_data_len)
        # check tracing results in the same outputs
        assert torch.allclose(outputs_normal, outputs_traced, atol=1e-5)

        export_onnx(
            traced_model,
            (dummy_data, dummy_data_len),
            f=f,
            verbose=True,
            input_names=["data", "data_len"],
            output_names=["classes"],
            dynamic_axes={
                "data": {0: "batch", 1: "time"},
                "data_len": {0: "batch"},
                "classes": {0: "batch", 1: "time"},
            },
        )

        session = ort.InferenceSession(f.name)
        outputs_onnx = torch.FloatTensor(
            session.run(None, {"data": dummy_data.numpy(), "data_len": dummy_data_len.numpy()})[0]
        )
        outputs_onnx_other = torch.FloatTensor(
            session.run(None, {"data": dummy_data.numpy(), "data_len": dummy_data_len_2.numpy()})[0]
        )

        # The default 1e-8 was slightly too strong
        assert torch.allclose(outputs_normal, outputs_onnx, atol=1e-5)
        # check that for different lengths we really get a different result
        assert not torch.allclose(outputs_normal, outputs_onnx_other, atol=1e-5)

        # in the future check with different batching and max size (20)
        # This has to fail now as we have non-batch-safe convolutions and unmasked batch-norm
        # outputs_onnx_diff_batch = torch.FloatTensor(
        #     session.run(
        #         None,
        #         {
        #             "data": dummy_data[(1, 2), :20, :].numpy(),
        #             "data_len": dummy_data_len[
        #                 (1, 2),
        #             ].numpy(),
        #         },
        #     )[0]
        # )
        # assert torch.allclose(outputs_normal[1, :20], outputs_onnx_diff_batch[0], atol=1e-6)
        # assert torch.allclose(outputs_normal[2, :15], outputs_onnx_diff_batch[1,:15], atol=1e-6)
