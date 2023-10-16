import onnxruntime as ort
import tempfile
import torch
from torch.onnx import export as export_onnx

from i6_models.parts.blstm import BlstmEncoderV1, BlstmEncoderV1Config


def test_blstm_onnx_export():
    with torch.no_grad(), tempfile.NamedTemporaryFile() as f:
        config = BlstmEncoderV1Config(num_layers=4, input_dim=50, hidden_dim=128, dropout=0.1, enforce_sorted=True)
        model = BlstmEncoderV1(config=config)
        scripted_model = torch.jit.optimize_for_inference(torch.jit.script(model.eval()))

        dummy_data = torch.randn(3, 30, 50)
        dummy_data_len = torch.IntTensor([30, 20, 15])
        dummy_data_len_2 = torch.IntTensor([30, 15, 10])

        outputs_normal = model(dummy_data, dummy_data_len)
        outputs_scripted = scripted_model(dummy_data, dummy_data_len)
        assert torch.allclose(outputs_normal, outputs_scripted)
        export_onnx(
            scripted_model,
            (dummy_data, dummy_data_len),
            f=f,
            verbose=True,
            input_names=["data", "data_len"],
            output_names=["classes"],
            dynamic_axes={
                # dict value: manually named axes
                "data": {0: "batch", 1: "time"},
                "data_len": {0: "batch"},
                "classes": {0: "batch", 1: "time"},
            },
        )
        session = ort.InferenceSession(f.name, providers=["CPUExecutionProvider"])
        outputs_onnx = torch.FloatTensor(
            session.run(None, {"data": dummy_data.numpy(), "data_len": dummy_data_len.numpy()})[0]
        )
        outputs_onnx_other = torch.FloatTensor(
            session.run(None, {"data": dummy_data.numpy(), "data_len": dummy_data_len_2.numpy()})[0]
        )
        # The default 1e-8 was slightly too strong
        assert torch.allclose(outputs_normal, outputs_onnx, atol=1e-6)
        # check that for different lengths we really get a different result
        assert not torch.allclose(outputs_normal, outputs_onnx_other, atol=1e-6)

        # check with different batching and max size
        outputs_onnx_diff_batch = torch.FloatTensor(
            session.run(
                None,
                {
                    "data": dummy_data[(1, 2), :20, :].numpy(),
                    "data_len": dummy_data_len[
                        (1, 2),
                    ].numpy(),
                },
            )[0]
        )
        assert torch.allclose(outputs_normal[2, :20], outputs_onnx_diff_batch[1], atol=1e-6)
