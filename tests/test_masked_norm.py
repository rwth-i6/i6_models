import torch

from i6_models.parts.masked_norm import MaskedBatchNorm1dV1, _lengths_to_mask


def test_masked_batch_norm():
    data = (
        torch.tensor(
            [
                list(range(10)),
                list(range(10, 20)),
            ]
        )
        .unsqueeze(-1)
        .transpose(1, 2)
    )
    data_lens = torch.tensor([10, 5])
    data_mask = _lengths_to_mask(data_lens)

    norm = MaskedBatchNorm1dV1(num_features=data.shape[1])
    norm.train()
    normed_data = norm(data, data_lens)
    assert normed_data[data_mask.unsqueeze(1)].mean() < 1e-3
    assert (normed_data[data_mask.unsqueeze(1)].var().sqrt() - 1) < 1e-1

    norm = MaskedBatchNorm1dV1(num_features=data.shape[1])
    norm.train()
    normed_data_via_mask: torch.Tensor = norm(data, data_mask)
    assert torch.allclose(normed_data, normed_data_via_mask)
