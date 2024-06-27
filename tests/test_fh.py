from itertools import product

import torch
import torch.nn as nn

from i6_models.parts.factored_hybrid import (
    DiphoneBackendV1,
    DiphoneBackendV1Config,
    PhonemeStateClassV1,
    DiphoneLogitsV1,
    DiphoneProbsV1,
)
from i6_models.parts.factored_hybrid.util import get_center_dim


def test_dim_calcs():
    n_ctx = 42

    assert get_center_dim(n_ctx, 1, PhonemeStateClassV1.none) == 42
    assert get_center_dim(n_ctx, 1, PhonemeStateClassV1.word_end) == 84
    assert get_center_dim(n_ctx, 3, PhonemeStateClassV1.word_end) == 252
    assert get_center_dim(n_ctx, 3, PhonemeStateClassV1.boundary) == 504


def test_output_shape():
    n_ctx = 42
    n_in = 32

    for we_class, states_per_ph in product(
        [PhonemeStateClassV1.none, PhonemeStateClassV1.word_end, PhonemeStateClassV1.boundary],
        [1, 3],
    ):
        backend = DiphoneBackendV1(
            DiphoneBackendV1Config(
                activation=lambda: nn.ReLU(),
                context_mix_mlp_dim=64,
                context_mix_mlp_num_layers=2,
                dropout=0.1,
                left_context_embedding_dim=32,
                n_contexts=n_ctx,
                num_hmm_states_per_phone=states_per_ph,
                num_inputs=n_in,
                phoneme_state_class=we_class,
            )
        )

        backend.train(True)
        for b, t in product([10, 50, 100], [10, 50, 100]):
            contexts_forward = torch.randint(0, n_ctx, (b, t))
            encoder_output = torch.rand((b, t, n_in))
            output = backend(features=encoder_output, contexts_left=contexts_forward)
            assert isinstance(output, DiphoneLogitsV1) and not isinstance(output, DiphoneProbsV1)
            assert output.output_left.shape == (b, t, n_ctx)
            cdim = get_center_dim(n_ctx, states_per_ph, we_class)
            assert output.output_center.shape == (b, t, cdim)

        backend.train(False)
        for b, t in product([10, 50, 100], [10, 50, 100]):
            encoder_output = torch.rand((b, t, n_in))
            output = backend(features=encoder_output)
            assert isinstance(output, DiphoneProbsV1)
            assert output.output_left.shape == (b, t, n_ctx)
            cdim = get_center_dim(n_ctx, states_per_ph, we_class)
            assert output.output_center.shape == (b, t, cdim * n_ctx)
            output_p = torch.exp(output.output_center)
            ones_hopefully = torch.sum(output_p, dim=-1)
            close_to_one = torch.abs(1 - ones_hopefully).flatten() < 1e-3
            assert all(close_to_one)
