from itertools import product

import torch
import torch.nn as nn

from i6_models.parts.factored_hybrid import (
    BoundaryClassV1,
    FactoredDiphoneBlockV1,
    FactoredDiphoneBlockV1Config,
    FactoredDiphoneBlockV2,
    FactoredDiphoneBlockV2Config,
)
from i6_models.parts.factored_hybrid.util import get_center_dim


def test_dim_calcs():
    n_ctx = 42

    assert get_center_dim(n_ctx, 1, BoundaryClassV1.none) == 42
    assert get_center_dim(n_ctx, 1, BoundaryClassV1.word_end) == 84
    assert get_center_dim(n_ctx, 3, BoundaryClassV1.word_end) == 252
    assert get_center_dim(n_ctx, 3, BoundaryClassV1.boundary) == 504


def test_v1_output_shape_and_norm():
    n_ctx = 42
    n_in = 32

    for we_class, states_per_ph in product(
        [BoundaryClassV1.none, BoundaryClassV1.word_end, BoundaryClassV1.boundary],
        [1, 3],
    ):
        block = FactoredDiphoneBlockV1(
            FactoredDiphoneBlockV1Config(
                activation=nn.ReLU,
                context_mix_mlp_dim=64,
                context_mix_mlp_num_layers=2,
                dropout=0.1,
                left_context_embedding_dim=32,
                num_contexts=n_ctx,
                num_hmm_states_per_phone=states_per_ph,
                num_inputs=n_in,
                boundary_class=we_class,
            )
        )

        for b, t in product([10, 50, 100], [10, 50, 100]):
            contexts_forward = torch.randint(0, n_ctx, (b, t))
            encoder_output = torch.rand((b, t, n_in))
            output_center, output_left, _ = block(features=encoder_output, contexts_left=contexts_forward)
            assert output_left.shape == (b, t, n_ctx)
            cdim = get_center_dim(n_ctx, states_per_ph, we_class)
            assert output_center.shape == (b, t, cdim)

            encoder_output = torch.rand((b, t, n_in))
            output = block.forward_joint(features=encoder_output)
            cdim = get_center_dim(n_ctx, states_per_ph, we_class)
            assert output.shape == (b, t, cdim * n_ctx)
            output_p = torch.exp(output)
            ones_hopefully = torch.sum(output_p, dim=-1)
            close_to_one = torch.abs(1 - ones_hopefully).flatten() < 1e-3
            assert all(close_to_one)


def test_v2_output_shape_and_norm():
    n_ctx = 42
    n_in = 32

    for we_class, states_per_ph in product(
        [BoundaryClassV1.none, BoundaryClassV1.word_end, BoundaryClassV1.boundary],
        [1, 3],
    ):
        block = FactoredDiphoneBlockV2(
            FactoredDiphoneBlockV2Config(
                activation=nn.ReLU,
                context_mix_mlp_dim=64,
                context_mix_mlp_num_layers=2,
                dropout=0.1,
                left_context_embedding_dim=32,
                center_state_embedding_dim=128,
                num_contexts=n_ctx,
                num_hmm_states_per_phone=states_per_ph,
                num_inputs=n_in,
                boundary_class=we_class,
            )
        )

        for b, t in product([10, 50, 100], [10, 50, 100]):
            contexts_left = torch.randint(0, n_ctx, (b, t))
            contexts_center = torch.randint(0, block.num_center, (b, t))
            encoder_output = torch.rand((b, t, n_in))
            output_center, output_left, output_right, _, _ = block(
                features=encoder_output, contexts_left=contexts_left, contexts_center=contexts_center
            )
            assert output_left.shape == (b, t, n_ctx)
            assert output_right.shape == (b, t, n_ctx)
            cdim = get_center_dim(n_ctx, states_per_ph, we_class)
            assert output_center.shape == (b, t, cdim)
