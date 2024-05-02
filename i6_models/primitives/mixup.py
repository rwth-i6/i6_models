__all__ = ["MixupConfig", "Mixup"]

import random
from dataclasses import dataclass

import torch

from i6_models.config import ModelConfiguration


@dataclass
class MixupConfig(ModelConfiguration):
    """
    Attributes:
        buffer_size: number of frames.
        apply_prob: probability to apply mixup at all
        lambda_min: minimum lambda value
        lambda_max: maximum lambda value
        max_num_mix: maximum number of mixups (random int in [1, max_num_mix])
    """

    buffer_size: int = 1_000_000
    apply_prob: float = 1.0
    max_num_mix: int = 4
    lambda_min: float = 0.003
    lambda_max: float = 0.3


class FeatureBuffer:
    """
    The FeatureBuffer saves the feature from previous timeframes
    after the buffer is full, the buffer will not be updated
    """

    def __init__(self, *, buffer_size: int, feature_dim: int):
        self.filled = False
        self.pos = 0
        self.buffer_size = buffer_size
        self.cache = torch.nn.parameter.Parameter(
            data=torch.zeros((self.buffer_size, feature_dim)), requires_grad=False
        )

    def append(self, tensor: torch.Tensor):
        t_dim = tensor.shape[0]
        if t_dim > self.buffer_size:
            tensor = tensor[: self.buffer_size]
            t_dim = self.buffer_size
        delta_pos = min(self.buffer_size - self.pos, t_dim)
        end_pos = self.pos + delta_pos
        self.cache[self.pos : end_pos] = tensor[:delta_pos]
        self.pos = end_pos

        if end_pos == self.buffer_size:
            self.filled = True
            end_pos = t_dim - delta_pos
            self.cache[:end_pos] = tensor[delta_pos:]
            self.pos = end_pos

    def get_random(self, b_dim: int, t_dim: int, max_num_mixup: int, n_mask: torch.tensor) -> torch.Tensor:
        if not self.filled and self.pos == 0:
            return None
        else:
            end_idx = self.buffer_size if self.filled else self.pos
            max_end_idx = end_idx - t_dim

            start_indicies = torch.randint(
                high=max_end_idx, size=(b_dim, max_num_mixup)
            )  # [B, M] (M denotes maximum of num_mixup over the batch)
            start_indicies_flat = torch.masked_select(
                start_indicies, n_mask
            )  # [B, M'] (M' denotes sum of num_mixup over the batch)

            idx = torch.arange(t_dim)
            idx = torch.unsqueeze(idx, dim=-1) + start_indicies_flat  # [T, M']
            mixup_values = self.cache[idx]  # [T, M', F]
            return mixup_values


class Mixup:
    """
    Implement the Mixup data augmentation method
    C.f. https://arxiv.org/abs/1710.09412

    The code is partly based on Albert's implementation in returnn front-end
    C.f. https://github.com/rwth-i6/i6_experiments/blob/main/users/zeyer/returnn/models/rf_mixup.py
    """

    def __init__(self, feature_dim: int, cfg: MixupConfig):
        self.apply_prob = cfg.apply_prob
        self.lambda_min = cfg.lambda_min
        self.lambda_max = cfg.lambda_max
        self.max_num_mix = cfg.max_num_mix
        self.buffer_size = cfg.buffer_size
        self.feature_dim = feature_dim

        self.feature_buffer = FeatureBuffer(self.buffer_size, feature_dim)

    def __call__(self, input: torch.Tensor, sequence_mask: torch.Tensor) -> torch.Tensor:
        assert (
            input.size()[-1] == self.feature_dim
        ), "the given feature dimension does not match input feature dimension"
        input_ = self._maybe_apply_mixup(input, sequence_mask)
        self._append_to_buffer(input, sequence_mask)
        return input_

    def _append_to_buffer(self, input: torch.Tensor, sequence_mask: torch.Tensor):
        _, _, f_dim = input.size()
        # mask out the padded frames before append it to feature_buffer
        input_flat = torch.masked_select(input, sequence_mask.unsqueeze(-1)).view(-1, f_dim)
        self.feature_buffer.append(input_flat)

    def _maybe_apply_mixup(self, input: torch.Tensor, sequence_mask: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.apply_prob:
            return input

        b_dim, t_dim, f_dim = input.size()  # [B, T, F]

        max_end_idx = self.feature_buffer.buffer_size if self.feature_buffer.filled else self.feature_buffer.pos
        if t_dim > max_end_idx:
            return input

        num_mixup = torch.randint(low=1, high=self.max_num_mix + 1, size=(b_dim,))  # [B]
        max_num_mixup = max(num_mixup)  # [M] (M denotes maximum of num_mixup)

        row_vector = torch.arange(0, max_num_mixup, 1)  # [M]
        n_mask = torch.unsqueeze(num_mixup, dim=-1) > row_vector  # [B, M]

        mixup_values = self.feature_buffer.getRandom(b_dim, t_dim, max_num_mixup, n_mask).to(
            input.device
        )  # [T, M', F] (M' denotes sum of num_mixup over the batch)

        lambda_ = torch.FloatTensor(b_dim, max(num_mixup)).uniform_(self.lambda_min, self.lambda_max)  # [B, M]
        mixup_scales = torch.FloatTensor(b_dim, max(num_mixup)).uniform_(0.001, 1)  # [B, M]

        mixup_scales *= lambda_ / torch.unsqueeze(torch.sum(mixup_scales * n_mask, axis=1), dim=-1)  # [B, M]
        mixup_scales_flat = torch.masked_select(mixup_scales, n_mask)  # [M']
        mixup_values = torch.einsum("ijk,j->ijk", mixup_values, mixup_scales_flat.to(mixup_values.device))  # [T, M', F]

        idx_b = torch.arange(0, b_dim, 1)  # [B]
        idx_b = torch.masked_select(idx_b.unsqueeze(-1).expand(n_mask.size()), n_mask)  # [M']
        ones = torch.ones(mixup_values.size())  # [T, M', F]
        idx_b = torch.einsum("ijk,j->ijk", ones, idx_b).to(mixup_values.device, dtype=torch.long)  # [T, M', F]

        mixup_value = torch.scatter_add(
            torch.zeros(t_dim, b_dim, f_dim).to(mixup_values.device), 1, idx_b, mixup_values
        )  # [T, B, F]
        mixup_value = mixup_value.transpose(0, 1)  # [B, T, F]

        # mask out padded frames
        mixup_value = torch.mul(mixup_value, sequence_mask.unsqueeze(-1))  # [B, T, F]

        input = input + mixup_value  # [B, T, F]
        return input
