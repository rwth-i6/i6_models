__all__ = ["NoiseContrastiveEstimationLossV1Config", "NoiseContrastiveEstimationLossV1"]

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional
import math

from i6_models.config import ModelConfiguration


@dataclass
class NoiseContrastiveEstimationLossV1Config(ModelConfiguration):
    num_samples: int
    model: nn.Module
    noise_distribution_sampler: nn.Module  # torch.utils.data.Sampler
    log_norm_term: Optional[float] = None


class NoiseContrastiveEstimationLossV1(nn.Module):  # (nn.modules.loss._Loss):
    __constants__ = ["num_samples", "log_norm_term", "reduction"]
    num_samples: int
    log_norm_term: float

    def __init__(self, cfg: NoiseContrastiveEstimationLossV1Config) -> None:
        super().__init__()
        self.num_samples = cfg.num_samples
        self.model = cfg.model  # only used to access weights of output layer for NCE computation
        self.noise_distribution_sampler = cfg.noise_distribution_sampler
        self.log_norm_term = cfg.log_norm_term

        self._bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, data_tensor: torch.Tensor, target: torch.Tensor):
        # input: [B x T, F] target: [B x T]

        with torch.no_grad():
            samples = self.noise_distribution_sampler.sample(self.num_samples).cuda()

            # log-probabilities for the noise distribution k * q(w|h)
            sampled_prob = math.log(self.num_samples) + self.noise_distribution_sampler.log_prob(
                samples
            )  # [num_samples]
            true_sample_prob = math.log(self.num_samples) + self.noise_distribution_sampler.log_prob(target)  # [B x T]

        all_classes = torch.cat((target, samples), 0)  # [B x T + num_sampled]

        # Steps:
        # - lookup embeddings + bias
        # - compute logits for log p(w|h) = s(w, h)
        # - compute log [p(w|h) / (k q(w|h))] = s(w, h) - log [k q(w|h)]
        # - feed into BCE with logits loss

        # do lookup once
        all_emb = F.embedding(all_classes, self.model.output.weight)  # [B x T + num_sampled, F]
        all_b = F.embedding(all_classes, torch.unsqueeze(self.model.output.bias, 1))  # [B X T + num_sampled, 1]

        # slice embeddings for targets and samples below
        true_emb = torch.narrow(all_emb, 0, 0, data_tensor.shape[0])  # [B x T, F]
        true_b = torch.narrow(all_b, 0, 0, data_tensor.shape[0])  # [B x T, 1]

        sampled_emb = torch.narrow(all_emb, 0, data_tensor.shape[0], self.num_samples)  # [num_sampled, F]
        sampled_b = torch.narrow(all_b, 0, data_tensor.shape[0], self.num_samples).squeeze(
            1
        )  # [num_sampled], remove dim for broadcasting

        # compute logits log p(w|h)
        sampled_logits = torch.matmul(data_tensor, sampled_emb.T)  # [B x T, num_sampled]

        # row-wise dot product
        true_logits = torch.multiply(data_tensor, true_emb)
        true_logits = torch.sum(true_logits, 1, keepdim=True)  # [B x T, 1]

        true_logits += true_b
        sampled_logits += sampled_b

        # divide by optional constant normalization term here. Default is 1 (or 0 in log-space)
        if self.log_norm_term is not None:
            true_logits -= self.log_norm_term
            sampled_logits -= self.log_norm_term

        true_logits -= torch.log(true_sample_prob.unsqueeze(1))
        sampled_logits -= torch.log(sampled_prob.unsqueeze(0))

        out_logits = torch.cat((true_logits, sampled_logits), 1)  # [B x T, 1 + num_sampled]

        targets = torch.cat(
            (torch.ones_like(true_logits), torch.zeros_like(sampled_logits)), 1
        )  # [B x T, 1 + num_sampled]

        # no reduction on last axis here: [B x T, 1 + num_sampled]
        return self._bce(out_logits, targets)
