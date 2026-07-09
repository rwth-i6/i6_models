__all__ = [
    "NoiseContrastiveEstimationLossV1",
]

import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional


class NoiseContrastiveEstimationLossV1(nn.Module):
    __constants__ = ["num_samples", "log_norm_term", "reduction"]
    num_samples: int
    log_norm_term: float

    def __init__(
        self,
        num_samples: int,
        *,
        model_output_weights: torch.Tensor,
        model_output_bias: torch.Tensor,
        noise_distribution_sampler: nn.Module,
        log_norm_term: Optional[float] = None,
        reduction: str = "none",
        device: Optional[str] = None,
    ) -> None:
        """
        Noise contrastive estimation (NCE) loss implementation.
        Used to estimate the softmax. Normally for very large softmax sizes, for example word-level LM.

        :param num_samples: num of samples for the estimation, normally a value between 1000-4000.
            2000 is a good starting point.
        :param model_output_weights: represents the weights of the output layer of the model.
        :param model_output_bias: represents the bias of the output layer of the model.
        :param noise_distribution_sampler: for example `i6_models.parts.samplers.LogUniformSampler`.
        :param log_norm_term: normalisation term for true/sampled logits.
        :param reduction: reduction method for binary cross entropy.
        :param device: device where the loss will be placed.
        """
        super().__init__()

        self.num_samples = num_samples
        self.model_output_weights = model_output_weights
        self.model_output_bias = model_output_bias
        self.noise_distribution_sampler = noise_distribution_sampler
        self.log_norm_term = log_norm_term
        self.device = device

        self._bce = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, encoder_output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """

        :param encoder_output: here the INPUT to the final linear layer (variable `output` in the encoder model). Logit
            computation is performed in this method, requiring the raw encoder output fo be fed. The implementation
            requires a tensor where batch and time are flattened, requiring an `encoder_output` tensor shape [B x T, F].
        :param labels: ground truth labels. A tensor with flattened batch and time is required, requiring a `labels`
            tensor shape [B x T].
        :return: [B x T, 1 + num_sampled] if reduction == "none"
        """
        with torch.no_grad():
            samples = self.noise_distribution_sampler.sample(self.num_samples).cuda()

            # log-probabilities for the noise distribution k * q(w|h)
            ws = torch.log(torch.tensor(float(self.num_samples)))
            sampled_prob = ws + self.noise_distribution_sampler.log_prob(samples)  # [num_samples]
            true_sample_prob = ws + self.noise_distribution_sampler.log_prob(labels)  # [B x T]

        all_classes = torch.cat((labels, samples), 0)  # [B x T + num_sampled]

        # Steps:
        # - lookup embeddings + bias
        # - compute logits for log p(w|h) = s(w, h)
        # - compute log [p(w|h) / (k q(w|h))] = s(w, h) - log [k q(w|h)]
        # - feed into BCE with logits loss

        # do lookup once
        all_emb = F.embedding(all_classes, self.model_output_weights)  # [B x T + num_sampled, F]
        all_b = F.embedding(all_classes, torch.unsqueeze(self.model_output_bias, 1))  # [B X T + num_sampled, 1]

        # slice embeddings for targets and samples below
        true_emb = torch.narrow(all_emb, 0, 0, encoder_output.shape[0])  # [B x T, F]
        true_b = torch.narrow(all_b, 0, 0, encoder_output.shape[0])  # [B x T, 1]

        sampled_emb = torch.narrow(all_emb, 0, encoder_output.shape[0], self.num_samples)  # [num_sampled, F]
        sampled_b = torch.narrow(all_b, 0, encoder_output.shape[0], self.num_samples).squeeze(
            1
        )  # [num_sampled], remove dim for broadcasting

        # compute logits log p(w|h)
        sampled_logits = torch.matmul(encoder_output, sampled_emb.T)  # [B x T, num_sampled]

        # row-wise dot product
        true_logits = torch.multiply(encoder_output, true_emb)  # [B x T, F]
        true_logits = torch.sum(true_logits, 1, keepdim=True)  # [B x T, 1]

        true_logits += true_b
        sampled_logits += sampled_b

        # divide by optional constant normalization term here. Default is 1 (or 0 in log-space)
        if self.log_norm_term is not None:
            true_logits -= self.log_norm_term
            sampled_logits -= self.log_norm_term

        true_logits -= true_sample_prob.unsqueeze(1)
        sampled_logits -= sampled_prob.unsqueeze(0)

        out_logits = torch.cat((true_logits, sampled_logits), 1)  # [B x T, 1 + num_sampled]

        targets = torch.cat(
            (torch.ones_like(true_logits), torch.zeros_like(sampled_logits)), 1
        )  # [B x T, 1 + num_sampled]

        # no reduction on last axis here: [B x T, 1 + num_sampled]
        return self._bce(out_logits, targets)
