import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import vector_norm

__all__ = [
    "RandomProjectionQuantizer",
]


class RandomProjectionQuantizer(nn.Module):
    """
    implement the fixed random projection quantizer from BestRQ
    C.f. https://arxiv.org/pdf/2202.01855 for theoretic background
    code adapted from https://github.com/speechbrain/speechbrain/blob/16b6420d4ff23210cfca2e888be8853264e0cb17/speechbrain/nnet/quantisers.py#L127
    """

    def __init__(self, input_dim, codebook_dim, codebook_num_vars):
        """
        :param input_dim: number of feature dimension of input
        :param codebook_dim: number of dimension for vocab in the codebook
        :param codebook_num_vars: vocab size of the codebook
        """
        super().__init__()

        self.input_dim = input_dim

        # projection matrix use Xavier initialization
        P_init = torch.empty((input_dim, codebook_dim))
        self.register_buffer("P", nn.init.xavier_uniform_(P_init))

        # normalize random matrix for codebook
        self.register_buffer("CB", F.normalize(torch.randn(codebook_num_vars, codebook_dim)))

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = F.normalize(x @ self.P)
        return vector_norm((self.CB.unsqueeze(1) - x.unsqueeze(1)), dim=-1).argmin(dim=1)
