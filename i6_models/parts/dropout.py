from typing import Optional, Literal

import torch
from torch import nn


class BroadcastDropout(nn.Module):
    """
    customized dropout module supporting dropout broadcasting
    supported variants are:
        - no broadcasting (default): dropout_broadcast_axes=None
        - broadcast over the batch axis: dropout_broadcast_axes='B'
        - broadcast over the time axis: dropout_broadcast_axes='T'
        - broadcast over the batch and time axes: dropout_broadcast_axes='BT'
    """

    def __init__(self, p: float, dropout_broadcast_axes: Optional[Literal["B", "T", "BT"]] = None):
        super().__init__()

        self.p = p
        assert dropout_broadcast_axes in [
            None,
            "B",
            "T",
            "BT",
        ], "invalid value, supported are None, 'B', 'T' and 'BT'"
        self.dropout_broadcast_axes = dropout_broadcast_axes

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        :param tensor: input tensor of shape [B, T, F]
        :return: tensor of shape [B, T, F]
        """
        if self.dropout_broadcast_axes is None:
            tensor = torch.nn.functional.dropout(tensor, p=self.p, training=self.training)
        elif self.dropout_broadcast_axes == "T":  # [B..., T, F] -> [B..., F, T] -> [B..., T, F]
            # torch.nn.functional.dropout1d expects a 3D tensor and broadcasts in the last dimension.
            tensor = torch.nn.functional.dropout1d(
                tensor.transpose(-1, -2), p=self.p, training=self.training
            ).transpose(-1, -2)
        elif self.dropout_broadcast_axes == "B":  # [B..., T, F] -> [T, F, prod(B...)] -> [B..., T, F]
            batch_dim_sizes = tensor.shape[:-2]
            time_dim_size = tensor.shape[-2]
            feature_dim_size = tensor.shape[-1]

            tensor = (
                torch.nn.functional.dropout1d(
                    tensor.reshape(-1, time_dim_size, feature_dim_size).permute(1, 2, 0),
                    p=self.p,
                    training=self.training,
                )
                .permute(2, 0, 1)
                .reshape(*batch_dim_sizes, time_dim_size, feature_dim_size)
            )
        elif (
            self.dropout_broadcast_axes == "BT"
        ):  # [B..., T, F] -> [prod(B...)*T, F] -> [F, prod(B...)*T] -> [prod(B...)*T, F] -> [B..., T, F]
            batch_dim_sizes = tensor.shape[:-2]
            feature_dim_size = tensor.shape[-1]

            tensor = (
                torch.nn.functional.dropout1d(
                    tensor.reshape(-1, feature_dim_size).transpose(0, 1), p=self.p, training=self.training
                )
                .transpose(0, 1)
                .reshape(*batch_dim_sizes, -1, feature_dim_size)
            )

        return tensor
