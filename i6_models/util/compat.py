"""
Compatibility support for different functions. This could be for example for onnx export.
"""

import torch


def logical_not(tensor: torch.Tensor, /) -> torch.Tensor:
    """
    Helper function to decide how to invert the sequence mask. For ONNX export use XOR with 1 since logical_not is not implemented.
    Else logical_not is applied for efficiency reasons.

    :param tensor: bool mask of shape (B, T) to be inverted.
    """
    if torch.onnx.is_in_onnx_export():
        return torch.logical_xor(tensor, torch.ones_like(tensor))
    else:
        return torch.logical_not(tensor)
