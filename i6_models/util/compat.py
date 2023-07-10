"""
Compatibility functions for different functions. This could be for example for onnx export.
"""

import torch


def logical_not(tensor: torch.Tensor):
    """
    Helper function to decide how to invert the sequence mask. For ONNX export use XOR with 1 since logical_not is not implemented.
    Else logical_not is applied for efficiency reasons.

    :param sequence_mask: bool mask of shape (B, T) to be inverted.
    """
    if torch.onnx.is_in_onnx_export():
        return torch.logical_xor(sequence_mask, torch.ones_like(sequence_mask))
    else:
        return torch.logical_not(sequence_mask)
