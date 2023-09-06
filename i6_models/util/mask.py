import torch


def mask_tensor(tensor: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
    """
    Create a "positive" mask for a tensor (boolean true means position is used)
    on the same device as the tensor.

    :param tensor: [B,T,....]
    :param seq_len: [B]
    :return: Mask of [B,T]
    """
    seq_len = seq_len.to(device=tensor.device)
    r = torch.arange(tensor.shape[1], device=tensor.device)  # [T]
    seq_mask = torch.less(r[None, :], seq_len[:, None])  # broadcast to [B,T]
    return seq_mask
