import torch


def _mask(tensor: torch.Tensor, batch_axis: int, axis: int, pos: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    :param tensor: e.g. [B, ..., A, ...] but arbitrary axis order
    :param batch_axis: index of the batch axis
    :param axis: which axis A to mask
    :param pos: at which positions along axis to start the mask (size [B])
    :param max_len: mask length drawn uniformly from [0, max_len]
    """
    batch_dim = tensor.shape[batch_axis]
    dim = tensor.shape[axis]
    amount = torch.randint(low=1, high=max_len + 1, size=(batch_dim,), dtype=torch.int32).to(device=tensor.device)
    pos2 = torch.min(pos + amount, torch.tensor([dim] * batch_dim).to(device=tensor.device))
    idxs = torch.arange(0, dim).to(device=tensor.device).unsqueeze(0)  # [1,dim]
    pos_bc = pos.unsqueeze(1)  # [B,1]
    pos2_bc = pos2.unsqueeze(1)  # [B,1]
    cond = torch.logical_and(torch.greater_equal(idxs, pos_bc), torch.less(idxs, pos2_bc))  # [B,dim]
    if batch_axis > axis:
        cond = cond.transpose(0, 1)  # [dim,B]
    cond = torch.reshape(
        cond, shape=[tensor.shape[i] if i in (batch_axis, axis) else 1 for i in range(len(tensor.shape))]
    )
    tensor = torch.where(cond, 0.0, tensor)
    return tensor


def _random_mask(tensor: torch.Tensor, batch_axis: int, axis: int, min_num: int, max_num: int, max_len: int):
    """
    Mask tensor along axis using N in [min_num, max_num] masks of length [0, max_len]

    :param tensor: e.g. [B, ..., A, ...] but arbitrary axis order
    :param batch_axis: index of the batch axis
    :param axis: which axis to mask
    :param min_num: minimum number of masks
    :param max_num: maximum number of masks
    :param max_amount: mask length drawn uniformly from [0, max_amount]
    """

    batch_dim = tensor.shape[batch_axis]
    num_masks = torch.randint(min_num, max_num, size=(batch_dim,)).to(device=tensor.device)  # [B]

    z = -torch.log(-torch.log(torch.rand((batch_dim, tensor.shape[axis])).to(device=tensor.device)))  # [B,dim]
    _, indices = torch.topk(z, num_masks.max().item(), dim=1)

    for i in range(num_masks.max().item()):
        tensor = _mask(tensor, batch_axis, axis, indices[:, i], max_len)
    return tensor


def zero_specaugment(
    audio_features: torch.Tensor,
    time_min_num_masks: int,
    time_max_num_masks: int,
    time_mask_max_size: int,
    freq_min_num_masks: int,
    freq_max_num_masks: int,
    freq_mask_max_size: int,
):
    """
    Specaugment from legacy rossenbach/zeineldeen/zeyer attention setups (usually called specaugment_v2.py or so),
    but without any step-based scheduling. Fills masks with zeros.

    Basically just a convenience wrapper around _random_mask.

    :param audio_features: e.g. log-mel features as [B, T, F]
    :param time_min_num_masks: minimum number of masks along T
    :param time_max_num_masks: maximum number of masks along T
    :param time_mask_max_size: maximum size of masks along T
    :param freq_min_num_masks: minimum number of masks along F
    :param freq_max_num_masks: maximum number of masks along F
    :param freq_mask_max_size: maximum size of masks along F
    :return: masked audio features
    """
    assert len(tensor.shape) == 3
    tensor = _random_mask(
        audio_features, 0, 1, time_min_num_masks, time_max_num_masks, time_mask_max_size
    )  # time masking
    tensor = _random_mask(
        audio_features, 0, 2, freq_min_num_masks, freq_max_num_masks, freq_mask_max_size
    )  # freq masking
    return tensor


def zero_specaugment_by_length(
    audio_features: torch.Tensor,
    time_mask_per_n_frames: int,
    time_min_num_masks: int,
    time_mask_max_size: int,
    freq_min_num_masks: int,
    freq_max_num_masks: int,
    freq_mask_max_size: int,
):
    """
    Convenience wrapper around zero_specaugment with time-length adaptive number of masks

    :param audio_features: e.g. log-mel features as [B, T, F]
    :param time_mask_per_n_frames: maximum number of masks depending on length T.
        They are still drawn depending on the full batch length, so shorter sequences
        might get more masks than that by chance, or none at all when all masks
        fall into the padding space.
    :param time_min_num_masks: minimum number of masks along T
    :param time_mask_max_size: maximum size of masks along T
    :param freq_min_num_masks: minimum number of masks along F
    :param freq_max_num_masks: maximum number of masks along F
    :param freq_mask_max_size: maximum size of masks along F
    :return: masked audio features
    """
    return zero_specaugment(
        audio_features,
        time_min_num_masks=time_min_num_masks,
        time_max_num_masks=audio_features.size(1) // time_mask_per_n_frames,
        time_mask_max_size=time_mask_max_size,
        freq_min_num_masks=freq_min_num_masks,
        freq_max_num_masks=freq_max_num_masks,
        freq_mask_max_size=freq_mask_max_size,
    )
