from typing import Optional

import torch


def cross_entropy_loss(
    probabilities: torch.Tensor, target: torch.Tensor, pad_idx: Optional[int] = None, reduction: Optional[str] = "mean"
) -> torch.Tensor:
    """Calculate cross entropy loss

    :param probabilities: [batch size; n classes] batch with logits
    :param target: [batch size; max classes] batch with padded class labels
    :param pad_idx: id of pad label
    :param reduction: how reduce a batch of losses, `None` mean no reduction
    :return: loss
    """
    gathered_logits = torch.gather(probabilities, 1, target)
    if pad_idx is not None:
        pad_mask = target == pad_idx
        gathered_logits[pad_mask] = 1
    batch_loss = -gathered_logits.log().sum(-1)
    if reduction is None:
        return batch_loss
    elif reduction == "mean":
        return batch_loss.mean()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")
