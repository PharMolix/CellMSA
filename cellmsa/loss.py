import torch
import torch.nn.functional as F


def masked_mse_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the masked MSE loss between input and target.
    """
    mask = mask.float()
    loss = F.mse_loss(input * mask, target * mask, reduction="sum")
    return loss / mask.sum()

def masked_ce_loss(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute masked cross entropy loss.

    Args:
        logits: [B, G, C]
        target: [B, G]
        mask:   [B, G] bool
    """
    if mask.sum() == 0:
        return logits.new_tensor(0.0)
    target = target.long()
    loss = F.cross_entropy(logits[mask], target[mask], reduction="mean")
    return loss
