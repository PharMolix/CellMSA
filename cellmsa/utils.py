import os
import torch
import torch.distributed as dist
import numpy as np

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        # Increase timeout to tolerate sequential loading of large files.
        from datetime import timedelta
        dist.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(minutes=60))

    return rank, world_size, local_rank


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def print_main(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)

@torch.no_grad()
def sync_grads(model, group=None, average=True):
    """Synchronize gradients with an all-reduce, similar to DDP."""
    world = dist.get_world_size(group=group) if dist.is_initialized() else 1
    if world == 1:
        return

    for p in model.parameters():
        if p.grad is None:
            continue
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=group)
        if average:
            p.grad.div_(world)


def mask_input(gene_ids, target_values, neighbor_matrix,
               mlm_probability, mask_replace_prob, random_replace_prob,
               mask_value, pad_id):
    # gene_ids: [B, G]
    # target_values: [B, G]
    # neighbor_matrix: [B, N, G]
    masked_values = target_values.clone()
    random_tensor = torch.rand_like(target_values)

    mlm_mask = random_tensor < mlm_probability
    mask_replace_mask = random_tensor < mlm_probability * mask_replace_prob
    random_replace_mask = (random_tensor >= mlm_probability * mask_replace_prob) & (random_tensor < mlm_probability * (mask_replace_prob + random_replace_prob))

    padding_mask = gene_ids.eq(pad_id)
    mlm_mask = mlm_mask & ~padding_mask
    mask_replace_mask = mask_replace_mask & ~padding_mask
    random_replace_mask = random_replace_mask & ~padding_mask

    # Replace masked positions with mask_value.
    masked_values[mask_replace_mask] = mask_value
    # Mask the same gene positions in neighbor cells.
    masked_neighbor_matrix = neighbor_matrix.clone()
    mask_expanded = mask_replace_mask.unsqueeze(1).expand_as(neighbor_matrix)
    masked_neighbor_matrix[mask_expanded] = mask_value

    # Random replacement uses either zero or the per-cell maximum value.
    replace_zero_mask = random_replace_mask & ~target_values.eq(0.0) & (torch.rand_like(target_values) < 0.5)
    masked_values[replace_zero_mask] = 0.0
    replace_max_mask = random_replace_mask & ~replace_zero_mask
    max_values = target_values.max(dim=1) # [B]
    masked_values[replace_max_mask] = max_values.values.unsqueeze(1).expand_as(target_values)[replace_max_mask]

    return masked_values, masked_neighbor_matrix, mlm_mask


def _digitize(x: np.ndarray, bins: np.ndarray, side: str = "both") -> np.ndarray:
    left_digits = np.digitize(x, bins)
    if side == "one":
        return left_digits
    right_digits = np.digitize(x, bins, right=True)
    rands = np.random.rand(len(x))
    digits = rands * (right_digits - left_digits) + left_digits
    return np.ceil(digits).astype(np.int64)


def binning_batch(values: torch.Tensor, n_bins: int, side: str = "both") -> torch.Tensor:
    """
    Per-cell quantile binning.
    Input/Output shape: [B, G]
    Bin 0 is reserved for zero values.
    """
    device = values.device
    values_np = values.detach().cpu().numpy()
    binned_rows = []
    for row in values_np:
        if row.max() == 0:
            binned_rows.append(np.zeros_like(row, dtype=np.int64))
            continue
        if row.min() <= 0:
            non_zero_ids = row.nonzero()
            non_zero_row = row[non_zero_ids]
            bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
            non_zero_digits = _digitize(non_zero_row, bins, side=side)
            binned_row = np.zeros_like(row, dtype=np.int64)
            binned_row[non_zero_ids] = non_zero_digits
        else:
            bins = np.quantile(row, np.linspace(0, 1, n_bins - 1))
            binned_row = _digitize(row, bins, side=side)
        binned_rows.append(binned_row)
    binned = torch.from_numpy(np.stack(binned_rows)).to(device=device, dtype=torch.long)
    return binned


def mask_input_binned(
    gene_ids,
    target_values,
    neighbor_matrix,
    mlm_probability,
    mask_replace_prob,
    random_replace_prob,
    pad_id,
    n_bins,
    mask_bin_id,
):
    """
    Binned masking path for category value embedding.
    """
    target_bins = binning_batch(target_values, n_bins=n_bins)
    neighbor_bins = torch.stack(
        [binning_batch(neighbor_matrix[:, i, :], n_bins=n_bins) for i in range(neighbor_matrix.size(1))],
        dim=1,
    )
    masked_values = target_bins.clone()
    masked_neighbor_matrix = neighbor_bins.clone()

    random_tensor = torch.rand_like(target_values)
    zero_mask = target_bins == 0
    num_zero = zero_mask.sum(dim=1, keepdim=True)
    G = gene_ids.size(1)
    num_nonzero = G - num_zero
    num_zero = num_zero.clamp(min=1)
    num_nonzero = num_nonzero.clamp(min=1)
    scale = torch.where(
        zero_mask,
        1 / (0.1 * G / num_zero),
        1 / (0.9 * G / num_nonzero),
    )
    random_tensor = random_tensor * scale

    mlm_mask = random_tensor < mlm_probability
    mask_replace_mask = random_tensor < mlm_probability * mask_replace_prob
    random_replace_mask = (random_tensor >= mlm_probability * mask_replace_prob) & (
        random_tensor < mlm_probability * (mask_replace_prob + random_replace_prob)
    )

    padding_mask = gene_ids.eq(pad_id)
    mlm_mask = mlm_mask & ~padding_mask
    mask_replace_mask = mask_replace_mask & ~padding_mask
    random_replace_mask = random_replace_mask & ~padding_mask

    # mask
    masked_values[mask_replace_mask] = mask_bin_id
    mask_expanded = mask_replace_mask.unsqueeze(1).expand_as(masked_neighbor_matrix)
    masked_neighbor_matrix[mask_expanded] = mask_bin_id

    # Random replace: 10% -> 0, 90% -> random non-zero bins.
    replace_zero_mask = random_replace_mask & (torch.rand_like(target_values) < 0.1)
    replace_nonzero_mask = random_replace_mask & ~replace_zero_mask
    masked_values[replace_zero_mask] = 0
    random_bins = torch.randint(1, n_bins, size=masked_values.shape, device=masked_values.device)
    masked_values[replace_nonzero_mask] = random_bins[replace_nonzero_mask]

    return masked_values, masked_neighbor_matrix, mlm_mask, target_bins, neighbor_bins
