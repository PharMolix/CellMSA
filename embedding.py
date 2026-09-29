import os
import gc
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import random
import argparse
import time
from tqdm import tqdm
import scanpy as sc

from cellmsa.model.cellmsa import CellMSAModule
from cellmsa.config import Config
from cellmsa.utils import print_main, setup_distributed, binning_batch
from cellmsa.dataset import (
    collate_fn,
    NEIGHBOR_RELATION_SAME_BATCH,
    NEIGHBOR_RELATION_SAME_TYPE_DIFF_BATCH,
    NEIGHBOR_RELATION_SIMILAR_CLASS,
    NEIGHBOR_RELATION_UNKNOWN,
    load_checkpoint_gene_info,
    load_prebuilt_gene_info,
    select_gene_indices,
)

def _extract_gene_matrix(adata, gene_ids, geneid2name, var_key_gene_name, desc):
    gene_names = (
        adata.var[var_key_gene_name].values
        if var_key_gene_name in adata.var
        else adata.var_names.values
    )
    gene_to_idx = {name: i for i, name in enumerate(gene_names)}

    target_positions = []
    source_positions = []
    for target_pos, gene_id in enumerate(gene_ids):
        gene_name = geneid2name[gene_id]
        source_pos = gene_to_idx.get(gene_name)
        if source_pos is not None:
            target_positions.append(target_pos)
            source_positions.append(source_pos)

    print_main(f"\n{desc}...")
    exprs = np.zeros((adata.shape[0], len(gene_ids)), dtype=np.float32)
    if source_positions:
        selected = adata.X[:, source_positions]
        if hasattr(selected, "toarray"):
            selected = selected.toarray()
        else:
            selected = np.asarray(selected)
        exprs[:, target_positions] = selected.astype(np.float32, copy=False)
    print_main(f"✓ {desc}: {len(source_positions)}/{len(gene_ids)} genes mapped")
    return exprs, len(source_positions)


def _filter_gene_ids_to_adata(adata, gene_ids, geneid2name, var_key_gene_name):
    gene_names = (
        adata.var[var_key_gene_name].values
        if var_key_gene_name in adata.var
        else adata.var_names.values
    )
    available = {str(name) for name in gene_names if str(name)}
    filtered_gene_ids = np.array(
        [gene_id for gene_id in gene_ids if geneid2name[gene_id] in available],
        dtype=np.int64,
    )
    if filtered_gene_ids.size == 0:
        raise ValueError(
            f"No genes from the checkpoint vocab were found in adata.var[{var_key_gene_name!r}]."
        )
    return filtered_gene_ids


def _validate_neighbor_indices(neighbor_indices, n_msa_obs, source_name):
    if neighbor_indices.ndim != 2:
        raise ValueError(
            f"{source_name} must be a 2D array, got shape {neighbor_indices.shape}"
        )
    valid_or_missing = (neighbor_indices == -1) | (
        (neighbor_indices >= 0) & (neighbor_indices < n_msa_obs)
    )
    if not np.all(valid_or_missing):
        bad_count = int((~valid_or_missing).sum())
        bad_min = int(neighbor_indices[~valid_or_missing].min())
        bad_max = int(neighbor_indices[~valid_or_missing].max())
        raise ValueError(
            f"{source_name} contains {bad_count} indices outside [0, {n_msa_obs}). "
            f"Bad range: [{bad_min}, {bad_max}]. Rebuild the neighbor arrays for "
            "the current query/MSA h5ad instead of reusing indices from a larger file."
        )


def _build_neighbor_sources(input_adata, n_msa_obs):
    if "neighbor_indices" in input_adata.obsm:
        neighbor_indices = np.asarray(input_adata.obsm["neighbor_indices"], dtype=np.int64)
        _validate_neighbor_indices(neighbor_indices, n_msa_obs, "obsm['neighbor_indices']")
        neighbor_relation_ids = np.full(
            neighbor_indices.shape,
            NEIGHBOR_RELATION_SAME_TYPE_DIFF_BATCH,
            dtype=np.int64,
        )
        return neighbor_indices, neighbor_relation_ids

    required_keys = ["stsb", "stdb", "dt"]
    if all(key in input_adata.obsm for key in required_keys):
        stsb = np.asarray(input_adata.obsm["stsb"], dtype=np.int64)
        stdb = np.asarray(input_adata.obsm["stdb"], dtype=np.int64)
        dt = np.asarray(input_adata.obsm["dt"], dtype=np.int64)

        neighbor_indices = np.concatenate([stsb, stdb, dt], axis=1)
        _validate_neighbor_indices(neighbor_indices, n_msa_obs, "obsm['stsb'/'stdb'/'dt']")
        neighbor_relation_ids = np.concatenate(
            [
                np.full(stsb.shape, NEIGHBOR_RELATION_SAME_BATCH, dtype=np.int64),
                np.full(stdb.shape, NEIGHBOR_RELATION_SAME_TYPE_DIFF_BATCH, dtype=np.int64),
                np.full(dt.shape, NEIGHBOR_RELATION_SIMILAR_CLASS, dtype=np.int64),
            ],
            axis=1,
        )

        neighbor_relation_ids[neighbor_indices < 0] = NEIGHBOR_RELATION_UNKNOWN
        return neighbor_indices, neighbor_relation_ids

    raise KeyError(
        "input_adata.obsm must contain either 'neighbor_indices' or all of ['stsb', 'stdb', 'dt']."
    )


def _chunk_output_path(output_prefix, output_name, output_suffix, start_idx, end_idx, chunk_size):
    chunk_label = f"{start_idx}-{end_idx}"
    return output_prefix.parent / f"{output_prefix.name}_{output_name}{output_suffix}_{chunk_label}.npy"


class DownstreamDataset(Dataset):
    def __init__(
        self,
        input_adata,
        msa_adata,
        gene_ids,
        var_key_gene_name,
        vocab,
        n_bins: int,
        gene_truncation_enable: bool = False,
        gene_truncation_length: int = 2048,
        gene_truncation_nonzero_ratio: float = 1.0,
        gene_truncation_diff_eps: float = 1e-4,
        padding_gene_id: int = 0,
        pad_value: float = 0.0,
    ):
        super().__init__()
        geneid2name = {v: k for k, v in vocab.items()}
        self.n_bins = n_bins
        self.gene_truncation_enable = gene_truncation_enable
        self.gene_truncation_length = gene_truncation_length
        self.gene_truncation_nonzero_ratio = gene_truncation_nonzero_ratio
        self.gene_truncation_diff_eps = gene_truncation_diff_eps
        self.padding_gene_id = padding_gene_id
        self.pad_value = pad_value

        mapped_gene_ids = _filter_gene_ids_to_adata(
            input_adata,
            gene_ids,
            geneid2name,
            var_key_gene_name,
        )
        print_main(
            f"✓ Using {len(mapped_gene_ids)}/{len(gene_ids)} checkpoint genes present in input adata"
        )

        input_exprs, hit_genes = _extract_gene_matrix(
            input_adata,
            mapped_gene_ids,
            geneid2name,
            var_key_gene_name,
            "Mapping genes from input adata to vocab",
        )
        self.input_exprs_raw = input_exprs
        self.input_id2msa_ids, self.neighbor_relation_ids = _build_neighbor_sources(
            input_adata,
            msa_adata.n_obs,
        )

        msa_exprs, _ = _extract_gene_matrix(
            msa_adata,
            mapped_gene_ids,
            geneid2name,
            var_key_gene_name,
            "Mapping genes from msa adata to vocab",
        )
        self.msa_exprs_raw = msa_exprs
        self.gene_ids_np = np.asarray(mapped_gene_ids, dtype=np.int64)
        self.gene_ids = torch.from_numpy(self.gene_ids_np)
        print_main(f"✓ Input data: {hit_genes}/{len(mapped_gene_ids)} genes found in input adata")

    def __len__(self):
        return self.input_exprs_raw.shape[0]

    def _bin_expr_rows(self, rows):
        rows = np.asarray(rows, dtype=np.float32)
        if rows.ndim == 1:
            rows = rows[None, :]
        if rows.shape[1] == 0:
            return np.zeros(rows.shape, dtype=np.int64)
        return (
            binning_batch(torch.from_numpy(rows), n_bins=self.n_bins, side="one")
            .cpu()
            .numpy()
            .astype(np.int64, copy=False)
        )

    def __getitem__(self, idx):
        expr_raw = self.input_exprs_raw[idx]
        neighbor_idx = self.input_id2msa_ids[idx]
        relation_ids = self.neighbor_relation_ids[idx].copy()
        valid_mask = neighbor_idx >= 0
        relation_ids[~valid_mask] = NEIGHBOR_RELATION_UNKNOWN
        gene_ids = self.gene_ids_np

        neighbor_matrix_raw = np.zeros((neighbor_idx.shape[0], self.msa_exprs_raw.shape[1]), dtype=np.float32)
        if np.any(valid_mask):
            neighbor_matrix_raw[valid_mask] = self.msa_exprs_raw[neighbor_idx[valid_mask]]

        if self.gene_truncation_enable:
            selected_idx = select_gene_indices(
                cell_expr=expr_raw,
                neighbor_exprs=neighbor_matrix_raw,
                target_length=self.gene_truncation_length,
                nonzero_ratio=self.gene_truncation_nonzero_ratio,
                diff_eps=self.gene_truncation_diff_eps,
            )
            expr_selected = np.asarray(expr_raw[selected_idx], dtype=np.float32)
            gene_ids = np.asarray(gene_ids[selected_idx], dtype=np.int64)
            neighbor_matrix_selected = np.asarray(neighbor_matrix_raw[:, selected_idx], dtype=np.float32)

            # Keep inference consistent with training: select genes first, then bin.
            expr = self._bin_expr_rows(expr_selected)[0]
            neighbor_matrix = self._bin_expr_rows(neighbor_matrix_selected)
            if expr.shape[0] < self.gene_truncation_length:
                pad_width = self.gene_truncation_length - expr.shape[0]
                expr = np.pad(expr, (0, pad_width), constant_values=self.pad_value)
                neighbor_matrix = np.pad(
                    neighbor_matrix,
                    ((0, 0), (0, pad_width)),
                    constant_values=self.pad_value,
                )
                gene_ids = np.pad(gene_ids, (0, pad_width), constant_values=self.padding_gene_id)
            else:
                expr = expr[:self.gene_truncation_length]
                neighbor_matrix = neighbor_matrix[:, :self.gene_truncation_length]
                gene_ids = gene_ids[:self.gene_truncation_length]
        else:
            expr = self._bin_expr_rows(expr_raw)[0]
            neighbor_matrix = self._bin_expr_rows(neighbor_matrix_raw)

        return {
            'expr': torch.as_tensor(expr, dtype=torch.long),
            'neighbor_matrix': torch.as_tensor(neighbor_matrix, dtype=torch.long),
            'gene_ids': torch.as_tensor(gene_ids, dtype=torch.long),
            'neighbor_relation_ids': torch.as_tensor(relation_ids, dtype=torch.long),
        }

@torch.no_grad()
def emb(
    model,
    loader,
    config,
    output_prefix,
    output_suffix="",
    chunk_size=10000,
    save_values=False,
):
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    start_time = time.time()
    model.eval()
    device = config.local_rank if config.local_rank >= 0 else 0

    embeddings = []
    values_all = [] if save_values else None
    chunk_start = 0
    chunk_count = 0
    total_count = 0

    def flush_chunk(force=False):
        nonlocal embeddings, values_all
        nonlocal chunk_start, chunk_count
        if chunk_count == 0 or (chunk_count < chunk_size and not force):
            return

        chunk_end = chunk_start + chunk_count
        embeddings_chunk = torch.cat(embeddings, dim=0).cpu().numpy()
        embeddings_path = _chunk_output_path(
            output_prefix,
            "embeddings",
            output_suffix,
            chunk_start,
            chunk_end,
            chunk_size,
        )
        np.save(embeddings_path, embeddings_chunk)

        if save_values:
            values_chunk = torch.cat(values_all, dim=0).cpu().numpy()
            np.save(
                _chunk_output_path(output_prefix, "values", output_suffix, chunk_start, chunk_end, chunk_size),
                values_chunk,
            )

        print_main(f"Saved embedding chunk: {embeddings_path}")
        embeddings = []
        values_all = [] if save_values else None
        chunk_start = chunk_end
        chunk_count = 0
        gc.collect()
        torch.cuda.empty_cache()

    for batch_data in tqdm(loader):
        gene_ids = batch_data['gene_ids'].to(device)
        values = batch_data['expr'].to(device)

        neighbor_matrix = batch_data['neighbor_matrix'].to(device)
        neighbor_relation_ids = batch_data.get('neighbor_relation_ids')
        if neighbor_relation_ids is not None:
            neighbor_relation_ids = neighbor_relation_ids.to(device)

        with torch.cuda.amp.autocast(enabled=config.amp):
            output = model(
                values,
                neighbor_matrix,
                gene_ids,
                neighbor_relation_ids=neighbor_relation_ids,
                use_embedding=False,
            )
            pooled_repr = output['cell_emb_raw'] # (b, d), cls token

        batch_size = pooled_repr.size(0)
        offset = 0
        while offset < batch_size:
            take = min(chunk_size - chunk_count, batch_size - offset)
            batch_slice = slice(offset, offset + take)
            embeddings.append(pooled_repr[batch_slice].cpu()) # (b, d)
            if save_values:
                values_all.append(values[batch_slice].cpu().to(torch.uint8)) # (b, n)
            chunk_count += take
            total_count += take
            offset += take
            flush_chunk()

        del output, pooled_repr, gene_ids, values, neighbor_matrix
        if neighbor_relation_ids is not None:
            del neighbor_relation_ids

    flush_chunk(force=True)
    end_time = time.time()
    print_main(f"Evaluation time: {end_time - start_time:.2f} seconds")
    return total_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_cell_types', type=int, default=None)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--query_h5ad', type=str, default='/to/path/')
    parser.add_argument('--msa_h5ad', type=str, default='/to/path/')
    parser.add_argument('--var_key_gene_name', type=str, default='var_names')
    parser.add_argument('--output_prefix', type=str, default='/to/path/')
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--chunk_size', type=int, default=100000)
    parser.add_argument('--save_values', action='store_true')
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    assert world_size == 1

    config = Config()
    config.local_rank = local_rank
    config.world_size = world_size
    config.batch_size = args.batch_size
    config.value_emb_style = "category"
    if args.num_workers is not None:
        config.num_workers = args.num_workers

    seed = config.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device(f'cuda:{local_rank}' if local_rank >= 0 else 'cuda:0')

    print_main("=" * 60)
    print_main("CellMSA + scGPT Training (Shared Pretrained Embedding)")
    print_main(f"World size: {world_size}")
    print_main(f"Embedding dim: {config.d_model}")
    print_main("=" * 60)

    checkpoint = None
    vocab = None
    gene_ids = None
    if args.resume and os.path.exists(args.resume):
        print_main(f"\nLoading checkpoint metadata: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        gene_info = load_checkpoint_gene_info(checkpoint)
        if gene_info is not None:
            vocab, gene_ids, gene_symbols = gene_info
            print_main(f"✓ Loaded vocab/gene ids from checkpoint: {len(vocab)} vocab, {len(gene_ids)} genes")
    else:
        assert False, f"Checkpoint not found: {args.resume}"

    adata_input = sc.read_h5ad(args.query_h5ad)
    adata_msa = sc.read_h5ad(args.msa_h5ad)

    dataset = DownstreamDataset(
        adata_input,
        adata_msa,
        gene_ids,
        var_key_gene_name=args.var_key_gene_name,
        vocab=vocab,
        n_bins=config.n_bins,
        gene_truncation_enable=config.gene_truncation_enable,
        gene_truncation_length=config.gene_truncation_length,
        gene_truncation_nonzero_ratio=1.0,
        gene_truncation_diff_eps=config.gene_truncation_diff_eps,
        padding_gene_id=vocab.get(config.pad_token, 0),
        pad_value=config.pad_value,
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        drop_last=False,
        pin_memory=True,
        prefetch_factor=2 if config.num_workers > 0 else None,
        persistent_workers=True if config.num_workers > 0 else False,
    )
    print_main(f"✓ DataLoader: {len(loader)} batches per GPU, {config.num_workers} workers")

    print_main("\nCreating model with shared embedding...")
    model = CellMSAModule(gene_vocab_size=len(vocab),
                          embedding_dim=config.d_model,
                          pair_embedding_dim=config.d_pair,
                          padding_gene_id=vocab.get(config.pad_token, 0),
                          cls_gene_id=vocab.get("<cls>", 1),
                          cce_temp=config.cce_temp,
                          msa_depth=config.msa_depth,
                          pairformer_depth=config.pairformer_depth,
                          value_emb_style=config.value_emb_style,
                          n_bins=config.n_bins,
                          msa_dim=config.msa_dim,
                          msa_outer_product_mean_dim_hidden=config.msa_outer_product_mean_dim_hidden,
                          msa_pwa_dropout_row_prob=config.msa_pwa_dropout_row_prob,
                          msa_pwa_heads=config.msa_pwa_heads,
                          msa_pwa_dim_head=config.msa_pwa_dim_head,
                          pairformer_pair_bias_attn_dim_head=config.pairformer_pair_bias_attn_dim_head,
                          pairformer_pair_bias_attn_heads=config.pairformer_pair_bias_attn_heads,
                          pairformer_dropout_row_prob=config.pairformer_dropout_row_prob)

    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True) if world_size > 1 else model

    n_params = sum(p.numel() for p in model.parameters())
    print_main(f"✓ Model: {n_params:,} parameters")

    if checkpoint is not None:
        print_main(f"\nResuming from checkpoint: {args.resume}")
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        del checkpoint
        torch.cuda.empty_cache()

    print_main("\nEvaluation:")
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    total_embeddings = emb(
        model,
        loader,
        config,
        output_prefix=output_prefix,
        chunk_size=args.chunk_size,
        save_values=args.save_values,
    )
    print_main(f"✓ Saved {total_embeddings} embeddings in chunks of {args.chunk_size}")

    if world_size > 1:
        dist.destroy_process_group()
    print_main("\n" + "=" * 60)
    print_main("Done!")
    print_main("=" * 60)


if __name__ == '__main__':
    main()
