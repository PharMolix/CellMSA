import os
import json
import time
import pickle
import zipfile
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import Dataset
from .utils import print_main

NEIGHBOR_RELATION_SAME_BATCH = 0
NEIGHBOR_RELATION_SAME_TYPE_DIFF_BATCH = 1
NEIGHBOR_RELATION_SIMILAR_CLASS = 2
NEIGHBOR_RELATION_UNKNOWN = 3
SPECIAL_TOKENS = {"<pad>", "<cls>", "<unk>", "<mask>", "<eoc>"}

class PretrainedEmbeddingLoader:
    """
    Load pretrained gene embeddings and build gene mappings.
    """

    def __init__(self, pretrain_dir: str):
        self.pretrain_dir = pretrain_dir

        # Load pretrained vocabulary (gene symbol -> id).
        vocab_path = os.path.join(pretrain_dir, 'vocab.json')
        with open(vocab_path, 'r') as f:
            self.pretrain_vocab = json.load(f)
        print_main(f"✓ Loaded pretrain vocab: {len(self.pretrain_vocab)} genes")

        # Load pretrained weights.
        model_path = os.path.join(pretrain_dir, 'best_model.pt')
        self.pretrain_weights = torch.load(model_path, map_location='cpu')
        self.pretrain_embedding = self.pretrain_weights['encoder.embedding.weight']
        print_main(f"✓ Loaded pretrain embedding: {self.pretrain_embedding.shape}")

    def get_embedding_for_genes(self, gene_symbols: List[str]) -> Dict[str, int]:
        """
        Create a vocabulary for the requested genes.

        Args:
            gene_symbols: Gene symbols to include.

        Returns:
            vocab: {gene_symbol: id} vocabulary.
        """
        # Build a new vocabulary.
        vocab = {'<pad>': 0, '<cls>': 1, '<unk>': 2}
        for i, gene in enumerate(gene_symbols):
            vocab[gene] = i + 3

        return vocab


def _ordered_symbols_from_vocab(vocab: Dict[str, int]) -> List[str]:
    size = max(vocab.values()) + 1
    symbols = [None] * size
    for token, idx in vocab.items():
        if idx < 0 or idx >= size:
            raise ValueError(f"Invalid vocab index {idx} for token {token}")
        symbols[idx] = token
    missing = [idx for idx, token in enumerate(symbols) if token is None]
    if missing:
        raise ValueError(f"Missing vocab entries for indices: {missing[:8]}")
    return symbols


def _is_special_token(token: str) -> bool:
    return isinstance(token, str) and token in SPECIAL_TOKENS


def _normalize_gene_symbols(symbols: List[str], vocab: Dict[str, int]) -> List[str]:
    normalized = []
    seen = set()
    for symbol in symbols:
        if symbol in seen or _is_special_token(symbol):
            continue
        if symbol not in vocab:
            raise KeyError(f"Gene symbol {symbol} is missing from vocab")
        normalized.append(symbol)
        seen.add(symbol)
    return normalized


def _gene_ids_from_symbols(gene_symbols: List[str], vocab: Dict[str, int]) -> np.ndarray:
    return np.array([vocab[symbol] for symbol in gene_symbols], dtype=np.int64)


def load_prebuilt_gene_info(
    training_data_path: str,
) -> Tuple[Dict[str, int], np.ndarray, List[str], Dict]:
    meta_path = os.path.join(training_data_path, "metadata.pkl")
    with open(meta_path, "rb") as f:
        training_meta = pickle.load(f)

    if "gene_vocab" in training_meta:
        vocab = dict(training_meta["gene_vocab"])
        if "gene_symbols" in training_meta:
            gene_symbols = list(training_meta["gene_symbols"])
        elif "hvg_symbols" in training_meta:
            gene_symbols = list(training_meta["hvg_symbols"])
        else:
            gene_symbols = _ordered_symbols_from_vocab(vocab)
        gene_symbols = _normalize_gene_symbols(gene_symbols, vocab)
        gene_ids = _gene_ids_from_symbols(gene_symbols, vocab)
    else:
        raise KeyError(f"Unsupported metadata keys in {meta_path}: {list(training_meta.keys())}")

    return vocab, gene_ids, gene_symbols, training_meta


def load_checkpoint_gene_info(checkpoint: Dict) -> Optional[Tuple[Dict[str, int], np.ndarray, List[str]]]:
    vocab = checkpoint.get("vocab")
    gene_symbols = checkpoint.get("gene_symbols")
    if gene_symbols is None:
        gene_symbols = checkpoint.get("hvg_symbols")

    if vocab is None or gene_symbols is None:
        return None

    vocab = dict(vocab)
    gene_symbols = _normalize_gene_symbols(list(gene_symbols), vocab)
    gene_ids = _gene_ids_from_symbols(gene_symbols, vocab)
    return vocab, gene_ids, gene_symbols


def _zip_member_data_offset(zip_path: str, member_name: str) -> int:
    with zipfile.ZipFile(zip_path, "r") as zf:
        info = zf.getinfo(member_name)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{zip_path}:{member_name} is compressed; mmap is unsupported")

        with open(zip_path, "rb") as fh:
            fh.seek(info.header_offset)
            local_header = fh.read(30)
            if len(local_header) != 30 or local_header[:4] != b"PK\x03\x04":
                raise ValueError(f"Invalid local zip header for {zip_path}:{member_name}")
            name_len = int.from_bytes(local_header[26:28], "little")
            extra_len = int.from_bytes(local_header[28:30], "little")
        return info.header_offset + 30 + name_len + extra_len


def _open_zip_npy_memmap(zip_path: str, member_name: str) -> np.memmap:
    data_offset = _zip_member_data_offset(zip_path, member_name)
    with open(zip_path, "rb") as fh:
        fh.seek(data_offset)
        version = np.lib.format.read_magic(fh)
        shape, fortran_order, dtype = np.lib.format._read_array_header(fh, version)
        array_offset = fh.tell()
    return np.memmap(
        zip_path,
        dtype=dtype,
        mode="r",
        offset=array_offset,
        shape=shape,
        order="F" if fortran_order else "C",
    )


class CSRRowAccessor:
    """Memory-map CSR arrays stored inside an uncompressed scipy sparse .npz archive."""

    def __init__(self, npz_path: str):
        self.npz_path = npz_path
        with zipfile.ZipFile(npz_path, "r") as zf:
            with zf.open("format.npy") as f:
                fmt = np.load(f)
            with zf.open("shape.npy") as f:
                shape = tuple(np.load(f).tolist())

        if isinstance(fmt, np.ndarray):
            fmt = fmt.item()
        if isinstance(fmt, np.bytes_):
            fmt = fmt.decode("utf-8")
        elif isinstance(fmt, bytes):
            fmt = fmt.decode("utf-8")
        else:
            fmt = str(fmt)
        if fmt != "csr":
            raise ValueError(f"Unsupported sparse format {fmt} in {npz_path}")

        self.shape = shape
        self.indices = _open_zip_npy_memmap(npz_path, "indices.npy")
        self.indptr = _open_zip_npy_memmap(npz_path, "indptr.npy")
        self.data = _open_zip_npy_memmap(npz_path, "data.npy")

    def get_rows(self, row_indices: np.ndarray) -> np.ndarray:
        row_indices = np.asarray(row_indices, dtype=np.int64).reshape(-1)
        out = np.zeros((len(row_indices), self.shape[1]), dtype=np.float32)
        for i, row_idx in enumerate(row_indices):
            start = int(self.indptr[row_idx])
            end = int(self.indptr[row_idx + 1])
            cols = np.asarray(self.indices[start:end], dtype=np.int64)
            vals = np.asarray(self.data[start:end], dtype=np.float32)
            out[i, cols] = vals
        return out


def build_neighbor_relation_ids(n_neighbors: int) -> np.ndarray:
    relation_ids = np.full((n_neighbors,), NEIGHBOR_RELATION_SIMILAR_CLASS, dtype=np.int64)
    relation_ids[: min(24, n_neighbors)] = NEIGHBOR_RELATION_SAME_BATCH
    if n_neighbors > 24:
        relation_ids[24:min(48, n_neighbors)] = NEIGHBOR_RELATION_SAME_TYPE_DIFF_BATCH
    return relation_ids


def select_gene_indices(
    cell_expr: np.ndarray,
    neighbor_exprs: np.ndarray,
    target_length: Optional[int],
    nonzero_ratio: float,
    diff_eps: float,
) -> np.ndarray:
    """
    Select informative genes for one cell with a fixed quota:
    1. prioritize non-zero genes ranked by expression
    2. when non-zero genes do not fill the full quota, use standardized diff
       only for the remaining quota
    3. if the non-zero expression quota is not filled, leave that quota's
       remainder for padding instead of backfilling it with diff-ranked genes
    """
    n_genes = cell_expr.shape[0]
    if target_length is None or target_length <= 0:
        return np.arange(n_genes, dtype=np.int64)

    nonzero_idx = np.flatnonzero(cell_expr > 0)

    nonzero_quota = min(int(round(target_length * nonzero_ratio)), target_length)
    diff_quota = target_length - nonzero_quota

    selected_chunks = []
    selected_mask = np.zeros(n_genes, dtype=bool)

    if nonzero_idx.size > 0 and nonzero_quota > 0:
        nz_order = np.argsort(-cell_expr[nonzero_idx], kind="stable")
        top_nonzero = nonzero_idx[nz_order[:nonzero_quota]]
        selected_chunks.append(top_nonzero)
        selected_mask[top_nonzero] = True

    if diff_quota <= 0:
        if not selected_chunks:
            return np.empty((0,), dtype=np.int64)
        selected_idx = np.concatenate(selected_chunks)
        if selected_idx.size > target_length:
            selected_idx = selected_idx[:target_length]
        return np.sort(selected_idx.astype(np.int64, copy=False))

    neighbor_exprs = np.asarray(neighbor_exprs, dtype=np.float32)
    if neighbor_exprs.ndim == 1:
        neighbor_exprs = neighbor_exprs[None, :]

    if neighbor_exprs.shape[0] == 0:
        neighbor_mean = np.zeros_like(cell_expr, dtype=np.float32)
        neighbor_std = np.ones_like(cell_expr, dtype=np.float32)
    else:
        neighbor_mean = neighbor_exprs.mean(axis=0, dtype=np.float32)
        neighbor_std = neighbor_exprs.std(axis=0, dtype=np.float32)

    diff_score = np.abs(cell_expr - neighbor_mean) / (neighbor_std + diff_eps)

    remaining_idx = np.flatnonzero(~selected_mask)
    if remaining_idx.size > 0 and diff_quota > 0:
        diff_order = np.argsort(-diff_score[remaining_idx], kind="stable")
        top_diff = remaining_idx[diff_order[:diff_quota]]
        selected_chunks.append(top_diff)
        selected_mask[top_diff] = True

    if not selected_chunks:
        return np.empty((0,), dtype=np.int64)

    selected_idx = np.concatenate(selected_chunks)
    if selected_idx.size > target_length:
        selected_idx = selected_idx[:target_length]
    return np.sort(selected_idx.astype(np.int64, copy=False))


def apply_gene_truncation(
    cell_expr: np.ndarray,
    neighbor_exprs: np.ndarray,
    gene_ids: np.ndarray,
    target_length: Optional[int],
    nonzero_ratio: float,
    diff_eps: float,
    pad_gene_id: int,
    pad_value: float = 0.0,
    selected_idx: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if selected_idx is None:
        selected_idx = select_gene_indices(
            cell_expr=cell_expr,
            neighbor_exprs=neighbor_exprs,
            target_length=target_length,
            nonzero_ratio=nonzero_ratio,
            diff_eps=diff_eps,
        )
    if target_length is None or target_length <= 0:
        target_length = selected_idx.size

    cell_selected = np.asarray(cell_expr[selected_idx], dtype=np.float32)
    neighbor_selected = np.asarray(neighbor_exprs[:, selected_idx], dtype=np.float32)
    gene_selected = np.asarray(gene_ids[selected_idx], dtype=np.int64)

    if cell_selected.shape[0] >= target_length:
        return (
            cell_selected[:target_length],
            neighbor_selected[:, :target_length],
            gene_selected[:target_length],
        )

    pad_width = target_length - cell_selected.shape[0]
    cell_selected = np.pad(cell_selected, (0, pad_width), constant_values=pad_value)
    neighbor_selected = np.pad(neighbor_selected, ((0, 0), (0, pad_width)), constant_values=pad_value)
    gene_selected = np.pad(gene_selected, (0, pad_width), constant_values=pad_gene_id)
    return cell_selected, neighbor_selected, gene_selected


class PrebuiltDataset(Dataset):
    """
    Load prebuilt sharded data, using one shard per epoch.
    Class-level shared state lets worker processes reuse the same mmap handles.
    """

    # Shared state across dataset instances and worker processes.
    _shared_expr = None
    _shared_sample_indices = None
    _shared_neighbor_indices = None
    _shared_shard_idx = -1
    _shared_expr_format = None

    def __init__(
        self,
        training_data_path: str,
        gene_ids: np.ndarray,
        n_neighbors: int = 40,
        rank: int = 0,
        world_size: int = 1,
        gene_truncation_enable: bool = False,
        gene_truncation_length: Optional[int] = None,
        gene_truncation_nonzero_ratio: float = 0.7,
        gene_truncation_diff_eps: float = 1e-4,
        padding_gene_id: int = 0,
        pad_value: float = 0.0,
    ):
        self.training_data_path = training_data_path
        self.gene_ids = gene_ids
        self.gene_ids_tensor = torch.from_numpy(gene_ids)
        self.n_neighbors = n_neighbors
        self.rank = rank
        self.world_size = world_size
        self.gene_truncation_enable = gene_truncation_enable
        self.gene_truncation_length = gene_truncation_length
        self.gene_truncation_nonzero_ratio = gene_truncation_nonzero_ratio
        self.gene_truncation_diff_eps = gene_truncation_diff_eps
        self.padding_gene_id = padding_gene_id
        self.pad_value = pad_value
        self._sample_cell_type_ids = None
        self._sample_rows_sorted_by_type = None
        self._type_row_starts = None
        self._type_row_counts = None
        self._cell_type_names = None

        # Load global metadata.
        meta_path = os.path.join(training_data_path, 'metadata.pkl')
        if rank == 0:
            print(f"Loading prebuilt training data from {training_data_path}...")
        with open(meta_path, 'rb') as f:
            self.global_meta = pickle.load(f)

        self.n_shards = self.global_meta['n_shards']
        self.shard_info = [dict(info) for info in self.global_meta['shard_info']]

        for info in self.shard_info:
            shard_dir = os.path.join(training_data_path, f"shard_{info['shard_idx']}")
            shard_meta_path = os.path.join(shard_dir, "metadata.pkl")
            if os.path.exists(shard_meta_path):
                with open(shard_meta_path, "rb") as f:
                    shard_meta = pickle.load(f)
                info.setdefault("n_cells", shard_meta.get("n_cells"))
                info["n_samples"] = shard_meta.get("n_samples", info.get("n_samples"))

            expr_dense_path = os.path.join(shard_dir, "expr_all.npy")
            expr_sparse_path = os.path.join(shard_dir, "expr_sparse.npz")
            if os.path.exists(expr_dense_path):
                info["expr_format"] = "dense"
                info["expr_path"] = expr_dense_path
            elif os.path.exists(expr_sparse_path):
                info["expr_format"] = "csr_npz"
                info["expr_path"] = expr_sparse_path
            else:
                raise FileNotFoundError(f"No expression file found under {shard_dir}")

            if info.get("n_samples") is None:
                raise ValueError(f"Missing n_samples for shard {info['shard_idx']} in {shard_dir}")

            info["memory_gb"] = os.path.getsize(info["expr_path"]) / (1024 ** 3)

        if rank == 0:
            print(f"  Total shards: {self.n_shards}")
            for info in self.shard_info:
                print(
                    f"    Shard {info['shard_idx']}: {info['n_samples']:,} samples, "
                    f"{info['expr_format']}, ~{info['memory_gb']:.1f} GB"
                )

        # Use a fixed epoch length to keep DistributedSampler stable.
        self.fixed_length = min(info['n_samples'] for info in self.shard_info)
        if rank == 0:
            print(f"  Fixed length per epoch: {self.fixed_length:,}")

        # Shards are loaded lazily in set_epoch().

    def load_shard(self, shard_idx: int):
        """Load one shard with mmap-backed arrays."""
        if PrebuiltDataset._shared_shard_idx == shard_idx:
            return

        shard_dir = os.path.join(self.training_data_path, f'shard_{shard_idx}')
        lock_dir = os.path.join(self.training_data_path, '.locks')
        os.makedirs(lock_dir, exist_ok=True)

        import gc

        # Let lower ranks finish opening the shard first.
        for prev_rank in range(self.rank):
            done_file = os.path.join(lock_dir, f'rank_{prev_rank}_done')
            while not os.path.exists(done_file):
                time.sleep(1)

        gc.collect()
        print(f"  [Rank {self.rank}] Loading shard {shard_idx} (mmap)...")
        start_time = time.time()

        expr_dense_path = os.path.join(shard_dir, "expr_all.npy")
        expr_sparse_path = os.path.join(shard_dir, "expr_sparse.npz")
        sample_indices_path = os.path.join(shard_dir, "sample_indices.npy")
        neighbor_indices_path = os.path.join(shard_dir, "neighbor_indices.npy")

        if not os.path.exists(sample_indices_path) or not os.path.exists(neighbor_indices_path):
            raise FileNotFoundError(
                f"Shard {shard_idx} is incomplete: expected sample_indices.npy and neighbor_indices.npy in {shard_dir}"
            )

        if os.path.exists(expr_dense_path):
            PrebuiltDataset._shared_expr = np.load(expr_dense_path, mmap_mode="r")
            PrebuiltDataset._shared_expr_format = "dense"
        elif os.path.exists(expr_sparse_path):
            PrebuiltDataset._shared_expr = CSRRowAccessor(expr_sparse_path)
            PrebuiltDataset._shared_expr_format = "csr_npz"
        else:
            raise FileNotFoundError(f"No expression file found for shard {shard_idx} in {shard_dir}")

        PrebuiltDataset._shared_sample_indices = np.load(sample_indices_path, mmap_mode="r")
        PrebuiltDataset._shared_neighbor_indices = np.load(neighbor_indices_path, mmap_mode="r")
        self._prepare_cce_index(shard_dir)

        PrebuiltDataset._shared_shard_idx = shard_idx
        print(f"  [Rank {self.rank}] Done in {time.time()-start_time:.1f}s")

        # Mark this rank as ready.
        done_file = os.path.join(lock_dir, f'rank_{self.rank}_done')
        with open(done_file, 'w') as f:
            f.write('done')

        # Wait for all ranks and then clean up lock files.
        if self.world_size > 1:
            dist.barrier()
            if self.rank == 0:
                for r in range(self.world_size):
                    try:
                        os.remove(os.path.join(lock_dir, f'rank_{r}_done'))
                    except:
                        pass

    def set_epoch(self, epoch: int):
        """Switch shard once per epoch."""
        shard_idx = epoch % self.n_shards
        self.load_shard(shard_idx)

    def __len__(self):
        return self.fixed_length

    def _prepare_cce_index(self, shard_dir: str) -> None:
        shard_meta_path = os.path.join(shard_dir, "metadata.pkl")
        with open(shard_meta_path, "rb") as f:
            shard_meta = pickle.load(f)

        batch_metadata = shard_meta.get("batch_metadata", [])
        if not batch_metadata:
            self._sample_cell_type_ids = None
            self._sample_rows_sorted_by_type = None
            self._type_row_starts = None
            self._type_row_counts = None
            self._cell_type_names = None
            return

        type_to_id: Dict[str, int] = {}
        segment_type_ids = np.empty(len(batch_metadata), dtype=np.int32)
        row_ends = np.empty(len(batch_metadata), dtype=np.int64)
        for i, item in enumerate(batch_metadata):
            cell_type = item["cell_type"]
            type_id = type_to_id.setdefault(cell_type, len(type_to_id))
            segment_type_ids[i] = type_id
            row_ends[i] = item["row_end"]

        self._cell_type_names = [None] * len(type_to_id)
        for name, idx in type_to_id.items():
            self._cell_type_names[idx] = name

        sample_indices = np.asarray(PrebuiltDataset._shared_sample_indices, dtype=np.int64)
        segment_ids = np.searchsorted(row_ends, sample_indices, side="right")
        sample_cell_type_ids = segment_type_ids[segment_ids]
        sample_rows = np.arange(len(sample_indices), dtype=np.int64)
        order = np.argsort(sample_cell_type_ids, kind="stable")
        sorted_type_ids = sample_cell_type_ids[order]
        unique_type_ids, start_idx, counts = np.unique(sorted_type_ids, return_index=True, return_counts=True)

        type_row_starts = np.full((len(type_to_id),), -1, dtype=np.int64)
        type_row_counts = np.zeros((len(type_to_id),), dtype=np.int64)
        type_row_starts[unique_type_ids] = start_idx
        type_row_counts[unique_type_ids] = counts

        self._sample_cell_type_ids = sample_cell_type_ids
        self._sample_rows_sorted_by_type = sample_rows[order]
        self._type_row_starts = type_row_starts
        self._type_row_counts = type_row_counts

    def _build_item(self, actual_idx: int) -> Dict[str, torch.Tensor]:
        cell_idx = PrebuiltDataset._shared_sample_indices[actual_idx]
        neighbor_idx = PrebuiltDataset._shared_neighbor_indices[actual_idx]
        neighbor_idx = np.atleast_1d(np.asarray(neighbor_idx, dtype=np.int64))
        neighbor_relation_ids = build_neighbor_relation_ids(len(neighbor_idx))

        if neighbor_idx.ndim > 0 and len(neighbor_idx) > self.n_neighbors:
            sampled_idx = np.random.choice(len(neighbor_idx), size=self.n_neighbors, replace=False)
            neighbor_idx = neighbor_idx[sampled_idx]
            neighbor_relation_ids = neighbor_relation_ids[sampled_idx]

        if PrebuiltDataset._shared_expr_format == "dense":
            cell_expr = np.asarray(PrebuiltDataset._shared_expr[cell_idx], dtype=np.float32)
            neighbor_exprs = np.asarray(PrebuiltDataset._shared_expr[neighbor_idx], dtype=np.float32)
        elif PrebuiltDataset._shared_expr_format == "csr_npz":
            cell_expr = PrebuiltDataset._shared_expr.get_rows(np.array([cell_idx]))[0]
            neighbor_exprs = PrebuiltDataset._shared_expr.get_rows(neighbor_idx)
        else:
            raise RuntimeError(f"Unsupported shared expr format: {PrebuiltDataset._shared_expr_format}")

        # Some prebuilt shards include four leading special-token columns.
        # Remove them when expression width exceeds the gene-id width by four.
        if cell_expr.shape[0] > len(self.gene_ids):
            if cell_expr.shape[0] == len(self.gene_ids) + 4:
                cell_expr = cell_expr[4:]
                neighbor_exprs = neighbor_exprs[:, 4:]
            else:
                raise ValueError(f"cell_expr has unexpected number of genes: {cell_expr.shape[0]} vs vocab {len(self.gene_ids)}")

        if self.gene_truncation_enable:
            cell_expr, neighbor_exprs, gene_ids = apply_gene_truncation(
                cell_expr=cell_expr,
                neighbor_exprs=neighbor_exprs,
                gene_ids=self.gene_ids,
                target_length=self.gene_truncation_length,
                nonzero_ratio=self.gene_truncation_nonzero_ratio,
                diff_eps=self.gene_truncation_diff_eps,
                pad_gene_id=self.padding_gene_id,
                pad_value=self.pad_value,
            )
        else:
            gene_ids = self.gene_ids

        cell_tensor = torch.as_tensor(cell_expr, dtype=torch.float32)
        neighbor_matrix = torch.as_tensor(neighbor_exprs, dtype=torch.float32)
        gene_ids_tensor = torch.as_tensor(gene_ids, dtype=torch.long)
        neighbor_relation_ids_tensor = torch.as_tensor(neighbor_relation_ids, dtype=torch.long)

        return {
            'expr': cell_tensor,
            'gene_ids': gene_ids_tensor,
            'neighbor_matrix': neighbor_matrix,
            'neighbor_relation_ids': neighbor_relation_ids_tensor,
            'sample_row': torch.tensor(actual_idx, dtype=torch.long),
            'cell_idx': torch.tensor(int(cell_idx), dtype=torch.long),
        }

    def sample_cce_batch(self, sample_rows: torch.Tensor | np.ndarray | List[int]) -> Dict[str, torch.Tensor]:
        if self._sample_cell_type_ids is None:
            raise RuntimeError("CCE index is not prepared for the current shard")

        sample_rows = np.asarray(sample_rows, dtype=np.int64).reshape(-1)
        items = []
        for sample_row in sample_rows:
            cell_type_id = int(self._sample_cell_type_ids[sample_row])
            start = int(self._type_row_starts[cell_type_id])
            count = int(self._type_row_counts[cell_type_id])
            if start < 0 or count <= 0:
                positive_row = int(sample_row)
            else:
                candidates = self._sample_rows_sorted_by_type[start:start + count]
                positive_row = int(candidates[np.random.randint(count)])
                if count > 1 and positive_row == int(sample_row):
                    for alt in candidates:
                        if int(alt) != int(sample_row):
                            positive_row = int(alt)
                            break

            items.append(self._build_item(positive_row))

        batch = collate_fn(items)
        assert batch is not None
        return batch

    def __getitem__(self, idx):
        actual_idx = idx % len(PrebuiltDataset._shared_sample_indices)
        return self._build_item(actual_idx)


def collate_fn(batch):
    if not batch:
        return None

    out = {}
    for key in batch[0].keys():
        value0 = batch[0][key]
        if torch.is_tensor(value0):
            out[key] = torch.stack([item[key] for item in batch])
        else:
            out[key] = [item[key] for item in batch]
    return out
