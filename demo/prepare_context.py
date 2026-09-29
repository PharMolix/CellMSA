#!/usr/bin/env python3
"""Prepare label-informed 16/16/8 demo context without overwriting source data."""
import argparse
from pathlib import Path


def sample_context(obs, seed=0):
    """Use the original notebook's positional-neighbor sampling rules."""
    import numpy as np
    required = ["cell_type", "donor_id", "supertype"]
    missing = [key for key in required if key not in obs]
    if missing:
        raise ValueError(f"Missing observation columns: {missing}")
    if obs.empty:
        raise ValueError("The demo input has no cells.")
    if obs[required].isna().any().any():
        raise ValueError("Demo context annotations must not contain missing values.")
    obs = obs[required].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    same_ct_exp = {
        key: np.asarray(idx, dtype=np.int32)
        for key, idx in obs.groupby(["cell_type", "donor_id"], sort=False, observed=False).indices.items()
    }
    same_ct = {
        key: np.asarray(idx, dtype=np.int32)
        for key, idx in obs.groupby("cell_type", sort=False, observed=False).indices.items()
    }
    same_super = {
        key: np.asarray(idx, dtype=np.int32)
        for key, idx in obs.groupby("supertype", sort=False, observed=False).indices.items()
    }
    same_ct_diff_exp = {}
    for cell_type, cell_idx in same_ct.items():
        donor_ids = obs.iloc[cell_idx]["donor_id"].to_numpy()
        for donor_id in np.unique(donor_ids):
            same_ct_diff_exp[(cell_type, donor_id)] = cell_idx[donor_ids != donor_id]
    same_super_diff_ct = {}
    for supertype, super_idx in same_super.items():
        cell_types = obs.iloc[super_idx]["cell_type"].to_numpy()
        for cell_type in np.unique(cell_types):
            same_super_diff_ct[(supertype, cell_type)] = super_idx[cell_types != cell_type]

    def sample(pool, size, self_pos):
        pool = np.asarray(pool, dtype=np.int32)
        pool = pool[pool != self_pos]
        if pool.size == 0:
            return np.full(size, -1, dtype=np.int32)
        return rng.choice(pool, size=size, replace=pool.size < size).astype(np.int32)

    result = {key: np.empty((len(obs), width), dtype=np.int32)
              for key, width in [("stsb", 16), ("stdb", 16), ("dt", 8)]}
    for i, (cell_type, donor_id, supertype) in enumerate(obs.itertuples(index=False, name=None)):
        result["stsb"][i] = sample(same_ct_exp[(cell_type, donor_id)], 16, i)
        result["stdb"][i] = sample(same_ct_diff_exp[(cell_type, donor_id)], 16, i)
        result["dt"][i] = sample(same_super_diff_ct[(supertype, cell_type)], 8, i)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Use a separate output path to preserve the original dataset.")
    if args.output.exists():
        parser.error("Output already exists; choose a new path or explicitly remove the generated copy.")
    if not args.input.is_file():
        parser.error(f"Input file not found: {args.input}.")
    import anndata as ad
    adata = ad.read_h5ad(args.input)
    neighbors = sample_context(adata.obs, args.seed)
    # The reader prioritizes generic indices; remove stale ones from this copy.
    if "neighbor_indices" in adata.obsm:
        del adata.obsm["neighbor_indices"]
    for key, values in neighbors.items():
        adata.obsm[key] = values
        print(f"{key}: shape={values.shape}, missing={(values < 0).sum()}")
    adata.uns["cellmsa_demo_context"] = {"seed": args.seed, "mode": "label-informed"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.output)
    print(f"Prepared context written to {args.output}")


if __name__ == "__main__":
    main()
