
from dataclasses import dataclass

@dataclass
class Config:

    # CellMSA parameters
    d_model: int = 512       # embedding dim
    d_pair: int = 8          # pairwise embedding dim
    msa_depth: int = 4
    pairformer_depth: int = 6
    n_neighbors: int = 40
    msa_dim: int = 128
    msa_outer_product_mean_dim_hidden: int = 8
    msa_pwa_dropout_row_prob: float = 0.15
    msa_pwa_heads: int = 8
    msa_pwa_dim_head: int = 32
    pairformer_pair_bias_attn_dim_head: int = 32
    pairformer_pair_bias_attn_heads: int = 8
    pairformer_dropout_row_prob: float = 0.25

    # Training parameters
    batch_size: int = 16
    learning_rate: float = 1e-5
    epochs: int = 1
    mlm_probability: float = 0.15
    mask_replace_prob: float = 0.8
    random_replace_prob: float = 0.1
    mask_value: float = -1.0
    pad_value: float = 0.0
    pad_token: str = "<pad>"
    gradient_accumulation_steps: int = 16
    use_binned_input: bool = True
    n_bins: int = 11
    value_emb_style: str = "category"
    use_cce: bool = True
    cce_weight: float = 10.
    cce_temp: float = 0.05
    cce_interval: int = 1
    use_recon: bool = True
    recon_probability: float = 0.10
    recon_weight: float = 0.01

    # Warmup and learning-rate schedule
    warmup_steps: int = 1000

    # Data parameters
    gene_truncation_enable: bool = True
    gene_truncation_length: int = 2048
    gene_truncation_nonzero_ratio: float = 1.0
    gene_truncation_diff_eps: float = 1e-4
    max_cells_per_batch: int = 500
    max_cells_to_load: int = 5000
    min_cells_per_batch: int = 10

    # Paths
    output_dir: str = "output"

    # DDP
    local_rank: int = -1
    world_size: int = 1

    # Misc
    log_interval: int = 100
    save_interval: int = 500
    amp: bool = True
    num_workers: int = 4
    seed: int = 42
