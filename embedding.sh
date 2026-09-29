#!/usr/bin/env bash
# Single-GPU inference. Override paths/settings with environment variables.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON="${PYTHON:-python}"
CHECKPOINT="${CHECKPOINT:-$SCRIPT_DIR/best_model.pt}"
QUERY_H5AD="${QUERY_H5AD:-$SCRIPT_DIR/demo/data/blood_subset.h5ad}"
MSA_H5AD="${MSA_H5AD:-$QUERY_H5AD}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-$SCRIPT_DIR/output/blood_subset_cellmsa}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"
CHUNK_SIZE="${CHUNK_SIZE:-100000}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
for input_file in "$CHECKPOINT" "$QUERY_H5AD" "$MSA_H5AD"; do
  if [[ ! -s "$input_file" ]]; then
    echo "Missing input: $input_file" >&2
    exit 1
  fi
done
# Do not combine new chunks with files from an earlier run.
shopt -s nullglob
existing=("${OUTPUT_PREFIX}"_embeddings_*.npy)
if ((${#existing[@]})); then
  echo "Output chunks already exist for $OUTPUT_PREFIX; choose a new OUTPUT_PREFIX." >&2
  exit 1
fi
exec "$PYTHON" embedding.py \
  --batch_size "$BATCH_SIZE" \
  --resume "$CHECKPOINT" \
  --query_h5ad "$QUERY_H5AD" \
  --msa_h5ad "$MSA_H5AD" \
  --var_key_gene_name "${VAR_KEY_GENE_NAME:-var_names}" \
  --output_prefix "$OUTPUT_PREFIX" \
  --num_workers "$NUM_WORKERS" \
  --chunk_size "$CHUNK_SIZE" \
  "$@"
