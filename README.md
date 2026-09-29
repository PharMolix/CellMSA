# CellMSA: Context Modeling for Single-Cell Representation Learning

Learning informative single-cell representations requires understanding both a cell's own expression profile and its biological context. We propose **CellMSA**, a framework that models each target cell together with related cells from different batches and biologically related cell types. Inspired by MSA-based modeling in AlphaFold 3, CellMSA captures cross-cell patterns as context-dependent gene-pair representations and uses them to guide a GenePairformer cell encoder. The resulting representations support batch integration, cell type/state analysis, and downstream perturbation modeling.

More information can be found in our paper.

# Install

[![Python 3.12](https://img.shields.io/badge/python-3.12-brightgreen)](https://www.python.org/)

```bash
pip install -r requirements.txt
pip install 'jupyterlab>=4,<5'
```

Embedding inference requires an NVIDIA GPU. Please install a [PyTorch 2.5.x build](https://pytorch.org/get-started/previous-versions/) compatible with your CUDA version.

# Checkpoint

The pretrained checkpoint is provided in `best_model.pt`, including the model weights and gene vocabulary. For loading details, please refer to `embedding.py`.

The checkpoint and demo data are stored with [Git LFS](https://git-lfs.com/). After cloning the repository, download them with:

```bash
git lfs install
git lfs pull
```

The blood demo and its precomputed embeddings are provided in [`demo/data/`](demo/data/).

# Usage

- **Data preprocess**
Prepare an AnnData file with nonnegative expression values in `.X` and Ensembl gene IDs in `.var_names`. The blood demo already includes context indices built from cell-type and donor annotations. To rebuild its context:

```bash
python demo/prepare_context.py --input demo/data/blood_subset.h5ad --output output/blood_subset_context.h5ad
```

- **Get embeddings**
Use `embedding.sh` to obtain cell embeddings. The default input is `demo/data/blood_subset.h5ad`; output is saved to `output/blood_subset_cellmsa_embeddings_0-5000.npy`. For custom data, set `QUERY_H5AD`, `MSA_H5AD`, and `OUTPUT_PREFIX`; the query file must contain context indices into the reference file.

```bash
CUDA_VISIBLE_DEVICES=0 bash embedding.sh
```

- **Interactive demo**
We recommend starting with [`demo/demo.ipynb`](demo/demo.ipynb) to explore context preparation, embedding inference, and visualization. A blood dataset containing 5,000 cells and its precomputed embeddings are included.

```bash
jupyter lab demo/demo.ipynb
```

- **Downstream tasks**
After obtaining cell embeddings, use `mlp_classifier.py` to train a classification head. Run `python mlp_classifier.py --help` for available options. Use context constructed without validation/test target labels for label-free evaluation.

# Citation

If you find CellMSA helpful to your research, please consider giving this repository a 🌟star and citing our paper.

```bibtex
@misc{zhao2026cellmsa,
  title  = {CellMSA: Context Modeling for Single-Cell Representation Learning},
  author = {Zhao, Suyuan and Liu, Minghao and Luo, Yizhen and Nie, Zaiqing},
  year   = {2026}
}
```
