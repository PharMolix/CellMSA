import argparse
import json
import random
from pathlib import Path

import anndata as ad
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _chunk_order(path, stem):
    suffix = path.stem.removeprefix(f"{stem}_embeddings_")
    start, end = suffix.rstrip("w").split("-")
    return int(start), int(end)


def load_embedding(h5ad_path, embedding_path, concat_emb_dir=None):
    """Load one embedding matrix, concatenate chunks, or read h5ad obsm."""
    h5ad_path = Path(h5ad_path)
    embedding_source = str(embedding_path)

    if embedding_source.startswith("obsm."):
        obsm_key = embedding_source.removeprefix("obsm.")
        if not obsm_key:
            raise ValueError("--embedding obsm.KEY requires a non-empty KEY")
        adata = ad.read_h5ad(h5ad_path, backed="r")
        try:
            if obsm_key not in adata.obsm:
                available = ", ".join(adata.obsm.keys())
                raise KeyError(f"{obsm_key!r} not found in h5ad.obsm. Available keys: {available}")
            emb = np.asarray(adata.obsm[obsm_key])
        finally:
            adata.file.close()
        return emb, [f"{h5ad_path}:obsm[{obsm_key!r}]"]

    embedding_path = Path(embedding_source)

    if embedding_source == "concat":
        if concat_emb_dir is None:
            raise ValueError("concat mode needs concat_emb_dir")
        emb_dir = Path(concat_emb_dir)
        stem = h5ad_path.stem
        emb_paths = sorted(
            emb_dir.glob(f"{stem}_embeddings_*-*.npy"),
            key=lambda path: _chunk_order(path, stem),
        )
        if not emb_paths:
            raise FileNotFoundError(f"No chunks found: {emb_dir}/{stem}_embeddings_*-*.npy")
        emb = np.concatenate([np.load(path) for path in emb_paths], axis=0)
        return emb, emb_paths

    if embedding_path.is_dir():
        stem = h5ad_path.stem
        emb_paths = sorted(
            embedding_path.glob(f"{stem}_embeddings_*-*.npy"),
            key=lambda path: _chunk_order(path, stem),
        )
        if emb_paths:
            emb = np.concatenate([np.load(path) for path in emb_paths], axis=0)
            return emb, emb_paths
        raise FileNotFoundError(f"No chunks found in directory: {embedding_path}")

    if not embedding_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
    return np.load(embedding_path), [embedding_path]


def load_data(h5ad_path, embedding_path, label_key, concat_emb_dir=None):
    adata = ad.read_h5ad(h5ad_path, backed="r")
    obs = adata.obs.copy()
    n_obs = adata.n_obs
    adata.file.close()

    if label_key not in obs.columns:
        raise KeyError(f"{label_key!r} not found in h5ad.obs")

    emb, emb_paths = load_embedding(h5ad_path, embedding_path, concat_emb_dir)
    if emb.shape[0] != n_obs:
        raise ValueError(f"Embedding rows ({emb.shape[0]}) != h5ad n_obs ({n_obs})")

    labels_raw = obs[label_key].astype(str).to_numpy()
    valid_mask = (labels_raw != "nan") & (labels_raw != "None") & (labels_raw != "")
    if not valid_mask.all():
        emb = emb[valid_mask]
        obs = obs.loc[valid_mask].copy()
        labels_raw = labels_raw[valid_mask]

    classes, labels = np.unique(labels_raw, return_inverse=True)
    return emb.astype(np.float32), labels.astype(np.int64), classes, obs, emb_paths


def parse_obs_filter_values(value):
    if not value:
        return []
    value = value.strip()
    if value.startswith("["):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError("--obs-filter-values JSON input must be a list")
        return [str(item) for item in parsed]
    return [item.strip() for item in value.split(",") if item.strip()]


def filter_obs_by_values(x, y, obs, filter_key, filter_values):
    if filter_key is None:
        return x, y, obs, None
    if filter_key not in obs.columns:
        raise KeyError(f"{filter_key!r} not found in h5ad.obs")
    if not filter_values:
        raise ValueError("--obs-filter-values must contain at least one value when --obs-filter-key is set")

    values = [str(value) for value in filter_values]
    obs_values = obs[filter_key].astype(str).to_numpy()
    keep_mask = np.isin(obs_values, values)
    if not keep_mask.any():
        raise ValueError(f"No cells matched {filter_key!r} in {values}")

    stats = {
        "key": filter_key,
        "values": values,
        "n_cells_before": int(len(y)),
        "n_cells_after": int(keep_mask.sum()),
    }
    return x[keep_mask], y[keep_mask], obs.loc[keep_mask].copy(), stats


def relabel_classes(labels, classes):
    labels_raw = classes[labels]
    new_classes, new_labels = np.unique(labels_raw, return_inverse=True)
    stats = {
        "classes_before": classes.tolist(),
        "classes_after": new_classes.tolist(),
        "n_classes_before": int(len(classes)),
        "n_classes_after": int(len(new_classes)),
    }
    return new_labels.astype(np.int64), new_classes, stats


def split_indices(obs, labels, split, val_size=0.1, test_size=0.2, seed=0, donor_key="donor_id"):
    if val_size <= 0 or test_size <= 0:
        raise ValueError(f"val_size and test_size must be positive, got {val_size}, {test_size}")
    if val_size + test_size >= 1:
        raise ValueError(
            f"val_size + test_size must be < 1, got {val_size + test_size:.3f}"
        )

    idx = np.arange(len(labels))
    if split == "random":
        stratify = labels if len(np.unique(labels)) > 1 else None
        train_val_idx, test_idx = train_test_split(
            idx,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
        relative_val_size = val_size / (1.0 - test_size)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=relative_val_size,
            random_state=seed,
            stratify=labels[train_val_idx] if stratify is not None else None,
        )
        return train_idx, val_idx, test_idx

    if donor_key not in obs.columns:
        raise KeyError(f"{donor_key!r} not found in h5ad.obs")

    donors = obs[donor_key].astype(str).to_numpy()
    unique_donors = np.unique(donors)
    if len(unique_donors) < 3:
        raise ValueError(f"Need at least 3 unique donors for train/val/test split, got {len(unique_donors)}")

    train_val_donors, test_donors = train_test_split(
        unique_donors,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )
    relative_val_size = val_size / (1.0 - test_size)
    train_donors, val_donors = train_test_split(
        train_val_donors,
        test_size=relative_val_size,
        random_state=seed,
        shuffle=True,
    )
    train_mask = np.isin(donors, train_donors)
    val_mask = np.isin(donors, val_donors)
    test_mask = np.isin(donors, test_donors)
    return idx[train_mask], idx[val_mask], idx[test_mask]


def filter_and_downsample_donors(x, y, obs, donor_key, min_cells=100, max_cells=1000, seed=0):
    if donor_key not in obs.columns:
        raise KeyError(f"{donor_key!r} not found in h5ad.obs")
    if min_cells < 1:
        raise ValueError(f"min_cells must be >= 1, got {min_cells}")
    if max_cells < 1:
        raise ValueError(f"max_cells must be >= 1, got {max_cells}")
    if max_cells < min_cells:
        raise ValueError(f"max_cells ({max_cells}) must be >= min_cells ({min_cells})")

    donors = obs[donor_key].astype(str).to_numpy()
    unique_donors, donor_counts = np.unique(donors, return_counts=True)
    kept_donors = unique_donors[donor_counts >= min_cells]
    if len(kept_donors) == 0:
        raise ValueError(f"No donors with at least {min_cells} cells for {donor_key!r}")

    rng = np.random.default_rng(seed)
    selected_indices = []
    downsampled_donors = 0
    for donor in kept_donors:
        donor_indices = np.flatnonzero(donors == donor)
        if len(donor_indices) > max_cells:
            donor_indices = rng.choice(donor_indices, size=max_cells, replace=False)
            downsampled_donors += 1
        selected_indices.append(donor_indices)

    selected_indices = np.sort(np.concatenate(selected_indices))
    filtered_obs = obs.iloc[selected_indices].copy()
    stats = {
        "donor_key": donor_key,
        "min_cells": min_cells,
        "max_cells": max_cells,
        "n_cells_before": int(len(y)),
        "n_cells_after": int(len(selected_indices)),
        "n_donors_before": int(len(unique_donors)),
        "n_donors_after": int(len(kept_donors)),
        "n_downsampled_donors": int(downsampled_donors),
    }
    return x[selected_indices], y[selected_indices], filtered_obs, stats


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=(256, 128), dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def make_loader(x, y, indices, batch_size, shuffle):
    x_tensor = torch.from_numpy(x[indices])
    y_tensor = torch.from_numpy(y[indices])
    return DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch_size, shuffle=shuffle)


def weighted_accuracy_score(y_true, y_pred):
    classes, counts = np.unique(y_true, return_counts=True)
    class_weights = {cls: 1.0 / count for cls, count in zip(classes, counts)}
    sample_weight = np.array([class_weights[label] for label in y_true], dtype=np.float64)
    return accuracy_score(y_true, y_pred, sample_weight=sample_weight)


def make_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = np.zeros(num_classes, dtype=np.float32)
    present = counts > 0
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    return weights


def l2_regularization(model):
    l2_norm = None
    for param in model.parameters():
        if not param.requires_grad:
            continue
        term = param.pow(2).sum()
        l2_norm = term if l2_norm is None else l2_norm + term
    return l2_norm


def format_confusion_matrix(matrix, classes):
    class_names = [str(cls) for cls in classes]
    row_labels = [f"true:{name}" for name in class_names]
    col_labels = [f"pred:{name}" for name in class_names]
    row_header = "true\\pred"
    widths = [
        max(len(row_header), *(len(label) for label in row_labels)),
        *[
            max(len(label), *(len(str(matrix[row_idx, col_idx])) for row_idx in range(matrix.shape[0])))
            for col_idx, label in enumerate(col_labels)
        ],
    ]
    lines = [
        " ".join(
            [row_header.rjust(widths[0])]
            + [label.rjust(widths[col_idx + 1]) for col_idx, label in enumerate(col_labels)]
        )
    ]
    for row_idx, row_label in enumerate(row_labels):
        lines.append(
            " ".join(
                [row_label.rjust(widths[0])]
                + [
                    str(matrix[row_idx, col_idx]).rjust(widths[col_idx + 1])
                    for col_idx in range(matrix.shape[1])
                ]
            )
        )
    return "\n".join(lines)


def majority_vote(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes)
    return int(np.flatnonzero(counts == counts.max())[0])


def vote_probabilities_by_group(y_prob, groups, classes):
    num_classes = len(classes)
    voted_pred = np.empty(y_prob.shape[0], dtype=np.int64)
    vote_rows = []

    for donor_id in np.unique(groups):
        mask = groups == donor_id
        mean_prob = y_prob[mask].mean(axis=0)
        pred_vote = int(mean_prob.argmax())

        voted_pred[mask] = pred_vote
        vote_rows.append(
            {
                "donor_id": str(donor_id),
                "n_cells": int(mask.sum()),
                "pred_label": int(pred_vote),
                "pred_class": str(classes[pred_vote]),
                "pred_probability": float(mean_prob[pred_vote]),
                "mean_probabilities": mean_prob.tolist(),
            }
        )

    return voted_pred, vote_rows


def evaluate(model, loader, device):
    model.eval()
    losses = []
    preds = []
    probs = []
    labels = []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            losses.append(loss.item() * x_batch.size(0))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            preds.append(logits.argmax(dim=1).cpu().numpy())
            labels.append(y_batch.cpu().numpy())

    y_true = np.concatenate(labels)
    y_pred = np.concatenate(preds)
    y_prob = np.concatenate(probs)
    return {
        "loss": float(np.sum(losses) / len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_accuracy": float(weighted_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def train_mlp(
    x,
    y,
    train_idx,
    val_idx,
    test_idx,
    hidden_dims=(256, 128),
    dropout=0.2,
    lr=1e-3,
    weight_decay=1e-4,
    l2_lambda=0.0,
    weighted_loss=True,
    batch_size=256,
    epochs=50,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = int(y.max()) + 1
    model = MLPClassifier(x.shape[1], num_classes, hidden_dims, dropout).to(device)
    train_loader = make_loader(x, y, train_idx, batch_size, shuffle=True)
    val_loader = make_loader(x, y, val_idx, batch_size, shuffle=False)
    test_loader = make_loader(x, y, test_idx, batch_size, shuffle=False)
    class_weights = make_class_weights(y[train_idx], num_classes)
    class_weight_tensor = torch.from_numpy(class_weights).to(device) if weighted_loss else None
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = []
    best_state = None
    best_val_weighted_acc = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            l2_loss = torch.zeros((), device=device)
            if l2_lambda > 0:
                l2_loss = 0.5 * l2_lambda * l2_regularization(model)
                loss = loss + l2_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)
            total_seen += x_batch.size(0)

        val_metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": float(total_loss / total_seen),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_weighted_accuracy": val_metrics["weighted_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "weighted_loss": weighted_loss,
            "l2_lambda": l2_lambda,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} "
            f"train_loss={row['train_loss']:.4f} "
            f"val_acc={row['val_accuracy']:.4f} "
            f"val_weighted_acc={row['val_weighted_accuracy']:.4f} "
            f"val_macro_f1={row['val_macro_f1']:.4f} "
            f"val_weighted_f1={row['val_weighted_f1']:.4f}"
        )
        if val_metrics["weighted_accuracy"] > best_val_weighted_acc:
            best_val_weighted_acc = val_metrics["weighted_accuracy"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    final_metrics = evaluate(model, test_loader, device)
    return model, final_metrics, history, class_weights


def parse_hidden_dims(value):
    if not value:
        return ()
    return tuple(int(item) for item in value.split(","))


def main():
    parser = argparse.ArgumentParser(description="Train/validate/test an MLP classifier on h5ad embeddings.")
    parser.add_argument("--h5ad", required=True, help="Path to h5ad file.")
    parser.add_argument(
        "--embedding",
        required=True,
        help="Path to .npy embedding file, a chunk directory, literal 'concat', or 'obsm.KEY'.",
    )
    parser.add_argument(
        "--concat-emb-dir",
        default=None,
        help="Directory with {h5ad_stem}_embeddings_0-1.npy or 0-1w.npy chunks when --embedding concat.",
    )
    parser.add_argument("--obs-filter-key", default=None, help="Optional h5ad.obs column used to filter cells.")
    parser.add_argument(
        "--obs-filter-values",
        default=None,
        help='Allowed values for --obs-filter-key, either comma-separated or JSON list.',
    )
    parser.add_argument("--label-key", required=True, help="Column in h5ad.obs used as labels, e.g. disease.")
    parser.add_argument("--split", choices=["random", "donor"], default="random")
    parser.add_argument("--donor-key", default="donor_id")
    parser.add_argument("--min-donor-cells", type=int, default=100)
    parser.add_argument("--max-donor-cells", type=int, default=1000)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dims", default="256,128")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--l2-lambda", type=float, default=0.0)
    parser.add_argument("--no-weighted-loss", action="store_false", dest="weighted_loss")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="output/mlp_classifier")
    parser.set_defaults(weighted_loss=True)
    args = parser.parse_args()
    if args.obs_filter_values is not None and args.obs_filter_key is None:
        raise ValueError("--obs-filter-key is required when --obs-filter-values is set")

    set_seed(args.seed)
    x, y, classes, obs, emb_paths = load_data(
        args.h5ad,
        args.embedding,
        args.label_key,
        concat_emb_dir=args.concat_emb_dir,
    )
    obs_filter_values = parse_obs_filter_values(args.obs_filter_values)
    x, y, obs, obs_filter_stats = filter_obs_by_values(
        x,
        y,
        obs,
        args.obs_filter_key,
        obs_filter_values,
    )
    x, y, obs, donor_filter_stats = filter_and_downsample_donors(
        x,
        y,
        obs,
        args.donor_key,
        min_cells=args.min_donor_cells,
        max_cells=args.max_donor_cells,
        seed=args.seed,
    )
    y, classes, class_filter_stats = relabel_classes(y, classes)
    train_idx, val_idx, test_idx = split_indices(
        obs,
        y,
        args.split,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        donor_key=args.donor_key,
    )
    print(f"loaded h5ad: {args.h5ad}")
    print(f"loaded embeddings: {[str(path) for path in emb_paths]}")
    if obs_filter_stats is not None:
        print(f"obs filter: {obs_filter_stats}")
    print(f"donor filter: {donor_filter_stats}")
    print(f"class filter: {class_filter_stats}")
    print(f"x shape: {x.shape}")
    print(f"classes: {dict(enumerate(classes.tolist()))}")
    print(
        f"split: {args.split}, "
        f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
    )
    print(f"weighted_loss: {args.weighted_loss}")
    print(f"l2_lambda: {args.l2_lambda}, weight_decay: {args.weight_decay}")

    model, metrics, history, class_weights = train_mlp(
        x,
        y,
        train_idx,
        val_idx,
        test_idx,
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        l2_lambda=args.l2_lambda,
        weighted_loss=args.weighted_loss,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
    )

    print("\nclassification report:")
    labels = np.arange(len(classes))
    print(
        classification_report(
            metrics["y_true"],
            metrics["y_pred"],
            labels=labels,
            target_names=classes,
            zero_division=0,
        )
    )
    conf_mat = confusion_matrix(metrics["y_true"], metrics["y_pred"], labels=labels)
    print(f"accuracy: {metrics['accuracy']:.4f}")
    print(f"weighted_accuracy: {metrics['weighted_accuracy']:.4f}")
    print(f"macro_f1: {metrics['macro_f1']:.4f}")
    print(f"weighted_f1: {metrics['weighted_f1']:.4f}")
    print("\nconfusion matrix:")
    print(format_confusion_matrix(conf_mat, classes))

if __name__ == "__main__":
    main()
