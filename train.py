"""Training script for Siamese DTA regression with 5-fold cross-validation.

Usage (after installing deps like tdc/pandas):
  python train.py --dataset DAVIS
  python train.py --dataset KIBA

This script:
  - Loads the dataset via PyTDC
  - Builds k-fold splits
  - Trains a Siamese GNN+CNN model
  - Saves best checkpoint per fold based on validation CI
  - Prints per-fold and average metrics

Note: For Windows environments without UTF-8 console, avoid fancy unicode output.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam

from torch_geometric.loader import DataLoader

from config import Config, DatasetName
from data.dataset import DtiDataset, load_dataset_items, load_tdc_splits, make_kfold_splits
from losses.combined_loss import CombinedLoss
from models.siamese_dta import build_model_from_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(cfg: Config) -> torch.device:
    if cfg.device.lower().startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))


def pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    if y_true.size < 2:
        return float("nan")
    yt = y_true - y_true.mean()
    yp = y_pred - y_pred.mean()
    denom = (np.sqrt(np.sum(yt**2)) * np.sqrt(np.sum(yp**2)))
    if denom == 0:
        return float("nan")
    return float(np.sum(yt * yp) / denom)


class _Fenwick:
    def __init__(self, n: int) -> None:
        self.n = n
        self.bit = np.zeros(n + 1, dtype=np.int64)

    def add(self, i: int, v: int) -> None:
        while i <= self.n:
            self.bit[i] += v
            i += i & -i

    def sum(self, i: int) -> int:
        s = 0
        while i > 0:
            s += int(self.bit[i])
            i -= i & -i
        return s


def concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute concordance index (CI) in O(n log n).

    CI considers all comparable pairs with different true values.
    For each pair (i, j) where y_true[i] != y_true[j], it is concordant if the
    ordering of predictions matches the ordering of truths.

    This implementation:
      - Sorts by y_true
      - Processes groups of equal y_true together (excluded from comparables)
      - Uses a Fenwick tree over discretized y_pred ranks
    """

    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    n = y_true.size
    if n < 2:
        return float("nan")

    order = np.argsort(y_true, kind="mergesort")
    yt = y_true[order]
    yp = y_pred[order]

    # Discretize predictions to ranks 1..M
    uniq = np.unique(yp)
    rank = np.searchsorted(uniq, yp) + 1
    ft = _Fenwick(len(uniq))

    concordant = 0
    ties = 0
    comparable = 0

    i = 0
    prev_count = 0
    while i < n:
        j = i
        while j < n and yt[j] == yt[i]:
            j += 1

        # Count against all previous groups (lower y_true)
        group_ranks = rank[i:j]
        for r in group_ranks:
            less = ft.sum(r - 1)
            leq = ft.sum(r)
            eq = leq - less
            greater = prev_count - leq

            concordant += less
            ties += eq
            comparable += less + eq + greater

        # Add this group into Fenwick (so within-group pairs aren't counted)
        for r in group_ranks:
            ft.add(int(r), 1)
            prev_count += 1

        i = j

    if comparable == 0:
        return float("nan")

    return float((concordant + 0.5 * ties) / comparable)


@torch.no_grad()
def predict_on_loader(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, float]:
    model.eval()

    ys: List[float] = []
    ps: List[float] = []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        pred = out["pred"]
        y_true = batch.y.view(-1)

        ys.append(y_true.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())

        n_batches += 1

    y_all = np.concatenate(ys, axis=0) if ys else np.array([])
    p_all = np.concatenate(ps, axis=0) if ps else np.array([])
    return y_all, p_all, total_loss


def train_one_fold(
    *,
    cfg: Config,
    fold_idx: int,
    train_items,
    val_items,
    device: torch.device,
    output_dir: str,
    num_tasks: int,
) -> Dict[str, float]:
    """Train and validate a single fold; returns best val metrics."""

    train_ds = DtiDataset(train_items, cfg=cfg, cache_graphs=True, cache_proteins=True)
    val_ds = DtiDataset(val_items, cfg=cfg, cache_graphs=True, cache_proteins=True)

    # Infer node feature dim from a single sample
    sample = train_ds[0]
    node_feature_dim = int(sample.x.shape[1])

    model = build_model_from_config(cfg, node_feature_dim=node_feature_dim, num_tasks=num_tasks).to(device)
    criterion = CombinedLoss(cfg=cfg)
    opt = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and device.type == "cuda",
    )

    best_ci = float("-inf")
    best_metrics: Dict[str, float] = {}

    ckpt_path = os.path.join(output_dir, f"best_fold_{fold_idx}.pt")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        n_steps = 0

        for batch in train_loader:
            batch = batch.to(device)
            out = model(batch)

            loss = criterion(
                pred=out["pred"],
                y_true=batch.y.view(-1),
                drug_emb=out["drug_emb"],
                protein_emb=out["protein_emb"],
                is_nonbinder=getattr(batch, "is_nonbinder", None),
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += float(loss.detach().cpu().item())
            n_steps += 1

        # Validation
        y_val, p_val, _ = predict_on_loader(model, val_loader, device)
        ci = concordance_index(y_val, p_val)
        mse = mean_squared_error(y_val, p_val)
        r = pearsonr(y_val, p_val)

        # CI can be NaN for degenerate cases (e.g., very small validation set).
        # We still want a deterministic "best" checkpoint.
        ci_score = ci
        if not np.isfinite(ci_score):
            ci_score = float("-inf")

        if ci_score > best_ci or not best_metrics:
            best_ci = ci_score
            best_metrics = {"ci": float(ci), "mse": float(mse), "pearson": float(r)}
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg.to_dict(),
                    "fold": fold_idx,
                    "epoch": epoch,
                    "val_metrics": best_metrics,
                },
                ckpt_path,
            )

        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
            avg_loss = running / max(n_steps, 1)
            print(
                f"Fold {fold_idx} | Epoch {epoch:03d} | train_loss={avg_loss:.4f} | val_CI={ci:.4f} | val_MSE={mse:.4f} | val_r={r:.4f}"
            )

    print(f"Fold {fold_idx} best | CI={best_metrics.get('ci', float('nan')):.4f} | MSE={best_metrics.get('mse', float('nan')):.4f} | r={best_metrics.get('pearson', float('nan')):.4f}")
    print(f"Saved: {ckpt_path}")

    return best_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, choices=["DAVIS", "KIBA"], default="DAVIS")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--use_tdc_splits", action="store_true")
    p.add_argument("--multitask_enabled", action="store_true")
    p.add_argument("--family_mapping_path", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config(dataset=args.dataset)  # type: ignore[arg-type]
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir  # type: ignore[misc]
    if args.epochs is not None:
        cfg.epochs = int(args.epochs)  # type: ignore[misc]
    if args.batch_size is not None:
        cfg.batch_size = int(args.batch_size)  # type: ignore[misc]
    if args.lr is not None:
        cfg.lr = float(args.lr)  # type: ignore[misc]
    if args.use_tdc_splits:
        cfg.use_tdc_splits = True  # type: ignore[misc]
    if args.multitask_enabled:
        cfg.multitask_enabled = True  # type: ignore[misc]
    if args.family_mapping_path is not None:
        cfg.family_mapping_path = args.family_mapping_path  # type: ignore[misc]

    os.makedirs(cfg.output_dir, exist_ok=True)

    set_seed(cfg.seed)
    device = get_device(cfg)

    # Save config snapshot
    with open(os.path.join(cfg.output_dir, f"config_{cfg.dataset}.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    print(f"Dataset: {cfg.dataset} | device={device} | folds={cfg.num_folds}")

    def infer_num_tasks(items: List[object]) -> int:
        if not cfg.multitask_enabled:
            return 1
        fam_ids = [int(getattr(i, "family_id")) for i in items]
        return max(fam_ids) + 1 if fam_ids else 1

    if cfg.use_tdc_splits:
        train_items, val_items, test_items = load_tdc_splits(cfg)
        num_tasks = infer_num_tasks(train_items + val_items + test_items)

        print(f"\n=== TDC split | train={len(train_items)} val={len(val_items)} test={len(test_items)} ===")

        metrics = train_one_fold(
            cfg=cfg,
            fold_idx=0,
            train_items=train_items,
            val_items=val_items,
            device=device,
            output_dir=cfg.output_dir,
            num_tasks=num_tasks,
        )

        # Evaluate on test split using best checkpoint
        test_ds = DtiDataset(test_items, cfg=cfg, cache_graphs=True, cache_proteins=True)
        sample = test_ds[0]
        model = build_model_from_config(
            cfg,
            node_feature_dim=int(sample.x.shape[1]),
            num_tasks=num_tasks,
        ).to(device)
        ckpt_path = os.path.join(cfg.output_dir, "best_fold_0.pt")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory and device.type == "cuda",
        )
        y_test, p_test, _ = predict_on_loader(model, test_loader, device)
        ci = concordance_index(y_test, p_test)
        mse = mean_squared_error(y_test, p_test)
        r = pearsonr(y_test, p_test)

        print(f"\n=== Test metrics ===\nCI={ci:.4f} | MSE={mse:.4f} | r={r:.4f}")
    else:
        items = load_dataset_items(cfg)
        num_tasks = infer_num_tasks(items)
        splits = make_kfold_splits(len(items), num_folds=cfg.num_folds, seed=cfg.seed)

        fold_metrics: List[Dict[str, float]] = []

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            train_items = [items[int(i)] for i in train_idx]
            val_items = [items[int(i)] for i in val_idx]

            print(f"\n=== Fold {fold_idx} | train={len(train_items)} val={len(val_items)} ===")

            metrics = train_one_fold(
                cfg=cfg,
                fold_idx=fold_idx,
                train_items=train_items,
                val_items=val_items,
                device=device,
                output_dir=cfg.output_dir,
                num_tasks=num_tasks,
            )
            fold_metrics.append(metrics)

        # Aggregate
        def avg(key: str) -> float:
            vals = [m.get(key, float("nan")) for m in fold_metrics]
            return float(np.nanmean(vals))

        print("\n=== Cross-validation summary ===")
        for k in range(len(fold_metrics)):
            m = fold_metrics[k]
            print(f"Fold {k}: CI={m.get('ci', float('nan')):.4f} | MSE={m.get('mse', float('nan')):.4f} | r={m.get('pearson', float('nan')):.4f}")
        print(f"AVG : CI={avg('ci'):.4f} | MSE={avg('mse'):.4f} | r={avg('pearson'):.4f}")


if __name__ == "__main__":
    main()
