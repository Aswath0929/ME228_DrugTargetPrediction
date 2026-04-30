"""Evaluation utilities for DTA regression.

This script evaluates saved fold checkpoints produced by train.py.

Typical usage:
  python evaluate.py --dataset DAVIS --output_dir outputs

It will:
  - Load dataset (same preprocessing as training)
  - Recreate the same K-fold splits
  - For each fold, load best checkpoint, run prediction on validation split
  - Report CI, MSE, Pearson r per fold and averages

Note: This uses the same metric implementations as train.py for consistency.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from config import Config
from data.dataset import DtiDataset, load_dataset_items, load_tdc_splits, make_kfold_splits
from models.siamese_dta import build_model_from_config


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
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    n = y_true.size
    if n < 2:
        return float("nan")

    order = np.argsort(y_true, kind="mergesort")
    yt = y_true[order]
    yp = y_pred[order]

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

        group_ranks = rank[i:j]
        for r in group_ranks:
            less = ft.sum(r - 1)
            leq = ft.sum(r)
            eq = leq - less
            greater = prev_count - leq

            concordant += less
            ties += eq
            comparable += less + eq + greater

        for r in group_ranks:
            ft.add(int(r), 1)
            prev_count += 1

        i = j

    if comparable == 0:
        return float("nan")

    return float((concordant + 0.5 * ties) / comparable)


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys: List[np.ndarray] = []
    ps: List[np.ndarray] = []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        ys.append(batch.y.view(-1).detach().cpu().numpy())
        ps.append(out["pred"].view(-1).detach().cpu().numpy())

    y = np.concatenate(ys, axis=0) if ys else np.array([])
    p = np.concatenate(ps, axis=0) if ps else np.array([])
    return y, p


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, choices=["DAVIS", "KIBA"], default="DAVIS")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--use_tdc_splits", action="store_true")
    p.add_argument("--multitask_enabled", action="store_true")
    p.add_argument("--family_mapping_path", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config(dataset=args.dataset)  # type: ignore[arg-type]
    if args.batch_size is not None:
        cfg.batch_size = int(args.batch_size)  # type: ignore[misc]
    if args.device is not None:
        cfg.device = str(args.device)  # type: ignore[misc]
    if args.use_tdc_splits:
        cfg.use_tdc_splits = True  # type: ignore[misc]
    if args.multitask_enabled:
        cfg.multitask_enabled = True  # type: ignore[misc]
    if args.family_mapping_path is not None:
        cfg.family_mapping_path = args.family_mapping_path  # type: ignore[misc]

    device = torch.device("cuda" if cfg.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    def infer_num_tasks(items: List[object]) -> int:
        if not cfg.multitask_enabled:
            return 1
        fam_ids = [int(getattr(i, "family_id")) for i in items]
        return max(fam_ids) + 1 if fam_ids else 1

    if cfg.use_tdc_splits:
        train_items, val_items, test_items = load_tdc_splits(cfg)
        num_tasks = infer_num_tasks(train_items + val_items + test_items)

        ckpt_path = os.path.join(args.output_dir, "best_fold_0.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

        test_ds = DtiDataset(test_items, cfg=cfg)
        sample = test_ds[0]

        model = build_model_from_config(cfg, node_feature_dim=int(sample.x.shape[1]), num_tasks=num_tasks).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
        y, p = predict(model, loader, device)

        ci = concordance_index(y, p)
        mse = mean_squared_error(y, p)
        r = pearsonr(y, p)

        print(f"Test: CI={ci:.4f} | MSE={mse:.4f} | r={r:.4f}")
    else:
        items = load_dataset_items(cfg)
        num_tasks = infer_num_tasks(items)
        splits = make_kfold_splits(len(items), num_folds=cfg.num_folds, seed=cfg.seed)

        metrics: List[Dict[str, float]] = []

        print(f"Evaluating dataset={cfg.dataset} | folds={cfg.num_folds} | device={device}")

        for fold_idx, (_, val_idx) in enumerate(splits):
            ckpt_path = os.path.join(args.output_dir, f"best_fold_{fold_idx}.pt")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

            val_items = [items[int(i)] for i in val_idx]
            val_ds = DtiDataset(val_items, cfg=cfg)
            sample = val_ds[0]

            model = build_model_from_config(cfg, node_feature_dim=int(sample.x.shape[1]), num_tasks=num_tasks).to(device)
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])

            loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
            y, p = predict(model, loader, device)

            ci = concordance_index(y, p)
            mse = mean_squared_error(y, p)
            r = pearsonr(y, p)

            m = {"ci": float(ci), "mse": float(mse), "pearson": float(r)}
            metrics.append(m)

            print(f"Fold {fold_idx}: CI={ci:.4f} | MSE={mse:.4f} | r={r:.4f}")

        def avg(key: str) -> float:
            return float(np.nanmean([m.get(key, float('nan')) for m in metrics]))

        print("\nAVG : CI={:.4f} | MSE={:.4f} | r={:.4f}".format(avg("ci"), avg("mse"), avg("pearson")))


if __name__ == "__main__":
    main()
