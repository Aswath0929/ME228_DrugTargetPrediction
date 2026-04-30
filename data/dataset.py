"""Dataset utilities for Drug-Target Interaction (DTI) prediction.

This module is responsible for:
  - Loading DAVIS / KIBA via PyTDC
  - Optional label normalization (e.g., DAVIS Kd -> pKd)
  - Protein sequence tokenization + padding/truncation
  - Building a torch.utils.data.Dataset that returns torch-geometric Data objects

Design choice:
We attach the protein tensor and regression label directly onto the returned
PyG Data object. This allows torch_geometric.loader.DataLoader to batch both
the molecular graphs and the per-sample tensors cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Tuple

import math

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from config import Config, DatasetName
from data.graph_utils import smiles_to_pyg_data

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


def encode_protein_sequence(
    sequence: str,
    *,
    vocab: Mapping[str, int],
    max_len: int,
) -> Tensor:
    """Convert an amino acid sequence into a fixed-length integer tensor.

    - Unknown characters map to vocab["<UNK>"]
    - Output is padded/truncated to `max_len` using vocab["<PAD>"]

    Returns:
        LongTensor of shape (max_len,)
    """

    pad_idx = vocab.get("<PAD>", 0)
    unk_idx = vocab.get("<UNK>", 1)

    seq = (sequence or "").strip().upper()
    out = torch.full((max_len,), fill_value=pad_idx, dtype=torch.long)

    n = min(len(seq), max_len)
    if n == 0:
        return out

    ids = [vocab.get(ch, unk_idx) for ch in seq[:n]]
    out[:n] = torch.tensor(ids, dtype=torch.long)
    return out


def maybe_convert_davis_y_to_pkd(y: float) -> float:
    """Convert DAVIS affinity to pKd if it looks like raw Kd.

    Many DAVIS pipelines use pKd = -log10(Kd in M).

    If Kd is given in nM, then:
        pKd = -log10(Kd * 1e-9) = 9 - log10(Kd)

    PyTDC may return either pKd-like values (typically ~5-10) or raw Kd values.
    We apply a lightweight heuristic:
      - if y > 20, treat as Kd in nM and convert to pKd
      - else, assume y is already on a log/PKd-like scale
    """

    if not np.isfinite(y):
        return float(y)

    if y > 20.0:
        # Treat as Kd in nM
        return float(9.0 - math.log10(max(y, 1e-12)))

    return float(y)


def load_tdc_dti_dataframe(name: DatasetName) -> "pd.DataFrame":
    """Load DAVIS/KIBA using PyTDC and return a normalized DataFrame.

    Output columns are standardized to:
      - smiles (str)
      - sequence (str)
      - y (float)
    """

    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise ImportError("pandas is required for dataset loading") from e

    try:
        from tdc.multi_pred import DTI
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "PyTDC is required. Install with `pip install pytdc` (package name: tdc)."
        ) from e

    data = DTI(name=name)
    df = data.get_data()

    # Expected PyTDC columns: Drug, Target, Y
    required = {"Drug", "Target", "Y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"TDC returned missing columns {sorted(missing)}; got {list(df.columns)}")

    out_cols = ["Drug", "Target", "Y"]
    if "Target_ID" in df.columns:
        out_cols.append("Target_ID")

    out = df[out_cols].copy()
    out = out.rename(columns={"Drug": "smiles", "Target": "sequence", "Y": "y", "Target_ID": "target_id"})

    # Ensure types
    out["smiles"] = out["smiles"].astype(str)
    out["sequence"] = out["sequence"].astype(str)
    out["y"] = out["y"].astype(float)
    if "target_id" not in out.columns:
        out["target_id"] = out["sequence"].astype(str)
    else:
        out["target_id"] = out["target_id"].astype(str)

    return out


def make_kfold_splits(
    n: int,
    *,
    num_folds: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Simple K-fold split generator (no sklearn dependency)."""

    if num_folds < 2:
        raise ValueError("num_folds must be >= 2")
    if n <= 0:
        raise ValueError("n must be > 0")

    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    folds = np.array_split(indices, num_folds)

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for k in range(num_folds):
        val_idx = folds[k]
        train_idx = np.concatenate([folds[i] for i in range(num_folds) if i != k])
        splits.append((train_idx, val_idx))

    return splits


@dataclass(frozen=True)
class DtiItem:
    """A single dataset row after preprocessing."""

    smiles: str
    sequence: str
    y: float
    target_id: str
    family_id: int


class DtiDataset(Dataset):
    """PyTorch Dataset returning PyG Data objects with protein+label attached."""

    def __init__(
        self,
        items: Sequence[DtiItem],
        *,
        cfg: Config,
        cache_graphs: bool = True,
        cache_proteins: bool = True,
    ) -> None:
        self.cfg = cfg
        self.items = list(items)

        self._graph_cache: Dict[str, "object"] = {} if cache_graphs else {}
        self._protein_cache: Dict[str, Tensor] = {} if cache_proteins else {}
        self._cache_graphs = cache_graphs
        self._cache_proteins = cache_proteins

    def __len__(self) -> int:
        return len(self.items)

    def _get_graph(self, smiles: str):
        if self._cache_graphs and smiles in self._graph_cache:
            return self._graph_cache[smiles]

        g = smiles_to_pyg_data(
            smiles,
            atomic_numbers=self.cfg.atom_atomic_numbers,
            degrees=self.cfg.atom_degrees,
            valences=self.cfg.atom_total_valence,
            num_hs=self.cfg.atom_total_num_hs,
        )

        if self._cache_graphs:
            self._graph_cache[smiles] = g
        return g

    def _get_protein(self, sequence: str) -> Tensor:
        if self._cache_proteins and sequence in self._protein_cache:
            return self._protein_cache[sequence]

        t = encode_protein_sequence(
            sequence,
            vocab=self.cfg.protein_vocab,
            max_len=self.cfg.max_protein_len,
        )
        if self._cache_proteins:
            self._protein_cache[sequence] = t
        return t

    def _process_y(self, y: float) -> float:
        if self.cfg.dataset == "DAVIS":
            return maybe_convert_davis_y_to_pkd(float(y))
        return float(y)

    def _is_nonbinder(self, y_processed: float) -> bool:
        if self.cfg.dataset == "DAVIS":
            # Non-binder if pKd < 6 (we treat y_processed as pKd-like)
            return y_processed < self.cfg.davis_nonbinder_pkd_threshold

        # KIBA: user-specified non-binder rule (typically higher score = weaker)
        return y_processed > self.cfg.kiba_nonbinder_score_threshold

    def __getitem__(self, idx: int):
        item = self.items[idx]

        y = self._process_y(item.y)
        g = self._get_graph(item.smiles)
        p = self._get_protein(item.sequence)

        # Important: return a fresh object each time. PyG Batch may mutate fields.
        data = g.clone() if hasattr(g, "clone") else g

        # Store as [1, L] so torch_geometric Batch stacks it to [B, L].
        data.protein = p.view(1, -1).clone()
        data.y = torch.tensor([y], dtype=torch.float32)
        data.is_nonbinder = torch.tensor([self._is_nonbinder(y)], dtype=torch.bool)
        data.family_id = torch.tensor([int(item.family_id)], dtype=torch.long)

        return data


def _load_family_mapping(path: Optional[str]) -> Dict[str, int]:
    if not path:
        return {}

    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise ImportError("pandas is required for family mapping") from e

    df = pd.read_csv(path)
    if "target_id" not in df.columns or "family_id" not in df.columns:
        raise ValueError("family mapping CSV must contain target_id and family_id columns")

    return {str(r.target_id): int(r.family_id) for r in df.itertuples(index=False)}


def _assign_family_ids(df: "pd.DataFrame", mapping: Dict[str, int]) -> Dict[str, int]:
    """Return a target_id -> family_id map, falling back to per-target grouping."""

    if mapping:
        # Preserve provided family ids, but ensure all targets are covered.
        fam = dict(mapping)
    else:
        fam = {}

    next_id = max(fam.values(), default=-1) + 1
    for target_id in df["target_id"].astype(str).unique():
        if target_id not in fam:
            fam[target_id] = next_id
            next_id += 1
    return fam


def dataframe_to_items(df: "pd.DataFrame", *, family_map: Dict[str, int]) -> List[DtiItem]:
    """Convert a normalized DataFrame into DtiItem records."""

    items: List[DtiItem] = []
    for r in df.itertuples(index=False):
        target_id = str(getattr(r, "target_id"))
        items.append(
            DtiItem(
                smiles=str(r.smiles),
                sequence=str(r.sequence),
                y=float(r.y),
                target_id=target_id,
                family_id=int(family_map[target_id]),
            )
        )
    return items


def load_dataset_items(cfg: Config) -> List[DtiItem]:
    """Load the configured dataset and return preprocessed items."""

    df = load_tdc_dti_dataframe(cfg.dataset)
    mapping = _load_family_mapping(cfg.family_mapping_path)
    family_map = _assign_family_ids(df, mapping)
    return dataframe_to_items(df, family_map=family_map)


def load_tdc_splits(cfg: Config) -> Tuple[List[DtiItem], List[DtiItem], List[DtiItem]]:
    """Load official TDC train/valid/test splits when available."""

    try:
        from tdc.multi_pred import DTI
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "PyTDC is required. Install with `pip install pytdc` (package name: tdc)."
        ) from e

    data = DTI(name=cfg.dataset)
    split = data.get_split()

    def _normalize(df):
        df = df.copy()
        cols = ["Drug", "Target", "Y"]
        if "Target_ID" in df.columns:
            cols.append("Target_ID")
        df = df[cols]
        df = df.rename(columns={"Drug": "smiles", "Target": "sequence", "Y": "y", "Target_ID": "target_id"})
        df["smiles"] = df["smiles"].astype(str)
        df["sequence"] = df["sequence"].astype(str)
        df["y"] = df["y"].astype(float)
        if "target_id" not in df.columns:
            df["target_id"] = df["sequence"].astype(str)
        else:
            df["target_id"] = df["target_id"].astype(str)
        return df

    train_df = _normalize(split["train"])
    valid_df = _normalize(split.get("valid") or split.get("val") or split["test"])
    test_df = _normalize(split.get("test") or split["valid"])

    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise ImportError("pandas is required for dataset loading") from e

    full_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

    mapping = _load_family_mapping(cfg.family_mapping_path)
    family_map = _assign_family_ids(full_df, mapping)

    return (
        dataframe_to_items(train_df, family_map=family_map),
        dataframe_to_items(valid_df, family_map=family_map),
        dataframe_to_items(test_df, family_map=family_map),
    )
