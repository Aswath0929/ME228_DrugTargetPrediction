"""Utilities for converting SMILES strings into torch-geometric graphs.

This module uses RDKit to parse SMILES and builds a torch_geometric.data.Data
object with:
  - x: node (atom) feature matrix
  - edge_index: COO adjacency (2, E)
  - edge_attr: optional bond features (kept minimal; extend if needed)

Node features follow the project spec:
  - atomic number (one-hot over configured bins)
  - degree (one-hot)
  - total valence (one-hot)
  - aromaticity (binary)
  - total hydrogen count (one-hot)

We keep the featurization deterministic and driven by Config so it’s easy to
reproduce and to adjust.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

try:
    from rdkit import Chem
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "RDKit is required for SMILES parsing. Install rdkit (e.g., via conda-forge)."
    ) from e

try:
    from torch_geometric.data import Data
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "torch-geometric is required. Install torch-geometric matching your PyTorch version."
    ) from e


def _one_hot_with_unknown(value: int, allowed: Sequence[int]) -> List[int]:
    """One-hot encode `value` over `allowed` with an extra 'unknown' bin."""
    out = [0] * (len(allowed) + 1)
    try:
        idx = allowed.index(value)
    except ValueError:
        idx = len(allowed)
    out[idx] = 1
    return out


def atom_features(
    atom: "Chem.Atom",
    *,
    atomic_numbers: Sequence[int],
    degrees: Sequence[int],
    valences: Sequence[int],
    num_hs: Sequence[int],
) -> List[int]:
    """Compute the node feature vector for a single RDKit atom."""

    feats: List[int] = []
    feats += _one_hot_with_unknown(atom.GetAtomicNum(), atomic_numbers)
    feats += _one_hot_with_unknown(atom.GetDegree(), degrees)
    feats += _one_hot_with_unknown(atom.GetTotalValence(), valences)
    feats += [1 if atom.GetIsAromatic() else 0]
    feats += _one_hot_with_unknown(atom.GetTotalNumHs(), num_hs)
    return feats


def bond_features(bond: "Chem.Bond") -> List[int]:
    """Minimal bond features.

    Not required by your spec, but useful for future upgrades. We include:
    - is_single, is_double, is_triple, is_aromatic
    """

    bt = bond.GetBondType()
    return [
        1 if bt == Chem.rdchem.BondType.SINGLE else 0,
        1 if bt == Chem.rdchem.BondType.DOUBLE else 0,
        1 if bt == Chem.rdchem.BondType.TRIPLE else 0,
        1 if bond.GetIsAromatic() else 0,
    ]


def smiles_to_pyg_data(
    smiles: str,
    *,
    atomic_numbers: Sequence[int],
    degrees: Sequence[int],
    valences: Sequence[int],
    num_hs: Sequence[int],
    add_hs: bool = False,
) -> Data:
    """Convert a SMILES string into a torch-geometric Data object.

    Args:
        smiles: Input SMILES.
        atomic_numbers/degrees/valences/num_hs: Bins used for one-hot features.
        add_hs: If True, explicit hydrogens are added before building the graph.

    Returns:
        torch_geometric.data.Data with fields: x, edge_index, edge_attr.

    Raises:
        ValueError: If the SMILES cannot be parsed.
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    if add_hs:
        mol = Chem.AddHs(mol)

    # Node features
    x_list: List[List[int]] = [
        atom_features(
            atom,
            atomic_numbers=atomic_numbers,
            degrees=degrees,
            valences=valences,
            num_hs=num_hs,
        )
        for atom in mol.GetAtoms()
    ]

    x = torch.tensor(x_list, dtype=torch.float)

    # Edges (bidirectional)
    edge_index_list: List[Tuple[int, int]] = []
    edge_attr_list: List[List[int]] = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)

        edge_index_list.append((i, j))
        edge_attr_list.append(bf)
        edge_index_list.append((j, i))
        edge_attr_list.append(bf)

    if len(edge_index_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 4), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


def smiles_to_pyg_data_from_config(smiles: str, cfg: "object") -> Data:
    """Convenience wrapper using a config-like object.

    `cfg` is expected to expose:
      - atom_atomic_numbers
      - atom_degrees
      - atom_total_valence
      - atom_total_num_hs
    """

    return smiles_to_pyg_data(
        smiles,
        atomic_numbers=getattr(cfg, "atom_atomic_numbers"),
        degrees=getattr(cfg, "atom_degrees"),
        valences=getattr(cfg, "atom_total_valence"),
        num_hs=getattr(cfg, "atom_total_num_hs"),
    )
