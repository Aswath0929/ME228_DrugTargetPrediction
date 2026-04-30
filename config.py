"""Project-wide configuration.

Keep *all* hyperparameters, constants, and thresholds in this file so training/eval
scripts and modules stay clean and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Mapping, Optional, Sequence

DatasetName = Literal["DAVIS", "KIBA"]


def default_protein_vocab() -> Dict[str, int]:
    """Return a 25-token protein vocabulary mapping.

    We reserve:
    - 0 for PAD
    - 1 for UNK (unknown/rare amino acids)

    Then map the 20 canonical amino acids plus common ambiguity tokens.
    """

    tokens = [
        "<PAD>",
        "<UNK>",
        "A",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "K",
        "L",
        "M",
        "N",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "V",
        "W",
        "Y",
        # Ambiguity / uncommon tokens (counted in the requested vocab size)
        "B",  # D or N
        "Z",  # E or Q
        "X",  # unknown amino acid
    ]

    if len(tokens) != 25:
        raise ValueError(f"Expected 25 protein tokens, got {len(tokens)}")

    return {tok: i for i, tok in enumerate(tokens)}


@dataclass
class Config:
    # -------------------------
    # Reproducibility / runtime
    # -------------------------
    seed: int = 42
    device: str = "cuda"  # train.py will fall back to cpu if cuda unavailable

    # --------
    # Dataset
    # --------
    dataset: DatasetName = "DAVIS"
    num_folds: int = 5
    use_tdc_splits: bool = False

    # Optional CSV mapping target_id -> family_id for multitask learning.
    # If None, we fall back to using target_id as the family grouping.
    family_mapping_path: Optional[str] = None
    multitask_enabled: bool = False

    # For DAVIS, many papers work in pKd (higher is stronger). We’ll enforce a
    # consistent "higher is stronger" convention in dataset.py.
    davis_nonbinder_pkd_threshold: float = 6.0

    # For KIBA, higher is stronger; the user-specified threshold is used as-is.
    kiba_nonbinder_score_threshold: float = 12.1

    # Protein encoding
    max_protein_len: int = 1000
    protein_vocab: Dict[str, int] = field(default_factory=default_protein_vocab)

    # --------------
    # Graph features
    # --------------
    # One-hot bins for atom featurization; unknown values go to the last bin.
    atom_atomic_numbers: List[int] = field(
        default_factory=lambda: [
            1,
            5,
            6,
            7,
            8,
            9,
            14,
            15,
            16,
            17,
            19,
            20,
            26,
            29,
            30,
            33,
            34,
            35,
            53,
        ]
    )
    atom_degrees: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    atom_total_valence: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6])
    atom_total_num_hs: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])

    # -----------------
    # Model hyperparams
    # -----------------
    embedding_dim: int = 256
    dropout: float = 0.1

    # Drug (GIN) encoder
    gin_num_layers: int = 3
    gin_hidden_dim: int = 256

    # Protein (1D CNN) encoder
    protein_embed_dim: int = 128
    protein_conv_channels: Sequence[int] = (32, 64, 96)
    protein_conv_kernel_size: int = 8

    # Regression head
    mlp_hidden_dims: Sequence[int] = (1024, 512)

    # -----------------
    # Loss / objectives
    # -----------------
    mse_weight: float = 1.0
    contrastive_weight: float = 0.1

    # For non-binder pairs, we discourage high similarity between the *drug* and
    # *protein* embeddings. If cosine_sim > nonbinder_max_cosine, we penalize.
    nonbinder_max_cosine: float = 0.2

    # --------
    # Training
    # --------
    lr: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 256
    epochs: int = 100

    # Dataloader
    num_workers: int = 0  # set >0 if your Windows env supports it reliably
    pin_memory: bool = True

    # Checkpointing / logging
    output_dir: str = "outputs"

    def nonbinder_threshold(self) -> float:
        """Return the affinity threshold used to define non-binders."""
        if self.dataset == "DAVIS":
            return self.davis_nonbinder_pkd_threshold
        return self.kiba_nonbinder_score_threshold

    @property
    def protein_vocab_size(self) -> int:
        return len(self.protein_vocab)

    def to_dict(self) -> Mapping[str, object]:
        """A JSON-serializable-ish view for logging."""
        # Avoid pulling in asdict() recursion for the vocab mapping; keep it simple.
        return {
            **{k: v for k, v in self.__dict__.items() if k != "protein_vocab"},
            "protein_vocab_size": self.protein_vocab_size,
        }
