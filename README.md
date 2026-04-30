# ME228 Drug-Target Prediction

This project trains a Siamese drug-target affinity (DTA) regression model with a graph neural network (GIN) drug encoder and a 1D CNN protein encoder. It supports DAVIS and KIBA datasets via PyTDC, optional official TDC splits, and an optional multitask head for protein family groupings.

## What It Does

- Converts SMILES strings to molecular graphs (RDKit + PyG).
- Encodes proteins with a 1D CNN over amino-acid tokens.
- Fuses drug and protein embeddings and predicts affinity.
- Trains with MSE plus a contrastive penalty on non-binders.
- Supports k-fold cross-validation or official TDC train/valid/test splits.
- Optional multitask regression head using family IDs.

## Setup

Create and activate a Python environment, then install dependencies:

```
pip install torch torchvision torchaudio
pip install torch-geometric
pip install pytdc pandas numpy rdkit
```

Notes:
- Torch Geometric wheels must match your PyTorch version. See https://pytorch-geometric.readthedocs.io/ for install guidance.
- RDKit is easiest to install from conda-forge if pip fails.

## Train

K-fold training (default):

```
python train.py --dataset DAVIS --output_dir outputs_davis
python train.py --dataset KIBA --output_dir outputs_kiba
```

Official TDC splits (single train/val/test run):

```
python train.py --dataset DAVIS --use_tdc_splits --output_dir outputs_davis
```

Enable multitask head (optional):

```
python train.py --dataset DAVIS --use_tdc_splits --multitask_enabled --output_dir outputs_davis
```

Provide a family mapping CSV (optional):

```
python train.py --dataset DAVIS --use_tdc_splits --multitask_enabled \
  --family_mapping_path path/to/family_map.csv --output_dir outputs_davis
```

The CSV must contain columns:
- target_id
- family_id

## Evaluate

K-fold evaluation:

```
python evaluate.py --dataset DAVIS --output_dir outputs_davis
```

TDC split evaluation:

```
python evaluate.py --dataset DAVIS --use_tdc_splits --output_dir outputs_davis
```

With multitask head:

```
python evaluate.py --dataset DAVIS --use_tdc_splits --multitask_enabled --output_dir outputs_davis
```

## Project Structure

- config.py: hyperparameters and dataset settings
- train.py: training script
- evaluate.py: evaluation script
- data/: dataset loading and graph utilities
- models/: model architecture (GIN + CNN + fusion)
- losses/: combined loss (MSE + contrastive)
- outputs_smoketest/: sample checkpoints

## Notes

- Non-binder thresholds are configured in config.py per dataset.
- If no family mapping is provided, each target is treated as its own family.
