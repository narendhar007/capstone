"""train.py — Stage 2/3: transfer-learning training + MLflow tracking + registry. 

Implement: build splits, train the ResNet18 head (CrossEntropy, Adam on trainable params,
early stop on val F1), log params/metrics/model to MLflow, register + promote to the
@production alias, evaluate on test, and save a clean-data reference baseline (image
features + embeddings) for drift monitoring.   Run: python -m src.train
"""
from __future__ import annotations

import json, random
from pathlib import Path
import numpy as np, torch, torch.nn as nn

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src import data_prep, evaluate
from src.dataset import CastingDataset
from src.model import build_model, trainable_parameters, save_model, EmbeddingExtractor
from torch.utils.data import DataLoader


def set_seed(seed: int = config.RANDOM_SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _subsample(items, cap):
    # TODO: stratified subsample to `cap` (or return items if cap falsy/too small).
    raise NotImplementedError


def class_weights(items) -> torch.Tensor:
    # TODO: inverse-frequency class weights for CrossEntropyLoss.
    raise NotImplementedError


def save_reference_baseline(net, ref_items) -> dict:
    # TODO 4: save reference_features.csv + reference_embeddings.npz for clean ref images.
    raise NotImplementedError


def main() -> int:
    import mlflow, mlflow.pytorch
    from mlflow import MlflowClient
    set_seed()
    root = data_prep.find_data_root()
    qc = data_prep.validate_quality(root)
    (config.ARTIFACT_DIR / "data_quality_report.json").write_text(json.dumps(qc, indent=2))
    data_prep.build_splits(root, "v1")
    # TODO 2/3: load splits, subsample train, build loaders, build model + optimiser + loss.
    # TODO 3 (MLflow): set_experiment; start_run; log_params; per-epoch log_metrics; early stop.
    # TODO 3 (eval + registry): test metrics; plot_eval; save_model; log_model + register +
    #         set @production alias; write model_meta.json + metrics.json.
    # TODO 4: save_reference_baseline on a clean val sample.
    raise NotImplementedError("Implement the training + MLflow + registry workflow")


if __name__ == "__main__":
    raise SystemExit(main())
