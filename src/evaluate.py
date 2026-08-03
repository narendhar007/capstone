"""evaluate.py — Stage 3: evaluation metrics, plots, failure-case analysis. 

Positive class = DEFECT (recall on defects is the headline QC metric). Implement
prediction, imbalance-aware metrics, confusion/ROC plots, and a misclassified-sample list.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


@torch.no_grad()
def predict(net, loader, return_embeddings: bool = False):
    # TODO 3: return (y_true, y_pred, y_prob_defect[, embeddings]) over the loader.
    raise NotImplementedError


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    # TODO 3: accuracy, precision/recall/f1 for the DEFECT class (pos_label=config.POSITIVE_IDX),
    #         macro_f1, roc_auc, confusion_matrix. Return a dict.
    raise NotImplementedError


def plot_eval(y_true, y_pred, y_prob, out: Path | None = None) -> Path:
    # TODO 3: save a confusion-matrix + ROC figure to artifacts/model_eval.png.
    raise NotImplementedError


def failure_cases(items, y_true, y_pred, y_prob, limit: int = 20) -> list[dict]:
    # TODO 3: list misclassified samples with predicted p_defect (error analysis).
    raise NotImplementedError
