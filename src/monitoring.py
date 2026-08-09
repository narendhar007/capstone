"""monitoring.py — Stage 4: statistical + embedding drift + confidence monitoring.

Implement the three drift signals against the clean reference baseline:
  1. statistical drift  — Evidently DataDriftPreset + PSI on image features
  2. embedding drift    — PSI on ResNet-embedding distance-to-centroid distribution
  3. confidence         — mean predicted confidence reference vs current
Use a corrupted copy of clean images as the simulated "current" production batch.
Outputs drift_report.html + drift_summary.json.   Run: python -m src.monitoring

Embedding drift (TODO 4) — see the conceptual walkthrough in
Operations_Monitoring_and_Evidence.ipynb (Stage 4.3):
  1. feature extraction  — penultimate 512-dim ResNet embedding (model.EmbeddingExtractor)
  2. embedding generation — embeddings for reference + current batches
  3. feature-space compare — reduce each to distance-to-reference-centroid (one distribution per batch)
  4. drift calculation   — PSI between the two distance distributions (> ~0.10 => drifted)
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src import data_prep
from src.model import load_model, EmbeddingExtractor


def psi(reference, current, bins: int = 10) -> float:
    # TODO 4: Population Stability Index between two 1-D distributions (quantile bins).
    """
    Calculate Population Stability Index between two
    one-dimensional distributions using reference quantile bins.
    """

    # Convert inputs to 1-D numpy arrays.
    reference = np.asarray(
        reference,
        dtype=float,
    ).ravel()

    current = np.asarray(
        current,
        dtype=float,
    ).ravel()

    # Create quantile-based bin edges from the reference distribution.
    bin_edges = np.quantile(
        reference,
        np.linspace(
            0.0,
            1.0,
            bins + 1,
        ),
    )

    # Remove duplicate edges caused by repeated values.
    bin_edges = np.unique(bin_edges)

    # If there are fewer than 2 unique bin edges, PSI cannot be computed.
    if len(bin_edges) < 2:
        return 0.0

    # Include values outside the reference range.
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Compute histogram counts for both distributions using the same bin edges.
    reference_counts, _ = np.histogram(
        reference,
        bins=bin_edges,
    )

    current_counts, _ = np.histogram(
        current,
        bins=bin_edges,
    )

    # Calculate proportions for both distributions.
    reference_proportions = (
        reference_counts
        / reference_counts.sum()
    )

    current_proportions = (
        current_counts
        / current_counts.sum()
    )

    # Avoid divide-by-zero and log(0).
    epsilon = 1e-6

    # Clip proportions to avoid zero values.
    reference_proportions = np.clip(
        reference_proportions,
        epsilon,
        None,
    )

    current_proportions = np.clip(
        current_proportions,
        epsilon,
        None,
    )

    # Calculate the Population Stability Index (PSI).
    psi_value = np.sum(
        (
            current_proportions
            - reference_proportions
        )
        * np.log(
            current_proportions
            / reference_proportions
        )
    )

    return float(psi_value)


def corrupt(img: Image.Image) -> Image.Image:
    # TODO 4: simulate camera/lighting drift (brightness/blur/rotate/noise via config.DRIFT_SIM).
    raise NotImplementedError


def run() -> dict:
    # TODO 4: build reference (clean) + current (corrupted) batches → features, embeddings,
    #         mean confidence. Run Evidently DataDriftPreset + PSI; embedding PSI on
    #         distance-to-centroid; confidence drop. Write drift_summary.json + drift_report.html
    #         and set retrain_recommended from the configured thresholds.
    raise NotImplementedError


if __name__ == "__main__":
    run()
