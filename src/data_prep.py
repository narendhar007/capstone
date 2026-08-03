"""data_prep.py — Stage 1: discovery, quality validation, versioned splits, transforms. 

Implement: locate the casting folders; run data-quality checks (missing/corrupt/duplicate/
dimension/class-distribution/consistency); build reproducible stratified train/val/test
splits with a versioned snapshot + metadata.json; define preprocessing + augmentation
transforms; and per-image feature extraction used by drift monitoring.
"""
from __future__ import annotations

import hashlib, json, os, random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageStat, UnidentifiedImageError

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_data_root(base: Path | None = None) -> Path:
    # TODO 1: search under (base or config.DATA_DIR) for a dir whose train/ has both
    #         ok_front and def_front subfolders; return it.
    raise NotImplementedError("Locate the casting train/test root")


def list_images(split_dir: Path) -> list[tuple[Path, int]]:
    items = []
    for cls, idx in config.CLASS_TO_IDX.items():
        for p in sorted((split_dir / cls).glob("*")):
            if p.suffix.lower() in IMG_EXTS:
                items.append((p, idx))
    return items


def validate_quality(root: Path) -> dict:
    # TODO 1 (data quality): for train+test, count images + class distribution; open each
    #         image (catch corrupt); record non-300x300 dims; md5-hash to find duplicates.
    #         Return a report dict with issues + a 'passed' flag.
    raise NotImplementedError("Implement data-quality validation")


def build_splits(root: Path, version: str = "v1") -> dict:
    # TODO 1 (versioning): stratified val carve-out from train/; test from test/. Save
    #         {train,val,test}.json file lists + metadata.json (version, date, class defs,
    #         split sizes/distribution, seed) under config.SPLIT_DIR/<version>/.
    raise NotImplementedError("Build versioned stratified splits + metadata")


def load_split(version: str, name: str, root: Path) -> list[tuple[Path, int]]:
    rel = json.loads((config.SPLIT_DIR / version / f"{name}.json").read_text())
    return [(root / r, y) for r, y in rel]


def get_transforms(train: bool):
    from torchvision import transforms
    # TODO 2 (preprocessing + augmentation): Grayscale(3) → Resize(224) → [train: flip,
    #         affine rotation/translate, ColorJitter] → ToTensor → Normalize(ImageNet).
    raise NotImplementedError("Define the train/eval transforms")


def image_features(img: Image.Image) -> dict:
    # TODO 4 (statistical drift): return brightness, contrast, edge_density, sharpness,
    #         mean_intensity for a PIL image (keys == config.DRIFT_FEATURES).
    raise NotImplementedError("Extract per-image drift features")
