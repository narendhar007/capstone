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
    """
    Locate the root directory of the casting image dataset.

    A valid dataset root must contain:

        train/
            ok_front/
            def_front/
        test/
            ok_front/
            def_front/

    Parameters
    ----------
    base:
        Directory beneath which the dataset should be searched.
        When omitted, config.DATA_DIR is used.

    Returns
    -------
    Path
        Resolved path to the casting dataset root.

    Raises
    ------
    FileNotFoundError
        If the configured search directory or expected dataset structure
        cannot be found.
    """

    search_base = Path(base) if base is not None else config.DATA_DIR #if base is None, use the default data directory from config
    search_base = search_base.expanduser().resolve() #expanduser() replaces ~ with the user home directory, resolve() returns the absolute path. 
    #Required as I am using symbolic links for my data directory.

    if not search_base.exists():
        raise FileNotFoundError(
            f"Data search directory does not exist: {search_base}"
        )

    # Define the required directory structure for a valid dataset root.
    required_directories = (
        ("train", "ok_front"),
        ("train", "def_front"),
        ("test", "ok_front"),
        ("test", "def_front"),
    )

    # Helper function to check if a candidate directory is a valid dataset root.
    def is_dataset_root(candidate: Path) -> bool:
        return all(
            (candidate / split / class_name).is_dir()
            for split, class_name in required_directories
        )

    # First check whether the configured directory is itself the dataset root.
    if is_dataset_root(search_base):
        return search_base

    # Search recursively. followlinks=True allows the Windows directory
    # junction under data/casting_data to be traversed.
    for current_directory, _, _ in os.walk(search_base, followlinks=True):
        candidate = Path(current_directory)

        if is_dataset_root(candidate):
            return candidate.resolve()

    raise FileNotFoundError(
        "Casting dataset could not be found beneath "
        f"{search_base}. Expected train/test directories containing "
        "ok_front and def_front class folders."
    )
    #raise NotImplementedError("Locate the casting train/test root")


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
