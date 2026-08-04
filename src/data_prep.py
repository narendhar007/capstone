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
from datetime import datetime, timezone

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
    #Required as I am using symbolic links for the data directory.

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

    # If we reach this point, no valid dataset root was found. Therefore raise a FileNotFoundError with a descriptive message.
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
    """
    Validate the structure, integrity and leakage safety of the dataset.

    The function checks:

    1. Required train/test and class directories exist.
    2. Every class directory contains image files.
    3. Each image can be opened and verified by Pillow.
    4. Each image has the expected 300 x 300 dimensions.
    5. Exact duplicate images are detected using an MD5 content hash.
    6. Duplicate images crossing train/test boundaries are identified.
    7. Duplicate images with conflicting class labels are identified.
    8. Class counts and proportions are recorded for each split.

    MD5 is used only for exact duplicate detection. 

    Parameters
    ----------
    root:
        Dataset directory containing train/ and test/ folders.

    Returns
    -------
    dict
        JSON-serialisable data-quality report containing detailed issues,
        class distributions, check results and an overall passed flag.
    """
    root = Path(root).expanduser().resolve()

    if not root.exists(): #if root does not exist, raise a FileNotFoundError with a descriptive message.
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    expected_dimensions = (300, 300) # Expected image dimensions
    expected_splits = ("train", "test") # Expected dataset splits
    expected_classes = tuple(config.CLASS_TO_IDX.keys()) # Expected class names

    missing_directories: list[str] = [] # List of missing directories
    empty_directories: list[str] = [] # List of empty directories
    corrupt_files: list[dict] = [] # List of corrupt files with their paths and error details
    invalid_dimensions: list[dict] = [] # List of files with invalid dimensions, including their paths, actual and expected dimensions, format, and mode

    # Initialize a nested dictionary to count images per split and class.
    split_counts: dict[str, dict[str, int]] = {
        split: {class_name: 0 for class_name in expected_classes}
        for split in expected_splits
    }

    # Maps an image-content hash to all relative paths sharing that hash.
    hash_to_paths: dict[str, list[str]] = {}

    for split in expected_splits: #for each split (train/test), check for the existence of class directories and process images.
        for class_name in expected_classes: #for each class (ok_front/def_front), check for the existence of the class directory and process images.
            class_directory = root / split / class_name # Construct the path to the class directory.
            relative_directory = Path(split) / class_name # Construct the relative path for reporting.

            if not class_directory.is_dir(): #if the class directory does not exist, add it to the list of missing directories and continue to the next class.
                missing_directories.append(relative_directory.as_posix()) #add the relative path of the missing directory to the list of missing directories.
                continue

            image_paths = sorted( #sorted list of image file paths in the class directory, filtered by valid image extensions.
                path 
                for path in class_directory.rglob("*") # Recursively find all files in the class directory and its subdirectories.
                if path.is_file() and path.suffix.lower() in IMG_EXTS # Filter for files with valid image extensions (jpg, jpeg, png, bmp).
            )

            #count the number of images found in the class directory and store it in the split_counts dictionary.
            split_counts[split][class_name] = len(image_paths) 

            #if the class directory is empty (no image files found), add it to the list of empty directories.
            if not image_paths:
                empty_directories.append(relative_directory.as_posix())

            # Process each image file in the class directory.
            for image_path in image_paths:
                # Compute the relative path of the image file with respect to the dataset root for reporting purposes.
                relative_path = image_path.relative_to(root).as_posix()

                try:
                    # Read the image bytes once so the same contents are used
                    # for duplicate hashing and file verification.
                    image_bytes = image_path.read_bytes()

                    digest = hashlib.md5(image_bytes).hexdigest()
                    hash_to_paths.setdefault(digest, []).append(relative_path)

                    with Image.open(image_path) as image:
                        # Store the actual dimensions, format, and mode of the image for validation.
                        actual_dimensions = image.size
                        image_format = image.format
                        image_mode = image.mode

                        # verify() checks file integrity without decoding it
                        # into an image array for modelling.
                        image.verify()

                    # Check if the actual dimensions of the image match the expected dimensions (300 x 300). If not, add the image details to the list of invalid dimensions.
                    if actual_dimensions != expected_dimensions:
                        invalid_dimensions.append(
                            {
                                "path": relative_path,
                                "actual_dimensions": list(actual_dimensions),
                                "expected_dimensions": list(expected_dimensions),
                                "format": image_format,
                                "mode": image_mode,
                            }
                        )

                # Handle exceptions that may occur during image processing, such as unreadable or corrupt files. If an exception is raised, add the image details and error information to the list of corrupt files.
                except (
                    UnidentifiedImageError,
                    OSError,
                    ValueError,
                    SyntaxError,
                ) as exc:
                    # Record the corrupt file details, including its relative path, error type, and error message, in the list of corrupt files.
                    corrupt_files.append(
                        {
                            "path": relative_path,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )

    duplicate_groups: list[dict] = []
    cross_split_duplicates: list[dict] = []
    cross_class_duplicates: list[dict] = []

    for digest, relative_paths in sorted(hash_to_paths.items()):
        if len(relative_paths) < 2:
            continue

        sorted_paths = sorted(relative_paths)

        split_names = {
            Path(relative_path).parts[0]
            for relative_path in sorted_paths
        }

        class_names = {
            Path(relative_path).parts[1]
            for relative_path in sorted_paths
            if len(Path(relative_path).parts) >= 3
        }

        duplicate_record = {
            "md5": digest,
            "file_count": len(sorted_paths),
            "files": sorted_paths,
            "cross_split": len(split_names) > 1,
            "cross_class": len(class_names) > 1,
        }

        duplicate_groups.append(duplicate_record)

        if duplicate_record["cross_split"]:
            cross_split_duplicates.append(duplicate_record)

        if duplicate_record["cross_class"]:
            cross_class_duplicates.append(duplicate_record)

    split_totals = {
        split: sum(split_counts[split].values())
        for split in expected_splits
    }

    overall_class_counts = {
        class_name: sum(
            split_counts[split][class_name]
            for split in expected_splits
        )
        for class_name in expected_classes
    }

    total_images = sum(split_totals.values())

    class_distribution = {}

    for split in expected_splits:
        split_total = split_totals[split]

        class_distribution[split] = {
            class_name: {
                "count": split_counts[split][class_name],
                "percentage": (
                    round(
                        100.0
                        * split_counts[split][class_name]
                        / split_total,
                        2,
                    )
                    if split_total > 0
                    else 0.0
                ),
            }
            for class_name in expected_classes
        }

    duplicate_file_count = sum(
        group["file_count"]
        for group in duplicate_groups
    )

    # Number of duplicate copies beyond the first unique occurrence.
    excess_duplicate_count = sum(
        group["file_count"] - 1
        for group in duplicate_groups
    )

    structure_passed = (
        len(missing_directories) == 0
        and len(empty_directories) == 0
    )

    integrity_passed = len(corrupt_files) == 0

    dimensions_passed = len(invalid_dimensions) == 0

    class_distribution_passed = all(
        split_counts[split][class_name] > 0
        for split in expected_splits
        for class_name in expected_classes
    )

    duplicates_passed = len(duplicate_groups) == 0

    leakage_passed = len(cross_split_duplicates) == 0

    label_consistency_passed = len(cross_class_duplicates) == 0

    checks = {
        "required_directories": {
            "status": "PASS" if structure_passed else "FAIL",
            "issue_count": (
                len(missing_directories) + len(empty_directories)
            ),
        },
        "corrupt_files": {
            "status": "PASS" if integrity_passed else "FAIL",
            "issue_count": len(corrupt_files),
        },
        "image_dimensions": {
            "status": "PASS" if dimensions_passed else "FAIL",
            "issue_count": len(invalid_dimensions),
        },
        "class_distribution": {
            "status": "PASS" if class_distribution_passed else "FAIL",
            "issue_count": sum(
                split_counts[split][class_name] == 0
                for split in expected_splits
                for class_name in expected_classes
            ),
        },
        "exact_duplicates": {
            "status": "PASS" if duplicates_passed else "FAIL",
            "issue_count": len(duplicate_groups),
        },
        "cross_split_leakage": {
            "status": "PASS" if leakage_passed else "FAIL",
            "issue_count": len(cross_split_duplicates),
        },
        "class_label_consistency": {
            "status": "PASS" if label_consistency_passed else "FAIL",
            "issue_count": len(cross_class_duplicates),
        },
    }

    failure_reasons: list[str] = []

    if not structure_passed:
        failure_reasons.append(
            "Required split/class directories are missing or empty."
        )

    if not integrity_passed:
        failure_reasons.append(
            "One or more image files are corrupt or unreadable."
        )

    if not dimensions_passed:
        failure_reasons.append(
            "One or more images do not have the expected 300 x 300 dimensions."
        )

    if not class_distribution_passed:
        failure_reasons.append(
            "One or more required split/class combinations contain no images."
        )

    if not duplicates_passed:
        failure_reasons.append(
            "Exact duplicate images were found in the dataset."
        )

    if not leakage_passed:
        failure_reasons.append(
            "Exact duplicate images cross train/test boundaries and create leakage."
        )

    if not label_consistency_passed:
        failure_reasons.append(
            "Identical image contents were assigned to different class labels."
        )

    passed = all(
        result["status"] == "PASS"
        for result in checks.values()
    )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "expected_dimensions": list(expected_dimensions),
        "expected_splits": list(expected_splits),
        "expected_classes": list(expected_classes),
        "total_images": total_images,
        "split_totals": split_totals,
        "split_counts": split_counts,
        "overall_class_counts": overall_class_counts,
        "class_distribution": class_distribution,
        "missing_directories": missing_directories,
        "empty_directories": empty_directories,
        "corrupt_files": corrupt_files,
        "invalid_dimensions": invalid_dimensions,
        "duplicates": {
            "duplicate_group_count": len(duplicate_groups),
            "duplicate_file_count": duplicate_file_count,
            "excess_duplicate_count": excess_duplicate_count,
            "cross_split_group_count": len(cross_split_duplicates),
            "cross_class_group_count": len(cross_class_duplicates),
            "groups": duplicate_groups,
            "cross_split_groups": cross_split_duplicates,
            "cross_class_groups": cross_class_duplicates,
        },
        "checks": checks,
        "failure_reasons": failure_reasons,
        "passed": passed,
    }
    #raise NotImplementedError("Implement data-quality validation")


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
