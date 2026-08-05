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
    raise NotImplementedError("Implement data-quality validation")


def build_splits(root: Path, version: str = "v1") -> dict:
    # TODO 1 (versioning): stratified val carve-out from train/; test from test/. Save
    #         {train,val,test}.json file lists + metadata.json (version, date, class defs,
    #         split sizes/distribution, seed) under config.SPLIT_DIR/<version>/
    """
    Create reproducible, duplicate-safe train, validation and test snapshots.

    Strategy
    --------
    - Use the supplied train folder as the source of train/validation data.
    - Use the supplied test folder as the source of final test data.
    - Deduplicate exact image contents using MD5.
    - When an exact image occurs in both train and test, retain the training
      copy and exclude the matching test copy.
    - Create a class-stratified validation split from the cleaned training set.
    - Save relative file paths so the manifests remain portable.
    - Treat an existing dataset version as immutable.

    Parameters
    ----------
    root:
        Dataset root containing train/ and test/ directories.

    version:
        Dataset snapshot identifier, for example ``v1``.

    Returns
    -------
    dict
        Metadata describing the versioned split.
    """
    root = Path(root).expanduser().resolve()

    #check if root exists, if not raise a FileNotFoundError with a descriptive message.
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    #check if version is valid, if not raise a ValueError with a descriptive message.
    if (
        not version
        or not version.replace("-", "").replace("_", "").isalnum()
    ):
        raise ValueError(
            "Version must contain only letters, numbers, hyphens "
            "or underscores."
        )

    source_train = list_images(root / "train")
    source_test = list_images(root / "test")

    #check if there are any training images, if not raise a ValueError with a descriptive message.
    if not source_train:
        raise ValueError("No training images were found.")

    #check if there are any test images, if not raise a ValueError with a descriptive message.
    if not source_test:
        raise ValueError("No test images were found.")

    #set the split policy to "prefer_train_remove_test_hash_overlap_v1" to indicate that the training set is preferred and any overlapping hashes in the test set will be removed.
    split_policy = "prefer_train_remove_test_hash_overlap_v1"

    # Helper functions for content hashing and relative path resolution.
    def content_hash(path: Path) -> str:
        return hashlib.md5(path.read_bytes()).hexdigest()

    # Helper function to get the relative path of a file with respect to the dataset root and return it as a POSIX-style string.
    def relative_path(path: Path) -> str:
        return path.resolve().relative_to(root).as_posix()

    # Calculate the source inventory before splitting.
    inventory = []

    # Iterate over the source training and test items, and for each item, create a dictionary containing the source split, relative path, label, and MD5 hash. Append this dictionary to the inventory list.
    for source_split, items in (
        ("train", source_train),
        ("test", source_test),
    ):
        for path, label in items:
            inventory.append(
                {
                    "source_split": source_split,
                    "path": relative_path(path),
                    "label": label,
                    "md5": content_hash(path),
                }
            )

    # Sort the inventory by source split, path, and label to ensure a consistent order for serialization and hashing.
    inventory.sort(
        key=lambda item: (
            item["source_split"],
            item["path"],
            item["label"],
        )
    )

    # Serialize the inventory to a JSON string with sorted keys and compact separators.
    inventory_serialised = json.dumps(
        inventory,
        sort_keys=True,
        separators=(",", ":"),
    )

    # Compute the SHA-256 hash of the serialized inventory to create a unique identifier for the source inventory.
    source_inventory_sha256 = hashlib.sha256(
        inventory_serialised.encode("utf-8")
    ).hexdigest()

    # Create the version directory and metadata path based on the specified version.
    version_dir = config.SPLIT_DIR / version
    metadata_path = version_dir / "metadata.json"

    # Dataset versions are immutable. Reuse the version only when the source
    # inventory and splitting policy are unchanged.
    if metadata_path.exists():
        existing_metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        same_inventory = (
            existing_metadata.get("source_inventory_sha256")
            == source_inventory_sha256
        )

        same_policy = (
            existing_metadata.get("split_policy")
            == split_policy
        )

        manifests_exist = all(
            (version_dir / f"{name}.json").exists()
            for name in ("train", "val", "test")
        )

        if same_inventory and same_policy and manifests_exist:
            return existing_metadata

        raise FileExistsError(
            f"Dataset version '{version}' already exists but does not "
            "match the current source inventory or split policy. "
            "Use a new version identifier."
        )

    # Group items by content hash to identify duplicates and ensure consistent label assignment.
    def group_by_hash(
        items: list[tuple[Path, int]],
    ) -> dict[str, list[tuple[Path, int]]]:
        groups: dict[str, list[tuple[Path, int]]] = {}

        for path, label in items:
            digest = content_hash(path)
            groups.setdefault(digest, []).append((path, label))

        return groups

    train_hash_groups = group_by_hash(source_train)
    test_hash_groups = group_by_hash(source_test)

    # Confirm that identical image content never has conflicting labels.
    all_hash_groups: dict[str, list[tuple[Path, int]]] = {}

    for groups in (train_hash_groups, test_hash_groups):
        for digest, items in groups.items():
            all_hash_groups.setdefault(digest, []).extend(items)

    conflicting_hashes = []

    for digest, items in all_hash_groups.items():
        labels = {label for _, label in items}

        if len(labels) > 1:
            conflicting_hashes.append(
                {
                    "md5": digest,
                    "files": [
                        relative_path(path)
                        for path, _ in items
                    ],
                    "labels": sorted(labels),
                }
            )

    if conflicting_hashes:
        raise ValueError(
            "Identical image contents have conflicting class labels. "
            "Resolve these records before splitting."
        )

    def representative(
        items: list[tuple[Path, int]],
    ) -> tuple[Path, int]:
        return sorted(
            items,
            key=lambda item: relative_path(item[0]),
        )[0]

    # Keep one representative for every unique training hash.
    unique_train = [
        representative(items)
        for _, items in sorted(train_hash_groups.items())
    ]

    train_hashes = set(train_hash_groups)
    test_hashes = set(test_hash_groups)
    overlapping_hashes = train_hashes & test_hashes

    # Keep only test hashes that were not already represented in train.
    unique_test = [
        representative(items)
        for digest, items in sorted(test_hash_groups.items())
        if digest not in overlapping_hashes
    ]

    excluded_test_files = sorted(
        relative_path(path)
        for digest in overlapping_hashes
        for path, _ in test_hash_groups[digest]
    )

    within_train_duplicates_removed = (
        len(source_train) - len(train_hash_groups)
    )

    within_test_duplicates_removed = (
        len(source_test) - len(test_hash_groups)
    )

    # Stratified train/validation split.
    rng = random.Random(config.RANDOM_SEED)

    train_items: list[tuple[Path, int]] = []
    val_items: list[tuple[Path, int]] = []

    for class_name, class_index in config.CLASS_TO_IDX.items():
        class_items = sorted(
            [
                item
                for item in unique_train
                if item[1] == class_index
            ],
            key=lambda item: relative_path(item[0]),
        )

        if len(class_items) < 2:
            raise ValueError(
                f"Class '{class_name}' requires at least two unique "
                "training images to create train and validation sets."
            )

        rng.shuffle(class_items)

        validation_count = int(
            round(len(class_items) * config.VAL_SPLIT)
        )

        validation_count = max(1, validation_count)
        validation_count = min(
            validation_count,
            len(class_items) - 1,
        )

        val_items.extend(class_items[:validation_count])
        train_items.extend(class_items[validation_count:])

    # Stable ordering makes the saved manifests easy to compare.
    train_items.sort(key=lambda item: relative_path(item[0]))
    val_items.sort(key=lambda item: relative_path(item[0]))
    unique_test.sort(key=lambda item: relative_path(item[0]))

    final_splits = {
        "train": train_items,
        "val": val_items,
        "test": unique_test,
    }

    # Final leakage assertion using the image contents, not file names.
    final_hash_sets = {
        name: {
            content_hash(path)
            for path, _ in items
        }
        for name, items in final_splits.items()
    }

    overlap_counts = {
        "train_val": len(
            final_hash_sets["train"]
            & final_hash_sets["val"]
        ),
        "train_test": len(
            final_hash_sets["train"]
            & final_hash_sets["test"]
        ),
        "val_test": len(
            final_hash_sets["val"]
            & final_hash_sets["test"]
        ),
    }

    if any(overlap_counts.values()):
        raise RuntimeError(
            f"Hash leakage remains after splitting: {overlap_counts}"
        )

    def split_summary(
        items: list[tuple[Path, int]],
    ) -> dict:
        label_counts = Counter(
            label
            for _, label in items
        )

        class_counts = {
            class_name: label_counts[class_index]
            for class_name, class_index
            in config.CLASS_TO_IDX.items()
        }

        size = len(items)

        class_percentages = {
            class_name: (
                round(100.0 * count / size, 2)
                if size
                else 0.0
            )
            for class_name, count in class_counts.items()
        }

        return {
            "size": size,
            "class_counts": class_counts,
            "class_percentages": class_percentages,
        }

    version_dir.mkdir(parents=True, exist_ok=False)

    manifest_hashes = {}

    for split_name, items in final_splits.items():
        manifest = [
            [relative_path(path), label]
            for path, label in items
        ]

        manifest_text = json.dumps(
            manifest,
            indent=2,
        )

        manifest_path = version_dir / f"{split_name}.json"
        manifest_path.write_text(
            manifest_text,
            encoding="utf-8",
        )

        manifest_hashes[split_name] = hashlib.sha256(
            manifest_text.encode("utf-8")
        ).hexdigest()

    metadata = {
        "version_id": version,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "random_seed": config.RANDOM_SEED,
        "validation_fraction": config.VAL_SPLIT,
        "class_to_idx": config.CLASS_TO_IDX,
        "positive_class": config.POSITIVE_CLASS,
        "positive_class_index": config.POSITIVE_IDX,
        "split_policy": split_policy,
        "source_inventory_sha256": source_inventory_sha256,
        "source_summary": {
            "total_images": (
                len(source_train) + len(source_test)
            ),
            "supplied_train_size": len(source_train),
            "supplied_test_size": len(source_test),
        },
        "duplicate_handling": {
            "hash_algorithm": "md5",
            "policy": (
                "Retain one representative per hash in the supplied "
                "training data. Remove test images whose hash occurs "
                "in training. Retain one representative per remaining "
                "test hash."
            ),
            "cross_split_duplicate_group_count": len(
                overlapping_hashes
            ),
            "cross_split_test_files_excluded": len(
                excluded_test_files
            ),
            "within_train_duplicates_removed":
                within_train_duplicates_removed,
            "within_test_duplicates_removed":
                within_test_duplicates_removed,
            "excluded_test_files": excluded_test_files,
        },
        "split_info": {
            split_name: split_summary(items)
            for split_name, items in final_splits.items()
        },
        "leakage_check": {
            **overlap_counts,
            "passed": not any(overlap_counts.values()),
        },
        "manifest_files": {
            split_name: f"{split_name}.json"
            for split_name in final_splits
        },
        "manifest_sha256": manifest_hashes,
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return metadata    
    raise NotImplementedError("Build versioned stratified splits + metadata")


def load_split(version: str, name: str, root: Path) -> list[tuple[Path, int]]:
    rel = json.loads((config.SPLIT_DIR / version / f"{name}.json").read_text())
    return [(root / r, y) for r, y in rel]


def get_transforms(train: bool):
    from torchvision import transforms
    # TODO 2 (preprocessing + augmentation): Grayscale(3) → Resize(224) → [train: flip,
    #         affine rotation/translate, ColorJitter] → ToTensor → Normalize(ImageNet).
    """
    Build the image transformation pipeline.

    Training images receive controlled random augmentation.
    Validation, test and inference images receive only deterministic
    preprocessing.
    """
    steps = [
        # ResNet18 expects three input channels.
        transforms.Grayscale(num_output_channels=3),

        # ResNet18 ImageNet input size.
        transforms.Resize(
            (config.IMG_SIZE, config.IMG_SIZE)
        ),
    ]

    # Add data augmentation steps for training images only.
    if train:
        steps.extend(
            [
                # Random horizontal flip with a probability defined in the config file. 
                # This augmentation helps the model generalize better by simulating different orientations of the objects in the images.
                transforms.RandomHorizontalFlip(
                    p=config.AUG["hflip_p"]
                ),
                # Random affine transformation with rotation and translation parameters defined in the config file.
                # This augmentation introduces variability in the training data, helping the model become more robust to changes
                # in object orientation and position.
                transforms.RandomAffine(
                    degrees=config.AUG["rotation_degrees"],
                    translate=(
                        config.AUG["translate"],
                        config.AUG["translate"],
                    ),
                ),
                # Color jittering to randomly change the brightness and contrast of the images.
                # This augmentation simulates different lighting conditions, which can help the model learn to recognize objects 
                # under varying illumination.
                transforms.ColorJitter(
                    brightness=config.AUG["brightness"],
                    contrast=config.AUG["contrast"],
                ),
            ]
        )

    # Convert the image to a tensor and normalize it using the mean and standard deviation values for the ImageNet dataset. 
    steps.extend(
        [
            transforms.ToTensor(),
            # Normalize the image tensor using the mean and standard deviation values for the ImageNet dataset.
            # This normalization step ensures that the input data has a consistent scale, which can improve the 
            # convergence and performance of the model during training.
            transforms.Normalize(
                mean=config.IMAGENET_MEAN,
                std=config.IMAGENET_STD,
            ),
        ]
    )

    return transforms.Compose(steps)
    raise NotImplementedError("Define the train/eval transforms")


def image_features(img: Image.Image) -> dict:
    # TODO 4 (statistical drift): return brightness, contrast, edge_density, sharpness,
    #         mean_intensity for a PIL image (keys == config.DRIFT_FEATURES).
    #Function to extract interpretable numerical characteristics from one image. 
    # The same function is used for Stage 1 exploratory data analysis and Stage 4 statistical drift monitoring.
    """
    Extract simple image statistics for EDA and drift monitoring.

    Returns:
        brightness      - normalised RMS brightness
        contrast        - standard deviation of pixel intensity
        edge_density    - proportion of strong edge pixels
        sharpness       - variance of the edge response
        mean_intensity  - average grayscale intensity
    """
    #Convert the input image to grayscale using the "L" mode, which represents 8-bit pixels, black and white.
    #This simplifies the analysis by reducing the image to a single channel of intensity values.
    gray = img.convert("L")

    #Convert the grayscale image to a NumPy array of type float32.
    pixels = np.asarray(gray, dtype=np.float32)

    # Edge response using Pillow's built-in filter.
    edge_image = gray.filter(ImageFilter.FIND_EDGES)
    edges = np.asarray(edge_image, dtype=np.float32)

    #Calculate the brightness, contrast, edge density, sharpness, and mean intensity of the image using various statistical measures.
    features = {
        "brightness": float(ImageStat.Stat(gray).rms[0] / 255.0),
        "contrast": float(pixels.std()),
        "edge_density": float((edges > 20).mean()),
        "sharpness": float(edges.var()),
        "mean_intensity": float(pixels.mean()),
    }

    #Return only the features specified in the configuration's DRIFT_FEATURES list, 
    #ensuring that the output is consistent with the expected feature set for drift monitoring.
    return {
        name: features[name]
        for name in config.DRIFT_FEATURES
    }
    raise NotImplementedError("Extract per-image drift features")
