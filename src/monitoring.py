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

from datetime import datetime, timezone

import torch
import torch.nn.functional as F


def psi(reference, current, bins: int = 10) -> float:
    # TODO 4: Population Stability Index between two 1-D distributions (quantile bins).
    """
    Calculate Population Stability Index between two
    one-dimensional distributions using reference quantile bins.

    PSI is a statistical metric used in model monitoring to measure the shift in distributions 
    between a reference dataset and a current dataset. 
    It is commonly used to detect data drift in production environments.
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
    # PSI is calculated as the sum of the product of the difference in proportions 
    # and the logarithm of the ratio of current to reference proportions across all bins.
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
    """
    Simulate production camera and lighting drift.

    Applies the corruption settings defined in config.DRIFT_SIM:
    brightness reduction, blur, rotation and sensor-like noise.
    """
    # Convert the image to grayscale (L mode) for simplicity.
    image = img.convert("L")

    # Simulate reduced lighting.
    image = ImageEnhance.Brightness(
        image
    ).enhance(
        config.DRIFT_SIM["brightness"]
    )

    # Simulate loss of camera focus.
    image = image.filter(
        ImageFilter.GaussianBlur(
            radius=config.DRIFT_SIM[
                "blur_radius"
            ]
        )
    )

    # Simulate camera alignment change.
    image = image.rotate(
        config.DRIFT_SIM["rotate"],
        resample=Image.Resampling.BILINEAR, # Use bilinear resampling for better quality during rotation
        expand=False,
        fillcolor=0,
    )

    # Define pixels as a numpy array for noise addition.
    pixels = np.asarray(
        image,
        dtype=np.float32,
    )

    # Simulate image-sensor noise.
    noise = np.random.normal(
        loc=0.0,
        scale=config.DRIFT_SIM["noise_std"],
        size=pixels.shape,
    )

    # Add noise to the image and clip pixel values to valid range [0, 255].
    pixels = np.clip(
        pixels + noise,
        0,
        255,
    ).astype(np.uint8)

    # Convert the noisy pixel array back to a PIL Image in grayscale mode.
    return Image.fromarray(
        pixels
    )


def run() -> dict:
    # TODO 4: build reference (clean) + current (corrupted) batches → features, embeddings,
    #         mean confidence. Run Evidently DataDriftPreset + PSI; embedding PSI on
    #         distance-to-centroid; confidence drop. Write drift_summary.json + drift_report.html
    #         and set retrain_recommended from the configured thresholds.
    """
    Compare a clean reference batch with a simulated drifted batch.

    Calculates:
      1. interpretable image-feature PSI
      2. Evidently data-drift report
      3. ResNet embedding drift
      4. mean prediction-confidence drop
      5. multi-signal retraining recommendation
    """

    # Reproducible corruption noise.
    np.random.seed(
        config.RANDOM_SEED
    )

    # ---------------------------------------------------------
    # Load the production model and matching dataset version.
    # ---------------------------------------------------------
    model = load_model(
        path=config.MODEL_PATH,
        freeze=True,
    )

    # Set the model to evaluation mode to disable dropout and batch normalization layers.
    model.eval()

    # Define an empty metadata dictionary to hold model metadata
    metadata = {}

    # If the model metadata file exists, read it and parse the JSON content into the metadata dictionary.
    if config.MODEL_META_PATH.exists():
        metadata = json.loads(
            config.MODEL_META_PATH.read_text(
                encoding="utf-8"
            )
        )

    # Get the dataset version from the metadata, defaulting to "v1" if not found.
    dataset_version = metadata.get(
        "dataset_version",
        "v1",
    )

    # Set the root directory for the dataset using the data_prep module.
    root = data_prep.find_data_root()

    # Load the reference items (image paths and labels) for the specified dataset version and split ("var") from the root directory.
    reference_items = data_prep.load_split(
        dataset_version,
        "val",
        root,
    )

    # ---------------------------------------------------------
    # Build clean reference and simulated current batches.
    # ---------------------------------------------------------
    reference_images = []
    current_images = []

    for image_path, _ in reference_items:
        # Load the image from the specified path, designate it as clean image and convert it to grayscale (L mode).
        with Image.open(image_path) as opened_image:
            clean_image = (
                opened_image
                .convert("L")
                .copy()
            )

        # Append the clean image to the reference_images list and a corrupted version of the clean image to the current_images list.
        reference_images.append(
            clean_image
        )

        #simulate drifted production images by applying corruption to the clean image and appending it to the current_images list.
        current_images.append(
            corrupt(clean_image)
        )


    #Print the dataset version, number of reference images, and number of current images for monitoring purposes.
    print(
        "Dataset version:",
        dataset_version,
    )

    print(
        "Reference images:",
        len(reference_images),
    )

    print(
        "Current images:",
        len(current_images),
    )

    # ---------------------------------------------------------
    # Stage 4.2 — interpretable image features.
    # ---------------------------------------------------------

    # Load the reference features baseline from the CSV file in the artifact directory (config.ARTIFACT_DIR).    
    reference_features_path = (
        config.ARTIFACT_DIR
        / "reference_features.csv"
    )

    # Check if the reference features baseline file exists. If not, raise a FileNotFoundError with a 
    # message indicating that the baseline file was not found and suggesting to run src/train.py/save_reference_baseline() first.
    if not reference_features_path.exists():
        raise FileNotFoundError(
            "Reference features baseline file was not found."
            "Run save_reference_baseline() first."
        )

    # Load the fixed clean reference feature baseline
    # created for the production model.
    saved_reference_features = pd.read_csv(
        reference_features_path
    )

    # Use only the numerical features monitored for drift.
    reference_features = (
        saved_reference_features[
            config.DRIFT_FEATURES
        ]
        .copy()
    )

    # Calculate features only for the current simulated-production batch.
    current_features = pd.DataFrame(
        [
            data_prep.image_features(image)
            for image in current_images
        ]
    )

    # Ensure the stored baseline matches the reference split.
    if len(reference_features) != len(
        reference_items
    ):
        raise ValueError(
            "Reference feature baseline does not match "
            "the configured reference split."
    )

    # Calculate the Population Stability Index (PSI) for each feature in the reference and current feature sets 
    # using the psi function defined earlier in src/monitoring.py.
    feature_psi = {

        # For every feature in the config.DRIFT_FEATURES list, calculate the PSI between the reference and 
        # current feature values using the psi function.
        feature: psi(
            reference_features[feature].values,
            current_features[feature].values,
        )
        for feature in config.DRIFT_FEATURES
    }

    # Identify features that have drifted by comparing their PSI values against the configured threshold (config.PSI_THRESHOLD).
    drifted_features = [
        feature
        for feature, value
        in feature_psi.items()
        if value > config.PSI_THRESHOLD
    ]

    # Calculate the share of drifted features relative to the total number of features being monitored.
    drift_share = (
        len(drifted_features)
        / len(config.DRIFT_FEATURES)
    )

    # Determine if the dataset has drifted based on whether the share of drifted features 
    # exceeds the configured threshold (config.DRIFT_SHARE_THRESHOLD).
    dataset_drift = (
        drift_share
        >= config.DRIFT_SHARE_THRESHOLD
    )

    # ---------------------------------------------------------
    # Evidently DataDriftPreset.
    # ---------------------------------------------------------
    from evidently import Report
    from evidently.presets import DataDriftPreset

    # Create an Evidently report with the DataDriftPreset to analyze the drift between the reference and current feature sets.
    report = Report(
        [
            DataDriftPreset()
        ]
    )

    # Run the report on the reference and current feature sets to generate a drift report.
    report_result = report.run(
        reference_data=reference_features,
        current_data=current_features,
    )

    # Save the generated drift report as an HTML file in the configured artifact directory 
    # (config.ARTIFACT_DIR) with the filename "drift_report.html".
    drift_report_path = (
        config.ARTIFACT_DIR
        / "drift_report.html"
    )

    # Save the generated drift report as an HTML file in the configured artifact directory 
    # (config.ARTIFACT_DIR) with the filename "drift_report.html".
    report_result.save_html(
        str(drift_report_path)
    )

    # ---------------------------------------------------------
    # Stage 4.3 — embedding generation.
    # ---------------------------------------------------------
    # Define the image transformation pipeline for the model, specifying that it is not for training (train=False).
    transform = data_prep.get_transforms(
        train=False
    )

    # Create an embedding extractor using the loaded model. The EmbeddingExtractor class is defined in src/model.py 
    # and is used to extract embeddings from the penultimate layer of the model.
    extractor = EmbeddingExtractor(
        model
    ).to(
        config.DEVICE
    )

    extractor.eval()  # Set the embedding extractor to evaluation mode to disable dropout and batch normalization layers.

    # Define a function to compute model outputs, including confidence scores and embeddings, for a given list of images.
    def model_outputs(images):
        confidence_batches = []
        embedding_batches = []

        # Process the images in batches to avoid memory issues and improve efficiency.
        for start in range(
            0,
            len(images),
            config.BATCH_SIZE,
        ):
            # Select a batch of images from the input list based on the current start index 
            # and the configured batch size (config.BATCH_SIZE).
            batch_images = images[
                start:
                start + config.BATCH_SIZE
            ]

            # Convert the batch of images to a tensor using the defined transformation pipeline 
            # and move it to the configured device (config.DEVICE).
            tensor_batch = torch.stack(
                [
                    transform(image)
                    for image in batch_images
                ]
            ).to(
                config.DEVICE
            )

            # Perform a forward pass through the model to obtain logits, probabilities, confidence scores, 
            # and embeddings for the current batch of images.
            with torch.no_grad():

                # Extract embeddings from the penultimate layer of the model for the current batch of images using the embedding extractor.
                embeddings = extractor(
                    tensor_batch
                )

                # Compute the logits (raw model outputs) for the current batch of embeddings using the model's 
                # fully connected layer (model.fc).
                logits = model.fc(
                    embeddings
                )

                # Compute the probabilities for each class by applying the softmax function to the logits along the class dimension (dim=1).
                probabilities = F.softmax(
                    logits,
                    dim=1,
                )

                confidence = (
                    probabilities
                    .max(dim=1)
                    .values
                )

            # Append the computed confidence scores and embeddings for the current batch to their respective lists,
            # converting them to NumPy arrays and moving them to the CPU for further analysis.
            confidence_batches.append(
                confidence.cpu().numpy()
            )

            # Append the computed embeddings for the current batch to the embedding_batches list, 
            # converting them to NumPy arrays and moving them to the CPU for further analysis.
            embedding_batches.append(
                embeddings.cpu().numpy()
            )

        # Concatenate the confidence and embedding batches along the first axis (batch dimension) 
        # and return them as NumPy arrays for further analysis.
        return (
            np.concatenate(
                confidence_batches
            ),
            np.concatenate(
                embedding_batches
            ),
        )

    # Compute model outputs (confidence scores and embeddings) for current image batches 
    # using the model_outputs function defined above.
    (
        reference_confidence,
        generated_reference_embeddings,
    ) = model_outputs(
        reference_images
    )

    (
        current_confidence,
        current_embeddings,
    ) = model_outputs(
        current_images
    )

    # ---------------------------------------------------------
    # Load reference embeddings saved during training.
    # ---------------------------------------------------------

    if not config.REFERENCE_EMBED.exists():
        raise FileNotFoundError(
            "Reference embedding baseline was not found. "
            "Run save_reference_baseline() first."
        )

    with np.load(
        config.REFERENCE_EMBED
    ) as reference_embedding_data:
        reference_embeddings = (
            reference_embedding_data[
                "embeddings"
            ]
        )

        reference_centroid = (
            reference_embedding_data[
                "centroid"
            ]
        )

    if (
        reference_embeddings.ndim != 2
        or reference_embeddings.shape[1]
        != config.EMBEDDING_DIM
    ):
        raise ValueError(
            "Stored reference embeddings have "
            "an unexpected shape."
        )
        

    if reference_centroid.shape != (
        config.EMBEDDING_DIM,
    ):
        raise ValueError(
            "Stored reference centroid has "
            "an unexpected shape."
        )

    if (
        reference_embeddings.shape[0]
        != len(reference_items)
    ):
        raise ValueError(
            "Stored reference embeddings do not "
            "match the configured reference split."
        )

    if not np.allclose(
        generated_reference_embeddings,
        reference_embeddings,
        rtol=1e-4,
        atol=1e-5,
    ):
        raise ValueError(
            "Stored reference embeddings do not "
            "match the currently loaded production model."
        )

    # Distance from each image embedding to the clean centroid.
    reference_distances = np.linalg.norm(
        reference_embeddings
        - reference_centroid,
        axis=1,
    )

    # Compute the distances from each current image embedding to the reference centroid for drift analysis.
    current_distances = np.linalg.norm(
        current_embeddings
        - reference_centroid,
        axis=1,
    )

    # Calculate the Population Stability Index (PSI) between the reference and current distance distributions
    # to assess embedding drift. A higher PSI value indicates a greater shift in the embedding space
    embedding_psi = psi(
        reference_distances,
        current_distances,
    )

    # Determine if embedding drift has occurred by comparing the calculated embedding PSI 
    # against the configured threshold (config.EMBEDDING_DRIFT_THRESHOLD).
    embedding_drift = (
        embedding_psi
        > config.EMBEDDING_DRIFT_THRESHOLD
    )

    # ---------------------------------------------------------
    # Stage 4.1 — confidence monitoring.
    # ---------------------------------------------------------

    # Calculate the mean confidence for both the reference and current batches to assess any drop in model confidence.
    reference_mean_confidence = float(
        reference_confidence.mean()
    )

    # Calculate the mean confidence for the current batch to assess any drop in model confidence.
    current_mean_confidence = float(
        current_confidence.mean()
    )

    # Calculate the drop in mean confidence between the reference and current batches.
    confidence_drop = (
        reference_mean_confidence
        - current_mean_confidence
    )

    # Determine if a confidence alert should be triggered based on whether the drop in mean confidence
    confidence_alert = (
        confidence_drop
        > config.CONFIDENCE_DROP_THRESHOLD
    )

    # ---------------------------------------------------------
    # Multi-signal retraining trigger.
    # ---------------------------------------------------------

    # Determine the reasons for triggering retraining based on the results of dataset drift, embedding drift, and confidence drop.
    trigger_reasons = []

    # If dataset drift is detected, add "feature_drift" to the list of trigger reasons.
    if dataset_drift:
        trigger_reasons.append(
            "feature_drift"
        )

    # If embedding drift is detected, add "embedding_drift" to the list of trigger reasons.
    if embedding_drift:
        trigger_reasons.append(
            "embedding_drift"
        )

    # If a confidence alert is triggered, add "confidence_drop" to the list of trigger reasons.
    if confidence_alert:
        trigger_reasons.append(
            "confidence_drop"
        )

    # Determine if retraining is recommended based on whether any of the drift signals have been triggered.
    retrain_recommended = bool(
        trigger_reasons
    )

    # Generate a summary dictionary containing all relevant information about the drift analysis, including timestamps, 
    # dataset version, sample counts, drift metrics, and retraining recommendations.
    summary = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset_version": dataset_version,
        "reference_samples": len(
            reference_images
        ),
        "current_samples": len(
            current_images
        ),
        "feature_drift": {
            "psi": {
                feature: float(value)
                for feature, value
                in feature_psi.items()
            },
            "threshold": (
                config.PSI_THRESHOLD
            ),
            "drifted_features": (
                drifted_features
            ),
            "drifted_feature_count": len(
                drifted_features
            ),
            "total_features": len(
                config.DRIFT_FEATURES
            ),
            "drift_share": float(
                drift_share
            ),
            "drift_share_threshold": (
                config.DRIFT_SHARE_THRESHOLD
            ),
            "dataset_drift": bool(
                dataset_drift
            ),
        },
        "embedding_drift": {
            "psi": float(
                embedding_psi
            ),
            "threshold": (
                config.EMBEDDING_DRIFT_THRESHOLD
            ),
            "drifted": bool(
                embedding_drift
            ),
        },
        "confidence": {
            "reference_mean": (
                reference_mean_confidence
            ),
            "current_mean": (
                current_mean_confidence
            ),
            "drop": float(
                confidence_drop
            ),
            "threshold": (
                config.CONFIDENCE_DROP_THRESHOLD
            ),
            "alert": bool(
                confidence_alert
            ),
        },
        "retrain_recommended": (
            retrain_recommended
        ),
        "trigger_reasons": (
            trigger_reasons
        ),
        "drift_report": str(
            drift_report_path
        ),
        "reference_embeddings": str(
            config.REFERENCE_EMBED
        ),
        "reference_baseline": {
            "split": "val",
            "features_path": str(
                reference_features_path
            ),
            "embeddings_path": str(
                config.REFERENCE_EMBED
            ),

            "reference_embedding_shape": list(
                reference_embeddings.shape
            ),

            "current_embedding_shape": list(
                current_embeddings.shape
            ),

            "embedding_dimension":
                config.EMBEDDING_DIM,
        },
    }

    # Save the drift summary as a JSON file in the configured artifact directory (config.ARTIFACT_DIR) with the filename 
    # "drift_summary.json".

    # Set the path for the drift summary JSON file in the artifact directory.
    summary_path = (
        config.ARTIFACT_DIR
        / "drift_summary.json"
    )

    # Save the summary dictionary as a JSON file with indentation for readability and UTF-8 encoding.
    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Print the results of the drift analysis, including feature PSI values, drifted feature share, 
    # embedding PSI, mean confidence values, confidence drop, retraining recommendation, trigger reasons, 
    # and paths to the drift report and summary files.
    print(
        "\nFeature PSI:"
    )

    # Print the PSI values for each feature in the feature_psi dictionary, formatted to four decimal places.
    for feature, value in feature_psi.items():
        print(
            f"  {feature}: {value:.4f}"
        )

    # Print the share of drifted features as a percentage.
    print(
        "\nDrifted feature share:",
        f"{drift_share:.2%}",
    )

    # Print the embedding PSI value, formatted to four decimal places.
    print(
        "Embedding PSI:",
        f"{embedding_psi:.4f}",
    )

    # Print the mean confidence values for both the reference and current batches, as well as the drop in mean confidence.
    print(
        "Reference mean confidence:",
        f"{reference_mean_confidence:.4f}",
    )

    # Print the mean confidence value for the current batch, formatted to four decimal places.
    print(
        "Current mean confidence:",
        f"{current_mean_confidence:.4f}",
    )

    # Print the drop in mean confidence between the reference and current batches, formatted to four decimal places.
    print(
        "Confidence drop:",
        f"{confidence_drop:.4f}",
    )

    # Print whether retraining is recommended based on the drift analysis results.
    print(
        "Retraining recommended:",
        retrain_recommended,
    )

    # Print the reasons for triggering retraining based on the detected drift signals.
    print(
        "Trigger reasons:",
        trigger_reasons,
    )

    # Print the paths to the generated drift report and drift summary files for reference.
    print(
        "\nDrift report:",
        drift_report_path,
    )

    # Print the path to the generated drift summary JSON file for reference.
    print(
        "Drift summary:",
        summary_path,
    )

    return summary


if __name__ == "__main__":
    run()
