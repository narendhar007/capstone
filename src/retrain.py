"""retrain.py — Stage 4: drift-triggered retraining, version compare, rollback.

Implement: read drift_summary.json; if retraining is recommended, measure the production
model on a drifted batch, train a drift-augmented candidate, compare, then PROMOTE the
candidate to @production only if it improves (within PROMOTE_EPSILON) else ROLL BACK.
Record retraining_decision.json + manage MLflow registry versions.   Run: python -m src.retrain
"""
from __future__ import annotations

import json, random
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src import data_prep, evaluate
from src.model import build_model, trainable_parameters, load_model, save_model
from src.monitoring import corrupt
from src.train import _subsample, set_seed, class_weights, train_model, save_reference_baseline, mlflow_tracking_uri

from datetime import datetime, timezone

# Around 40% of the retraining data is exposed to
# simulated production drift.
RETRAIN_CORRUPT_FRACTION = 0.40

# Define a class for a dataset that contains a deterministic mixture of clean and simulated-drift images. 
# This class will be used to create a DataLoader for retraining the model on drifted data.
class DriftAugmentedDataset(Dataset):
    """
    Dataset containing a deterministic mixture of clean and
    simulated-drift images.
    """

    # The constructor initializes the dataset with a list of items (image paths and labels), a fraction of images to be corrupted, 
    # and a flag indicating whether the dataset is for training or evaluation. It also sets up the image transformations and 
    # determines which images will be corrupted based on the specified fraction.
    def __init__(
        self,
        items,
        corrupt_fraction: float,
        train: bool,
    ):
        # Initialize the dataset with a list of items (image paths and labels), a fraction of images to be corrupted, 
        # and a flag indicating whether the dataset is for training or evaluation.
        self.items = list(items)

        # Set up the image transformations based on whether the dataset is for training or evaluation.
        self.transform = data_prep.get_transforms(
            train=train
        )

        # Determine the number of images to be corrupted based on the specified fraction and 
        # randomly select which images will be corrupted.
        corrupt_count = int(
            round(
                len(self.items)
                * corrupt_fraction
            )
        )

        # Use a fixed random seed for reproducibility when selecting which images to corrupt.
        rng = np.random.default_rng(
            config.RANDOM_SEED
        )

        # If the number of images to be corrupted is greater than zero, randomly select indices of images to corrupt.
        if corrupt_count > 0:
            # Use a fixed random seed for reproducibility when selecting which images to corrupt. Raise an error if the number of images 
            # to be corrupted exceeds the total number of images in the dataset.
            selected = rng.choice(
                len(self.items),
                size=corrupt_count,
                replace=False,
            )

            # Store the selected indices in a set for efficient lookup when determining whether to corrupt an image during data loading.
            self.corrupt_indices = set(
                int(index)
                for index in selected
            )
        else:
            # If no images are to be corrupted, initialize an empty set for corrupt indices.
            self.corrupt_indices = set()


    # The __len__ method returns the total number of items in the dataset.
    def __len__(self):
        return len(self.items)
    
    # The __getitem__ method retrieves an item (image and label) from the dataset at the specified index.
    # If the image is marked for corruption, it applies a repeatable corruption before returning the image tensor and label.
    def __getitem__(self, index):
        image_path, label = self.items[index]

        # Open the image file, convert it to grayscale, and create a copy of the image to avoid modifying the original.
        with Image.open(image_path) as opened_image:
            image = (
                opened_image
                .convert("L")
                .copy()
            )

        # If the current index is in the set of corrupt indices, apply a repeatable corruption to the image using a fixed random seed 
        # based on the index. This ensures that the same image will always be corrupted in the same way across different runs. 
        # After applying any necessary corruption, transform the image into a tensor and return it along with its label.
        if index in self.corrupt_indices:
            # Give every image a repeatable corruption.
            # This is particularly important when comparing
            # production and candidate models on the same
            # drifted evaluation batch.
            random_state = np.random.get_state()

            # Use a fixed random seed based on the index to ensure that the corruption applied to the image is repeatable 
            # across different runs.
            np.random.seed(
                config.RANDOM_SEED
                + index
            )

            try:
                image = corrupt(image)
            finally:
                np.random.set_state(
                    random_state
                )

        # Transform the image into a tensor using the specified transformations and return the image tensor along with its label.
        image_tensor = self.transform(
            image
        )

        return image_tensor, label

# Define a function to create a DataLoader for the DriftAugmentedDataset, allowing for reproducible loading of clean and drift-mixed data.
def make_loader(
    items,
    corrupt_fraction: float,
    train: bool,
    shuffle: bool,
):
    """Build a reproducible clean/drift-mixed DataLoader."""

    # Create an instance of the DriftAugmentedDataset with the specified items, corruption fraction, and training flag.
    dataset = DriftAugmentedDataset(
        items=items,
        corrupt_fraction=corrupt_fraction,
        train=train,
    )

    # Set up a random number generator for shuffling the dataset if the shuffle flag is set to True. 
    # This ensures that the order of the data is randomized in a reproducible manner.
    generator = None

    # If the shuffle flag is set to True, create a random number generator with a fixed seed for reproducibility.
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(
            config.RANDOM_SEED
        )

    # Create a DataLoader for the DriftAugmentedDataset, specifying the batch size, shuffle option, number of worker processes,
    # and the random number generator for reproducibility. The DataLoader will yield batches of data from the dataset during training or 
    # evaluation.
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        generator=generator,
    )

    return loader

# Define a function to evaluate retraining triggers based on drift metrics, determining whether retraining 
# is recommended and providing details on the signals that triggered the decision.
def retraining_triggers(
    drift: dict,
):
    """
    Evaluate the three measurable retraining signals.
    """

    # Extract the relevant drift metrics from the input dictionary, including feature drift, embedding drift, and confidence drop.
    feature = drift[
        "feature_drift"
    ]

    embedding = drift[
        "embedding_drift"
    ]

    confidence = drift[
        "confidence"
    ]

    # Evaluate the three measurable retraining signals based on the extracted drift metrics, determining whether 
    # retraining is recommended and providing details on the signals that triggered the decision.
    signals = {
        "feature_drift": {
            "value": float(
                feature["drift_share"]
            ),
            "threshold": float(
                feature[
                    "drift_share_threshold"
                ]
            ),
            "fired": bool(
                feature["drift_share"]
                >= feature[
                    "drift_share_threshold"
                ]
            ),
        },

        "embedding_drift": {
            "value": float(
                embedding["psi"]
            ),
            "threshold": float(
                embedding["threshold"]
            ),
            "fired": bool(
                embedding["psi"]
                > embedding["threshold"]
            ),
        },

        "confidence_drop": {
            "value": float(
                confidence["drop"]
            ),
            "threshold": float(
                confidence["threshold"]
            ),
            "fired": bool(
                confidence["drop"]
                > confidence["threshold"]
            ),
        },
    }

    # Determine which retraining signals have fired by checking the "fired" status of each signal in the signals dictionary.
    trigger_reasons = [
        signal_name
        for signal_name, details
        in signals.items()
        if details["fired"]
    ]

    # Determine whether retraining is recommended based on whether any of the retraining signals have fired.
    retrain = bool(
        trigger_reasons
    )

    # Return a tuple containing the retraining recommendation, the signals dictionary, and the list of trigger reasons.
    return (
        retrain,
        signals,
        trigger_reasons,
    )



def main() -> int:

    """Main entrypoint for drift-triggered retraining, version compare, rollback."""
    import mlflow
    import mlflow.pytorch

    from mlflow import MlflowClient

    set_seed() # From train.py, for reproducibility.

    # Define paths for drift summary and retraining decision JSON files in the artifact directory.
    drift_summary_path = (
        config.ARTIFACT_DIR
        / "drift_summary.json"
    )

    # Define the path for the retraining decision JSON file in the artifact directory.
    decision_path = (
        config.ARTIFACT_DIR
        / "retraining_decision.json"
    )

    # Check if the drift summary JSON file exists; if not, raise a FileNotFoundError and instruct the user to run the monitoring script first.
    if not drift_summary_path.exists():
        raise FileNotFoundError(
            "drift_summary.json was not found. "
            "Run `python -m src.monitoring` first."
        )

    # Read the drift summary JSON file and parse its contents into a dictionary.
    drift = json.loads(
        drift_summary_path.read_text(
            encoding="utf-8"
        )
    )
    
    # Evaluate retraining triggers based on the drift metrics and determine whether retraining is recommended.
    (
        retrain,
        trigger_signals,
        trigger_reasons,
    ) = retraining_triggers(
        drift
    )

    # Print the retraining recommendation and the reasons for the decision, including the values 
    # and thresholds of the relevant drift signals.
    print(
        "Retraining trigger:",
        retrain,
    )

    print(
        "Trigger reasons:",
        trigger_reasons,
    )

    # Print the details of each retraining signal, including its value, threshold, and whether it fired.
    for signal_name, details in (
        trigger_signals.items()
    ):
        print(
            f"  {signal_name}: "
            f"value={details['value']:.4f}, "
            f"threshold="
            f"{details['threshold']:.4f}, "
            f"fired={details['fired']}"
        )


    # Evaluate retraining triggers based on the drift metrics and determine whether retraining is recommended.
    # ---------------------------------------------------------
    # No drift trigger → record decision and stop.
    # ---------------------------------------------------------
    if not retrain:
        # TODO 4: record a 'no_retrain' decision and return.

        # Record the retraining decision in a JSON file, including the timestamp, whether retraining was triggered,
        # the signals that triggered the decision, and the action taken (in this case, "no_retrain").
        decision = {
            "generated_at_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "retrain_triggered": False,

            "trigger_signals":
                trigger_signals,

            "trigger_reasons":
                trigger_reasons,

            "action":
                "no_retrain",
        }

        # Write the retraining decision to a JSON file in the artifact directory for record-keeping and future reference.
        decision_path.write_text(
            json.dumps(
                decision,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            "No retraining required."
        )

        return 0

    # ---------------------------------------------------------
    # Load the immutable dataset version associated with the
    # monitoring result.
    # ---------------------------------------------------------
    dataset_version = drift.get(
        "dataset_version",
        "v1",
    )

    # Find the root directory containing the dataset splits for the specified version, and load the training, 
    # validation, and test items (image paths and labels) from the corresponding splits.
    root = data_prep.find_data_root()

    train_items = data_prep.load_split(
        dataset_version,
        "train",
        root,
    )

    val_items = data_prep.load_split(
        dataset_version,
        "val",
        root,
    )

    test_items = data_prep.load_split(
        dataset_version,
        "test",
        root,
    )

    # Subsample the training items to limit the number of images used for retraining.
    train_items = _subsample(
        train_items,
        config.MAX_TRAIN_IMAGES,
    )

    # ---------------------------------------------------------
    # Candidate training:
    # 60% clean + 40% simulated-drift images.
    # ---------------------------------------------------------

    # Create DataLoaders for the training, validation, and fully drifted test datasets, using the DriftAugmentedDataset class to
    # generate a mixture of clean and drifted images for training and validation, and a fully drifted dataset for evaluation.
    train_loader = make_loader(
        train_items,
        corrupt_fraction=(
            RETRAIN_CORRUPT_FRACTION
        ),
        train=True,
        shuffle=True,
    )

    val_loader = make_loader(
        val_items,
        corrupt_fraction=(
            RETRAIN_CORRUPT_FRACTION
        ),
        train=False,
        shuffle=False,
    )

    # Fully drifted evaluation batch for Stage 4.5.
    drifted_test_loader = make_loader(
        test_items,
        corrupt_fraction=1.0,
        train=False,
        shuffle=False,
    )

    # ---------------------------------------------------------
    # Production and candidate start from exactly the same
    # production checkpoint.
    # ---------------------------------------------------------

    # Load the production and candidate models from the same production
    # checkpoint. The ResNet18 backbone remains frozen, while the candidate
    # classification head remains trainable.
    production_model = load_model(
        path=config.MODEL_PATH,
        freeze=True,
    )

    candidate_model = load_model(
        path=config.MODEL_PATH,
        freeze=True,
    )

    # Compute class weights for the training dataset to handle class imbalance during training, and move the weights 
    # to the appropriate device (CPU or GPU) for use in the loss function.
    weights = class_weights(
        train_items
    ).to(
        config.DEVICE
    )

    # Define the loss function for training the candidate model, using cross-entropy loss with the computed class weights 
    # to account for class imbalance.
    loss_function = nn.CrossEntropyLoss(
        weight=weights
    )

    # Define the optimizer for training the candidate model, using the Adam optimizer with the specified learning rate and weight decay,
    # and only optimizing the parameters of the candidate model that require gradients (i.e., those that are not frozen).
    optimizer = torch.optim.Adam(
        trainable_parameters(
            candidate_model
        ),
        lr=config.LEARNING_RATE,
        weight_decay=(
            config.WEIGHT_DECAY
        ),
    )

    # Define the path for saving the candidate model checkpoint in the artifact directory, which will be used for 
    # evaluation and potential promotion to production.
    candidate_path = (
        config.ARTIFACT_DIR
        / "candidate_model.pt"
    )

    # ---------------------------------------------------------
    # MLflow setup.
    # ---------------------------------------------------------
    tracking_uri = mlflow_tracking_uri()

    mlflow.set_tracking_uri(
        tracking_uri
    )

    mlflow.set_experiment(
        config.MLFLOW_EXPERIMENT
    )

    client = MlflowClient(
        tracking_uri=tracking_uri
    )

    # Determine which model version currently owns
    # the production alias.
    production_version_before = (
        client.get_model_version_by_alias(
            name=config.REGISTERED_MODEL,
            alias=config.PRODUCTION_ALIAS,
        )
    )

    # Get the version number of the production model before retraining, and convert it to a string for logging and comparison purposes.
    production_version_before = str(
        production_version_before.version
    )

    print(
        "\nProduction version before retraining:",
        production_version_before,
    )

    # Define a unique run name for the MLflow run associated with drift-triggered retraining, incorporating 
    # the dataset version into the name for clarity and traceability.
    run_name = (
        f"drift_retraining_"
        f"{dataset_version}"
    )

    # Start an MLflow run for drift-triggered retraining, logging relevant parameters, training the candidate model,
    # saving the model, registering it, and comparing its performance against the production model on a fully drifted 
    # evaluation batch. Depending on the comparison results, the candidate model may be promoted to production or rolled back.
    with mlflow.start_run(
        run_name=run_name
    ) as run:

        # Log relevant parameters for the drift-triggered retraining run in MLflow, including the stage, dataset version, trigger reasons,
        # corruption fraction, number of training and validation images, number of drifted test images,
        mlflow.log_params(
            {
                "stage":
                    "drift_retraining",

                "dataset_version":
                    dataset_version,

                "trigger_reasons":
                    json.dumps(
                        trigger_reasons
                    ),

                "corrupt_fraction":
                    RETRAIN_CORRUPT_FRACTION,

                "training_images":
                    len(train_items),

                "validation_images":
                    len(val_items),

                "drifted_test_images":
                    len(test_items),

                "production_version_before":
                    production_version_before,

                "epochs_requested":
                    config.EPOCHS,

                "learning_rate":
                    config.LEARNING_RATE,

                "weight_decay":
                    config.WEIGHT_DECAY,
            }
        )

        # TODO 4: train a drift-augmented candidate (corrupt_frac~0.4, few epochs, capped).
        # -----------------------------------------------------
        # Train the candidate.
        # -----------------------------------------------------
        (
            history,
            best_epoch,
            best_val_f1,
        ) = train_model(
            candidate_model,
            train_loader,
            val_loader,
            loss_function,
            optimizer,
            log_to_mlflow=True,
        )

        # Save the trained candidate model to the specified path in the artifact directory for later evaluation 
        # and potential promotion to production.
        save_model(
            candidate_model,
            candidate_path,
        )

        # -----------------------------------------------------
        # Register the candidate as a new model version.
        # Do NOT move production yet.
        # -----------------------------------------------------
        # Log the trained candidate model to MLflow, registering it as a new model version in the MLflow model registry 
        # under the specified registered model name.
        mlflow.pytorch.log_model(
            pytorch_model=(
                candidate_model
            ),
            name="candidate_model",
        )

        # Define the model URI for the registered candidate model version in MLflow, using the run ID of the current MLflow run and
        # the name of the logged model artifact.
        model_uri = (
            f"runs:/{run.info.run_id}"
            f"/candidate_model"
        )

        # Register the candidate model as a new version in the MLflow model registry, using the defined model URI and 
        # the specified registered model name. This allows for versioning and management of the candidate model in the registry.
        registered_version = (
            mlflow.register_model(
                model_uri=model_uri,
                name=(
                    config.REGISTERED_MODEL
                ),
            )
        )

        # Get the version number of the newly registered candidate model and convert it to a string for logging and comparison purposes.
        candidate_version = str(
            registered_version.version
        )

        print(
            "\nRegistered candidate version:",
            candidate_version,
        )

        client.set_model_version_tag(
            name=config.REGISTERED_MODEL,
            version=candidate_version,
            key="validation_status",
            value="candidate",
        )

        # -----------------------------------------------------
        # Stage 4.5 comparison is performed in this same run so
        # that we do not register another candidate later.
        # -----------------------------------------------------

        # TODO 4: evaluate current production on a fully-drifted eval batch (corrupt_frac=1.0).

        # Evaluate the production model and candidate model on a fully drifted evaluation batch, computing their 
        # respective true labels, predicted labels, and predicted probabilities. Then, calculate the performance metrics 
        # (e.g., F1 score) for both models based on their predictions.
        (
            prod_y_true,
            prod_y_pred,
            prod_y_prob,
        ) = evaluate.predict(
            production_model,
            drifted_test_loader,
        )

        # Compute the performance metrics for the production model based on its predictions on the fully drifted evaluation batch.
        production_metrics = (
            evaluate.compute_metrics(
                prod_y_true,
                prod_y_pred,
                prod_y_prob,
            )
        )

        # Evaluate the candidate model on the same fully drifted evaluation batch, computing its true labels, 
        # predicted labels, and predicted probabilities.
        (
            cand_y_true,
            cand_y_pred,
            cand_y_prob,
        ) = evaluate.predict(
            candidate_model,
            drifted_test_loader,
        )

        # Compute the performance metrics for the candidate model based on its predictions on the fully drifted evaluation batch.
        candidate_metrics = (
            evaluate.compute_metrics(
                cand_y_true,
                cand_y_pred,
                cand_y_prob,
            )
        )

        # Extract the F1 scores for the production and candidate models from their respective performance metrics.
        production_f1 = float(
            production_metrics[
                "defect_f1"
            ]
        )

        candidate_f1 = float(
            candidate_metrics[
                "defect_f1"
            ]
        )

        # Determine the required candidate F1 score for promotion to production, which is the production F1 score 
        # plus a configured epsilon value.
        required_candidate_f1 = (
            production_f1
            + config.PROMOTE_EPSILON
        )

        # TODO 4: compare candidate vs production F1 on the drifted batch.

        # Determine whether the candidate model should be promoted to production based on whether 
        # its F1 score meets or exceeds the required threshold.
        promote = (
            candidate_f1
            >= required_candidate_f1
        )

        # TODO 4: promote candidate→@production iff cand_f1 >= prod_f1 + PROMOTE_EPSILON, else
        #  rollback (keep incumbent). Register the candidate version; move/keep the alias.

        # If the candidate model meets the promotion criteria, update the production alias in the MLflow model registry 
        # to point to the newly registered candidate version.
        if promote:
            client.set_registered_model_alias(
                name=(
                    config.REGISTERED_MODEL
                ),
                alias=(
                    config.PRODUCTION_ALIAS
                ),
                version=(
                    registered_version.version
                ),
            )

            # Update the locally deployed production checkpoint.
            save_model(
                candidate_model,
                config.MODEL_PATH,
                )

            # Save a reference baseline for monitoring purposes, which includes the candidate model and the validation items used 
            # during training.
            reference_baseline = save_reference_baseline(
                candidate_model,
                val_items,
            )

            # Update the validation status tag for the candidate model version in the MLflow model registry to 
            # indicate that it has been promoted to production.
            client.set_model_version_tag(
                name=config.REGISTERED_MODEL,
                version=candidate_version,
                key="validation_status",
                value="promoted",
            )

            action = "promote"

            # Update the production version after retraining to reflect the newly promoted candidate version.
            production_version_after = (
                candidate_version
            )

            model_metadata = {}

            # If a model metadata JSON file already exists, read its contents and parse it into a dictionary. 
            # This allows for updating the existing metadata with new information related to the retraining process.
            if config.MODEL_META_PATH.exists():
                model_metadata = json.loads(
                    config.MODEL_META_PATH.read_text(
                        encoding="utf-8"
                    )
                )

            # Remove the incumbent model's clean-test metrics because they belong
            # to the previous production model. The promoted candidate's evaluation
            # on the simulated drifted batch is stored separately as
            # drifted_test_metrics.         
            model_metadata.pop(
                "test_metrics",
                None,
            )

            model_metadata.update(
                {
                    "mlflow_run_id":
                        run.info.run_id,

                    "dataset_version":
                        dataset_version,

                    "registered_model":
                        config.REGISTERED_MODEL,

                    "registered_model_version":
                        candidate_version,

                    "production_alias":
                        config.PRODUCTION_ALIAS,

                    "production_model_uri": (
                        f"models:/{config.REGISTERED_MODEL}"
                        f"@{config.PRODUCTION_ALIAS}"
                    ),

                    "promotion_source":
                        "drift_retraining",

                    "epochs_completed":
                        len(history),

                    "best_epoch":
                        best_epoch,

                    "best_val_f1":
                        float(best_val_f1),

                    "history":
                        history,

                    "drifted_test_metrics":
                        candidate_metrics,

                    "reference_baseline":
                        reference_baseline,

                    "promotion_comparison": {
                        "previous_production_version":
                            production_version_before,

                        "production_drifted_f1":
                            production_f1,

                        "candidate_drifted_f1":
                            candidate_f1,

                        "promotion_epsilon":
                            config.PROMOTE_EPSILON,
                    },
                }
            )

            # Write the updated model metadata to a JSON file in the specified path, ensuring that it is formatted with 
            # indentation for readability.
            config.MODEL_META_PATH.write_text(
                json.dumps(
                    model_metadata,
                    indent=2,
                ),
                encoding="utf-8",
            )

            # Log the updated model metadata JSON file and the reference features CSV file as artifacts in MLflow for traceability 
            # and reproducibility.
            mlflow.log_artifact(
                str(config.MODEL_META_PATH),
                artifact_path="promoted_model",
            )

            mlflow.log_artifact(
                str(
                    config.ARTIFACT_DIR
                    / "reference_features.csv"
                ),
                artifact_path="monitoring_baseline",
            )

            mlflow.log_artifact(
                str(config.REFERENCE_EMBED),
                artifact_path="monitoring_baseline",
            )
        else:
            # Candidate remains registered for auditability,
            # but production alias stays on the incumbent.
            client.set_model_version_tag(
                name=config.REGISTERED_MODEL,
                version=candidate_version,
                key="validation_status",
                value="rejected",
            )

            action = (
                "rollback_keep_incumbent"
            )

            production_version_after = (
                production_version_before
            )

        # Log the relevant metrics and tags for the drift-triggered retraining run in MLflow, including the production and 
        # candidate F1 scores, the best validation F1 score for the candidate, the candidate version, the governance action 
        # taken (promote or rollback), and the production version after the retraining decision.
        mlflow.log_metrics(
            {
                "production_drifted_f1":
                    production_f1,

                "candidate_drifted_f1":
                    candidate_f1,

                "candidate_best_val_f1":
                    best_val_f1,
            }
        )

        mlflow.set_tags(
            {
                "candidate_version":
                    candidate_version,

                "governance_action":
                    action,

                "production_version_after":
                    production_version_after,
            }
        )

        # -----------------------------------------------------
        # One decision artifact supports both 4.4 and 4.5.
        # -----------------------------------------------------
        

        # Record the retraining decision in a JSON file, including the timestamp, whether retraining was triggered, the signals 
        # that triggered the decision, the candidate training details, the comparison with the production model, the action taken, 
        # and the production and candidate versions.
        decision = {
            "generated_at_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "retrain_triggered":
                True,

            "trigger_signals":
                trigger_signals,

            "trigger_reasons":
                trigger_reasons,

            "candidate_training": {
                "dataset_version":
                    dataset_version,

                "training_images":
                    len(train_items),

                "validation_images":
                    len(val_items),

                "corrupt_fraction":
                    RETRAIN_CORRUPT_FRACTION,

                "epochs_completed":
                    len(history),

                "best_epoch":
                    best_epoch,

                "best_val_f1":
                    float(best_val_f1),

                "mlflow_run_id":
                    run.info.run_id,

                "registered_model":
                    config.REGISTERED_MODEL,

                "candidate_version":
                    candidate_version,

                "candidate_model_path":
                    str(candidate_path),

                "history":
                    history,
            },

            "comparison": {
                "production_drifted_f1":
                    production_f1,

                "candidate_drifted_f1":
                    candidate_f1,

                "promotion_epsilon":
                    config.PROMOTE_EPSILON,

                "required_candidate_f1":
                    required_candidate_f1,
            },

            "action":
                action,

            "production_version_before":
                production_version_before,

            "candidate_version":
                candidate_version,

            "production_version_after":
                production_version_after,
        }

        # TODO 4: write retraining_decision.json (scores, action, version history).
        # Write the retraining decision to a JSON file in the artifact directory for record-keeping and future reference.
        decision_path.write_text(
            json.dumps(
                decision,
                indent=2,
            ),
            encoding="utf-8",
        )

        # Log the retraining decision JSON file and the candidate model checkpoint as artifacts in MLflow for traceability 
        # and reproducibility.
        mlflow.log_artifact(
            str(decision_path)
        )

        mlflow.log_artifact(
            str(candidate_path)
        )

    print(
        "\nCandidate best epoch:",
        best_epoch,
    )

    print(
        "Candidate best validation F1:",
        f"{best_val_f1:.4f}",
    )

    print(
        "Production drifted F1:",
        f"{production_f1:.4f}",
    )

    print(
        "Candidate drifted F1:",
        f"{candidate_f1:.4f}",
    )

    print(
        "Governance action:",
        action,
    )

    print(
        "Production version after:",
        production_version_after,
    )

    print(
        "\nDecision artifact:",
        decision_path,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
