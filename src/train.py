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

from collections import Counter
from copy import deepcopy

from sklearn.metrics import f1_score

def set_seed(seed: int = config.RANDOM_SEED) -> None:
    """Set random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _subsample(items, cap):
    # TODO: stratified subsample to `cap` (or return items if cap falsy/too small).
    """
    Return a reproducible class-balanced sample.

    If the cap is zero, larger than the dataset, or too small to
    include every class, the original items are returned.
    """
    items = list(items)
    classes = sorted({label for _, label in items})

    if not cap or cap >= len(items) or cap < len(classes):
        return items

    random_generator = random.Random(config.RANDOM_SEED)

    # Group items by class label.
    items_by_class = {
        label: [
            item
            for item in items
            if item[1] == label
        ]
        for label in classes
    }

    items_per_class = cap // len(classes)
    remainder = cap % len(classes)

    selected_items = []

    # Distribute the remainder across the first few classes to ensure a balanced sample.
    for index, label in enumerate(classes):
        sample_count = items_per_class

        if index < remainder:
            sample_count += 1

        # Ensure we don't sample more items than available in the class.
        sample_count = min(
            sample_count,
            len(items_by_class[label]),
        )

        # Select a random sample of items for the current class.
        selected_items.extend(
            random_generator.sample(
                items_by_class[label],
                sample_count,
            )
        )

    random_generator.shuffle(selected_items)

    return selected_items
    raise NotImplementedError


def class_weights(items) -> torch.Tensor:
    # TODO: inverse-frequency class weights for CrossEntropyLoss.
    """Calculate inverse-frequency weights for each class."""

    # Count the number of occurrences of each class label in the dataset.
    class_counts = Counter(
        label
        for _, label in items
    )

    # Ensure that the number of unique classes matches the expected number of classes.
    if len(class_counts) != config.NUM_CLASSES:
        raise ValueError(
            "Training data must contain every configured class."
        )

    total_images = len(items)

    # Calculate the inverse-frequency weights for each class.
    weights = [
        total_images
        / (
            config.NUM_CLASSES
            * class_counts[class_index]
        )
        for class_index in range(config.NUM_CLASSES)
    ]

    #return the weights as a PyTorch tensor of type float32.
    return torch.tensor(
        weights,
        dtype=torch.float32,
    )
    raise NotImplementedError

# implement the training epoch with optional validation, returning average loss and defect-class F1 score.
def run_epoch( net, loader, loss_function, optimizer=None) -> tuple[float, float]:
    """
    Run one training or validation epoch.

    Providing an optimizer enables training. Without an optimizer,
    the function performs validation only.
    """
    # Determine if we are in training mode based on the presence of an optimizer.
    training = optimizer is not None

    # Set the model to training or evaluation mode based on the `training` flag.
    if training:
        net.train()
    else:
        net.eval()

    total_loss = 0.0
    true_labels = []
    predicted_labels = []

    # Use the appropriate context manager for gradient computation based on the training mode.
    context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    # Iterate over the data loader to process batches of images and labels.
    with context:
        for images, labels in loader:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            # Zero the gradients of the optimizer if we are in training mode.
            if training:
                optimizer.zero_grad()

            # Forward pass: compute the model's predictions (logits) for the input images.
            logits = net(images)
            # Compute the loss between the predicted logits and the true labels using the provided loss function.
            loss = loss_function(logits, labels)

            # Backward pass and optimization step if we are in training mode.
            if training:
                loss.backward()
                optimizer.step()

            # Accumulate the total loss, weighted by the number of samples in the current batch.
            total_loss += (
                loss.item() * labels.size(0)
            )

            # Determine the predicted class labels by taking the argmax of the logits along the class dimension.
            predictions = logits.argmax(dim=1)

            # Extend the lists of true and predicted labels with the current batch's labels and predictions,
            # converting them to CPU and list format for further evaluation.
            true_labels.extend(
                labels.cpu().tolist()
            )

            # Extend the list of predicted labels with the current batch's predictions,
            # converting them to CPU and list format for further evaluation.
            predicted_labels.extend(
                predictions.cpu().tolist()
            )


    # Calculate the average loss over the entire dataset.
    average_loss = total_loss / len(loader.dataset)

    # Calculate the F1 score for the defect class using the true and predicted labels.
    defect_f1 = f1_score(
        true_labels,
        predicted_labels,
        pos_label=config.POSITIVE_IDX,
        zero_division=0,
    )

    # Return the average loss and defect class F1 score as floats.
    return float(average_loss), float(defect_f1)

# implement the training loop with early stopping based on validation F1 score.
def train_model( net, train_loader, val_loader, loss_function, optimizer, log_to_mlflow=False,) -> tuple[list[dict], int, float]:
    """
    Train the model and stop when validation F1 stops improving.

    The best-performing model weights are restored before returning.
    """

    #If MLflow logging is enabled, import the mlflow module for experiment tracking.
    if log_to_mlflow:
        import mlflow

    history = []

    # Track the best validation F1 score and corresponding epoch and model state.
    best_val_f1 = -1.0
    best_epoch = 0
    best_model_state = deepcopy(net.state_dict())

    # Track the number of consecutive epochs without improvement in validation F1 score.
    epochs_without_improvement = 0

    # Loop over the specified number of epochs for training.
    for epoch in range(1, config.EPOCHS + 1):
        # Run a training epoch and obtain the average loss and defect-class F1 score for the training set.
        train_loss, train_f1 = run_epoch(
            net,
            train_loader,
            loss_function,
            optimizer,
        )

        # Run a validation epoch and obtain the average loss and defect-class F1 score for the validation set.
        # Note that the optimizer is not provided, indicating that this is a validation pass.
        val_loss, val_f1 = run_epoch(
            net,
            val_loader,
            loss_function,
        )

        # Store the results of the current epoch in a dictionary.
        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_f1": val_f1,
        }

        # Append the current epoch's results to the training history list.
        history.append(epoch_result)

        # If MLflow logging is enabled, log the metrics for the current epoch to MLflow for experiment tracking.
        if log_to_mlflow:
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_f1": train_f1,
                    "val_loss": val_loss,
                    "val_f1": val_f1,
                },
                step=epoch,
            )

        # Print the results of the current epoch, including training and validation loss and F1 scores.
        print(
            f"Epoch {epoch}/{config.EPOCHS} | "
            f"train loss: {train_loss:.4f} | "
            f"train F1: {train_f1:.4f} | "
            f"val loss: {val_loss:.4f} | "
            f"val F1: {val_f1:.4f}"
        )

        # Check if the validation F1 score has improved compared to the best recorded value.
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_model_state = deepcopy(
                net.state_dict()
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

            if (
                epochs_without_improvement
                >= config.EARLY_STOP_PATIENCE
            ):
                print(
                    "Early stopping: validation F1 "
                    "did not improve."
                )
                break

    # Return the model to its best validation state.
    net.load_state_dict(best_model_state)

    # Return the training history, the epoch with the best validation F1 score, and the best validation F1 score itself.
    return history, best_epoch, best_val_f1

def save_reference_baseline(net, ref_items) -> dict:
    # TODO 4: save reference_features.csv + reference_embeddings.npz for clean ref images.

    raise NotImplementedError


def main() -> int:
    import mlflow, mlflow.pytorch
    from mlflow import MlflowClient
    from mlflow.models import infer_signature
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

    dataset_version = "v1"

    # Reuse the immutable split manifests created in Stage 1.
    # Load the training and validation items from the dataset split manifests.
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

    # Subsample the training items to a maximum number of images specified in the configuration.
    train_items = _subsample(
        train_items,
        config.MAX_TRAIN_IMAGES,
    )

    # The generator makes training-data shuffling reproducible.
    data_generator = torch.Generator()
    data_generator.manual_seed(config.RANDOM_SEED)

    # Create DataLoader instances for the training and validation datasets using the CastingDataset class.
    # The DataLoader handles batching, shuffling, and parallel data loading. CastingDataset is defined in dataset.py
    train_loader = DataLoader(
        CastingDataset(
            train_items,
            train=True,
        ),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        generator=data_generator,
    )

    # Create a DataLoader for the validation dataset without shuffling, as validation data should be evaluated in a consistent order.
    val_loader = DataLoader(
        CastingDataset(
            val_items,
            train=False,
        ),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    # Build the model, compute class weights, define the loss function and optimizer for training.
    # build_model() is defined in model.py and returns a ResNet18 model with a new classification head.
    net = build_model().to(config.DEVICE)

    # Compute class weights for the training dataset to handle class imbalance.
    # The class_weights function calculates inverse-frequency weights for each class.
    weights = class_weights(train_items).to(
        config.DEVICE
    )

    # Define the loss function as CrossEntropyLoss with the computed class weights to handle class imbalance during training.
    loss_function = nn.CrossEntropyLoss(
        weight=weights
    )

    # Define the optimizer as Adam, optimizing only the trainable parameters of the model (the new classification head).
    optimizer = torch.optim.Adam(
        trainable_parameters(net),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # Set the MLflow tracking URI and experiment name for logging training parameters, metrics, and models.
    mlflow.set_tracking_uri(
        config.MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        config.MLFLOW_EXPERIMENT
    )

    # Print out the dataset version, number of training and validation images, class weights,
    # and the number of trainable parameters in the model for reference.
    print("Dataset version:", dataset_version)
    print("Training images:", len(train_items))
    print("Validation images:", len(val_items))
    print("Class weights:", weights.cpu().tolist())
    print(
        "Trainable parameters:",
        sum(
            parameter.numel()
            for parameter in trainable_parameters(net)
        ),
    )

    # Define the MLflow run name based on the model backbone and dataset version for easy identification of the
    # training run in the MLflow UI.
    run_name = (
        f"{config.BACKBONE}_{dataset_version}"
    )

    # Start an MLflow run for logging training parameters, metrics, and the trained model. The run name is set to the defined run_name.
    with mlflow.start_run(
        run_name=run_name
    ) as run:

        mlflow.log_params(
            {
                "dataset_version": dataset_version,
                "backbone": config.BACKBONE,
                "freeze_backbone":
                    config.FREEZE_BACKBONE,
                "num_classes": config.NUM_CLASSES,
                "positive_class":
                    config.POSITIVE_CLASS,
                "training_images": len(train_items),
                "validation_images": len(val_items),
                "batch_size": config.BATCH_SIZE,
                "epochs_requested": config.EPOCHS,
                "learning_rate":
                    config.LEARNING_RATE,
                "weight_decay":
                    config.WEIGHT_DECAY,
                "early_stop_patience":
                    config.EARLY_STOP_PATIENCE,
                "random_seed":
                    config.RANDOM_SEED,
                "image_size":
                    config.IMG_SIZE,
                "class_weights":
                    json.dumps(
                        weights.cpu().tolist()
                    ),
            }
        )

        # Set MLflow tags for the current run, including the stage of training, dataset version, and model family.
        mlflow.set_tags(
            {
                "stage": "baseline_training",
                "dataset_version":
                    dataset_version,
                "model_family": "ResNet",
            }
        )

        # Train the model using the train_model function, which implements the training loop with early stopping based on validation F1 score.
        history, best_epoch, best_val_f1 = (
            train_model(
                net,
                train_loader,
                val_loader,
                loss_function,
                optimizer,
                log_to_mlflow=True,
            )
        )

        # Save the trained model to disk using the save_model function, which handles saving the
        # model's state dictionary and any necessary metadata. save_model() is defined in model.py and
        # saves the model to the path specified in config.MODEL_PATH.
        save_model(net)

        # Log the best validation F1 score, the epoch at which it occurred, and the
        # total number of epochs completed to MLflow for tracking.
        mlflow.log_metrics(
            {
                "best_val_f1": best_val_f1,
                "best_epoch": best_epoch,
                "epochs_completed":
                    len(history),
            }
        )

        # Save the model metadata, including training parameters, best epoch, and best validation F1 score, to a JSON file for reference.
        model_metadata = {
            "mlflow_run_id": run.info.run_id,
            "dataset_version":
                dataset_version,
            "random_seed":
                config.RANDOM_SEED,
            "device": config.DEVICE,
            "training_images":
                len(train_items),
            "validation_images":
                len(val_items),
            "batch_size":
                config.BATCH_SIZE,
            "epochs_requested":
                config.EPOCHS,
            "epochs_completed":
                len(history),
            "best_epoch":
                best_epoch,
            "best_val_f1":
                best_val_f1,
            "learning_rate":
                config.LEARNING_RATE,
            "weight_decay":
                config.WEIGHT_DECAY,
            "class_weights":
                weights.cpu().tolist(),
            "history": history,
        }

        # Save the model metadata to a JSON file for reference, allowing for reproducibility and tracking of training parameters and results.
        config.MODEL_META_PATH.write_text(
            json.dumps(
                model_metadata,
                indent=2,
            ),
            encoding="utf-8",
        )

        # Set the model to evaluation mode before saving the reference baseline, ensuring that
        # any layers like dropout or batch normalization behave correctly during inference.
        net.eval()

        # Create a dummy input example for the model, which is a zero tensor with the shape
        # expected by the model (batch size of 1, 3 color channels, and the configured image size).
        # This input example is used for logging the model signature in MLflow.
        input_example = np.zeros(
            (
                1,
                3,
                config.IMG_SIZE,
                config.IMG_SIZE,
            ),
            dtype=np.float32,
        )

        # Use the model to generate an example output for the dummy input, which will be used to
        # infer the model's input-output signature for logging in MLflow.
        with torch.no_grad():
            example_output = net(
                torch.from_numpy(
                    input_example
                ).to(config.DEVICE)
            ).cpu().numpy()

        # Infer the model's input-output signature using the dummy input and example output,
        # which will be logged in MLflow for reproducibility and tracking of the model's expected input and output formats.
        signature = infer_signature(
            input_example,
            example_output,
        )

        # Log the trained model to MLflow, including the model's state dictionary, input example, and inferred signature.
        mlflow.pytorch.log_model(
            pytorch_model=net,
            name="model",
            input_example=input_example,
            signature=signature,
        )

        # Log the model metadata and model checkpoint files as artifacts in MLflow for tracking and reproducibility.
        mlflow.log_artifact(
            str(config.MODEL_META_PATH),
            artifact_path="metadata",
        )

        mlflow.log_artifact(
            str(config.MODEL_PATH),
            artifact_path="checkpoints",
        )

        # Log the dataset split metadata to MLflow if it exists, allowing for tracking of the dataset version
        # and splits used for training and validation.
        split_metadata_path = (
            config.SPLIT_DIR
            / dataset_version
            / "metadata.json"
        )

        # If the split metadata file exists, log it as an artifact in MLflow for tracking the dataset version
        # and splits used for training and validation.
        if split_metadata_path.exists():
            mlflow.log_artifact(
                str(split_metadata_path),
                artifact_path="dataset",
            )

        # Print out the MLflow experiment name, run ID, and model URI for reference, allowing for easy access
        # to the MLflow UI to view the training run and model artifacts.
        print(
            "MLflow experiment:",
            config.MLFLOW_EXPERIMENT,
        )

        print(
            "MLflow run ID:",
            run.info.run_id,
        )

        print(
            "MLflow model URI:",
            f"runs:/{run.info.run_id}/model",
        )

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"Model saved to: {config.MODEL_PATH}")
    print(
        f"Metadata saved to: "
        f"{config.MODEL_META_PATH}"
    )

    return 0
    raise NotImplementedError("Implement the training + MLflow + registry workflow")


if __name__ == "__main__":
    raise SystemExit(main())
