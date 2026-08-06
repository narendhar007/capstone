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

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

@torch.no_grad()
def predict(net, loader, return_embeddings: bool = False):
    # TODO 3: return (y_true, y_pred, y_prob_defect[, embeddings]) over the loader.
    """
    Generate labels, predictions and defect probabilities.

    When requested, the 512-dimensional input to the final
    classification layer is also returned as the embedding.
    """

    # Set the model to evaluation mode to ensure that layers like dropout and batch normalization behave correctly during inference.
    net.eval()

    # Initialize lists to store true labels, predicted labels, predicted probabilities, and embeddings (if requested) for each batch of data.
    true_batches = []
    prediction_batches = []
    probability_batches = []
    embedding_batches = []

    # Register a forward pre-hook to capture embeddings from the final fully connected layer of the model 
    # if return_embeddings is True. The hook will store the embeddings in the embedding_batches list.
    hook = None

    if return_embeddings:
        # Define a function to capture embeddings from the model's final fully connected layer during the forward pass.
        def capture_embeddings(module, inputs):
            # Append the detached embeddings (from the first input to the fully connected layer) to the embedding_batches list.
            embedding_batches.append(
                # Detach the embeddings from the computation graph and move them to the CPU for storage.
                inputs[0].detach().cpu()
            )

        # Register the forward pre-hook to the model's final fully connected layer (net.fc) to capture embeddings during the forward pass.
        hook = net.fc.register_forward_pre_hook(
            capture_embeddings
        )

    try:
        # Iterate over the data loader to process each batch of images and labels.
        for images, labels in loader:
            # Move the images to the specified device (CPU or GPU) as defined in the configuration.
            images = images.to(config.DEVICE)

            # Capture the model's output logits by passing the images through the network.
            logits = net(images)

            # Apply the softmax function to the logits to obtain predicted probabilities for each class (DEFECT and NON-DEFECT).
            probabilities = F.softmax(
                logits,
                dim=1,
            )

            # Determine the predicted class labels by taking the index of the maximum probability for each sample in the batch.
            predictions = logits.argmax(dim=1)

            # Append the true labels, predicted labels, and predicted probabilities for the DEFECT class to their respective lists.
            true_batches.append(
                labels.cpu().numpy()
            )

            prediction_batches.append(
                predictions.cpu().numpy()
            )

            probability_batches.append(
                probabilities[
                    :,
                    config.POSITIVE_IDX,
                ].cpu().numpy()
            )

    finally:
        if hook is not None:
            hook.remove()


    # Concatenate the lists of true labels, predicted labels, and predicted probabilities into single NumPy arrays for evaluation.
    y_true = np.concatenate(true_batches)
    y_pred = np.concatenate(prediction_batches)
    y_prob = np.concatenate(probability_batches)

    # If embeddings were requested, concatenate the embedding batches into a single NumPy array and return it 
    # along with the true labels, predicted labels, and predicted probabilities.
    if return_embeddings:
        embeddings = torch.cat(
            embedding_batches
        ).numpy()

        return (
            y_true,
            y_pred,
            y_prob,
            embeddings,
        )

    return y_true, y_pred, y_prob
    raise NotImplementedError


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    # TODO 3: accuracy, precision/recall/f1 for the DEFECT class (pos_label=config.POSITIVE_IDX),
    #         macro_f1, roc_auc, confusion_matrix. Return a dict.
    """
    Calculate binary classification metrics.

    Precision, recall and F1 use the defect class as the
    positive class.
    """
    # Convert the input lists of true labels, predicted labels, and predicted probabilities into NumPy arrays for evaluation.
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    # Compute the confusion matrix using the true labels and predicted labels, specifying the range of class labels 
    # based on the number of classes defined in the configuration.
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(config.NUM_CLASSES)),
    )

    roc_auc = None

    # If there are exactly two unique classes in the true labels, compute the ROC AUC score using the true labels and predicted probabilities for the DEFECT class.
    if len(np.unique(y_true)) == 2:
        roc_auc = float(
            roc_auc_score(
                y_true,
                y_prob,
            )
        )

    # Return a dictionary containing various evaluation metrics, including the number of test samples, accuracy, precision, recall, 
    # F1 score for the DEFECT class, macro F1 score, ROC AUC score (if applicable), and the confusion matrix.
    return {
        "test_samples": int(len(y_true)),
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "defect_precision": float(
            precision_score(
                y_true,
                y_pred,
                pos_label=config.POSITIVE_IDX,
                zero_division=0,
            )
        ),
        "defect_recall": float(
            recall_score(
                y_true,
                y_pred,
                pos_label=config.POSITIVE_IDX,
                zero_division=0,
            )
        ),
        "defect_f1": float(
            f1_score(
                y_true,
                y_pred,
                pos_label=config.POSITIVE_IDX,
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "roc_auc": roc_auc,
        "confusion_matrix": matrix.tolist(),
    }
    raise NotImplementedError


def plot_eval(y_true, y_pred, y_prob, out: Path | None = None) -> Path:
    # TODO 3: save a confusion-matrix + ROC figure to artifacts/model_eval.png.
    """
    Save the confusion matrix and ROC curve.
    """
    # Set the output path for saving the evaluation plot. If an output path is provided, use it; 
    # otherwise, create a default path in the artifacts directory.
    output_path = (
        out
        or config.ARTIFACT_DIR
        / "model_eval.png"
    )

    # Create the parent directories for the output path if they do not already exist, ensuring that the directory structure 
    # is in place for saving the plot.
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Compute the confusion matrix using the true labels and predicted labels, specifying the range of class labels 
    # based on the number of classes defined in the configuration.
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(config.NUM_CLASSES)),
    )

    import matplotlib.pyplot as plt

    # Create a figure with two subplots: one for the confusion matrix and one for the ROC curve. 
    # The figure size is set to (11, 4.5) inches.
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.5),
    )

    # Confusion matrix
    matrix_image = axes[0].imshow(matrix)

    # Set the title, x-label, and y-label for the confusion matrix subplot.
    axes[0].set_title("Test Confusion Matrix")
    axes[0].set_xlabel("Predicted class")
    axes[0].set_ylabel("Actual class")

    # Set the x-ticks and y-ticks for the confusion matrix subplot, labeling them with the class names defined in the configuration.
    axes[0].set_xticks(
        range(config.NUM_CLASSES),
        config.CLASSES,
        rotation=25,
        ha="right",
    )

    # Set the y-ticks for the confusion matrix subplot, labeling them with the class names defined in the configuration.
    axes[0].set_yticks(
        range(config.NUM_CLASSES),
        config.CLASSES,
    )

    # Add text annotations to each cell of the confusion matrix subplot, displaying the corresponding value from the confusion matrix.
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axes[0].text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
            )

    # Add a colorbar to the confusion matrix subplot to indicate the scale of values in the confusion matrix.
    figure.colorbar(
        matrix_image,
        ax=axes[0],
        fraction=0.046,
        pad=0.04,
    )

    # ROC curve
    # If there are exactly two unique classes in the true labels, compute the false positive rate, true positive rate, and ROC AUC score
    # using the true labels and predicted probabilities for the DEFECT class, and plot the ROC curve on the second subplot.
    if len(np.unique(y_true)) == 2:
        false_positive_rate, true_positive_rate, _ = (
            roc_curve(
                y_true,
                y_prob,
                pos_label=config.POSITIVE_IDX,
            )
        )

        auc_value = roc_auc_score(
            y_true,
            y_prob,
        )

        axes[1].plot(
            false_positive_rate,
            true_positive_rate,
            label=f"ROC AUC = {auc_value:.3f}",
        )

        axes[1].plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            label="Chance",
        )

        axes[1].legend()
    else:
        axes[1].text(
            0.5,
            0.5,
            "ROC unavailable:\ntest data contains one class",
            ha="center",
            va="center",
        )

    axes[1].set_title("Test ROC Curve")
    axes[1].set_xlabel("False-positive rate")
    axes[1].set_ylabel("True-positive rate")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)

    figure.tight_layout()

    # Save the figure to the specified output path with a resolution of 150 DPI and tight bounding box to minimize whitespace.
    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)

    # Return the output path for the saved evaluation plot.
    return output_path
    raise NotImplementedError


def failure_cases(items, y_true, y_pred, y_prob, limit: int = 20) -> list[dict]:
    # TODO 3: list misclassified samples with predicted p_defect (error analysis).
    """
    Return the most confident incorrect predictions.
    """
    # Initialize an empty list to store information about misclassified samples, including their file paths, true labels, 
    # predicted labels, error types, predicted probabilities for the DEFECT class, and predicted confidence scores.
    errors = []

    # Iterate over the zipped lists of items, true labels, predicted labels, and predicted probabilities to identify misclassified samples.
    for (
        (path, _),
        true_label,
        predicted_label,
        defect_probability,
    ) in zip(
        items,
        y_true,
        y_pred,
        y_prob,
    ):
        true_label = int(true_label)
        predicted_label = int(predicted_label)
        defect_probability = float(
            defect_probability
        )

        # If the true label matches the predicted label, skip this sample as it is correctly classified.
        if true_label == predicted_label:
            continue

        # Determine the type of error (false negative or false positive) based on the true label and predicted label.
        if (
            true_label == config.POSITIVE_IDX
            and predicted_label != config.POSITIVE_IDX
        ):
            error_type = "false_negative"
        else:
            error_type = "false_positive"

        # Calculate the predicted confidence score based on the predicted label and the predicted probability for the DEFECT class.
        predicted_confidence = (
            defect_probability
            if predicted_label == config.POSITIVE_IDX
            else 1.0 - defect_probability
        )

        # Append a dictionary containing information about the misclassified sample to the errors list.
        errors.append(
            {
                "path": str(path),
                "file_name": Path(path).name,
                "true_label": config.IDX_TO_CLASS[
                    true_label
                ],
                "predicted_label": config.IDX_TO_CLASS[
                    predicted_label
                ],
                "error_type": error_type,
                "p_defect": defect_probability,
                "predicted_confidence": float(
                    predicted_confidence
                ),
            }
        )

    # Sort the list of misclassified samples in descending order based on the predicted confidence score,
    # so that the most confident incorrect predictions appear first.
    errors.sort(
        key=lambda item: item[
            "predicted_confidence"
        ],
        reverse=True,
    )

    # Return a limited number of the most confident incorrect predictions based on the specified limit.
    return errors[:limit]
    raise NotImplementedError
