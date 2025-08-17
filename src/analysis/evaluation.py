#!/usr/bin/env python3

import numpy as np
from sklearn.metrics import (
    roc_curve, 
    roc_auc_score,  
    confusion_matrix,  
    precision_recall_curve, 
    average_precision_score,
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score
)
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Literal


def youden_threshold(
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
) -> Optional[float]:
    """
    Return the threshold that maximizes Youden's J = TPR - FPR,
    computed from a precomputed ROC (fpr, tpr, thresholds).

    Skips the sentinel +inf threshold that sklearn puts first.

    Returns
    -------
    float or None
        Best finite threshold (Youden). Returns None if no finite
        threshold is available (extremely rare).
    """
    finite = np.isfinite(thresholds)
    if not finite.any():
        return None
    j = tpr[finite] - fpr[finite]
    idx = int(np.argmax(j))
    return float(thresholds[finite][idx])




def threshold_at_target_tpr(
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
    target_tpr: float = 0.95,
) -> Optional[float]:
    """
    Return the threshold whose TPR is closest to `target_tpr`,
    using a precomputed ROC (fpr, tpr, thresholds).

    Skips the sentinel +inf threshold that sklearn puts first.

    Returns
    -------
    float or None
        Threshold at target TPR (closest). Returns None if no finite
        threshold is available.
    """
    target_tpr = float(np.clip(target_tpr, 0.0, 1.0))
    finite = np.isfinite(thresholds)
    if not finite.any():
        return None
    fpr_f, tpr_f, thr_f = fpr[finite], tpr[finite], thresholds[finite]
    idx = int(np.argmin(np.abs(tpr_f - target_tpr)))
    return float(thr_f[idx])




def compute_metrics(
    scores_id: np.ndarray,
    scores_ood: np.ndarray,
    plot: bool = False,
    save_path: str | None = None,
    threshold_strategy: Literal["youden", "target_tpr"] = "youden",
    target_tpr: float = 0.95,
) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute auROC, FPR@TPR=95% (FPR95), AUC-PR, and select a decision threshold
    according to a chosen strategy, for OOD  detection using OOD/ID scores.

    This function:
    - Concatenates ID (label 0) and OOD (label 1) scores,
    - Computes auROC, FPR95, and AUC-PR and optionally plots the ROC curve 
    - Determines the optimal threshold using either:
        - 'youden'     : maximizes Youden's J = TPR - FPR (finite thresholds only)
        - 'target_tpr' : picks the threshold whose TPR is closest to `target_tpr`

    Metrics:
    --------
    auROC (Area Under the ROC Curve): 
        A global measure of the model's ability to distinguish between two classes (ID vs OOD)
        across all possible thresholds. The closer to 1, the better the separation.
    FPR95 (False Positive Rate at 95% TPR): 
        The false positive rate when the true positive rate reaches 95%. Indicates the proportion 
        of ID samples incorrectly classified as OOD when 95% of OOD samples are correctly detected.
    AUC-PR (Area Under the Precision-Recall Curve): 
        The area under the precision-recall curve, which is especially useful for evaluating model
        performance on imbalanced datasets. Higher values indicate better ability to identify OOD 
        among ID samples.

    Parameters
    ----------
    scores_id : np.ndarray
        Scores for in-distribution samples
    scores_ood : np.ndarray
        Scores for out-of-distribution samples
    plot : bool, optional (default=False)
        If True, plot the ROC curve.
    save_path : str, optional (default=None)
        If provided, save the ROC curve image to this path (e.g. "plot.png").
    threshold_strategy : {"youden", "target_tpr"}, default "youden"
        Strategy used to select the operating threshold.
    target_tpr : float, default 0.95
        Target TPR used when `threshold_strategy="target_tpr"`.


    Returns
    -------
    auroc : float
        Area under the ROC curve.
    fpr95 : float
        False positive rate when TPR is closest to 95%.
    auc_pr : float
        Area under the precision-recall curve (AUC-PR).
    fpr : np.ndarray
        False positive rates at different thresholds.
    tpr : np.ndarray
        True positive rates at different thresholds.
    thresholds : np.ndarray
        Thresholds corresponding to (fpr, tpr). Note: first element may be +inf.
    selected_threshold : float
        The decision threshold chosen per `threshold_strategy`.
         Samples with score <= threshold are classified as ID,
        samples with score > threshold are classified as OOD.

    Notes
    -----
    - Youden's J threshold:
      Finds the ROC point that maximizes J = TPR - FPR, i.e., pushes TPR up while keeping FPR down.
      This yields a single operating point that balances sensitivity and specificity.
    
    - Target-TPR threshold:
      Picks the threshold whose TPR is closest to a desired target (e.g., 95%).
      Common in OOD evaluations where high detection rate (recall for OOD) is required;
      it then reports the corresponding FPR at that operating point.
    
    - sklearn's roc_curve includes a sentinel +inf as the first threshold;
      threshold selection here ignores non-finite thresholds and falls back safely if needed.
    """

    # Concatenate scores and create labels
    all_scores = np.concatenate([scores_id, scores_ood])
    all_labels = np.concatenate([
        np.zeros_like(scores_id),  # 0 for ID
        np.ones_like(scores_ood)   # 1 for OOD
    ])

    # Some scoring pipelines can produce NaN/Inf => unpredictable roc_curve/roc_auc_score
    # Keeps only finite scores and drops the corresponding labels
    m = np.isfinite(all_scores)
    if not np.all(m): 
        all_scores = all_scores[m]
        all_labels = all_labels[m]

    # Compute ROC curve and auROC
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores, pos_label=1)
    auroc = roc_auc_score(all_labels, all_scores)

    # Compute FPR95: find the threshold where TPR is closest to 0.95
    tpr_95_idx = int(np.argmin(np.abs(tpr - 0.95)))
    fpr95 = fpr[tpr_95_idx]

    # Compute Precision-Recall curve and AUC-PR
    precision, recall, _ = precision_recall_curve(all_labels, all_scores, pos_label=1) # OOD as positive class (1)
    auc_pr = average_precision_score(all_labels, all_scores, pos_label=1)

    ''' # Debug
    print(
    "ID min/med/max:", np.min(scores_id), np.median(scores_id), np.max(scores_id),
    "| OOD min/med/max:", np.min(scores_ood), np.median(scores_ood), np.max(scores_ood)
    )
    print("unique score count (all):", np.unique(np.concatenate([scores_id, scores_ood])).size)
    '''

    # Select threshold per strategy (robust to +inf)
    selected_threshold: Optional[float] = None
    if threshold_strategy == "youden":
        selected_threshold = youden_threshold(fpr, tpr, thresholds)
    elif threshold_strategy == "target_tpr":
        selected_threshold = threshold_at_target_tpr(
            fpr, tpr, thresholds, target_tpr=target_tpr
        )
    else:
        raise ValueError("threshold_strategy must be 'youden' or 'target_tpr'.")

    # Fallback if selection failed (degenerate ROC): picks the median of the finite scores
    if selected_threshold is None or not np.isfinite(selected_threshold):
        finite_scores = all_scores[np.isfinite(all_scores)]
        selected_threshold = float(np.median(finite_scores)) if finite_scores.size else 0.0

    # Display results
    if plot:
        print(f"auROC: {auroc:.4f}")
        print(f"FPR95: {fpr95:.4f}")
        print(f"AUC-PR: {auc_pr:.4f}")
        print(f"Optimal Threshold ({threshold_strategy}): {selected_threshold:.4f}")

    # Optionally plot the ROC curve
    if plot or save_path:
        plt.figure()
        plt.plot(fpr, tpr, label=f"auROC = {auroc:.4f}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        if plot:
            plt.show()
        else:
            plt.close()

    return auroc, fpr95, auc_pr, fpr, tpr, thresholds, float(selected_threshold)




def compute_confusion_matrix_and_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    plot: bool = True,
    normalize: bool = False,
) -> Tuple[np.ndarray, float, float, float, float]:
    """
    Compute and optionally plot the 2x2 confusion matrix and classification metrics
    for OOD detection, given ground-truth and predicted labels.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels (0 = ID, 1 = OOD).
    y_pred : np.ndarray
        Predicted labels (0 = predicted ID, 1 = predicted OOD).
    plot : bool, optional (default=True)
        If True, plot the confusion matrix using seaborn and displays accuracy metrics.
    normalize : bool, optional (default=False)
        If True, normalize each row of the confusion matrix to sum to 1.

    Returns
    -------
    cm : np.ndarray
        Confusion matrix of shape (2, 2):
            - Rows: actual labels (0 = ID, 1 = OOD)
            - Columns: predicted labels (0 = predicted ID, 1 = predicted OOD)
    accuracy : float
        Accuracy of the predictions.
    f1 : float
        F1-score of the predictions.
    precision : float
        Precision of the predictions.
    recall : float
        Recall of the predictions.

    """
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Compute accuracy
    accuracy = accuracy_score(y_true, y_pred)
    # Compute f1-score
    f1 = f1_score(y_true, y_pred, zero_division=0)
    # Compute precision
    precision = precision_score(y_true, y_pred, zero_division=0)
    # Compute recall 
    recall = recall_score(y_true, y_pred, zero_division=0)

    if normalize:
        cm = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)

    if plot:
        # Display metrics 
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")

        # Plot confusion matrix
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt=".2f" if normalize else "d", 
            cmap="Blues",
            xticklabels=["Predicted ID", "Predicted OOD"],
            yticklabels=["True ID", "True OOD"]
        )
        plt.xlabel("Prediction")
        plt.ylabel("Ground Truth")
        title = "Row normalized" if normalize else "Raw"
        plt.title(f"{title} Confusion Matrix")
        plt.tight_layout()
        plt.show()

    return cm, accuracy, f1, precision, recall




def compute_confusion_matrix_with_attribute_split(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    attribute: np.ndarray,
    attr_labels: tuple = ("A", "U"),
    attr_true_value=True,
    normalize: bool = True,
    class_names: list = ["ID", "OOD"]
):
    """
    Plot a confusion matrix (normalized or raw) with an attribute breakdown in each cell.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels (e.g. 0 = ID, 1 = OOD)
    y_pred : np.ndarray
        Predicted binary labels (e.g. 0 = ID, 1 = OOD)
    attribute : np.ndarray
        Array used to split data (e.g. is_unanswerable), must be same length
    attr_labels : tuple
        Labels to display for attribute split (default: ("A", "U"))
    attr_true_value : bool
        The value in `attribute` that should be associated with the second label in `attr_labels`
        (i.e., `attr_labels[1]`). All other values will be counted under `attr_labels[0]`.
    normalize : bool
        If True, normalize confusion matrix by row
    class_names : list
        Class names for rows/columns (e.g. ["ID", "OOD"])
            class_names[0] => corresponds to class 0
            class_names[1] => corresponds to class 1
    """

    # Check consistency
    assert len(y_true) == len(y_pred) == len(attribute), "All input arrays must be the same length."

    # Raw and normalized confusion matrices
    cm = confusion_matrix(y_true, y_pred)
    cm_display = confusion_matrix(y_true, y_pred, normalize="true") if normalize else cm

    # Build annotations with attribute stats
    annotations = np.empty_like(cm, dtype=object)

    # Loop over true labels (row index in confusion matrix)
    for i in range(2):  
        # Mask where true label matches class i (i = true class)
        true_mask = y_true == i 
        # Loop over predicted labels (column index in confusion matrix)
        for j in range(2):  
            # Mask where pred label matches class j (j = predicted class)
            pred_mask = y_pred == j 
            # identify samples classified true=i, pred=j
            cell_mask = true_mask & pred_mask 
            # Total number of samples in this cell of the confusion matrix
            total = np.sum(cell_mask) 
            if total == 0:
                # If no data in the cell, leave it empty
                annotations[i, j] = "" 
            else:
                # Count how many of the samples in this cell have attribute == attr_true_value
                n_attr_true = np.sum(attribute[cell_mask] == attr_true_value)
                # All others are considered as attr_false
                n_attr_false = total - n_attr_true
                # Convert counts to percentages (relative to total cell count)
                p_attr_true = 100 * n_attr_true / total
                p_attr_false = 100 * n_attr_false / total
                # Unpack the attribute labels: first = false group, second = true group
                label_false, label_true = attr_labels
                # Value to display: either normalized percentage or raw count
                val_str = f"{cm_display[i, j]*100:.1f}%" if normalize else f"{cm[i, j]}"
                # Combine all info into one string: confusion value + attribute breakdown
                annotations[i, j] = (
                    f"{val_str}\n"
                    f"{label_false}:{p_attr_false:.0f}% / {label_true}:{p_attr_true:.0f}%"
                )

    # Plot heatmap
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(
        cm_display,
        annot=annotations,
        fmt='',
        cmap="Blues",
        xticklabels=["Predicted ID", "Predicted OOD"],
        yticklabels=["True ID", "True OOD"],
        cbar=True
    )
    title = "Row normalized" if normalize else "Raw"
    plt.title(f"{title} Confusion Matrix with Attribute Split")
    plt.xlabel("Prediction")
    plt.ylabel("Ground Truth")
    plt.tight_layout()
    plt.show()