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
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import umap.umap_ as umap



def compute_auroc(
    scores_id: np.ndarray,
    scores_ood: np.ndarray,
    plot: bool = False,
    save_path: str = None,
) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute the Area Under the ROC Curve (auROC), FPR95, and AUC-PR for OOD 
    detection using scores.

    This function:
    - Concatenates the scores from ID and OOD samples,
    - Assigns binary labels (0 for ID, 1 for OOD)
    - Computes auROC, FPR95, and AUC-PR and optionally plots the ROC curve 
    - Determines the optimal threshold using Youden's J statistic. 

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
        Thresholds used to compute FPR and TPR.
    youden_threshold : float
        Optimal threshold according to Youden's J statistic.
        Samples with score <= threshold are classified as ID,
        samples with score > threshold are classified as OOD.

    Notes
    -----
    Youden's J statistic looks for the optimal point on the ROC curve, where :
    - The rate of true positives is as high as possible
    - The false positive rate is as low as possible
    This is the threshold that best separates the classes without  
    compromising either side too much.
    """
    # Concatenate scores and create labels
    all_scores = np.concatenate([scores_id, scores_ood])
    all_labels = np.concatenate([
        np.zeros_like(scores_id),  # 0 for ID
        np.ones_like(scores_ood)   # 1 for OOD
    ])

    # Compute ROC curve and auROC
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    auroc = roc_auc_score(all_labels, all_scores)

    # Compute Youden's J statistic
    j_scores = tpr - fpr
    best_index = np.argmax(j_scores)
    youden_threshold = thresholds[best_index]

    # Compute FPR95
    # Find the threshold where TPR is closest to 0.95
    tpr_95_idx = np.argmin(np.abs(tpr - 0.95))
    fpr95 = fpr[tpr_95_idx]

    # Compute Precision-Recall curve and AUC-PR
    precision, recall, _ = precision_recall_curve(all_labels, all_scores)
    auc_pr = average_precision_score(all_labels, all_scores)

    # Display results
    print(f"auROC: {auroc:.4f}")
    print(f"FPR95: {fpr95:.4f}")
    print(f"AUC-PR: {auc_pr:.4f}")
    print(f"Optimal Threshold (Youden's J statistic): {youden_threshold:.4f}")

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

    return auroc, fpr95, auc_pr, fpr, tpr, thresholds, youden_threshold



def plot_score_distributions(
    scores_id: np.ndarray,
    scores_ood: np.ndarray,
    xlabel: str = "Score",
    title: str = "Distribution of ID and OOD scores",
    figsize: Tuple[int, int] = (10, 6),
    save_path: str = None,
    bandwidth: float = 1.0
) -> plt.Figure:
    """
    Plot smoothed KDE distributions for ID and OOD scores with vertical mean lines and color-coded legend.

    Parameters
    ----------
    scores_id : np.ndarray
        Scores for in-distribution samples
    scores_ood : np.ndarray
        Scores for out-of-distribution samples
    xlabel : str
        X-axis label
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size
    save_path : str
        If set, saves the figure to this path
    bandwidth : float
        Bandwidth adjustment for KDE smoothing

    Returns
    -------
    plt.Figure
        The Matplotlib figure
    """
    plt.figure(figsize=figsize)

    # Define colors
    color_id = "#1f77b4"    # blue
    color_ood = "#ff7f0e"   # orange

    # KDE curves
    sns.kdeplot(scores_id, label="ID", fill=True, linewidth=2, bw_adjust=bandwidth, color=color_id)
    sns.kdeplot(scores_ood, label="OOD", fill=True, linewidth=2, bw_adjust=bandwidth, color=color_ood)

    # Compute statistics
    id_mean, id_std = scores_id.mean(), scores_id.std()
    ood_mean, ood_std = scores_ood.mean(), scores_ood.std()

    # Vertical mean lines
    plt.axvline(id_mean, color=color_id, linestyle="--", linewidth=1.5, label=f"ID mean: {id_mean:.2f} ± {id_std:.2f}")
    plt.axvline(ood_mean, color=color_ood, linestyle="--", linewidth=1.5, label=f"OOD mean: {ood_mean:.2f} ± {ood_std:.2f}")

    # Labels and formatting
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()



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
    f1 = f1_score(y_true, y_pred)
    # Compute precision
    precision = precision_score(y_true, y_pred)
    # Compute recall 
    recall = recall_score(y_true, y_pred)

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



def plot_dim_reduction_3d_embeddings(
    id_test_embeddings,
    od_test_embeddings,
    labels_id=None,
    labels_ood=None,
    pca_config=None,
    tsne_config=None,
    umap_config=None,
    marker_size=3,
    random_state=42,
):
    """
    Visualize embeddings in 3D using PCA, t-SNE, and/or UMAP, with optional 
    chaining (PCA->t-SNE/UMAP).

    This function supports two usage cases:
    ----------
    1) labels_id and labels_ood are provided as arrays encoding classification results with 4 classes:
       - 0: ID well classified   (true ID, score > threshold)
       - 1: OOD well classified  (true OOD, score < threshold)
       - 2: OOD not detected     (true OOD, score > threshold)
       - 3: ID not detected      (true ID, score < threshold)
    2) labels_id and labels_ood are None, in which case simple binary labels are created:
       - 0 for all id_test_embeddings (ID)
       - 1 for all od_test_embeddings (OOD)

    Supports 3 dimension reduction methods:
    ----------
    - PCA reduces dimensionality by projecting the data onto orthogonal axes that capture 
        the most variance, revealing the global structure of the dataset.
    - t-SNE maps high-dimensional data into a lower-dimensional space by preserving local 
        neighborhood similarities, making clusters more visually separable.
    - UMAP reduces dimensionality by preserving both local and some global data structure, 
        offering fast and meaningful visualizations of complex datasets.

    Parameters
    ----------
    id_test_embeddings : torch.Tensor
        Embeddings of ID test samples, Shape (N_id_samples, hidden_size)
    od_test_embeddings : torch.Tensor
        Embeddings of OOD test samples, Shape (N_ood_samples, hidden_size)
    labels_id : np.ndarray or None, optional
        Array of labels for ID test samples, same length as id_test_embeddings.
        Shape (N_id_samples)
    labels_ood : np.ndarray or None, optional
        Array of labels for OOD test samples, same length as od_test_embeddings.
        Shape (N_od_samples)
    pca_config : dict or None
        Parameters for PCA (e.g. {'n_components': 3}). If None, PCA is skipped unless used as preprocessing.
       If provided alone, the function computes and plots a 3D PCA projection 
       (n_components is forced to 3 for visualization).
    tsne_config : dict or None
        Parameters for t-SNE (e.g. {'perplexity': 30}). If None, t-SNE is skipped. 
        t-SNE always projects data into 3D (n_components is forced to 3), and a 3D t-SNE plot is always shown.
    umap_config : dict or None
        Parameters for UMAP (e.g. {'n_neighbors': 15, 'min_dist': 0.1}). If None, UMAP is skipped. 
        UMAP always projects data into 3D (n_components is forced to 3), and a 3D UMAP plot is always shown.
    marker_size : int, default=3
        Size of the points in the plot and legend.
    random_state : int, default=44
        Random seed for reproducibility.

    Returns
    -------
    figs : dict
        Dictionary of plotly.graph_objects.Figure objects for each method computed.
        Keys: 'pca', 'tsne', 'umap' (present only if the method was computed).

    Notes
    -----
    - Embeddings are always standardized before any reduction.
    - If both PCA and t-SNE/UMAP configs are provided, PCA is used as preprocessing step
        to accelerate t-SNE/UMAP and improve quality.
    - If only t-SNE or UMAP is provided, they are applied directly on the standardized embeddings.
    - If labels are not provided, a binary classification scheme is assumed (ID=0, OOD=1).
    - The function displays the interactive plot inline and returns the Plotly figure object.

    Example Usage
    -------------
        --- Configs ---
    pca_config =  {'n_components': 3}
    tsne_config = {'perplexity': 30}
    umap_config = {'n_neighbors': 15, 'min_dist': 0.1}
        --- PCA only ---
    plot_dim_reduction_3d_embeddings(id_test_embeddings, od_test_embeddings, 
    labels_id, labels_ood, pca_config=pca_config)
        --- UMAP only ---
    plot_dim_reduction_3d_embeddings(id_test_embeddings, od_test_embeddings, 
    labels_id, labels_ood, umap_config=umap_config)
        --- PCA + t-SNE ---
    plot_dim_reduction_3d_embeddings(id_test_embeddings, od_test_embeddings,labels_id,
    labels_ood, pca_config=pca_config, tsne_config=tsne_config)
        --- The 3 methods ---
    plot_dim_reduction_3d_embeddings(id_test_embeddings, od_test_embeddings, labels_id,
    labels_ood, pca_config=pca_config, tsne_config=tsne_config, umap_config=umap_config)
    """

    # ==============================
    # Prepare embeddings and labels 
    # ==============================
    # Convert embeddings to numpy arrays
    id_emb_np = id_test_embeddings.cpu().numpy()
    od_emb_np = od_test_embeddings.cpu().numpy()

    # Labels and concatenation
    if labels_id is not None and labels_ood is not None:
        all_embeddings = np.concatenate([id_emb_np, od_emb_np], axis=0)
        all_labels = np.concatenate([labels_id, labels_ood])
    else:
        all_embeddings = np.concatenate([id_emb_np, od_emb_np], axis=0)
        all_labels = np.concatenate([
            np.zeros(len(id_emb_np), dtype=int),  # label 0 = ID
            np.ones(len(od_emb_np), dtype=int)    # label 1 = OOD
        ])

    # ==============================
    # Standardize
    # ==============================
    scaler = StandardScaler()
    all_embeddings_scaled = scaler.fit_transform(all_embeddings)

    # ==============================
    # Prepare colors and names
    # ==============================
    unique_labels = np.unique(all_labels)
    colors_4 = ['royalblue', 'crimson', 'pink', 'seagreen']
    names_4 = [
        'Answerable ID - (identified)',      # 0: ID well classified
        'Unanswerable OOD - (identified)',   # 1: OOD well classified
        'Unanswerable OOD - (unidentified)', # 2: OOD not detected
        'Answerable ID - (unidentified)'     # 3: ID not detected
    ]
    colors_2 = ['royalblue', 'crimson']
    names_2 = ['ID test samples', 'OOD test samples']
    if set(unique_labels).issubset({0,1}):
        colors = colors_2
        names = names_2
    else:
        colors = colors_4
        names = names_4

    figs = {}

    # ==============================
    # PCA
    # ==============================
    if pca_config is not None:
        pca_params = dict(pca_config)
        if 'random_state' not in pca_params:
            pca_params['random_state'] = random_state
        pca = PCA(**pca_params)
        data_pca = pca.fit_transform(all_embeddings_scaled)
        
        # Plot if n_components == 3
        if pca_params.get('n_components', 3) == 3:
            traces = []
            for i, name in zip(unique_labels, [names[int(i)] for i in unique_labels]):
                idx = all_labels == i
                traces.append(go.Scatter3d(
                    x=data_pca[idx, 0],
                    y=data_pca[idx, 1],
                    z=data_pca[idx, 2],
                    mode='markers',
                    marker=dict(size=marker_size, color=colors[int(i)]),
                    name=name
                ))
            fig_pca = go.Figure(data=traces)
            fig_pca.update_layout(
                scene=dict(
                    xaxis_title='PC1',
                    yaxis_title='PC2',
                    zaxis_title='PC3',
                    aspectmode='cube'
                ),
                legend=dict(font=dict(size=12)),
                title='3D PCA of test embeddings'
            )
            fig_pca.show()
            figs['pca'] = fig_pca
    else:
        data_pca = None

    # ==============================
    # t-SNE
    # ==============================
    if tsne_config is not None:
        tsne_params = dict(tsne_config)
        tsne_params['n_components'] = 3  # Always force 3D for t-SNE
        if 'random_state' not in tsne_params:
            tsne_params['random_state'] = random_state
        if 'init' not in tsne_params:
            # a PCA init in t-SNE is not the same as doing a complete PCA before launching t-SNE.
            tsne_params['init'] = 'pca' 
        # If PCA is present, use its output, else use standardized embeddings
        tsne_input = data_pca if (data_pca is not None and data_pca.shape[1] >= tsne_params.get('n_components', 3)) else all_embeddings_scaled
        tsne = TSNE(**tsne_params)
        data_tsne = tsne.fit_transform(tsne_input)
        traces = []
        for i, name in zip(unique_labels, [names[int(i)] for i in unique_labels]):
            idx = all_labels == i
            traces.append(go.Scatter3d(
                x=data_tsne[idx, 0],
                y=data_tsne[idx, 1],
                z=data_tsne[idx, 2],
                mode='markers',
                marker=dict(size=marker_size, color=colors[int(i)]),
                name=name
            ))
        fig_tsne = go.Figure(data=traces)
        fig_tsne.update_layout(
            scene=dict(
                xaxis_title='t-SNE 1',
                yaxis_title='t-SNE 2',
                zaxis_title='t-SNE 3',
                aspectmode='cube'
            ),
            legend=dict(font=dict(size=12)),
            title='3D t-SNE of test embeddings'
        )
        fig_tsne.show()
        figs['tsne'] = fig_tsne

    # ==============================
    # UMAP
    # ==============================
    if umap_config is not None:
        umap_params = dict(umap_config)
        umap_params['n_components'] = 3  # Always force 3D for UMAP
        if 'random_state' not in umap_params:
            umap_params['random_state'] = random_state
        if 'init' not in umap_params:
            umap_params['init'] = 'spectral'
        # If PCA is present, use its output, else use standardized embeddings
        umap_input = data_pca if (data_pca is not None and data_pca.shape[1] >= umap_params.get('n_components', 3)) else all_embeddings_scaled
        umap_model = umap.UMAP(**umap_params)
        data_umap = umap_model.fit_transform(umap_input)
        traces = []
        for i, name in zip(unique_labels, [names[int(i)] for i in unique_labels]):
            idx = all_labels == i
            traces.append(go.Scatter3d(
                x=data_umap[idx, 0],
                y=data_umap[idx, 1],
                z=data_umap[idx, 2],
                mode='markers',
                marker=dict(size=marker_size, color=colors[int(i)]),
                name=name
            ))
        fig_umap = go.Figure(data=traces)
        fig_umap.update_layout(
            scene=dict(
                xaxis_title='UMAP 1',
                yaxis_title='UMAP 2',
                zaxis_title='UMAP 3',
                aspectmode='cube'
            ),
            legend=dict(font=dict(size=12)),
            title='3D UMAP of test embeddings'
        )
        fig_umap.show()
        figs['umap'] = fig_umap

    return figs
