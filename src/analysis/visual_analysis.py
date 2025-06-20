#!/usr/bin/env python3

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
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
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute the Area Under the ROC Curve (auROC) for OOD detection using scores.

    This function:
    - Concatenates the scores from ID and OOD samples,
    - Assigns binary labels (0 for ID, 1 for OOD)
    - Computes the ROC curve and auROC and optionally plots the ROC curve
    - Determines the optimal threshold using Youden's J statistic. 

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

    print(f"auROC: {auroc:.4f}")
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

    return auroc, fpr, tpr, thresholds, youden_threshold



def plot_score_distributions(
    scores_id: np.ndarray,
    scores_ood: np.ndarray,
    bins: int = 50,
    xlabel: str = "Distance to k-th NN",
    title: str = "Distribution of DKNN Scores",
    figsize: Tuple[int, int] = (10, 6),
    save_path: str = None,
) -> plt.Figure:
    """
    Plot overlapping histograms of ID and OOD scores to compare their distributions.

    Parameters
    ----------
    scores_id : np.ndarray
        Scores for in-distribution samples
    scores_ood : np.ndarray
        Scores for out-of-distribution samples
    bins : int, optional (default=50)
        Number of histogram bins
    xlabel : str, optional
        Label for x-axis
    title : str, optional
        Plot title
    figsize : Tuple[int, int], optional
        Figure size
    save_path : str, optional
        If provided, the plot will be saved to this path (e.g. "plot.png")

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    plt.figure(figsize=figsize)
    
    # Plot histograms with transparency
    plt.hist(scores_id, bins=bins, alpha=0.5, label="ID", density=True, color="blue")
    plt.hist(scores_ood, bins=bins, alpha=0.5, label="OOD", density=True, color="red")
    
    # Add labels and styling
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.7)
    
    # Add text box with summary stats
    stats_text = (f"ID mean: {scores_id.mean():.2f} ± {scores_id.std():.2f}\n"
                  f"OOD mean: {scores_ood.mean():.2f} ± {scores_ood.std():.2f}")
    plt.gca().text(
        0.0, 0.95, stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", alpha=0.2, facecolor="w")
    )
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()
    #return plt.gcf() # return figure if needed



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
