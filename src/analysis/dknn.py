#!/usr/bin/env python3

import torch
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from typing import Tuple, Any
# if you have cuda version 12:
# uv pip install faiss-gpu-cu12
import faiss 


# **Note:** If we use an L2 FAISS index on normalized vectors, 
# the Euclidean distance becomes equivalent to the cosine 
# distance (because the norm of the vectors is 1).
def l2_normalize(feat: np.ndarray) -> np.ndarray:
    """
    L2 normalization of a tensor along the last dimension.

    Args:
        feat (np.ndarray): the tensor to normalize

    Returns:
        np.ndarray: the normalized tensor
    """
    return feat / (np.linalg.norm(feat, ord=2, axis=-1, keepdims=True) + 1e-10)



def fit_to_dataset(fit_embeddings: torch.tensor, use_gpu: bool = True) -> faiss.Index:
    """
    Constructs the FAISS index from ID data.

    Parameters
    ----------
    fit_embeddings : torch.tensor
        ID embeddings, shape (N, D)
    use_gpu : bool
        Whether to use GPU acceleration for FAISS

    Returns
    -------
    faiss.Index
        FAISS index built on the ID embeddings,  ready for k-NN search
    """
    dim = fit_embeddings.shape[1]  # embedding dimension
    fit_embeddings = np.array(fit_embeddings).astype(np.float32) # Convert to array 
    norm_fit_embeddings  = l2_normalize(fit_embeddings) # Normalize embeddings

    cpu_index = faiss.IndexFlatL2(dim) # Create a flat L2 index (exact search, not approximate)

    # If GPU requested, move index to GPU
    if use_gpu:
        res = faiss.StandardGpuResources() # Allocate GPU memory
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index) # Move index to GPU
    else:
        index = cpu_index

    # Add normalized ID embeddings to index
    index.add(norm_fit_embeddings)
    return index



def score_tensor(
    index: faiss.Index,
    inputs: torch.tensor,
    nearest: int = 50,
    batch_size: int = 4
) -> np.ndarray:
    """
    Compute DKNN OOD score for test embeddings.

    Parameters
    ----------
    index : faiss.Index
        FAISS index built from ID data
    inputs : torch.tensor
        Array of test embeddings shape (N, D)
    nearest : int
        Number of nearest neighbors (k)
    batch_size : int
        Batch size for processing

    Returns
    -------
    np.ndarray
        Array of distances to the k-th nearest neighbor
    """

    # Convert to numpy float32 array
    inputs = np.array(inputs, dtype=np.float32)

    # Normalize the test embeddings
    norm_inputs = l2_normalize(inputs)

    # Allocate list to store distances
    all_scores = []

    # Process in mini-batches to avoid memory overflow
    for i in range(0, norm_inputs.shape[0], batch_size):
        batch = norm_inputs[i:i + batch_size]           # Select batch
        distances, _ = index.search(batch, nearest)     # FAISS k-NN search
        kth_dist = distances[:, -1]                     # Take the k-th distance
        all_scores.append(kth_dist)                     # Collect scores
    

    # Concatenate results from all batches
    return np.concatenate(all_scores)


def compute_auroc(
    dknn_scores_id: np.ndarray,
    dknn_scores_ood: np.ndarray,
    plot: bool = False
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Area Under the ROC Curve (auROC) for OOD detection using DKNN scores.

    This function concatenates the DKNN scores from ID and OOD samples,
    assigns binary labels (0 for ID, 1 for OOD), computes the ROC curve and auROC, 
    and optionally plots the ROC curve.

    Parameters
    ----------
    dknn_scores_id : np.ndarray
        DKNN scores for ID samples.
    dknn_scores_ood : np.ndarray
        DKNN scores for OOD samples.
    plot : bool, optional (default=False)
        If True, plot the ROC curve.

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
    """
    # Concatenate scores and create labels
    all_scores = np.concatenate([dknn_scores_id, dknn_scores_ood])
    all_labels = np.concatenate([
        np.zeros_like(dknn_scores_id),  # 0 for ID
        np.ones_like(dknn_scores_ood)   # 1 for OOD
    ])

    # Compute ROC curve and auROC
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    auroc = roc_auc_score(all_labels, all_scores)

    print(f"auROC: {auroc:.4f}")

    # Optionally plot the ROC curve
    if plot:
        plt.figure()
        plt.plot(fpr, tpr, label=f"auROC = {auroc:.4f}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(True)
        plt.show()

    return auroc, fpr, tpr, thresholds
