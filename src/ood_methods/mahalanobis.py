#!/usr/bin/env python3
"""
============================================================
Mahalanobis-Based OOD Detection with Multi-Center Support
============================================================

This module provides an implementation of out-of-distribution (OOD) detection
using Mahalanobis distance between test embeddings and in-distribution (ID) reference points.
It supports both simple and multimodal modeling of the ID distribution through different
strategies for selecting representative centers.

************************************************************
A high Mahalanobis score => OOD data (far from ID distribution)
A low Mahalanobis score  => ID data (close to ID distribution)
************************************************************

Mahalanobis Distance
--------------------
Mahalanobis distance measures how far a point is from the center of a distribution,
taking into account the variance and correlations between features (via the covariance matrix).

- If the distance is LOW: the test sample is considered close to the ID distribution.
- If the distance is HIGH: the sample is likely OOD.

Classic Approach:
- Compute the **mean vector** of the ID embeddings.
- Measure the Mahalanobis distance from each test embedding to this mean.

Extended Approach (Multimodal):
- Use **multiple centers** to better capture a multimodal ID distribution.
  - `k-means`: computes k synthetic centroids.
  - `k-medoids`: selects k real data points as cluster medoids.
- For each test point, the Mahalanobis distance is computed to all centers.
- The **minimum distance** is retained, assuming that closeness to at least one mode indicates ID.

Main Features
-------------
- Computes full covariance matrix over ID embeddings with regularization
- Supports single (mean) or multiple centers (via k-means or k-medoids)
- Handles both in-distribution and OOD test embeddings
- Returns per-sample Mahalanobis scores (lower means more likely ID)
"""
import torch
import numpy as np
from typing import Literal, Tuple
from src.ood_methods.ood_utils import compute_kmeans_centroids, compute_kmedoids_centers
from src.analysis.evaluation import compute_metrics
from sklearn.covariance import LedoitWolf

def compute_mahalanobis_distance(
    id_fit_embeddings: torch.Tensor,
    id_test_embeddings: torch.Tensor,
    od_test_embeddings: torch.Tensor,
    seed: int = 44,
    center_type: Literal["mean", "kmeans", "kmedoids"] = "mean",
    k: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs OOD detection by computing Mahalanobis distance between test embeddings
    and representative ID centers (mean, k-means centroids, or medoids).

    The Mahalanobis distance measures how far a sample lies from the ID training distribution
    while accounting for its covariance structure. Higher scores suggest the sample is far 
    from the ID manifold (likely OOD), while lower scores indicate it lies closer to the 
    expected distribution.
    
    Why use different centers?
    ---------------------------
    => "mean": uses the average of all ID embeddings as a single representative center.
        It is the most common method to apply Mahalanobis distance. 
    => "kmeans": applies K-Means clustering on ID embeddings and uses the centroids of the k clusters 
        as representative centers.
    => "kmedoids": Uses K-Medoids clustering, which selects k real embeddings (medoids)
        that best represent each cluster.

    Parameters
    ----------
    id_fit_embeddings : torch.Tensor
        Embeddings from in-distribution training examples. [n_fit_samples, hidden_size]
    id_test_embeddings : torch.Tensor
        Embeddings from in-distribution test examples. [n_id_test_samples, hidden_size]
    od_test_embeddings : torch.Tensor
        Embeddings from OOD test examples. [n_ood_test_samples, hidden_size]
    seed : int
        Random state for reproducibility.
    center_type : {'mean', 'kmeans', 'kmedoids'}
        Method to define the reference ID centers.
    k : int
        Number of clusters or medoids to use (if method is "kmeans" or "kmedoids").

    Returns
    -------
    maha_scores_id : np.ndarray
        Mahalanobis distance of ID test embeddings to closest center(s).
        Shape: [n_id_test_samples]
     maha_scores_ood : np.ndarray
        Mahalanobis distance of OOD test embeddings to closest center(s).
        Shape: [n_ood_test_samples]
    """
    # Convert to numpy
    id_fit_np  = id_fit_embeddings.detach().cpu().numpy().astype(np.float64)
    id_test_np = id_test_embeddings.detach().cpu().numpy().astype(np.float64)
    od_test_np = od_test_embeddings.detach().cpu().numpy().astype(np.float64)

    # Compute covariance matrix on ID fit set
    # Regularized covariance matrix
    '''
    reg_eps = 1e-6
    cov = np.cov(id_fit_np, rowvar=False)
    cov += reg_eps * np.eye(cov.shape[0]) # Regularize for numerical stability
    inv_cov = np.linalg.inv(cov)
    '''
    # Ledoit-Wolf covariance matrix: 
    lw = LedoitWolf().fit(id_fit_np)   # fit only on ID data
    inv_cov = lw.precision_            # inverse cov stable 

    # Select ID reference centers
    # No normalization/standarization for Mahalanobis distance as the method already 
    # integrates scale and correlations within the covariance matrix. 
    if center_type == 'mean':
        centers = np.mean(id_fit_np, axis=0, keepdims=True)  # Shape: [1, hidden_size]
    elif center_type == 'kmeans':
        centers = compute_kmeans_centroids(id_fit_embeddings, k=k, normalize=False, seed=seed)  # Shape: [k, hidden_size]
    elif center_type == 'kmedoids':
        centers = compute_kmedoids_centers(id_fit_embeddings, k=k, normalize=False, seed=seed)  # Shape: [k, hidden_size]
    else:
        raise ValueError(f"Unknown center_type: {center_type}")

    # Function to compute Mahalanobis distance to all centers
    def maha_dist(X, centers, inv_cov):
        """
        X: [n_samples, hidden_size], centers: [k, hidden_size], inv_cov: [hidden_size, hidden_size]
        Returns [n_samples, k]
        """
        # Differences between each point and each center
        diffs = X[:, None, :] - centers[None, :, :]        # Shape: [n_samples, k, hidden_size]
        # Mahalanobis: sqrt((x-mu)^T @ inv_cov @ (x-mu))
        left = np.einsum('nkd,dd->nkd', diffs, inv_cov)    # Shape: [n_samples, k, hidden_size]
        dists = np.sqrt(np.maximum(np.einsum('nkd,nkd->nk', left, diffs)),0) # Shape: [n_samples, k]
        return dists

    # Compute Mahalanobis distances
    maha_id = maha_dist(id_test_np, centers, inv_cov)   # Shape: [n_id_test_samples, k]
    maha_ood = maha_dist(od_test_np, centers, inv_cov)  # Shape: [n_ood_test_samples, k]

    # Keep minimum distance across centers for each test sample
    # We take the minimum distance because we want to know if a test point is close to any 
    # center of the ID: If the point is close to at least one center (low minimum distance), 
    # it can be considered ID.
    min_maha_id = np.min(maha_id, axis=1)
    min_maha_ood = np.min(maha_ood, axis=1)

    maha_scores_id =  min_maha_id  # Shape: [n_id_test_samples]
    maha_scores_ood = min_maha_ood # Shape: [n_ood_test_samples]

    return maha_scores_id, maha_scores_ood 


