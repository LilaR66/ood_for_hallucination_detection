#!/usr/bin/env python3
"""
============================================================
Cosine Similarity-Based OOD Detection with Multi-Center Support
============================================================

This module implements an out-of-distribution (OOD) detection method based on
cosine similarity between test embeddings and representative in-distribution (ID) centers.
It supports both simple and multimodal modeling of the ID distribution through different
strategies for selecting representative centers.

************************************************************
A high Cosine similarity score => ID data (aligned with ID center)
A low Cosine similarity score  => OOD data (different direction)
************************************************************

Cosine Similarity
-----------------
Cosine similarity measures the alignment (angle) between two vectors, independent of their magnitude.
It is widely used to compare embeddings, as it captures semantic similarity based on direction.

- If the similarity is HIGH: the test sample is considered close to the ID distribution.
- If the similarity is LOW: the sample is likely OOD.

Single-Center Approach:
- Compute the **mean vector** of the ID embeddings (after L2-normalization).
- Calculate cosine similarity between each test point and this mean.

Multi-Center Extension:
- Use **multiple centers** to better model multimodal ID distributions:
  - `k-means`: generates synthetic centroids.
  - `k-medoids`: selects real representative examples.
- Compute cosine similarity between each test point and all centers.
- The **maximum similarity** is retained, assuming that high similarity to any center
  suggests the sample belongs to the ID distribution.

Main Features
-------------
- Computes cosine similarity between test samples and ID reference centers
- Supports single-center (mean) and multi-center (k-means / k-medoids) strategies
- Returns per-sample similarity scores for both ID and OOD test sets
- Embeddings are automatically L2-normalized to ensure cosine validity
"""


import torch
import torch.nn.functional as F
import numpy as np
from typing import Literal
from src.analysis.analysis_utils import compute_kmeans_centroids, compute_kmedoids_centers


def compute_cosine_similarity(
    id_fit_embeddings: torch.Tensor,
    id_test_embeddings: torch.Tensor,
    od_test_embeddings: torch.Tensor,
    seed: int = 44,
    center_type: Literal["mean", "kmeans", "kmedoids"] = "mean",
    k: int = 5,
) -> dict:
    """
    Performs OOD detection by computing cosine similarity between test embeddings
    and representative ID centers (mean, k-means centroids, or medoids).

    Why use different centers?
    ---------------------------
    => "mean": uses the average of all ID embeddings as a single representative center.
    => "kmeans": applies K-Means clustering on ID embeddings and uses the centroids of the k clusters 
        as representative centers.
    => "kmedoids": Uses K-Medoids clustering, which selects k real embeddings (medoids)
        that best represent each cluster.

    Parameters
    ----------
    id_fit_embeddings : torch.Tensor
        Embeddings from in-distribution training examples. Shape: [n_fit_samples, hidden_size]
    id_test_embeddings : torch.Tensor
        Embeddings from in-distribution test examples. Shape: [n_id_test_samples, hidden_size]
    od_test_embeddings : torch.Tensor
        Embeddings from OOD test examples. Shape: [n_ood_test_samples, hidden_size]
    seed : int
        Random state for reproducibility.
    center_type : {'mean', 'kmeans', 'kmedoids'}
        Method to define the reference ID centers:
        - 'mean': uses the average embedding
        - 'kmeans': uses FAISS-based k-means centroids
        - 'kmedoids': uses sklearn_extra KMedoids (real embeddings as centers)
    k : int
        Number of clusters or medoids to use (if method is "kmeans" or "kmedoids").

    Returns
    -------
    dict
        Dictionary with cosine similarity scores:
        {
            'cossim_scores_id': cosine similarity of ID test embeddings to closest center(s),  # Shape: [n_id_test_samples]
            'cossim_scores_ood': cosine similarity of OOD test embeddings to closest center(s) # Shape: [n_ood_test_samples]
        }

    Notes
    -----
    The idea is to compare each test embedding to representative embeddings 
    from the ID training set. A high similarity suggests the input is in-distribution, 
    while a low similarity may indicate OOD behavior.
    """

    # Normalize test embeddings (L2 norm along last dim)
    id_test_norm = F.normalize(id_test_embeddings, p=2, dim=1)
    od_test_norm = F.normalize(od_test_embeddings, p=2, dim=1)

    # Select ID reference centers
    if center_type == 'mean':
        # Compute mean and normalize
        centers = id_fit_embeddings.mean(dim=0, keepdim=True) # Shape: [1, hidden_size]
        centers = F.normalize(centers, p=2, dim=1)
    elif center_type == 'kmeans':
        centers = compute_kmeans_centroids(id_fit_embeddings, k=k, normalize=True, seed=seed)  # Shape: [k, hidden_size]
    elif center_type == 'kmedoids':
        centers = compute_kmedoids_centers(id_fit_embeddings, k=k, normalize=True, seed=seed)  # Shape: [k, hidden_size]
    else:
        raise ValueError(f"Unknown center_type: {center_type}")

    # Convert test embeddings to numpy for dot product
    id_test_np = id_test_norm.cpu().numpy()   # Shape: [n_id_test_samples, hidden_size]
    od_test_np = od_test_norm.cpu().numpy()   # Shape: [n_ood_test_samples, hidden_size]

    # Convert centers to numpy if needed
    if isinstance(centers, torch.Tensor):
        centers = centers.cpu().numpy()

    # Cosine similarity = dot product (since all are L2-normalized)
    sim_id = id_test_np @ centers.T    # Shape: [n_id_test_samples, k]
    sim_ood = od_test_np @ centers.T   # Shape: [n_ood_test_samples, k]

    # Keep max similarity across centers for each test sample
    # We take the maximum cosine similarity because we want to know if a test point 
    # is highly similar to any  center of the ID: If the point is highly similar to 
    # at least one center (high maximum similarity), it can be considered ID.
    max_sim_id = np.max(sim_id, axis=1)
    max_sim_ood = np.max(sim_ood, axis=1)

    return {
        'cossim_scores_id': max_sim_id,  # Shape: [n_id_test_samples]
        'cossim_scores_ood': max_sim_ood # Shape: [n_ood_test_samples]
    }