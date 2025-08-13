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

Therefore, to be consistent with the rest of the code, we return the opposite of the 
cosine distance such that higher scores indicate a OOD sample. 
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
from typing import Literal, Tuple, Optional
from src.ood_methods.ood_utils import compute_kmeans_centroids, compute_kmedoids_centers
from tqdm import tqdm

def compute_cosine_similarity(
    id_fit_embeddings: torch.Tensor,
    id_test_embeddings: torch.Tensor,
    od_test_embeddings: torch.Tensor,
    seed: int = 44,
    center_type: Literal["mean", "all", "kmeans", "kmedoids"] = "mean",
    k: int = 5,
    batch_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs OOD detection by computing cosine similarity between test embeddings
    and representative ID centers (mean, k-means centroids, or medoids).

    The cosine similarity score represents the angular closeness between a sample and 
    the center of the ID training distribution. Higher scores indicate stronger similarity 
    (likely ID), while lower scores suggest dissimilarity and potential OOD status.
    Therefore, to be consistent with the rest of the code, we return the opposite of the 
    cosine distance such that higher scores indicate a OOD sample.  

    Why use different centers?
    ---------------------------
    => "mean": uses the average of all ID embeddings as a single representative center.
    => "all": uses all ID embeddings as the ID reprensentation
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
    batch_size : int
        If center_type='all' and batch_size is not None, process samples per batch. 

    Returns
    -------
    cossim_scores_id : np.ndarray
        (Oposite of) cosine similarity of ID test embeddings to closest center(s).
        Shape: [n_id_test_samples]
    cossim_scores_ood : np.ndarray
        (Oposite of) cosine similarity of OOD test embeddings to closest center(s).
        Shape: [n_ood_test_samples]

    Notes
    -----
    The idea is to compare each test embedding to representative embeddings 
    from the ID training set. A high similarity suggests the input is in-distribution, 
    while a low similarity may indicate OOD behavior.
    """
    device = id_fit_embeddings.device

    # Normalize test embeddings (L2 norm along last dim)
    # (ensures cosine similarity = dot product)
    id_test_norm = F.normalize(id_test_embeddings, p=2, dim=1)
    od_test_norm = F.normalize(od_test_embeddings, p=2, dim=1)

    # Select ID reference centers
    if center_type == 'mean':
        # Compute mean and normalize
        id_fit_norm = F.normalize(id_fit_embeddings, p=2, dim=1) # normalize each embedding
        centers = id_fit_norm.mean(dim=0, keepdim=True) # Shape: [1, hidden_size]
        centers = F.normalize(centers, p=2, dim=1) # normalize centers to obtain spherical centroids
    elif center_type == 'kmeans':
        centers = compute_kmeans_centroids(id_fit_embeddings, k=k, normalize=True, seed=seed)  # Shape: [k, hidden_size]
    elif center_type == 'kmedoids':
        centers = compute_kmedoids_centers(id_fit_embeddings, k=k, normalize=True, seed=seed)  # Shape: [k, hidden_size]
    elif center_type == 'all':
        # Keep all normalized embeddings as centers
        centers = F.normalize(id_fit_embeddings, p=2, dim=1)  
    else:
        raise ValueError(f"Unknown center_type: {center_type}")

    # Convert centers to torch tensor
    if isinstance(centers, np.ndarray):
            centers = torch.from_numpy(centers).to(device).float()

    def batch_max_sim(test_embeddings, centers, batch_size):
        n = test_embeddings.shape[0]
        max_sims = []
        for i in tqdm(range(0, n, batch_size)):
            sim = test_embeddings[i:i+batch_size] @ centers.T
            max_sim = torch.max(sim, dim=1).values
            max_sims.append(max_sim)
        return torch.cat(max_sims, dim=0)

    if batch_size is not None and center_type == "all":
        # Batch computation to avoid OOM
        max_sim_id = batch_max_sim(id_test_norm, centers, batch_size)
        max_sim_ood = batch_max_sim(od_test_norm, centers, batch_size)
    else:
        # Cosine similarity = dot product (since all are L2-normalized)
        sim_id = id_test_norm @ centers.T   # Shape: [n_id_test_samples, k]
        sim_ood = od_test_norm @ centers.T  # Shape: [n_ood_test_samples, k]
        # Keep max similarity across centers for each test sample
        # We take the maximum cosine similarity because we want to know if a test point 
        # is highly similar to any center of the ID: If the point is highly similar to 
        # at least one center (high maximum similarity), it can be considered ID.
        max_sim_id = torch.max(sim_id, dim=1).values
        max_sim_ood = torch.max(sim_ood, dim=1).values

    cossim_scores_id = max_sim_id.cpu().numpy()    # Shape: [n_id_test_samples]
    cossim_scores_ood = max_sim_ood.cpu().numpy()  # Shape: [n_ood_test_samples]

    return - cossim_scores_id, - cossim_scores_ood
