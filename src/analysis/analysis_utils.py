#!/usr/bin/env python3
"""
============================================================
Clustering Utilities for Embedding-Based OOD Detection
============================================================

This module provides clustering utilities to compute representative centers
from a set of embeddings.

It supports both synthetic centroids (via k-means) and real representative points (via k-medoids),
with optional L2-normalization for cosine-based similarity.

Main Features
-------------
- Computes k-means centroids using FAISS for scalable clustering
- Computes k-medoids using scikit-learn-extra for robust, real-example-based centers
- Optional L2-normalization to enable cosine-based clustering
- Compatible with PyTorch tensors and FAISS GPU acceleration
- Can be used with either Euclidean or cosine-based distances
"""


import torch
import numpy as np
from sklearn_extra.cluster import KMedoids
# if you have cuda version 12:
# uv pip install faiss-gpu-cu12
import faiss


# **Note:** If we use an L2 FAISS index on normalized vectors, 
# the Euclidean distance becomes equivalent to the cosine 
# distance (because the norm of the vectors is 1).
def l2_normalize(feat: np.ndarray) -> np.ndarray:
    """
    L2-normalize a numpy array along the last dimension.

    Parameters
    ----------
    feat : np.ndarray
        The array to normalize.

    Returns
    -------
    np.ndarray
        L2-normalized array.
    """
    return feat / (np.linalg.norm(feat, ord=2, axis=-1, keepdims=True) + 1e-10)



def compute_kmeans_centroids(
    embeddings: torch.Tensor,
    k: int = 5,
    niter: int = 20,
    normalize: bool = False,
    seed: int = 44,
) -> np.ndarray:
    """
    Computes k-means centroids using FAISS on L2-normalized embeddings.

    Parameters
    ----------
    embeddings : torch.Tensor
        Embeddings to cluster. Shape: [n_samples, hidden_size].
    k : int
        Number of centroids/clusters.
    niter : int
        Number of k-means iterations.
    normalize : bool
        If True, applies L2-normalization to embeddings before clustering.
        Use True for cosine-based clustering, False for Euclidean-based.
    seed : int
        Random state for reproducibility.

    Returns
    -------
    np.ndarray
        Array of centroids. Shape: [k, hidden_size].
        L2-normalized if `normalize=True`.

    Notes
    -----
    FAISS performs k-means clustering using Euclidean distance (L2), but if
    the embeddings are normalized, it becomes equivalent to cosine-based clustering.
    """
    # Convert PyTorch tensor to NumPy array and cast to float32 (FAISS requirement)
    X = embeddings.detach().cpu().numpy().astype(np.float32)

    # Optional L2-normalization (ensures cosine similarity = dot product)
    if normalize:
        X = l2_normalize(X) 

    hidden_size = X.shape[1] 

    # Initialize FAISS's clustering object with k clusters
    clus = faiss.Clustering(hidden_size, k)
    clus.niter = niter             # Number of iterations for convergence
    clus.verbose = False           # Set to True to print detailed logs
    clus.seed = seed

    # Create the FAISS index where clustering will be done
    # IndexFlatL2 is a brute-force index using Euclidean (L2) distance
    index = faiss.IndexFlatL2(hidden_size)

    # Move the index to GPU 
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        res = faiss.StandardGpuResources()  # Allocate GPU resources
        index = faiss.index_cpu_to_gpu(res, 0, index)  # Transfer index to GPU

    # Train the FAISS clustering algorithm on the embeddings
    # This performs the actual k-means algorithm and populates the cluster centroids
    clus.train(X, index)

    # Extract the final centroids from the FAISS clustering object
    # clus.centroids is a flat array, so we reshape it to [k, hidden_size]
    centroids = faiss.vector_to_array(clus.centroids).reshape(k, hidden_size)

    # To obtain spherical centroids
    if normalize:
        centroids = l2_normalize(centroids) 

    return centroids



def compute_kmedoids_centers(
    embeddings: torch.Tensor,
    k: int = 5,
    normalize: bool = False, 
    seed : int = 44
) -> np.ndarray:
    """
    Computes k representative medoids from the input embeddings using KMedoids clustering.
    Uses cosine distance for medoid computation.

    Parameters
    ----------
    embeddings : torch.Tensor
        Input embeddings of shape [n_samples, hidden_size].
    k : int
        Number of medoids to extract.
    normalize : bool
        If True, applies L2-normalization to embeddings before clustering.
        Use True for cosine-based clustering, False for Euclidean-based.
    seed : int
        Random state for reproducibility.

    Returns
    -------
    np.ndarray
        Array of selected medoid vectors. Shape: [k, hidden_size]. 
        L2-normalized if `normalize=True`.
    """
    # Convert tensor to NumPy and float32 (for compatibility)
    X = embeddings.detach().cpu().numpy().astype(np.float32)

    # Optional L2-normalization (ensures cosine similarity = dot product)
    if normalize:
        X = l2_normalize(X) 

    # Initialize KMedoids clustering
    kmedoids = KMedoids(n_clusters=k, metric='cosine', init='k-medoids++', random_state=seed)

    # Fit the model
    kmedoids.fit(X)

    # Return the actual medoid vectors 
    medoids = X[kmedoids.medoid_indices_]
    return medoids
