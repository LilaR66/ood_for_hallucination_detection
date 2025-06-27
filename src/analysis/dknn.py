#!/usr/bin/env python3
"""
============================================================
DKNN-Based OOD Detection with FAISS and L2-Normalized Embeddings
============================================================

This module implements utilities for out-of-distribution (OOD) detection using
Deep k-Nearest Neighbors (DKNN) over embedding spaces. It relies on FAISS for
fast nearest neighbor search and L2-normalized representations to approximate
cosine similarity.

The DKNN method scores test samples based on their distance to the in-distribution
(ID) training set in feature space. Specifically, the distance to the k-th nearest
neighbor serves as an OOD score.

************************************************************
A high DeepKNN score (distance to k-th NN) => OOD data (far from ID neighbors)
A low DeepKNN score                        => ID data (close to ID neighbors)
************************************************************

Main Features
-------------
- L2-normalizes ID and test embeddings to enable cosine-based comparison
- Constructs a FAISS index for efficient k-NN search over ID embeddings
- Computes DKNN scores as distance to the k-th nearest neighbor
- Supports batched scoring for memory efficiency on large test sets
- Compatible with GPU acceleration via FAISS

Intended Use
------------
This method assumes that test and ID embeddings are from the same feature space
(e.g., from a pretrained encoder), and that proximity in this space reflects
semantic similarity. High DKNN scores suggest out-of-distribution behavior.
"""

import torch
import numpy as np
# if you have cuda version 12:
# uv pip install faiss-gpu-cu12
import faiss 
from src.analysis.analysis_utils import l2_normalize


def fit_to_dataset(fit_embeddings: torch.tensor) -> faiss.Index:
    """
    Constructs the FAISS index from ID data.

    Parameters
    ----------
    fit_embeddings : torch.tensor
        ID embeddings, shape (N, D)

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
    use_gpu = torch.cuda.is_available()
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
        Array of distances to the k-th nearest neighbor of shape (N,)
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

