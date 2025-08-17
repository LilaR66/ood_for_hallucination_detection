#!/usr/bin/env python3
"""
============================================================
Unified OOD Scoring Frontend (Dispatch API)
============================================================

This module provides a single, unified entry point to compute
out-of-distribution (OOD) scores using multiple backends
(DKNN, cosine, Mahalanobis, one-class SVM, isolation forest),
with consistent shape conventions and a common return format.

It dispatches to the concrete implementations defined in
`src.ood_methods.*` and handles input validation, tensor conversion,
and the "raw_scores" bypass for already-scalar descriptors.

Score Convention
----------------
All backends follow the same convention:
    **higher score = more OOD (more anomalous)**

Supported Backends
------------------
- "raw_scores"   : directly return 1D descriptors as OOD scores
                   (useful when you already computed scalar scores
                    in "attention", "logit", or a 1D "hidden" score).
- "dknn"         : distance to the k-th nearest ID neighbor (FAISS).
- "cosine"       : (negative) max cosine similarity to ID centers.
- "mahalanobis"  : min Mahalanobis distance to ID center(s).
- "ocsvm"        : (negative) one-class SVM decision function.
- "isoforest"    : (negative) isolation-forest score_samples().


Main Features
-------------
- Single function `compute_ood_scores(...)` for all methods.
- Strict input checks with clear error messages.
- No silent broadcasting/reshaping (makes errors obvious).
- Returns NumPy arrays for both ID and OOD scores.

Note
----
- This module does *not* normalize/scale features; normalization
  is handled by the specific backends where relevant (e.g. cosine).
"""

import numpy as np
from typing import Dict, Tuple, Any, Literal
import torch

from src.ood_methods.dknn import compute_dknn_scores
from src.ood_methods.cosine import compute_cosine_similarity
from src.ood_methods.mahalanobis import compute_mahalanobis_distance
from src.ood_methods.ocsvm import compute_ocsvm_score
from src.ood_methods.isoforest import isolation_forest


# Registry of available OOD methods
_OOD_REGISTRY: Dict[str, Any] = {
    "dknn": compute_dknn_scores,
    "cosine": compute_cosine_similarity,
    "mahalanobis": compute_mahalanobis_distance,
    "ocsvm": compute_ocsvm_score,
    "isoforest": isolation_forest,
}


def compute_ood_scores(
    method: Literal["dknn", "cosine", "mahalanobis", "ocsvm", "isoforest", "raw_scores"],
    id_fit_descriptors: np.ndarray,
    id_test_descriptors: np.ndarray,
    od_test_descriptors: np.ndarray,
    **method_kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute OOD scores for ID and OOD test samples using a selected method.

    Parameters
    ----------
    method : str
        One of:
          - "raw_scores":       directly use given 1D descriptors as OOD scores
                                (for "attention", "logits" or "token_svd_score" in "hidden" 
                                already computed as scalar scores).
          - "dknn":             Deep k-NN distance to k-th neighbor (higher => OOD).
          - "cosine":           (sign flipped) Cosine similarity (higher => OOD).
          - "mahalanobis":      Mahalanobis distance to ID distribution (higher => OOD).
          - "ocsvm":            (sign flipped) One-class SVM scores (higher => OOD).
          - "isoforest":        (sign flipped) Isolation forest anomaly scores (higher => OOD).
    id_fit_descriptors : np.ndarray[float]
        ID fit descriptors.
        Shape:
          - 2D (n_samples, feat_dim) for "hidden"
          - 1D (n_samples,) for "attention", "logits" or "token_svd_score" in "hidden"
            will be reshaped to (n_samples, 1) if method != "raw_scores".
    id_test_descriptors : np.ndarray[float]
        ID test descriptors (same shape convention as above).
    od_test_descriptors : np.ndarray[float]
        OOD test descriptors (same shape convention as above).
    **method_kwargs : Any
        Extra keyword arguments forwarded to the backend method
        (e.g., k=5 for DKNN, kernel/nu/gamma for OCSVM).

    Returns
    -------
    scores_id : np.ndarray, shape (n_id_test,), float
        OOD scores for ID test samples (larger means more OOD).
    scores_ood : np.ndarray, shape (n_ood_test,), float
        OOD scores for OOD test samples (larger means more OOD).

    Raises
    ------
    ValueError
        If `method='raw_scores'` is used with non-1D descriptors.
    """
    if method not in ("dknn", "cosine", "mahalanobis", "ocsvm", "isoforest", "raw_scores"):
        raise ValueError(f"method must be one of: 'dknn','cosine','mahalanobis','ocsvm','isoforest','raw_scores'; but '{method}' was chosen.")
    
    # "raw_scores": directly use 1D descriptors as scores
    if method == "raw_scores":
        if id_test_descriptors.ndim != 1 or od_test_descriptors.ndim != 1:
            raise ValueError("method='raw_scores' requires 1D descriptors (precomputed scalar desriptors).")
        return id_test_descriptors.astype(float), od_test_descriptors.astype(float)

    
    # We do not allow to apply OOD methods on 1D descriptors:
    # Raise errors if descriptors are 1D 
    for name, arr in [("id_fit_descriptors", id_fit_descriptors),
                      ("id_test_descriptors", id_test_descriptors),
                      ("od_test_descriptors", od_test_descriptors)]:
        if np.asarray(arr).ndim == 1:
            raise ValueError(
                f"{name} is 1D but method='{method}' requires 2D features. "
                f"Use method='raw_scores' or reshape to (n,1) explicitly."
            )
    id_fit_t  = torch.as_tensor(id_fit_descriptors)  # (n,d)
    id_test_t = torch.as_tensor(id_test_descriptors) # (n,d)
    od_test_t = torch.as_tensor(od_test_descriptors) # (n,d)

    if method not in _OOD_REGISTRY:
        raise ValueError(f"Unknown OOD method: {method}")
    fn = _OOD_REGISTRY[method]
    
    kwargs = dict(method_kwargs)

    # Apply OOD detection method
    scores_id, scores_ood = fn(
        id_fit_embeddings=id_fit_t,
        id_test_embeddings=id_test_t,
        od_test_embeddings=od_test_t,
        **kwargs
    )

    return np.asarray(scores_id, dtype=float), np.asarray(scores_ood, dtype=float)