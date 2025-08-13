#!/usr/bin/env python3
"""
============================================================
One-Class SVM-Based OOD Detection (ThunderSVM/Scikit-learn)
============================================================

This module implements an out-of-distribution (OOD) detection method based on
the One-Class Support Vector Machine (One-Class SVM) algorithm, which learns the
boundary of the in-distribution (ID) data in the embedding space. It supports
both linear and cosine similarity-based kernels, and can leverage GPU acceleration
via ThunderSVM or run on CPU via scikit-learn.

************************************************************
A high OCSVM decision score => ID data (inside learned boundary)
A low OCSVM decision score  => OOD data (outside learned boundary)

Therefore, to be consistent with the rest of the code, we return the opposite of the 
ocsvm score such that higher scores indicate a OOD sample. 
************************************************************

One-Class SVM for OOD Detection
-------------------------------
One-Class SVM is an unsupervised anomaly detection algorithm that learns to
separate the majority of the training data (ID samples) from the origin in
feature space. At test time, it assigns a score to each sample indicating how
well it fits the learned ID distribution.

- If the decision function score is HIGH: the test sample is considered in-distribution.
- If the score is LOW: the sample is likely out-of-distribution (OOD).

Linear vs Cosine Kernel:
- 'linear': The SVM is trained directly on the raw embeddings.
- 'cosine': All embeddings are L2-normalized, so the linear kernel effectively computes cosine similarity.

Main Features
-------------
- Trains a One-Class SVM on ID training embeddings
- Supports both linear and cosine (L2-normalized) kernels
- Returns decision function scores for both ID and OOD test sets
- Compatible with ThunderSVM (GPU) and scikit-learn (CPU)
- Embeddings are automatically L2-normalized if 'cosine' kernel is selected

Returns
-------
- For each test sample, the decision function score (higher = more likely ID).
- You can threshold these scores to separate ID and OOD samples.
"""
import torch
from sklearn.preprocessing import StandardScaler, normalize
from typing import Literal, Tuple
import numpy as np
try:
    from thundersvm import OneClassSVM
    _BACKEND = "thundersvm"
except ImportError:
    from sklearn.svm import OneClassSVM
    _BACKEND = "sklearn"



def compute_ocsvm_score(
    id_fit_embeddings: torch.Tensor,
    id_test_embeddings: torch.Tensor,
    od_test_embeddings: torch.Tensor,
    kernel: Literal["linear", "cosine", "rbf"] = "linear",
    nu: float = 0.1,
    gamma: float | str = "scale"
)-> Tuple[np.ndarray, np.ndarray] :
    """
    Performs OOD detection by training a One-Class SVM on in-distribution (ID) embeddings
    and scoring both ID and OOD test samples using the SVM decision function.
    Note: One-Class SVM  with 'rbf' kernel is equivalent to SVDD with 'rbf' kernel. 

    The decision function score reflects how well a sample fits the learned ID distribution:
    - Higher scores indicate the sample is likely in-distribution (ID)
    - Lower scores indicate potential out-of-distribution (OOD) status
    Therefore, to be consistent with the rest of the code, we return the opposite of the 
    ocsvm score such that higher scores indicate a OOD sample. 

    Parameters
    ----------
    id_fit_embeddings : torch.Tensor, shape [n_train_samples, n_features]
        Embeddings for in-distribution samples used for training.
    id_test_embeddings : torch.Tensor, shape [n_id_test_samples, n_features]
        Embeddings for in-distribution samples to evaluate.
    od_test_embeddings : torch.Tensor, shape [n_ood_test_samples, n_features]
        Embeddings for out-of-distribution samples to evaluate.
    kernel : str, default='linear'
        Kernel type for the SVM:
        - 'linear': Use standardized embeddings
        - 'cosine': L2-normalize all embeddings, so the linear kernel computes cosine similarity.
        - 'rbf': Use standardized embeddings
    nu : float, default=0.1
        An upper bound on the fraction of training errors and a lower bound of the fraction
        of support vectors (0 < nu <= 1). Controls the sensitivity of the decision boundary.
    gamma : float or 'scale', default='scale'
        Kernel coefficient for 'rbf'. If 'scale', uses 1 / (n_features * X.var()).

    Returns
    -------
    ocsvm_id_scores : np.ndarray, shape [n_id_test_samples]
        (Oposite of) the decision function scores for the ID test samples (higher = more likely OOD).
    ocsvm_ood_scores : np.ndarray, shape [n_ood_test_samples]
        (Oposite of) the decision function scores for the OOD test samples (lower = more likely ID).

    Notes
    -----
    - The One-Class SVM is trained only on ID samples.
    - At test time, both ID and OOD samples are scored by the trained model.
    - For GPU acceleration, use ThunderSVM; for CPU, use scikit-learn.
    """
    # Ensure numpy arrays on CPU
    X_train    = id_fit_embeddings.detach().cpu().numpy()
    X_id_test  = id_test_embeddings.detach().cpu().numpy()
    X_od_test = od_test_embeddings.detach().cpu().numpy()

    if kernel == 'linear':
        # Standardize for linear kernel as it is sensitive to scale
        scaler = StandardScaler().fit(X_train)   # fit ID only
        X_train_proc = scaler.transform(X_train)
        X_id_test_proc = scaler.transform(X_id_test)
        X_od_test_proc = scaler.transform(X_od_test)
        model = OneClassSVM(kernel='linear', nu=nu) 
    elif kernel == 'cosine':
        # L2-normalize for cosine similarity with linear kernel
        # (so that cosine similarity = dot product)
        X_train_proc = normalize(X_train, norm='l2')
        X_id_test_proc = normalize(X_id_test, norm='l2')
        X_od_test_proc = normalize(X_od_test, norm='l2')
        model = OneClassSVM(kernel='linear', nu=nu)
    elif kernel == 'rbf':
        # Standardize for gaussian kernel as it is sensitive to scale
        scaler = StandardScaler().fit(X_train)  # fit ID only
        X_train_proc = scaler.transform(X_train)
        X_id_test_proc = scaler.transform(X_id_test)
        X_od_test_proc = scaler.transform(X_od_test)
        # Computation of gamma depending on backend
        if _BACKEND == "thundersvm":
            if gamma == "scale":
                gamma_val = 1.0 / (X_train_proc.shape[1] * X_train_proc.var())
            elif gamma == "auto":
                gamma_val = 1.0 / X_train_proc.shape[1]
            else:
                gamma_val = gamma
            model = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma_val)
        else:
            model = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
    else:
        raise ValueError("Kernel must be 'linear', 'cosine' or 'rbf'.")

    model.fit(X_train_proc)

    # Decision function gives anomaly scores: higher = more in-distribution, lower = more outlier
    ocsvm_id_scores = model.decision_function(X_id_test_proc)
    ocsvm_ood_scores = model.decision_function(X_od_test_proc)

    ocsvm_id_scores = ocsvm_id_scores.reshape(-1)
    ocsvm_ood_scores = ocsvm_ood_scores.reshape(-1)

    return - ocsvm_id_scores, - ocsvm_ood_scores
