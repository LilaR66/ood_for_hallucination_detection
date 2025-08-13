#!/usr/bin/env python3
"""
============================================================
Isolation Forest-Based OOD Detection (scikit-learn)
============================================================

This module implements an out-of-distribution (OOD) detection method based on
the Isolation Forest algorithm, which detects anomalies by isolating data points
in the feature space. It is an unsupervised method that builds an ensemble of
random trees to partition the data; outliers are more easily isolated and thus
receive higher anomaly scores.

************************************************************
A high Isolation Forest score =>  ID data (harder to isolate) 
A low Isolation Forest score  => OOD data (easily isolated)

Therefore, to be consistent with the rest of the code, we return the opposite of the 
isolation forest score such that higher scores indicate a OOD sample. 
************************************************************

Isolation Forest for OOD Detection
----------------------------------
Isolation Forest is an unsupervised anomaly detection algorithm that recursively
partitions the data using random splits. The number of splits required to isolate
a sample is lower for outliers and higher for inliers. At test time, each sample
receives an anomaly score based on how easily it is isolated by the ensemble.

- If the anomaly score is HIGH: the test sample is likely out-of-distribution (OOD).
- If the score is LOW: the sample is likely in-distribution (ID).

Main Features
-------------
- Trains an Isolation Forest on ID training embeddings
- Returns anomaly scores for both ID and OOD test sets
- Uses scikit-learn's efficient implementation
- Embeddings are automatically converted to numpy arrays if needed

Returns
-------
- For each test sample, the anomaly score (higher = more likely OOD).
- You can threshold these scores to separate ID and OOD samples.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import torch

def isolation_forest(
    id_fit_embeddings: torch.Tensor,
    id_test_embeddings: torch.Tensor,
    od_test_embeddings: torch.Tensor,
    n_estimators: int = 100,
    contamination: str = "auto",
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs OOD detection by training an Isolation Forest on in-distribution (ID) embeddings
    and scoring both ID and OOD test samples using the anomaly score.

    The anomaly score reflects how easily a sample is isolated by the random trees:
    - Higher scores indicate the sample is likely out-of-distribution (OOD)
    - Lower scores indicate the sample is likely in-distribution (ID)

    Parameters
    ----------
    id_fit_embeddings : torch.Tensor, shape [n_train_samples, n_features]
        Embeddings for in-distribution samples used for training.
    id_test_embeddings : torch.Tensor, shape [n_id_test_samples, n_features]
        Embeddings for in-distribution samples to evaluate.
    od_test_embeddings : torch.Tensor, shape [n_ood_test_samples, n_features]
        Embeddings for out-of-distribution samples to evaluate.
    n_estimators : int, default=100
        Number of trees in the Isolation Forest.
    contamination : 'auto' or float, default='auto'
        Proportion of outliers in the data set. Used for thresholding.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    forest_id_scores : np.ndarray, shape [n_id_test_samples]
        Anomaly scores for the ID test samples (higher = more likely OOD).
    forest_ood_scores : np.ndarray, shape [n_ood_test_samples]
        Anomaly scores for the OOD test samples (higher = more likely OOD).

    Notes
    -----
    - The Isolation Forest is trained only on ID samples.
    - At test time, both ID and OOD samples are scored by the trained model.
    - These scores can be thresholded or used as continuous anomaly/OOD scores.
    - Higher scores correspond to more anomalous (OOD) samples.
    """
    # Convert torch.Tensor to numpy if needed
    if hasattr(id_fit_embeddings, "cpu"):
        X_train = id_fit_embeddings.cpu().numpy()
        X_id_test = id_test_embeddings.cpu().numpy()
        X_ood_test = od_test_embeddings.cpu().numpy()
    else:
        X_train = id_fit_embeddings
        X_id_test = id_test_embeddings
        X_ood_test = od_test_embeddings

    # Apply standard scaler
    scaler = StandardScaler().fit(X_train)  
    X_train = scaler.transform(X_train)
    X_id_test = scaler.transform(X_id_test)
    X_ood_test = scaler.transform(X_ood_test)

    # Train Isolation Forest on ID embeddings
    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=seed
    )
    clf.fit(X_train)

    # The higher, the more abnormal. 
    forest_id_scores = clf.score_samples(X_id_test)
    forest_ood_scores = clf.score_samples(X_ood_test)
    
    # Flip sign to match "higher = more OOD"
    return - forest_id_scores, - forest_ood_scores
