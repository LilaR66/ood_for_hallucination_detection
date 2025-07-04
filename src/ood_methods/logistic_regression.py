import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from typing import Tuple

def train_logistic_regression_on_embeddings(
    id_test_embeddings: torch.Tensor,
    od_test_embeddings: torch.Tensor,
    random_state: int = 42,
    test_size: float = 0.2,
) -> Tuple[LogisticRegression, np.ndarray, np.ndarray, np.ndarray]:
    """
    Train and evaluate a logistic regression classifier to distinguish 
    in-distribution (ID) from out-of-distribution (OOD) embeddings.

    Parameters
    ----------
    id_test_embeddings : torch.Tensor
        Embeddings of in-distribution samples. Shape: [n_id_samples, hidden_size]
    od_test_embeddings : torch.Tensor
        Embeddings of out-of-distribution samples. Shape: [n_ood_samples, hidden_size]
    random_state : int, optional (default=42)
        Random seed for reproducibility.
    test_size : float, optional (default=0.2)
        Proportion of the dataset to include in the test split (e.g., 0.2 = 20%).

    Returns
    -------
    clf : sklearn.linear_model.LogisticRegression
        The trained logistic regression classifier.
    y_test : np.ndarray
        Ground-truth labels for the test split (0 = ID, 1 = OOD).
    y_pred : np.ndarray
        Predicted labels for the test split (0 = predicted ID, 1 = predicted OOD).
    idx_test : np.ndarray
        Indexes of samples used for testing
    """

    # 1. Concatenate embeddings and create labels
    X = torch.cat([id_test_embeddings, od_test_embeddings], dim=0).cpu().numpy()
    y = np.concatenate([
        np.zeros(id_test_embeddings.shape[0]),  # 0 = ID
        np.ones(od_test_embeddings.shape[0])    # 1 = OOD
    ])

    # 2. Stratified train/test split to preserve class proportions
    indices = np.arange(len(X))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices, test_size=test_size, random_state=random_state, stratify=y
    )

    # 3. Train logistic regression
    clf = LogisticRegression(max_iter=1000, random_state=random_state)
    clf.fit(X_train, y_train)

    # 4. Predict and evaluate on test set
    y_pred = clf.predict(X_test)

    return clf, y_test, y_pred, idx_test