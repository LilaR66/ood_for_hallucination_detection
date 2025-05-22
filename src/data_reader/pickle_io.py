#!/usr/bin/env python3
"""
==================================================
Utility functions for managing inference results via Pickle
==================================================

This module provides helper functions to handle reading, writing,
and appending data to `.pkl` files for storing inference results.

These functions are primarily used to:
- Initialize a structured result dictionary if no pickle exists
- Incrementally store batched outputs (e.g., generated answers, activations, scores)
- Load full batches of results for later evaluation or analysis

Data Format
-----------
The result dictionary managed by this module uses the following structure:
{
    "id": List[str],
    "original_indices": List[int],
    "gen_answers": List[str],
    "ground_truths": List[str],
    "activations": List[Tensor or np.ndarray],
    "is_correct": List[int],
    "sbert_scores": List[float or None],
    "rouge_scores": List[float]
}

Usage
-------
from pickle_io import append_to_pickle, load_or_create_pickle

results = load_or_create_pickle("outputs/results.pkl")
append_to_pickle("outputs/results.pkl", new_batch)
"""

import pickle
import os
from typing import Dict, List, Any


def load_or_create_pickle(filepath: str) -> dict:
    """
    Load an existing pickle file or return an empty dictionary if it doesn't exist.

    Parameters
        filepath : str
            Path to the pickle file.

    Returns
        dict
            Dictionary loaded from the pickle or an empty one.
    """
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)
    else:
        return {
            "id": [],
            "original_indices": [],
            "gen_answers": [],
            "ground_truths": [],
            "activations": [],
            "is_correct": [],
            "sbert_scores": [],
            "rouge_scores": []
        }


def append_to_pickle(filepath: str, new_data: dict):
    """
    Append new data to an existing pickle file, or create it if it doesn't exist.

    Parameters
        filepath : str
            Path to the pickle file.
        new_data : dict
            New data to append. Must have same keys as existing pickle or empty.
    """
    existing_data = load_or_create_pickle(filepath)

    for key in new_data:
        existing_data[key].extend(new_data[key])

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(existing_data, f)


def load_pickle_batches(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a list of dictionaries from a pickle file.
    Each element typically represents one batch of results 
    (e.g., prompts, answers, activations, scores, etc.).

    Parameters
    ----------
    file_path : str
        Path to the pickle file.

    Returns
    -------
    List[Dict[str, Any]]
        List where each item is a dictionary with keys
    """
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            n_samples = len(data[list(data.keys())[0]])
        print(f"Loaded {n_samples} samples from: {file_path}")
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []
