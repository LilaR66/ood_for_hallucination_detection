#!/usr/bin/env python3

import torch
import numpy as np
import random

def print_time_elapsed(start, end, label=""):
    elapsed = end - start
    mins, secs = divmod(elapsed, 60)
    print(f"{label}Time elapsed: {int(mins):02d} min {int(secs):02d} sec\n")


def seed_all(seed: int = 44) -> None:
    random.seed(seed)                          # Python random
    np.random.seed(seed)                       # NumPy
    torch.manual_seed(seed)                    # PyTorch CPU
    torch.cuda.manual_seed(seed)               # PyTorch GPU
    torch.cuda.manual_seed_all(seed)           # If multiple GPUs
    from transformers import set_seed
    set_seed(seed) # Deterministic HuggingFace .generate() calls


def filter_correct_entries(data: dict) -> dict:
    """
    Filter a result dictionary to keep only entries where is_correct == 1.

    Parameters
    ----------
    data : dict
        Dictionary with keys like 'id', 'gen_answers', 'is_correct', etc.

    Returns
    -------
    dict
        Filtered dictionary with only correct entries.
    """
    # Compute size before filtering
    original_size = len(data['is_correct'])
    # Find indices to keep
    keep_indices = [i for i, ok in enumerate(data['is_correct']) if ok == 1]
    # Filter the dictionary
    filtered_data = {
        key: [values[i] for i in keep_indices]
        for key, values in data.items()
    }
    # Compute and print size after filtering
    filtered_size = len(filtered_data['is_correct'])
    # Display information
    print(f"Size before filtering incorrect samples: {original_size}.\nSize after filtering: {filtered_size}. Filtered {original_size - filtered_size} samples.")
    return filtered_data