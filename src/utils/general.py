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


def filter_entries(data: dict, column: str, value=1) -> dict:
    """
    Filter a result dictionary to keep only entries where the specified 
    column equals a given value.

    Parameters
    ----------
    data : dict
        Dictionary with keys like 'id', 'gen_answers', 'is_correct', etc.
    column : str
        The key of the column to filter on (e.g., 'is_correct', 'is_unanswerable').
    value : any, optional (default=1)
        The value to keep in the specified column. Only entries where 
        data[column][i] == value are kept.

    Returns
    -------
    dict
        Filtered dictionary with only the selected entries.

    """
    # Compute size before filtering
    original_size = len(data[column])
    # Find indices to keep
    keep_indices = [i for i, val in enumerate(data[column]) if val == value]
    # Filter the dictionary
    filtered_data = {
        key: [values[i] for i in keep_indices]
        for key, values in data.items()
    }
    # Compute and print size after filtering
    filtered_size = len(filtered_data[column])
    # Display information
    print(f"Size before filtering: {original_size}. Size after filtering: {filtered_size}. Filtered {original_size - filtered_size} samples.")
    return filtered_data


def add_unanswerable_flag(data: dict) -> dict:
    """
    Add a boolean column 'is_unanswerable' indicating if the generated answer 
    contains the word 'unanswerable' (case-insensitive, anywhere in the string).

    Parameters
    ----------
    data : dict
        Dictionary with at least the key 'gen_answers', which should be a list of strings.
        Other keys (e.g., 'id') are preserved.

    Returns
    -------
    dict
        A copy of the input dictionary with an additional key 'is_unanswerable',
        which is a list of booleans corresponding to each answer in 'gen_answers'.
        True if 'unanswerable' is present in the answer, False otherwise.

    """
    answers = data.get('gen_answers', [])
    is_unanswerable = [
        'unanswerable' in str(ans).lower() for ans in answers
    ]
    # Return a new dict with the extra column
    new_data = dict(data)
    new_data['is_unanswerable'] = is_unanswerable
    return new_data
