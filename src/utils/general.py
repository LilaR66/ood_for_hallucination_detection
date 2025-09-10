#!/usr/bin/env python3
"""
=====================================================
General Utilities: Timing, Seeding, and Data Filters
=====================================================

Small helper utilities for experiment hygiene and quick
dataset post-processing: elapsed-time printing, global
random seeding (NumPy / PyTorch / Transformers), simple
row filtering on dicts, and an "unanswerable" flag adder.

Main Features
-------------
- **print_time_elapsed(start, end, label="")**
  Prints a human-readable elapsed time (mm:ss) with an optional label.

- **seed_all(seed=44)**
  Sets seeds for Python `random`, NumPy, PyTorch (CPU/GPU, all devices),
  and (if available) Hugging Face `transformers.set_seed`. Also configures
  `PYTHONHASHSEED`, and sets:
    - `torch.backends.cudnn.deterministic = True`
    - `torch.backends.cudnn.benchmark = False`
  to favor deterministic behavior.

- **filter_entries(data, column, value=1)**
  Returns a new dict keeping only rows where `data[column] == value`
  for every key in `data`. Prints before/after sizes.

- **add_unanswerable_flag(data)**
  Appends a boolean list `is_unanswerable` indicating whether each
  string in `data["gen_answers"]` contains the substring "unanswerable"
  (case-insensitive).
"""

import torch
import numpy as np
import random
import os 

def print_time_elapsed(start, end, label=""):
    elapsed = end - start
    mins, secs = divmod(elapsed, 60)
    print(f"{label}Time elapsed: {int(mins):02d} min {int(secs):02d} sec\n")



def seed_all(seed: int = 44) -> None:
    random.seed(seed)                          # Python random
    np.random.seed(seed)                       # NumPy
    os.environ["PYTHONHASHSEED"] = str(seed)   # Hashed objects (e.g. dictionaries)
    torch.manual_seed(seed)                    # PyTorch CPU
    torch.cuda.manual_seed(seed)               # PyTorch GPU
    torch.cuda.manual_seed_all(seed)           # If multiple GPUs
    torch.backends.cudnn.deterministic = True  # /.\ Can slow inference
    torch.backends.cudnn.benchmark = False     # /.\ Can slow inference
    try:
        from transformers import set_seed
        set_seed(seed) # Deterministic HuggingFace .generate() calls
    except ImportError:
        pass



def filter_entries(data: dict, column: str, value=1) -> dict:
    """
    Filter a result dictionary to keep only entries where the specified 
    column equals a given value i.e. where data[column][i] == value. 
    Works with nested numpy arrays in the 'descriptors' key. 

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

    original_size = len(data[column])
    keep_indices = [i for i, val in enumerate(data[column]) if val == value]
    filtered_size = len(keep_indices)
    print(f"Size before filtering: {original_size}. Size after filtering: {filtered_size}. Filtered {original_size - filtered_size} samples.")

    filtered_data = {}

    # Filter top-level lists
    for k, v in data.items():
        if k == "descriptors":
            continue  # handled separately
        filtered_data[k] = [v[i] for i in keep_indices]

    # Filter descriptors nested structure
    def filter_numpy_arrays(descr):
        filtered_descr = {}
        for layer, layer_dict in descr.items():
            filtered_layer = {}
            for dtype, dtype_dict in layer_dict.items():
                filtered_dtype = {}
                for mode, arr in dtype_dict.items():
                    if isinstance(arr, np.ndarray):
                        filtered_dtype[mode] = arr[keep_indices]
                    else:
                        filtered_dtype[mode] = arr  # fallback if not ndarray
                filtered_layer[dtype] = filtered_dtype
            filtered_descr[layer] = filtered_layer
        return filtered_descr

    filtered_data["descriptors"] = filter_numpy_arrays(data["descriptors"])

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
