#!/usr/bin/env python3
"""
==================================================
Batch Pickle Management Utilities
==================================================

This module provides utility functions for managing the storage and aggregation of batched 
inference results using Python pickle files.

Key functionalities:
- Create an output directory for storing batch pickle files.
- Save each batch as a separate pickle file (e.g., batch_00001.pkl, batch_00002.pkl, ...).
- Load and merge all batch pickle files from a directory into a single results dictionary.
- Save the merged results as a single pickle file (named after the directory).
- Optionally, delete the directory containing batch files after aggregation.

Typical use case:
-----------------
- During inference or training, save each batch's results as a separate pickle file for fast, 
incremental disk writes.
- After processing, aggregate all batches into a single file for easy downstream analysis.

Data Format:
------------
Each batch pickle file should be a dictionary with the following structure 
(keys may vary depending on your workflow):

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

Example usage:
--------------
    from pickle_batch_utils import (
        save_batch_pickle, load_and_merge_pickles, save_merged_pickle, delete_directory
    )

    # Save a batch
    output_path = "outputs/monjob"
    for batch_idx, batch_data in enumerate(batches):
        save_batch_pickle(batch_data, output_path, batch_idx)

    # 1. Merge all batches and save as a single file
    merged = load_and_merge_pickles(output_path)
    save_merged_pickle(merged, output_path)

    # 2. Optionally delete the batch directory
    delete_directory(output_path, confirm=True)

    # 3. Or to do step 1. and 2. all together:
    merged = merge_batches_and_cleanup(output_path, delete=True, confirm=True) 
"""

import pickle
import os
from typing import Dict, List, Any
import glob
import shutil

def ensure_dir(directory: str):
    """
    Ensure that a directory exists. If it does not exist, create it.

    Parameters
    ----------
    directory : str
        Path to the directory to create.
    """
    os.makedirs(directory, exist_ok=True)



def save_batch_pickle(batch_data: dict, output_dir: str, batch_idx: int):
    """
    Save a single batch of results as a pickle file in the specified directory.

    Parameters
    ----------
    batch_data : dict
        Dictionary containing batch results (see module docstring for structure).
    output_dir : str
        Directory where batch pickle files will be saved.
    batch_idx : int
        Index of the batch (used to name the file, e.g., batch_00001.pkl).
    """
    ensure_dir(output_dir)
    filepath = os.path.join(output_dir, f"batch_{batch_idx:05d}.pkl")
    with open(filepath, "wb") as f:
        pickle.dump(batch_data, f)



def load_and_merge_pickles(directory: str) -> Dict[str, List[Any]]:
    """
    Load and merge all batch pickle files from a directory into a single results dictionary.

    Each batch file should be a dictionary with the same keys.
    The merged dictionary will concatenate the lists for each key across all batches.

    Parameters
    ----------
    directory : str
        Path to the directory containing batch pickle files.

    Returns
    -------
    Dict[str, List[Any]]
        Merged dictionary where each key corresponds to a concatenated list of results from all batches.
    """
    files = sorted(glob.glob(os.path.join(directory, "*.pkl")))
    merged = None
    for file in files:
        with open(file, "rb") as f:
            batch = pickle.load(f)
            if merged is None:
                merged = {k: list(v) for k, v in batch.items()}
            else:
                for k in batch:
                    merged[k].extend(batch[k])
    return merged



def save_merged_pickle(merged_data: Dict[str, List[Any]], directory: str):
    """
    Save the merged results dictionary as a single pickle file.

    The output file will be named after the input directory (e.g., outputs/myjob.pkl).

    Parameters
    ----------
    merged_data : Dict[str, List[Any]]
        The merged results dictionary to save.
    directory : str
        The directory whose name will be used for the output file (e.g., outputs/myjob -> outputs/myjob.pkl).
    """
    output_file = directory.rstrip("/\\") + ".pkl" # output_file = os.path.join(directory + ".pkl")
    with open(output_file, "wb") as f:
        pickle.dump(merged_data, f)



def delete_directory(directory: str, confirm: bool = False):
    """
    Delete a directory and all its contents, if confirmation is given.

    Parameters
    ----------
    directory : str
        Path to the directory to delete.
    confirm : bool or str, optional (default=False)
        If True, the directory will be deleted.
        If False, no action is taken.
        If 'user', prompt the user for confirmation.
    """
    if not os.path.isdir(directory):
        print(f"Directory '{directory}' does not exist. Nothing to delete.")
        return

    do_delete = False
    if confirm == 'user':
        user_input = input(f"Delete directory '{directory}'? [y/n]: ").strip().lower()
        do_delete = user_input == 'y'
    else:
        do_delete = bool(confirm)

    if do_delete:
        shutil.rmtree(directory)
        print(f"Directory '{directory}' deleted.")
    else:
        print(f"Directory '{directory}' NOT deleted.")



def merge_batches_and_cleanup(directory: str, delete: bool = True, confirm: bool = True) -> dict:
    """
    Load and merge all batch pickle files from a directory, save the merged result as a single pickle file,
    and optionally delete the batch directory (using existing utility functions).

    Parameters
    ----------
    directory : str
        Path to the directory containing batch pickle files.
    delete : bool, optional (default=True)
        If True, the directory and its contents will be deleted after merging.
   confirm : bool or str, optional (default=False)
        If True, the directory will be deleted.
        If False, no action is taken.
        If 'user', prompt the user for confirmation.

    Returns
    -------
    dict
        The merged results dictionary.
    """
    merged = load_and_merge_pickles(directory)
    save_merged_pickle(merged, directory)
    if delete:
        delete_directory(directory, confirm=confirm)
    return merged



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


"""
==================================================
Auxiliary functions to compute directory and file size

Exemple of usage
----------------
dir_size = get_dir_size(dir_path)
file_size = get_file_size(file_path)
==================================================
"""

def convert_bytes(num_bytes):
    for unit in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0



def get_dir_size(path: str) -> int:
    """
    Calculate the total size of all files in the given directory and its subdirectories.

    Parameters
    ----------
    path : str
        Path to the directory.

    Returns
    -------
    int
        Total size in bytes of all files contained in the directory tree.
    """

    total_size = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file(follow_symlinks=False):
                total_size += entry.stat(follow_symlinks=False).st_size
            elif entry.is_dir(follow_symlinks=False):
                total_size += get_dir_size(entry.path)

    print(f"Size of Directory {path}: {convert_bytes(total_size)}")
    return total_size



def get_file_size(path: str) -> int:
    """
    Calculate the total size of a file.

    Parameters
    ----------
    path : str
        Path to the file.

    Returns
    -------
    int
        Total size in bytes of the file.
        """
    file_size = os.path.getsize(path)
    print(f"Size of file {path}: {convert_bytes(file_size)}")
    return file_size