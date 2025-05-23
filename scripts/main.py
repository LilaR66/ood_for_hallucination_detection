#!/usr/bin/env python3
"""
============================================================
Main pipeline for extracting and saving embeddings
============================================================

This script provides a modular pipeline for:
- Seeding reproducibility
- Loading a Llama model and tokenizer
- Loading ID dataset and test datasets (that can have OOD and ID samples)
- Extracting and saving token-level activations (embeddings) for ID and test sets

Usage:
------
Run this script directly to execute the full pipeline:
    python3 main.py

Outputs:
--------
- outputs/id_fit_results.pkl
- outputs/id_test_results.pkl
- outputs/od_test_results.pkl
"""
# ====================================
# Find source repository 
# ====================================
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# ====================================
# Import librairies
# ====================================
from typing import Optional, Tuple
import time
from src.utils.general import seed_all, print_time_elapsed
from src.model_loader.llama_loader import load_llama
from src.data_reader.squad_loader import (
    load_id_fit_dataset, 
    load_id_test_dataset, 
    load_od_test_dataset
)
from src.inference.inference_utils import (
    batch_extract_token_activations_with_generation, 
    batch_extract_token_activations, 
    build_prompt, 
    get_layer_output, 
    extract_last_token_activations
)

# ====================================
# Global variables  
# ====================================
SEED = 44
BATCH_SIZE = 16
MODEL_NAME =  "meta-llama/Llama-2-7b-chat-hf"
OUTPUT_DIR = "../results/raw/test/"


# ====================================
# Mains funtions 
# ====================================
def prepare_fit_embeddings(
    model_name: str = MODEL_NAME,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
    output_path: str = OUTPUT_DIR + "id_fit_results.pkl",
    shuffle: bool = False,
    select_slice: Optional[Tuple[int, int]]  = None
) -> None:
    """
    Extract and save embeddings for the in-distribution (ID) training set.

    Parameters
    ----------
    model_name : str
        Name or path of the pretrained Llama model to load.
    batch_size : int
        Number of samples per batch for inference.
    seed : int
        Random seed for reproducibility.
    output_path : str
        Path to save the extracted embeddings as a pickle file.
    shuffle : bool
        If true the dataset if shuffled with seed. 
    select_slice : Optional[Tuple[int, int]]
        (start, end) indices for slicing the dataset.
    """
    # Seed everything
    seed_all(seed)

    # Load model
    model, tokenizer = load_llama(model_name)

    # Load ID dataset
    id_fit_dataset = load_id_fit_dataset()

    # Shuffle if needed 
    if shuffle:
        print("Shuffle dataset")
        id_fit_dataset = id_fit_dataset.shuffle(seed)

    # Select a subset of the dataset if needed
    if (select_slice is not None
        and len(select_slice) == 2
        and 0 <= select_slice[0] < select_slice[1] <= len(id_fit_dataset)
        ):
        print(f"Select dataset slice: {select_slice}")
        id_fit_dataset = id_fit_dataset.slice(idx_start=select_slice[0], idx_end=select_slice[1])

    # Retrieve ID embeddings and save results
    print("\nStart retrieving ID embeddings...")
    t0 = time.time()
    batch_extract_token_activations_with_generation(
        model=model,
        tokenizer=tokenizer,
        dataset=id_fit_dataset,
        batch_size=batch_size,
        idx_start_sample=0,
        max_samples=len(id_fit_dataset),
        output_path=output_path,
        build_prompt_fn=build_prompt,
        get_layer_output_fn=get_layer_output,
        layer_idx=-1,
        extract_token_activations_fn=extract_last_token_activations,
        offset=-5
    )
    t1 = time.time()
    print("...end!")
    print_time_elapsed(t0, t1, label="ID embeddings: ")

    # Free memory
    del id_fit_dataset


def prepare_test_embeddings(
    model_name: str = MODEL_NAME,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
    id_output_path: str = OUTPUT_DIR + "id_test_results.pkl",
    od_output_path: str = OUTPUT_DIR + "od_test_results.pkl",
    shuffle: bool = False,
    select_slice: Optional[Tuple[int, int]]  = None
) -> None:
    """
    Extract and save embeddings for the in-distribution (ID) and 
    out-of-distribution (OOD) test sets.

    Parameters
    ----------
    model_name : str
        Name or path of the pretrained Llama model to load.
    batch_size : int
        Number of samples per batch for inference.
    seed : int
        Random seed for reproducibility.
    id_output_path : str
        Path to save the ID test embeddings as a pickle file.
    od_output_path : str
        Path to save the OOD test embeddings as a pickle file.
    shuffle : bool
        If true the dataset if shuffled with seed. 
    select_slice : Optional[Tuple[int, int]]
        (start, end) indices for slicing the dataset.
    """
    # Seed everything
    seed_all(seed)

    # Load model
    model, tokenizer = load_llama(model_name)

    # Load  ID test dataset
    id_test_dataset = load_id_test_dataset()

    # Load a OOD test dataset
    od_test_dataset = load_od_test_dataset()

    # Shuffle if needed 
    if shuffle:
        print("Shuffle dataset")
        id_test_dataset = id_test_dataset.shuffle(seed)
        od_test_dataset = od_test_dataset.shuffle(seed)

    # Select a subset of the dataset if needed
    if (select_slice is not None
        and len(select_slice) == 2
        and 0 <= select_slice[0] < select_slice[1] <= len(id_test_dataset)
        and select_slice[1] <= len(od_test_dataset)
        ):
        print(f"Select dataset slice: {select_slice}")
        id_test_dataset = id_test_dataset.slice(idx_start=select_slice[0], idx_end=select_slice[1])
        od_test_dataset = od_test_dataset.slice(idx_start=select_slice[0], idx_end=select_slice[1])


    # Retrieve test embeddings and save results 
    print("\nStart retrieving test impossible embeddings...")
    t2 = time.time()
    batch_extract_token_activations(
        model=model,
        tokenizer=tokenizer,
        dataset=od_test_dataset,
        batch_size=batch_size,
        idx_start_sample=0,
        max_samples=len(od_test_dataset),
        save_to_pkl=True,
        output_path=od_output_path,
        build_prompt_fn=build_prompt,
        get_layer_output_fn=get_layer_output,
        layer_idx=-1,
        extract_token_activations_fn=extract_last_token_activations,
        offset=-5,
    )
    t3 = time.time()
    print("...end!")
    print_time_elapsed(t2, t3, label="Impossible test embeddings: ")

    # Extract ID test embeddings
    print("\nStart retrieving test possible embeddings...")
    t4 = time.time()
    batch_extract_token_activations(
        model=model,
        tokenizer=tokenizer,
        dataset=id_test_dataset,
        batch_size=batch_size,
        idx_start_sample=0,
        max_samples=len(id_test_dataset),
        save_to_pkl=True,
        output_path=id_output_path,
        build_prompt_fn=build_prompt,
        get_layer_output_fn=get_layer_output,
        layer_idx=-1,
        extract_token_activations_fn=extract_last_token_activations,
        offset=-5,
    )
    t5 = time.time()
    print("end!")
    print_time_elapsed(t4, t5, label="Possible test embeddings: ")

    # Free memory
    del od_test_dataset
    del id_test_dataset


def main() -> None:
    """
    Main entry point for the embedding extraction pipeline.
    """
    prepare_fit_embeddings(shuffle=True, select_slice=(0,100))
    prepare_test_embeddings(shuffle=True, select_slice=(0,100))


if __name__ == "__main__":
    main()

