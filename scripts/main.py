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
import gc
import torch
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# ====================================
# Import librairies
# ====================================
from typing import Optional, Tuple, Callable, Literal, List
import time
import pickle
from datasets import  Dataset
from functools import partial
from src.utils.general import seed_all, print_time_elapsed
from src.model_loader.llama_loader import load_llama
from src.data_reader.squad_loader import (
    load_id_fit_dataset, 
    load_id_test_dataset, 
    load_od_test_dataset
)
from src.inference.run_extraction import (
    run_filter_generated_answers_by_similarity, 
    run_prompt_descriptor_extraction, 
    run_prompt_and_generation_descriptor_extraction,
)
from src.inference.generation_utils import build_prompt

from src.data_reader.pickle_io import merge_batches_and_cleanup, load_pickle_batches
from src.utils.general import filter_entries

# ====================================
#  Define funtions 
# ====================================
def clear_cache():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()


def prepare_fit_dataset(
    seed: int = 44,
    shuffle: bool = False,
    select_slice: Optional[Tuple[int, int]]  = None,
) -> Dataset:
    """
    Extract and save embeddings for the in-distribution (ID) training set.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    shuffle : bool
        If true the dataset if shuffled with seed. 
    select_slice : Optional[Tuple[int, int]]
        (start, end) indices for slicing the dataset. 

    Returns
    -------
    id_fit_dataset : Dataset
        Hugging Face dataset containing ID fit data. 
    """
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
    
    return id_fit_dataset


def prepare_test_dataset(
    seed: int = 44,
    shuffle: bool = False,
    select_slice: Optional[Tuple[int, int]]  = None,
) -> Dataset:
    """
    Extract and save embeddings for the in-distribution (ID) training set.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    shuffle : bool
        If true the dataset if shuffled with seed. 
    select_slice : Optional[Tuple[int, int]]
        (start, end) indices for slicing the dataset.
    custom_dataset_path : str
        Path to a custom dataset to extract token embeddings. 

    Returns
    -------
    id_test_dataset : Dataset
        Hugging Face dataset containing ID test data. 
    od_test_dataset : Dataset
        Hugging Face dataset containing OOD test data. 
    """
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
    
    return id_test_dataset, od_test_dataset



def run_filter_generated_answers_by_similarity_pipeline(
    model_name: str,
    seed: int,
    output_path: str,
    save_dataset_path: str,
    shuffle: bool = False,
    select_slice: Optional[Tuple[int, int]]  = None,
    batch_size: int = 16,
    build_prompt_fn: Callable = None,
) -> None:
    """
    Generate model answers for a dataset, compare them to ground-truth, and filter to keep only correct responses.

    This function runs batched inference to generate answers with a decoder-only language model,
    compares generated responses to ground-truth answers using semantic similarity metrics,
    and saves only the dataset entries where the generated answer is considered correct.

    Parameters
    ----------
    model_name : str
        Name or path of the pretrained Llama model to load.
    seed : int
        Random seed for reproducibility.
    output_path : str
        Path to the directory where extracted answers will be saved as individual pickle batch files.
    save_dataset_path : str
        Path to save the new created dataset containing only correct answers.
    shuffle : bool
        If true the dataset if shuffled with seed. 
    select_slice : Optional[Tuple[int, int]]
        (start, end) indices for slicing the dataset.
    batch_size : int
        Number of samples per batch for inference.
    build_prompt_fn : callable
        Function to build a prompt from context and question.
    """   

    # Seed everything
    # -----------------------------------
    seed_all(seed)

    # Load model
    # -----------------------------------
    model, tokenizer = load_llama(model_name)

    # Load ID dataset
    # -----------------------------------
    id_fit_dataset = prepare_fit_dataset(seed, shuffle, select_slice)

    # Retrieve ID generated responses and compare them to ground-truth 
    # -----------------------------------
    print("\nStart generating ID answers and comparing them to ground-truth...")
    t0 = time.time()
    run_filter_generated_answers_by_similarity(
        model=model,
        tokenizer=tokenizer,
        dataset=id_fit_dataset,
        batch_size=batch_size,
        idx_start_sample=0,
        max_samples=len(id_fit_dataset),
        output_path=output_path,
        build_prompt_fn=build_prompt_fn
    )
    t1 = time.time()
    print("...end!")
    print_time_elapsed(t0, t1, label="ID answers: ")

    # Merge all batches, save as a single file and delete batch directory
    merged = merge_batches_and_cleanup(directory=output_path, delete=True, confirm=True) 

    # Load ID responses and only keep correct entries 
    # -----------------------------------
    # Load extracted answers 
    id_fit_answers = load_pickle_batches(output_path + ".pkl")
    # Only keep rows where the generated responses are similar to the ground-truth answers
    ids_correct_answers = filter_entries(id_fit_answers, column='is_correct', value=1)["id"]
    # Create a new dataset contaning only the correct answers 
    id_fit_correct_dataset =  id_fit_dataset.filter_by_column('id', ids_correct_answers)
    # Save the new correct dataset for later use
    id_fit_correct_dataset.save(save_dataset_path) 
    # Free memory
    del model, tokenizer, merged, id_fit_answers, ids_correct_answers, id_fit_correct_dataset


def retrieve_fit_inputs_descriptor_pipeline(
    model_name: str,
    seed: int,
    output_path: str,
    custom_dataset_path: str = None,
    shuffle: bool = False,
    select_slice: Optional[Tuple[int, int]]  = None,
    batch_size: int = 16,
    build_prompt_fn: Callable = None,
    layers: List[int] = [-1],  
    hidden_agg: List[str] = ["avg_emb", "last_emb", "max_emb", "first_gen_emb", "hidden_score", "feat_var_emb"],
    attn_agg: List[str] = ["attn_score"],
    logit_agg: List[str] = ["perplexity_score", "logit_entropy_score", "window_logit_entropy_score"],
    logit_config: dict = {"top_k": 50, "window_size": 1, "stride": 1},
    start_offset: int = 0,
    end_offset: int = 0
) -> None:
    """
    Extract and save activations/attention/logits descriptors for the in-distribution (ID) training set.

    Parameters
    ----------
    model_name : str
        Name or path of the pretrained Llama model to load.
    seed : int
        Random seed for reproducibility.
    output_path : str
        Path to the directory where extracted answers will be saved as individual pickle batch files.
    custom_dataset_path : str
        Path to a custom dataset to extract token embeddings. 
        If None, the default dataset is loaded with prepare_fit_dataset() 
    shuffle : bool
        If true the dataset if shuffled with seed. 
    select_slice : Optional[Tuple[int, int]]
        (start, end) indices for slicing the dataset.
    batch_size : int
        Number of samples per batch for inference.
    build_prompt_fn : callable
        Function to build a prompt from context and question.
    layers : List[int]
        List of indices of the transformer layers to extract activations from (default: [-1] for last layer).
    hidden_agg : List[str], optional
        List of aggregation modes to compute on token activations. Possible modes include:
            "avg_emb", "last_emb", "max_emb", "first_gen_emb", "hidden_score", "feat_var_emb".
    attn_agg : List[str], optional
        List of attention-based descriptors to compute. Supported: "attn_eig_prod".
    logit_agg : List[str], optional
        List of logit-based descriptors to compute. Supported:
            "perplexity_score", "logit_entropy_score", "window_logit_entropy_score".
    logit_config : dict, optional
        Configuration dictionary for logit-based descriptor functions, with keys such as:
            - "top_k": int, number of top logits considered (default 50)
            - "window_size": int, window size for windowed entropy (default 1)
            - "stride": int, stride for windowed entropy (default 1)
    start_offset : int
        Number of tokens to skip from the beginning of the sequence when extracting token 
        activations with extract_token_activations_fn.
    end_offset : int
        The number of tokens to skip from the end of the sequence when extracting token
        activations with extract_token_activations_fn.
    """  

    # Seed everything
    # -----------------------------------
    seed_all(seed)

    # Load model
    # -----------------------------------
    model, tokenizer = load_llama(model_name)

    # Load ID dataset
    # -----------------------------------
    if custom_dataset_path:
        with open(custom_dataset_path, "rb") as f:
            id_fit_dataset = pickle.load(f)
        print(f"loaded dataset from {custom_dataset_path}")
    else:
        id_fit_dataset = prepare_fit_dataset(seed, shuffle, select_slice)

    # Retrieve ID descriptors and save results
    # -----------------------------------
    print("\nStart retrieving ID fit descriptors from inputs...")
    t0 = time.time()
    run_prompt_descriptor_extraction(
        model=model,
        tokenizer=tokenizer,
        dataset=id_fit_dataset,
        batch_size=batch_size,
        idx_start_sample=0,
        max_samples=len(id_fit_dataset),
        save_to_pkl=True,
        output_path=output_path,
        build_prompt_fn=build_prompt_fn,
        layers=layers,  
        hidden_agg=hidden_agg,
        attn_agg=attn_agg,
        logit_agg=logit_agg,
        logit_config=logit_config,
        start_offset=start_offset,
        end_offset=end_offset
    )
    t1 = time.time()
    print("...end!")
    print_time_elapsed(t0, t1, label="ID descriptors: ")

    # Merge all batches, save as a single file and delete batch directory
    merged = merge_batches_and_cleanup(directory=output_path, delete=True, confirm=True) 

    # Free memory
    del id_fit_dataset, model, tokenizer, merged, 


def retrieve_test_inputs_descriptor_pipeline(
    model_name: str,
    seed: int,
    id_output_path: str,
    od_output_path: str,
    shuffle: bool = False,
    select_slice: Optional[Tuple[int, int]]  = None,
    batch_size: int = 16,
    build_prompt_fn: Callable = None,
    layers: List[int] = [-1],  
    hidden_agg: List[str] = ["avg_emb", "last_emb", "max_emb", "first_gen_emb", "hidden_score", "feat_var_emb"],
    attn_agg: List[str] = ["attn_score"],
    logit_agg: List[str] = ["perplexity_score", "logit_entropy_score", "window_logit_entropy_score"],
    logit_config: dict = {"top_k": 50, "window_size": 1, "stride": 1},
    start_offset: int = 0,
    end_offset: int = 0
) -> None:
    """
    Extract and save activations/attention/logits descriptors for the in-distribution (ID)  
    and out-of-distribution (OOD) test sets.

    Parameters
    ----------
    model_name : str
        Name or path of the pretrained Llama model to load.
    id_output_path : str
        Path to the directory where ID test embeddings will be saved as individual pickle batch files.
    od_output_path : str
        Path to the directory where OOD test embeddings will be saved as individual pickle batch files.
    seed : int
        Random seed for reproducibility.
    shuffle : bool
        If true the dataset if shuffled with seed. 
    select_slice : Optional[Tuple[int, int]]
        (start, end) indices for slicing the dataset.
    batch_size : int
        Number of samples per batch for inference.
    build_prompt_fn : callable
        Function to build a prompt from context and question.
    layers : List[int]
        List of indices of the transformer layers to extract activations from (default: [-1] for last layer).
    hidden_agg : List[str], optional
        List of aggregation modes to compute on token activations. Possible modes include:
            "avg_emb", "last_emb", "max_emb", "first_gen_emb", "hidden_score", "feat_var_emb".
    attn_agg : List[str], optional
        List of attention-based descriptors to compute. Supported: "attn_eig_prod".
    logit_agg : List[str], optional
        List of logit-based descriptors to compute. Supported:
            "perplexity_score", "logit_entropy_score", "window_logit_entropy_score".
    logit_config : dict, optional
        Configuration dictionary for logit-based descriptors functions, with keys such as:
            - "top_k": int, number of top logits considered (default 50)
            - "window_size": int, window size for windowed entropy (default 1)
            - "stride": int, stride for windowed entropy (default 1)
    start_offset : int
        Number of tokens to skip from the beginning of the sequence when extracting token 
        activations with extract_token_activations_fn.
    end_offset : int
        The number of tokens to skip from the end of the sequence when extracting token
        activations with extract_token_activations_fn.
    """  
    # Seed everything
    # -----------------------------------
    seed_all(seed)

    # Load model
    # -----------------------------------
    model, tokenizer = load_llama(model_name)

    # Load ID and OOD test dataset
    # -----------------------------------
    id_test_dataset, od_test_dataset = prepare_test_dataset(seed, shuffle, select_slice)

    # Retrieve test descriptors and save results 
    # -----------------------------------
    # Extract OOD test descriptors
    print("\nStart retrieving test impossible descriptors from inputs...")
    t2 = time.time()
    run_prompt_descriptor_extraction(
        model=model,
        tokenizer=tokenizer,
        dataset=od_test_dataset,
        batch_size=batch_size,
        idx_start_sample=0,
        max_samples=len(od_test_dataset),
        save_to_pkl=True,
        output_path=od_output_path,
        build_prompt_fn=build_prompt_fn,
        layers=layers,  
        hidden_agg=hidden_agg,
        attn_agg=attn_agg,
        logit_agg=logit_agg,
        logit_config=logit_config,
        start_offset=start_offset,
        end_offset=end_offset
    )
    t3 = time.time()
    print("...end!")
    print_time_elapsed(t2, t3, label="Impossible test descriptors: ")

    # Extract ID test descriptors
    print("\nStart retrieving test possible descriptors from inputs...")
    t4 = time.time()
    run_prompt_descriptor_extraction(
        model=model,
        tokenizer=tokenizer,
        dataset=id_test_dataset,
        batch_size=batch_size,
        idx_start_sample=0,
        max_samples=len(id_test_dataset),
        save_to_pkl=True,
        output_path=id_output_path,
        build_prompt_fn=build_prompt_fn,
        layers=layers,  
        hidden_agg=hidden_agg,
        attn_agg=attn_agg,
        logit_agg=logit_agg,
        logit_config=logit_config,
        start_offset=start_offset,
        end_offset=end_offset
    )
    t5 = time.time()
    print("end!")
    print_time_elapsed(t4, t5, label="Possible test descriptors: ")

    # Merge all batches, save as a single file and delete batch directory
    od_merged = merge_batches_and_cleanup(directory=od_output_path, delete=True, confirm=True) 
    id_merged = merge_batches_and_cleanup(directory=id_output_path, delete=True, confirm=True) 

    # Free memory
    del od_test_dataset, id_test_dataset, model, tokenizer, od_merged, id_merged


def retrieve_fit_answers_descriptor_pipeline(
    model_name: str,
    seed: int,
    output_path: str,
    custom_dataset_path: str = None,
    shuffle: bool = False,
    select_slice: Optional[Tuple[int, int]]  = None,
    batch_size: int = 16,
    build_prompt_fn: Callable = None,
    layers: List[int] = [-1],  
    hidden_agg: List[str] = ["avg_emb", "last_emb", "max_emb", "first_gen_emb", "hidden_score", "feat_var_emb"],
    attn_agg: List[str] = ["attn_score"],
    logit_agg: List[str] = ["perplexity_score", "logit_entropy_score", "window_logit_entropy_score"],
    logit_config: dict = {"top_k": 50, "window_size": 1, "stride": 1},
    activation_source: Literal["prompt", "generation", "promptGeneration"] = "generation",
    start_offset: int = 0,
    end_offset: int = 0,
) -> None:
    """
    Extract and save activations/attention/logits descriptors for the in-distribution (ID) 
    training set of the generated answers.

    Parameters
    ----------
    model_name : str
        Name or path of the pretrained Llama model to load.
    seed : int
        Random seed for reproducibility.
    output_path : str
        Path to the directory where extracted answers will be saved as individual pickle batch files
    custom_dataset_path : str
        Path to a custom dataset to extract token embeddings. 
        If None, the default dataset is loaded.
    shuffle : bool
        If true, the dataset will be shuffled with the given seed.
    select_slice : Optional[Tuple[int, int]]
        (start, end) indices for slicing the dataset.
    batch_size : int
        Number of samples per batch for inference.
    build_prompt_fn : callable
        Function to build a prompt from context and question.
    layers : List[int]
        List of indices of the transformer layers to extract activations from (default: [-1] for last layer).
    hidden_agg : List[str], optional
        List of aggregation modes to compute on token activations. Possible modes include:
            "avg_emb", "last_emb", "max_emb", "first_gen_emb", "hidden_score", "feat_var_emb".
    attn_agg : List[str], optional
        List of attention-based descriptors to compute. Supported: "attn_eig_prod".
    logit_agg : List[str], optional
        List of logit-based descriptors to compute. Supported:
            "perplexity_score", "logit_entropy_score", "window_logit_entropy_score".
    logit_config : dict, optional
        Configuration dictionary for logit-based descriptors functions, with keys such as:
            - "top_k": int, number of top logits considered (default 50)
            - "window_size": int, window size for windowed entropy (default 1)
            - "stride": int, stride for windowed entropy (default 1)
    activation_source : {"prompt", "generation", "promptGeneration"}
        Which part of the sequence to extract activations from:
        - "prompt": only from the prompt
        - "generation": only from the generated answer
        - "promptGeneration": prompt and generation answer both concatenated
    start_offset : int
        Number of tokens to skip from the beginning of the sequence when extracting token activations.
    end_offset : int
        The number of tokens to skip from the end of the sequence when extracting token activations.
    """

    # Seed everything
    # -----------------------------------
    seed_all(seed)

    # Load model
    # -----------------------------------
    model, tokenizer = load_llama(model_name)

    # Load ID dataset
    # -----------------------------------
    if custom_dataset_path:
        with open(custom_dataset_path, "rb") as f:
            id_fit_dataset = pickle.load(f)
        print(f"loaded dataset from {custom_dataset_path}")
    else:
        id_fit_dataset = prepare_fit_dataset(seed, shuffle, select_slice)

    # Retrieve ID descriptors and save results
    # -----------------------------------
    print(f"\nStart retrieving ID fit descriptors from {activation_source}...")
    t0 = time.time()
    run_prompt_and_generation_descriptor_extraction(
        model=model,
        tokenizer=tokenizer,
        dataset=id_fit_dataset,
        batch_size=batch_size,
        idx_start_sample=0,
        max_samples=len(id_fit_dataset),
        save_to_pkl=True,
        output_path=output_path,
        build_prompt_fn=build_prompt_fn,
        layers=layers,  
        activation_source=activation_source,
        hidden_agg=hidden_agg,
        attn_agg=attn_agg,
        logit_agg=logit_agg,
        logit_config=logit_config,
        start_offset=start_offset,
        end_offset=end_offset
    )
    t1 = time.time()
    print("...end!")
    print_time_elapsed(t0, t1, label="ID descriptors: ")

    # Merge all batches, save as a single file and delete batch directory
    merged = merge_batches_and_cleanup(directory=output_path, delete=True, confirm=True) 

    # Free memory
    del id_fit_dataset, model, tokenizer, merged



def retrieve_test_answers_descriptor_pipeline(
    model_name: str,
    seed: int,
    id_output_path: str,
    od_output_path: str,
    shuffle: bool = False,
    select_slice: Optional[Tuple[int, int]]  = None,
    batch_size: int = 16,
    build_prompt_fn: Callable = None,
    layers: List[int] = [-1],  
    hidden_agg: List[str] = ["avg_emb", "last_emb", "max_emb", "first_gen_emb", "hidden_score", "feat_var_emb"],
    attn_agg: List[str] = ["attn_score"],
    logit_agg: List[str] = ["perplexity_score", "logit_entropy_score", "window_logit_entropy_score"],
    logit_config: dict = {"top_k": 50, "window_size": 1, "stride": 1},
    activation_source: Literal["prompt", "generation", "promptGeneration"] = "generation",
    start_offset: int = 0,
    end_offset: int = 0,
) -> None:
    """
    Extract and save activations/attention/logits descriptors for the in-distribution (ID) 
    and out-of-distribution (OOD) test sets of the generated answers.

    Parameters
    ----------
    model_name : str
        Name or path of the pretrained Llama model to load.
    id_output_path : str
        Path to the directory where ID test embeddings will be saved as individual pickle batch files.
    od_output_path : str
        Path to the directory where OOD test embeddings will be saved as individual pickle batch files.
    seed : int
        Random seed for reproducibility.
    shuffle : bool
        If true the dataset if shuffled with seed. 
    select_slice : Optional[Tuple[int, int]]
        (start, end) indices for slicing the dataset.
    batch_size : int
        Number of samples per batch for inference.
    build_prompt_fn : callable
        Function to build a prompt from context and question.
    layers : List[int]
        List of indices of the transformer layers to extract activations from (default: [-1] for last layer).
    hidden_agg : List[str], optional
        List of aggregation modes to compute on token activations. Possible modes include:
            "avg_emb", "last_emb", "max_emb", "first_gen_emb", "hidden_score", "feat_var_emb".
    attn_agg : List[str], optional
        List of attention-based descriptors to compute. Supported: "attn_eig_prod".
    logit_agg : List[str], optional
        List of logit-based descriptors to compute. Supported:
            "perplexity_score", "logit_entropy_score", "window_logit_entropy_score".
    logit_config : dict, optional
        Configuration dictionary for logit-based descriptors functions, with keys such as:
            - "top_k": int, number of top logits considered (default 50)
            - "window_size": int, window size for windowed entropy (default 1)
            - "stride": int, stride for windowed entropy (default 1)
    activation_source : {"prompt", "generation", "promptGeneration"}
        Which part of the sequence to extract activations from:
        - "prompt": only from the prompt
        - "generation": only from the generated answer
        - "promptGeneration": prompt and generation answer both concatenated 
    start_offset : int
        Number of tokens to skip from the beginning of the sequence when extracting token 
        activations with extract_token_activations_fn.
    end_offset : int
        The number of tokens to skip from the end of the sequence when extracting token
        activations with extract_token_activations_fn.
    """  
    # Seed everything
    # -----------------------------------
    seed_all(seed)

    # Load model
    # -----------------------------------
    model, tokenizer = load_llama(model_name)

    # Load ID and OOD test dataset
    # -----------------------------------
    id_test_dataset, od_test_dataset = prepare_test_dataset(seed, shuffle, select_slice)

    # Retrieve test descriptors and save results 
    # -----------------------------------
    # Extract OOD test descriptors
    print(f"\nStart retrieving test impossible descriptors from {activation_source}...")
    t2 = time.time()
    run_prompt_and_generation_descriptor_extraction(
        model=model,
        tokenizer=tokenizer,
        dataset=od_test_dataset,
        batch_size=batch_size,
        idx_start_sample=0,
        max_samples=len(od_test_dataset),
        save_to_pkl=True,
        output_path=od_output_path,
        build_prompt_fn=build_prompt_fn,
        layers=layers,  
        activation_source=activation_source,
        hidden_agg=hidden_agg,
        attn_agg=attn_agg,
        logit_agg=logit_agg,
        logit_config=logit_config,
        start_offset=start_offset,
        end_offset=start_offset
    )
    t3 = time.time()
    print("...end!")
    print_time_elapsed(t2, t3, label="Impossible test descriptors: ")

    # Extract ID test descriptors
    print(f"\nStart retrieving test possible descriptors from {activation_source}...")
    t4 = time.time()
    run_prompt_and_generation_descriptor_extraction(
        model=model,
        tokenizer=tokenizer,
        dataset=id_test_dataset,
        batch_size=batch_size,
        idx_start_sample=0,
        max_samples=len(id_test_dataset),
        save_to_pkl=True,
        output_path=id_output_path,
        build_prompt_fn=build_prompt_fn,
        layers=layers,  
        activation_source=activation_source,
        hidden_agg=hidden_agg,
        attn_agg=attn_agg,
        logit_agg=logit_agg,
        logit_config=logit_config,
        start_offset=start_offset,
        end_offset=end_offset
    )
    t5 = time.time()
    print("end!")
    print_time_elapsed(t4, t5, label="Possible test descriptors: ")

     # Merge all batches, save as a single file and delete batch directory
    od_merged = merge_batches_and_cleanup(directory=od_output_path, delete=True, confirm=True) 
    id_merged = merge_batches_and_cleanup(directory=id_output_path, delete=True, confirm=True) 

    # Free memory
    del od_test_dataset, id_test_dataset, model, tokenizer, od_merged, id_merged



# ====================================
# Global variables  
# ====================================
SEED = 777 #42, 44, 777, 123, 2025, 42
BATCH_SIZE = 16
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
OUTPUT_DIR = f"../results/raw/small_dataset_correct_split_allConfig_{SEED}/"
PLOT_DIR   = f"../results/figures/small_dataset_correct_split_allConfig_{SEED}/"
ACTIVATION_SOURCE = "promptGeneration" # can be 'generation', 'prompt', 'promptGeneration'
START_OFFSET = 0 #40
END_OFFSET = 0   #-4

HIDDEN_AGG = ["avg_emb", "last_emb", "max_emb", "first_gen_emb", "hidden_score", "feat_var_emb"]
ATTN_AGG = ["attn_score"]
LOGIT_AGG = ["perplexity_score", "logit_entropy_score", "window_logit_entropy_score"]
LOGIT_CONFIG = {"top_k": 50, "window_size": 1, "stride": 1}

STR_AGG = 'All'

LAYERS = list(range(1, 31, 2)) + [-1] # (List[int]) - Layers from witch retrieve the scores 
STR_LAYERS = '1:32:2' #"_".join(str(x) for x in LAYERS)

# ====================================
# Main function 
# ====================================
def main() -> None:
    """
    Main entry point for the embedding extraction pipeline.
    """

    if False:
        clear_cache()
        run_filter_generated_answers_by_similarity_pipeline(
            model_name=MODEL_NAME,
            seed=SEED,
            output_path=OUTPUT_DIR + f"id_fit_results_answers_BIG_correct_split",
            save_dataset_path="../data/datasets/id_fit_correct_dataset_BIG_correct_split.pkl",
            shuffle=True,
            select_slice=(0, 10_000), # None
            batch_size=BATCH_SIZE,
            build_prompt_fn=build_prompt
        )

    OUTPUT_TITLE =  f"_layer{STR_LAYERS}_agg{STR_AGG}_{ACTIVATION_SOURCE}_so{START_OFFSET}_eo{END_OFFSET}_seed{SEED}"

    print(f"\n\n===========================================================")
    print(f"Processing OUTPUT_TITLE : {OUTPUT_TITLE}")
    print(f"===========================================================\n\n")

    if False: 
        clear_cache()
        retrieve_fit_inputs_descriptor_pipeline(
            model_name=MODEL_NAME,
            seed=SEED,
            output_path=f"{OUTPUT_DIR}id_fit_results{OUTPUT_TITLE}",
            custom_dataset_path="../data/datasets/id_fit_correct_dataset.pkl",
            shuffle=True,
            select_slice=(0,10_000),
            batch_size=BATCH_SIZE,
            build_prompt_fn=build_prompt,
            layers=LAYERS,  
            hidden_agg=HIDDEN_AGG,
            attn_agg=ATTN_AGG,
            logit_agg=LOGIT_AGG,
            logit_config=LOGIT_CONFIG,
            start_offset=START_OFFSET,
            end_offset=END_OFFSET
        )


    if False: 
        clear_cache()
        retrieve_test_inputs_descriptor_pipeline(
            model_name=MODEL_NAME,
            seed=SEED,
            id_output_path=f"{OUTPUT_DIR}id_test_results{OUTPUT_TITLE}",
            od_output_path=f"{OUTPUT_DIR}od_test_results{OUTPUT_TITLE}",
            shuffle=True,
            select_slice=(0,1000),
            batch_size=BATCH_SIZE,
            build_prompt_fn=build_prompt,
            layers=LAYERS,  
            hidden_agg=HIDDEN_AGG,
            attn_agg=ATTN_AGG,
            logit_agg=LOGIT_AGG,
            logit_config=LOGIT_CONFIG,
            start_offset=START_OFFSET,
            end_offset=END_OFFSET
        )


    if True:
        clear_cache()
        retrieve_fit_answers_descriptor_pipeline(
            model_name=MODEL_NAME,
            seed=SEED,
            output_path=f"{OUTPUT_DIR}id_fit_results{OUTPUT_TITLE}",
            custom_dataset_path=f"../data/datasets/id_fit_correct_dataset_small_seed{SEED}.pkl",
            shuffle=True,
            select_slice=(0,10_000), 
            batch_size=BATCH_SIZE,
            build_prompt_fn=build_prompt,
            layers=LAYERS,  
            hidden_agg=HIDDEN_AGG,
            attn_agg=ATTN_AGG,
            logit_agg=LOGIT_AGG,
            logit_config=LOGIT_CONFIG,
            activation_source=ACTIVATION_SOURCE,
            start_offset=START_OFFSET,
            end_offset=END_OFFSET      
        )

    
    if True:
        clear_cache()
        retrieve_test_answers_descriptor_pipeline(
            model_name=MODEL_NAME,
            seed=SEED,
            id_output_path=f"{OUTPUT_DIR}id_test_results{OUTPUT_TITLE}",
            od_output_path=f"{OUTPUT_DIR}od_test_results{OUTPUT_TITLE}",
            shuffle=True,
            select_slice=(0,1000), # 8760 big wrong split, 5920 big corr split, 1000 small all config
            batch_size=BATCH_SIZE,
            build_prompt_fn=build_prompt,
            layers=LAYERS,  
            hidden_agg=HIDDEN_AGG,
            attn_agg=ATTN_AGG,
            logit_agg=LOGIT_AGG,
            logit_config=LOGIT_CONFIG,
            activation_source=ACTIVATION_SOURCE,
            start_offset=START_OFFSET,
            end_offset=END_OFFSET      
        )

    
    # Evil Code for squatting on deel machines 
    # (｀∀´)Ψ mouahahah
    if False:
        import time
        try:
            while True: 
                clear_cache()
                #OUTPUT_DIR = "../results/raw/TEST/"
                print("Infinite loop.")
                retrieve_test_inputs_embeddings_pipeline(
                    model_name=MODEL_NAME,
                    seed=SEED,
                    id_output_path=f"{OUTPUT_DIR}id_test_results{OUTPUT_PROMPT_TITLE}",
                    od_output_path=f"{OUTPUT_DIR}od_test_results{OUTPUT_PROMPT_TITLE}",
                    shuffle=True,
                    select_slice=(0,100), #(0,1000),
                    batch_size=BATCH_SIZE,
                    build_prompt_fn=build_prompt,
                    layer_idx=LAYER,
                    extract_token_activations_fn=partial(extract_token_activations, mode=EXTRACTION_MODE),
                    start_offset=START_OFFSET,
                    end_offset=END_OFFSET
                )
                time.sleep(120)  # Pause after execution
        except KeyboardInterrupt:
            print("Stop of the infinite loop.")




if __name__ == "__main__":
    main()

