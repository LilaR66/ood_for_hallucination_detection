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
import torch
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# ====================================
# Import librairies
# ====================================
from typing import Optional, Tuple, Callable, Literal
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
from src.inference.inference_utils import (
    run_filter_generated_answers_by_similarity, 
    run_prompt_activation_extraction, 
    run_prompt_and_generation_activation_extraction,
    build_prompt
)
from src.inference.activation_utils import (
    register_forward_activation_hook,
    register_generation_activation_hook, 
    extract_token_activations
)

from src.data_reader.pickle_io import load_pickle_batches
from src.utils.general import filter_correct_entries
# ====================================
#  Define funtions 
# ====================================
def clear_cache():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


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
        Path to save the extracted answers as a pickle file.
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
        idx_start_sample= 0,
        max_samples=len(id_fit_dataset),
        output_path=output_path,
        build_prompt_fn=build_prompt_fn
    )
    t1 = time.time()
    print("...end!")
    print_time_elapsed(t0, t1, label="ID answers: ")

    # Load ID responses and only keep correct entries 
    # -----------------------------------
    # Load extracted answers 
    id_fit_answers = load_pickle_batches(output_path)
    # Only keep rows where the generated responses are similar to the ground-truth answers
    ids_correct_answers = filter_correct_entries(id_fit_answers)["id"]
    # Create a new dataset contaning only the correct answers 
    id_fit_correct_dataset =  id_fit_dataset.filter_by_column('id', ids_correct_answers)
    # Save the new correct dataset for later use
    id_fit_correct_dataset.save(save_dataset_path) 


def retrieve_fit_inputs_embeddings_pipeline(
    model_name: str,
    seed: int,
    output_path: str,
    custom_dataset_path: str = None,
    shuffle: bool = False,
    select_slice: Optional[Tuple[int, int]]  = None,
    batch_size: int = 16,
    build_prompt_fn: Callable = None,
    layer_idx: int = -1,
    extract_token_activations_fn: Callable = None,
    start_offset: int = 0,
    end_offset: int = 0
) -> None:
    """
    Extract and save embeddings for the in-distribution (ID) training set.

    Parameters
    ----------
    model_name : str
        Name or path of the pretrained Llama model to load.
    seed : int
        Random seed for reproducibility.
    output_path : str
        Path to save the extracted embeddings as a pickle file.
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
    layer_idx : int
        Index of the transformer layer to extract activations from.
    extract_token_activations_fn : callable
        Function to extract token activations from a model layer.
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

    # Retrieve ID embeddings and save results
    # -----------------------------------
    print("\nStart retrieving ID embeddings from inputs...")
    t0 = time.time()
    run_prompt_activation_extraction(
        model=model,
        tokenizer=tokenizer,
        dataset=id_fit_dataset,
        batch_size=batch_size,
        idx_start_sample=0,
        max_samples=len(id_fit_dataset),
        save_to_pkl=True, 
        output_path=output_path,
        build_prompt_fn=build_prompt_fn,
        register_forward_activation_hook_fn=register_forward_activation_hook,
        layer_idx=layer_idx,
        extract_token_activations_fn=extract_token_activations_fn,
        start_offset=start_offset,
        end_offset=end_offset
    )
    t1 = time.time()
    print("...end!")
    print_time_elapsed(t0, t1, label="ID embeddings: ")

    # Free memory
    del id_fit_dataset


def retrieve_test_inputs_embeddings_pipeline(
    model_name: str,
    seed: int,
    id_output_path: str,
    od_output_path: str,
    shuffle: bool = False,
    select_slice: Optional[Tuple[int, int]]  = None,
    batch_size: int = 16,
    build_prompt_fn: Callable = None,
    layer_idx: int = -1,
    extract_token_activations_fn: Callable = None,
    start_offset: int = 0,
    end_offset: int = 0
) -> None:
    """
    Extract and save embeddings for the in-distribution (ID) and 
    out-of-distribution (OOD) test sets.

    Parameters
    ----------
    model_name : str
        Name or path of the pretrained Llama model to load.
    id_output_path : str
        Path to save the ID test embeddings as a pickle file.
    od_output_path : str
        Path to save the OOD test embeddings as a pickle file.
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
    layer_idx : int
        Index of the transformer layer to extract activations from.
    extract_token_activations_fn : callable
        Function to extract token activations from a model layer.
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

    # Retrieve test embeddings and save results 
    # -----------------------------------
    # Extract OOD test embeddings
    print("\nStart retrieving test impossible embeddings from inputs...")
    t2 = time.time()
    run_prompt_activation_extraction(
        model=model,
        tokenizer=tokenizer,
        dataset=od_test_dataset,
        batch_size=batch_size,
        idx_start_sample=0,
        max_samples=len(od_test_dataset),
        save_to_pkl=True, 
        output_path=od_output_path,
        build_prompt_fn=build_prompt_fn,
        register_forward_activation_hook_fn=register_forward_activation_hook,
        layer_idx=layer_idx,
        extract_token_activations_fn=extract_token_activations_fn,
        start_offset=start_offset,
        end_offset=end_offset
    )
    t3 = time.time()
    print("...end!")
    print_time_elapsed(t2, t3, label="Impossible test embeddings: ")

    # Extract ID test embeddings
    print("\nStart retrieving test possible embeddings from inputs...")
    t4 = time.time()
    run_prompt_activation_extraction(
        model=model,
        tokenizer=tokenizer,
        dataset=id_test_dataset,
        batch_size=batch_size,
        idx_start_sample=0,
        max_samples=len(id_test_dataset),
        save_to_pkl=True, 
        output_path=id_output_path,
        build_prompt_fn=build_prompt_fn,
        register_forward_activation_hook_fn=register_forward_activation_hook,
        layer_idx=layer_idx,
        extract_token_activations_fn=extract_token_activations_fn,
        start_offset=start_offset,
        end_offset=end_offset
    )
    t5 = time.time()
    print("end!")
    print_time_elapsed(t4, t5, label="Possible test embeddings: ")

    # Free memory
    del od_test_dataset
    del id_test_dataset


def retrieve_fit_answers_embeddings_pipeline(
    model_name: str,
    seed: int,
    output_path: str,
    custom_dataset_path: str = None,
    shuffle: bool = False,
    select_slice: Optional[Tuple[int, int]]  = None,
    batch_size: int = 16,
    build_prompt_fn: Callable = None,
    layer_idx: int = -1,
    extract_token_activations_fn: Callable = None,
    activation_source: Literal["prompt", "generation", "promptGeneration"] = "generation",
    k_beams : int = 1,
    start_offset: int = 0,
    end_offset: int = 0,
) -> None:
    """
    Extract and save embeddings for the in-distribution (ID) training 
    set of the generated answers.

    Parameters
    ----------
    model_name : str
        Name or path of the pretrained Llama model to load.
    seed : int
        Random seed for reproducibility.
    output_path : str
        Path to save the extracted embeddings as a pickle file.
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
    layer_idx : int
        Index of the transformer layer to extract activations from.
    extract_token_activations_fn : callable
        Function to extract token activations from a model layer.
    activation_source : {"prompt", "generation", "promptGeneration"}
        Which part of the sequence to extract activations from:
        - "prompt": only from the prompt
        - "generation": only from the generated answer
        - "promptGeneration": prompt and generation answer both concatenated
    k_beams : int, optional
        Number of beams for beam search during generation (default: 1). If 1, uses sampling. 
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

    # Retrieve ID embeddings and save results
    # -----------------------------------
    print("\nStart retrieving ID embeddings from generated answers...")
    t0 = time.time()
    run_prompt_and_generation_activation_extraction(
        model=model,
        tokenizer=tokenizer,
        dataset=id_fit_dataset,
        batch_size=batch_size,
        idx_start_sample=0,
        max_samples=len(id_fit_dataset),
        save_to_pkl=True, 
        output_path=output_path,
        build_prompt_fn=build_prompt_fn,
        register_generation_activation_hook_fn=register_generation_activation_hook,
        layer_idx=layer_idx,
        extract_token_activations_fn=extract_token_activations_fn,
        activation_source=activation_source,
        k_beams=k_beams,
        start_offset=start_offset,
        end_offset=end_offset
    )
    t1 = time.time()
    print("...end!")
    print_time_elapsed(t0, t1, label="ID embeddings: ")

    # Free memory
    del id_fit_dataset


def retrieve_test_answers_embeddings_pipeline(
    model_name: str,
    seed: int,
    id_output_path: str,
    od_output_path: str,
    shuffle: bool = False,
    select_slice: Optional[Tuple[int, int]]  = None,
    batch_size: int = 16,
    build_prompt_fn: Callable = None,
    layer_idx: int = -1,
    extract_token_activations_fn: Callable = None,
    activation_source: Literal["prompt", "generation", "promptGeneration"] = "generation",
    k_beams : int = 1,
    start_offset: int = 0,
    end_offset: int = 0,
) -> None:
    """
    Extract and save embeddings for the in-distribution (ID) and 
    out-of-distribution (OOD) test sets of the generated answers.

    Parameters
    ----------
    model_name : str
        Name or path of the pretrained Llama model to load.
    id_output_path : str
        Path to save the ID test embeddings as a pickle file.
    od_output_path : str
        Path to save the OOD test embeddings as a pickle file.
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
    layer_idx : int
        Index of the transformer layer to extract activations from.
    extract_token_activations_fn : callable
        Function to extract token activations from a model layer.
    activation_source : {"prompt", "generation", "promptGeneration"}
        Which part of the sequence to extract activations from:
        - "prompt": only from the prompt
        - "generation": only from the generated answer
        - "promptGeneration": prompt and generation answer both concatenated
    k_beams : int, optional
        Number of beams for beam search during generation (default: 1). If 1, uses sampling. 
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

    # Retrieve test embeddings and save results 
    # -----------------------------------
    # Extract OOD test embeddings
    print("\nStart retrieving test impossible embeddings from answers...")
    t2 = time.time()
    run_prompt_and_generation_activation_extraction(
        model=model,
        tokenizer=tokenizer,
        dataset=od_test_dataset,
        batch_size=batch_size,
        idx_start_sample=0,
        max_samples=len(od_test_dataset),
        save_to_pkl=True, 
        output_path=od_output_path,
        build_prompt_fn=build_prompt_fn,
        register_generation_activation_hook_fn=register_generation_activation_hook,
        layer_idx=layer_idx,
        extract_token_activations_fn=extract_token_activations_fn,
        activation_source=activation_source,
        k_beams=k_beams,
        start_offset=start_offset,
        end_offset=end_offset
    )
    t3 = time.time()
    print("...end!")
    print_time_elapsed(t2, t3, label="Impossible test embeddings: ")

    # Extract ID test embeddings
    print("\nStart retrieving test possible embeddings from answers...")
    t4 = time.time()
    run_prompt_and_generation_activation_extraction(
        model=model,
        tokenizer=tokenizer,
        dataset=id_test_dataset,
        batch_size=batch_size,
        idx_start_sample=0,
        max_samples=len(id_test_dataset),
        save_to_pkl=True, 
        output_path=id_output_path,
        build_prompt_fn=build_prompt_fn,
        register_generation_activation_hook_fn=register_generation_activation_hook,
        layer_idx=layer_idx,
        extract_token_activations_fn=extract_token_activations_fn,
        activation_source=activation_source,
        k_beams=k_beams,
        start_offset=start_offset,
        end_offset=end_offset
    )
    t5 = time.time()
    print("end!")
    print_time_elapsed(t4, t5, label="Possible test embeddings: ")

    # Free memory
    del od_test_dataset
    del id_test_dataset


# ====================================
# Global variables  
# ====================================
SEED = 44
BATCH_SIZE = 16 #32
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
OUTPUT_DIR = "../results/raw/analyse_answers/"
PLOT_DIR   = "../results/figures/analyse_answers/"
K_BEAMS = 1 #3
ACTIVATION_SOURCE = "promptGeneration" # can be 'Generation', 'Prompt', 'PromptGeneration'
START_OFFSET = 0 #40
END_OFFSET = 0   #-4


LAYER_LIST = [-1, 16]  # integer
TOKENS_LIST = ["-1", "Avg", "Max"]  # string

# ====================================
# Main function 
# ====================================
def main() -> None:
    """
    Main entry point for the embedding extraction pipeline.
    """
    for LAYER in LAYER_LIST:
        for TOKENS in TOKENS_LIST:

            # Select function to extract token embeddings
            if TOKENS=="-1":
                EXTRACTION_MODE = "last"
            elif TOKENS=="Avg":
                EXTRACTION_MODE = "average"
            elif TOKENS=="Max":
                EXTRACTION_MODE = "max"

            OUTPUT_PROMPT_TITLE = f"_layer{LAYER}_token{TOKENS}_prompt_so{START_OFFSET}_eo{END_OFFSET}"
            OUTPUT_GEN_TITLE =  f"_layer{LAYER}_token{TOKENS}_{ACTIVATION_SOURCE}_kbeams{K_BEAMS}_so{START_OFFSET}_eo{END_OFFSET}"

            print(f"\n\n===========================================================")
            print(f"Processing OUTPUT_PROMPT_TITLE : {OUTPUT_PROMPT_TITLE}")
            print(f"           OUTPUT_GEN_TITLE    : {OUTPUT_GEN_TITLE}")
            print(f"\n\n===========================================================")

            if False:
                clear_cache()
                run_filter_generated_answers_by_similarity_pipeline(
                    model_name=MODEL_NAME,
                    seed=SEED,
                    output_path=OUTPUT_DIR + f"id_fit_results_answers.pkl",
                    save_dataset_path="../data/datasets/id_fit_correct_dataset.pkl",
                    shuffle=True,
                    select_slice=(0,10_000),
                    batch_size=BATCH_SIZE,
                    build_prompt_fn=build_prompt
                )

            if False:
                clear_cache()
                retrieve_fit_inputs_embeddings_pipeline(
                    model_name=MODEL_NAME,
                    seed=SEED,
                    output_path=f"{OUTPUT_DIR}id_fit_results{OUTPUT_PROMPT_TITLE}.pkl",
                    custom_dataset_path="../data/datasets/id_fit_correct_dataset.pkl",
                    shuffle=True,
                    select_slice=(0,10_000),
                    batch_size=BATCH_SIZE,
                    build_prompt_fn=build_prompt,
                    layer_idx=LAYER,
                    extract_token_activations_fn=partial(extract_token_activations, mode=EXTRACTION_MODE),
                    start_offset=START_OFFSET,
                    end_offset=END_OFFSET
                )


            if False: 
                clear_cache()
                retrieve_test_inputs_embeddings_pipeline(
                    model_name=MODEL_NAME,
                    seed=SEED,
                    id_output_path=f"{OUTPUT_DIR}id_test_results{OUTPUT_PROMPT_TITLE}.pkl",
                    od_output_path=f"{OUTPUT_DIR}od_test_results{OUTPUT_PROMPT_TITLE}.pkl",
                    shuffle=True,
                    select_slice=(0,1000),
                    batch_size=BATCH_SIZE,
                    build_prompt_fn=build_prompt,
                    layer_idx=LAYER,
                    extract_token_activations_fn=partial(extract_token_activations, mode=EXTRACTION_MODE),
                    start_offset=START_OFFSET,
                    end_offset=END_OFFSET
                )


            if True:
                clear_cache()
                retrieve_fit_answers_embeddings_pipeline(
                    model_name=MODEL_NAME,
                    seed=SEED,
                    output_path=f"{OUTPUT_DIR}id_fit_results{OUTPUT_GEN_TITLE}.pkl",
                    custom_dataset_path="../data/datasets/id_fit_correct_dataset.pkl",
                    shuffle=True,
                    select_slice=(0,10_000),
                    batch_size=BATCH_SIZE,
                    build_prompt_fn=build_prompt,
                    layer_idx=LAYER,
                    extract_token_activations_fn=partial(extract_token_activations, mode=EXTRACTION_MODE),
                    activation_source=ACTIVATION_SOURCE,
                    k_beams=K_BEAMS,
                    start_offset=START_OFFSET,
                    end_offset=END_OFFSET,       
                )

            
            if True:
                clear_cache()
                retrieve_test_answers_embeddings_pipeline(
                    model_name=MODEL_NAME,
                    seed=SEED,
                    id_output_path=f"{OUTPUT_DIR}id_test_results{OUTPUT_GEN_TITLE}.pkl",
                    od_output_path=f"{OUTPUT_DIR}od_test_results{OUTPUT_GEN_TITLE}.pkl",
                    shuffle=True,
                    select_slice=(0,1000),
                    batch_size=BATCH_SIZE,
                    build_prompt_fn=build_prompt,
                    layer_idx=LAYER,
                    extract_token_activations_fn=partial(extract_token_activations, mode=EXTRACTION_MODE),
                    activation_source=ACTIVATION_SOURCE,
                    k_beams=K_BEAMS,
                    start_offset=START_OFFSET,
                    end_offset=END_OFFSET,
                )



if __name__ == "__main__":
    main()

