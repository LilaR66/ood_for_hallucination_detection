#!/usr/bin/env python3
"""
============================================================
Utilities for batched inference and activation extraction with LLMs
============================================================

This module provides a suite of functions for conducting batched inference using
decoder-only language models (e.g., LLaMA), including structured prompt construction,
answer generation, and extraction of internal transformer activations for each input.

The design is modular, allowing users to specify custom prompt builders, token extractors,
and layer selection strategies. It also supports saving inference results (activations,
generated answers, and evaluation metrics) to disk in a pickle format.

Main Features
-------------
- Constructs LLaMA-compatible prompts for question answering
- Performs batched tokenization and generation
- Hooks and retrieves activations from arbitrary transformer layers
- Extracts contextual embeddings of tokens (e.g., at the beginning of answer span)
- Computes semantic similarity using ROUGE and Sentence-BERT
- Exports results incrementally to disk for large-scale evaluation
"""


from transformers import PreTrainedTokenizer, PreTrainedModel, BatchEncoding
import torch
from datasets import  Dataset
from tqdm import tqdm
from typing import Dict, List, Any, Callable
import time

from src.evaluation.similarity_metrics import rouge_l_simScore, sentence_bert_simScore
from src.data_reader.pickle_io import append_to_pickle

# Specific to Llama tokenizer: 
def build_prompt(context:str, question:str) -> str:
    """
    Construct a structured prompt for question answering with an LLM.

    The prompt includes the special `[INST] "Input intruction here" [/INST]` 
    (to indicate the start of the instruction block).

    **Note:** the llama tokenizer adds a special `<s> ` before `[INST]`. 
    **Note:** we also experimented with giving 2-3 few shot prompts. 
    However, just appending the sentence "Just give the answer, without a complete sentence." 
    at the begining of the prompt seemed to work best. 

    Parameters
    ----------
    context : str
        The input passage or context from which the answer should be extracted.
    question : str
        The question to be answered based on the provided context.

    Returns
    -------
    str
        A formatted prompt string ready to be fed to a language model.
    
    """
    prompt = f"[INST]\n\nJust give the answer, without a complete sentence.\n\nContext:\n" + context + "\n\nQuestion:\n" + question  + "\n\nAnswer:\n[/INST]" 
    return prompt

# Specific to Llama tokenizer: 
def build_impossible_prompt(context:str, question:str) -> str:
    """
    Construct a structured prompt for question answering with an LLM.

    The prompt includes the special `<s>[INST] "Input intruction here" [/INST]` 
    (to indicate the start of the instruction block).
    
    **Note:** the llama tokenizer adds a special `<s> ` before `[INST]`.
    **Note:** we also experimented with giving 2-3 few shot prompts. 
    However, just appending the sentence "Just give the answer, without a complete sentence." 
    at the begining of the prompt seemed to work best. 
    **Note:** Since this answer is impossible, we add a prompt to inform the model. 

    Parameters
    ----------
    context : str
        The input passage or context from which the answer should be extracted.
    question : str
        The question to be answered based on the provided context.

    Returns
    -------
    str
        A formatted prompt string ready to be fed to a language model.
    
    """
    prompt = f"[INST]\n\nJust give the answer, without a complete sentence. Reply with 'Impossible to answer' if answer not in context.\n\nContext:\n" + context + "\n\nQuestion:\n" + question  + "\n\nAnswer:\n[/INST]" 
    return prompt


def get_layer_output(
    model: PreTrainedModel, 
    inputs: Dict[str, torch.Tensor], 
    layer_idx: int
) -> torch.Tensor:
    """
    Run a forward pass and extract the hidden states from a specific transformer layer
    (more memory-efficient than using output_hidden_states=True).
    Transformer layer = self-attention + FFN + normalization.

    Parameters
    ----------
    model : PreTrainedModel
        A Hugging Face causal language model (e.g., LLaMA, GPT-2).
    inputs : dict
        Tokenized inputs returned by a tokenizer with return_tensors="pt".
    layer_idx : int
        Index of the transformer block to capture:
        - Use 0 to N-1 for internal layers.
        - Use -1 to retrieve the final transformer block (not logits).

    Returns
    -------
    torch.Tensor
        Hidden states from the selected transformer layer.
        Shape: (batch_size, seq_len, hidden_size)
    """
    # If layer_idx = -1, interpret as "last transformer block"
    if layer_idx == -1:
        layer_idx = len(model.model.layers) - 1  # last layer index

    captured_hidden = {}

    def hook_fn(module, input, output):
        # output is a tuple (hidden_states,) → keep [0]
        captured_hidden["layer_output"] = output[0]

    # Register hook on the transformer block
    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model(**inputs, return_dict=True)

    handle.remove()

    if "layer_output" not in captured_hidden:
        raise RuntimeError(f"Layer {layer_idx} did not produce an output.")

    return captured_hidden["layer_output"]  # shape: (batch_size, seq_len, hidden_size)



def extract_batch(
    dataset: Dataset,
    idx_start: int, 
    batch_size: int
) -> List[Dict[str, Any]]:
    """
    Extract a slice from a HuggingFace Dataset and convert it to a list of dictionaries.

    Parameters
    ----------
    dataset : Dataset
        The HuggingFace Dataset to slice.
    idx_start : int
        Start index (inclusive) of the batch.
    batch_size : int
        Number of examples to include in the batch.

    Returns
    -------
    List[Dict[str, Any]]
        List of examples as dictionaries (one per example).
    """
    end_idx = min(idx_start + batch_size, len(dataset))
    batch = dataset.select(range(idx_start, end_idx))
    batch_dicts = batch.to_dict()
    return [dict(zip(batch_dicts.keys(), vals)) for vals in zip(*batch_dicts.values())]


# Specific to Llama tokenizer: 
def extract_last_token_activations(
    selected_layer: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    offset: int = -5
) -> torch.Tensor:
    """
    Extract the activation vector of the token at a specific offset 
    from the last non-padding token in each sequence. 
    
    Example: when working with structured question-answering (QA) prompts, such as:
    <s> [INST]\n\nContext:\n...\n\nQuestion:\n...\n\nAnswer:\n[/INST]
    for the last token embeddings, we want to extract the token embedding of the '\n' just after 'Answer:'. 
    Since the llama tokeniser tokenizes '\n[/INST]' into '\n', '[', '/', 'INST', ']' we 
    select '\n' with an offset of -5. 

    Parameters
    ----------
    selected_layer : torch.Tensor
        Output tensor from the selected model layer (shape: batch_size x seq_len x hidden_size).
    attention_mask : torch.Tensor
        Attention mask indicating real tokens (1) vs padding (0) (shape: batch_size x seq_len).
    device : torch.device
        Device to perform indexing operations on.
    offset : int, optional
        Negative offset from the last non-padding token (e.g., -5 for the 5th token before the end). Default is -5.

    Returns
    -------
    torch.Tensor
        Activation vectors of the selected token for each input (shape: batch_size x hidden_size).
    """
    # Find the index of the last non-padding token for each sequence
    # Handle left and right padding
    if (attention_mask[:, 0] == 0).any():  # If any sequence starts with padding, assume left padding
        last_indices = (attention_mask.size(1) - 1) - attention_mask.flip(dims=[1]).argmax(dim=1)
    else:  # right padding
        last_indices = (attention_mask.sum(dim=1) - 1)
    last_indices = last_indices.to(device)

    # Compute the target index using the offset
    target_indices = (last_indices + offset + 1)
    batch_indices = torch.arange(selected_layer.size(0), device=device)
    # Extract the activations at the target indices
    return selected_layer[batch_indices, target_indices]


def generate_answers(
    model: PreTrainedModel,
    inputs: BatchEncoding,
    tokenizer: PreTrainedTokenizer,
    max_new_tokens: int = 50
) -> torch.Tensor:
    
    """
    Generate answers from the model for a batch of inputs.

    Parameters
    ----------
    model : PreTrainedModel
        The language model to use for generation.
    inputs : BatchEncoding
        Tokenized input prompts.
    tokenizer : PreTrainedTokenizer
        Tokenizer for special token IDs.
    max_new_tokens : int, optional
        Maximum number of tokens to generate per answer.

    Returns
    -------
    torch.Tensor
        Generated token IDs for each example in the batch.
    """
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            top_k=50,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id 
        )
    return output_ids


def analyze_single_generation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    sample_idx: int = 0,
    build_prompt_fn: Callable[[str, str], str] = None,
    get_layer_output_fn: Callable = None,
    layer_idx: int = -1,
    extract_token_activations_fn: Callable = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze the computation time and output of a single sample through the model pipeline.

    This function processes one sample from the dataset: it builds a prompt, tokenizes it,
    extracts layer activations, generates an answer, decodes the output, computes similarity
    scores with the ground-truth answer, and prints computation times for each step.

    Parameters
    ----------
    model : PreTrainedModel
        The Hugging Face transformer model (e.g., LLaMA, GPT-2).
    tokenizer : PreTrainedTokenizer
        The corresponding tokenizer.
    dataset : Dataset
        The Hugging Face dataset containing context, question, and answers fields.
    sample_idx : int, optional
        Index of the sample to analyze (default is 0).
    build_prompt_fn : Callable[[str, str], str], optional
        Function to build a prompt from context and question.
    get_layer_output_fn : Callable = None,
        Function to extract the output of a specific model layer.
    layer_idx : int, optional
        Index of the transformer layer to extract activations from (default is -1 for last layer).
    extract_token_activations_fn : Callable = None,
        Function to extract token activations from a model layer.
    **kwargs :
        Additional keyword arguments passed to extract_token_activations_fn.
    Returns
    -------
    Dict[str, Any]
        Dictionary containing prompt, generated answer, ground-truth answer, similarity scores,
        and computation times for each step.
    """
    sample = dataset[sample_idx]

    print("========= Analyze one generation  =========")
    times = {}

    # Prompt construction
    t0 = time.time()
    prompt = build_prompt_fn(sample["context"], sample["question"])
    answer = sample["answers"]['text']
    times['prompt_construction'] = time.time() - t0
    print(f"----- Prompt construction: {times['prompt_construction']:.3f} sec")

    # Tokenization
    t1 = time.time()
    inputs = tokenizer(prompt, truncation=True, padding=True, return_tensors="pt").to(model.device)
    times['tokenization'] = time.time() - t1
    print(f"----- Tokenization: {times['tokenization']:.3f} sec")

    # Layer output extraction
    t2 = time.time()
    selected_layer = get_layer_output_fn(model, inputs, layer_idx)
    times['layer_output'] = time.time() - t2
    print(f"----- Token extraction: {times['layer_output']:.3f} sec")

    # Token activations extraction
    t3 = time.time()
    selected_token_vecs = extract_token_activations_fn(
        selected_layer,
        inputs["attention_mask"],
        device=selected_layer.device,
        **kwargs
    )
    times['token_activations'] = time.time() - t3

    # Generation
    t4 = time.time()
    output_ids = generate_answers(model, inputs, tokenizer)
    times['generation'] = time.time() - t4
    print(f"----- Generation: {times['generation']:.3f} sec")

    # Decoding
    t5 = time.time()
    prompt_len = len(inputs["input_ids"][0])
    generated_answer_ids = output_ids[0][prompt_len:]
    generated_answer = tokenizer.decode(generated_answer_ids, skip_special_tokens=True).strip()
    times['decoding'] = time.time() - t5
    print(f"----- Decoding: {times['decoding']:.3f} sec")

    # Similarity scoring
    t6 = time.time()
    rouge_l_score = rouge_l_simScore(generated_answer, answer) 
    sbert_sim = sentence_bert_simScore(generated_answer, answer)
    is_correct = (rouge_l_score >= 0.5) or (sbert_sim >= 0.4)
    times['similarity_scoring'] = time.time() - t6
    print(f"----- Similarity scoring: {times['similarity_scoring']:.3f} sec")

    # Display information
    print("\n=== Prompt ===")
    print(prompt)
    print("\n=== Shapes ===")
    print(f"Shape - number of tokens: {inputs['input_ids'].shape}")
    print(f"Shape - selected_layer: {selected_layer.shape}")
    print("\n=== Generated Answer ===")
    print(generated_answer)
    print("\n=== Ground-truth Answer ===")
    print(answer)
    print("\n=== Similarity Scores ===")
    print(f"ROUGE-L F1: {rouge_l_score:.4f}")
    print(f"Sentence-BERT Cosine Similarity: {sbert_sim:.4f}")
    print(f"Is generated answer correct: {is_correct}")

    return {
        "prompt": prompt,
        "generated_answer": generated_answer,
        "ground_truth_answer": answer,
        "rouge_l_score": rouge_l_score,
        "sbert_score": sbert_sim,
        "is_correct": is_correct,
        "computation_times": times,
        "input_shape": inputs['input_ids'].shape,
        "layer_shape": selected_layer.shape,
        "token_activations": selected_token_vecs,
    }



def batch_extract_token_activations_with_generation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    batch_size: int = 4,
    idx_start_sample: int = 0,
    max_samples: int = 1000,
    output_path: str = "outputs/all_batch_results.pkl",
    build_prompt_fn: Callable[[str, str], str] = None,
    get_layer_output_fn: Callable = None,
    layer_idx: int = -1,  
    extract_token_activations_fn: Callable = None,
    **kwargs
):
    """
    Runs batched inference on a dataset using a decoder-only language model.
    For each batch, generates answers, computes semantic similarity scores, extracts token-level activations,
    and appends the results to a pickle file.

    Parameters
    ----------
    model : PreTrainedModel
        The causal language model to evaluate (e.g., LLaMA).
    tokenizer : PreTrainedTokenizer
        The corresponding tokenizer.
    dataset : Dataset
        The input dataset.
    batch_size : int
        Number of samples per batch.
    idx_start_sample : int
        Index of the first sample to process from the dataset.
    max_samples : int
        Total number of examples to process from the dataset, starting from idx_start_sample. 
    output_path : str
        Path to the pickle file for saving intermediate results.
    build_prompt_fn : Callable
        Function to build a prompt from context and question.
    get_layer_output_fn : Callable
        Function to extract the output of a specific model layer.
    layer_idx : int
        Index of the transformer layer to extract activations from (default: -1 for last layer).
    extract_token_activations_fn : Callable
        Function to extract token activations from a model layer.
    **kwargs :
        Extra keyword arguments passed to extract_token_activations_fn.
    """
    for i in tqdm(range(idx_start_sample, idx_start_sample + max_samples, batch_size)):
        batch_answers = []               # Generated answers
        batch_gt_answers = []            # Ground-truth answers
        batch_is_correct = []            # 0/1 labels indicating correctness
        batch_dataset_ids = []           # 'id' field from dataset
        batch_dataset_original_idx = []  # Original indices from dataset
        batch_rouge_scores = []          # Rouge-L scores
        batch_sbert_scores = []          # Sentence-Bert scores

        batch = extract_batch(dataset, i, batch_size)
        prompts = [build_prompt_fn(s["context"], s["question"]) for s in batch]
        answers = [s["answers"]["text"] for s in batch]
        inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(model.device)
        selected_layer = get_layer_output_fn(model, inputs, layer_idx)
        selected_token_vecs = extract_token_activations_fn(
                selected_layer, 
                inputs["attention_mask"], 
                device=selected_layer.device,
                **kwargs) 
        output_ids = generate_answers(model, inputs, tokenizer)
        
        for j in range(len(prompts)):
            # --- Decode token IDs into text ---
            prompt_len = len(inputs["input_ids"][j]) # Length of prompt j
            generated_answer_ids = output_ids[j][prompt_len:] # Remove prompt prefix to isolate the generated answer
            generated_answer = tokenizer.decode(generated_answer_ids, skip_special_tokens=True).strip()

            # --- Compute semantic similarity between model's answer and ground-truth ---    
            rouge_l_score = rouge_l_simScore(generated_answer, answers[j])
            if rouge_l_score >= 0.5:
                is_correct = True
                sbert_score = None
            else:
                sbert_score = sentence_bert_simScore(generated_answer, answers[j])
                is_correct = (sbert_score >= 0.4)

            # --- Store everything ---
            batch_dataset_ids.append(batch[j]['id'])
            batch_dataset_original_idx.append(batch[j]['original_index'])
            batch_answers.append(generated_answer)
            batch_gt_answers.append(answers[j])
            batch_is_correct.append(int(is_correct))
            batch_rouge_scores.append(rouge_l_score)
            batch_sbert_scores.append(sbert_score)

        # --- Save progress to pickle after each batch ---
        batch_results = {
            "id": batch_dataset_ids,
            "original_indices": batch_dataset_original_idx,
            "gen_answers": batch_answers,
            "ground_truths": batch_gt_answers,
            "activations": [selected_token_vecs[i].unsqueeze(0).cpu() for i in range(selected_token_vecs.size(0))],
            "is_correct": batch_is_correct,
            "sbert_scores": batch_sbert_scores,
            "rouge_scores": batch_rouge_scores
        }
        append_to_pickle(output_path, batch_results)
    

def batch_extract_token_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    batch_size: int = 4,
    idx_start_sample: int = 0,
    max_samples: int = 1000,
    save_to_pkl: bool = False,
    output_path: str = "outputs/all_batch_results.pkl",
    build_prompt_fn: Callable[[str, str], str] = None,
    get_layer_output_fn: Callable = None,
    layer_idx: int = -1,  
    extract_token_activations_fn: Callable = None,
    **kwargs
):
    """
    Runs batched inference on a dataset using a decoder-only language model.
    For each batch, gextracts token-level activations, and appends the results to a pickle file.

    Parameters
    ----------
    model : PreTrainedModel
        The causal language model to evaluate (e.g., LLaMA).
    tokenizer : PreTrainedTokenizer
        The corresponding tokenizer.
    dataset : Dataset
        The input dataset.
    batch_size : int
        Number of samples per batch.
    idx_start_sample : int
        Index of the first sample to process from the dataset.
    max_samples : int
        Total number of examples to process from the dataset, starting from idx_start_sample. 
    save_to_pkl : bool
        If True, activations are appended to the pickle file at output_path.
        If False, the function returns a list of activations.
    output_path : str
        Path to the pickle file for saving intermediate results.
    build_prompt_fn : Callable
        Function to build a prompt from context and question.
    get_layer_output_fn : Callable
        Function to extract the output of a specific model layer.
    layer_idx : int
        Index of the transformer layer to extract activations from (default: -1 for last layer).
    extract_token_activations_fn : Callable
        Function to extract token activations from a model layer.
    **kwargs :
        Extra keyword arguments passed to extract_token_activations_fn.
    
    Returns
    -------
    List[torch.Tensor]
        List of activations, each of shape (1, hidden_size), for all processed examples
        (returned only if save_to_pkl is False).
    """
    batch_activations = []  # Chosen token activation vectors

    for i in tqdm(range(idx_start_sample, idx_start_sample + max_samples, batch_size)):
        
        batch = extract_batch(dataset, i, batch_size)
        prompts = [build_prompt_fn(s["context"], s["question"]) for s in batch]
        inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(model.device)
        selected_layer = get_layer_output_fn(model, inputs, layer_idx)
        selected_token_vecs = extract_token_activations_fn(
                selected_layer, 
                inputs["attention_mask"], 
                device=selected_layer.device,
                **kwargs) 

        activations = [selected_token_vecs[j].unsqueeze(0).cpu() for j in range(selected_token_vecs.size(0))]
        batch_dataset_ids = [s['id'] for s in batch]
        batch_dataset_original_idx = [s['original_index'] for s in batch]
        
        batch_results = {
            "id": batch_dataset_ids,
            "original_indices": batch_dataset_original_idx,
            "activations": activations 
        }

        if save_to_pkl:
            append_to_pickle(output_path, batch_results)
        else:
            batch_activations.extend(activations)
        
    if not save_to_pkl:
        return batch_activations



