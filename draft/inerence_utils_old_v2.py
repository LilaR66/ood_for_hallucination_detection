#!/usr/bin/env python3
"""
============================================================
Utilities for Prompt Construction, Batched Inference, and Answer Evaluation with LLMs
============================================================

This module provides high-level utilities for running batched inference
using decoder-only language models (e.g., LLaMA). It includes structured prompt
construction, answer generation, evaluation against ground-truth answers, and
optional extraction of token-level activations.


Main Features
-------------
- Constructs LLaMA-compatible prompts for question answering
- Performs batched tokenization and generation
- Computes semantic similarity using ROUGE and Sentence-BERT
- Supports per-sample or batched evaluation and analysis
- Extracts token-level activations from generated or prompted sequences
- Saves generation results (answers, scores, metadata) incrementally to disk
"""


from transformers import PreTrainedTokenizer, PreTrainedModel, BatchEncoding
import torch
from datasets import  Dataset
from tqdm import tqdm
from typing import Dict, List, Any, Callable, Tuple
import time

from src.answer_similarity.similarity_metrics import rouge_l_simScore, sentence_bert_simScore
from src.data_reader.pickle_io import append_to_pickle

# Specific to Llama tokenizer: 
def build_prompt(context:str, question:str) -> str:
    """
    Construct a structured prompt for question answering with an LLM.

    The prompt includes the special formatting: `[INST]`, [/INST]`, `<<SYS>>`
    as recommended here: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

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
    prompt = f"[INST] <<SYS>>\nJust give the answer, without a complete sentence. Reply with 'Impossible to answer' if answer not in context.\n<<SYS>>\n\nContext:\n" + context + "\n\nQuestion:\n" + question  + "\n\nAnswer:\n[/INST]" 
    return prompt



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
        If None, no token extraction is performed. 
    layer_idx : int
        Index of the transformer layer to extract activations from (default: -1 for last layer).
    extract_token_activations_fn : Callable
        Function to extract token activations from a model layer.
        If None, no token extraction is performed. 
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

        # Extract token activations from the specified model layer,
        # and format them as a list of tensors (one per sample) for saving.
        if (get_layer_output_fn is not None) and (extract_token_activations_fn is not None):
            selected_layer = get_layer_output_fn(model, inputs, layer_idx)
            selected_token_vecs = extract_token_activations_fn(
                    selected_layer, 
                    inputs["attention_mask"], 
                    device=selected_layer.device,
                    **kwargs)
            # format activations to save them later
            activations = [selected_token_vecs[i].unsqueeze(0).cpu() for i in range(selected_token_vecs.size(0))]
        else:
            activations = [None for _ in range(batch_size)]

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
            "activations": activations,
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



def batch_extract_answer_token_activations(
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
    include_prompt: bool = True,
    **kwargs
):
    """
    Runs batched inference on a dataset using a decoder-only language model.
    For each batch, generates answers, extracts token-level activations for the generated answer,
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
        Function to extract token activations from a model layer (default is average).
    include_prompt : bool
        Whether to include the prompt in the embedding extraction. 
        - If include_prompt=False: 
            start_offset is set to prompt length and end_offset is set to 0.
        - If include_prompt=True: 
            uses start_offset and end_offset specified in **kwargs (defaults to 0).
        *Note:* Tokenization will always include the prompt.  
    **kwargs :
        Extra keyword arguments passed to extract_token_activations_fn, including start_offset.
    """    
    batch_activations = []  # Chosen token activation vectors

    for i in tqdm(range(idx_start_sample, idx_start_sample + max_samples, batch_size)):
        batch_answers = []   # Generated answers
 
        # Extract a batch from the dataset
        batch = extract_batch(dataset, i, batch_size)
        prompts = [build_prompt_fn(s["context"], s["question"]) for s in batch]

        # Tokenize the prompt 
        inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(model.device)
        
        # Compute the number of non-padding tokens in each prompt (true prompt length)
        prompt_non_pad_len = inputs["attention_mask"].sum(dim=1).tolist()  # Shape (batch_size,)

        # Generate the answers for the batch
        output_ids = generate_answers(model, inputs, tokenizer)

        # Build full sequences (prompt + generated answer) for each sample in the batch
        full_sequences = []
        for j in range(len(prompts)):
            # --- Total length of the tokenized prompt, padding included ---
            prompt_len = len(inputs["input_ids"][j])  # Length of the prompt for example j
            # --- Decode token IDs into text ---
            generated_answer_ids = output_ids[j][prompt_len:]  # Remove prompt part
            generated_answer = tokenizer.decode(generated_answer_ids, skip_special_tokens=True).strip()
            # --- Decode the prompt tokens to text ---
            prompt_text = tokenizer.decode(inputs["input_ids"][j], skip_special_tokens=True)
            # --- Combine prompt and answer for full sequence ---
            full_sequences.append(prompt_text + generated_answer)
            # --- Store generated answers ---
            batch_answers.append(generated_answer)

        # Tokenize the full sequences (prompt + answer) again, with padding and truncation
        # We need to retokenize and cannot directly use `output_ids` since we need attention_mask for get_layer_output_fn
        full_inputs = tokenizer(full_sequences, padding=True, truncation=True, return_tensors="pt").to(model.device)

        # Extract activations from the specified model layer for all sequences in the batch
        selected_layer = get_layer_output_fn(model, full_inputs, layer_idx)

        # Compute the start offsets for activation extraction
        if include_prompt:
            # --- If include_prompt is True, use the value from kwargs (or zeros if not provided) ---
            start_offsets = kwargs.get("start_offset", torch.zeros(len(prompts), device=selected_layer.device)) # Shape (batch_size,)
            end_offsets = kwargs.get("end_offset", torch.zeros(len(prompts), device=selected_layer.device)) # Shape (batch_size,) 
        else:
            # --- If include_prompt is False, use the true prompt length (non-padding tokens) and end_offset=0 ---
            start_offsets = torch.tensor(prompt_non_pad_len, device=selected_layer.device)  # Shape (batch_size,) 
            end_offsets = torch.zeros(len(prompts), device=selected_layer.device) # Shape (batch_size,) 

        # Remove from kwargs to avoid passing it twice to the extraction function
        kwargs.pop("start_offset", None)
        kwargs.pop("end_offset", None)

        # Call the specified activation extraction function
        selected_token_vecs = extract_token_activations_fn(
            selected_layer,
            full_inputs["attention_mask"],
            device=selected_layer.device,
            start_offset=start_offsets,  # Shape (batch_size,) 
            end_offset=end_offsets,     # Shape (batch_size,) 
            **kwargs
        )
        
        # --- Store everything ---
        batch_dataset_ids = [s['id'] for s in batch]  # 'id' field from dataset
        batch_dataset_original_idx = [s['original_index'] for s in batch] # Original indices from dataset
        activations = [selected_token_vecs[j].unsqueeze(0).cpu() for j in range(selected_token_vecs.size(0))] # Embeddings of generated answers

       
        # --- Save progress to pickle after each batch ---
        batch_results = {
            "id": batch_dataset_ids,
            "original_indices": batch_dataset_original_idx,
            "gen_answers": batch_answers,
            "activations": activations
        }

        if save_to_pkl:
            append_to_pickle(output_path, batch_results)
        else:
            batch_activations.extend(activations)
        
    if not save_to_pkl:
        return batch_activations
