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
from typing import Dict, List, Any, Callable, Tuple, Union, Literal
import time

from src.evaluation.similarity_metrics import rouge_l_simScore, sentence_bert_simScore
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



def generate(
    model: PreTrainedModel,
    inputs: BatchEncoding,
    tokenizer: PreTrainedTokenizer,
    max_new_tokens: int = 50,
    k_beams: int = 1,
    **generate_kwargs
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Generate sequences from the model with optional beam search.
    Supports advanced options via **generate_kwargs (e.g., output_attentions).

    Parameters
    ----------
    model : PreTrainedModel
        The language model to use for generation.
    inputs : BatchEncoding
        Tokenized input prompts.
    tokenizer : PreTrainedTokenizer
        Tokenizer providing eos and pad token IDs.
    max_new_tokens : int, optional
        Maximum number of new tokens to generate.
    k_beams : int, optional
        Number of beams to use. If 1, uses sampling. If >1, beam search is enabled.
    **generate_kwargs : dict
        Additional keyword arguments passed to `model.generate()`.

    Returns
    -------
    Union[torch.Tensor, Dict[str, torch.Tensor]]
        - If k_beams == 1:
            Returns a tensor of generated token IDs: shape (batch_size, prompt_len + gen_len)
        - If k_beams > 1:
            Returns a dictionary with keys:
                - "sequences": the generated token IDs
                - "beam_indices": the beam path for each token
    """
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True    if k_beams == 1 else False,
            temperature=0.6   if k_beams == 1 else None,
            top_p=0.9         if k_beams == 1 else None,
            top_k=50          if k_beams == 1 else None,
            num_beams=k_beams,
            use_cache=True, 
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id, # Ensures clean padding (right padding using eos token)
            output_hidden_states=False,      # We rely on the hook to extract hidden states instead (more memory efficient)
            return_dict_in_generate=True,    # Needed for access to beam_indices when num_beams > 1
            early_stopping=False if k_beams == 1 else True, #Generation stops as soon as any sequence hits EOS, even if other candidates have not yet finished.
            **generate_kwargs                # For future flexibility (e.g., output_attentions, output_scores)
        )
        return outputs 



def build_generation_attention_mask(
    gen_ids: torch.Tensor,
    eos_token_id: int
) -> torch.Tensor:
    """
    Build an attention mask for the generated part of sequences, marking all tokens up to and 
    including the first EOS token as valid (True), and the rest as padding (False).

    Parameters
    ----------
    gen_ids : torch.Tensor
        Tensor of shape (batch_size, gen_len) containing only generated sequences.
    eos_token_id : int
        ID of the EOS token used for padding and stopping generation.

    Returns
    -------
    torch.Tensor
        Boolean tensor of shape (batch_size, gen_len), where True marks valid generated tokens.
    """
    batch_size, _ = gen_ids.shape

    # Extract only the generated tokens IDs (excluding the prompt part)
    gen_len = gen_ids.shape[1]

    # Create a boolean mask with True values where tokens equal to eos_token_id
    eos_mask = (gen_ids == eos_token_id) # Shape: (batch_size, gen_len)
    
    # Default eos position = gen_len (means: no eos -> whole sequence is valid)
    eos_positions = torch.full((batch_size,), gen_len, dtype=torch.long, device=gen_ids.device)

    # Find first eos position for sequences that have one
    any_eos = eos_mask.any(dim=1)  # Find which sequences actually contain at least one eos_token_id - Shape: (batch_size,)
    eos_positions[any_eos] = eos_mask[any_eos].float().argmax(dim=1) # argmax returns the 1st position where eos_token_id == True

    # Generate a position index tensor (e.g., [0, 1, ..., gen_len-1]), repeated for each batch item
    position_ids = torch.arange(gen_len, device=gen_ids.device).unsqueeze(0).expand(batch_size, -1) # Shape: (batch_size, gen_len)

    # Final generation attetion mask: True for all positions <= first eos (included)
    generation_attention_mask = position_ids <= eos_positions.unsqueeze(1) # Shape (batch_size, gen_len)

    return generation_attention_mask.int()



def align_generation_hidden_states(
    generation_activations: List[torch.Tensor],
    beam_indices: torch.Tensor = None,
    k_beams: int = 1
) -> torch.Tensor:
    """
    If k_beams > 1, aligns (extracts) the hidden states from `activations` that 
    correspond to the generated sequence selected by the beam search algorithm. 
    If k_beams == 1, returns stacked outputs for greedy/top-k decoding.

    Parameters
    ----------
    generation_activations : List[torch.Tensor]
        List of activation tensors per generation step. 
        activations = [prompt] + [gen_step_1, gen_step_2, ..., gen_step_49], if max_new_tokens=50.
        so generation_activations = [gen_step_1, gen_step_2, ..., gen_step_49] (without prompt),
        Each tensor has shape (batch_size * k_beams, seq_len, hidden_size) if k_beams > 1,
        or (batch_size, seq_len, hidden_size) if k_beams == 1.

    beam_indices : torch.Tensor
        Tensor of shape (batch_size, gen_len) indicating which beam was selected at each step.
        Need to be specified only if k_beams > 1. 

    k_beams : int
        Number of beams used during generation (1 = greedy/top-k, >1 = beam search).

    Returns
    -------
    torch.Tensor
        Aligned hidden states of shape (batch_size, gen_len (= max_new_tokens - 1), hidden_size).
    """
    
    gen_len = len(generation_activations) # gen_len = max_new_tokens - 1 from `model.generate()`
    hidden_size = generation_activations[0].shape[-1]
    batch_size = beam_indices.shape[0] if k_beams > 1 else generation_activations[0].shape[0]

    if k_beams > 1:
        # Allocate tensor for aligned hidden states for selected beams
        aligned_hidden_states = torch.zeros((batch_size, gen_len, hidden_size), dtype=generation_activations[0].dtype)
        
        # Align hidden states for the selected beams
        for step in range(gen_len):
            h = generation_activations[step]  # Shape: (batch_size * k_beams, seq_len, hidden_size)
            indices = beam_indices[:, step].to(h.device)  # Shape: (batch_size,)
            valid = indices >= 0
            if valid.any():
                # For each batch item, select the last generated hidden state at this step, for the selected beam sequence 
                aligned_hidden_states[valid, step, :] = h[indices[valid], -1, :] # Shape (batch_size_valid, hidden_size)
    else:
        # No beam alignment needed, output comes directly from top-k sampling
        # For each batch item, take the last generated hidden state at this step
        aligned_hidden_states = torch.stack(
            [h[:, -1, :] for h in generation_activations], dim=1
        ).to(generation_activations[0].device)  # Shape: (batch_size, gen_len, hidden_size)

    return aligned_hidden_states



def align_prompt_hidden_states(
    prompt_activations: torch.Tensor,
    k_beams: int
) -> torch.Tensor:
    """
    If k_beams > 1, aligns (extracts) the prompt hidden states to match the original batch size
    by removing beam duplication introduced for decoding. 
    If k_beams == 1, returns the prompt hidden states as-is.

    During generation with beam search (k_beams > 1), the encoder prompt is computed once per input 
    and duplicated `k_beams` times to initialize multiple decoding paths. This function removes 
    the beam-level redundancy from the prompt representations to restore a shape that matches 
    the actual number of input samples.

    Parameters
    ----------
    prompt_activations : torch.Tensor
        List of activation tensors per generation step. 
        activations = [prompt] + [gen_step_1, gen_step_2, ..., gen_step_49], if max_new_tokens=50.
        so prompt_activations = prompt,  only the prompt part
        Each tensor has shape (batch_size * k_beams, seq_len, hidden_size) if k_beams > 1,
        or (batch_size, seq_len, hidden_size) if k_beams == 1.
    
    k_beams : int
        Number of beams used during generation.

    Returns
    -------
    torch.Tensor
        Aligned prompt hidden states of shape (batch_size, prompt_len, hidden_size).
    """

    if k_beams == 1:
        return prompt_activations  # Already (batch_size, prompt_len, hidden_size)
    
    # If beam search: keep only the first beam per batch
    # Since beam search only applies to the generation part, the prompt 
    # is encoded once, then duplicated k_beams times to initialize generation.
    return prompt_activations[::k_beams]  # Take every k_beams-th item



def analyze_single_generation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    sample_idx: int = 0,
    build_prompt_fn: Callable[[str, str], str] = None,
    register_forward_activation_hook_fn: Callable = None,
    layer_idx: int = -1,
    extract_token_activations_fn: Callable = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze a single sample from the dataset through the full inference pipeline:
    - Build prompt
    - Run forward pass and capture hidden states
    - Extract token-level activations from a specific layer
    - Generate an answer
    - Decode output and compute similarity scores with ground-truth

    Also returns timing information for each processing stage.

    Parameters
    ----------
    model : PreTrainedModel
        The causal language model to evaluate (e.g., LLaMA).
    tokenizer : PreTrainedTokenizer
        The corresponding tokenizer.
    dataset : Dataset
        The input dataset.
    sample_idx : int
        Index of the sample to analyze.
    build_prompt_fn : Callable
        Function to build a prompt from context and question.
    register_forward_activation_hook_fn : Callable
        Function that registers a forward hook on the model during a forward pass. 
    layer_idx : int
        Index of the transformer layer to extract activations from (default -1: last layer).
    extract_token_activations_fn : Callable
        Function that selects and aggregates token-level activations.
    **kwargs :
        Additional keyword arguments passed to extract_token_activations_fn.

    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - prompt, generated and ground-truth answers
        - similarity scores (ROUGE-L, Sentence-BERT)
        - whether generation is deemed correct
        - activations and tensor shapes
        - timing breakdown
    """
    sample = dataset[sample_idx]

    print("========= Analyze one generation  =========")
    times = {}

    # ==============================
    # 1. Prompt construction
    # ==============================
    t0 = time.time()
    prompt = build_prompt_fn(sample["context"], sample["question"])
    answer = sample["answers"]['text']
    times['prompt_construction'] = time.time() - t0
    print(f"----- Prompt construction: {times['prompt_construction']:.3f} sec")

    # ==============================
    # 2. Tokenization
    # ==============================
    t1 = time.time()
    inputs = tokenizer(prompt, truncation=True, padding=True, return_tensors="pt").to(model.device)
    times['tokenization'] = time.time() - t1
    print(f"----- Tokenization: {times['tokenization']:.3f} sec")

    t2 = time.time()
    # ==============================
    # Capture hidden states with forward hook
    # ==============================
    # Hook to collect the hidden states after the forward pass
    captured_hidden = {}
    handle, call_counter = register_forward_activation_hook_fn(model, captured_hidden, layer_idx=layer_idx)

    # Pass inputs through the model. When the target layer is reached,
    # the hook executes and saves its output in captured_hidden.
    with torch.no_grad():
        _ = model(**inputs, return_dict=True)
    # Remove the hook to avoid memory leaks or duplicate logging
    handle.remove() 

    #print(f"Hook was called {call_counter['count']} times.")
    if "activations" not in captured_hidden:
        raise RuntimeError("Hook failed to capture activations.")

    layer_output = captured_hidden["activations"]  # Shape: (batch_size, seq_len, hidden_size)

    # ==============================
    # Extract token activations from captured layer
    # ==============================
    times['layer_output'] = time.time() - t2
    print(f"----- Token extraction with single forward pass: {times['layer_output']:.3f} sec")

    # Token activations extraction
    t3 = time.time()
    selected_token_vecs = extract_token_activations_fn(
        selected_layer=layer_output,
        attention_mask=inputs["attention_mask"],
        device=layer_output.device,
        **kwargs
    )
    times['token_activations'] = time.time() - t3

    # ==============================
    # Run generation
    # ==============================
    t4 = time.time()
    outputs = generate(model, inputs, tokenizer)
    outputs_ids = outputs.sequences
    times['generation'] = time.time() - t4
    print(f"----- Generation: {times['generation']:.3f} sec")

    # ==============================
    # Decode generated output
    # ==============================
    t5 = time.time()
    prompt_len = len(inputs["input_ids"][0])
    generated_answer_ids = outputs_ids[0][prompt_len:]
    generated_answer = tokenizer.decode(generated_answer_ids, skip_special_tokens=True).strip()
    times['decoding'] = time.time() - t5
    print(f"----- Decoding: {times['decoding']:.3f} sec")

    # ==============================
    # Compute similarity scores
    # ==============================
    t6 = time.time()
    rouge_l_score = rouge_l_simScore(generated_answer, answer) 
    sbert_sim = sentence_bert_simScore(generated_answer, answer)
    is_correct = (rouge_l_score >= 0.5) or (sbert_sim >= 0.4)
    times['similarity_scoring'] = time.time() - t6
    print(f"----- Similarity scoring: {times['similarity_scoring']:.3f} sec")

    # ==============================
    # Display results
    # ==============================
    print("\n=== Prompt ===")
    print(prompt)
    print("\n=== Shapes ===")
    print(f"Shape - number of tokens: {inputs['input_ids'].shape}")
    print(f"Shape - selected_layer: {layer_output.shape}")
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
        "layer_shape": layer_output.shape,
        "token_activations": selected_token_vecs,
    }



def run_filter_generated_answers_by_similarity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    batch_size: int = 4,
    idx_start_sample: int = 0,
    max_samples: int = 1000,
    output_path: str = "outputs/all_batch_results.pkl",
    build_prompt_fn: Callable[[str, str], str] = None,
) -> None:
    """
    Generates answers in batch using a decoder-only language model, evaluates their semantic 
    similarity against ground-truth answers using ROUGE-L and SBERT, and stores the results
    to a pickle file.

    An answer is considered correct if:
        - ROUGE-L ≥ 0.5, or
        - SBERT ≥ 0.4 (only computed if ROUGE-L < 0.5)

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
    """
    for i in tqdm(range(idx_start_sample, idx_start_sample + max_samples, batch_size)):
        
        # ==============================
        # Initialize batch containers
        # ==============================
        batch_answers = []               # Generated answers
        batch_gt_answers = []            # Ground-truth answers
        batch_is_correct = []            # 0/1 labels indicating correctness
        batch_dataset_ids = []           # 'id' field from dataset
        batch_dataset_original_idx = []  # Original indices from dataset
        batch_rouge_scores = []          # Rouge-L scores
        batch_sbert_scores = []          # Sentence-Bert scores

        # ==============================
        # Prepare input batch
        # ==============================
        batch = extract_batch(dataset, i, batch_size)
        prompts = [build_prompt_fn(s["context"], s["question"]) for s in batch]
        answers = [s["answers"]["text"] for s in batch]
        inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(model.device)

        # ==============================
        # Generate model predictions
        # ==============================
        outputs = generate(model, inputs, tokenizer)
        outputs_ids = outputs.sequences 
        
        for j in range(len(prompts)):
            # ==============================
            # Decode generated tokens
            # ==============================
            prompt_len = len(inputs["input_ids"][j]) # Length of prompt j
            generated_answer_ids = outputs_ids[j][prompt_len:] # Remove prompt prefix to isolate the generated answer
            generated_answer = tokenizer.decode(generated_answer_ids, skip_special_tokens=True).strip()

            # ==============================
            # Compute semantic similarity between model's answer and ground-truth
            # ==============================
            rouge_l_score = rouge_l_simScore(generated_answer, answers[j])
            if rouge_l_score >= 0.5:
                is_correct = True
                sbert_score = None
            else:
                sbert_score = sentence_bert_simScore(generated_answer, answers[j])
                is_correct = (sbert_score >= 0.4)

            # ==============================
            # Store per-example results
            # ==============================
            batch_dataset_ids.append(batch[j]['id'])
            batch_dataset_original_idx.append(batch[j]['original_index'])
            batch_answers.append(generated_answer)
            batch_gt_answers.append(answers[j])
            batch_is_correct.append(int(is_correct))
            batch_rouge_scores.append(rouge_l_score)
            batch_sbert_scores.append(sbert_score)

        # ==============================
        # Store results (to file)
        # ==============================
        batch_results = {
            "id": batch_dataset_ids,
            "original_indices": batch_dataset_original_idx,
            "gen_answers": batch_answers,
            "ground_truths": batch_gt_answers,
            "is_correct": batch_is_correct,
            "sbert_scores": batch_sbert_scores,
            "rouge_scores": batch_rouge_scores,
            "activations": [None] * len(batch_answers) 
            # empty list for 'activations' to be consistent with `append_to_pickle`
        }
        
        append_to_pickle(output_path, batch_results)
   


def run_prompt_activation_extraction(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    batch_size: int = 4,
    idx_start_sample: int = 0,
    max_samples: int = 1000,
    save_to_pkl: bool = False,
    output_path: str = "outputs/all_batch_results.pkl",
    build_prompt_fn: Callable[[str, str], str] = None,
    register_forward_activation_hook_fn: Callable = None,
    layer_idx: int = -1,  
    extract_token_activations_fn: Callable = None,
    **kwargs
) -> Union [Tuple[List[torch.Tensor]], None]:
    """
    Runs batched inference on a dataset using a decoder-only language model.
    For each batch, it extracts token-level hidden activations 
    (from the prompt only) from a specified transformer layer.

    Hidden states are captured via a forward hook during a single forward pass.
    These representations can be saved to a pickle file or returned directly.

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
    register_forward_activation_hook_fn : Callable
        Function that registers a forward hook on the model during a forward pass. 
    layer_idx : int
        Index of the transformer layer to extract activations from (default: -1 for last layer).
    extract_token_activations_fn : Callable
        Function that selects and aggregates token-level activations. 
    **kwargs :
        Extra keyword arguments passed to extract_token_activations_fn.
    
    Returns
    -------
    Union[
        Tuple[List[torch.Tensor],
        None
    ]
        If save_to_pkl is False:
            Returns batch_activations: list of length `num_samples`, each element is a tensor 
            of shape (1, hidden_size), containing the selected and aggragated token activations.
        If save_to_pkl is True:
            Returns None (activations are saved incrementally to output_path).
    """
    
    batch_activations = []  # Chosen prompt token activation vectors

    for i in tqdm(range(idx_start_sample, idx_start_sample + max_samples, batch_size)):
        
        # ==============================
        # Prepare input batch
        # ==============================
        batch = extract_batch(dataset, i, batch_size)
        prompts = [build_prompt_fn(s["context"], s["question"]) for s in batch]
        inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(model.device)

        # ==============================
        # Register forward hook to capture layer output
        # ==============================
        # Hook to collect the hidden states after the forward pass
        captured_hidden = {}
        handle, call_counter = register_forward_activation_hook_fn(model, captured_hidden, layer_idx=layer_idx)
        
        # ==============================
        # Run model forward pass (hook captures activations)
        # ==============================
        # Pass inputs through the model. When the target layer is reached,
        # the hook executes and saves its output in captured_hidden.
        with torch.no_grad():
            _ = model(**inputs, return_dict=True)
        # Remove the hook to avoid memory leaks or duplicate logging
        handle.remove() 
        
        #print(f"Hook was called {call_counter['count']} times.")
        if "activations" not in captured_hidden:
            raise RuntimeError("Hook failed to capture activations.")

        # ==============================
        # Extract token activations from captured layer
        # ==============================
        layer_output = captured_hidden["activations"]  # Shape: (batch_size, seq_len, hidden_size)

        selected_token_vecs = extract_token_activations_fn(
                selected_layer=layer_output , 
                attention_mask=inputs["attention_mask"], 
                device=layer_output.device,
                **kwargs)  # Shape (batch_size, hidden_size)
        
        # ==============================
        # Store results (to file or memory)
        # ==============================
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



def run_prompt_and_generation_activation_extraction(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    batch_size: int = 4,
    idx_start_sample: int = 0,
    max_samples: int = 1000,
    save_to_pkl: bool = False,
    output_path: str = "outputs/all_batch_results.pkl",
    build_prompt_fn: Callable[[str, str], str] = None,
    register_generation_activation_hook_fn: Callable = None,
    layer_idx: int = -1,  
    extract_token_activations_fn: Callable = None,
    activation_source: Literal["prompt", "generation", "promptGeneration"] = "generation",
    k_beams : int = 1,
    **kwargs
) -> Union[List[torch.Tensor], None]:
    """
    Runs batched inference on a dataset using a decoder-only language model.
    For each batch, it performs text generation and extracts token-level hidden activations 
    (both from the prompt and the generated text depending on `activation_source`) 
    from a specified transformer layer.

    Hidden states are captured via a forward hook during generation, then aligned and 
    filtered using attention masks. These representations can be saved to a pickle file 
    or returned directly.

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
    register_generation_activation_hook_fn : Callable
        Function that registers a forward hook on the model during autoregressive text generation.
    layer_idx : int
        Index of the transformer layer to extract activations from (default: -1 for last layer).
    extract_token_activations_fn : Callable
        Function that selects and aggregates token-level activations. 
    activation_source : {"prompt", "generation", "promptGeneration"}
        Which part of the sequence to extract activations from:
        - "prompt": only from the prompt
        - "generation": only from the generated answer
        - "promptGeneration": prompt and generation answer both concatenated
    k_beams : int, optional
        Number of beams for beam search during generation (default: 1). If 1, uses sampling. 
    **kwargs :
        Extra keyword arguments passed to extract_token_activations_fn.
    
    Returns
    -------
    Union[
        List[torch.Tensor],
        None
    ]
        If save_to_pkl is False 
            Returns batch_activations: list of length `num_samples`, each element is a tensor 
            of shape (1, hidden_size), containing the selected and aggragated token activations.
        If save_to_pkl is True:
            Returns None (activations are saved incrementally to output_path).
    """

    batch_activations = []  # Chosen token activation vectors

    for i in tqdm(range(idx_start_sample, idx_start_sample + max_samples, batch_size)):
        
        # ==============================
        # Prepare input batch
        # ==============================
        batch = extract_batch(dataset, i, batch_size)
        prompts = [build_prompt_fn(s["context"], s["question"]) for s in batch]
        inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1] # Assumes prompts are padded to same length

        # ==============================
        # Register forward hook to capture layer output
        # ==============================
        # This hook collects the hidden states at each decoding step
        # activations = [prompt] + [gen_step_1, gen_step_2, ..., gen_step_49], len(activations)=50, if max_new_tokens=50.
        activations = [] # activations[k] of Shape: (batch_size * k_beams, seq_len, hidden_size)
        handle, call_counter = register_generation_activation_hook_fn(model, activations, layer_idx=layer_idx)

        # ==============================
        # Run model forward pass (hook captures activations)
        # ==============================
        # Generate text from prompts using beam search or sampling. 
        # When the target layer is reached, the hook executes and saves its output in activations.
        outputs = generate(model, inputs, tokenizer, max_new_tokens=50, k_beams=k_beams)
        # Remove the hook to avoid memory leaks or duplicate logging
        handle.remove() 
        
        #print(f"Hook was called {call_counter['count']} times.")
        if len(activations)==0:
            raise RuntimeError("Hook failed to capture activations.")
        
        # Define prompt and generation hidden states 
        prompt_activations=activations[0]      # `[0]` to include only the prompt part 
        generation_activations=activations[1:] # `[1:]` to exclude the prompt part 

        # ===============================
        # Truncate activations to match real generation steps (cf. Understanding Note #1)
        # ===============================
        # During generation, the model may run extra forward passes (especially with beam search)
        # beyond the number of tokens in the final output. This results in activations being longer
        # than needed — we need to truncate them accordingly.
        # (see Understanding Note #1).
        if k_beams > 1:
            # In beam search, we use beam_indices.shape[1] to determine the actual number of generation steps
            gen_len = outputs.beam_indices.shape[1]
        else:
            # In greedy/top-k sampling, gen_len is simply the number of new tokens beyond the prompt
            gen_len = outputs.sequences.shape[1] - prompt_len

        # Sometimes, activations may include extra "ghost" steps (e.g., due to internal padding/sync in beam search)
        bool_truncate_activations = (len(generation_activations) >= gen_len) 
 
        if bool_truncate_activations:
            # Truncate extra steps to ensure alignment with generated tokens
            generation_activations = generation_activations[:gen_len]

        """
        ==================================
        Understanding Note #1:
        ==================================
        When using beam search in Hugging Face Transformers, the number of decoder hidden states
        (len(outputs.hidden_states)) can be greater than the number of tokens in the final generated 
        sequence (outputs.sequences[:,prompt_len:].shape[1] = outputs.beam_indices.shape[1]). 
        This happens because, during beam search, the model explores multiple candidate sequences 
        (beams) at each generation step and continues generating until a stopping condition is met 
        (such as all beams reaching EOS or the maximum number of tokens). But because beams can 
        finish at different steps (some hitting EOS early, others continuing), the model must keep
        generating for the remaining active beams. 
        *Note* that in our code, outputs.hidden_states and activations are the same. 
      
        Explanation from Hugging Face, January 2023: 
        (https://github.com/huggingface/transformers/issues/21374)
        "Beam Search: Here it's trickier. In essence, beam search looks for candidate outputs until it hits 
        a stopping condition. The candidate outputs can have fewer tokens than the total number of generation 
        steps -- for instance, in an encoder-decoder text model, if your input is How much is 2 + 2? and the 
        model generates as candidates <BOS>4<EOS> (3 tokens) and <BOS>The answer is potato<EOS> 
        (for argument's sake, 6 tokens) before deciding to stop, you should see sequences with shape [1, 3] 
        and decoder_hidden_states with length 5, because 5 tokens were generated internally before settling 
        on the 1st candidate."    
        """

        # ===============================
        # Truncate generated token IDs to match activations (cf. Understanding Note #2) 
        # ===============================
        # - When N tokens are generated, only the first N-1 tokens have corresponding hidden states.
        #   So activations[1:] covers only the first N-1 steps (cf. Understanding Note #2).
        #   Therefore, we exclude the last generated token from outputs.sequences and beam_indices
        #   to match activations[1:]
        # - Exception: if activations were truncated earlier (bool_truncate_activations = True),
        #   then we already lost activations of the final decoding step(s), and our activations[1:]
        #   only cover the available tokens. In that case, we keep the full `gen_len` to match.
        # (see Understanding Note #2)
        if bool_truncate_activations:
            expected_gen_len = gen_len  # All generated tokens have hidden states
        else: 
            expected_gen_len  = gen_len - 1 # Drop final token to match activations[1:]

        # Truncate generated sequences and beam paths accordingly
        truncated_gen_sequences = outputs.sequences[:, prompt_len : prompt_len + expected_gen_len]
        if k_beams > 1:
            truncated_beam_indices = outputs.beam_indices[:, :expected_gen_len] 

        """
        ==================================
        Understanding Note #2:
        ==================================
        When using model.generate() with output_hidden_states=True (what we are replicating here with the hook),
        use_cache=True and max_new_tokens=30, there is always an offset between the length of the 
        generated sequence (outputs.sequences.shape[1][prompt_len:]) and the length of len(outputs.hidden_states) : 
        * outputs.sequences.shape[1] = prompt_len (17) + max_new_tokens (30) = 47
        * len(outputs.hidden_states) = max_new_tokens (30)
            With : 
            * outputs.hidden_states[0][layer_idx].shape = (batch_size, prompt_len, hidden_size)           --> includes the prompt ! 
            * outputs.hidden_states[i][layer_idx].shape = (batch_size, 1, hidden_size) with 1 <= i <= 29  --> stops at 29 ! 
        *Note* that in our code, outputs.hidden_states and activations are the same. 
            
        Explanation from Hugging Face, April 2024 
        (https://github.com/huggingface/transformers/issues/30036):
        "If you have 30 tokens at the end of generation, you'll always have 29 hidden states.
        The token with index N is used to produce hidden states with index N, which is then used 
        to get the token with index N+1. The generation ends as soon as the target number of 
        tokens is obtained so, when we obtain the 30th token, we don't spend compute to get the 30th 
        set of hidden states. You can, however, manually run an additional forward pass to obtain the 
        30th set of hidden states, corresponding to the 30th token and used to obtain the 31st token.
        """

        # ===============================
        # Align generated and prompt hidden states
        # ===============================
        # Extract the hidden states that correspond to the generated sequence
        # selected by the beam search (or top-k sampling if k_beams = 1)
        aligned_generation_hidden_states = align_generation_hidden_states(
            generation_activations=generation_activations, 
            beam_indices=truncated_beam_indices if k_beams < 1 else None,
            k_beams=k_beams
        ) # Shape: (batch_size, gen_len, hidden_size)

        # Extract the hidden states that correspond to the prompt
        aligned_prompt_hidden_states = align_prompt_hidden_states(
            prompt_activations=prompt_activations, 
            k_beams=k_beams
        ) # Shape: (batch_size, prompt_len, hidden_size)

        # Concatenate the prompt and generation aligned hidden states  
        aligned_prompt_and_gen_hidden_states = torch.cat(
            [aligned_prompt_hidden_states, 
             aligned_generation_hidden_states], 
             dim=1
        ) # Shape: (batch_size, prompt_len + gen_len, hidden_size)

        # ===============================
        # Build generation and prompt attention mask
        # ===============================
        # This mask marks which generated tokens are valid (i.e., not padding).
        # Positions are marked True up to and including the first eos_token_id
        generation_attention_mask = build_generation_attention_mask(
            gen_ids=truncated_gen_sequences, 
            eos_token_id=tokenizer.eos_token_id
        ) # Shape (batch_size, gen_len)

        # Prompt attention mask
        prompt_attention_mask = inputs["attention_mask"] 
        # Shape (batch_size, prompt_len)
        
        # Concatenate the prompt and generation attention mask
        prompt_and_gen_attention_mask = torch.cat(
            [prompt_attention_mask,
            generation_attention_mask],
            dim=1
        ) # Shape (batch_size, prompt_len + gen_len)

        # ==============================
        # Extract token activations from captured layer, based on source
        # ==============================
        if activation_source == "generation":
            # Return only the token activations from the generated answer 
            selected_token_vecs = extract_token_activations_fn(
                    selected_layer=aligned_generation_hidden_states, 
                    attention_mask=generation_attention_mask, 
                    device=aligned_generation_hidden_states.device,
                    **kwargs) # Shape (batch_size, hidden_size)
            
        elif activation_source == "prompt":    
            # Return only the token activations from the prompt
            selected_token_vecs = extract_token_activations_fn(
                    selected_layer=aligned_prompt_hidden_states, 
                    attention_mask=prompt_attention_mask, 
                    device=aligned_prompt_hidden_states.device,
                    **kwargs) # Shape (batch_size, hidden_size)
            
        elif activation_source == "promptGeneration":
            # Return token activations from the concatenated prompt + generated answer 
            selected_token_vecs = extract_token_activations_fn(
                    selected_layer=aligned_prompt_and_gen_hidden_states, 
                    attention_mask=prompt_and_gen_attention_mask, 
                    device=aligned_prompt_and_gen_hidden_states.device,
                    **kwargs) # Shape (batch_size, hidden_size)

        else:
            raise ValueError(
                f"Invalid value for `activation_source`: '{activation_source}'. "
                f"Expected one of: ['prompt', 'generation', 'promptGeneration']."
            )    
        
        # ==============================
        # Store results (to file or memory)
        # ==============================
        activations = [selected_token_vecs[j].unsqueeze(0).cpu() for j in range(selected_token_vecs.size(0))]

        batch_dataset_ids = [s['id'] for s in batch]
        batch_dataset_original_idx = [s['original_index'] for s in batch]
        
        batch_results = {
            "id": batch_dataset_ids,
            "original_indices": batch_dataset_original_idx,
            "activations": activations,
        }

        if save_to_pkl:
            append_to_pickle(output_path, batch_results)
        else:
            batch_activations.extend(activations)
        
    if not save_to_pkl:
        return batch_activations