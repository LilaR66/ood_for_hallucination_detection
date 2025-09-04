#!/usr/bin/env python3
"""
============================================================
Batched Inference and Activation/Attention/Logit Extraction Utilities for LLM QA
============================================================

This module provides functions for running batched inference with decoder-only 
language models such as LLaMA on question answering datasets. It supports construction of 
question prompts, text generation, extraction of token-level activations, attentions and logits, 
semantic evaluation of generated answers (ROUGE-L, SBERT), and incremental batch-wise saving 
of results.

Key features:
-------------
- `analyze_single_generation`: [DEBUG function] 
    Process one sample—construct prompt, pass through model, extract activations, generate answer,
    score with ROUGE-L and SBERT, return detailed outputs and timing.

- `run_filter_generated_answers_by_similarity`: 
    - Process a batch of samples
    - Generate answers, 
    - Compute similarity scores (ROUGE-L and Sentence-BERT) between generated and ground-truth answers,
    - Filter and label generated answers as correct/incorrect based on similarity thresholds,
    - Save results incrementally in batch-wise pickle files.

- `run_prompt_score_extraction`: 
    - For each batch, register and use forward hooks to capture token-level hidden states (activations)
     or attention maps at specified transformer layers.
    - Extract and aggregate activations using configurable modes (average, max, last, etc.).
    - Optional scoring on logit and attention tensors (perplexity, entropy, and attention eigenvalue scores).
    - Supports configurable masking and offset handling for precise score extraction regions.
    - Save results, including generated answers and scores, incrementally in batch-wise pickle files.

- `run_prompt_and_generation_score_extraction`: 
    - For each batch, register and use forward hooks to capture token-level hidden states (activations)
     or attention maps at specified transformer layers.
    - Captures activations/attentions/logits from both from the prompt and the generated text depending
     on `activation_source`
    - Extract and aggregate activations using configurable modes (average, max, last, etc.).
    - Optional scoring on logit and attention tensors (perplexity, entropy, and attention eigenvalue scores).
    - Supports configurable masking and offset handling for precise score extraction regions.
    - Save results, including generated answers and scores, incrementally in batch-wise pickle files.
"""


from transformers import PreTrainedTokenizer, PreTrainedModel
import torch
from datasets import  Dataset
from tqdm import tqdm
from typing import Dict, List, Any, Callable, Union, Literal, Optional
import time

from src.answer_similarity.similarity_metrics import rouge_l_simScore, sentence_bert_simScore
from src.data_reader.pickle_io import save_batch_pickle
from src.inference.generation_utils import (
    generate, 
    extract_batch,
    build_generation_attention_mask,
    apply_logit_lens,
)
from src.inference.hooks import (
    patched_LlamaAttention_forward,
    register_activation_hook,
    register_attention_hook,
    verify_call_counters,
    )
from src.inference.compute_descriptors import (
    extract_token_activations,
    compute_attn_eig_prod,
    compute_perplexity,
    compute_logit_entropy,
)
from src.inference.offset_utils import (
    compute_offset_attention_mask
)


def analyze_single_generation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    sample_idx: int = 0,
    layers: List[int] = [-1],
    build_prompt_fn: Callable[[str, str], str] = None,
) -> Dict[str, Any]:
    """
    Analyze a single sample from the dataset through the full inference pipeline:
    - Build prompt
    - Run forward pass and capture hidden states
    - Extract token-level hidden states and attention maps from a specific layer 
        with custom hooks
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
    layers : int
        List of indices of the transformer layers to extract hidden states and 
        attention maps from (default: [-1] for last layer).

    Notes
    -----
    - Expects dataset fields: "context", "question", "answers" (with key "text").
    - Restores original attention forwards after generation to avoid side effects.
    """
    # ---------------------------
    # Helpers for pretty printing
    # ---------------------------
    def hr(ch: str = "─", n: int = 80) -> None:
        print(ch * n)

    def title(s: str) -> None:
        hr("═"); print(s); hr("═")

    def section(s: str) -> None:
        print(); hr(); print(s); hr()

    def shape_of(x) -> str:
        shp = getattr(x, "shape", None)
        return str(tuple(shp)) if shp is not None else "N/A"
    
    # ---------------------------
    # Setup
    # ---------------------------
    times: Dict[str, float] = {}
    def tic(k: str): times[k] = time.time()
    def toc(k: str): times[k] = time.time() - times[k]

    sample = dataset[sample_idx]
    context = sample["context"]
    question = sample["question"]
    answer = sample["answers"]["text"]

    title("Analyze single generation")
    # --------------------------- 
    # Replace LlamaAttention.forward on target layers by
    # custom module to extract attention weights
    # ---------------------------
    for idx in layers:  
        model.model.layers[idx].self_attn.forward = patched_LlamaAttention_forward.__get__(
            model.model.layers[idx].self_attn,
            model.model.layers[idx].self_attn.__class__
    )

    # ---------------------------
    # Prompt
    # ---------------------------
    tic("prompt_construction")
    prompt = build_prompt_fn(context, question)
    toc("prompt_construction")
    print(f"> Prompt construction: {times['prompt_construction']:.3f}s")
    print(f"> Prompt preview:\n\n{prompt}\n\n")

    # ---------------------------
    # Tokenization
    # ---------------------------
    tic("tokenization")
    inputs = tokenizer(prompt, truncation=True, padding=True, return_tensors="pt").to(model.device)
    toc("tokenization")
    prompt_ids = inputs["input_ids"]
    prompt_len = prompt_ids.shape[1]
    print(f"> Tokenization: {times['tokenization']:.3f}s")
    print(f"> Prompt shape: {tuple(prompt_ids.shape)} (batch_size, prompt_len)")

    # ---------------------------
    # Capture hidden states and attention maps with forward hook
    # ---------------------------
    # Hook to collect the hidden states after the forward pass
    activations_lists = [[] for _ in layers]  # one empty list per layer 
    handle_act, call_counter_act = register_activation_hook(model, activations_lists, layers)
    # Hook to collect attention maps after the forward pass
    attentions_lists = [[] for _ in layers]  # one empty list per layer
    handle_attn, call_counter_attn = register_attention_hook(model, attentions_lists, layers)

    # ---------------------------
    # Generation
    # ---------------------------
    tic("generation")
    outputs = generate(model, inputs, tokenizer, max_new_tokens=50, k_beams=1)
    gen_ids = outputs.sequences[:, prompt_len:]
    toc("generation")
    print(f"> Generation: {times['generation']:.3f}s")
    print(f"> Generated output shape: {tuple(gen_ids.shape)} (batch_size, gen_len)")

    # Remove the hooks to avoid memory leaks or duplicate logging
    for h in handle_act:  h.remove()
    for h in handle_attn: h.remove()

    # ---------------------------
    # Decode
    # ---------------------------
    tic("decoding")
    generated_answer = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
    toc("decoding")
    print(f"> Decoding: {times['decoding']:.3f}s")

    # ---------------------------
    # Similarity scoring
    # ---------------------------
    tic("similarity_scoring")
    rouge_l_score = rouge_l_simScore(generated_answer, answer)
    sbert_sim = sentence_bert_simScore(generated_answer, answer)
    is_correct = (rouge_l_score >= 0.5) or (sbert_sim >= 0.4)
    toc("similarity_scoring")
    print(f"> Scoring time: {times['similarity_scoring']:.3f}s")

    # ---------------------------
    # Results
    # ---------------------------
    section("Results")
    print("> Generated Answer:")
    print(generated_answer)
    print("\n> Ground-truth Answer:")
    print(answer)
    print("\n> Scores:")
    print(f"ROUGE-L F1: {rouge_l_score:.4f}")
    print(f"Sentence-BERT Cosine Similarity: {sbert_sim:.4f}")
    print(f"Is generated answer correct: {is_correct}")

    # ---------------------------
    # Timing summary
    # ---------------------------
    section("Timing summary (seconds)")
    for k in ("prompt_construction", "tokenization", "generation", "decoding", "similarity_scoring"):
        if k in times:
            print(f"> {k:>22}: {times[k]:.3f}")

    # ---------------------------
    # Hook prints
    # ---------------------------
    section("Hook capture summary\nHook retrieved: Hidden States in `activations_lists` and Attention Maps in `attentions_lists`")
    print("> Transformer layers to retrive hidden states/attention maps from:", layers)
    selected_layer = layers[-1]
    try:
        pos = layers.index(selected_layer)  # position within our lists
    except ValueError:
        pos = len(layers) - 1

    print("\n> Hidden States (Activations) Summary for layer {selected_layer}:")
    print(f"  - Number of analyzed layers [len(activations_lists)]: {len(activations_lists)}")
    print(f"  - Number of hidden states for layer {selected_layer} [len(activations_lists[{selected_layer}])]: {len(activations_lists[pos])}")

    if len(activations_lists[pos]) > 0:
        h0 = activations_lists[pos][0]
        print(f"  - First hidden state corresponds to the prompt:")
        print(f"    activations_lists[{selected_layer}][0].shape = {shape_of(h0)}")
        print("    Shape format: (batch_size, prompt_len, hidden_size)")

    if len(activations_lists[pos]) > 1:
        h1 = activations_lists[pos][1]
        print(f"  - Subsequent hidden states correspond to the generation tokens:")
        print(f"    activations_lists[{selected_layer}][i].shape for i in 1..gen_len-1 = {shape_of(h1)}")
        print("    Shape format: (batch_size, 1, hidden_size)")
        print("    NOTE: The model does not return the hidden state for the last generated token.")

    print(f"\n> Attention Maps Summary for layer {selected_layer}:")
    print(f"  - Number of analyzed layers [len(attentions_lists)]: {len(attentions_lists)}")
    print(f"  - Number of attention maps [len(attentions_lists[{selected_layer}])]: {len(attentions_lists[pos])}")

    if len(attentions_lists[pos]) > 0:
        a0 = attentions_lists[pos][0]
        print(f"  - First attention map corresponds to the prompt:")
        print(f"    attentions_lists[{selected_layer}][0].shape = {shape_of(a0)}")
        print("    Shape format: (batch_size, num_attention_heads, prompt_len, hidden_size)")

    if len(attentions_lists[pos]) > 1:
        a1 = attentions_lists[pos][1]
        print(f"  - Subsequent attention maps correspond to the generation tokens:")
        print(f"    attentions_lists[{selected_layer}][i].shape for i in 1..gen_len-1 = {shape_of(a1)}")
        print("    Shape format: (batch_size, num_attention_heads, 1, hidden_size)")
        print("    NOTE: The model does not return the attention map for the last generated token.")



def run_filter_generated_answers_by_similarity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    batch_size: int = 4,
    idx_start_sample: int = 0,
    max_samples: int = 1000,
    output_path: str = "outputs/all_batch_results",
    build_prompt_fn: Callable[[str, str], str] = None,
) -> None:
    """
    Generates answers in batch using a decoder-only language model, evaluates their semantic 
    similarity against ground-truth answers using ROUGE-L and SBERT.
    The results are saved as individual batch files in a specified pickle directory, 
    allowing efficient incremental storage and later aggregation.

    An answer is considered correct if:
        - ROUGE-L >= 0.5, or
        - SBERT >= 0.4 (only computed if ROUGE-L < 0.5)

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
        Path to the directory where extracted answers will be saved as individual pickle batch files.
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
        }
         
        save_batch_pickle(batch_data=batch_results, output_dir=output_path, batch_idx=i)
   


def run_prompt_descriptor_extraction(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    batch_size: int = 4,
    idx_start_sample: int = 0,
    max_samples: int = 1000,
    save_to_pkl: bool = False,
    output_path: str = "outputs/all_batch_results.pkl",
    build_prompt_fn: Callable[[str, str], str] = None,
    layers: List[int] = [-1],  
    hidden_agg: List[str] = ["avg_emb", "last_emb", "max_emb", "first_gen_emb", "hidden_score", "feat_var_emb"],
    attn_agg: List[str] = ["attn_score"],
    logit_agg: List[str] = ["perplexity_score", "logit_entropy_score", "window_logit_entropy_score"],
    logit_config: dict = {"top_k": 50, "window_size": 1, "stride": 1},
    start_offset : int = 0,
    end_offset : int = 0,
) -> Union[List[torch.Tensor], None]:
    """
    Runs batched inference on a dataset using a decoder-only language model.
    For each batch, it runs a forward pass on the prompt and extracts token-level hidden 
    activations, attention maps and logit descriptors from specified transformer layers.

    The function supports multiple descriptor aggregation modes for the activations (`hidden_agg`),
    attentions (`attn_agg`), and logits (`logit_agg`). The `logit_config` argument provides 
    configuration parameters for logit-based score functions.
    
    Hidden states and attention maps are captured via forward hooks, 
    then aggregated based on token position and attention masks.
    
    These activations are saved as individual batch files in a specified pickle directory, 
    allowing efficient incremental storage and later aggregation.
    Alternatively, the representations can be returned directly.

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
        Path to the directory where extracted answers will be saved as individual pickle batch files.
    build_prompt_fn : Callable
        Function to build a prompt from context and question.
    layers : List[int]
        List of indices of the transformer layers to extract activations from (default: [-1] for last layer).
    hidden_agg : List[str], optional
        List of aggregation modes to compute on token activations. Possible modes include:
            "avg_emb", "last_emb", "max_emb", "first_gen_emb", "hidden_score", "feat_var_emb".
        These modes are passed to `extract_token_activations` for aggregation. Default includes the above.
    attn_agg : List[str], optional
        List of attention-based descriptors to compute. Supported: "attn_score".
    logit_agg : List[str], optional
        List of logit-based descriptors to compute. Supported:
            "perplexity_score", "logit_entropy_score", "window_logit_entropy_score".
    logit_config : dict, optional
        Configuration dictionary for logit-based descriptors functions, with keys such as:
            - "top_k": int, number of top logits considered (default 50)
            - "window_size": int, window size for windowed entropy (default 1)
            - "stride": int, stride for windowed entropy (default 1)
    start_offset : int
        Offset from the first non-padding token (must be >= 0). 
    end_offset : int
        Offset from the last non-padding token (must be <= 0, e.g., -3 to remove 3 tokens).
    
    Returns
    -------
     Union[List[dict], None]
        If `save_to_pkl` is False, returns a list of dictionaries, one per batch, with each element
         of the list having the following structure:
            {
                "id": List[str],  # IDs of batch samples
                "original_indices": List[int],  # Original dataset indices
                "context": List[str],
                "question": List[str],
                "gt_answers": List[str],        # Ground-truth reference answers
                "descriptors": {
                    "layer_{layer_idx}": {
                        "hidden": { 
                            "{mode}": np.ndarray[(batch_size, hidden_size), float],
                             # for mode in {'avg_emb','last_emb','max_emb','first_gen_emb','feat_var_emb'}.
                            "{mode}": np.ndarray[(batch_size,), float],
                             # if mode=='hidden_score' 
                            ... 
                        },
                        "attention": {
                            "{attn_score}": np.ndarray[(batch_size,), float],  
                            ...
                        }
                        "logit": {
                            "perplexity_score": np.ndarray[(batch_size,), float],
                            "logit_entropy_score": np.ndarray[(batch_size,), float],
                            "window_logit_entropy_score": np.ndarray[(batch_size,), float] 
                        }
                    },
                    
                }
            },

        If `save_to_pkl` is True, saves each batch's dictionary incrementally to disk and returns None.
    """

    # ==============================================================  
    # [PATCH] Replace LlamaAttention.forward on target layers by
    #  custom module to extract attention weights
    # ==============================================================
    for idx in layers:  
        model.model.layers[idx].self_attn.forward = patched_LlamaAttention_forward.__get__(
            model.model.layers[idx].self_attn,
            model.model.layers[idx].self_attn.__class__
    )
        
    # ==============================================================  
    # [LOOP] Process batches of examples  
    # ==============================================================
    all_batch_results = []  

    for i in tqdm(range(idx_start_sample, idx_start_sample + max_samples, batch_size)):
      
        # ----------------------------------------------------------
        # [BATCH INPUT] Extract and tokenize prompts
        # ----------------------------------------------------------
        batch = extract_batch(dataset, i, batch_size)
        prompts = [build_prompt_fn(s["context"], s["question"]) for s in batch]
        inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(model.device)
        prompt_ids = inputs["input_ids"] # (batch_size, prompt_len)
        prompt_attention_mask = inputs["attention_mask"] 

        # ----------------------------------------------------------
        # [HOOKS] Register hooks to capture hidden states and attentions
        # The activations/attention retrieved by the hooks are have similar values 
        # as the ones from `output_hidden_states=True`/`output_attentions=True` in `model.generate()`
        # ----------------------------------------------------------
        # This hook collects the hidden states. For layer l: 
        # activations_lists[l] = [act_prompt], 
        # activations_lists[l][0] of shape: (batch_size, prompt_len, hidden_size) 
        activations_lists = [[] for _ in layers]  # one empty list per layer 
        handle_act, call_counter_act = register_activation_hook(model, activations_lists, layers)

        # This hook collects the activations at each decoding step. For layer l: 
        # attentions_lists[l] = [attn_prompt], 
        # activations_lists[l][0] of shape: (batch_size, n_heads, prompt_len, prompt_len)
        attentions_lists = [[] for _ in layers]  # one empty list per layer
        handle_attn, call_counter_attn = register_attention_hook(model, attentions_lists, layers)
        
        # ----------------------------------------------------------
        # [FOWARD PASS] Run model with hooks to capture intermediate states
        # ----------------------------------------------------------
        # Pass inputs through the model. When the target layer is reached,
        # the hook executes and saves its output in captured_hidden.
        with torch.no_grad():
            outputs = model(**inputs, return_dict=True, return_logits=True)
        
        # Remove hooks to avoid memory leaks or duplicate logging
        for h in handle_act: h.remove()
        for h in handle_attn: h.remove()
        
        # Verify that hooks worked properly
        verify_call_counters(call_counter_act, name="activation hooks")
        verify_call_counters(call_counter_attn, name="attention hooks")


        # ----------------------------------------------------------
        # [OFFSET] Modify prompt mask with offset, if specified
        # ----------------------------------------------------------
        if start_offset !=0 or end_offset !=0:
            prompt_attention_mask, start_indices, end_indices = compute_offset_attention_mask(
                attention_mask=prompt_attention_mask, 
                start_offset=start_offset, 
                end_offset=end_offset
            ) # (batch_size, prompt_len), (batch_size,), (batch_size,)


        # **********************************************************
        # [LAYER LOOP] Extract activation and attention-based descriptors for each specified layer 
        # **********************************************************
        save_layers_descriptors = {}

        for l, layer_idx in enumerate(layers):

            activations = activations_lists[l]
            attentions = attentions_lists[l]

            # Define prompt and generation hidden states 
            prompt_activations=activations[0]    
            
            # Define prompt and generation attention maps
            prompt_attentions=attentions[0]        

            # ------------------------------------------------------
            # [HIDDEN DESCRIPTORS] Extract token-level activations/hidden-states
            # ------------------------------------------------------
            if hidden_agg is not None and len(hidden_agg) > 0:
                # Return only the token activations from the prompt
                selected_token_vecs = extract_token_activations(
                        selected_layer=prompt_activations, 
                        attention_mask=prompt_attention_mask, 
                        device=prompt_activations.device,
                        modes=hidden_agg,
                    ) # (batch_size, hidden_size)
 
                # Save results to dict
                hidden_results = {}
                for mode in hidden_agg:
                    if mode in selected_token_vecs:
                        hidden_results[mode] = selected_token_vecs[mode].cpu().numpy()
                save_layers_descriptors.setdefault(f"layer_{layer_idx}", {}).update({"hidden": hidden_results})

            # ------------------------------------------------------
            # [ATTENTION DESCRIPTORS] Extract attention eigenvalue-based metric
            # ------------------------------------------------------
            if attn_agg is not None and 'attn_score' in attn_agg:
                attn_eig_prod = compute_attn_eig_prod(
                        prompt_attentions=prompt_attentions, 
                        generation_attentions=None,
                        prompt_attention_mask=prompt_attention_mask, 
                        generation_attention_mask=None,
                        mode='prompt',
                )
                # Save results to dict
                save_layers_descriptors.setdefault(f"layer_{layer_idx}", {}).update({"attention": {"attn_score": attn_eig_prod}}) 


            # ------------------------------------------------------
            # [LOGIT DESCRIPTORS] Compute metrics from model logits
            # ------------------------------------------------------
            if logit_agg is not None and len(logit_agg) > 0: 
                logits_results = {}
                
                # If this is not the last layer, the only way to compute logits is from logitLens and activations
                if layer_idx != -1 and layer_idx != model.config.num_hidden_layers-1:
                    with torch.no_grad():
                        prompt_logits = apply_logit_lens(model, prompt_activations) # (batch, prompt_len, vocab_size)
                
                # If this is the last layer, compute logits using 'right way of computing logits'. 
                # There are small differences when computing prompt activations from forward pass and prompt activations
                # from model.generate() resulting in slightly different logits.
                else: #last layer
                    prompt_logits = outputs.logits # (batch, prompt_len, vocab_size)
                
                if 'perplexity_score' in logit_agg:
                    perplexity = compute_perplexity(
                        prompt_logits=prompt_logits, 
                        gen_logits=None,
                        prompt_ids=prompt_ids, 
                        gen_ids=None,
                        prompt_attention_mask=prompt_attention_mask,
                        gen_attention_mask=None,
                        prepend_last_prompt_logit=False,
                        mode='prompt',
                        min_k=None
                    )
                    # Save results to dict
                    logits_results['perplexity_score'] = perplexity 

                if 'logit_entropy_score' in logit_agg:
                    if logit_config is None:
                        raise ValueError("logit_entropy_score is required but logit_config is None")
                    logit_entropy = compute_logit_entropy(
                        prompt_logits=prompt_logits,
                        gen_logits=None,
                        prompt_attention_mask=prompt_attention_mask,
                        gen_attention_mask=None,
                        mode='prompt',
                        prepend_last_prompt_logit=False,
                        top_k=logit_config['top_k'], 
                        window_size=None,
                        stride=None
                    )
                    # Save results to dict
                    logits_results['logit_entropy_score'] = logit_entropy 
        
                if 'window_logit_entropy_score' in logit_agg:
                    if logit_config is None:
                        raise ValueError("window_logit_entropy_score is required but logit_config is None")
                    window_logit_entropy = compute_logit_entropy(
                        prompt_logits=prompt_logits,
                        gen_logits=None,
                        prompt_attention_mask=prompt_attention_mask,
                        gen_attention_mask=None,
                        mode='prompt',
                        prepend_last_prompt_logit=False,
                        top_k=logit_config['top_k'],
                        window_size=logit_config['window_size'], 
                        stride=logit_config['stride'] 
                    )
                    # Save results to dict
                    logits_results['window_logit_entropy_score'] = window_logit_entropy 
                    
                if logits_results:
                    save_layers_descriptors.setdefault(f"layer_{layer_idx}", {}).update({"logit": logits_results})

        # **********************************************************
        # [END LAYER LOOP] 
        # **********************************************************

        # ==========================================================
        # [OUTPUT] Store extracted results (to memory or file)
        # ==========================================================
        batch_results = {
            "id": [s['id'] for s in batch],
            "original_indices": [s['original_index'] for s in batch],
            "context": [s['context'] for s in batch],
            "question": [s['question'] for s in batch],
            "gt_answers": [s['answers'] for s in batch],
            "descriptors": {**save_layers_descriptors}
        }

        from src.data_reader.pickle_io import save_batch_pickle

        if save_to_pkl:
            save_batch_pickle(batch_data=batch_results, output_dir=output_path, batch_idx=i)
        else:
            all_batch_results.append(batch_results)

    if not save_to_pkl:
        return all_batch_results



def run_prompt_and_generation_descriptor_extraction(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    batch_size: int = 4,
    idx_start_sample: int = 0,
    max_samples: int = 1000,
    save_to_pkl: bool = False,
    output_path: str = "outputs/all_batch_results.pkl",
    build_prompt_fn: Callable[[str, str], str] = None,
    layers: List[int] = [-1],  
    activation_source: Literal["prompt", "generation", "promptGeneration"] = "generation",
    hidden_agg: List[str] = ["avg_emb", "last_emb", "max_emb", "first_gen_emb", "hidden_score", "feat_var_emb"],
    attn_agg: List[str] = ["attn_score"],
    logit_agg: List[str] = ["perplexity_score", "logit_entropy_score", "window_logit_entropy_score"],
    logit_config: dict = {"top_k": 50, "window_size": 1, "stride": 1},
    start_offset : int = 0,
    end_offset : int = 0,
) -> Union[List[torch.Tensor], None]:
    """
    Runs batched inference on a dataset using a decoder-only language model.
    For each batch, it performs text generation and extracts token-level 
    hidden activations, attention maps and logit descriptors from specified transformer layers.
    (both from the prompt and the generated text depending on `activation_source`) 

    The function supports multiple descriptor aggregation modes for the activations (`hidden_agg`),
    attentions (`attn_agg`), and logits (`logit_agg`). The `logit_config` argument provides 
    configuration parameters for logit-based score functions.
    
    Hidden states and attention maps are captured via forward hooks during generation, 
    then aggregated based on token position and attention masks.
    
    These activations are saved as individual batch files in a specified pickle directory, 
    allowing efficient incremental storage and later aggregation.
    Alternatively, the representations can be returned directly.

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
        Path to the directory where extracted answers will be saved as individual pickle batch files.
    build_prompt_fn : Callable
        Function to build a prompt from context and question.
    layers : List[int]
        List of indices of the transformer layers to extract activations from (default: [-1] for last layer).
    activation_source : {"prompt", "generation", "promptGeneration"}
        Which part of the sequence to extract activations/attentions/logits from:
        - "prompt": only from the prompt
        - "generation": only from the generated answer
        - "promptGeneration": prompt and generation answer both concatenated
    hidden_agg : List[str], optional
        List of aggregation modes to compute on token activations. Possible modes include:
            "avg_emb", "last_emb", "max_emb", "first_gen_emb", "hidden_score", "feat_var_emb".
        These modes are passed to `extract_token_activations` for aggregation. Default includes the above.
    attn_agg : List[str], optional
        List of attention-based descriptors to compute. Supported: "hidden_score".
    logit_agg : List[str], optional
        List of logit-based descriptors to compute. Supported:
            "perplexity_score", "logit_entropy_score", "window_logit_entropy_score".
    logit_config : dict, optional
        Configuration dictionary for logit-based descriptors functions, with keys such as:
            - "top_k": int, number of top logits considered (default 50)
            - "window_size": int, window size for windowed entropy (default 1)
            - "stride": int, stride for windowed entropy (default 1)
    start_offset : int
        Offset from the first non-padding token (must be >= 0). 
    end_offset : int
        Offset from the last non-padding token (must be <= 0, e.g., -3 to remove 3 tokens).
    
    Returns
    -------
    Union[List[dict], None]
        If `save_to_pkl` is False, returns a list of dictionaries, one per batch, with each element
         of the list having the following structure:
            {
                "id": List[str],  # IDs of batch samples
                "original_indices": List[int],  # Original dataset indices
                "context": List[str],
                "question": List[str],
                "gt_answers": List[str],        # Ground-truth reference answers
                "gen_answers": List[str],       # Generated model answers
                "descriptors": {
                    "layer_{layer_idx}": {
                        "hidden": { 
                            "{mode}": np.ndarray[(batch_size, hidden_size), float],
                             # for mode in {'avg_emb','last_emb','max_emb','first_gen_emb','feat_var_emb'}.
                            "{mode}": np.ndarray[(batch_size,), float],
                             # if mode=='hidden_score' 
                            ... 
                        },
                        "attention": {
                            "{attn_score}": np.ndarray[(batch_size,), float],  
                            ...
                        }
                        "logit": {
                            "perplexity_score": np.ndarray[(batch_size,), float],
                            "logit_entropy_score": np.ndarray[(batch_size,), float],
                            "window_logit_entropy_score": np.ndarray[(batch_size,), float] 
                        }
                    },
                    
                }
            },

        If `save_to_pkl` is True, saves each batch's dictionary incrementally to disk and returns None.

    Notes
    -----
    When using model.generate() with output_hidden_states=True (what we are replicating here with the ,
    activation hook) use_cache=True and max_new_tokens=30, there is always an offset between the length of the 
    generated sequence (outputs.sequences.shape[1][prompt_len:]) and the length of len(outputs.hidden_states) : 
    * outputs.sequences.shape[1] = prompt_len (17) + max_new_tokens (30) = 47
    * len(outputs.hidden_states) = max_new_tokens (30)
        With : 
        * outputs.hidden_states[0][layer_idx].shape = (batch_size, prompt_len, hidden_size)           --> includes the prompt ! 
        * outputs.hidden_states[i][layer_idx].shape = (batch_size, 1, hidden_size) with 1 <= i <= 29  --> stops at 29 ! 
    *Note* that in our code, outputs.hidden_states and activations are the same. 
        
    Explanation from Hugging Face, April 2024 
    (https://github.com/huggingface/transformers/issues/30036):
    """

    # ==============================================================
    # [VALIDATION] Ensure activation_source is correctly defined
    # ==============================================================
    if activation_source not in ('prompt', 'generation', 'promptGeneration'):
        raise ValueError(
                f"Invalid value for `activation_source`: '{activation_source}'. "
                f"Expected one of: ['prompt', 'generation', 'promptGeneration']."
            )    
        
    # ==============================================================  
    # [PATCH] Replace LlamaAttention.forward on target layers by
    #  custom module to extract attention weights
    # ==============================================================
    for idx in layers:  
        model.model.layers[idx].self_attn.forward = patched_LlamaAttention_forward.__get__(
            model.model.layers[idx].self_attn,
            model.model.layers[idx].self_attn.__class__
    )
        
    # ==============================================================  
    # [LOOP] Process batches of examples  
    # ==============================================================
    all_batch_results = []  

    for i in tqdm(range(idx_start_sample, idx_start_sample + max_samples, batch_size)):
      
        # ----------------------------------------------------------
        # [BATCH INPUT] Extract and tokenize prompts
        # ----------------------------------------------------------
        batch = extract_batch(dataset, i, batch_size)
        prompts = [build_prompt_fn(s["context"], s["question"]) for s in batch]
        inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(model.device)
        prompt_ids = inputs["input_ids"] # (batch_size, prompt_len)
        prompt_len = prompt_ids.shape[1] # Assumes prompts are padded to same length

        # ----------------------------------------------------------
        # [HOOKS] Register hooks to capture hidden states and attentions
        # ----------------------------------------------------------
        # This hook collects the hidden states at each decoding step. For layer l: 
        # activations_lists[l] = [act_prompt, act_gen_step1, ..., act_gen_step49] of length 50, if max_new_tokens=50.
        # activations_lists[l][k] of shape: (batch_size, seq_len, hidden_size) 
        activations_lists = [[] for _ in layers]  # one empty list per layer 
        handle_act, call_counter_act = register_activation_hook(model, activations_lists, layers)

        # This hook collects the activations at each decoding step. For layer l: 
        # attentions_lists[l] = [attn_prompt, attn_gen_step1, ..., attn_gen_step49], of length 50, if max_new_tokens=50.
        # activations_lists[l][k] of shape: (batch_size, n_heads, tgt_seq_len, src_seq_len)
        #   tgt_seq_len: length of the sequence the model is currently producing (query)
        #   src_seq_len: length of the sequence the model is focusing on (key/value)
        attentions_lists = [[] for _ in layers]  # one empty list per layer
        handle_attn, call_counter_attn = register_attention_hook(model, attentions_lists, layers)
        
        # ----------------------------------------------------------
        # [GENERATION] Run model with hooks to capture intermediate states
        # ----------------------------------------------------------
        # When target layers are reached, hooks execute and saves their output in activations and attentions
        outputs = generate(model, inputs, tokenizer, max_new_tokens=50, k_beams=1)
        gen_ids = outputs.sequences[:, prompt_len:]
    
        # Remove hooks to avoid memory leaks or duplicate logging
        for h in handle_act: h.remove()
        for h in handle_attn: h.remove()
        
        # Verify that hooks worked properly
        verify_call_counters(call_counter_act, name="activation hooks")
        verify_call_counters(call_counter_attn, name="attention hooks")

        # Retrieve text of generated answers
        gen_answers = tokenizer.batch_decode(
            outputs.sequences[:, prompt_len:], 
            skip_special_tokens=True
        ) # (batch_size,)

        # ----------------------------------------------------------
        # [FOWARD PASS] Forward pass to the model to retrieve prompt logits 
        # ----------------------------------------------------------
        if logit_agg is not None and len(logit_agg) > 0:
            gen_logits = torch.stack(outputs.logits, dim=1) 
            with torch.no_grad():
                prompt_logits = model(input_ids=inputs["input_ids"]).logits
        
        # ----------------------------------------------------------
        # [MASKING] Build attention masks for prompt and generation
        # ----------------------------------------------------------
        # This mask marks which generated tokens are valid (i.e., not padding).
        # Positions are marked True up to and including the first eos_token_id
        generation_attention_mask = build_generation_attention_mask(
            gen_ids=gen_ids, 
            eos_token_id=tokenizer.eos_token_id
        ) # (batch_size, gen_len)

        prompt_attention_mask = inputs["attention_mask"] 
        # (batch_size, prompt_len)

        # ----------------------------------------------------------
        # [OFFSET] Modify prompt mask with offset, if specified
        # ----------------------------------------------------------
        if start_offset !=0 or end_offset !=0:
            prompt_attention_mask, start_indices, end_indices = compute_offset_attention_mask(
                attention_mask=prompt_attention_mask, 
                start_offset=start_offset, 
                end_offset=end_offset
            ) # (batch_size, prompt_len), (batch_size,), (batch_size,)

        # ----------------------------------------------------------
        # [MASKING] Concatenate the prompt and generation attention mask
        # ----------------------------------------------------------
        prompt_and_gen_attention_mask = torch.cat(
            [prompt_attention_mask,
            generation_attention_mask],
            dim=1
        ) # (batch_size, prompt_len + gen_len)

        # ----------------------------------------------------------
        # [TRUNCATE] Remove final token from generated outputs to align with activations/attentions
        # ----------------------------------------------------------
        # When N tokens are generated, only the first N-1 tokens have corresponding hidden states.
        # So activations[1:] covers only the first N-1 steps. Therefore, we exclude the last
        # generated token from outputs.sequences to match activations[1:]. Same for attentions.
        truncated_gen_ids = gen_ids[:,:-1] # (gen_len-1,)
        truncated_generation_attention_mask = generation_attention_mask[:,:-1] # (batch_size, gen_len-1)
        truncated_prompt_and_gen_attention_mask = prompt_and_gen_attention_mask[:,:-1] # (batch_size, prompt_len + gen_len-1)

        # **********************************************************
        # [LAYER LOOP] Extract activation and attention-based descriptors for each specified layer 
        # **********************************************************
        save_layers_descriptors = {}

        for l, layer_idx in enumerate(layers):

            activations = activations_lists[l]
            attentions = attentions_lists[l]

            # Define prompt and generation hidden states 
            prompt_activations=activations[0]       # `[0]` to include only the prompt part 
            generation_activations=activations[1:]  # `[1:]` to exclude the prompt part 
            
            # Define prompt and generation attention maps
            prompt_attentions=attentions[0]         # `[0]` to include only the prompt part 
            generation_attentions=attentions[1:]    # `[1:]` to exclude the prompt part 

            # ------------------------------------------------------
            # [ALIGNMENT] Stack and concatenate prompt + generation activations
            # ------------------------------------------------------
            # For each batch item, take the last generated hidden state at this step
            stacked_generation_activations = torch.stack(
                [h[:, -1, :] for h in generation_activations], dim=1
            ) # (batch_size, gen_len, hidden_size)

            # Concatenate the prompt and generation hidden states  
            prompt_and_gen_activations = torch.cat(
                [stacked_generation_activations, # (batch_size, gen_len, hidden_size)
                prompt_activations],             # (batch_size, prompt_len, hidden_size)
                dim=1
            ) # (batch_size, prompt_len + gen_len, hidden_size)
            
            # ------------------------------------------------------
            # [HIDDEN DESCRIPTORS] Extract token-level activations/hidden-states
            # ------------------------------------------------------
            if hidden_agg is not None and len(hidden_agg) > 0:
                if activation_source == "generation":
                    # Return only the token activations from the generated answer 
                    selected_token_vecs = extract_token_activations(               
                            selected_layer=stacked_generation_activations, 
                            attention_mask=truncated_generation_attention_mask, 
                            device=stacked_generation_activations.device,
                            modes=hidden_agg,
                        ) # (batch_size, hidden_size)
                    
                elif activation_source == "prompt":    
                    # Return only the token activations from the prompt
                    selected_token_vecs = extract_token_activations(
                            selected_layer=prompt_activations, 
                            attention_mask=prompt_attention_mask, 
                            device=prompt_activations.device,
                            modes=hidden_agg,
                        ) # (batch_size, hidden_size)
                    
                else: # activation_source == "promptGeneration"
                    # Return token activations from the concatenated prompt + generated answer 
                    selected_token_vecs = extract_token_activations(
                            selected_layer=prompt_and_gen_activations, 
                            attention_mask=truncated_prompt_and_gen_attention_mask, 
                            device=prompt_and_gen_activations.device,
                            skip_length=prompt_len,
                            modes=hidden_agg,
                            # skip_length: exclude prompt from computation if 
                            # mode=='first_generated' in `extract_token_activations_fn`
                        ) # (batch_size, hidden_size)

                # Save results to dict
                hidden_results = {}
                for mode in hidden_agg:
                    if mode in selected_token_vecs:
                        hidden_results[mode] = selected_token_vecs[mode].cpu().numpy()
                save_layers_descriptors.setdefault(f"layer_{layer_idx}", {}).update({"hidden": hidden_results})

            # ------------------------------------------------------
            # [ATTENTION DESCRIPTORS] Extract attention eigenvalue-based metric
            # ------------------------------------------------------
            if attn_agg is not None and 'attn_score' in attn_agg:
                attn_eig_prod = compute_attn_eig_prod(
                        prompt_attentions=prompt_attentions, 
                        generation_attentions=generation_attentions,
                        prompt_attention_mask=prompt_attention_mask, 
                        generation_attention_mask=truncated_generation_attention_mask,
                        mode=activation_source,
                )
                # Save results to dict
                save_layers_descriptors.setdefault(f"layer_{layer_idx}", {}).update({"attention": {"attn_score": attn_eig_prod}}) 


            # ------------------------------------------------------
            # [LOGIT DESCRIPTORS] Compute metrics from model logits
            # ------------------------------------------------------
            if logit_agg is not None and len(logit_agg) > 0: 
                logits_results = {}

                # If this is not the last layer, the only way to compute logits is from logitLens and activations
                if layer_idx != -1 and layer_idx != model.config.num_hidden_layers-1:
                    with torch.no_grad():
                        prompt_logits = apply_logit_lens(model, prompt_activations) # (batch, prompt_len, vocab_size)
                        gen_logits = apply_logit_lens(model, stacked_generation_activations) # (batch, gen_len-1, vocab_size)
                        # When computing logits from activations, the first logit of gen_logits is missing (shape gen_len-1) 
                        # -> use `prepend_last_prompt_logit=True` (see spec from function `compute_perplexity` for details)
                    prepend_last_prompt_logit = True

                # If this is the last layer, compute logits using 'right way of computing logits'. 
                # There are small differences when computing prompt activations from forward pass and prompt activations
                # from model.generate() resulting in slightly different logits.
                else: #last layer
                    with torch.no_grad():
                        prompt_logits = model(input_ids=inputs["input_ids"]).logits # (batch, prompt_len, vocab_size)
                    gen_logits = torch.stack(outputs.logits, dim=1)  # (batch, gen_len, vocab_size)
                    prepend_last_prompt_logit = False

                if 'perplexity_score' in logit_agg:
                    perplexity = compute_perplexity(
                        prompt_logits=prompt_logits, 
                        gen_logits=gen_logits,
                        prompt_ids=prompt_ids, 
                        gen_ids=gen_ids,
                        prompt_attention_mask=prompt_attention_mask,
                        gen_attention_mask=generation_attention_mask,
                        prepend_last_prompt_logit=prepend_last_prompt_logit,
                        mode=activation_source,
                        min_k=None
                    )
                    # Save results to dict
                    logits_results['perplexity_score'] = perplexity
                
                if 'logit_entropy_score' in logit_agg:
                    if logit_config is None:
                        raise ValueError("logit_entropy_score is required but logit_config is None")
                    logit_entropy = compute_logit_entropy(
                        prompt_logits=prompt_logits,
                        gen_logits=gen_logits,
                        prompt_attention_mask=prompt_attention_mask,
                        gen_attention_mask=generation_attention_mask,
                        mode=activation_source,
                        prepend_last_prompt_logit=prepend_last_prompt_logit,
                        top_k=logit_config['top_k'],
                        window_size=None,
                        stride=None
                    )
                    # Save results to dict
                    logits_results['logit_entropy_score'] = logit_entropy

                if 'window_logit_entropy_score' in logit_agg:
                    if logit_config is None:
                        raise ValueError("window_logit_entropy_score is required but logit_config is None")
                    window_logit_entropy = compute_logit_entropy(
                        prompt_logits=prompt_logits,
                        gen_logits=gen_logits,
                        prompt_attention_mask=prompt_attention_mask,
                        gen_attention_mask=generation_attention_mask,
                        mode=activation_source,
                        prepend_last_prompt_logit=prepend_last_prompt_logit,
                        top_k=logit_config['top_k'], 
                        window_size=logit_config['window_size'], 
                        stride=logit_config['stride'] 
                    )
                    # Save results to dict
                    logits_results['window_logit_entropy_score'] = window_logit_entropy

                if logits_results:
                    save_layers_descriptors.setdefault(f"layer_{layer_idx}", {}).update({"logit": logits_results})

        # **********************************************************
        # [END LAYER LOOP] 
        # **********************************************************

        # ==========================================================
        # [OUTPUT] Store extracted results (to memory or file)
        # ==========================================================
        batch_results = {
            "id": [s['id'] for s in batch],
            "original_indices": [s['original_index'] for s in batch],
            "context": [s['context'] for s in batch],
            "question": [s['question'] for s in batch],
            "gt_answers": [s['answers'] for s in batch],
            "gen_answers": gen_answers,
            "descriptors":  {**save_layers_descriptors},
        }

        from src.data_reader.pickle_io import save_batch_pickle

        if save_to_pkl:
            save_batch_pickle(batch_data=batch_results, output_dir=output_path, batch_idx=i)
        else:
            all_batch_results.append(batch_results)

    if not save_to_pkl:
        return all_batch_results


