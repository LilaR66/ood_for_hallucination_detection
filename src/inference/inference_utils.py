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
from typing import Dict, List, Any, Callable, Tuple
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


def compute_token_offsets(
    text: str,
    tokenizer: PreTrainedTokenizer,
    start_phrase: str,
    end_phrase: str,
    debug: bool = False
) -> Tuple[int, int]:
    """
    Compute start_offset (number of tokens before the first occurrence of start_phrase)
    and end_offset (number of tokens after the last occurrence of end_phrase)
    in the tokenized version of `text`.
    
     Parameters
    ----------
    text : str
        The input prompt text.
    tokenizer : PreTrainedTokenizer
        HuggingFace tokenizer instance.
    start_phrase : str
        The phrase before which to count tokens (e.g., "Context:").
    end_phrase : str
        The phrase after which to count tokens (e.g., "[/INST]").
    debug : bool = False
        Wheter to display information
    
    Returns
    -------
    start_offset : int
        Number of tokens before the first occurrence of `start_phrase`.
    end_offset : int
        Number of tokens after the last occurrence of `end_phrase`.
    """
    if debug:
        print("===== Input text =====")
        print(text)

    #----- Tokenize the input text and get the mapping from each token to its character position in the text
    encoding = tokenizer( 
        text,
        return_offsets_mapping=True, # match each token to its exact position in the text
        add_special_tokens=True #False
    ) 
    offsets = encoding['offset_mapping']  # List of (start_char, end_char) for each token
    input_ids = encoding['input_ids']     # List of token IDs for the text
 
    #----- Find character indices of the phrases in the text
    # Find the character index where the start_phrase first appears in the text
    start_char_idx = text.find(start_phrase)
    # Find the character index where the end_phrase last appears in the text
    end_char_idx = text.rfind(end_phrase)
    # Raise an error if either phrase is not found
    if start_char_idx == -1:
        raise ValueError(f"Start phrase '{start_phrase}' not found in text.")
    if end_char_idx == -1:
        raise ValueError(f"End phrase '{end_phrase}' not found in text.")

    #---- Find the index of the first token whose span covers or starts after the start_char_idx
    start_offset = None
    for idx, (start, end) in enumerate(offsets):
        # If the token covers start_char_idx or starts after it, use this token
        if start <= start_char_idx < end or start >= start_char_idx:
            start_offset = idx
            break
    # If not found, default to the length of offsets (all tokens before)
    if start_offset is None:
        start_offset = len(offsets)

    #---- Find last token whose span includes (or ends after) the end of end_phrase
    # Compute the end character index for the end_phrase (i.e., where it finishes)
    end_phrase_end = end_char_idx + len(end_phrase)
    last_token_idx = None
    # Find the index of the first token whose end position is at or after the end of end_phrase
    for idx, (start, end) in enumerate(offsets):
        if end >= end_phrase_end:
            last_token_idx = idx
            break
    # If not found, default to the last token
    if last_token_idx is None:
        last_token_idx = len(offsets) - 1

    #---- Calculate how many tokens are after the last token of end_phrase
    end_offset = len(offsets) - (last_token_idx + 1)
    end_offset = -end_offset

    #---- Display text between start_offset and end_offset for verification
    if debug:
        print("\n===== Decoded text between `start_offset` and `end_offset` =====")
        print(f"----START TEXT---{tokenizer.decode(input_ids[start_offset:end_offset])}----END TEXT---")
        print(f"\nstart_offset: {start_offset}, end_offset:{end_offset}")

    #---- Return the number of tokens before start_phrase and after end_phrase
    return start_offset, end_offset


# Specific to Llama tokenizer: 
def extract_last_token_activations(
    selected_layer: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    end_offset: int = 0,
    start_offset: int = 0
) -> torch.Tensor:
    """
    Extract the activation vector of the token at a specific end_offset 
    from the last non-padding token in each sequence. 

    Parameters
    ----------
    selected_layer : torch.Tensor
        Output tensor from the selected model layer (shape: batch_size x seq_len x hidden_size).
    attention_mask : torch.Tensor
        Attention mask indicating real tokens (1) vs padding (0) (shape: batch_size x seq_len).
    device : torch.device
        Device to perform indexing operations on.
    end_offset : int, optional
        Negative offset from the last non-padding token. Default is 0
    start_offset : int, optionnal
        Not used here, only useful for the rest of the pipeline. 

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
    target_indices = last_indices + end_offset #+1
    # Convert indices to integer type
    target_indices = target_indices.to(torch.long)  
    batch_indices = torch.arange(selected_layer.size(0), device=device)
    # Extract the activations at the target indices
    return selected_layer[batch_indices, target_indices]


# Specific to Llama tokenizer: 
def extract_average_token_activations(
    selected_layer: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    start_offset: int = 0,
    end_offset: int = 0
) -> torch.Tensor:
    """
    Extract the mean activation vector over a token span for each sequence in a batch.
    The span is defined by applying start_offset (from the first non-padding token)
    and end_offset (from the last non-padding token).

    Parameters
    ----------
    selected_layer : torch.Tensor
        Output tensor from the selected model layer (batch_size x seq_len x hidden_size).
    attention_mask : torch.Tensor
        Attention mask (batch_size x seq_len).
    device : torch.device
        Device for computation.
    start_offset : int
        Offset from first non-padding token (to skip e.g. [INST]).
    end_offset : int
        Offset from last non-padding token (to skip e.g. [/INST]).

    Returns
    -------
    torch.Tensor
        Averaged embeddings (batch_size x hidden_size)
    """
    batch_size, seq_len, _ = selected_layer.shape

    # Detect left padding if any sequence starts with padding
    is_left_padding = (attention_mask[:, 0] == 0).any()

    # Find the index of the first and the  last non-padding token for each sequence
    if is_left_padding:
        #--- For left padding, first non-padding token is at index: number of padding tokens
        first_indices = attention_mask.argmax(dim=1)
        #--- For left padding, last non-padding token is at the end: compute its index by flipping and offsetting from the end
        last_indices = (attention_mask.size(1) - 1) - attention_mask.flip(dims=[1]).argmax(dim=1)
    else:
        #--- For right padding, first non-padding token is always at index 0
        first_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
        #--- For right padding, last non-padding token is at: (number of non-padding tokens) - 1
        last_indices = (attention_mask.sum(dim=1) - 1)

    first_indices = first_indices.to(device)
    last_indices = last_indices.to(device)

    # Apply offsets (e.g., skip <s> [INST] or [\INST])
    target_first_indices = first_indices + start_offset #-1
    target_last_indices = last_indices + end_offset #+1

    # Clamp indices to valid range
    target_first_indices = torch.clamp(target_first_indices, min=0, max=seq_len - 1)
    target_last_indices = torch.clamp(target_last_indices, min=0, max=seq_len - 1)

    # Compute mask for averaging
    #--- Create a tensor of positions: shape (1, seq_len), then expand to (batch_size, seq_len)
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
    #--- Build a boolean mask: True where the position is within [target_first_indices, target_last_indices] for each sequence
    mask = (positions >= target_first_indices.unsqueeze(1)) & (positions <= target_last_indices.unsqueeze(1))
    #--- Convert the boolean mask to float and add a singleton dimension for broadcasting with selected_layer
    mask = mask.float().unsqueeze(-1)  # (batch_size, seq_len, 1)

    # Apply mask and compute mean
    #--- Apply the mask to the activations: zero out tokens outside the target interval
    masked = selected_layer * mask
    #--- Count the number of selected tokens for each sequence (avoid division by zero with clamp)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    #--- Compute the mean activation vector for each sequence over the selected interval
    avg = masked.sum(dim=1) / counts # (batch_size, hidden_size)

    # Optionally, return also the indices used
    #indices = torch.stack([target_first_indices, target_last_indices], dim=1)

    return avg 


# Specific to Llama tokenizer: 
def extract_max_token_activations(
    selected_layer: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    start_offset: int = 5,
    end_offset: int = -5
) -> torch.Tensor:
    """
    Extract the maximum activation vector over a token span for each sequence in a batch.
    The span is defined by applying start_offset (from the first non-padding token)
    and end_offset (from the last non-padding token).

    Parameters
    ----------
    selected_layer : torch.Tensor
        Output tensor from the selected model layer (batch_size x seq_len x hidden_size).
    attention_mask : torch.Tensor
        Attention mask (batch_size x seq_len).
    device : torch.device
        Device for computation.
    start_offset : int
        Offset from first non-padding token (to skip e.g. [INST]).
    end_offset : int
        Offset from last non-padding token (to skip e.g. [/INST]).

    Returns
    -------
    torch.Tensor
        Averaged embeddings (batch_size x hidden_size)
    """
    batch_size, seq_len, _ = selected_layer.shape

    # Detect left padding if any sequence starts with padding
    is_left_padding = (attention_mask[:, 0] == 0).any()

    # Find the index of the first and the  last non-padding token for each sequence
    if is_left_padding:
        #--- For left padding, first non-padding token is at index: number of padding tokens
        first_indices = attention_mask.argmax(dim=1)
        #--- For left padding, last non-padding token is at the end: compute its index by flipping and offsetting from the end
        last_indices = (attention_mask.size(1) - 1) - attention_mask.flip(dims=[1]).argmax(dim=1)
    else:
        #--- For right padding, first non-padding token is always at index 0
        first_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
        #--- For right padding, last non-padding token is at: (number of non-padding tokens) - 1
        last_indices = (attention_mask.sum(dim=1) - 1)

    first_indices = first_indices.to(device)
    last_indices = last_indices.to(device)

    # Apply offsets (e.g., skip <s> [INST] or [\INST])
    target_first_indices = first_indices + start_offset #-1
    target_last_indices = last_indices + end_offset #+1

    # Clamp indices to valid range
    target_first_indices = torch.clamp(target_first_indices, min=0, max=seq_len - 1)
    target_last_indices = torch.clamp(target_last_indices, min=0, max=seq_len - 1)

    # Compute mask for averaging
    #--- Create a tensor of positions: shape (1, seq_len), then expand to (batch_size, seq_len)
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
    #--- Build a boolean mask: True where the position is within [target_first_indices, target_last_indices] for each sequence
    mask = (positions >= target_first_indices.unsqueeze(1)) & (positions <= target_last_indices.unsqueeze(1))
    #--- Convert the boolean mask to float and add a singleton dimension for broadcasting with selected_layer
    mask = mask.float().unsqueeze(-1)  # (batch_size, seq_len, 1)

    # Apply mask and compute mean
    #--- Apply the mask to the activations: zero out tokens outside the target interval
    masked = selected_layer * mask.float()
    #--- Replace padding with -inf to exclude from max calculation
    masked = masked.masked_fill(mask.logical_not(), float('-inf'))
    #--- Extract maximum values across sequence dimension
    max, _ = masked.max(dim=1) # (batch_size, hidden_size)

    # Optionally, return also the indices used
    #indices = torch.stack([target_first_indices, target_last_indices], dim=1)

    return max 

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
