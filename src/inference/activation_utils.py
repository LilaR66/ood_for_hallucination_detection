#!/usr/bin/env python3
"""
============================================================
Utilities for Layer Access and Token-Level Activation Extraction in Transformer Models
============================================================

This module provides low-level tools for accessing and manipulating hidden states
within transformer-based language models. It enables extraction of token-level
activation vectors from arbitrary model layers.

These functions are designed to integrate flexibly into inference pipelines and
support various token selection strategies (e.g., final token, average span, max span).

Main Features
-------------
- Retrieves hidden states from specific transformer layers using forward hooks
- Extracts contextual embeddings from attention-masked sequences
- Supports different strategies for selecting token representations (last, average, max)
- Computes token offsets based on phrases in the input text
- Compatible with left- or right-padded tokenization
"""


from transformers import PreTrainedTokenizer, PreTrainedModel
import torch
from typing import Dict, Any, Tuple, Literal


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
        """Function called automatically by PyTorch just after
        the layer has produced its output during the forward pass."""
        # output is a tuple (hidden_states,) → keep [0]
        if layer_idx == -1:
            captured_hidden["layer_output"] = model.model.norm(output[0])  # post RMSNorm!
        else:
            captured_hidden["layer_output"] = output[0]

    # Register hook on the transformer block
    # When Pytorch pass through this layer during forward pass, it also execute hook_fn.
    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)

    # Pass inputs through the model
    # When the target layer is reached, the hook executes and saves its output in captured_hidden.
    with torch.no_grad():
        _ = model(**inputs, return_dict=True)

    # Remove the hook to avoid polluting future passages
    handle.remove()

    if "layer_output" not in captured_hidden:
        raise RuntimeError(f"Layer {layer_idx} did not produce an output.")

    return captured_hidden["layer_output"]  # shape: (batch_size, seq_len, hidden_size)


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



def extract_token_activations(
    selected_layer: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    mode: Literal["average", "last", "max"] = "average",
    start_offset: int = 0,
    end_offset: int = 0
) -> torch.Tensor:
    """
    Extract token-level activations using one of three modes: average, last, or max.
    Extract the activations vector over a token span for each sequence in a batch.
    The span is defined by applying start_offset (from the first non-padding token)
    and end_offset (from the last non-padding token).
    Supports left/right/mixed padding.

    Parameters
    ----------
    selected_layer : torch.Tensor
        Output tensor from the selected model layer of shape (batch_size, seq_len, hidden_size).
    attention_mask : torch.Tensor
        Attention mask of shape (batch_size, seq_len).
    device : torch.device
        Device for computation.
    mode : str
        Aggregation method: "average", "last", or "max".
    start_offset : int
        Offset from the first non-padding token, used in "average"/"max" (to skip e.g. [INST]).
    end_offset : int
        Offset from the last non-padding token (to skip e.g. [\INST]).

    Returns
    -------
    torch.Tensor
        Aggregated embeddings of shape (batch_size, hidden_size)
    """
    batch_size, seq_len, _ = selected_layer.shape

    # =======================================
    # Compute first and last valid token positions (regardless of padding side)
    # =======================================
    # First non-padding token is at index: number of padding tokens
    first_indices = attention_mask.float().argmax(dim=1)  
    # Last non-padding token is at the end: compute its index by flipping and computing position from end
    flipped_mask = attention_mask.flip(dims=[1])
    last_offsets = flipped_mask.float().argmax(dim=1)
    last_indices = seq_len - 1 - last_offsets
    # Move to device
    first_indices = first_indices.to(device)
    last_indices = last_indices.to(device)

    # =======================================
    # Select the last token with optional offset
    # =======================================
    if mode == "last":
        # Compute the target index using the offset and convert to integer type
        target_last_indices = (last_indices + end_offset).to(torch.long) 
        # Clamp indices to valid range 
        target_last_indices = torch.clamp(target_last_indices, min=0, max=seq_len - 1)
        batch_indices = torch.arange(batch_size, device=device)
        # Optionally, return also the indices used
        #indices = target_last_indices
        # Extract last token activation
        last = selected_layer[batch_indices, target_last_indices] # Shape: (batch_size, hidden_size)
        return last
    
    # =======================================
    # Build mask to select the token span 
    # =======================================
    # Compute target indices using the offsets and convert to integer type
    target_first_indices = (first_indices + start_offset).to(torch.long) 
    target_last_indices  = (last_indices + end_offset).to(torch.long) 

    # Clamp indices to valid range
    target_first_indices = torch.clamp(target_first_indices, min=0, max=seq_len - 1)
    target_last_indices  = torch.clamp(target_last_indices, min=0, max=seq_len - 1)

    # Compute span mask
    #--- Create a tensor of positions: shape (1, seq_len), then expand to (batch_size, seq_len)
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
    #--- Build a boolean mask: True where the position is within [target_first_indices, target_last_indices] for each sequence
    mask = (positions >= target_first_indices.unsqueeze(1)) & (positions <= target_last_indices.unsqueeze(1))
    #--- Convert the boolean mask to float and add a singleton dimension for broadcasting with selected_layer
    mask = mask.float().unsqueeze(-1)  # Shape: (batch_size, seq_len, 1)

    # =======================================
    # Apply mask and compute aggregation over the selected span  
    # =======================================
    if mode == "average":
        #--- Apply the mask to the activations: zero out tokens outside the target interval
        masked = selected_layer * mask
        #--- Count the number of selected tokens for each sequence (avoid division by zero with clamp)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        #--- Compute the mean activation vector for each sequence over the selected interval
        avg = masked.sum(dim=1) / counts # Shape: (batch_size, hidden_size)
        #--- Optionally, return also the indices used
        #indices = torch.stack([target_first_indices, target_last_indices], dim=1)
        return avg

    elif mode == "max":
        #--- Apply the mask to the activations: zero out tokens outside the target interval
        masked = selected_layer * mask.float()
        #--- Replace padding with -inf to exclude from max calculation
        masked = masked.masked_fill(mask.logical_not(), float('-inf'))
        #--- Extract maximum values across sequence dimension
        max, _ = masked.max(dim=1) # Shape: (batch_size, hidden_size)
        #--- Optionally, return also the indices used
        #indices = torch.stack([target_first_indices, target_last_indices], dim=1)
        return max

    else:
        raise ValueError(f"Unsupported mode: {mode}. Choose from 'average', 'last', or 'max'.")