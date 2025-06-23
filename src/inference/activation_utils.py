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
from torch.utils.hooks import RemovableHandle
from typing import Tuple, Literal, List, Optional


def register_forward_activation_hook(
    model: PreTrainedModel,
    captured_hidden: dict,
    layer_idx: int = -1
) -> RemovableHandle:
    """
    Attaches a forward hook to a specific transformer layer to capture hidden states 
    during a single forward pass (more memory-efficient than using output_hidden_states=True).
    Transformer layer = self-attention + FFN + normalization.

    Parameters
    ----------
    model : PreTrainedModel
        The Hugging Face causal language model (e.g., GPT, LLaMA).
    captured_hidden : dict
        Dictionary used to store the hidden states from the forward pass (will be overwritten).
        captured_hidden["activations"] of shape (batch_size, seq_len, hidden_size).
    layer_idx : int
        Index of the transformer block to hook. Defaults to -1 (the last layer).
        Use a positive integer if you want to hook an intermediate layer instead.

    Returns
    ----------
    RemovableHandle : A handle object
        Call `handle.remove()` after generation to remove the hook.
    call_counter : int 
        Stores the number of times the hook is activated.
    """
    # Raise error if layer_idx not in correct range
    num_layers = len(model.model.layers)
    if not (layer_idx == -1 or 0 <= layer_idx < num_layers):
        raise ValueError(
            f"`layer_idx` must be -1 or in [0, {num_layers - 1}], but got {layer_idx}."
        )
    
    call_counter = {"count": 0} # count how many times the hook is triggered
    
    def hook_fn(module, input, output):
        """Function called automatically by PyTorch just after
        the layer has produced its output during the forward pass."""
        
        call_counter["count"] += 1 
        
        # output is a tuple (hidden_states,) → keep [0]
        if layer_idx == -1:
            captured_hidden["activations"] = model.model.norm(output[0])  # post RMSNorm!
        else:
            captured_hidden["activations"] = output[0]

    # Register hook on the transformer block
    # When Pytorch pass through this layer during a forward pass, it also execute hook_fn.
    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)

    return handle, call_counter



def register_generation_activation_hook(
    model: PreTrainedModel,
    captured_hidden_list: List[torch.Tensor],
    layer_idx: int = -1
) -> RemovableHandle:
    """
    Attaches a forward hook to a specific transformer layer to capture hidden states
    during autoregressive text generation i.e., at each decoding step.
    (more memory-efficient than using output_hidden_states=True).
    Transformer layer = self-attention + FFN + normalization.

    Parameters
    ----------
    model : PreTrainedModel
        The Hugging Face causal language model (e.g., GPT, LLaMA).
    captured_hidden_list : List[torch.Tensor]
        A list that will be filled with hidden states for each generation step. 
        Each tensor has shape (batch_size * num_beams, seq_len, hidden_size).
    layer_idx : int
        Index of the transformer block to hook. Defaults to -1 (the last layer).
        Use a positive integer if you want to hook an intermediate layer instead.

    Returns
    ----------
    RemovableHandle : A handle object
        Call `handle.remove()` after generation to remove the hook.
    call_counter : int 
        Stores the number of times the hook is activated.
    """
    # Raise error if layer_idx not in correct range
    num_layers = len(model.model.layers)
    if not (layer_idx == -1 or 0 <= layer_idx < num_layers):
        raise ValueError(
            f"`layer_idx` must be -1 or in [0, {num_layers - 1}], but got {layer_idx}."
        )
    
    call_counter = {"count": 0} # count how many times the hook is triggered

    def hook_fn(module, input, output):
        """Function called automatically by PyTorch just after
            the layer has produced its output during the forward pass."""
        
        call_counter["count"] += 1 

        # output is a tuple (hidden_states,) → keep [0]
        if layer_idx == -1:
            # Capture the final normalized output 
            captured_hidden_list.append(model.model.norm(output[0]).detach().cpu())  # post RMSNorm!
        else:
            # Capture raw hidden states before layer normalization
            captured_hidden_list.append(output[0].detach().cpu()) 
    
    # Register hook on the transformer block
    # When Pytorch pass through this layer during forward pass, it also execute hook_fn.
    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    
    return handle, call_counter


def compute_token_offsets(
    text: str,
    tokenizer: PreTrainedTokenizer,
    start_phrase: str,
    end_phrase: str,
    include_start_phrase: bool = False,
    include_end_phrase: bool = False,
    debug: bool = False,
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
    include_start_phrase : bool, optional
        If True, include the start_phrase itself in the offset.
    include_end_phrase : bool, optional
        If True, include the end_phrase itself in the offset.
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

    # ==============================
    # Tokenize the input text and get the mapping from each token to its character position in the text
    # ==============================
    encoding = tokenizer( 
        text,
        return_offsets_mapping=True, # match each token to its exact position in the text
        add_special_tokens=True 
    ) 
    offsets = encoding['offset_mapping']  # List of (start_char, end_char) for each token
    input_ids = encoding['input_ids']     # List of token IDs for the text
 
    # ==============================
    # Find character indices of the phrases in the text
    # ==============================
    # Find the character index where the start_phrase first appears in the text
    start_char_idx = text.find(start_phrase)
    if not include_start_phrase: 
        start_char_idx = start_char_idx + len(start_phrase)
    # Find the character index where the end_phrase last appears in the text
    end_char_idx = text.rfind(end_phrase)
    if not include_end_phrase :
        end_char_idx = end_char_idx - len(end_phrase)
    # Raise an error if either phrase is not found
    if start_char_idx == -1:
        raise ValueError(f"Start phrase '{start_phrase}' not found in text.")
    if end_char_idx == -1:
        raise ValueError(f"End phrase '{end_phrase}' not found in text.")

    # ==============================
    # Find the index of the first token whose span covers or starts after the start_char_idx
    # ==============================
    start_offset = None
    for idx, (start, end) in enumerate(offsets):
        # If the token covers start_char_idx or starts after it, use this token
        if start <= start_char_idx < end or start >= start_char_idx:
            start_offset = idx
            break
    # If not found, default to the length of offsets (all tokens before)
    if start_offset is None:
        start_offset = len(offsets)

    # ==============================
    # Find last token whose span includes (or ends after) the end of end_phrase
    # ==============================
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

    # ==============================
    # Calculate how many tokens are after the last token of end_phrase
    # ==============================
    end_offset = len(offsets) - (last_token_idx + 1)
    end_offset = -end_offset

    # ==============================
    # Display text between start_offset and end_offset for verification
    # ==============================
    if debug:
        print("\n===== Decoded text between `start_offset` and `end_offset` =====")
        if end_offset != 0:
            print(f"----START TEXT----{tokenizer.decode(input_ids[start_offset:end_offset])}----END TEXT----")
        else :
            print(f"----START TEXT----{tokenizer.decode(input_ids[start_offset:])}----END TEXT----")
        print(f"\nstart_offset: {start_offset}, end_offset:{end_offset}")

    # Return the number of tokens before start_phrase and after end_phrase
    return start_offset, end_offset



def compute_offset_attention_mask(
    attention_mask: torch.Tensor,
    start_offset: int = 0,
    end_offset: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns a modified attention mask selecting a token span based on padding-aware offsets.

    For each sequence in the batch:
    - The function identifies the span of real (non-padding) tokens using the `attention_mask`.
    - It then shifts the start and end of this real span using:
        - `start_offset` >= 0: number of tokens to skip from the beginning of the real tokens.
        - `end_offset` <= 0: number of tokens to exclude from the end of the real tokens.
    - It outputs a new boolean attention mask of the same shape as `attention_mask`, 
      marking only the tokens inside the selected subspan as `1`, and everything else 
      (including original padding) as `0`.

    Examples
    --------
    - If a sequence has real tokens at positions 10 to 50:
        - `start_offset=0, end_offset=0` selects tokens 10 to 50 (inclusive start, exclusive end).
        - `start_offset=5, end_offset=-3` selects tokens 15 to 47.
    - The resulting span always lies within the non-padding region.
            
    Parameters
    ----------
    attention_mask : torch.Tensor
        Shape (batch_size, seq_len). 1 for real tokens, 0 for padding.
    start_offset : int
        Offset from the first non-padding token (must be >= 0). 
    end_offset : int
        Offset from the last non-padding token (must be <= 0, e.g., -3 to remove 3 tokens).

    Returns
    -------
    offset_attention_mask : torch.Tensor
        Boolean tensor of shape (batch_size, seq_len). 
        `True` (or `1`) where tokens fall inside the offset-adjusted span.
    start : torch.Tensor
        Tensor of shape (batch_size,) indicating the inclusive start index of the span for each sequence.
    end : torch.Tensor
        Tensor of shape (batch_size,) indicating the exclusive end index of the span for each sequence.

    Raises
    ------
    ValueError
        If start_offset is negative or end_offset is positive.
        If the offsets result in an empty or invalid span for any sequence.
    """
    # =======================================
    # Validate offsets: start_offset must be non-negative, end_offset must be zero or negative.
    # =======================================
    if start_offset < 0:
        raise ValueError(f"`start_offset` must be non-negative, got: {start_offset}")
    if end_offset > 0:
        raise ValueError(f"`end_offset` must be zero or negative, got: {end_offset}")

    batch_size, seq_len = attention_mask.shape
    device = attention_mask.device
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
    # Compute start and end indices and clamp to valid range
    # =======================================
    # Compute target indices using the offsets and convert to integer type
    start = (first_indices + start_offset).to(torch.long)
    end = (last_indices + 1 + end_offset).to(torch.long)  # +1 for exclusive

    # Clamp to valid, non-padding region
    start = torch.clamp(start, min=first_indices, max=last_indices + 1)
    end = torch.clamp(end, min=start, max=last_indices + 1)

    # =======================================
    # Validate that the slice is non-empty
    # =======================================
    empty = (start >= end).nonzero(as_tuple=True)[0]
    if len(empty) > 0:
        raise ValueError(
            f"Token offsets result in an empty slice for at least one sequence: "
            f"start_offset={start_offset}, end_offset={end_offset}, "
            f"indices={[(int(start[i]), int(end[i])) for i in empty]}"
        )

    # =======================================
    # Build boolean span selection mask
    # =======================================
    # Create a tensor of positions: shape (1, seq_len), then expand to (batch_size, seq_len)
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
    # Build a boolean mask: True where the position is within [target_first_indices, target_last_indices] for each sequence
    offset_attention_mask = (positions >= start.unsqueeze(1)) & (positions < end.unsqueeze(1)) # Shape (batch_size, seq_len)

    return offset_attention_mask.int(), start, end



def extract_token_activations(
    selected_layer: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    mode: Literal["average", "last", "max", "first_generated"] = "average",
    skip_length: Optional[int] = None,
) -> torch.Tensor:
    """   
    Aggregate token-level activations over a specified span for each sequence in a batch,
    using attention mask.

    This function takes as input:
      - The layer activations (selected_layer) for each token in a batch of sequences,
      - An attention mask (attention_mask) of the same shape, where 1 indicates tokens to include
        in the aggregation and 0 marks tokens to ignore.

    The attention mask may be the original model mask, or a custom mask generated using
    `compute_offset_attention_mask` to dynamically select a sub-span of tokens.

    Modes
    -----
    - "average": Computes the mean activation over all tokens selected by `attention_mask`.
    - "max": Computes the element-wise maximum activation vector over all tokens selected by `attention_mask`.
    - "last": Selects the activation of the last token selected by the `attention_mask` (the last 1).
    - "first_generated": Selects the activation of the first token selected by `attention_mask`,
        or at index `skip_length` if provided.

    Parameters
    ----------
    selected_layer : torch.Tensor
        Tensor of shape (batch_size, seq_len, hidden_size) containing model activations for each token.
    attention_mask : torch.Tensor
        Attention mask of shape (batch_size, seq_len),  1 for real tokens, 0 for padding.
    device : torch.device
        Device for computation.
    mode : str
        Aggregation method: "average", "last", "max", or "first_generated".
    skip_length : Optional[int]
        If provided, used to explicitly select the first generated token (useful for "first_generated" mode).

    Returns
    -------
    torch.Tensor
        Aggregated activations of shape (batch_size, hidden_size), 
        according to the selected mode.
    """
    batch_size, seq_len, _ = selected_layer.shape

    # Move to device 
    attention_mask = attention_mask.to(selected_layer.device)

    # =======================================
    # Select the first tokenwith optional offset `skip_length`
    # =======================================
    if mode == "first_generated":
        batch_indices = torch.arange(batch_size, device=device)
        if skip_length is not None:
            first_indices = torch.full((batch_size,), skip_length, device=device, dtype=torch.long)
        else:
            first_indices = (attention_mask == 1).float().argmax(dim=1)
        first = selected_layer[batch_indices, first_indices] # Shape: (batch_size, hidden_size)
        aggregated_tokens = first

    # =======================================
    # Select the last token 
    # =======================================
    elif mode == "last":
        last_indices = attention_mask.shape[1] - 1 - attention_mask.flip(dims=[1]).float().argmax(dim=1)
        batch_indices = torch.arange(batch_size, device=device)
        last = selected_layer[batch_indices, last_indices]  # Shape: (batch_size, hidden_size)
        aggregated_tokens = last
    
    # =======================================
    # Apply mask and compute aggregation 
    # =======================================
    elif mode == "average":
        # Add one dimension for the broadcast on hidden_size
        attention_mask = attention_mask.float().unsqueeze(-1)  # (batch_size, seq_len, 1)
        # Apply the mask to the activations: zero out tokens outside the target interval
        masked = selected_layer * attention_mask
        #  Count the number of selected tokens for each sequence (avoid division by zero with clamp)
        counts = attention_mask.sum(dim=1).clamp(min=1e-6)
        #  Compute the mean activation vector for each sequence over the selected interval
        avg = masked.sum(dim=1) / counts # Shape: (batch_size, hidden_size)
        aggregated_tokens =  avg

    elif mode == "max":
        # Add one dimension for the broadcast on hidden_size
        attention_mask = attention_mask.float().unsqueeze(-1)  # (batch_size, seq_len, 1)
        #  Apply the mask to the activations: zero out tokens outside the target interval
        masked = selected_layer * attention_mask.float()
        #  Replace padding with -inf to exclude from max calculation
        masked = masked.masked_fill(attention_mask.logical_not(), float('-inf'))
        #  Extract maximum values across sequence dimension
        max, _ = masked.max(dim=1) # Shape: (batch_size, hidden_size)
        aggregated_tokens = max
    
    else:
        raise ValueError(f"Unsupported mode: {mode}. Choose from 'average', 'last', 'max', or 'first_generated'.")

    return aggregated_tokens