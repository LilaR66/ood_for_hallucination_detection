#!/usr/bin/env python3
"""
============================================================
Token Offset Computation and Attention Mask Adjustment Utilities for LLM Prompts
============================================================

This module provides functions to compute token offsets corresponding to specific 
phrases within a tokenized text prompt and to create modified attention masks that select
precise token spans based on these offsets. These utilities help isolate sub-spans of tokens
in prompts (e.g., context or instruction segments) for targeted processing or scoring in
language model pipelines.

Key features:
-------------
- Compute token-level start and end offsets for given start and end phrases in a text prompt,
  with options to include or exclude the phrases themselves from the offsets.
- Use tokenizer's offset mapping to align character-level phrase positions to token indices reliably.
- Generate an adjusted attention mask that selects tokens within the offset-defined span,
  handling padding tokens correctly.
"""

from transformers import PreTrainedTokenizer
import torch
from typing import Tuple


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



