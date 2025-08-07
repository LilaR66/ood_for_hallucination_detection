#!/usr/bin/env python3
"""
============================================================
Activation, Attention and Logit Scores Computation Module for Language Models
============================================================

This module provides functions to extract and aggregate token-level activations 
(= hidden states) from hidden layers of language models, as well as to compute  
metrics based on attention maps and logits.

Main Features
-------------
- ACTIVATION/HIDDEN STATES scores:
    - Flexible aggregation of token activations with multiple modes  
        (average, max, last token, first_generated)
    - Computation of scores based on centered Gram matrix, covariance, and SVD decomposition
        (token_svd_score, feat_var)

- ATTENTION scores:
    - Attention score calculations inspired by eigenvalues of attention maps

- LOGIT scores:
    - Pseudo-perplexity and logit entropy computations to analyze model internal consistency

- Support for attention masks to dynamically filter valid tokens
- Support for computing the scores on prompt, generation or concatenated prompt + generation
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Literal, List, Optional, Dict


# *********************************
# Hidden/Activations scores
# *********************************
def extract_token_activations(
    selected_layer: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    modes: List[Literal[
        "average", "last", "max", "first_generated", 
        "token_svd_score", "feat_var"
    ]] = ["average"],
    skip_length: Optional[int] = None,
    alpha: int = 0.001,
) -> Dict[str, torch.Tensor]:
    """   
    Aggregate token-level activations over a specified span for each sequence in a batch,
    using various aggregation modes and attention mask.

    This function takes as input:
      - The layer activations (selected_layer) for each token in a batch of sequences,
      - An attention mask (attention_mask) of the same shape, where 1 indicates tokens to include
        in the aggregation and 0 marks tokens to ignore.

    The attention mask may be the original model mask, or a custom mask generated using
    `compute_offset_attention_mask` to dynamically select a sub-span of tokens.

    Parameters
    ----------
    selected_layer : torch.Tensor
        Tensor of shape (batch_size, seq_len, hidden_size) containing model activations for each token.
    attention_mask : torch.Tensor
        Attention mask of shape (batch_size, seq_len),  1 for real tokens, 0 for padding.
    device : torch.device
        Device for computation.

    modes : List[str]
        List of aggregation modes to compute. Computed using only valid tokens where attention_mask == 1.
        Supported:
        - "average": Mean activation vector across valid tokens. Shape: (batch_size, hidden_size)
        - "max": Element-wise max activation across valid tokens. Shape: (batch_size, hidden_size)
        - "last": Activation vector of last valid token in each sequence. Shape: (batch_size, hidden_size)
        - "first_generated": Activation of the first generated valid token in each sequence. Shape: (batch_size, hidden_size)
             If skip_length is provided, selects the token starting from that offset. 
        - "token_svd_score": Mean log singular value of the centered Gram matrix over tokens. Shape (batch_size,)
            The Gram matrix is computed as Gram_token = Z·J·Z^T, where J is the centering matrix on features.
            It quantifies the pairwise similarity between token representations after removing the mean value 
            of each feature across tokens. Note: This is not a classical covariance matrix.
            The log singular values quantifies the effective dimensionality or diversity of the token 
            activations: higher values reflect more diverse (less redundant) token representations, lower values 
            indicate more redundancy or alignment.
            NOTE: Implementation inpired by "LLM-Check: Investigating Detection of Hallucinations in 
            Large Language Models" (Sriramanan et al. 2024)
        - "feat_var": Diagonal of the centered feature covariance matrix (variances). Shape: (batch_size, hidden_size)

    skip_length : Optional[int]
        If provided, used to explicitly select the first generated token (useful for "first_generated" mode).
    alpha : float
        Regularization parameter added to the covariance matrix.

    Returns
    -------
    Dict[str, torch.Tensor or np.ndarray]
        Dictionary mapping each mode to its result:
            - (batch_size, hidden_size) for "average", "max", "last", "first_generated", "feat_var"
            - (batch_size,) for "token_svd_score"
    """

    batch_size, seq_len, hidden_size = selected_layer.shape
    aggregated_tokens = {}
    
    # Move to device 
    attention_mask = attention_mask.to(selected_layer.device)

    # =======================================
    # Select the first token with optional offset `skip_length`
    # =======================================
    if "first_generated" in modes:
        batch_indices = torch.arange(batch_size, device=device)
        if skip_length is not None:
            first_indices = torch.full((batch_size,), skip_length, device=device, dtype=torch.long)
        else:
            first_indices = (attention_mask == 1).float().argmax(dim=1)
        first = selected_layer[batch_indices, first_indices] # Shape: (batch_size, hidden_size)
        aggregated_tokens["first_generated"] = first

    # =======================================
    # Select the last token 
    # =======================================
    if "last" in modes:
        last_indices = attention_mask.shape[1] - 1 - attention_mask.flip(dims=[1]).float().argmax(dim=1)
        batch_indices = torch.arange(batch_size, device=device)
        last = selected_layer[batch_indices, last_indices]  # Shape: (batch_size, hidden_size)
        aggregated_tokens["last"] = last

    # =======================================
    # Apply mask and compute aggregation 
    # =======================================
    if "average" in modes or "max" in modes:
        # Add one dimension for the broadcast on hidden_size
        mask_float = attention_mask.float().unsqueeze(-1)  # (batch_size, num_valid_tokens, 1)
        # Apply the mask to the activations: zero out tokens outside the target interval
        masked = selected_layer * mask_float
        #  Count the number of selected tokens for each sequence (avoid division by zero with clamp)
        counts = mask_float.sum(dim=1).clamp(min=1e-6)
        if "average" in modes:
            # Compute the mean activation vector for each sequence over the selected interval
            avg = masked.sum(dim=1) / counts # Shape: (batch_size, hidden_size)
            aggregated_tokens["average"] = avg
        if "max" in modes:
            # Replace padding with -inf to exclude from max calculation
            masked_max = masked.masked_fill(mask_float.logical_not(), float('-inf'))
            # Extract maximum values across sequence dimension
            max_vals, _ = masked_max.max(dim=1) # Shape: (batch_size, hidden_size)
            aggregated_tokens["max"] = max_vals

    # =======================================
    # Covariance-based metrics
    # =======================================
    if any(m in modes for m in ["token_svd_score", "feat_var"]):
        token_svd_score = [] 
        feat_var = []
        
        for i in range(batch_size):
            # Select valid tokens 
            mask = attention_mask[i].bool()
            Z = selected_layer[i][mask]  # (num_valid_tokens, hidden_size)
            
            if Z.shape[0] == 0:
                feat_var.append(torch.full((hidden_size,), float('nan')))
                token_svd_score.append(float('nan'))
                continue
            
            if Z.dtype != torch.float32:
                Z = Z.to(torch.float32)

            if "token_svd_score" in modes:
                # Compute Gram matrix on tokens : Gram_token = Z·J·Z^T
                # ---------------------------------------
                # Assumes Z is in full precision
                # Center the features of Z (i.e., subtract the mean value of each feature across tokens)
                J = torch.eye(hidden_size, device=Z.device, dtype=Z.dtype) - (1 / hidden_size) * torch.ones(hidden_size, hidden_size, device=Z.device, dtype=Z.dtype)
                # The Gram matrix Gram_token reflects the inner products (similarities) between tokens
                Gram_token = torch.matmul(torch.matmul(Z, J), Z.t()) # (num_valid_tokens, num_valid_tokens)
                # Regularization for stabilization
                Gram_token = Gram_token + alpha * torch.eye(Gram_token.shape[0], device=Z.device, dtype=Z.dtype)
            
                # Singular value decomposition (SVD) of the token Gram matrix
                # ---------------------------------------
                if Gram_token.dtype != torch.float32:
                    Gram_token = Gram_token.to(torch.float32)
                token_svdvals = torch.linalg.svdvals(Gram_token) # Singular Value Decomposition
                token_eigscore = torch.log(token_svdvals).mean()  # mult by 2 missing from the paper? 
                token_svd_score.append(token_eigscore)

            if "feat_var" in modes:
                # Compute covariance matrix on features 
                # ---------------------------------------
                Z_feat_centered = Z - Z.mean(dim=0, keepdim=True) # (num_valid_tokens, hidden_size)
                Cov_feat = (Z_feat_centered.t() @ Z_feat_centered) / max(1, Z.shape[0] - 1) # (hidden_size, idden_size)
                Cov_feat += alpha * torch.eye(Z.shape[1], device=Z.device, dtype=Z.dtype)
                feat_var.append(Cov_feat.diag())
            
        # Return scores
        # ---------------------------------------
        if "feat_var" in modes:
            aggregated_tokens["feat_var"] = torch.stack(feat_var, dim=0) # (batch_size, hidden_size) 
        if "token_svd_score" in modes:
            aggregated_tokens["token_svd_score"] = torch.stack(token_svd_score) # (batch_size,) 
        
        # Put everything on CPU
        # ---------------------------------------
        for key in aggregated_tokens:
            aggregated_tokens[key] = aggregated_tokens[key].detach().cpu()

    return aggregated_tokens


# *********************************
# Attention scores
# *********************************
def compute_attn_eig_prod(
    prompt_attentions: torch.Tensor,
    generation_attentions: List[torch.Tensor],
    prompt_attention_mask: torch.Tensor,
    generation_attention_mask: torch.Tensor,
    mode: Literal["prompt", "generation", "promptGeneration"] = "prompt"
) -> np.ndarray:
    """
    Compute a mean log-diagonal attention score (eigenvalue-inspired) for a single layer's 
    attention map, using attention mask. 
    
    NOTE: Implementation inspired by 
    "LLM-Check: Investigating Detection of Hallucinations in Large Language Models"
    (Sriramanan et al. 2024)

    Parameters
    ----------
    prompt_attentions: torch.Tensor
        Tensor of shape (batch_size, n_heads, prompt_len, prompt_len)
        Self-attention over the prompt tokens. 
    generation_attentions: list of torch.Tensor
        List of tensors of shape (batch_size, n_heads, 1, prompt_len + t)
        Self-attention for each generated token at generation step t (t >= 1).
    prompt_attention_mask: torch.Tensor
        Tensor of shape (batch_size, prompt_len), 1 where token valid, 0 for padding.
    generation_attention_mask: torch.Tensor  
        Tensor of shape (batch_size, gen_len), 1 where token valid, 0 for padding.
    mode : str, optional
        Specifies which part of the attention map to use for the score computation.
        Must be one of the following:
        - "prompt":
            Only uses the prompt self-attention map (prompt_attentions). 
            It is a matrix of shape (batch_size, n_heads, prompt_len, prompt_len).
            The diagonal (i.e., self-attention values per token) is extracted,
            then the log is taken, followed by a mean over prompt tokens and sum over heads.
        - "generation":
            Only uses the generated self-attention maps (generation_attentions).
            Each tensor in generation_attentions has shape (batch_size, n_heads, 1, prompt_len + t),
            where t is the generation step. 
            Instead of concatenating these tensors to obtain the generation attention matrix, 
            for each step, we directly take the last value along the last axis (i.e., the self-attention
            of the newly generated token). These values are stacked across time steps, then we take the log,
            compute the mean over time, and sum over heads.
        - "promptGeneration":
            Combines the diagonals from both the prompt and generation attention maps as described above
            for "prompt" and "generation" mode. The two diagonals are concatenated along the token/time axis, 
            then the log is taken, followed by a mean across all tokens and a sum over heads.
            Note: we do **not** concatenate the full prompt and generation attention matrices,
            since the diagonal of the combined matrix would only include values from the prompt attention
            due to mismatched matrix shapes.

    Returns
    -------
    np.ndarray
        A NumPy array of shape (batch_size,), where each value is the per-sample attention score.
        The score is summed across heads and averaged across tokens (in log-space).
    """
    if mode not in ("prompt", "generation", "promptGeneration"):
        raise ValueError(f"Invalid mode: {mode}. Must be 'prompt', 'generation' or 'promptGeneration'.")

    batch_size, n_heads = prompt_attentions.shape[:2]
    if generation_attentions is not None:
        gen_len = len(generation_attentions)    
    diag_blocks = []

    # Move to device
    device = prompt_attentions.device
    prompt_attention_mask = prompt_attention_mask.to(device)
    if generation_attention_mask is not None:
        generation_attention_mask = generation_attention_mask.to(device)

    # ==============================
    # Prompt mode or combined
    # ==============================
    if mode in ("prompt", "promptGeneration"):
        # Extract diagonal of prompt attentions
        prompt_diag = torch.diagonal(prompt_attentions, dim1=-2, dim2=-1) # (batch_size, n_heads, prompt_len)
        # Expand prompt mask to (batch_size, n_heads, prompt_len)
        p_mask_ext = prompt_attention_mask.unsqueeze(1).expand(-1, n_heads, -1)
        diag_blocks.append(prompt_diag)

    # ==============================
    # Generation mode or combined
    # ==============================
    if mode in ("generation", "promptGeneration") and gen_len > 0:
        # For each generation step, take the last value along last dim.
        gen_diag_steps = [attn[..., -1].squeeze(-1) for attn in generation_attentions] # list of (batch_size, n_heads)
        # Stack along time axis (= newly generated tokens)
        gen_diag = torch.stack(gen_diag_steps, dim=-1) if gen_diag_steps else None # (batch_size, n_heads, gen_len)
        # Expand generation mask to (batch_size, n_heads, gen_len)
        g_mask_ext = generation_attention_mask.unsqueeze(1).expand(-1, n_heads, -1)
        if gen_diag is not None:
            diag_blocks.append(gen_diag)


    # Concatenate diagonals along tokens/time dim
    all_diags = torch.cat(diag_blocks, dim=-1) # (batch_size, n_heads, N) where N = prompt_len + n_generated (or a subset)
    # Build full mask concatenated similarly: (batch_size, n_heads, N)
    if mode == "prompt":
        full_mask = p_mask_ext # (batch_size, n_heads, prompt_len)
    elif mode == "generation":
        full_mask = g_mask_ext  # (batch_size, n_heads, gen_len)
    else:  # "promptGeneration"
        full_mask = torch.cat([p_mask_ext, g_mask_ext], dim=-1)  # (batch_size, n_heads, total_len)

    # ==============================
    # Compute attention eigen product, ignoring padding tokens 
    # ==============================
    # Clamp very small values to avoid log(0)
    all_diags = all_diags.clamp(min=1e-6)
    # Compute log
    log_all_diags = torch.log(all_diags) # (batch_size, n_heads, N)
    # Mask out padding tokens by zeroing out their logs
    masked_log_all_diags = log_all_diags * full_mask # (batch_size, n_heads, N)
    # Count valid tokens per batch and head to compute mean properly (avoid div by zero)
    valid_token_counts = full_mask.sum(dim=-1).clamp(min=1) # (batch_size, n_heads)
    # Mean log diag over valid tokens dimension (N)
    mean_log_diag = masked_log_all_diags.sum(dim=-1) / valid_token_counts  # (batch_size, n_heads)
    # Sum over heads to get final per-sample scores
    scores = mean_log_diag.sum(dim=-1).cpu().numpy() # (batch_size,)

    return scores  # (batch_size,)


# *********************************
# Logit scores
# *********************************
def compute_perplexity(
        prompt_logits: torch.Tensor, 
        gen_logits: torch.Tensor,
        prompt_ids: torch.Tensor, 
        gen_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        gen_attention_mask: torch.Tensor,
        mode: Literal["prompt", "generation", "promptGeneration"] = "prompt",
        prepend_last_prompt_logit: bool = False,
        min_k: float = None,
    ) -> np.ndarray:
    """
    Computes the per-sample perplexity of language model outputs using logits 
    and corresponding input token IDs. Logits maked by 0 in the attention mask 
    are ignored in the computation of the perplexity. 
    If `min_k` is provided,
    it filters the lowest probabilities to compute a restricted perplexity.

    Perplexity is defined as:
        Perplexity = exp(- mean(log P(token_i | context))) 
        where token_i is the next token actually predicted

    NOTE: This implementation is inspired by:
    "LLM-Check: Investigating Detection of Hallucinations in Large Language Models"
    (Sriramanan et al., 2024)

    Parameters
    ----------
    prompt_logits : torch.Tensor
        Tensor of shape (batch_size, prompt_len, vocab_size) 
        These are the model's output logits obtained from a standard forward pass over the prompt sequence.
    gen_logits : torch.Tensor
        Tensor of shape (batch_size, gen_len, vocab_size).
        These are the logits obtained during autoregressive decoding using `model.generate()`.
    prompt_ids : torch.Tensor
        Tensor of shape (batch_size, prompt_len), containing the input token IDs for the prompt.
    gen_ids : torch.Tensor
        Tensor of shape (batch_size, gen_len), containing the token IDs generated by the model.
    prompt_attention_mask: torch.Tensor
        Tensor of shape (batch_size, prompt_len), 1 where token valid, 0 for padding.
    gen_attention_mask: torch.Tensor  
        Tensor of shape (batch_size, gen_len), 1 where token valid, 0 for padding.
    mode : str, optional
        One of {"prompt", "generation", "promptGeneration"}:
        - "prompt": compute perplexity only over the prompt.
        - "generation": compute perplexity only over the generated tokens.
        - "promptGeneration": compute perplexity over both prompt and generation.
    prepend_last_prompt_logit : bool, optional
        If True, appends the last logit from the prompt to the beginning of the generation logits.
        This is useful when generation logits were computed manually from hidden states 
        and are therefore shifted by one position (they lack the first prediction step).
        => see Notes 2)C) below for more detail. Default is False.
        Carreful! The gen_attention_mask must match.
    min_k : float, optional
        Optional value between 0 and 1. If specified, only the bottom-k lowest-probability
        tokens are used for perplexity calculation.

    Returns
    --------
        np.ndarray: Per-sample perplexity scores of shape (batch_size,)

    Notes
    -----
    1) This function computes a "Pseudo Perplexity".

        The Standard Perplexity requires ground truth tokens:
            PPL = exp(-1/N ∑_{t=1}^N log p(w_real_t | w_real_{<t})) where w_real_t are the true next tokens

        In our case, we are in pure generation mode (equivalent to teacher forcing on the generated text)
        and we don't have acces the real tokens. We thefore compute the Pseudo Perplexity:
            PPL_gen = exp(-1/N ∑_{t=1}^N log p(w_gen_t | w_gen_{<t})) where w_gen_t are the generated next tokens
        This measures the internal consistency of the model, and how well the model finds its own generation probable. 
    
    2) About token shifting in autoregressive models:

        A) When extracting prompt logits with with a standard autoregressive forward pass:
            Example: 
            prompt_outputs = model(inputs['input_ids'], output_logits=True)
            prompt_logits = outputs.logits # Tensor of shape (batch_size, promp_len, vocab_size)

            - The logit at position *t* predicts the token at position *t+1*.
            - The first token has no preceding context and is not predicted.
            - When computing log-probabilities, we must **shift the targets one position to the left** 
            to correctly align logits with target tokens.
            
            Example: Suppose we have a sequence of tokens (with their token IDs):
            | Index | Token | ID  |
            |-------|-------|-----| - The model produces logits at positions 0, 1, 
            | 0     | A     | 10  | and 2 to predict the tokens B, C, and D, respectively.
            | 1     | B     | 29  |
            | 2     | C     | 305 |  - The logits at position 0 are used to predict
            | 3     | D     | 24  |  token B (ID 29).

        B) When extracting gen logits during during generation:
            Example: 
            gen_outputs = model.generate(**inputs, max_new_tokens=10, output_logits=True) # gen_len=max_new_tokens
            gen_logits = torch.stack(outputs.logits,dim=1)  # Tensor of shape (batch_size, gen_len, vocab_size)
            
            - The logit at time step *t* predicts the token generated at position *t*.
            - Each logit already corresponds to the prediction of the token at this step 
            - No shifting is needed in this case.

        REMARK:
        The last prompt logit (i.e., `prompt_logits[:, -1, :]`) predicts the first generated token.
        This means: prompt_logits[:, -1, :] = gen_logits[:, 0, :]

        C) When computing gen_logits directly from the activations (no build-in methods) 
            Example:
            computed_gen_logits = model.lm_head(gen_hidden_states) 
            # Tensor of shape (batch_size, gen_len-1, vocab_size), gen_len=max_new_tokens
            Typically, `gen_hidden_states` is a Tensor of shape (batch_size, gen_len-1, hidden_size) 
            computed from gen_outputs.hidden_states. It has shape `gen_len-1`
            because there is no hidden state for the final generated token.

            - Here, the logit at position *t* predicts the generated token at position *t+1* !! 
            - Since the last prompt_logits = the first gen_logits, we repend the last prompt
            logit to the beginning of computed_gen_logits with `prepend_last_prompt_logit=True`
            - We recover case B) with the full gen_logits (as would be returned by model.generate) 
            - We can use the same gen_attenion_mask as in case B)

        D) When computing prompt_logits directly from the activations (no build-in methods) 
            Example:
            computed_prompt_logits = model.lm_head(prompt_hidden_states) 
            # Tensor of shape (batch_size, prompt_len, vocab_size),
            Typically, `prompt_hidden_states` is a Tensor of shape (batch_size, prompt_len, hidden_size) 

            - The logit at position *t* predicts the token at position *t+1* => recover case A)
            - We can use the same prompt_attenion_mask as in case A)

        Summary of alignment:
            - A) Prompt Logits from forward: 
                logit at position *t* predicts token at position *t+1* -> shift targets left.
            - B) Generation Logits from generate: 
                logit at position *t* predicts token at position *t* -> no shift.
            - C) Computed Generation Logits: 
                set `append_last_prompt_logit=True` and no shift needed -> go back to case B)
            - D) Computed Prompt Logits:
                similar to case A), nothing to do. 

        NOTE: help from issue https://github.com/huggingface/transformers/issues/29664
    """  

    if min_k is not None:
        if min_k < 0 or min_k > 1: raise ValueError("min_k must be between 0 and 1")

    if mode not in ('prompt','generation','promptGeneration'):
        raise ValueError("mode must be in {'prompt','generation','promptGeneration'}")
    
    # ==============================
    # Move to device
    # ==============================
    prompt_logits = prompt_logits.to(prompt_attention_mask.device)
    if gen_logits is not None:
        gen_logits = gen_logits.to(gen_attention_mask.device)

    # ==============================
    # Prepend last logit of prompt to the generation logits if specifed
    # ==============================
    if prepend_last_prompt_logit:
        last_prompt_logit = prompt_logits[:, -1:, :] # (batch_size, 1, vocab_size)
        gen_logits = torch.cat([last_prompt_logit, gen_logits], dim=1) # (batch_size, gen_len+1, vocab_size)
           
    # ==============================
    # Apply softmax over vocabulary dimension and take log to get log-probabilities
    # ==============================
    prompt_log_probs = torch.log_softmax(prompt_logits, dim=-1)  # (batch_size, prompt_len, vocab_size)
    if gen_logits is not None:
        gen_log_probs = torch.log_softmax(gen_logits, dim=-1)    # (batch_size, gen_len, vocab_size)

    # ==============================
    # Extraction of prompt log-probs
    # ==============================
    if mode in ("prompt", "promptGeneration"):
        # In prompt: logit at position t predicts token at t+1 (requires shifting)
        # Remove first token from target (no context to predict it)
        prompt_target_tokens = prompt_ids[:, 1:] # (batch_size, prompt_len - 1)

        prompt_attention_mask_shifted = prompt_attention_mask[:, 1:]  # (batch_size, prompt_len - 1)

        # Remove last logit position (since it predicts next token)
        prompt_pred_log_probs = prompt_log_probs[:, :-1, :] # shape: (batch_size, prompt_len - 1, vocab_size)

        # Retrieves, for each position and each batch, the log-probability corresponding to the next token 
        # (the one in target_tokens) from all the probas on the vocabulary.
        prompt_token_log_probs = prompt_pred_log_probs.gather(
            dim=2, index=prompt_target_tokens.unsqueeze(-1)
            ).squeeze(-1) # shape: (batch_size, prompt_len - 1)
      
        # Mask paddings
        prompt_token_log_probs = prompt_token_log_probs * prompt_attention_mask_shifted
        
    # ==============================
    # Extraction of generation log-probs
    # ==============================
    if mode in ("generation", "promptGeneration"):
        # In generation: logit at position t predicts token at position t (no shift needed)
        gen_token_log_probs = gen_log_probs.gather(
            dim=2, index=gen_ids.unsqueeze(-1)
            ).squeeze(-1)  # shape: (batch_size, gen_len)
        
        # Mask paddings
        gen_token_log_probs = gen_token_log_probs * gen_attention_mask
    

    # ==============================
    # Select log-probs according to mode
    # ==============================
    if mode == "promptGeneration":
        # Last logit of prompt from the forward pass == first logit of generation from `model.generate()`. 
        # To compute perplexity over the full sequence:
        # - Use prompt_token_log_probs (excluding final prompt token)
        # - Use gen_token_log_probs from generation
        # Concatenate both to form a complete sequence of predicted log-probs
        token_log_probs = torch.cat(
            [prompt_token_log_probs, gen_token_log_probs],  
            dim=1) # (batch_size, prompt_len - 1 + gen_len)
        total_mask = torch.cat(
            [prompt_attention_mask_shifted, gen_attention_mask],
            dim=1) # (batch_size, prompt_len - 1 + gen_len)
    
    elif mode == "prompt":
        token_log_probs = prompt_token_log_probs    # (batch_size, prompt_len - 1)
        total_mask = prompt_attention_mask_shifted  # (batch_size, prompt_len - 1)
    
    elif mode == "generation":
        token_log_probs = gen_token_log_probs  # (batch_size, gen_len)
        total_mask = gen_attention_mask        # (batch_size, gen_len)

    # ==============================
    # Compute Perplexity ignoring padded tokens
    # ==============================
    eps = 1e-12  # to avoid division by zero

    # Optionally focus only on the k% hardest tokens (lowest log-probs)
    if min_k is not None:
        # Keep only the min_k fraction of tokens with the lowest log-probs 
        k = int(min_k * token_log_probs.size(1))  # number of tokens to keep per sample
        
        # Exclude padding tokens from topk selection
        masked_log_probs = token_log_probs.clone()
        masked_log_probs[total_mask == 0] = 1e6  

        # Use topk with largest=False to get the k tokens with the lowest log-probabilities
        topk_vals, _ = torch.topk(masked_log_probs, k=k, dim=1, largest=False)

        # Compute perplexity using only the selected subset
        ppls = torch.exp(-topk_vals.mean(dim=1))

    else:
        # Compute perplexity over all predicted tokens
        sum_log_probs = (token_log_probs * total_mask).sum(dim=1)
        count = total_mask.sum(dim=1).clamp(min=eps)
        mean_log_prob = sum_log_probs / count
        ppls = torch.exp(-mean_log_prob)

    return ppls.cpu().numpy()




def compute_logit_entropy(
    prompt_logits: torch.Tensor,
    gen_logits: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    gen_attention_mask: torch.Tensor,
    mode: str = "prompt",
    prepend_last_prompt_logit: bool = False,
    top_k: int = None,
    window_size: int = None,
    stride: int = None
) -> np.ndarray:
    """
    Computes the per-sample entropy of a language model's output distributions
    using its logits and attention masks.
    For each token position, the function computes the entropy of the softmax distribution
    over the vocabulary. Entropy is averaged over the valid tokens (i.e., those marked
    as 1 in the attention mask). If `top_k` is specified, the entropy is computed only
    over the top-k logits (highest values) for each position.

    Entropy is defined as:
        Entropy = -Sum_i p_i * log(p_i)
        where p_i = softmax(logits)_i, i=1..vocab_size
    
    There are two main usage patterns:
      - Classic token-level average entropy (if window_size is None): computes the per-token entropy over the
        sequence, averages over all valid tokens per sample (optionally using top_k).
      - Windowed maximum mean entropy (if window_size is specified): slides a window of width `window_size`
        and stride `stride` (default equals window_size: non-overlapping windows, else user-specified) across
        the sequence of token entropies, and returns the maximum mean entropy observed in any window for each sample.

    Padding tokens are always ignored (via the provided attention masks); only windows where all tokens are valid
    are considered in the windowed mode.

    NOTE: This implementation is inspired by:
    "LLM-Check: Investigating Detection of Hallucinations in Large Language Models"
    (Sriramanan et al., 2024)
    
    Parameters
    ----------
    prompt_logits : torch.Tensor
        Tensor of shape (batch_size, prompt_len, vocab_size).
        These are the model's output logits obtained from a standard forward pass over the prompt sequence.
    gen_logits : torch.Tensor
        Tensor of shape (batch_size, gen_len, vocab_size).
        These are the logits obtained during autoregressive decoding using `model.generate()`.
    prompt_attention_mask : torch.Tensor
        Tensor of shape (batch_size, prompt_len). Contains 1 where the token is valid and 0 for padding.
    gen_attention_mask : torch.Tensor  
        Tensor of shape (batch_size, gen_len). Contains 1 where the token is valid and 0 for padding.
    mode : str, optional
        Which tokens to use for entropy computation:
        - "prompt": compute entropy only over the prompt logits/mask.
        - "generation": compute entropy only over the generated logits/mask.
        - "promptGeneration": compute entropy over both concatenated prompt and generated logits/mask.
    prepend_last_prompt_logit : bool, optional
        If True, appends the last logit from the prompt to the beginning of the generation logits.
        This is useful when generation logits were computed manually from hidden states 
        and are therefore shifted by one position (they lack the first prediction step so 
        the first logit is missing). Default is False.
        Carreful! The gen_attention_mask must match.
    top_k : int, optional
        If specified, only the top_k logits (per token) are used to compute the entropy.
        If None, use all logits.
    window_size : int, optional
        If not None, apply a sliding window of this size across the (valid) sequence of token entropies,
        and return the maximum mean entropy over any complete window, for each sample.
        If None, simply average the per-token entropies over all valid tokens.
    stride : int, optional
        Sliding window stride. Only used if window_size is specified.
        - If None, defaults to window_size (non-overlapping windows).
        - If set, must be a positive integer <= window_size.

    Returns
    -------
    np.ndarray
        Array of shape (batch_size,). For each batch sample, either the average logit entropy
        over valid tokens (if window_size is None) or the maximum windowed mean entropy (if window_size is given).

    Notes
    -----
    - Padding tokens are always ignored, both in classic and windowed entropy.
    - In windowed mode, only windows where all tokens in the window are valid are considered.
    - Uses torch.special.entr for numerically stable entropy calculation.
    """
    # ==============================
    # Move to device
    # ==============================
    prompt_logits = prompt_logits.to(prompt_attention_mask.device)
    if gen_logits is not None:
        gen_logits = gen_logits.to(gen_attention_mask.device)

    # Prepend last logit of prompt to the generation logits if specifed
    if prepend_last_prompt_logit:
        last_prompt_logit = prompt_logits[:, -1:, :] # (batch_size, 1, vocab_size)
        gen_logits = torch.cat([last_prompt_logit, gen_logits], dim=1) # (batch_size, gen_len+1, vocab_size)

    def entropy_from_logits(logits, attention_mask, top_k=None):
        """
        Parameters
        ----------
        logits: (batch_size, seq_len, vocab_size)
        attention_mask: (batch_size, seq_len)
        top_k: int > 0

        Returns
        -------
        entropy: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)
        """

        # Convert float16 -> float32 for better accuracy during computations
        logits = logits.float()
        attention_mask = attention_mask.float()

        if top_k is not None:
            topk_vals = torch.topk(logits, k=top_k, dim=-1).values  # (batch_size, seq_len, top_k)
            probs = F.softmax(topk_vals, dim=-1) # (batch_size, seq_len, top_k)
        else:
            probs = F.softmax(logits, dim=-1) # (batch_size, seq_len, vocab_size)

        # Use torch.special.entr, which automatically handles edge cases
        # entropy(x) = -x * log(x) with entropy(0) = 0
        entropy = torch.special.entr(probs).sum(dim=-1)  # (batch_size, seq_len)
        return entropy, attention_mask # both are (batch_size, seq_len)

    def average_entropy(entropy, mask):
        """
        Parameters
        ----------
        entropy: (batch_size, seq_len)
        mask: (batch_size, seq_len)
        
        Returns
        -------
        avg_entropy: (batch_size,)
        """
        entropy_masked = entropy * mask                    # (batch_size, seq_len)
        total_entropy = entropy_masked.sum(dim=-1)         # (batch_size,)
        valid_count = mask.sum(dim=-1)                     # (batch_size,)
        avg_entropy = total_entropy / (valid_count + 1e-9) # (batch_size,)
        return avg_entropy

    def max_sliding_window_entropy(entropy, mask, w, stride):
        """
        Parameters
        ----------
        entropy: (batch_size, seq_len)
        mask: (batch_size, seq_len)
        w: int > 0
        stride: int > 0

        Returns
        -------
        max_avg_entropy: (batch_size,)
        """
        # Add one dummy channel dimension since conv1d requires 3D tensors
        entropy = entropy.unsqueeze(1)  # (batch_size, 1, seq_len)
        mask = mask.unsqueeze(1)        # (batch_size, 1, seq_len)

        kernel = torch.ones(1, 1, w, device=entropy.device) / w  # shape: (1,1,w)

        # padding=0 to avoid artificial values and distorting the calculation
        # Ignore windows for which there are not enough elements to form a complete window.
        moving_avg = F.conv1d(entropy, kernel, stride=stride, padding=0)  # sliding mean entropy
        
        # All windows where there is at least one padding token will be ignored with valid_mask
        valid_counts = F.conv1d(mask, kernel, stride=stride, padding=0)   # sliding mean mask (valid token ratio)
        valid_mask = (valid_counts == 1.0)  # full valid windows only

        moving_avg = moving_avg.masked_fill(~valid_mask, float('-inf')) # put -inf where valid_mask==0

        max_avg_entropy, _ = moving_avg.max(dim=-1)  # (batch_size, 1)
        
        return max_avg_entropy.squeeze(1) # (batch_size,)

    if top_k is not None:
        top_k = int(top_k)
        if top_k <= 0 or top_k > prompt_logits.shape[2]:
            raise ValueError("top_k must be a positive integer less or equal to vocab size")
        
    if window_size is not None:
        if stride is None:
            stride = window_size
        else:
            stride = int(stride)
            if stride <= 0 or stride > window_size:
                raise ValueError("stride must be a positive integer less or equal to window_size.")
    else:
        stride = None

    if mode == "prompt":
        entropy, mask = entropy_from_logits(prompt_logits, prompt_attention_mask, top_k) # both are (batch_size, prompt_len)
    elif mode == "generation":
        entropy, mask = entropy_from_logits(gen_logits, gen_attention_mask, top_k)       # both are (batch_size, gen_len)
    elif mode == "promptGeneration":
        ent_p, mask_p = entropy_from_logits(prompt_logits, prompt_attention_mask, top_k) # both are (batch_size, prompt_len)
        ent_g, mask_g = entropy_from_logits(gen_logits, gen_attention_mask, top_k)       # both are (batch_size, gen_len)
        entropy = torch.cat([ent_p, ent_g], dim=1) # (batch_size, prompt_len + gen_len)
        mask = torch.cat([mask_p, mask_g], dim=1)  # (batch_size, prompt_len + gen_len)
    else:
        raise ValueError("mode must be in {'prompt','generation','promptGeneration'}")

    if window_size is None:
        result = average_entropy(entropy, mask)
    
    else:
        if window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        if window_size > entropy.shape[1]:
            raise ValueError("window_size greater than sequence length")
        if stride is None:
            stride = window_size
        else:
            stride = int(stride)
            if stride <= 0 or stride > window_size:
                raise ValueError("stride must be a positive integer less or equal to window_size.")
        
        window_size = int(window_size)
        result = max_sliding_window_entropy(entropy, mask, window_size, stride)
    return result.cpu().numpy()
