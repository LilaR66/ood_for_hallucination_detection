#!/usr/bin/env python3
"""
============================================================
Custom Llama Attention Patch and Hooks for Activation and Attention Extraction
============================================================
This module provides tools for reliable extraction of attention weights and hidden states 
from Hugging Face Llama language models during autoregressive generation, addressing 
numerical instability issues encountered with the default eager attention implementation.

NOTE: 
    Problem: When working with Hugging Face's Llama models
    -------
    The default "eager" attention backend exposes attention weights but suffers from numerical 
    instability leading to NaNs in hidden states and CUDA crashes, especially with mixed-precision 
    (fp16) training and inference. The more stable "sdpa" attention backend does not expose attention
    weights, making analysis of attention maps impossible via that route.

    Solution
    --------
    To achieve both stability and transparency:
    - Implement a custom patch for the LlamaAttention forward method, but only on the specific layers 
    where we wanted to access attention weights. The main computation of hidden states uses the stable 
    backend (sdpa by default). This ensures the forward pass and generated sequences remain numerically 
    stable.
    - In parallel, the patch computes attention weights using the "eager" mechanism, but solely to extract 
    and return them for inspection. These weights are not used in the model's forward pass and do not 
    affect generation, so any instabilities or NaN handling for these analytical values do not impact 
    the model's outputs.

Features
--------
- `patched_LlamaAttention_forward`: Wraps the original LlamaAttention with dual computation: stable output 
plus attention weights extraction.
- `custom_eager_attention_forward`: Computes attention weights in the eager mode for inspection.
- Hook registration functions to capture hidden states and attention weights during generation at 
specified layers efficiently.
- Utilities to verify that hooks are properly triggered during generation.
"""

from typing import Tuple, List, Optional, Dict
from transformers import PreTrainedModel
from torch.utils.hooks import RemovableHandle

import torch
from torch import nn 
from typing import Callable, Optional, Tuple, Unpack
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, eager_attention_forward, repeat_kv
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.utils import logging
logger = logging.get_logger(__name__)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS



def custom_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    return attn_weights




def patched_LlamaAttention_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:  
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        
        # ========================================================
        # [1] Forward pass using main attention backend (sdpa / flash)
        # This is the output used in the model's autoregressive loop.
        # These implementations are optimized (for memory + stability).
        # Does not compute attn_weights.
        # ========================================================
        attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        # ========================================================
        # [2] Parallel computation of attention weights using eager attention
        # This is to retrieve attention weights only (not used in forward loop)
        # It is more numerically unstable (NaN possible with fp16)
        # ========================================================
        try:
            attn_weights = custom_eager_attention_forward(
                self,
                query_states, 
                key_states, 
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling, 
                **kwargs,
            )

            # Replace NaNs (if any) by 0.0 (no attention)
            if torch.isnan(attn_weights).any():
                print("[WARN] NaNs detected in attn_weights — replacing with 0.0 (no renormalization)")
                attn_weights = attn_weights.masked_fill(torch.isnan(attn_weights), 0.0)

        except Exception as ex:
            print(f"[ERROR] Exception in custom_eager_attention_forward: {ex}")
            attn_weights = None

        return attn_output, attn_weights




def register_activation_hook(
    model: PreTrainedModel,
    captured_hidden_lists: List[List[torch.Tensor]],  
    layers_idx_list: List[int]
) -> Tuple[List[RemovableHandle], List[Dict[str,int]]]:
    """
    Attaches a forward hook to a specific transformer layer to capture hidden states
    during autoregressive text generation i.e., at each decoding step.
    (more memory-efficient than using output_hidden_states=True).
    Transformer layer = self-attention + FFN + normalization.

    Parameters
    ----------
    model : PreTrainedModel
        The Hugging Face causal language model (e.g., GPT, LLaMA).
    captured_hidden_lists : List[List[torch.Tensor]]
        A list containing one list per hooked layer.
        Each inner list will be filled with hidden states for each generation step,
        each tensor of shape (batch_size, seq_len, hidden_size).
    layers_idx_list : List[int]
        List of transformer block indices to hook.
        Use -1 to denote the last layer.
    
    Returns
    -------
    List[RemovableHandle]
        List of handle objects to remove hooks after generation by calling `handle.remove()`.
    List[Dict[str, int]]
        List of counters (dicts with key 'count') storing how many times each hook was activated.
    """
    handles = []
    call_counters = [] # count how many times the hook is triggered

    # Raise error if layer_idx not in correct range
    num_layers = len(model.model.layers)
    for idx in layers_idx_list:
        if not (idx == -1 or 0 <= idx < num_layers):
            raise ValueError(f"`layer_idx` must be -1 or in [0, {num_layers-1}] but got {idx}")

        call_counter = {"count": 0}  
        call_counters.append(call_counter)

        def hook_fn(module, input, output, call_counter=call_counter, idx=idx):
            """Function called automatically by PyTorch just after
            the layer has produced its output during the forward pass."""
            
            call_counter["count"] += 1
            # output is a tuple (hidden_states,) -> keep [0]
            if idx == -1:
                # Capture the final normalized output 
                captured_hidden_lists[layers_idx_list.index(idx)].append(
                    model.model.norm(output[0]).detach())  # post RMSNorm!
            else: 
                # Capture raw hidden states before layer normalization
                captured_hidden_lists[layers_idx_list.index(idx)].append(output[0].detach())

        # Register hook on the transformer block
        # When Pytorch pass through this layer during forward pass, it also execute hook_fn.
        handle = model.model.layers[idx].register_forward_hook(hook_fn)
        handles.append(handle)

    return handles, call_counters




def register_attention_hook(
    model: PreTrainedModel,
    captured_attn_lists: List[List[torch.Tensor]],
    layers_idx_list: List[int]
) -> Tuple[List[RemovableHandle], List[Dict[str, int]]]:
    """
    Attaches forward hooks to multiple specified Llama layers' self-attention modules
    to capture attention maps (weights) during autoregressive text generation.

    Parameters
    ----------
    model : PreTrainedModel
        The Hugging Face causal language model (e.g., Llama 2).
    captured_attn_lists : List[List[torch.Tensor]]
        A list containing one list per hooked layer.
        Each inner list will receive attention tensors after each decoding step.
        Each tensor shape: (batch_size, n_heads, tgt_seq_len, src_seq_len)
    layers_idx_list : List[int]
        List of indices of layers to hook.
        Use -1 to denote the last layer.

    Returns
    -------
    List[RemovableHandle]
        List of handle objects to remove hooks after generation by calling `handle.remove()`.
    List[Dict[str, int]]
        List of counters (dicts with key 'count') storing how many times each hook fired.
    """
    num_layers = len(model.model.layers)
    handles = []
    call_counters = [] # count how many times the hook is triggered

    # Raise error if layer_idx not in correct range
    for i, idx in enumerate(layers_idx_list):
        if idx == -1:
            idx = num_layers - 1
        if not (0 <= idx < num_layers):
            raise ValueError(f"`layer_idx` must be -1 or in [0, {num_layers - 1}], but got {idx}.")

        call_counter = {"count": 0}
        call_counters.append(call_counter)

        def attn_hook_fn(module, input, output, call_counter=call_counter, list_idx=i):
            """Hook: captures the attention weights after the forward pass.
            For Llama (transformers >=4.31/hf), output is a tuple:
            (attn_output, attn_weights)"""
            call_counter["count"] += 1
            attn_weights = output[1]  # (batch, n_heads, tgt_seq_len, src_seq_len)
            captured_attn_lists[list_idx].append(attn_weights.detach())

        # The attention submodule for Llama
        attention_module = model.model.layers[idx].self_attn
        # Register hook on the Attention block
        # When Pytorch pass through this layer during forward pass, it also execute attn_hook_fn.
        handle = attention_module.register_forward_hook(attn_hook_fn)
        handles.append(handle)

    return handles, call_counters




def verify_call_counters(
        call_counters: List[Dict[str, int]], 
        name: str = "hooks"
    ) -> None:
    """
     Checks that all call counters are > 0 and equal.

    Args:
        call_counters: List of dictionaries with the key 'count'.
        name: Descriptive name for the error message.
    
    Raises:
        RuntimeError if a counter is 0 or if the counters differ.
    """
    if not all(counter['count'] > 0 for counter in call_counters):
        raise RuntimeError(f"At least one {name} did not capture any events.")
    
    counts = [counter['count'] for counter in call_counters]
    if len(set(counts)) > 1:
        raise RuntimeError(f"{name.capitalize()} have inconsistent call counts: {counts}")

