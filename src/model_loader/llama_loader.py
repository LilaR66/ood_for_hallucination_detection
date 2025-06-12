#!/usr/bin/env python3

"""
===========================================
Loads a Hugging Face LLaMA model and tokenizer
===========================================

This module provides a helper function to load a LLaMA-like causal language model
along with its tokenizer using Hugging Face's Transformers library.

Usage:
-------
from llama_loader import load_llama
model, tokenizer = load_llama("meta-llama/Llama-2-7b-chat-hf")

Notes
-----
- Requires prior Hugging Face authentication via: `huggingface-cli login`
- The default model requires structuring prompts with special tags: 
    `<s>[INST] "Instruction here" [/INST] "Generated answer here"`
- Model loading may use multiple GPUs automatically if available.
- For debugging model cache size: 
    `watch -n 1 du -sh ~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/`
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_llama(model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
    model_name = model_name  # fine-tuned version of LLaMA for conversational uses

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    tokenizer.model_max_length = 1024  # LLaMA-2’s max length tokens is 4096
    tokenizer.pad_token = tokenizer.eos_token  # pad_token not defined by default: reuse the EOS token (</s>) as the padding token.

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16, 
        device_map="auto",          # load model to device 
        low_cpu_mem_usage=True,     # reduce RAM usage during loading
        #output_hidden_states=True, # to hidden activations -> memory overload since we access ALL hidden states 
        #force_download=True        # redo complete download 
    )

    model.config.pad_token_id = model.config.eos_token_id # ensures that during generation all sequences are aligned with the EOS token, and not with random tokens. 
    
    return model, tokenizer