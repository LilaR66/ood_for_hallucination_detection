#!/usr/bin/env python3
"""
============================================================
Prompt Building, Batch Extraction, and Controlled Generation Utilities
============================================================

This module provides utilities to build structured instruction-style prompts for question 
answering with LLMs (especially Llama models), extract batches from HuggingFace datasets as
lists of dictionaries, run controlled autoregressive generation and construct attention
masks for generated token sequences.

Key features:
-------------
- Build instruction-style prompts featuring Llama-specific formatting tokens 
  ([INST], [/INST], <<SYS>>) following recommended conventions for Llama 2/3 prompt
  formats and relevant literature.
- Extract contiguous batches of examples from HuggingFace Datasets, converting 
  them into plain Python list-of-dictionaries for convenient downstream processing.
- Generate sequences from language models with flexible configurations
  The generation setup favors memory-efficiency and hook-based extraction of hidden 
  states and attentions by disabling model generation outputs such as hidden states and 
  attentions directly.
- Build generation attention masks that identify valid generated tokens up to and 
  including the first EOS token, serving for alignment and filtering in analysis pipelines.
"""


from transformers import PreTrainedTokenizer, PreTrainedModel, BatchEncoding
import torch
from datasets import  Dataset
from tqdm import tqdm
from typing import Dict, List, Any, Callable, Tuple, Union, Literal
import time

from src.answer_similarity.similarity_metrics import rouge_l_simScore, sentence_bert_simScore
from src.data_reader.pickle_io import save_batch_pickle
from src.inference.offset_utils import compute_offset_attention_mask


# Specific to Llama tokenizer: 
'''
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
    prompt = f"[INST] <<SYS>>\nJust give the answer, without a complete sentence. Reply with 'Impossible to answer' if answer not in context.\n<</SYS>>\n\nContext:\n" + context + "\n\nQuestion:\n" + question  + "\n\nAnswer:\n[/INST]" 
    return prompt
'''

# Specific to Llama tokenizer:
def build_prompt(context:str, question:str) -> str:
    """
    Construct a structured prompt for question answering with an LLM.

    The prompt includes the special formatting: `[INST]`, [/INST]`, `<<SYS>>`
    as recommended here: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

    The prompt is formatted according to this paper:
    "The Curious Case of Hallucinatory (Un)answerability: Finding Truths in
    the Hidden States of Over-Confident Large Language Models (2023)" 

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
    B_INST = "[INST]"
    E_INST = "[/INST]"
    B_SYS = "<<SYS>>\n"
    E_SYS = "\n<</SYS>>\n\n"
    HINT = "Given the following passage and question, answer the question by only giving the answer without a complete sentence.\nIf it cannot be answered based on the passage, reply 'unanswerable':"
    prompt = f"{B_INST} {B_SYS}{HINT}{E_SYS}Passage: {context}\nQuestion: {question} {E_INST}"
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
            pad_token_id=tokenizer.eos_token_id, # Ensures clean padding (right padding)
            output_hidden_states=False,      # We rely on the hook to extract hidden states instead (more memory efficient)
            output_attentions=False,         # We rely on the hook to extract attention map instead (more memory efficient)
            output_logits=True,              # Logits not filtered/truncated by top-k/top-p sampling. Note: `output_scores=True` returns filtered logits. 
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



def apply_logit_lens(
        model: PreTrainedModel, 
        hidden_states: torch.Tensor
    )-> torch.Tensor:
    """
    Applies the model's LM head to hidden states to produce logits.

    Args:
        model: PreTrainedModel with `lm_head` attribute.
        hidden_states: Tensor (batch_size, seq_len, hidden_size).

    Returns:
        logits: Tensor (batch_size, seq_len, vocab_size).

    NOTE: 
    We do not apply layer norm to match transformers llama implementation
    """
    # Apply LM head (linear projection)
    logits = model.lm_head(hidden_states)
    return logits