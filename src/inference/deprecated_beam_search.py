#!/usr/bin/env python3
"""
Old version of inference_utils with functions supporting only the extraction of a 
single hidden activation score per run. Unlike the current version, which allows 
retrieving multiple types of scores simultaneously (e.g., minimum, maximum, last, 
or first generated), this version returns only one score at a time. It also includes
beam search for generation, a feature no longer used in the new inference_utils. 
Furthermore, the associated hook can capture activations from only one layer at a time,
not multiple layers simultaneously.
"""
from transformers import PreTrainedTokenizer, PreTrainedModel
import torch
from datasets import  Dataset
from tqdm import tqdm
from typing import List, Callable, Union, Literal, Tuple
from torch.utils.hooks import RemovableHandle

from src.data_reader.pickle_io import save_batch_pickle
from src.inference.offset_utils import compute_offset_attention_mask

from src.inference.generation_utils import (
    extract_batch, 
    generate, 
    build_generation_attention_mask
    )


# OLD VERSION: HOOKS ONLY 1 LAYER
def register_forward_activation_hook(
    model: PreTrainedModel,
    captured_hidden: dict,
    layer_idx: int = -1
) -> Tuple[RemovableHandle, dict]:
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
            captured_hidden["activations"] = model.model.norm(output[0].detach())  # post RMSNorm!
        else:
            captured_hidden["activations"] = output[0].detach()

    # Register hook on the transformer block
    # When Pytorch pass through this layer during a forward pass, it also execute hook_fn.
    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)

    return handle, call_counter



# OLD VERSION: HOOKS ONLY 1 LAYER
def register_generation_activation_hook(
    model: PreTrainedModel,
    captured_hidden_list: List[torch.Tensor],
    layer_idx: int = -1
) -> Tuple[RemovableHandle, dict]:
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
            captured_hidden_list.append(model.model.norm(output[0]).detach())  # post RMSNorm!
        else:
            # Capture raw hidden states before layer normalization
            captured_hidden_list.append(output[0].detach()) 
    
    # Register hook on the transformer block
    # When Pytorch pass through this layer during forward pass, it also execute hook_fn.
    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    
    return handle, call_counter



# OLD VERSION: FOR BEAM SEARCH ONLY
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



# OLD VERSION: FOR BEAM SEARCH ONLY
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



# OLD VERSION: HOOKS ONLY 1 LAYER, COMPUTE ONLY ONE ACTIVATION SCORE
def run_prompt_activation_extraction(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    batch_size: int = 4,
    idx_start_sample: int = 0,
    max_samples: int = 1000,
    save_to_pkl: bool = False,
    output_path: str = "outputs/all_batch_results",
    build_prompt_fn: Callable[[str, str], str] = None,
    register_forward_activation_hook_fn: Callable = None,
    layer_idx: int = -1,  
    extract_token_activations_fn: Callable = None,
    start_offset: int = 0,
    end_offset: int = 0
) -> Union [Tuple[List[torch.Tensor]], None]:
    """
    Runs batched inference on a dataset using a decoder-only language model.
    For each batch, it extracts token-level hidden activations 
    (from the prompt only) from a specified transformer layer.

    Hidden states are captured via a forward hook during a single forward pass.
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
    register_forward_activation_hook_fn : Callable
        Function that registers a forward hook on the model during a forward pass. 
    layer_idx : int
        Index of the transformer layer to extract activations from (default: -1 for last layer).
    extract_token_activations_fn : Callable
        Function that selects and aggregates token-level activations. 
    start_offset : int
        Offset from the first non-padding token (must be >= 0). 
    end_offset : int
        Offset from the last non-padding token (must be <= 0, e.g., -3 to remove 3 tokens).
    
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
        prompt_attention_mask = inputs["attention_mask"]

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

        layer_output = captured_hidden["activations"]  # Shape: (batch_size, seq_len, hidden_size)

        # ===============================
        # Modify prompt attention mask with offsets
        # ===============================
        if start_offset !=0 or end_offset !=0:
            prompt_attention_mask, start_indices, end_indices = compute_offset_attention_mask(
                attention_mask=prompt_attention_mask, 
                start_offset=start_offset, 
                end_offset=end_offset
            ) # Shape (batch_size, seq_len), (batch_size,), (batch_size,)
          
        # ==============================
        # Extract token activations from captured layer
        # ==============================
        selected_token_vecs = extract_token_activations_fn(
                selected_layer=layer_output, 
                attention_mask=prompt_attention_mask, 
                device=layer_output.device,
            )  # Shape (batch_size, hidden_size)
        
        # ==============================
        # Store results (to file or memory)
        # ==============================
        activations = [selected_token_vecs[j].unsqueeze(0).cpu().numpy() for j in range(selected_token_vecs.size(0))]
        batch_dataset_ids = [s['id'] for s in batch]
        batch_dataset_original_idx = [s['original_index'] for s in batch]
        
        batch_results = {
            "id": batch_dataset_ids,
            "original_indices": batch_dataset_original_idx,
            "activations": activations 
        }

        if save_to_pkl:
            #append_to_pickle(output_path, batch_results)
            save_batch_pickle(batch_data=batch_results, output_dir=output_path, batch_idx=i)
        else:
            batch_activations.extend(activations)
        
    if not save_to_pkl:
        return batch_activations



# OLD VERSION: HOOKS ONLY 1 LAYER, COMPUTE ONLY ONE ACTIVATION SCORE
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
    start_offset : int = 0,
    end_offset : int = 0,
) -> Union[List[torch.Tensor], None]:
    """
    Runs batched inference on a dataset using a decoder-only language model.
    For each batch, it performs text generation and extracts token-level hidden activations 
    (both from the prompt and the generated text depending on `activation_source`) 
    from a specified transformer layer.

    Hidden states are captured via a forward hook during generation, then aligned and 
    filtered using attention masks. 
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
    start_offset : int
        Offset from the first non-padding token (must be >= 0). 
    end_offset : int
        Offset from the last non-padding token (must be <= 0, e.g., -3 to remove 3 tokens).
    
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
        
        # Retrieve text of generated answers
        gen_answers = tokenizer.batch_decode(
            outputs.sequences[:, prompt_len:], 
            skip_special_tokens=True
        ) # Shape: [batch_size,]
        
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
            beam_indices=truncated_beam_indices if k_beams > 1 else None,
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
        
        # ===============================
        # Modify prompt attention mask with offsets
        # ===============================
        if start_offset !=0 or end_offset !=0:
            prompt_attention_mask, start_indices, end_indices = compute_offset_attention_mask(
                attention_mask=prompt_attention_mask, 
                start_offset=start_offset, 
                end_offset=end_offset
            ) # Shape (batch_size, prompt_len), (batch_size,), (batch_size,)

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
                ) # Shape (batch_size, hidden_size)
            
        elif activation_source == "prompt":    
            # Return only the token activations from the prompt
            selected_token_vecs = extract_token_activations_fn(
                    selected_layer=aligned_prompt_hidden_states, 
                    attention_mask=prompt_attention_mask, 
                    device=aligned_prompt_hidden_states.device,
                ) # Shape (batch_size, hidden_size)
            
        elif activation_source == "promptGeneration":
            # Return token activations from the concatenated prompt + generated answer 
            selected_token_vecs = extract_token_activations_fn(
                    selected_layer=aligned_prompt_and_gen_hidden_states, 
                    attention_mask=prompt_and_gen_attention_mask, 
                    device=aligned_prompt_and_gen_hidden_states.device,
                    skip_length=prompt_len 
                    # skip_length: exclude prompt from computation if 
                    # mode=='first_generated' in `extract_token_activations_fn`
                ) # Shape (batch_size, hidden_size)

        else:
            raise ValueError(
                f"Invalid value for `activation_source`: '{activation_source}'. "
                f"Expected one of: ['prompt', 'generation', 'promptGeneration']."
            )    
        
        # ==============================
        # Store results (to file or memory)
        # ==============================
        activations = [selected_token_vecs[j].unsqueeze(0).cpu().numpy() for j in range(selected_token_vecs.size(0))]

        batch_dataset_ids = [s['id'] for s in batch]
        batch_dataset_original_idx = [s['original_index'] for s in batch]
        batch_context = [s['context'] for s in batch]
        batch_question = [s['question'] for s in batch]
        batch_gt_answers = [s['answers'] for s in batch]
        
        batch_results = {
            "id": batch_dataset_ids,
            "original_indices": batch_dataset_original_idx,
            "activations": activations,
            "gen_answers": gen_answers,
            "gt_answers": batch_gt_answers,
            "context": batch_context,
            "question": batch_question
        }

        if save_to_pkl:
            #append_to_pickle(output_path, batch_results)
            save_batch_pickle(batch_data=batch_results, output_dir=output_path, batch_idx=i)
        else:
            batch_activations.extend(activations)
        
    if not save_to_pkl:
        return batch_activations
