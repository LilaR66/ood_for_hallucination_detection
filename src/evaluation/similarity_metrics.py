#!/usr/bin/env python3
"""
============================================================
Utilities for evaluating answer similarity in QA tasks
============================================================

This module provides a collection of functions for computing 
various similarity and matching scores between model-generated answers and 
ground-truth references in question answering (QA) settings. 

Functions
---------
- `normalize_answer`: Standardizes text by removing punctuation, articles, and extra whitespace.
- `exact_match_simScore`: Computes strict match (EM) between normalized prediction and reference.
- `f1_simScore`: Calculates token-level F1 overlap between prediction and ground truth.
- `compute_fuzzy_f1_simScore`: Computes fuzzy character-level similarity using SequenceMatcher.
- `compute_rouge_l_simScore`: Computes ROUGE-L F1 score for sequence overlap.
- `compute_sentence_similarity_simScore`: Computes semantic similarity using SBERT embeddings.
"""
import re
import string
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from difflib import SequenceMatcher
import pandas as pd


def normalize_answer(s: str) -> str:
    """
    Normalize a text string by lowercasing, removing punctuation, articles, and extra whitespace.

    Parameters
        s : str
            Input text string to normalize.
    Returns
        str
            Normalized version of the input string.
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text): # removes spaces, new lines, tabs
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_simScore(prediction: str, ground_truth: str) -> int:
    """
    Compute the Exact Match (EM) score between two strings after normalization.

    Parameters
        prediction : str
            Model-generated answer.
        ground_truth : str
            Reference or correct answer.

    Returns
        int
            1 if the normalized prediction matches ground truth exactly, otherwise 0.
    """
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_simScore(prediction: str, ground_truth: str) -> float:
    """
    Compute the word-level F1 score between a predicted and ground-truth answer,
    after normalization and tokenization.

    Parameters
        prediction : str
            Model-generated answer.
        ground_truth : str
            Reference or correct answer.

    Returns
        float
            F1 similarity score between 0.0 and 1.0.
    """
    # normalize and split per words
    pred_tokens = normalize_answer(prediction).split() 
    gt_tokens = normalize_answer(ground_truth).split()
    # count common words 
    common = set(pred_tokens) & set(gt_tokens) 
    # count how many overlapping words appear, considering duplicates.
    num_same = sum(min(pred_tokens.count(w), gt_tokens.count(w)) for w in common) 

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return int(pred_tokens == gt_tokens)
    if num_same == 0: # if no tokens matched, F1 is 0
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def fuzzy_f1_simScore(prediction: str, ground_truth: str) -> float:
    """
    Compute a fuzzy character-level similarity score between two strings
    using Levenshtein-like matching.

    Parameters
        pred : str
            Model-generated answer.
        ref : str
            Reference answer.

    Returns
        float
            Fuzzy match score between 0.0 and 1.0.
    """

    return SequenceMatcher(None, normalize_answer(prediction), normalize_answer(ground_truth)).ratio()


def rouge_l_simScore(prediction: str, ground_truth: str) -> float:
    """
    Compute the ROUGE-L F1 score between prediction and reference,
    based on the longest common subsequence.

    Parameters
        prediction : str
            Model-generated answer.
        ground_truth : str
            Reference answer.

    Returns
        float
            ROUGE-L F1 score between 0.0 and 1.0.
    """
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(ground_truth, prediction)
    return scores['rougeL'].fmeasure  # Can also return precision or recall if needed


# Load Sentence-BERT model (medium-sized, general-purpose)
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

def sentence_bert_simScore(prediction: str, ground_truth: str) -> float:
    """
    Compute semantic similarity between two strings using Sentence-BERT and cosine similarity.

    Parameters
        prediction : str
            Model-generated answer.
        ground_truth : str
            Reference answer.

    Returns
        float
            Cosine similarity score between 0.0 and 1.0.
    """
    embeddings = sbert_model.encode([prediction, ground_truth], convert_to_tensor=True)
    cosine_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return float(cosine_sim)
