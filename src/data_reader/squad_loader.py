#!/usr/bin/env python3

"""
============================================================
Utilities for loading, transforming, and inspecting SQuAD-style datasets
============================================================

This module defines a `SquadDataset` class that wraps a Hugging Face dataset object
and ensures it follows a consistent SQuAD-like structure. It also provides convenience
methods for data preprocessing such as shuffling, slicing, filtering, and reformatting
answers. Additionally, it provides utility functions to load different ID and OOD
variants of the SQuAD dataset.

Main Features
-------------
- Ensures required dataset fields are present
- Provides helper methods to modify the dataset format
- Supports filtering for impossible questions (SQuAD v2)
- Simplifies downstream evaluation by standardizing structure

"""

import numpy as np
from datasets import load_dataset, Dataset
import pickle
import json

class SquadDataset:
    """
    A utility class to wrap a HuggingFace Dataset and enforce SQuAD-style structure.
    Automatically checks for required fields and provides utility methods.
    """

    REQUIRED_FIELDS = ['id', 'title', 'context', 'question', 'answers']

    def __init__(self, dataset: Dataset):
        """
        Initialize the HalluDataset wrapper.

        Parameters
        ----------
        dataset : datasets.Dataset
            A Hugging Face dataset object expected to contain SQuAD-style fields.
        """
        self.dataset = dataset
        self._validate_fields()

    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def __len__(self):
        """
        Return the number of examples in the dataset.
        """
        return len(self.dataset)

    def _validate_fields(self):
        """
        Ensure the dataset contains all required fields.

        Raises
        ------
        ValueError if any required field is missing.
        """
        for field in self.REQUIRED_FIELDS:
            if field not in self.dataset.features:
                raise ValueError(f"Missing required field: '{field}'")

    def add_original_index(self):
        """
        Add an 'original_index' field representing the original row index in the dataset.

        Returns
        -------
        SquadDataset
            Self with updated dataset.
        """
        self.dataset = self.dataset.map(lambda sample, idx: {"original_index": idx}, with_indices=True)
        return self

    def shuffle(self, seed:int=44):
        """
        Shuffle the dataset.

         Parameters
        ----------
        seed : int
            Seed for reproductibility

        Returns
        -------
        SquadDataset
        """
        self.dataset = self.dataset.shuffle(seed=seed)
        return self
    
    def select(self, indices):
        """
        Create a new SquadDataset containing only the rows at the specified indices
        
        Parameters
        ----------
        indices : Sequence[int]
            Indices of the rows to select (relative to the current dataset).

        Returns
        -------
        SquadDataset
            A new `SquadDataset` instance containing only the selected rows.
        """
        return SquadDataset(self.dataset.select(indices))

    def to_dict(self) -> dict:
        """
        Convert the underlying HuggingFace Dataset to a Python dictionary.

        Returns
        -------
        dict
            A dictionary representation of the dataset, where each key is a column name
            and each value is a list of values for that column.
        """
        return self.dataset.to_dict()

    def slice(self, idx_start: int, idx_end: int = None):
        """
        Select a range of rows from the dataset.

        Parameters
        ----------
        idx_start : int
            Start index (inclusive)
        idx_end : int
            End index (exclusive). If None, selects until the end of the dataset.

        Returns
        -------
        SquadDataset
        """
        dataset_len = len(self.dataset)
        if idx_end is None:
            idx_end = dataset_len
        idx_start = max(0, idx_start)
        idx_end = min(dataset_len, idx_end)
        if idx_start >= idx_end:
            self.dataset = self.dataset.select([])
            print("Return empty dataset")
            return self 
        indices = list(range(idx_start, idx_end))
        self.dataset = self.dataset.select(indices)
        return self


    def filter_by_column(self, column: str, values_to_keep: list):
        """
        Returns a new SquadDataset containing only the samples where the specified column's value
        is present in the provided list of values.

        Parameters
        ----------
        column : str
            The name of the key/field in each sample (dictionary) on which to filter (e.g., 'id', 'title').
        values_to_keep : list
            A list of values. Only samples whose `column` value is in this list will be retained.

        Returns
        -------
        SquadDataset
            A new SquadDataset instance containing only the filtered samples.
        """
        values_set = set(values_to_keep)
        indices = [i for i, ex in enumerate(self.dataset) if ex[column] in values_set]
        return self.select(indices)
    

    def save(self, path : str, format="pickle"):
        """
        Save the dataset to disk in either pickle or JSON format.

        Parameters
        ----------
        path : str
            The file path where the dataset will be saved.
        format : str, optional
            The format to use: 'pickle' (default) or 'json'.
        """
        if format == "pickle":
            with open(path, "wb") as f:
                pickle.dump(self, f)
                print(f"Dataset saved to {path}")
        elif format == "json":
            # Save only the data part (list of dicts)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.dataset, f, ensure_ascii=False, indent=2)
                print(f"Dataset saved to {path}")
        else:
            raise ValueError("Unsupported format: choose 'pickle' or 'json'")


    def filter_impossible(self):
        """
        Filter the dataset to keep only questions with no answer (i.e., empty 'answers["text"]').

        Returns
        -------
        SquadDataset
        """
        self.dataset = self.dataset.filter(lambda x: len(x["answers"]["text"]) == 0)
        return self

    def filter_possible(self):
        """
        Filter the dataset to keep only questions with at least one answer
        (i.e., non-empty 'answers["text"]').

        Returns
        -------
        SquadDataset
        """
        self.dataset = self.dataset.filter(lambda x: len(x["answers"]["text"]) > 0)
        return self

    def keep_first_answer_possible(self):
        """
        Keep only the first available answer for each question with answers.

        Returns
        -------
        SquadDataset
        """
        def _map_fn(sample):
            return {
                'answers': {
                    'text': sample['answers']['text'][0],
                    'answer_start': sample['answers']['answer_start'][0]
                }
            }
        self.dataset = self.dataset.map(_map_fn)
        return self

    def keep_first_answer_impossible(self):
        """
        Overwrite all answers with an impossible answer (empty string and -1).

        Returns
        -------
        SquadDataset
        """
        def _map_fn(_):
            return {
                'answers': {
                    'text': "",
                    'answer_start': -1
                }
            }
        self.dataset = self.dataset.map(_map_fn)
        return self

    def add_impossible_flag(self, value: int):
        """
        Add a binary 'is_impossible' flag to all examples.

        Parameters
        ----------
        value : int
            Either 0 (possible question) or 1 (impossible question)

        Returns
        -------
        SquadDataset
        """
        self.dataset = self.dataset.map(lambda x: {"is_impossible": value})
        return self

    def print_info(self):
        """
        Print dataset structure, stats about answer and context lengths,
        and list samples with multiple answers (if applicable).
        """
        print("\n===== Dataset Information =====")
        print(self.dataset)

        # Skip samples with empty text to avoid index errors
        valid_texts = [ans['text'] for ans in self.dataset['answers'] if ans['text']!= ""]
        lengths = [len(text.split()) for text in valid_texts]

        if lengths:
            print(f"Mean ground-truth answer length: {np.mean(lengths):.2f}, Max length: {max(lengths)}")
        else:
            print("No valid ground-truth answers to compute length stats.")

        context_question_lengths = [len((ex['context'] + ' ' + ex['question']).split()) for ex in self.dataset]
        print(f"Mean context + question length: {np.mean(context_question_lengths):.2f}, Max length: {max(context_question_lengths)}")

    def get(self):
        """
        Get the internal HuggingFace Dataset object.

        Returns
        -------
        datasets.Dataset
        """
        return self.dataset


# Dataset loading wrappers using SquadDataset
'''
def load_id_fit_dataset():
    """
    Load the training split of SQuAD v1.1 and return it wrapped in SquadDataset.
    Adds index and formats answers for consistency.

    Returns
    -------
    SquadDataset
    """
    ds = load_dataset("squad")['train']
    return SquadDataset(ds)\
        .add_original_index()\
        .keep_first_answer_possible()\
        .add_impossible_flag(0)


def load_id_test_dataset():
    """
    Load the validation split of SQuAD v1.1.
    Useful for evaluating model accuracy on in-distribution examples.

    Returns
    -------
    SquadDataset
    """
    ds = load_dataset("squad")['validation']
    return SquadDataset(ds)\
        .add_original_index()\
        .keep_first_answer_possible()\
        .add_impossible_flag(0)


def load_od_test_dataset():
    """
    Load the training split of SQuAD v2.0 and extract only unanswerable questions.
    Sets the "is_impossible" flag to 1.

    Returns
    -------
    SquadDataset
    """
    ds = load_dataset("squad_v2")['train']
    return SquadDataset(ds)\
        .filter_impossible()\
        .add_original_index()\
        .add_impossible_flag(1)\
        .keep_first_answer_impossible()
'''


def load_id_fit_dataset():
    """
    Load the training split of SQuAD v2.0, keep only answerable questions,
    and return the result wrapped in a SquadDataset.
    Adds index and formats answers for consistency.

    Returns
    -------
    SquadDataset
    """
    ds = load_dataset("squad_v2")['train']
    return SquadDataset(ds)\
        .filter_possible()\
        .add_original_index()\
        .keep_first_answer_possible()\
        .add_impossible_flag(0)


def load_id_test_dataset():
    """
    Load the validation split of SQuAD v2.0, keep only answerable questions,
    and return the result wrapped in a SquadDataset.
    Useful for evaluating model accuracy on in-distribution examples.

    Returns
    -------
    SquadDataset
    """
    ds = load_dataset("squad_v2")['validation']
    return SquadDataset(ds)\
        .filter_possible()\
        .add_original_index()\
        .keep_first_answer_possible()\
        .add_impossible_flag(0)


def load_od_test_dataset():
    """
    Load the validation split of SQuAD v2.0, keep only unanswerable questions,
    and return the result wrapped in a SquadDataset.
    Sets the "is_impossible" flag to 1.
    Useful for evaluating model accuracy on out-of-distribution examples.

    Returns
    -------
    SquadDataset
    """
    ds = load_dataset("squad_v2")['validation']
    return SquadDataset(ds)\
        .filter_impossible()\
        .add_original_index()\
        .add_impossible_flag(1)\
        .keep_first_answer_impossible()
