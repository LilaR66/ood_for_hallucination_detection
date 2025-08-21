#!/usr/bin/env python3
"""
============================================================
Analyze OOD Grid — Config Discovery, Descriptor Access, & Sweeps
============================================================

This module centralizes everything needed to **enumerate**, **extract**, and
**evaluate** configurations for out-of-distribution (OOD) detection across
layers / groups / aggregations discovered directly from the `descriptors` tree.

What it contains
----------------
- `discover_config_space(data_dict)`: 
    Parse the `descriptors` structure to list available layers and, per layer,
    the available aggregations for each group (hidden / attention / logit).

- `extract_descriptors(data_dict, layer, group, aggregation)`:
    Robust accessor that returns the numpy array for a given (layer, group, agg),
    with informative errors if the path does not exist or the agg is under the
    wrong group.

- `grid_search_best_config(...)`:
    Full sweep over all discovered (layer, group, agg), computes OOD scores using
    a chosen method (DKNN, cosine, Mahalanobis, OCSVM, IsolationForest, or
    raw scalar scores), then reports AUROC / FPR@95 / AUPRC and thresholded
    metrics (accuracy / F1 / precision / recall). Also performs **linear probing**
    (logistic regression) on the descriptors themselves.

- `grid_search_best_config(...)`:
    consolidates multiple result tables produced by
    (typically one per random seed/run) and
    summarizes performance **per unique configuration**.

Expected input structure
------------------------
Each split dict (e.g., `id_fit_data`, `id_test_data`, `od_test_data`) must include:

    {
      "descriptors": {
        "layer_{i}": {
          "hidden": {
            "<hidden_agg>": np.ndarray[(n_samples, feat_dim), float]
                         or  np.ndarray[(n_samples,), float]  # if pre-scored
          },
          "attention": {
            "<attn_agg>":  np.ndarray[(n_samples,), float]
          },
          "logit": {
            "<logit_agg>": np.ndarray[(n_samples,), float]
          }
        },
        ...
      }
    }

Conventions
-----------
- **Group names**: "hidden" | "attention" | "logit"
- **Shape**: hidden → 2D `(n, d)`; attention/logit → 1D `(n,)` (scalar per sample)
- **OOD score sign**: larger score means **more OOD**
- If descriptors are already **1D scalar scores**, the grid enforces `method='raw_scores'`
  for that config (no extra OOD method run).
"""

from __future__ import annotations

from typing import Dict, Optional, Any, List, Literal, Union
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from src.analysis.evaluation import compute_metrics, compute_confusion_matrix_and_metrics
from src.ood_methods.logistic_regression import train_logistic_regression_on_descriptors
from src.ood_methods.ood_main import compute_ood_scores
from src.utils.debug import _describe_array

 

def discover_config_space(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Discover the available configuration space directly from the key `descriptors` 
    from the data_dict dictionary

    Parameters
    ----------
    data_dict : Dict[str, Any]
        One split's dictionary containing the `descriptors` key with the structure: 
        {
            "descriptors": {
                "layer_{i}": {
                    "hidden": {
                        "<hidden_agg>": np.ndarray[(n_samples, feat_dim), float]
                                     or np.ndarray[(n_samples,), float]
                        ...
                    },
                    "attention": {
                        "<attn_agg>": np.ndarray[(n_samples,), float]
                        ...
                    },
                    "logit": {
                        "<logit_agg>": np.ndarray[(n_samples,), float]
                        ...
                    }
                },
            }
        }

    Returns
    -------
    space : Dict[str, Any]
        {
          "layers": List[int],
          "hidden_aggs_by_layer": Dict[int, List[str]],
          "attn_aggs_by_layer": Dict[int, List[str]],
          "logit_aggs_by_layer": Dict[int, List[str]]
        }

    """
    assert "descriptors" in data_dict, "data_dict must contain key 'descriptors'."
    S = data_dict["descriptors"]

    # Extract layer indices from keys like "layer_12"
    layers: List[int] = []
    for k in S.keys():
        if k.startswith("layer_"):
            try:
                layers.append(int(k.split("_", 1)[1]))
            except Exception:
                pass
    layers = sorted(set(layers))

    hidden_aggs_by_layer: Dict[int, List[str]] = {}
    attn_aggs_by_layer: Dict[int, List[str]] = {}
    logit_aggs_by_layer: Dict[int, List[str]] = {}

    for li in layers:
        layer_node = S.get(f"layer_{li}", {})

        # Hidden aggregations: 2D descriptors (n_samples, feat_dim)
        hidden_aggs_by_layer[li] = (
            sorted(list(layer_node.get("hidden", {}).keys()))
            if isinstance(layer_node.get("hidden"), dict) else []
        )

        # Attention "scores": 1D descriptors (n_samples,)
        attn_aggs_by_layer[li] = (
            sorted(list(layer_node.get("attention", {}).keys()))
            if isinstance(layer_node.get("attention"), dict) else []
        )

        # Logits "scores": 1D descriptors (n_samples,)
        logit_aggs_by_layer[li] = (
            sorted(list(layer_node.get("logit", {}).keys()))
            if isinstance(layer_node.get("logit"), dict) else []
        )

    return {
        "layers": layers,
        "hidden_aggs_by_layer": hidden_aggs_by_layer,
        "attn_aggs_by_layer": attn_aggs_by_layer,
        "logit_aggs_by_layer": logit_aggs_by_layer,
    }




def retrieve_one_config_from_dict(
    data_dict: Dict[str, Any],
    layer: int,
    group: str,
    aggregation: str
) -> np.ndarray:
    """
    Extract descriptors (e.g., embeddings or other features
    derived from hidden representations) for a given configuration:
    layer, group and aggregation

    Parameters
    ----------  
    data_dict : Dict[str, Any]
        One split's dictionary containing the `descriptors` key with the structure: 
        {
            "descriptors": {
                "layer_{i}": {
                    "hidden": {
                        "<hidden_agg>": np.ndarray[(n_samples, feat_dim), float]
                                     or np.ndarray[(n_samples,), float]
                        ...
                    },
                    "attention": {
                        "<attn_agg>": np.ndarray[(n_samples,), float]
                        ...
                    },
                    "logit": {
                        "<logit_agg>": np.ndarray[(n_samples,), float]
                        ...
                    }
                },
            }
        }
    layer : int
        Layer index to read from (applies to `hidden`, `attention`, `logit`).
    group : str
        One of {"hidden", "attention", "logit"}.
    aggregation : str
        Name of the aggregation mode within the chosen group.
        One of  {"perplexity_score", "logit_entropy_score", "window_logit_entropy_score"} from "logit" group, 
                {"avg_emb", "last_emb", "max_emb", "first_gen_emb", "feat_var_emb", "hidden_score"} from "hidden" group 
                {"attn_score"} from "attention" group

    Returns
    -------
    arr : np.ndarray[float]
        - If aggregation mode ends with '*_emb': shape (n_samples, feat_dim), float
        - If aggregation mode ends with '*_score': shape (n_samples), float

    Raises
    ------
    KeyError
        If the requested path does not exist in the dictionary.
    """
    
    S = data_dict["descriptors"]
    layer_node = S.get(f"layer_{layer}", {})

    if group == "hidden":
        if "hidden" in layer_node and aggregation in layer_node["hidden"]:
            return np.asarray(layer_node["hidden"][aggregation])
        raise KeyError(f"Missing aggregation '{aggregation}' from group 'hidden' for layer {layer}")

    if group == "attention":
        if "attention" in layer_node and aggregation in layer_node["attention"]:
            return np.asarray(layer_node["attention"][aggregation])
        raise KeyError(f"Missing aggregation '{aggregation}' from group 'attention' for layer {layer}")
    
    if group == "logit":
        if "logit" in layer_node and aggregation in layer_node["logit"]:
            return np.asarray(layer_node["logit"][aggregation])
        raise KeyError(f"Missing aggregation '{aggregation}' from group 'logit' for layer {layer}")

    raise ValueError(f"Unknown group: {group}. Expected 'hidden' | 'attention' | 'logit'.")




def ensure_2d(X: np.ndarray) -> torch.Tensor:
    '''if X.shape==(n,) -> becomes (n,1);  if X.shape==(n,d) -> unchanged
    if X.shape==(n,1) -> unchanged
    '''
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return torch.from_numpy(X) 




def grid_search_best_config(
    id_fit_data: Dict[str, Any],
    id_test_data: Dict[str, Any],
    od_test_data: Dict[str, Any],
    method: Literal["dknn", "cosine", "mahalanobis", "ocsvm", "isoforest"]  = "dknn",
    method_params: Optional[Dict[str, Any]] = None,
    probing_test_size: float = 0.1,
    probing_random_state: int = 42,
    show_progress: bool = True,
    save_path: Optional[str] = None,  
    threshold_strategy: Literal["youden", "target_tpr"] = "youden",
    target_tpr: float = 0.95,
    sort_by: Optional[List[str]] = None,
    ascending: Optional[List[bool]] = None,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Enumerate and evaluate all available (layer, group, aggregation) configurations
    discovered from the `descriptors` keys in dicts id_fit_data/id_test_data/od_test_data,
    using the chosen OOD method.

    For each configuration, this function:
      1) Extracts descriptors for ID fit / ID test / OOD test.
      2) Computes OOD scores (ID vs OOD) using `method`.
      3) Computes ROC metrics (AUROC, FPR95, AUPRC) and selects a decision threshold
         using `threshold_strategy`:
           - 'youden'     : maximizes Youden's J = TPR - FPR
           - 'target_tpr' : picks the threshold whose TPR is closest to `target_tpr`
      4) Thresholds predictions (ID=0 / OOD=1) at the selected threshold and reports
         accuracy/F1/precision/recall.
      5) Runs **linear probing** (logistic regression) on the descriptors themselves
         and reports accuracy/F1/precision/recall for the probe.

    Parameters
    ----------
    id_fit_data : Dict[str, Any]
        ID fit split with the `descriptors` key structure.           
    id_test_data : Dict[str, Any]
        ID test split with the`descriptors` key structure.     
    od_test_data : Dict[str, Any]
        OOD test split with the `descriptors` key structure.  
    method : str, optional (default="dknn")
        OOD method. One of:
          "dknn", "cosine", "mahalanobis", "ocsvm", "isoforest".
        - Method is forced internally to "raw_scores" when descriptors are already
          scalar scores (1D). In that case, we take the raw scores as OOD scores.
    method_params : Optional[Dict[str, Any]], optional
        Extra keyword arguments for the chosen method (e.g., k=1 for DKNN).
        Optional metadata (e.g., "prompt" | "generation" | "PromptGeneration").
        Stored in the resulting DataFrame for traceability.
    probing_test_size : float, optional (default=0.1)
        Test split proportion used inside the logistic-regression probing.
    probing_random_state : int, optional (default=42)
        Random seed for the logistic-regression probing split.
    show_progress : bool, optional (default=True)
        If True, display tqdm progress bars.
    save_path : Optional[str], default None
        Path to the output file. If `None`, no file is saved. Supported extensions: 
        - ".csv"  -> save CSV via `DataFrame.to_csv`
        - ".xlsx" or ".xls" -> save Excel via `DataFrame.to_excel`
    threshold_strategy : {"youden","target_tpr"}, default "youden"
        Strategy to select the operating threshold from ROC.
    target_tpr : float, default 0.95
        Target TPR used when `threshold_strategy="target_tpr"`.
    sort_by : Optional[List[str]], default None
        Column(s) to sort the final DataFrame by. If None, defaults to
        ["auroc", "auprc"] in descending order when possible.
    ascending : Optional[List[bool]], default None
        Sort order per `sort_by`. Can be a single bool or a list with the same
        length as `sort_by`. If None and `sort_by` is provided, defaults to all False
        (i.e., descending for each column).
    debug : bool, default False
        If True, print small descriptive stats for each descriptor array.

    
    Structure of the input dicts
    ----
    id_fit_data, id_test_data and od_test_data have the following structure: 
        {
            "descriptors": {
                "layer_{i}": {
                    "hidden": {
                        "<hidden_agg>": np.ndarray[(n_samples, feat_dim), float]
                                     or np.ndarray[(n_samples,), float]
                        ...
                    },
                    "attention": {
                        "<attn_agg>": np.ndarray[(n_samples,), float]
                        ...
                    },
                    "logit": {
                        "<logit_agg>": np.ndarray[(n_samples,), float]
                        ...
                    }
                },
            }
        }

    Returns
    -------
    df_results : pd.DataFrame
        One row per (layer, group, aggregation) configuration with columns:
        - "layer": int
        - "group": str  in {"hidden", "attention", "logit"}
        - "aggregation": str  (e.g., "avg_emb", "attn_score", "logit_entropy_score", ...)
        - "method": str
        - "method_params": Dict[str, Any]

        OOD-score thresholding:
        - "auroc": float
        - "fpr95": float
        - "auprc": float
        - "threshold": float  (selected per strategy)
        - "acc_thresh": float
        - "f1_thresh": float
        - "prec_thresh": float
        - "recall_thresh": float

        Linear probing (logistic regression on descriptors):
        - "acc_probe": float
        - "f1_probe": float
        - "prec_probe": float
        - "recall_probe": float

    Notes
    -----
    - Descriptor shapes:
        hidden   -> (n_samples, feat_dim)  (2D)
        attention/logits -> (n_samples,)   (1D), reshaped to (n_samples, 1) for probing.
    - OOD score convention: higher score means "more OOD".
    - For the first_generated + prompt configuration, it's expected that all embeddings end up 
      (nearly) identical. Because the first generated token in the prompt is always the BOS/special 
      token (e.g., <s>). Since every sample shares the same first token, its representation is the 
      same across the dataset. As a result, distance-based  OOD scores (DKNN, cosine, etc.) collapse 
      to zero for everyone, and the method has no discriminative power in this setting.
    """
    method_params = method_params or {}

    # Retrieve all space configurations
    space = discover_config_space(id_fit_data)

    # Convenience for optional progress bars
    LayerIter = tqdm(space["layers"], desc="Layers") if show_progress else space["layers"]

    rows: List[Dict[str, Any]] = []

    # Helper: evaluate one configuration and append a row
    def _evaluate_and_append_row(
        layer: int,
        group: str,
        agg: str,
        method: str,
    ) -> None:
                
        # Descriptor extraction for a given configuration
        fit = retrieve_one_config_from_dict(id_fit_data, layer, group, agg)
        idt = retrieve_one_config_from_dict(id_test_data, layer, group, agg)
        odt = retrieve_one_config_from_dict(od_test_data, layer, group, agg)

        # Check if descriptors are 1D scores. In that case, use 
        # them directly as OOD scores and do not apply OOD methods
        if fit.ndim==1:
            method='raw_scores'

        # OOD scores (ID vs OOD)
        scores_id, scores_ood = compute_ood_scores(
            method=method,
            id_fit_descriptors=fit,  # shape: (n_fit,d) or (n_fit,)
            id_test_descriptors=idt, # shape: (n_idt,d) or (n_idt,)
            od_test_descriptors=odt, # shape: (n_odt,d) or (n_odt,)
            **method_params
        )

        if debug:
            print("="*30)
            for name, X in [("fit", fit), ("id", idt), ("ood", odt)]:
                _describe_array(name, X)


        # ROC metrics + threshold
        auroc, fpr95, auc_pr, fpr, tpr, thresholds, selected_threshold = compute_metrics(
            scores_id=scores_id, scores_ood=scores_ood, plot=False, save_path=None,
            threshold_strategy=threshold_strategy, target_tpr=target_tpr,
        )

        # Generate labels for each test point with Thresholding 
        # Predicted class is assigned as follows:
        # - score >  threshold -> predicted OOD (label 1)
        # - score <= threshold -> predicted ID (label 0)
        y_true = np.concatenate(   # Ground-truth labels: 0=ID, 1=OOD
            [np.zeros(len(scores_id), dtype=int), np.ones(len(scores_ood), dtype=int)]
        )
        y_pred = np.concatenate(   # Predicted labels based on threshold: 0=ID, 1=OOD
            [(scores_id > selected_threshold).astype(int), (scores_ood > selected_threshold).astype(int)]
        ) 

        _, acc, f1, prec, rec = compute_confusion_matrix_and_metrics(
            y_true=y_true, y_pred=y_pred, plot=False, normalize=True
        )

        if debug:
            print("unique y_true:", np.unique(y_true, return_counts=True))
            print("unique y_pred:", np.unique(y_pred, return_counts=True))
            print("used threshold:", selected_threshold)
    
        # Linear probing on descriptors; reshape 1D to 2D for probing (ensure_2d)
        clf, y_test_probe, y_pred_probe, idx_test = train_logistic_regression_on_descriptors(
            id_test_descriptors=ensure_2d(idt), 
            od_test_descriptors=ensure_2d(odt),
            random_state=probing_random_state,
            test_size=probing_test_size
        )
        _, acc_p, f1_p, prec_p, rec_p = compute_confusion_matrix_and_metrics(
            y_true=y_test_probe, y_pred=y_pred_probe, plot=False, normalize=True
        )


        rows.append({
            "layer": layer,
            "group": group,
            "aggregation": agg,
            "method": method,
            "method_params": dict(method_params),

            "auroc": float(auroc),
            "fpr95": float(fpr95),
            "auprc": float(auc_pr),
            "threshold": float(selected_threshold),

            "acc_thresh": float(acc),
            "f1_thresh": float(f1),
            "prec_thresh": float(prec),
            "recall_thresh": float(rec),

            "acc_probe": float(acc_p),
            "f1_probe": float(f1_p),
            "prec_probe": float(prec_p),
            "recall_probe": float(rec_p),
        })

    # -------------------------
    # Hidden (per layer, 2D)
    # -------------------------
    for layer in LayerIter:
        
        hidden_aggs = space["hidden_aggs_by_layer"].get(layer, [])
        if show_progress:
            hidden_aggs = tqdm(hidden_aggs, desc=f"Hidden aggs @ layer {layer}", leave=False)
        for agg in hidden_aggs:
            #print(f"\naggregation {agg}")
            _evaluate_and_append_row(layer, "hidden", agg, method)

        # -------------------------
        # Attention (per layer, 1D)
        # -------------------------
        attn_aggs = space["attn_aggs_by_layer"].get(layer, [])
        if show_progress:
            attn_aggs = tqdm(attn_aggs, desc=f"Attention aggs @ layer {layer}", leave=False)
        for agg in attn_aggs:
            #print(f"\naggregation {agg}")
            _evaluate_and_append_row(layer, "attention", agg, method)

        # -------------------------
        # Logits (per layer, 1D)
        # -------------------------
        logits_aggs = space["logit_aggs_by_layer"].get(layer, [])
        if show_progress:
            logits_aggs = tqdm(logits_aggs, desc=f"Logit aggs @ layer {layer}", leave=False)
        for agg in logits_aggs:
            #print(f"\naggregation {agg}")
            _evaluate_and_append_row(layer, "logit", agg, method)


    # -------------------------
    # Build & sort DataFrame
    # -------------------------
    df = pd.DataFrame(rows)
    if not df.empty:  # Sort if there is at least one row.
        if sort_by is None:
            sort_cols = ["auroc", "auprc"] # Default sort keys
            sort_ord = [False, False]  # Sort in descending order
        else:
            # Normalize `sort_by` into a list (accept a single string or a list/tuple of strings).
            sort_cols = list(sort_by) if isinstance(sort_by, (list, tuple)) else [sort_by]  
            if ascending is None:
                # If no `ascending` is provided, default to descending for each sort column.
                sort_ord = [False] * len(sort_cols)
            elif isinstance(ascending, bool):
                # If a single bool is provided, broadcast it to all sort columns.
                sort_ord = [ascending] * len(sort_cols)
            else:
                # If a list/tuple is provided, it must match the length of `sort_cols`.
                if len(ascending) != len(sort_cols):
                    raise ValueError("`ascending` must be a bool or a list with the same length as `sort_by`.")
                sort_ord = list(ascending) # Make a shallow copy as a list.

            # Keep only sort columns that actually exist in the DataFrame (avoids KeyError).
            present = [c for c in sort_cols if c in df.columns]
            if len(present) < len(sort_cols):
                # If we dropped some columns, drop their corresponding sort orders to keep alignment.
                sort_ord = [ord for c, ord in zip(sort_cols, sort_ord) if c in present]
                sort_cols = present

        if sort_cols: # If there is at least one valid column to sort on, sort the DataFrame 
            df = df.sort_values(sort_cols, ascending=sort_ord).reset_index(drop=True)

    # Build DataFrame and sort by AUROC (desc) then AUPRC (desc)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["auroc", "auprc"], ascending=[False, False]).reset_index(drop=True)

    # -------------------------
    # Save results if requested
    # -------------------------
    if save_path is not None:
        path = Path(save_path)
        if path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)

        ext = path.suffix.lower()
        if ext == ".csv":
            df.to_csv(path, index=False)
        elif ext in (".xlsx", ".xls"):
            df.to_excel(path, index=False)
        else:
            raise ValueError(
                f"Unsupported extension '{ext}'. Use '.csv', '.xlsx', or '.xls'."
            )


    return df




def average_ood_grid_results(
    paths: List[str],
    round_digits: int = 4,
    sort_by: Optional[Union[str, List[str]]] = None,
    ascending: Optional[Union[bool, List[bool]]] = None,
    save_path: Optional[str] = None,
    readable_writing: bool = False, 
) -> pd.DataFrame:
    """
    Read multiple grid_search_best_config result files (CSV/XLSX) and compute
    the mean/std per unique configuration:
    (layer, group, aggregation, method, method_params)

    Parameters
    ----------
    paths : List[str]
        List of CSV/XLSX files produced by `grid_search_best_config`.
    round_digits : int, default 4
        Round aggregated numeric results to this number of decimals.
    sort_by : str or List[str], optional
        Column(s) to sort by in the final table. If None, defaults to
        ["auroc_mean", "auprc_mean"] when present.
    ascending : bool or List[bool], optional
        Sort order per `sort_by`. If None and `sort_by` is given, defaults to
        all False (descending). If a single bool is given, it is broadcast.
    save_path : Optional[str], default None
        If provided, save the aggregated table. Extension decides the format:
        - ".csv"              -> DataFrame.to_csv
        - ".xlsx" or ".xls"   -> DataFrame.to_excel (requires openpyxl)
    readable_writing : bool, default False
        If True, convert metric columns to human-readable strings:
        percentage with two decimals: "XX.XX ± YY.YY". 
        With XX.XX = mean, YY.YY = std. Drops *_std columns.

    Returns
    -------
    pd.DataFrame
        Aggregated results with one row per unique config:
        (layer, group, aggregation, method, method_params), plus averaged metrics
        and a `n_runs` column indicating how many files contributed.
    """
    if not paths:
        raise ValueError("`paths` must contain at least one file path.")

    # Define the columns that uniquely identify a configuration group in the data
    group_cols = ["layer", "group", "aggregation", "method", "method_params"]

    # ------------------------------
    # Load all input files
    # ------------------------------
    # Initialize an empty list to store DataFrames read from each file
    frames = []
    # Loop through each file path provided in 'paths'
    for p in paths:
        ext = Path(p).suffix.lower() # Extract the file extension to determine file type
        if ext == ".csv":
            df = pd.read_csv(p)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(p)
        else:
            raise ValueError(f"Unsupported extension for '{p}'. Use .csv, .xlsx or .xls.")

        # Append the DataFrame loaded from this file into the list
        frames.append(df)

    # ------------------------------
    # Concatenate & schema checks
    # ------------------------------
    # Concatenate all loaded DataFrames into a single DataFrame, reset the index
    df_all = pd.concat(frames, ignore_index=True)

    # Check if all grouping columns exist in the combined DataFrame
    missing = [c for c in group_cols if c not in df_all.columns]
    if missing:
        raise KeyError(f"Missing required columns in input files: {missing}")

    # Make params groupable (string form assumed consistent across runs)
    df_all["method_params"] = df_all["method_params"].astype(str)

    # ------------------------------
    # Identify metric columns
    # ------------------------------
    # Metrics = all numeric columns except the group keys (if any were numeric)
    numeric_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
    metric_cols = [c for c in numeric_cols if c not in group_cols]
    if not metric_cols:
        raise ValueError("No numeric metric columns found to aggregate.")
    
    # ------------------------------
    # Aggregate per configuration
    # ------------------------------
    # Group the DataFrame by the grouping columns; keep NaN as a group
    grouped = df_all.groupby(group_cols, dropna=False)
    # Aggregate (mean and std) numeric metrics per group, reset index for result
    agg_df = grouped[metric_cols].agg(['mean', 'std']).reset_index()

    # Flatten the MultiIndex columns from aggregation into simple string column names
    agg_df.columns = [
        (f"{col}_{stat}" if stat else col)
        for col, stat in (
            [(c, '') for c in group_cols] +  # group columns without suffix
            [(m, s) for m in metric_cols for s in ('mean', 'std')]  # metric + aggregation suffix
        )
    ]

    # Count how many rows (runs) contributed to each group, rename to 'n_runs'
    df_count = grouped.size().rename("n_runs").reset_index()
    # Merge aggregated metrics DataFrame and count DataFrame on the group keys
    out = pd.merge(agg_df, df_count, on=group_cols, how="left")

    # ------------------------------
    # Post-processing of metrics
    # ------------------------------
    # std can be NaN when n_runs == 1 -> set to 0.0 for convenience
    for m in metric_cols:
        std_col = f"{m}_std"
        if std_col in out.columns:
            out[std_col] = out[std_col].fillna(0.0)

    # Rounding (only metric mean/std; not n_runs)
    if round_digits is not None:
        to_round = [f"{m}_mean" for m in metric_cols] + [f"{m}_std" for m in metric_cols]
        to_round = [c for c in to_round if c in out.columns]  # safely select existing columns
        out[to_round] = out[to_round].round(round_digits)

    # ------------------------------
    # Sorting of the final table
    # ------------------------------
    if sort_by is None:
        sort_cols = [c for c in ("auroc_mean", "auprc_mean") if c in out.columns]
        sort_ord = [False] * len(sort_cols)
    else:
        sort_cols = list(sort_by) if isinstance(sort_by, (list, tuple)) else [sort_by]
        if ascending is None:
            sort_ord = [False] * len(sort_cols)
        elif isinstance(ascending, bool):
            sort_ord = [ascending] * len(sort_cols)
        else:
            if len(ascending) != len(sort_cols):
                raise ValueError("`ascending` must be a bool or a list with the same length as `sort_by`.")
            sort_ord = list(ascending)
        # Keep only existing columns
        present = [c for c in sort_cols if c in out.columns]
        if len(present) < len(sort_cols):
            sort_ord = [ord for c, ord in zip(sort_cols, sort_ord) if c in present]
            sort_cols = present

   
    # Sort output descending by these metric columns and reset index afterward
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)


    # ------------------------------
    # Human-readable formatting
    # ------------------------------
    if readable_writing:
        # Format: percentage with two decimals, combine mean & std into one column named after the metric.
        # Example: "85.37 ± 1.24"
        formatted_cols = []
        for m in metric_cols:
            mean_col = f"{m}_mean"
            std_col = f"{m}_std"
            if mean_col in out.columns:
                mean_pct = (out[mean_col] * 100).astype(float)
                if std_col in out.columns:
                    std_pct = (out[std_col] * 100).astype(float)
                    out[m] = mean_pct.map(lambda x: f"{x:.2f}") + " ± " + std_pct.map(lambda x: f"{x:.2f}")
                    # Drop std/mean columns after merging
                    out = out.drop(columns=[mean_col, std_col])
                else:
                    out[m] = mean_pct.map(lambda x: f"{x:.2f}")
                    out = out.drop(columns=[mean_col])
                formatted_cols.append(m)

        # Reorder columns: group keys, formatted metrics, then the rest (e.g., n_runs)
        prefix = group_cols
        suffix = [c for c in out.columns if c not in (group_cols + formatted_cols)]
        out = out[prefix + formatted_cols + suffix]


    # ------------------------------
    # Save aggregated results if requests
    # ------------------------------
    # If a save_path is provided, save the aggregated DataFrame to the specified file
    if save_path is not None:
        path = Path(save_path)
        # Create parent directories if they don't exist already
        if path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        # Determine file extension of save_path for deciding output format
        ext = path.suffix.lower()
        if ext == ".csv":
            out.to_csv(path, index=False)
        elif ext in (".xlsx", ".xls"):
            try:
                out.to_excel(path, index=False)
            except Exception as e:
                # Inform user if openpyxl is missing or Excel saving failed
                raise RuntimeError(
                    "Writing Excel requires `openpyxl`. Install it or choose a .csv save_path."
                ) from e
        else:
            raise ValueError(f"Unsupported extension '{ext}'. Use '.csv', '.xlsx', or '.xls'.")

    return out
