#!/usr/bin/env python3


from typing import Sequence, Dict, Any, Optional, Literal, Union, List
import numpy as np
from src.analysis.analyze_ood_results import retrieve_one_config_from_dict
import matplotlib.pyplot as plt


AGG_TO_GROUP = {
    "logit_entropy_score":        "logit",
    "window_logit_entropy_score": "logit",
    "perplexity_score":           "logit",
    "hidden_score":               "hidden",
    "attn_score":                 "attention",
}


def collect_layerwise_scalar_series(
    layers: Sequence[int],
    id_fit_data: Dict[str, Any],
    id_test_data: Dict[str, Any],
    od_test_data: Dict[str, Any],
    aggregations: Optional[Sequence[str]] = None,
    output: Literal["array", "list"] = "list",
) -> Dict[str, Dict[str, Any]]:
    """
    For each scalar aggregation (e.g., logit_entropy_score, hidden_score, ...),
    gather the value *per layer* and stack them so each sample becomes a vector
    of length L (number of used layers).

    Parameters
    ----------
    layers : Sequence[int]
        Ordered list of layers to try (e.g., [1,3,5,...,-1]).
    id_fit_data, id_test_data, od_test_data : dict
        Split dicts containing the 'descriptors' tree.
    aggregations : list[str] or None
        Which aggregations to collect. If None, uses:
        ["logit_entropy_score","window_logit_entropy_score","perplexity_score",
         "hidden_score","attn_score"].
    output : {"array","list"}, default "list"
        Output format:
        - if "array": each split is a numpy array of shape (n_samples, L_used).
        - ig "list" : each split is a Python list of length n_samples, where each
          element is a list of length L_used (one value per used layer).

    Returns
    -------
    result : dict
        {
          "<agg_name>": {
             "group":  "<group_name>",
             "layers": [list of used layers in order],
             "id_fit":  (n_fit, L_used) array or list-of-lists,
             "id_test": (n_id,  L_used) array or list-of-lists,
             "od_test": (n_ood, L_used) array or list-of-lists,
          },
          ...
        }

    Notes
    -----
    - This function requires the chosen aggregation to be 1D (scalar per sample)
      at each layer. If a retrieved array is not 1D, a ValueError is raised.
    - Layers are kept in the order given by the `layers` argument.
    - With `missing="skip"`, only layers present in **all three splits** are kept.
    """
    if aggregations is None:
        aggregations = list(AGG_TO_GROUP.keys())

    # Validate aggs and map to groups
    agg_group_pairs = []
    for agg in aggregations:
        if agg not in AGG_TO_GROUP:
            raise ValueError(f"Unknown aggregation '{agg}'. Allowed keys: {list(AGG_TO_GROUP.keys())}")
        agg_group_pairs.append((agg, AGG_TO_GROUP[agg]))

    # Utility function: Stack 1D arrays for several layers into a 2D matrix for a split
    def _stack_for_split(data_dict: Dict[str, Any], used_layers, group: str, agg: str) -> np.ndarray:
        cols = []
        n_samples = None # Track number of samples for consistency check
        for l in used_layers:
            # Get the 1D array for this (layer, group, agg)
            v = retrieve_one_config_from_dict(data_dict, layer=l, group=group, aggregation=agg)
            v = np.asarray(v) # shape: (n_samples),
            if v.ndim != 1:
                raise ValueError(f"{group}/{agg} at layer {l} is not 1D (got shape {v.shape}).")
            # Set/check the number of samples across layers
            if n_samples is None:
                n_samples = v.shape[0]
            elif v.shape[0] != n_samples:
                raise ValueError(
                    f"Inconsistent number of samples across layers for {group}/{agg}: "
                    f"expected {n_samples}, got {v.shape[0]} at layer {l}."
                )
            cols.append(v.astype(float))
        return np.column_stack(cols)  # (n_samples, L_used)

    out: Dict[str, Dict[str, Any]] = {}

    for agg, group in agg_group_pairs:
        # Build (n_samples, L_used) matrices for each split
        fit_mat = _stack_for_split(id_fit_data,  layers, group, agg)
        id_mat  = _stack_for_split(id_test_data, layers, group, agg)
        od_mat  = _stack_for_split(od_test_data, layers, group, agg)

        if output == "list":
            fit_mat = fit_mat.tolist()
            id_mat  = id_mat.tolist()
            od_mat  = od_mat.tolist()

        out[agg] = {
            "group":  group,
            "layers": layers,
            "id_fit": fit_mat,
            "id_test": id_mat,
            "od_test": od_mat,
        }

    return out




def plot_layerwise_series(
    series: dict,
    agg: str = "perplexity_score",
    idx=range(60, 80),
    fit_max: int = 500,
) -> None:
    """
    Plot per-layer scalar sequences for a given aggregation, with:
      - Top row:   ID test (left) and OOD test (right), overlaying the curves for `idx`
      - Bottom row: ID fit (sampling up to `fit_max` curves to avoid clutter)
    The function expects each  sequence to be 1-dimensional.
      
    Expected `series` structure
    ---------------------------
    series = {
      "<agg>": {
        "id_test": List[np.ndarray or list],  # length = n_id_test, each item is a sequence over layers
        "od_test": List[np.ndarray or list],  # length = n_ood_test, each item is a sequence over layers
        "id_fit" : List[np.ndarray or list],  # length = n_id_fit , each item is a sequence over layers
      },
      ...
    }

    Parameters
    ----------
    series : dict
        Container holding the per-layer sequences for each aggregation.
    agg : str, default "perplexity_score"
        Which aggregation key from `series` to visualize.
    idx : iterable of int, default range(60, 80)
        Indices of samples to overlay in ID test and OOD test panels.
    fit_max : int, default 500
        Max number of ID-fit curves to plot in the bottom panel.

    Notes
    -----
    - Each curve corresponds to one sample; the x-axis is the layer index (in order of your sequence).
    - All three panels share the same y-limits for fair comparison.
    """
    if agg not in series:
        raise KeyError(f"Aggregation '{agg}' not found in `series`.")

    block = series[agg]
    for key in ("id_test", "od_test"):
        if key not in block:
            raise KeyError(f"`series['{agg}']` must contain key '{key}'.")

    id_test_list = block.get("id_test", [])
    od_test_list = block.get("od_test", [])
    id_fit_list  = block.get("id_fit", [])

    # Normalize and bound indices to available ranges
    idx = list(idx)
    idx_id  = [i for i in idx if 0 <= i < len(id_test_list)]
    idx_ood = [i for i in idx if 0 <= i < len(od_test_list)]
    fit_idx = list(range(min(len(id_fit_list), int(fit_max))))

    # Helper to collect finite values for y-limits
    def collect_vals(lst, indices):
        vals = []
        for i in indices:
            arr = np.asarray(lst[i]).ravel()
            arr = arr[np.isfinite(arr)]
            if arr.size:
                vals.append(arr)
        return vals

    # Fix min and max values for all panels by taking the min and max of all collected values 
    # The 3 panels of the graph will have the same y-limits, for fair comparison
    top_vals = collect_vals(id_test_list, idx_id) + collect_vals(od_test_list, idx_ood)
    fit_vals = collect_vals(id_fit_list, fit_idx)
    if top_vals or fit_vals:
        ymin = min([v.min() for v in (top_vals + fit_vals)]) if (top_vals or fit_vals) else None
        ymax = max([v.max() for v in (top_vals + fit_vals)]) if (top_vals or fit_vals) else None
    else:
        ymin = ymax = None

    # Figure layout: 2 columns (ID / OOD) on top, 1 wide panel (ID fit) below
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    ax_id  = fig.add_subplot(gs[0, 0])
    ax_ood = fig.add_subplot(gs[0, 1], sharey=ax_id)
    ax_fit = fig.add_subplot(gs[1, :],  sharey=ax_id)

    # --- Top-left: ID test ---
    for i in idx_id:
        ax_id.plot(id_test_list[i], alpha=0.6)
    ax_id.set_title(f"{agg} — ID test (n={len(idx_id)})")
    ax_id.set_xlabel("Layer index")
    ax_id.set_ylabel("Score")
    ax_id.grid(True, alpha=0.3)

    # --- Top-right: OOD test ---
    for i in idx_ood:
        ax_ood.plot(od_test_list[i], alpha=0.6)
    ax_ood.set_title(f"{agg} — OOD test (n={len(idx_ood)})")
    ax_ood.set_xlabel("Layer index")
    ax_ood.grid(True, alpha=0.3)

    # --- Bottom: ID fit ---
    for i in fit_idx:
        ax_fit.plot(id_fit_list[i], alpha=0.2)
    ax_fit.set_title(f"{agg} — ID fit (n={len(fit_idx)})")
    ax_fit.set_xlabel("Layer index")
    ax_fit.set_ylabel("Score")
    ax_fit.grid(True, alpha=0.3)

    # Unify y-limits if we could compute them
    if ymin is not None and ymax is not None:
        ax_id.set_ylim(ymin, ymax)
        ax_ood.set_ylim(ymin, ymax)
        ax_fit.set_ylim(ymin, ymax)

    fig.suptitle(f"Per-layer {agg} sequences", y=0.98)
    fig.tight_layout()
    plt.show()



# Supporte la version Array pour les embeddings ? 
def compute_dimension_joint_volatility(
    series: Dict[str, Dict[str, List[Union[np.ndarray, list]]]],
    agg: str,
    split: str = "id_test",
) -> np.ndarray:
    """
    Compute dimension joint volatility V_j(s) for per-layer sequences s 
    extracted from a nested series structure.
    
    The dimension joint volatility V_j(s) for one trajectory 
    s = (y_0, ..., y_{T-1}) with per-layer vectors/scores y_l is:
    
        V_j(s) = (1 / L) * sum_{l=1}^L || y_l - y_{l-1} ||_2
    
    where L is the number of valid consecutive segments after ignoring 
    invalid diffs (e.g., due to NaN or infinite values).
    
    NOTE: Inspired from the paper:
    "Embedding Trajectory for Out-of-Distribution Detection in 
    Mathematical Reasoning (2024)"

    Parameters
    ----------
    series : dict
        Nested dictionary with structure series[agg][split], where:
            - agg (str) keys select aggregation keys
            - split (str) keys select data splits ('id_test', 'od_test', etc.)
        Each series[agg][split] is expected to be a list of length n_samples,
        where each element is either a 1D sequence (T,) of scalars or a 2D array 
        (T, D) of per-layer features.
    agg : str
        Aggregation key to select within the series.
    split : str, default "id_test"
        Data split to select within the aggregation.
    
    Returns
    -------
    np.ndarray, shape (n_samples,)
        Array of path variation values, one per sample in series[agg][split].
        
    Notes
    ------
    - For 1D, the L2 norm reduces to absolute value.
    - Returns 0.0 if no valid segment exists or T < 2.
    """

    def path_variation(seq: Union[np.ndarray, list]) -> float:
        """
        Compute V_j(s) = (1 / L) * sum_{l=1}^L || y_l - y_{l-1} ||_2 for one trajectory.
        - seq can be 1D (T,) for scalar per-layer scores or 2D (T, D) for vector per-layer features.
        - L is the number of *valid* consecutive segments (after dropping non-finite diffs).
        """
        x = np.asarray(seq, dtype=np.float64)
        if x.ndim == 1:
            diffs = np.diff(x)  # (T-1,)
            mask = np.isfinite(diffs)
            L = np.sum(mask)
            if L == 0:
                return 0.0
            return np.sum(np.abs(diffs[mask])) / L
        elif x.ndim == 2:
            diffs = np.diff(x, axis=0)  # (T-1, D)
            mask = np.all(np.isfinite(diffs), axis=1)
            if not np.any(mask):
                return 0.0
            norms = np.linalg.norm(diffs[mask], axis=1) # (L_valid,)
            return np.sum(norms) / norms.size
        else:
            raise ValueError("Each sequence must be 1D (T,) or 2D (T, D) array.")

    if agg not in series:
        raise KeyError(f"Aggregation '{agg}' not found in series.")
    if split not in series[agg]:
        raise KeyError(f"Split '{split}' not found in series['{agg}'].")

    data = series[agg][split]
    return np.array([path_variation(s) for s in data], dtype=np.float64)




