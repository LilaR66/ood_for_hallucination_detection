#!/usr/bin/env python3
"""
============================================================
Layerwise Series & Trajectory Analysis Utilities
============================================================

This module provides tools to:
- Collect per-layer **series** for a single (group, aggregation) across splits
  (ID-fit / ID-test / OOD-test), returning compact NumPy arrays:
    - 1D scalars per layer  -> shape (n, L)
    - 2D embeddings per layer -> shape (n, L, D)
- Visualize **scalar** (1D) layerwise sequences side-by-side for ID/OOD/ID-fit.
- Compute the **dimension joint volatility** (trajectory total variation) for
  1D and 2D per-layer trajectories.
- Estimate per-layer ID **mean/variance** and compute **diagonal Mahalanobis**
  distances per layer for new samples.
"""

from typing import Sequence, Dict, Any, Literal, Iterable, Tuple
import numpy as np
from src.analysis.analyze_ood_results import retrieve_one_config_from_dict
import matplotlib.pyplot as plt


def collect_layerwise_series(
    layers: Sequence[int],
    id_fit_data: Dict[str, Any],
    id_test_data: Dict[str, Any],
    od_test_data: Dict[str, Any],
    *,
    group: Literal["hidden", "attention", "logit"],
    aggregation: str,
) -> Dict[str, Any]:
    """
    Collect a single (group, aggregation) across layers and stack into trajectories.

    Behavior
    --------
    Let L = len(layers), n = number of samples per split, and D = feature dimension (2D case).
      - If each per-layer array is 1D (n,), you get output of shape (n, L)
      - If each per-layer array is 2D (n, D), you get output of shape (n, L, D)

    Parameters
    ----------
    layers : Sequence[int]
        Ordered list of layers to retrieve (e.g., [1,3,5,...,-1]).
    id_fit_data, id_test_data, od_test_data : dict
        Split dicts containing the 'descriptors' key tree.
    group : {"hidden", "attention", "logit"}
        Group to read from.
    aggregation : str
        Aggregation name under the chosen group (e.g., "hidden_score", "avg_emb",
        "logit_entropy_score", ...).
   

    Returns
    -------
    dict
        {
          "group":  group,
          "aggregation": aggregation,
          "layers": list(layers),
          "id_fit":  (n, L) or (n, L, D) array,
          "id_test": (n, L) or (n, L, D) array,
          "od_test": (n, L) or (n, L, D) array,
        }

    Raises
    ------
    KeyError
        If the requested (layer, group, aggregation) path does not exist.
    ValueError
        If per-layer arrays are neither 1D nor 2D, or shapes are inconsistent
        across layers/splits.
    """
    # Helper to build (n, L) or (n, L, D) for one split without nested functions.
    def build_for_split(split_dict: Dict[str, Any]) -> np.ndarray:
        mats = []
        n = None
        D = None
        mode = None  # "1d" or "2d"

        for l in layers:
            v = retrieve_one_config_from_dict(split_dict, layer=l, group=group, aggregation=aggregation)
            v = np.asarray(v) # shape: (n,) or (n,D)

            if v.ndim == 1:
                if mode is None:
                    mode = "1d"
                elif mode != "1d":
                    raise ValueError(f"Mixed 1D/2D encountered at layer {l} for {group}/{aggregation}.")
                if n is None:
                    n = v.shape[0]
                elif v.shape[0] != n:
                    raise ValueError(
                        f"Inconsistent sample counts across layers for {group}/{aggregation}: "
                        f"expected {n}, got {v.shape[0]} at layer {l}."
                    )
                mats.append(v.astype(float))  # (n,)

            elif v.ndim == 2:
                if mode is None:
                    mode = "2d"
                elif mode != "2d":
                    raise ValueError(f"Mixed 1D/2D encountered at layer {l} for {group}/{aggregation}.")
                if n is None:
                    n = v.shape[0]
                    D = v.shape[1]
                else:
                    if v.shape[0] != n or v.shape[1] != D:
                        raise ValueError(
                            f"Inconsistent shapes across layers for {group}/{aggregation}: "
                            f"expected (n={n}, D={D}), got {tuple(v.shape)} at layer {l}."
                        )
                mats.append(v.astype(float))  # (n, D)

            else:
                raise ValueError(
                    f"{group}/{aggregation} at layer {l} has unsupported ndim={v.ndim}; "
                    f"expected 1 or 2."
                )

        # Stack across layers into the desired (n, L) or (n, L, D).
        if mode == "1d":
            # list of (n,) -> (n, L)
            return np.column_stack(mats)
        else:
            # list of (n, D) -> stack to (L, n, D) then transpose -> (n, L, D)
            LND = np.stack(mats, axis=0)
            return np.transpose(LND, (1, 0, 2))

    # Build arrays for each split and ensure they share the same mode/shape form.
    id_fit_out   = build_for_split(id_fit_data)
    id_test_out  = build_for_split(id_test_data)
    od_test_out  = build_for_split(od_test_data)

    return {
        "group": group,
        "aggregation": aggregation,
        "layers": list(layers),
        "id_fit":  id_fit_out,
        "id_test": id_test_out,
        "od_test": od_test_out,
    }




def plot_layerwise_scalar_series(
    series: Dict[str, Any],
    idx: Iterable[int] = range(60, 80),
    fit_max: int = 500,
) -> None:
    """
    Plot per-layer **scalar** (1D) sequences for a single (group, aggregation).

    Layout:
      - Top-left:   ID test (overlay curves for indices in `idx`)
      - Top-right:  OOD test (overlay curves for indices in `idx`)
      - Bottom:     ID fit   (overlay up to `fit_max` curves)

    Expected `series` structure
    -----------------------------------------------------------
    series = {
        "group": "<hidden|attention|logit>",    # optional, for title
        "aggregation": "<name>",                # optional, for title
        "layers": [l1, l2, ...],                # optional
        "id_test": np.ndarray,  # shape (n_id_test,  L)
        "od_test": np.ndarray,  # shape (n_ood_test, L)
        "id_fit":  np.ndarray,  # shape (n_id_fit,   L)
    }

    Notes
    -----
    - This function **only supports scalar sequences** stacked as 2D arrays:
      each row is one sample; columns are layers (length L).
    - All three panels share the same y-limits for fair visual comparison.
    """
    # -----------------------------
    # Basic presence checks
    # -----------------------------
    for key in ("id_test", "od_test", "id_fit"):
        if key not in series:
            raise KeyError(f"`series` must contain key '{key}'.")

    id_test = series["id_test"]
    od_test = series["od_test"]
    id_fit  = series["id_fit"]

    # -----------------------------
    # Validate: strictly 2D arrays of scalars
    # -----------------------------
    def _check_2d(name: str, arr: Any) -> np.ndarray:
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"`{name}` must be a NumPy array; got {type(arr)}.")
        if arr.ndim != 2:
            raise ValueError(f"`{name}` must be 2D (n_samples, L); got shape {arr.shape}.")
        if arr.shape[1] < 1:
            raise ValueError(f"`{name}` must have at least one layer (shape[1] >= 1); got {arr.shape}.")
        return arr

    id_test = _check_2d("id_test", id_test)
    od_test = _check_2d("od_test", od_test)
    id_fit  = _check_2d("id_fit",  id_fit)

    # -----------------------------
    # Normalize and bound indices
    # -----------------------------
    idx = list(idx)
    idx_id  = [i for i in idx if 0 <= i < id_test.shape[0]]
    idx_ood = [i for i in idx if 0 <= i < od_test.shape[0]]
    fit_n   = min(id_fit.shape[0], int(fit_max))

    # -----------------------------
    # Compute shared y-limits (finite only)
    # -----------------------------
    def _finite_minmax(rows: np.ndarray) -> tuple[float, float] | None:
        # rows: (k, L) of selected curves
        if rows.size == 0:
            return None
        vals = rows[np.isfinite(rows)]
        if vals.size == 0:
            return None
        return float(vals.min()), float(vals.max())

    # Fix min and max values for all panels by taking the min and max of all collected values 
    # The 3 panels of the graph will have the same y-limits, for fair comparison
    mins_maxs = []
    if idx_id:
        mins_maxs.append(_finite_minmax(id_test[idx_id, :]))
    if idx_ood:
        mins_maxs.append(_finite_minmax(od_test[idx_ood, :]))
    if fit_n > 0:
        mins_maxs.append(_finite_minmax(id_fit[:fit_n, :]))

    mins_maxs = [mm for mm in mins_maxs if mm is not None]
    if mins_maxs:
        ymin = min(m for m, _ in mins_maxs)
        ymax = max(M for _, M in mins_maxs)
    else:
        ymin = ymax = None

    # Titles
    group = series.get("group", None)
    agg   = series.get("aggregation", None)
    title_prefix = (f"{group}: {agg}" if group and agg else (agg or group or "Selected series"))

    # -----------------------------
    # Figure and axes
    # -----------------------------
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

    ax_id  = fig.add_subplot(gs[0, 0])
    ax_ood = fig.add_subplot(gs[0, 1], sharey=ax_id)
    ax_fit = fig.add_subplot(gs[1, :],  sharey=ax_id)

    # -----------------------------
    # Top-left: ID test
    # -----------------------------
    for i in idx_id:
        ax_id.plot(id_test[i, :], alpha=0.6)
    ax_id.set_title(f"{title_prefix} — ID test (n={len(idx_id)})")
    ax_id.set_xlabel("Layer index")
    ax_id.set_ylabel("Score")
    ax_id.grid(True, alpha=0.3)

    # -----------------------------
    # Top-right: OOD test
    # -----------------------------
    for i in idx_ood:
        ax_ood.plot(od_test[i, :], alpha=0.6)
    ax_ood.set_title(f"{title_prefix} — OOD test (n={len(idx_ood)})")
    ax_ood.set_xlabel("Layer index")
    ax_ood.grid(True, alpha=0.3)

    # -----------------------------
    # Bottom: ID fit
    # -----------------------------
    for i in range(fit_n):
        ax_fit.plot(id_fit[i, :], alpha=0.2)
    ax_fit.set_title(f"{title_prefix} — ID fit (n={fit_n})")
    ax_fit.set_xlabel("Layer index")
    ax_fit.set_ylabel("Score")
    ax_fit.grid(True, alpha=0.3)

    # unify y-limits if available
    if ymin is not None and ymax is not None:
        ax_id.set_ylim(ymin, ymax)
        ax_ood.set_ylim(ymin, ymax)
        ax_fit.set_ylim(ymin, ymax)

    fig.suptitle("Per-layer scalar sequences", y=0.98)
    fig.tight_layout()
    plt.show()




def compute_dimension_joint_volatility_1D(data: np.ndarray) -> np.ndarray:
    """
    Compute the dimension joint volatility V_j(s) for **1D** per-layer
    trajectories, vectorized over samples.
    NOTE: Implementation inspired by paper:
    "Embedding Trajectory for Out-of-Distribution Detection 
    in Mathematical Reasoning (2024)"

    Definition
    ----------
    For one trajectory s = (y_0, ..., y_{L-1}) of scalars,
        V_j(s) = (1 / L_valid) * sum_{l=1}^{L-1} | y_l - y_{l-1} |,
    where L_valid is the number of *valid* consecutive differences after dropping
    non-finite diffs (NaN/Inf). If L_valid == 0 (e.g., L < 2 or all diffs invalid),
    the volatility is defined as 0.0.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, L)
        Batch of scalar per-layer trajectories. Each row corresponds to one sample,
        each column to a layer index. Must be a real-valued numeric array.

    Returns
    -------
    np.ndarray, shape (n_samples,)
        One volatility value per sample.
    """
    if data.ndim != 2:
        raise ValueError(f"compute_dimension_joint_volatility_1D expects (n, L); got {data.shape}")
    # diffs between consecutive layers
    diffs = np.diff(data, axis=1)                       # (n, L-1)
    # keep only finite diffs
    mask  = np.isfinite(diffs)                          # (n, L-1)
    # sum of absolute valid diffs per sample
    sums  = np.abs(np.where(mask, diffs, 0.0)).sum(axis=1)   # (n,)
    # number of valid diffs per sample
    Lcnt  = mask.sum(axis=1)                            # (n,)
    # average, safe when Lcnt == 0 -> 0
    out   = np.divide(sums, Lcnt, out=np.zeros_like(sums, dtype=float), where=Lcnt > 0)
    return out




def compute_dimension_joint_volatility_2D(data: np.ndarray) -> np.ndarray:
    """
    Compute the dimension joint volatility V_j(s) of **2D** per-layer
    trajectories (vector features per layer), vectorized over samples.
    NOTE: Implementation inspired by paper:
    "Embedding Trajectory for Out-of-Distribution Detection 
    in Mathematical Reasoning (2024)"

    Definition
    ----------
    For one trajectory s = (y_0, ..., y_{L-1}) with y_l in R^D,
        V_j(s) = (1 / L_valid) * sum_{l=1}^{L-1} || y_l - y_{l-1} ||_2,
    where L_valid is the number of *valid* consecutive differences after dropping
    non-finite rows (any NaN/Inf in a difference vector). If L_valid == 0
    (e.g., L < 2 or all diffs invalid), the volatility is defined as 0.0.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, L, D)
        Batch of vector per-layer trajectories. Each row corresponds to one sample,
        the second axis to layer index, and the third axis to feature dimensions.
        Must be a real-valued numeric array.

    Returns
    -------
    np.ndarray, shape (n_samples,)
        One volatility value per sample.
    """
    if data.ndim != 3:
        raise ValueError(f"compute_dimension_joint_volatility_2D expects (n, L, D); got {data.shape}")
    # diffs between consecutive layers
    diffs  = np.diff(data, axis=1)                      # (n, L-1, D)
    # keep rows where all dims are finite
    finite = np.all(np.isfinite(diffs), axis=2)         # (n, L-1)
    # zero-out non-finite rows before norm
    diffs_masked = np.where(finite[:, :, None], diffs, 0.0)
    # L2 norm per (valid) step
    norms = np.linalg.norm(diffs_masked, axis=2)        # (n, L-1)
    # counts of valid steps and sum of norms
    Lcnt  = finite.sum(axis=1)                          # (n,)
    sums  = norms.sum(axis=1)                           # (n,)
    # average, safe when Lcnt == 0 -> 0
    out   = np.divide(sums, Lcnt, out=np.zeros_like(sums, dtype=float), where=Lcnt > 0)
    return out




def compute_id_layer_stats(
    emb_id_fit: np.ndarray, eps: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes per-layer mean and (diagonal) variance statistics 
    from ID-fit embeddings.
    NOTE: Implementation inspired by paper:
    "Embedding Trajectory for Out-of-Distribution Detection 
    in Mathematical Reasoning (2024)"

    Parameters
    ----------
    emb_id_fit (np.ndarray): Array of shape (n_fit, L, D) containing ID-fit embeddings,
        where n_fit = number of fit samples,
                L     = number of layers,
                D     = embedding dimension per layer.
    eps (float, optional): Minimum variance floor for numerical stability. 
        All variance elements will be floored at eps.

    Returns
    -------
    mean (np.ndarray): Array of shape (L, D) containing the mean embedding per layer.
    var  (np.ndarray): Array of shape (L, D) containing the unbiased variance (diagonal)
        per layer, floored at eps for numerical stability.
    """
    if emb_id_fit.ndim != 3:
        raise ValueError(f"emb_id_fit must be (n_fit, L, D); got {emb_id_fit.shape}")
    mean = emb_id_fit.mean(axis=0)            # (L, D)
    var  = emb_id_fit.var(axis=0, ddof=1)     # (L, D) unbiased variance
    var  = np.maximum(var, eps)               # avoid /0 and inf
    return mean, var




def layerwise_diag_mahalanobis(
    emb_split: np.ndarray,
    mean_LD: np.ndarray,
    var_LD: np.ndarray,
    eps: float = 1e-12
) -> np.ndarray:
    """
    Computes the diagonal Mahalanobis distance between embeddings 
    in a split (e.g. ID-test or OOD-test)
    and the reference distribution of ID-fit per layer.
    NOTE: Implementation inspired by paper:
    "Embedding Trajectory for Out-of-Distribution Detection 
    in Mathematical Reasoning (2024)"

    Parameters
    ----------
    emb_split (np.ndarray): Array of shape (n, L, D) containing embeddings 
    for a split (ID-test or OOD) to score
        n = number of evaluated samples,
        L = number of layers,
        D = embedding dimension per layer.
    mean_LD (np.ndarray): Array of shape (L, D) with per-layer means (from ID-fit).
    var_LD  (np.ndarray): Array of shape (L, D) with per-layer variances (from ID-fit).
    eps (float, optional): Small value for numerical stability (used to floor variance).

    Returns
    -------
    md (np.ndarray): Array of shape (n, L) containing  diagonal Mahalanobis distance 
    per sample and layer.
    """
    if emb_split.ndim != 3:
        raise ValueError(f"emb_split must be (n, L, D); got {emb_split.shape}")
    if mean_LD.shape != var_LD.shape:
        raise ValueError("mean_LD and var_LD must have the same (L, D) shape.")
    if emb_split.shape[1:] != mean_LD.shape:
        raise ValueError(
            f"Layer/feat mismatch: emb_split[:, L, D]={emb_split.shape[1:]}, "
            f"stats(L, D)={mean_LD.shape}"
        )

    diff = emb_split - mean_LD[None, :, :]           # (n, L, D)
    inv_var = 1.0 / np.maximum(var_LD, eps)[None, :, :]  # (1, L, D)
    m2 = np.sum((diff * diff) * inv_var, axis=2)     # (n, L)
    return np.sqrt(m2, dtype=np.float64) # (n, L)