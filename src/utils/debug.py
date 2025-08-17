#!/usr/bin/env python3

import numpy as np


def _describe_array(
    name: str,
    X,
    eps: float = 1e-12,
    near_eps: float = 1e-8,
    uniq_round_decimals: int = 6,
    uniq_sample_rows: int = 2048,
) -> None:
    """
    Print diagnostics on an array/tensor:
      - shape, fraction of non-finite, 'zero-like' fraction (rows with ~zero norm for 2D or values ~0 for 1D)
      - mean/std/min/max over finite values
      - near-identical check:
          * 2D: fraction of features with near-zero std across samples,
                approximate unique-rows fraction after rounding (on a subset)
          * 1D: approximate unique-values fraction after rounding
    """
    Xnp = np.asarray(X)
    finite_mask = np.isfinite(Xnp)
    # upcast for stable stats
    Xf = np.where(finite_mask, Xnp, np.nan).astype(np.float64, copy=False)

    # zero-like fraction
    if Xnp.ndim == 1:
        frac_zero_like = float(np.nanmean(np.abs(Xf) <= eps))
    elif Xnp.ndim == 2:
        Xsafe = np.where(finite_mask, Xnp, 0.0).astype(np.float64, copy=False)
        norms = np.linalg.norm(Xsafe, axis=1)
        frac_zero_like = float(np.mean(norms <= eps))
    else:
        frac_zero_like = float(np.nanmean(np.abs(Xf) <= eps))

    # global stats on finite values
    nonfinite_frac = float(1.0 - np.mean(finite_mask))
    mean_f = float(np.nanmean(Xf))
    std_f  = float(np.nanstd(Xf))
    min_f  = float(np.nanmin(Xf))
    max_f  = float(np.nanmax(Xf))

    # near-identical diagnostics
    near_flag = False
    near_msg = ""
    if Xnp.ndim == 2 and Xnp.shape[0] > 1:
        # variability across samples (per-feature std)
        feat_std = np.nanstd(Xf, axis=0)
        mean_feat_std = float(np.nanmean(feat_std))
        frac_low_var_feats = float(np.mean(feat_std <= near_eps))

        # approximate unique-rows fraction (on finite subset, rounded)
        N = min(Xnp.shape[0], uniq_sample_rows)
        sub = Xf[:N]
        row_finite = np.all(np.isfinite(sub), axis=1)
        sub = sub[row_finite]
        if sub.size > 0:
            sub_rounded = np.round(sub, uniq_round_decimals)
            unique_rows = np.unique(sub_rounded, axis=0).shape[0]
            unique_frac = float(unique_rows / max(sub_rounded.shape[0], 1))
        else:
            unique_frac = 0.0

        near_flag = (mean_feat_std <= near_eps) or (frac_low_var_feats >= 0.99) or (unique_frac <= 0.01)
        near_msg = (
            f" | mean_feat_std={mean_feat_std:.3e} "
            f"low_var_feats={frac_low_var_feats:.2%} "
            f"unique_rows_frac≈{unique_frac:.4f} "
            f"near_identical_rows≈{near_flag}"
        )

    elif Xnp.ndim == 1 and Xnp.size > 1:
        N = min(Xnp.size, uniq_sample_rows)
        sub = Xf[:N]
        sub = sub[np.isfinite(sub)]
        if sub.size > 0:
            sub_rounded = np.round(sub, uniq_round_decimals)
            unique_vals = np.unique(sub_rounded).size
            unique_frac = float(unique_vals / max(sub_rounded.size, 1))
        else:
            unique_frac = 0.0

        near_flag = (std_f <= near_eps) or (unique_frac <= 0.01)
        near_msg = f" | unique_vals_frac≈{unique_frac:.4f} near_identical_vals≈{near_flag}"

    print(
        f"{name} shape={Xnp.shape} "
        f"nonfinite={nonfinite_frac:.4f} zero_like={frac_zero_like:.4f} "
        f"mean={mean_f:.6f} std={std_f:.6f} min={min_f:.6f} max={max_f:.6f}"
        f"{near_msg}"
    )
