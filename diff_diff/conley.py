"""Conley (1999) spatial HAC helpers for diff-diff.

This module contains the geographic-distance and kernel helpers that
implement the Conley (1999) spatial heteroskedasticity-and-autocorrelation-
consistent variance estimator. Two operating modes:

**Cross-sectional (single-period or any pooled cross-section):**

    Var̂(β) = (X'X)^{-1} · ( Σ_{i,j} K(d_ij/h) · X_i ε_i ε_j X_j' ) · (X'X)^{-1}

**Panel block-decomposed (matches R conleyreg lag_cutoff > 0):**

    XeeX_spatial = Σ_t  Σ_{i,j in units}   K_space(d_ij/h) · X_{i,t} ε_{i,t} ε_{j,t} X_{j,t}'
    XeeX_serial  = Σ_u  Σ_{|t-s|<=L, t!=s} (1 - |t-s|/(L+1)) · X_{u,t} ε_{u,t} ε_{u,s} X_{u,s}'
    Var̂(β) = (X'X)^{-1} · (XeeX_spatial + XeeX_serial) · (X'X)^{-1}

The block decomposition (NOT a multiplicative product kernel) matches R
``conleyreg`` (Düsterhöft 2021, CRAN v0.1.9) at ~1e-14 on the parity
fixtures. The temporal kernel is hardcoded Bartlett-style regardless of
the ``conley_kernel`` argument, matching ``conleyreg::time_dist``.

The public dispatch (``compute_robust_vcov`` in :mod:`diff_diff.linalg`)
imports :func:`_validate_conley_kwargs` and :func:`_compute_conley_vcov` and
calls them from the ``vcov_type="conley"`` branch. Tests exercise the
inner helpers directly.

Earth radius constant is 6371.01 km (mean radius), matching R
``conleyreg::haversine_dist`` (Düsterhöft 2021, CRAN v0.1.9). See
``benchmarks/R/README.md`` for the cross-language parity convention.

A sparse k-d-tree fast path for ``n > 20_000`` is a follow-up; the dense
O(n²) distance matrix is emitted with a ``UserWarning`` for now.
"""

from __future__ import annotations

import warnings
from typing import Callable, Literal, Optional, Union

import numpy as np

# Public type alias for the ``conley_metric`` parameter accepted by
# ``compute_robust_vcov``, ``solve_ols``, and ``LinearRegression``. The
# implementation accepts the two named strings as well as a user-supplied
# callable ``(coords1, coords2) -> n×n distance matrix`` for custom
# (e.g. network) distance metrics. Exported so the public signatures
# in :mod:`diff_diff.linalg` can advertise the full accepted type to
# static checkers and IDEs.
ConleyMetric = Union[
    Literal["haversine", "euclidean"],
    Callable[[np.ndarray, np.ndarray], np.ndarray],
]

# Earth's mean radius (km), matching R conleyreg's haversine convention
# (Düsterhöft 2021, conleyreg::haversine_dist in src/distance_functions.cpp,
# CRAN v0.1.9). WGS-84 equatorial radius is 6378.137 km; the 0.01 km delta
# vs 6371.0 is methodologically negligible (Earth mean radius is approximate
# at many more digits) but matters for the 1e-6 cross-language parity bound.
_CONLEY_EARTH_RADIUS_KM = 6371.01

# Empirical threshold for warning about dense O(n²) distance matrix memory
_CONLEY_DENSE_WARN_N = 20_000


def _haversine_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Vectorized great-circle distance in km between two sets of points.

    Inputs are in DEGREES. NumPy broadcasting applies, so passing
    ``lat1=lats[:, None]`` and ``lat2=lats[None, :]`` (with matching
    ``lon1``, ``lon2``) yields the full pairwise n×n distance matrix.

    Earth radius is 6371.01 km (mean radius), matching R ``conleyreg``.
    """
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lat2_r = np.radians(lat2)
    lon2_r = np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    # Clip for numerical robustness — antipodal pairs can produce a >1 by ~eps
    a = np.clip(a, 0.0, 1.0)
    return _CONLEY_EARTH_RADIUS_KM * 2.0 * np.arcsin(np.sqrt(a))


def _pairwise_distance_matrix(coords: np.ndarray, metric: ConleyMetric) -> np.ndarray:
    """Build the dense n×n pairwise distance matrix.

    ``metric`` is one of ``"haversine"`` (lat/lon in degrees, distance in km),
    ``"euclidean"`` (any units), or a callable ``f(coords1, coords2) -> n×n``.
    """
    if metric == "haversine":
        lats = coords[:, 0]
        lons = coords[:, 1]
        return _haversine_km(lats[:, None], lons[:, None], lats[None, :], lons[None, :])
    if metric == "euclidean":
        # Vectorized via squared-distance identity; avoids scipy import path
        # while matching scipy.spatial.distance.cdist to ~1e-14
        diff = coords[:, None, :] - coords[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=-1))
    if callable(metric):
        return np.asarray(metric(coords, coords), dtype=np.float64)
    raise ValueError(
        f"conley_metric must be 'haversine', 'euclidean', or callable; got {metric!r}."
    )


def _bartlett_kernel(u: np.ndarray) -> np.ndarray:
    """Bartlett (linear taper) kernel on pairwise distance: K(u) = max(0, 1 - |u|).

    This is the radial 1-D specialization of Conley (1999)'s Bartlett window
    that R ``conleyreg`` (Düsterhöft 2021), Stata ``acreg`` (Colella et al.
    2019), and Hsiang (2010) all use as their Bartlett path. Conley's
    explicit PSD formula (Eq 3.14, page 12) is the **2-D separable product
    window** ``K(j, k) = (1 - |j|/L_M)(1 - |k|/L_N)`` indexed on a lattice;
    the 1-D radial form on pairwise distance is a practitioner specialization
    that is not explicitly written in the paper and is therefore **not
    PSD-guaranteed**. The caller checks the resulting meat for indefiniteness
    and emits a ``UserWarning`` if the smallest eigenvalue is materially
    negative (regardless of kernel).
    """
    return np.maximum(0.0, 1.0 - np.abs(u))


def _uniform_kernel(u: np.ndarray) -> np.ndarray:
    """Uniform (truncated) kernel: K(u) = 1 if |u| <= 1 else 0.

    Cited as White (1980) truncated estimator; Conley (1999) page 11. Easier
    to interpret than Bartlett but the spectral window is negative in regions
    (Conley 1999 footnote 11), so the resulting meat is not guaranteed PSD.
    Caller emits ``UserWarning`` if any meat eigenvalue is materially negative.
    """
    return (np.abs(u) <= 1.0).astype(np.float64)


def _validate_conley_kwargs(
    coords: Optional[np.ndarray],
    cutoff: Optional[float],
    metric: ConleyMetric,
    kernel: str,
    n: int,
    *,
    time: Optional[np.ndarray] = None,
    unit: Optional[np.ndarray] = None,
    lag_cutoff: Optional[int] = None,
) -> None:
    """Validate the Conley kwargs against the design's row count.

    The first five positional args define the cross-sectional contract (Phase 1).
    The three keyword-only ``time`` / ``unit`` / ``lag_cutoff`` args define the
    panel block-decomposed contract (Phase 2); they are three-way co-required.

    Raises
    ------
    ValueError
        Missing/malformed coords or cutoff; lat/lon out of range under
        haversine; unknown kernel/metric; non-finite or non-positive cutoff;
        partial panel arg set (must pass all three of time/unit/lag_cutoff or none).

    Warnings
    --------
    UserWarning
        Emitted when ``n > 20_000`` to flag the dense O(n²) memory cost.
    """
    if coords is None:
        raise ValueError(
            "vcov_type='conley' requires conley_coords (n×2 array of [lat, lon] "
            "or projected coords). Pass via LinearRegression(conley_coords=...) "
            "or compute_robust_vcov(conley_coords=...)."
        )
    coords_arr = np.asarray(coords, dtype=np.float64)
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 2:
        raise ValueError(f"conley_coords must be a 2-D (n, 2) array; got shape {coords_arr.shape}.")
    if coords_arr.shape[0] != n:
        raise ValueError(f"conley_coords has {coords_arr.shape[0]} rows but X has {n} rows.")
    if not np.isfinite(coords_arr).all():
        raise ValueError("conley_coords contains NaN or inf values.")

    if cutoff is None:
        raise ValueError(
            "vcov_type='conley' requires conley_cutoff_km (a positive finite "
            "bandwidth). No defensible default — see Conley 1999 Section 5 "
            "for the sensitivity-grid recommendation."
        )
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError(f"conley_cutoff_km must be a positive finite number; got {cutoff!r}.")

    if not (metric in ("haversine", "euclidean") or callable(metric)):
        raise ValueError(
            f"conley_metric must be 'haversine', 'euclidean', or callable; got {metric!r}."
        )

    # Lat/lon range checks under haversine. Skipped for euclidean (user's
    # responsibility to pass projected coords with consistent units) and
    # for callables (user supplies their own distance function).
    if metric == "haversine":
        if not ((coords_arr[:, 0] >= -90.0) & (coords_arr[:, 0] <= 90.0)).all():
            raise ValueError(
                "conley_metric='haversine' requires latitude in [-90, 90]; "
                f"got min={coords_arr[:, 0].min()}, max={coords_arr[:, 0].max()}."
            )
        if not ((coords_arr[:, 1] >= -180.0) & (coords_arr[:, 1] <= 180.0)).all():
            raise ValueError(
                "conley_metric='haversine' requires longitude in [-180, 180]; "
                f"got min={coords_arr[:, 1].min()}, max={coords_arr[:, 1].max()}."
            )

    if kernel not in ("bartlett", "uniform"):
        raise ValueError(f"conley_kernel must be 'bartlett' or 'uniform'; got {kernel!r}.")

    # Panel block-decomposed contract: time / unit / lag_cutoff are three-way co-required.
    panel_flags = (time is not None, unit is not None, lag_cutoff is not None)
    n_panel_set = sum(panel_flags)
    if n_panel_set not in (0, 3):
        raise ValueError(
            "conley_time, conley_unit, and conley_lag_cutoff must all be passed "
            "together (all None for cross-sectional Conley; all set for the "
            "panel block-decomposed form). Got "
            f"conley_time set={panel_flags[0]}, conley_unit set={panel_flags[1]}, "
            f"conley_lag_cutoff set={panel_flags[2]}."
        )
    if n_panel_set == 3:
        time_arr = np.asarray(time)
        if time_arr.ndim != 1 or time_arr.shape[0] != n:
            raise ValueError(
                f"conley_time must be a 1-D array of length {n}; got shape {time_arr.shape}."
            )
        # Use pandas.isna for object/categorical dtypes; np.isfinite catches
        # NaN/inf for numeric dtypes only.
        import pandas as _pd

        if _pd.isna(time_arr).any():
            raise ValueError("conley_time contains NaN or missing values.")
        if time_arr.dtype.kind in "fc" and not np.isfinite(time_arr).all():
            raise ValueError("conley_time contains NaN or inf values.")
        unit_arr = np.asarray(unit)
        if unit_arr.ndim != 1 or unit_arr.shape[0] != n:
            raise ValueError(
                f"conley_unit must be a 1-D array of length {n}; got shape {unit_arr.shape}."
            )
        # conley_unit may be any hashable (int, string, categorical). Use
        # pandas.isna for cross-dtype NA detection. NaN unit IDs would silently
        # drop those rows from the per-unit serial HAC sum at
        # `np.unique(unit_arr) + mask_u = unit_arr == u_val`.
        if _pd.isna(unit_arr).any():
            raise ValueError("conley_unit contains NaN or missing values.")
        if not (isinstance(lag_cutoff, (int, np.integer)) and int(lag_cutoff) >= 0):
            raise ValueError(
                f"conley_lag_cutoff must be a non-negative integer; got {lag_cutoff!r}."
            )

    if n > _CONLEY_DENSE_WARN_N:
        memory_gb = (n * n * 8) / 1e9
        warnings.warn(
            f"vcov_type='conley' builds a dense {n}x{n} distance matrix "
            f"(~{memory_gb:.1f} GB float64). The sparse k-d-tree fast path is "
            "deferred to a follow-up PR.",
            UserWarning,
            stacklevel=3,
        )


def _compute_conley_vcov(
    X: np.ndarray,
    residuals: np.ndarray,
    coords: np.ndarray,
    cutoff: float,
    metric: ConleyMetric,
    kernel: str,
    bread_matrix: np.ndarray,
    *,
    time: Optional[np.ndarray] = None,
    unit: Optional[np.ndarray] = None,
    lag_cutoff: Optional[int] = None,
) -> np.ndarray:
    """Conley (1999) spatial HAC sandwich variance.

    Two operating modes:

    **Cross-sectional** (``time`` / ``unit`` / ``lag_cutoff`` all None):

        Var̂(β) = bread_inv · (Σ_{i,j} K(d_ij/h) · X_i ε_i ε_j X_j') · bread_inv

    Implemented via ``meat = S' K S`` where ``S = X * residuals[:, None]``.

    **Panel block-decomposed** (all three keyword-only args set):

        XeeX_spatial = Σ_t  S_t' · K_space_t · S_t                    (within-period sum)
        XeeX_serial  = Σ_u  S_u' · K_time_u · S_u   if lag_cutoff > 0  (within-unit sum)
        Var̂(β) = bread_inv · (XeeX_spatial + XeeX_serial) · bread_inv

    The serial Bartlett kernel ``K_time_u[i, j] = 1{|t_i-t_j| <= L, i != j} ·
    (1 - |t_i-t_j|/(L+1))`` is hardcoded regardless of the user-supplied
    ``kernel`` argument, matching R ``conleyreg::time_dist``. The ``lag != 0``
    exclusion avoids double-counting the diagonal already covered by the
    spatial component.

    Inputs are assumed already validated by :func:`_validate_conley_kwargs`;
    the helper only does the math. Caller is responsible for the validator.

    Returns
    -------
    vcov : ndarray of shape (k, k)

    Notes
    -----
    Neither the uniform kernel (negative spectral regions, Conley 1999
    footnote 11) nor the **radial 1-D Bartlett** specialization implemented
    here is PSD-guaranteed. Conley's explicit PSD formula (Eq 3.14) is the
    2-D separable product window on a lattice; the radial pairwise form is
    a practitioner specialization (R ``conleyreg``, Stata ``acreg``, Hsiang
    2010) that is not formally PSD. We emit a ``UserWarning`` if the smallest
    meat eigenvalue is materially negative (< -1e-12) regardless of kernel.
    """
    coords_arr = np.asarray(coords, dtype=np.float64)
    S = X * residuals[:, np.newaxis]

    def _kernel_fn(u: np.ndarray) -> np.ndarray:
        if kernel == "bartlett":
            return _bartlett_kernel(u)
        if kernel == "uniform":
            return _uniform_kernel(u)
        raise ValueError(f"conley_kernel must be 'bartlett' or 'uniform'; got {kernel!r}.")

    # Suppress spurious BLAS-level "divide by zero / overflow" warnings on
    # macOS Accelerate when K is sparse-ish (most off-diagonals are exactly
    # 0 outside the cutoff). The matmul result is mathematically correct;
    # the warning is a subnormal-handling false-positive in the AVX path.
    # We verify finiteness immediately after.
    if time is None:
        # Phase 1 cross-sectional path: full n×n spatial sandwich.
        D = _pairwise_distance_matrix(coords_arr, metric)
        K = _kernel_fn(D / cutoff)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            meat = S.T @ K @ S
    else:
        # Phase 2 panel block-decomposed path (matches R conleyreg).
        time_arr = np.asarray(time)
        unit_arr = np.asarray(unit)
        k = X.shape[1]
        meat = np.zeros((k, k))
        # Spatial component: within-period sandwich, summed across periods.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            for t_val in np.unique(time_arr):
                mask_t = time_arr == t_val
                S_t = S[mask_t]
                D_t = _pairwise_distance_matrix(coords_arr[mask_t], metric)
                K_t = _kernel_fn(D_t / cutoff)
                meat += S_t.T @ K_t @ S_t
            # Serial component: within-unit Bartlett HAC for lag in {1..L},
            # excluding lag=0 to avoid double-counting the spatial diagonal.
            # Bartlett form hardcoded (matches conleyreg::time_dist).
            # The validator guarantees lag_cutoff is not None when time is not None.
            assert lag_cutoff is not None
            L = int(lag_cutoff)
            if L > 0:
                for u_val in np.unique(unit_arr):
                    mask_u = unit_arr == u_val
                    S_u = S[mask_u]
                    t_u = np.asarray(time_arr[mask_u], dtype=np.float64)
                    lag_mat = np.abs(t_u[:, None] - t_u[None, :])
                    K_u = ((lag_mat <= L) & (lag_mat != 0)).astype(np.float64) * (
                        1.0 - lag_mat / (L + 1.0)
                    )
                    meat += S_u.T @ K_u @ S_u
    if not np.all(np.isfinite(meat)):
        raise ValueError(
            "Conley meat contains non-finite values; check residuals and "
            "score matrix for NaN/Inf."
        )

    # PSD guard. Neither the uniform kernel (Conley 1999 fn 11) nor the
    # radial 1-D Bartlett specialization is formally PSD-guaranteed —
    # Conley's explicit PSD Bartlett formula (Eq 3.14) is the 2-D separable
    # product window, not the 1-D radial pairwise form that R `conleyreg`,
    # Stata `acreg`, and this implementation use. Check both kernels.
    eigvals = np.linalg.eigvalsh(meat)
    if eigvals.size and eigvals.min() < -1e-12:
        warnings.warn(
            f"Conley meat with conley_kernel={kernel!r} has a materially "
            f"negative eigenvalue ({eigvals.min():.2e}); the variance "
            "estimator is not guaranteed PSD on this design. Both "
            "supported kernels (radial bartlett and uniform) are "
            "practitioner specializations of Conley 1999 and are not "
            "formally PSD-guaranteed; consider varying conley_cutoff_km "
            "or reviewing the design for collinearity / degenerate "
            "residual structure.",
            UserWarning,
            stacklevel=3,
        )

    # Sandwich via two solves (mirrors _compute_cr2_bm pattern in linalg.py)
    try:
        temp = np.linalg.solve(bread_matrix, meat)
        vcov = np.linalg.solve(bread_matrix, temp.T).T
    except np.linalg.LinAlgError as e:
        if "Singular" in str(e):
            raise ValueError(
                "Design matrix is rank-deficient (singular X'X matrix). "
                "Cannot compute Conley spatial HAC variance."
            ) from e
        raise

    return vcov
