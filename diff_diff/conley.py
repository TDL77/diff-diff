"""Conley (1999) spatial HAC helpers for diff-diff.

This module contains the geographic-distance and kernel helpers that
implement the Conley (1999) spatial heteroskedasticity-and-autocorrelation-
consistent variance estimator:

    Var̂(β) = (X'X)^{-1} · ( Σ_{i,j} K(d_ij/h) · X_i ε_i ε_j X_j' ) · (X'X)^{-1}

The public dispatch (``compute_robust_vcov`` in :mod:`diff_diff.linalg`)
imports :func:`_validate_conley_kwargs` and :func:`_compute_conley_vcov` and
calls them from the ``vcov_type="conley"`` branch. Tests exercise the
inner helpers directly.

Earth radius constant is 6371.01 km (mean radius), matching R
``conleyreg::haversine_dist`` (Düsterhöft 2021, CRAN v0.1.9). See
``benchmarks/R/README.md`` for the cross-language parity convention.

Phase 2 will add the time-dimension extension (Driscoll-Kraay product
kernel) and a sparse k-d-tree fast path; both will live in this module.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

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


def _pairwise_distance_matrix(coords: np.ndarray, metric) -> np.ndarray:
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
    metric,
    kernel: str,
    n: int,
) -> None:
    """Validate the four Conley kwargs against the design's row count.

    Raises
    ------
    ValueError
        Missing/malformed coords or cutoff; lat/lon out of range under
        haversine; unknown kernel/metric; non-finite or non-positive cutoff.

    Warnings
    --------
    UserWarning
        Emitted when ``n > 20_000`` to flag the dense O(n²) memory cost.
    """
    if coords is None:
        raise ValueError(
            "vcov_type='conley' requires conley_coords (n×2 array of [lat, lon] "
            "or projected coords). Pass via LinearRegression(conley_coords=...) "
            "or compute_robust_vcov(conley_coords=...) on a cross-sectional "
            "design (Phase 1 supports cross-sectional Conley only; panel "
            "estimators are deferred to Phase 2)."
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
    metric,
    kernel: str,
    bread_matrix: np.ndarray,
) -> np.ndarray:
    """Conley (1999) spatial HAC sandwich variance.

    Var̂(β) = bread_inv · (Σ_{i,j} K(d_ij/h) · X_i ε_i ε_j X_j') · bread_inv

    Implemented via the vectorized identity ``meat = S' K S`` where
    ``S = X * residuals[:, None]`` is the (n, k) score matrix and ``K`` is
    the (n, n) kernel matrix. The diagonal contributes the standard White
    (1980) HC0 term ``X_i ε_i² X_i'``.

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
    D = _pairwise_distance_matrix(coords_arr, metric)

    # Apply kernel. Both supported kernels vanish strictly outside the cutoff,
    # so explicit zeroing of D > cutoff is unnecessary (the kernel handles it).
    u = D / cutoff
    if kernel == "bartlett":
        K = _bartlett_kernel(u)
    elif kernel == "uniform":
        K = _uniform_kernel(u)
    else:
        raise ValueError(f"conley_kernel must be 'bartlett' or 'uniform'; got {kernel!r}.")

    # Score matrix S = X * residuals[:, None] is (n, k). Conley meat:
    #   meat[a, b] = Σ_{i,j} K_{ij} · S_{i,a} · S_{j,b}
    # which equals S.T @ K @ S. The K(0) = 1 diagonal contributes the HC0 term.
    # Suppress spurious BLAS-level "divide by zero / overflow" warnings on
    # macOS Accelerate when K is sparse-ish (most off-diagonals are exactly
    # 0 outside the cutoff). The matmul result is mathematically correct;
    # the warning is a subnormal-handling false-positive in the AVX path.
    # We verify finiteness immediately after the matmul.
    S = X * residuals[:, np.newaxis]
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        meat = S.T @ K @ S
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
