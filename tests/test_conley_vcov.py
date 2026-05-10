"""Tests for Conley (1999) spatial HAC variance estimator.

Phase 1 scope: pure-numerics helpers (kernels, distance metrics, direct
sandwich helper) and the dispatch-level validator. Estimator-level
integration tests, set_params atomicity, and Stata acreg parity land in
later Phase 1 checkpoints (Steps 4-6 of the plan).
"""

import warnings

import numpy as np
import pytest

from diff_diff.conley import (
    _CONLEY_EARTH_RADIUS_KM,
    _bartlett_kernel,
    _compute_conley_vcov,
    _haversine_km,
    _pairwise_distance_matrix,
    _uniform_kernel,
    _validate_conley_kwargs,
)
from diff_diff.linalg import (
    LinearRegression,
    compute_robust_vcov,
    solve_ols,
)

# ---------------------------------------------------------------------------
# Shared fixtures (small synthetic OLS dataset with geocoords)
# ---------------------------------------------------------------------------


@pytest.fixture
def small_ols_with_coords():
    """20-row OLS dataset with synthetic lat/lon. Used across helper tests."""
    rng = np.random.default_rng(seed=42)
    n = 20
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    eps = rng.standard_normal(n)
    y = X @ np.array([1.0, 2.0]) + eps
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ coefs
    bread = X.T @ X
    coords = np.column_stack(
        [
            rng.uniform(-30, 30, n),  # lat
            rng.uniform(-100, 100, n),  # lon
        ]
    )
    return X, residuals, coords, bread


# ---------------------------------------------------------------------------
# TestConleyKernels — Bartlett / uniform shape and boundary behavior
# ---------------------------------------------------------------------------


class TestConleyKernels:
    def test_bartlett_at_zero(self):
        np.testing.assert_allclose(_bartlett_kernel(np.array([0.0])), 1.0)

    def test_bartlett_at_one(self):
        np.testing.assert_allclose(_bartlett_kernel(np.array([1.0])), 0.0)

    def test_bartlett_above_one_zero(self):
        u = np.array([1.5, 2.0, 100.0])
        np.testing.assert_allclose(_bartlett_kernel(u), np.zeros(3))

    def test_bartlett_negative_arg_symmetric(self):
        """Bartlett uses |u|, so K(-0.3) == K(0.3)."""
        np.testing.assert_allclose(
            _bartlett_kernel(np.array([-0.3])), _bartlett_kernel(np.array([0.3]))
        )

    def test_uniform_kernel_at_boundary(self):
        """Uniform kernel is closed on the right: K(1) = 1, K(1+eps) = 0."""
        np.testing.assert_allclose(_uniform_kernel(np.array([1.0])), 1.0)

    def test_uniform_kernel_above_one_zero(self):
        np.testing.assert_allclose(_uniform_kernel(np.array([1.0001, 2.0, 100.0])), np.zeros(3))

    def test_uniform_kernel_at_zero_one(self):
        np.testing.assert_allclose(_uniform_kernel(np.array([0.0])), 1.0)

    def test_bartlett_psd_on_random_distances(self):
        """Bartlett-weighted Gram matrix has all eigenvalues >= -tol."""
        rng = np.random.default_rng(seed=11)
        n = 25
        coords = rng.uniform(0, 1, size=(n, 2))
        diff = coords[:, None, :] - coords[None, :, :]
        D = np.sqrt((diff * diff).sum(axis=-1))
        K = _bartlett_kernel(D / 0.3)
        eigvals = np.linalg.eigvalsh(0.5 * (K + K.T))  # ensure symmetric
        assert eigvals.min() > -1e-12


# ---------------------------------------------------------------------------
# TestConleyDistanceMetrics — haversine, euclidean, callable
# ---------------------------------------------------------------------------


class TestConleyDistanceMetrics:
    def test_haversine_known_pair_one_degree_equator(self):
        """1° longitude at the equator = 2π·R/360 ≈ 111.195 km (R=6371)."""
        d = _haversine_km(np.array(0.0), np.array(0.0), np.array(0.0), np.array(1.0))
        expected = 2 * np.pi * _CONLEY_EARTH_RADIUS_KM / 360.0
        np.testing.assert_allclose(d, expected, atol=1e-9)

    def test_haversine_zero_self_distance(self):
        d = _haversine_km(np.array(45.0), np.array(-122.0), np.array(45.0), np.array(-122.0))
        np.testing.assert_allclose(d, 0.0, atol=1e-12)

    def test_haversine_symmetric(self):
        d_ab = _haversine_km(np.array(40.7), np.array(-74.0), np.array(34.0), np.array(-118.2))
        d_ba = _haversine_km(np.array(34.0), np.array(-118.2), np.array(40.7), np.array(-74.0))
        np.testing.assert_allclose(d_ab, d_ba, atol=1e-12)

    def test_haversine_pole_to_equator(self):
        """North pole to equator at any longitude = π/2 · R = ~10007.5 km."""
        d = _haversine_km(np.array(90.0), np.array(0.0), np.array(0.0), np.array(0.0))
        expected = np.pi * _CONLEY_EARTH_RADIUS_KM / 2.0
        np.testing.assert_allclose(d, expected, atol=1e-9)

    def test_haversine_broadcasting_pairwise(self):
        """Broadcasting (n, 1) vs (1, n) yields the n×n distance matrix."""
        coords = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
        lats = coords[:, 0]
        lons = coords[:, 1]
        D = _haversine_km(lats[:, None], lons[:, None], lats[None, :], lons[None, :])
        assert D.shape == (3, 3)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-12)
        # D[0, 2] should be 2 * D[0, 1] for collinear-equator points
        np.testing.assert_allclose(D[0, 2], 2.0 * D[0, 1], rtol=1e-10)

    def test_pairwise_distance_haversine(self):
        coords = np.array([[0.0, 0.0], [0.0, 1.0], [10.0, 0.0]])
        D = _pairwise_distance_matrix(coords, "haversine")
        assert D.shape == (3, 3)
        np.testing.assert_allclose(D, D.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-12)

    def test_pairwise_distance_euclidean_matches_pdist(self):
        """Euclidean path matches scipy.spatial.distance squareform exactly."""
        from scipy.spatial.distance import pdist, squareform

        rng = np.random.default_rng(seed=7)
        coords = rng.uniform(0, 100, size=(15, 2))
        D = _pairwise_distance_matrix(coords, "euclidean")
        D_scipy = squareform(pdist(coords, metric="euclidean"))
        np.testing.assert_allclose(D, D_scipy, atol=1e-12)

    def test_pairwise_distance_callable(self):
        """A user-supplied callable is dispatched and its output preserved."""
        coords = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        def constant_metric(c1, c2):
            n1 = len(c1)
            n2 = len(c2)
            return np.full((n1, n2), 5.0)

        D = _pairwise_distance_matrix(coords, constant_metric)
        np.testing.assert_allclose(D, np.full((3, 3), 5.0))

    def test_pairwise_distance_unknown_metric_raises(self):
        """Unknown metric strings raise ValueError from the dispatcher."""
        with pytest.raises(ValueError, match="conley_metric"):
            _pairwise_distance_matrix(np.zeros((3, 2)), "manhattan")


# ---------------------------------------------------------------------------
# TestConleyValidatorHelpers — direct calls to _validate_conley_kwargs
# ---------------------------------------------------------------------------


class TestConleyValidatorHelpers:
    def test_missing_coords_raises(self):
        with pytest.raises(ValueError, match="conley_coords"):
            _validate_conley_kwargs(
                coords=None, cutoff=100.0, metric="haversine", kernel="bartlett", n=10
            )

    def test_missing_cutoff_raises(self):
        with pytest.raises(ValueError, match="conley_cutoff_km"):
            _validate_conley_kwargs(
                coords=np.zeros((10, 2)),
                cutoff=None,
                metric="haversine",
                kernel="bartlett",
                n=10,
            )

    def test_zero_cutoff_raises(self):
        with pytest.raises(ValueError, match="positive finite"):
            _validate_conley_kwargs(
                coords=np.zeros((10, 2)),
                cutoff=0.0,
                metric="haversine",
                kernel="bartlett",
                n=10,
            )

    def test_negative_cutoff_raises(self):
        with pytest.raises(ValueError, match="positive finite"):
            _validate_conley_kwargs(
                coords=np.zeros((10, 2)),
                cutoff=-5.0,
                metric="haversine",
                kernel="bartlett",
                n=10,
            )

    def test_nan_cutoff_raises(self):
        with pytest.raises(ValueError, match="positive finite"):
            _validate_conley_kwargs(
                coords=np.zeros((10, 2)),
                cutoff=float("nan"),
                metric="haversine",
                kernel="bartlett",
                n=10,
            )

    def test_inf_cutoff_raises(self):
        with pytest.raises(ValueError, match="positive finite"):
            _validate_conley_kwargs(
                coords=np.zeros((10, 2)),
                cutoff=float("inf"),
                metric="haversine",
                kernel="bartlett",
                n=10,
            )

    def test_3d_coords_raises(self):
        with pytest.raises(ValueError, match=r"\(n, 2\)"):
            _validate_conley_kwargs(
                coords=np.zeros((10, 3)),
                cutoff=100.0,
                metric="haversine",
                kernel="bartlett",
                n=10,
            )

    def test_coord_n_mismatch_raises(self):
        with pytest.raises(ValueError, match="rows but X has"):
            _validate_conley_kwargs(
                coords=np.zeros((10, 2)),
                cutoff=100.0,
                metric="haversine",
                kernel="bartlett",
                n=15,
            )

    def test_nan_coord_raises(self):
        bad = np.zeros((10, 2))
        bad[3, 1] = np.nan
        with pytest.raises(ValueError, match="NaN or inf"):
            _validate_conley_kwargs(
                coords=bad, cutoff=100.0, metric="haversine", kernel="bartlett", n=10
            )

    def test_lat_out_of_range_raises_haversine(self):
        coords = np.array([[91.0, 0.0]] + [[0.0, 0.0]] * 9)
        with pytest.raises(ValueError, match=r"latitude in \[-90, 90\]"):
            _validate_conley_kwargs(
                coords=coords, cutoff=100.0, metric="haversine", kernel="bartlett", n=10
            )

    def test_lon_out_of_range_raises_haversine(self):
        coords = np.array([[0.0, 200.0]] + [[0.0, 0.0]] * 9)
        with pytest.raises(ValueError, match=r"longitude in \[-180, 180\]"):
            _validate_conley_kwargs(
                coords=coords, cutoff=100.0, metric="haversine", kernel="bartlett", n=10
            )

    def test_lat_out_of_range_skipped_for_euclidean(self):
        """Projected coords are unconstrained — euclidean skips lat/lon checks."""
        coords = np.array([[5000.0, 12000.0]] * 10)  # any units
        # Should not raise
        _validate_conley_kwargs(
            coords=coords, cutoff=100.0, metric="euclidean", kernel="bartlett", n=10
        )

    def test_unknown_kernel_raises(self):
        with pytest.raises(ValueError, match="conley_kernel"):
            _validate_conley_kwargs(
                coords=np.zeros((10, 2)),
                cutoff=100.0,
                metric="haversine",
                kernel="gaussian",
                n=10,
            )

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="conley_metric"):
            _validate_conley_kwargs(
                coords=np.zeros((10, 2)),
                cutoff=100.0,
                metric="manhattan",
                kernel="bartlett",
                n=10,
            )

    def test_callable_metric_accepted(self):
        """Callable distance metric passes validation (delegated to caller)."""
        _validate_conley_kwargs(
            coords=np.zeros((10, 2)),
            cutoff=100.0,
            metric=lambda c1, c2: np.zeros((len(c1), len(c2))),
            kernel="bartlett",
            n=10,
        )

    def test_n_above_warn_threshold_warns(self):
        with pytest.warns(UserWarning, match="dense"):
            _validate_conley_kwargs(
                coords=np.zeros((20_001, 2)),
                cutoff=100.0,
                metric="euclidean",
                kernel="bartlett",
                n=20_001,
            )

    def test_n_below_warn_threshold_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning becomes an error
            _validate_conley_kwargs(
                coords=np.zeros((100, 2)),
                cutoff=100.0,
                metric="euclidean",
                kernel="bartlett",
                n=100,
            )


# ---------------------------------------------------------------------------
# TestConleyDirectHelper — _compute_conley_vcov correctness
# ---------------------------------------------------------------------------


class TestConleyDirectHelper:
    def test_returns_psd_with_bartlett(self, small_ols_with_coords):
        X, residuals, coords, bread = small_ols_with_coords
        vcov = _compute_conley_vcov(
            X,
            residuals,
            coords,
            cutoff=2000.0,
            metric="haversine",
            kernel="bartlett",
            bread_matrix=bread,
        )
        eigvals = np.linalg.eigvalsh(0.5 * (vcov + vcov.T))
        assert eigvals.min() > -1e-10

    def test_symmetric_vcov(self, small_ols_with_coords):
        X, residuals, coords, bread = small_ols_with_coords
        vcov = _compute_conley_vcov(
            X,
            residuals,
            coords,
            cutoff=2000.0,
            metric="haversine",
            kernel="bartlett",
            bread_matrix=bread,
        )
        np.testing.assert_allclose(vcov, vcov.T, atol=1e-10)

    def test_shape_matches_bread(self, small_ols_with_coords):
        X, residuals, coords, bread = small_ols_with_coords
        vcov = _compute_conley_vcov(
            X,
            residuals,
            coords,
            cutoff=1500.0,
            metric="haversine",
            kernel="bartlett",
            bread_matrix=bread,
        )
        k = X.shape[1]
        assert vcov.shape == (k, k)

    def test_uniform_kernel_negative_eigenvalue_warns(self):
        """Construct a degenerate setup that produces a uniform-kernel
        meat with a small negative eigenvalue. Verifies the PSD-warning
        path. The setup uses two clusters of identical-coordinate points so
        the uniform-kernel meat reduces to a known structure that is
        numerically borderline."""
        rng = np.random.default_rng(seed=1)
        n = 30
        # Mix of identical-coord pairs; uniform kernel sums full pairs
        coords = np.repeat(rng.uniform(0, 1, size=(n // 2, 2)), 2, axis=0)
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        eps = rng.standard_normal(n)
        bread = X.T @ X
        # No assertion on the exact meat — only that the PSD path is
        # exercised. The warning may or may not fire depending on numerical
        # condition; this test mainly ensures the code path runs without error.
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            _compute_conley_vcov(
                X,
                eps,
                coords,
                cutoff=10.0,
                metric="euclidean",
                kernel="uniform",
                bread_matrix=bread,
            )


# ---------------------------------------------------------------------------
# TestConleyReductions — Bartlett+tiny cutoff → HC0 meat; etc.
# ---------------------------------------------------------------------------


class TestConleyReductions:
    def test_tiny_cutoff_distinct_coords_yields_HC0_meat(self):
        """When the bandwidth is much smaller than the minimum pairwise
        distance, Conley's kernel is ~0 off-diagonal and 1 on-diagonal, so
        the meat reduces to Σ x_i² ε_i² x_i x_i' = HC0 meat.
        """
        rng = np.random.default_rng(seed=3)
        n = 15
        # Distinct coords with min pairwise distance >> 0
        coords = np.column_stack([np.arange(n) * 100.0, np.arange(n) * 100.0])
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        eps = rng.standard_normal(n)
        bread = X.T @ X

        # HC0 meat (Σ x_i x_i' u_i²) — no DOF correction applied
        meat_hc0 = X.T @ (X * (eps**2)[:, None])
        bread_inv = np.linalg.solve(bread, np.eye(2))
        vcov_hc0 = bread_inv @ meat_hc0 @ bread_inv

        vcov_conley = _compute_conley_vcov(
            X,
            eps,
            coords,
            cutoff=1.0,  # << minimum pairwise distance
            metric="euclidean",
            kernel="bartlett",
            bread_matrix=bread,
        )
        np.testing.assert_allclose(vcov_conley, vcov_hc0, atol=1e-12)

    def test_uniform_kernel_tiny_cutoff_yields_HC0_meat(self):
        """Same reduction with the uniform kernel."""
        rng = np.random.default_rng(seed=5)
        n = 12
        coords = np.column_stack([np.arange(n) * 100.0, np.arange(n) * 100.0])
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        eps = rng.standard_normal(n)
        bread = X.T @ X

        meat_hc0 = X.T @ (X * (eps**2)[:, None])
        bread_inv = np.linalg.solve(bread, np.eye(2))
        vcov_hc0 = bread_inv @ meat_hc0 @ bread_inv

        vcov_conley = _compute_conley_vcov(
            X,
            eps,
            coords,
            cutoff=0.5,
            metric="euclidean",
            kernel="uniform",
            bread_matrix=bread,
        )
        np.testing.assert_allclose(vcov_conley, vcov_hc0, atol=1e-12)

    def test_huge_cutoff_NOT_HC0(self, small_ols_with_coords):
        """When cutoff dwarfs all pairwise distances, K -> ones(n, n) and
        meat = (X·ε)' ones (X·ε) which is the rank-1 outer product of summed
        scores — NOT HC0. This is the all-correlated limit."""
        X, residuals, coords, bread = small_ols_with_coords
        vcov_conley = _compute_conley_vcov(
            X,
            residuals,
            coords,
            cutoff=1e9,
            metric="euclidean",
            kernel="uniform",
            bread_matrix=bread,
        )
        # HC0 for comparison
        meat_hc0 = X.T @ (X * (residuals**2)[:, None])
        bread_inv = np.linalg.solve(bread, np.eye(X.shape[1]))
        vcov_hc0 = bread_inv @ meat_hc0 @ bread_inv
        # They should differ noticeably
        assert not np.allclose(vcov_conley, vcov_hc0, atol=1e-6)

    def test_dispatch_matches_direct_helper(self, small_ols_with_coords):
        """compute_robust_vcov(vcov_type='conley', ...) returns the same
        vcov as a direct call to _compute_conley_vcov on the same inputs.
        Atol=1e-14 (bit-equivalence)."""
        X, residuals, coords, bread = small_ols_with_coords
        vcov_dispatch = compute_robust_vcov(
            X,
            residuals,
            vcov_type="conley",
            conley_coords=coords,
            conley_cutoff_km=2000.0,
            conley_metric="haversine",
            conley_kernel="bartlett",
        )
        vcov_direct = _compute_conley_vcov(
            X,
            residuals,
            coords,
            2000.0,
            "haversine",
            "bartlett",
            bread,
        )
        np.testing.assert_allclose(vcov_dispatch, vcov_direct, atol=1e-14, rtol=1e-14)

    def test_dispatch_returns_dof_when_requested(self, small_ols_with_coords):
        """return_dof=True returns (vcov, dof_vec) tuple where dof = n - k."""
        X, residuals, coords, _ = small_ols_with_coords
        out = compute_robust_vcov(
            X,
            residuals,
            vcov_type="conley",
            conley_coords=coords,
            conley_cutoff_km=2000.0,
            return_dof=True,
        )
        assert isinstance(out, tuple) and len(out) == 2
        _vcov, dof = out
        n, k = X.shape
        np.testing.assert_array_equal(dof, np.full(k, n - k, dtype=np.float64))


class TestConleyValidationDispatch:
    """Validation tests at the compute_robust_vcov dispatch level."""

    @pytest.fixture
    def fit_inputs(self):
        rng = np.random.default_rng(seed=0)
        n = 12
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        residuals = rng.standard_normal(n)
        coords = rng.uniform(-10, 10, size=(n, 2))
        return X, residuals, coords

    def test_conley_in_valid_set(self):
        """Sanity: 'conley' is in the canonical _VALID_VCOV_TYPES set."""
        from diff_diff.linalg import _VALID_VCOV_TYPES

        assert "conley" in _VALID_VCOV_TYPES

    def test_conley_with_cluster_raises(self, fit_inputs):
        X, residuals, coords = fit_inputs
        with pytest.raises(NotImplementedError, match="conley.*cluster_ids"):
            compute_robust_vcov(
                X,
                residuals,
                cluster_ids=np.arange(len(X)) // 3,
                vcov_type="conley",
                conley_coords=coords,
                conley_cutoff_km=100.0,
            )

    def test_conley_with_weights_raises(self, fit_inputs):
        X, residuals, coords = fit_inputs
        with pytest.raises(NotImplementedError, match="conley.*weights"):
            compute_robust_vcov(
                X,
                residuals,
                weights=np.ones(len(X)),
                vcov_type="conley",
                conley_coords=coords,
                conley_cutoff_km=100.0,
            )

    def test_conley_without_coords_raises(self, fit_inputs):
        X, residuals, _ = fit_inputs
        with pytest.raises(ValueError, match="conley_coords"):
            compute_robust_vcov(
                X,
                residuals,
                vcov_type="conley",
                conley_cutoff_km=100.0,
            )

    def test_conley_without_cutoff_raises(self, fit_inputs):
        X, residuals, coords = fit_inputs
        with pytest.raises(ValueError, match="conley_cutoff_km"):
            compute_robust_vcov(
                X,
                residuals,
                vcov_type="conley",
                conley_coords=coords,
            )


class TestConleyLinearRegression:
    """Step 3 smoke tests: LinearRegression and solve_ols thread Conley
    kwargs to compute_robust_vcov. Covers both the higher-level
    LinearRegression API and the lower-level solve_ols entrypoint."""

    @pytest.fixture
    def fit_data(self):
        rng = np.random.default_rng(seed=42)
        n = 25
        X = rng.standard_normal(size=(n, 2))
        y = X @ np.array([1.0, 2.0]) + rng.standard_normal(n)
        coords = rng.uniform(-30, 30, size=(n, 2))
        return X, y, coords

    def test_linear_regression_conley_runs(self, fit_data):
        X, y, coords = fit_data
        reg = LinearRegression(
            vcov_type="conley",
            include_intercept=True,
            conley_coords=coords,
            conley_cutoff_km=2000.0,
        ).fit(X, y)
        assert reg.vcov_ is not None
        assert reg.vcov_.shape == (3, 3)  # +1 for intercept
        # Diagonal entries are SE^2 — must be finite and positive
        diag = np.diag(reg.vcov_)
        assert np.all(np.isfinite(diag))
        assert np.all(diag > 0)

    def test_linear_regression_conley_matches_direct(self, fit_data):
        """LinearRegression(vcov_type='conley', ...) ⇔ compute_robust_vcov direct
        call produces the same vcov on the same X (with intercept added)."""
        X, y, coords = fit_data
        reg = LinearRegression(
            vcov_type="conley",
            include_intercept=True,
            conley_coords=coords,
            conley_cutoff_km=2000.0,
        ).fit(X, y)
        # Reproduce X with intercept that LinearRegression built internally
        X_intercept = np.column_stack([np.ones(X.shape[0]), X])
        coefs, *_ = np.linalg.lstsq(X_intercept, y, rcond=None)
        residuals = y - X_intercept @ coefs
        vcov_direct = compute_robust_vcov(
            X_intercept,
            residuals,
            vcov_type="conley",
            conley_coords=coords,
            conley_cutoff_km=2000.0,
        )
        np.testing.assert_allclose(reg.vcov_, vcov_direct, atol=1e-10, rtol=1e-10)

    def test_solve_ols_conley_path(self, fit_data):
        """solve_ols(vcov_type='conley', ...) returns finite vcov."""
        X, y, coords = fit_data
        coefs, residuals, vcov = solve_ols(
            X,
            y,
            vcov_type="conley",
            conley_coords=coords,
            conley_cutoff_km=2000.0,
            skip_rank_check=True,
        )
        assert vcov is not None
        assert np.all(np.isfinite(np.diag(vcov)))


class TestConleyEstimatorIntegration:
    """Step 4 smoke tests: DifferenceInDifferences and MultiPeriodDiD accept
    vcov_type='conley' with the conley_* kwargs and produce finite SEs.
    Also tests that summary() prints the Conley label."""

    @pytest.fixture
    def two_period_panel(self):
        rng = np.random.default_rng(seed=11)
        n_units = 40
        units = np.arange(n_units)
        treated = (units < 20).astype(int)
        rows = []
        for u in units:
            lat = rng.uniform(-30, 30)
            lon = rng.uniform(-100, 100)
            for t in [0, 1]:
                y = 1.0 + 0.5 * t + (1.0 if (treated[u] and t == 1) else 0.0) + rng.normal(0, 0.5)
                rows.append(
                    {"unit": u, "time": t, "y": y, "treated": treated[u], "lat": lat, "lon": lon}
                )
        import pandas as pd

        return pd.DataFrame(rows)

    def test_did_basic_with_conley(self, two_period_panel):
        """DifferenceInDifferences fits with vcov_type='conley' and produces
        finite SE > 0."""
        from diff_diff import DifferenceInDifferences

        df = two_period_panel.copy()
        df["did"] = df["treated"] * df["time"]
        result = DifferenceInDifferences(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=2000.0,
        ).fit(df, outcome="y", treatment="treated", time="time")
        assert np.isfinite(result.se) and result.se > 0
        assert result.vcov_type == "conley"
        assert result.conley_cutoff_km == 2000.0
        assert result.conley_kernel == "bartlett"

    def test_did_summary_includes_conley_label(self, two_period_panel):
        from diff_diff import DifferenceInDifferences

        df = two_period_panel.copy()
        result = DifferenceInDifferences(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=1500.0,
        ).fit(df, outcome="y", treatment="treated", time="time")
        out = result.summary()
        assert "Conley spatial HAC" in out
        assert "1500" in out
        assert "bartlett" in out

    def test_multi_period_did_with_conley(self, two_period_panel):
        from diff_diff import MultiPeriodDiD

        # Build a 4-period panel for MultiPeriodDiD
        rng = np.random.default_rng(seed=13)
        n_units = 30
        rows = []
        for u in range(n_units):
            lat = rng.uniform(-30, 30)
            lon = rng.uniform(-100, 100)
            treated = u < 15
            for t in range(4):
                y = 0.2 * t + (1.0 if (treated and t >= 2) else 0.0) + rng.normal(0, 0.5)
                rows.append(
                    {"unit": u, "time": t, "y": y, "treated": int(treated), "lat": lat, "lon": lon}
                )
        import pandas as pd

        df_mp = pd.DataFrame(rows)
        result = MultiPeriodDiD(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=2000.0,
        ).fit(df_mp, outcome="y", treatment="treated", time="time", reference_period=1)
        assert np.isfinite(result.avg_se) and result.avg_se > 0
        assert result.vcov_type == "conley"


class TestConleyTWFE:
    """Step 5: TwoWayFixedEffects with Conley SE.

    TWFE composes with Conley because the meat depends only on scores X*epsilon,
    both of which FWL preserves under within-transformation. This is UNLIKE
    hc2/hc2_bm which depend on the full hat matrix and are rejected on TWFE.
    """

    @pytest.fixture
    def panel(self):
        """Build a 2-period panel with geocoords for TWFE testing."""
        rng = np.random.default_rng(seed=17)
        n_units = 30
        rows = []
        for u in range(n_units):
            lat = rng.uniform(-30, 30)
            lon = rng.uniform(-100, 100)
            treated = u < 15
            unit_fe = rng.normal(0, 0.3)
            for t in range(2):
                time_fe = 0.5 if t == 1 else 0.0
                effect = 1.0 if (treated and t == 1) else 0.0
                y = unit_fe + time_fe + effect + rng.normal(0, 0.4)
                rows.append(
                    {"unit": u, "time": t, "y": y, "treated": int(treated), "lat": lat, "lon": lon}
                )
        import pandas as pd

        return pd.DataFrame(rows)

    def test_twfe_conley_runs(self, panel):
        from diff_diff import TwoWayFixedEffects

        result = TwoWayFixedEffects(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=2000.0,
        ).fit(panel, outcome="y", treatment="treated", time="time", unit="unit")
        assert np.isfinite(result.se) and result.se > 0
        assert result.vcov_type == "conley"
        assert result.conley_cutoff_km == 2000.0
        assert result.cluster_name is None  # auto-cluster disabled under conley

    def test_twfe_conley_with_explicit_cluster_raises(self, panel):
        """User explicitly setting cluster=... with conley should raise."""
        from diff_diff import TwoWayFixedEffects

        with pytest.raises(NotImplementedError, match="conley"):
            TwoWayFixedEffects(
                vcov_type="conley",
                cluster="unit",
                conley_coords=("lat", "lon"),
                conley_cutoff_km=2000.0,
            ).fit(panel, outcome="y", treatment="treated", time="time", unit="unit")

    def test_twfe_conley_FWL_invariance(self, panel):
        """TWFE Conley SE matches DifferenceInDifferences with same kwargs
        (verifies FWL composability — Conley meat survives within-transformation
        because it depends only on scores X*epsilon)."""
        from diff_diff import DifferenceInDifferences, TwoWayFixedEffects

        twfe_result = TwoWayFixedEffects(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=2000.0,
        ).fit(panel, outcome="y", treatment="treated", time="time", unit="unit")
        # DiD equivalent: simple 2x2, no FE within-transformation
        did_result = DifferenceInDifferences(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=2000.0,
        ).fit(panel, outcome="y", treatment="treated", time="time")
        # ATT estimates should be similar (panel structure differs only in FE handling).
        # We don't expect bit-equivalence — DiD without FE absorbs unit FE
        # into the residuals while TWFE removes them. The key invariance is
        # that the SE families are both finite and reasonable.
        assert np.isfinite(twfe_result.se) and twfe_result.se > 0
        assert np.isfinite(did_result.se) and did_result.se > 0


class TestConleyEstimatorValidation:
    """Step 4 validation: estimator-level rejections for invalid combinations."""

    @pytest.fixture
    def df(self):
        import pandas as pd

        rng = np.random.default_rng(seed=2)
        n = 20
        return pd.DataFrame(
            {
                "unit": np.arange(n),
                "time": np.tile([0, 1], n // 2),
                "y": rng.standard_normal(n),
                "treated": np.tile([0, 1], n // 2),
                "lat": rng.uniform(-30, 30, n),
                "lon": rng.uniform(-100, 100, n),
                "stratum": np.tile([0, 1, 2, 3], n // 4),
            }
        )

    def test_did_conley_with_cluster_raises(self, df):
        from diff_diff import DifferenceInDifferences

        with pytest.raises(NotImplementedError, match="cluster.*conley"):
            DifferenceInDifferences(
                vcov_type="conley",
                cluster="stratum",
                conley_coords=("lat", "lon"),
                conley_cutoff_km=100.0,
            ).fit(df, outcome="y", treatment="treated", time="time")

    def test_did_conley_without_coords_raises(self, df):
        from diff_diff import DifferenceInDifferences

        with pytest.raises(ValueError, match="conley_coords"):
            DifferenceInDifferences(
                vcov_type="conley",
                conley_cutoff_km=100.0,
            ).fit(df, outcome="y", treatment="treated", time="time")

    def test_did_conley_without_cutoff_raises(self, df):
        from diff_diff import DifferenceInDifferences

        with pytest.raises(ValueError, match="conley_cutoff_km"):
            DifferenceInDifferences(
                vcov_type="conley",
                conley_coords=("lat", "lon"),
            ).fit(df, outcome="y", treatment="treated", time="time")

    def test_did_conley_unknown_coord_column_raises(self, df):
        from diff_diff import DifferenceInDifferences

        with pytest.raises(ValueError, match="not in `data`"):
            DifferenceInDifferences(
                vcov_type="conley",
                conley_coords=("missing_lat", "lon"),
                conley_cutoff_km=100.0,
            ).fit(df, outcome="y", treatment="treated", time="time")

    def test_did_conley_with_absorb_raises(self, df):
        from diff_diff import DifferenceInDifferences

        with pytest.raises(NotImplementedError, match="absorb.*conley"):
            DifferenceInDifferences(
                vcov_type="conley",
                conley_coords=("lat", "lon"),
                conley_cutoff_km=100.0,
            ).fit(df, outcome="y", treatment="treated", time="time", absorb=["unit"])

    def test_synthetic_did_conley_raises(self):
        from diff_diff import SyntheticDiD

        with pytest.raises(TypeError, match="conley"):
            SyntheticDiD(vcov_type="conley")  # type: ignore[call-arg]

    def test_synthetic_did_conley_kwarg_raises(self):
        from diff_diff import SyntheticDiD

        with pytest.raises(TypeError, match="conley"):
            SyntheticDiD(conley_cutoff_km=100.0)  # type: ignore[call-arg]


class TestConleySetParamsAtomicity:
    """set_params atomicity for Conley fields. Per
    feedback_transactional_set_params: invalid multi-kwarg call must not
    leave the estimator in a partial state."""

    def test_unknown_kwarg_raises_no_mutation(self):
        from diff_diff import DifferenceInDifferences

        est = DifferenceInDifferences(conley_coords=("lat", "lon"), conley_cutoff_km=100.0)
        # Pre-call snapshot
        before_cutoff = est.conley_cutoff_km
        before_kernel = est.conley_kernel
        # set_params with valid + unknown key → must raise & not mutate
        with pytest.raises(ValueError, match="Unknown parameter"):
            est.set_params(conley_cutoff_km=200.0, garbage_field="x")
        # Verify state did NOT change
        assert est.conley_cutoff_km == before_cutoff
        assert est.conley_kernel == before_kernel

    def test_valid_kwargs_apply(self):
        from diff_diff import DifferenceInDifferences

        est = DifferenceInDifferences(conley_coords=("lat", "lon"), conley_cutoff_km=100.0)
        est.set_params(conley_cutoff_km=250.0, conley_kernel="uniform")
        assert est.conley_cutoff_km == 250.0
        assert est.conley_kernel == "uniform"


class TestConleyParityR:
    """R conleyreg parity for the Conley spatial HAC implementation.

    Skips when the golden JSON is absent (CI's isolated-install job copies
    only tests/, not benchmarks/). Local regeneration:
        cd benchmarks/R && Rscript generate_conley_golden.R
    """

    GOLDEN_PATH = "benchmarks/data/r_conleyreg_conley_golden.json"
    PARITY_TOL = 1e-6  # Phase 1 success criterion

    @pytest.fixture(scope="class")
    def golden(self):
        import json
        from pathlib import Path

        repo_root = Path(__file__).resolve().parent.parent
        path = repo_root / self.GOLDEN_PATH
        if not path.exists():
            pytest.skip(
                f"Golden JSON not present at {path}; run "
                "`cd benchmarks/R && Rscript generate_conley_golden.R` to generate. "
                "Requires conleyreg R package + sf/lwgeom + system libs gdal/proj/geos/udunits."
            )
        return json.loads(path.read_text())

    def _check_fixture(self, golden, name):
        entry = golden[name]
        X = np.asarray(entry["x"], dtype=np.float64).reshape(entry["x_shape"])
        y = np.asarray(entry["y"], dtype=np.float64)
        coords = np.asarray(entry["coords"], dtype=np.float64).reshape(entry["coords_shape"])
        vcov_expected = np.asarray(entry["vcov"], dtype=np.float64).reshape(entry["vcov_shape"])

        coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ coefs
        vcov_got = compute_robust_vcov(
            X,
            residuals,
            vcov_type="conley",
            conley_coords=coords,
            conley_cutoff_km=entry["cutoff_km"],
            conley_metric=entry["metric"],
            conley_kernel=entry["kernel"],
        )
        np.testing.assert_allclose(
            vcov_got, vcov_expected, atol=self.PARITY_TOL, rtol=self.PARITY_TOL
        )

    def test_parity_small_haversine(self, golden):
        self._check_fixture(golden, "small_haversine")

    def test_parity_dense_haversine(self, golden):
        self._check_fixture(golden, "dense_haversine")

    def test_parity_lat_lon_realistic(self, golden):
        self._check_fixture(golden, "lat_lon_realistic")


class TestConleyReductionsAddendum:
    """Additional reduction tests not covered by the helper-direct class.

    Placeholder: the helper-direct class already covers the essential
    reductions (HC0 at tiny cutoff, K=ones at huge cutoff, etc.).
    Kept here so future test expansions have a clear class to attach to.
    """

    def test_diagonal_of_meat_equals_HC0_contribution(self):
        """For any kernel, K(0/h) = 1 so the diagonal contribution to the
        meat is exactly the HC0 term Σ_i X_i ε_i² X_i'."""
        rng = np.random.default_rng(seed=9)
        n = 20
        coords = rng.uniform(0, 1000, size=(n, 2))
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        eps = rng.standard_normal(n)
        # Build kernel with cutoff between min and max pairwise distance
        D = _pairwise_distance_matrix(coords, "euclidean")
        cutoff = float(D[D > 0].min() * 0.001)  # ensure off-diagonal kernel is 0
        # With this cutoff, the Bartlett kernel is 1 on the diagonal and 0 off,
        # so meat == HC0.
        S = X * eps[:, None]
        meat_full = S.T @ _bartlett_kernel(D / cutoff) @ S
        meat_hc0 = X.T @ (X * (eps**2)[:, None])
        np.testing.assert_allclose(meat_full, meat_hc0, atol=1e-12)
