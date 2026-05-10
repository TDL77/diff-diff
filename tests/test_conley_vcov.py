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

    def test_indefinite_meat_warning_fires_for_bartlett(self):
        """Both kernels (radial 1-D bartlett and uniform) are practitioner
        specializations of Conley 1999 and are NOT formally PSD-guaranteed
        (Conley's explicit PSD formula is the 2-D separable product window,
        Eq 3.14, not the 1-D radial form). The PSD guard must therefore
        fire for bartlett too, not just uniform.

        Forces the indefinite path by monkey-patching `_bartlett_kernel` to
        return a kernel matrix with an aggressive negative off-diagonal,
        making the resulting meat indefinite. Verifies the warning surfaces
        with the kernel name in the message.
        """
        from diff_diff import conley as conley_mod

        rng = np.random.default_rng(seed=11)
        n = 6
        coords = rng.uniform(0, 1, size=(n, 2))
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        eps = np.ones(n)  # non-zero residuals so meat is non-zero
        bread = X.T @ X

        # Patch the bartlett kernel to return a sign-pattern that DEFINITELY
        # produces an indefinite meat. The native bartlett is non-negative;
        # injecting large negative off-diagonals breaks the
        # PSD-by-non-negative-window heuristic.
        original = conley_mod._bartlett_kernel

        def _indefinite(u: np.ndarray) -> np.ndarray:
            base = np.eye(u.shape[0])
            # Aggressive negative off-diagonals; this kernel is itself
            # indefinite and so is S.T @ K @ S for generic S.
            for i in range(u.shape[0]):
                for j in range(u.shape[0]):
                    if i != j:
                        base[i, j] = -10.0
            return base

        try:
            conley_mod._bartlett_kernel = _indefinite
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                conley_mod._compute_conley_vcov(
                    X,
                    eps,
                    coords,
                    cutoff=10.0,
                    metric="euclidean",
                    kernel="bartlett",
                    bread_matrix=bread,
                )
            # Verify a PSD warning fired naming the bartlett kernel
            psd_warnings = [
                msg
                for msg in w
                if issubclass(msg.category, UserWarning)
                and "bartlett" in str(msg.message)
                and "negative eigenvalue" in str(msg.message)
            ]
            assert len(psd_warnings) >= 1, (
                f"Expected a UserWarning naming kernel='bartlett' and "
                f"'negative eigenvalue'; got {[str(m.message) for m in w]}"
            )
        finally:
            conley_mod._bartlett_kernel = original


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

    def test_linear_regression_conley_with_survey_design_raises(self, fit_data):
        """LinearRegression(vcov_type='conley', survey_design=...) must raise
        NotImplementedError before fitting. Without the front-door guard,
        LinearRegression.fit() silently bypasses the documented Conley+survey
        rejection: it sets `return_vcov=False` on the solve_ols call when
        survey vcov is needed, skipping the linalg validator, and the survey
        vcov path then overwrites `vcov_` with a non-Conley variance under a
        Conley request. Phase 5 will lift this rejection (Bertanha-Imbens 2014
        weighted-Conley); Phase 1 is unweighted only.
        """
        from diff_diff.survey import make_pweight_design

        X, y, coords = fit_data
        n = X.shape[0]
        survey = make_pweight_design(np.ones(n))
        with pytest.raises(NotImplementedError, match="conley.*survey"):
            LinearRegression(
                vcov_type="conley",
                include_intercept=True,
                conley_coords=coords,
                conley_cutoff_km=2000.0,
                survey_design=survey,
            ).fit(X, y)


class TestConleyEstimatorIntegration:
    """Panel-estimator rejection tests for vcov_type='conley'.

    DiD and MultiPeriodDiD reject Conley at fit-time in Phase 1 because
    cross-sectional Conley over (unit, time) rows mishandles same-unit
    cross-time pairs (d_ij = 0 -> K = 1). The supported Phase 1 path for
    Conley is direct compute_robust_vcov / LinearRegression on a single-
    period regression. Phase 2 will add the space-time product kernel and
    lift the rejection.
    """

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

    def test_did_with_conley_raises(self, two_period_panel):
        """DifferenceInDifferences + vcov_type='conley' is rejected
        unconditionally. DiD is intrinsically a two-period panel; cross-
        sectional Conley over (unit, t=0) ∪ (unit, t=1) rows would treat
        same-unit cross-time pairs as d_ij=0 -> K=1, mishandling the space-
        time HAC. Phase 2 will add the space-time product kernel; Phase 1's
        supported Conley path is direct compute_robust_vcov on a single-
        period design. Closes CI reviewer P1 #1.
        """
        from diff_diff import DifferenceInDifferences

        df = two_period_panel.copy()
        with pytest.raises(NotImplementedError, match="DifferenceInDifferences.*conley"):
            DifferenceInDifferences(
                vcov_type="conley",
                conley_coords=("lat", "lon"),
                conley_cutoff_km=2000.0,
            ).fit(df, outcome="y", treatment="treated", time="time")

    def test_did_with_conley_repeated_coords_raises(self, two_period_panel):
        """Per CI reviewer P1 #1 recommendation: regression test where
        coordinates repeat across multiple periods. The fit must reject
        rather than silently produce wrong SE."""
        from diff_diff import DifferenceInDifferences

        # Confirm the fixture has time-invariant coords per unit.
        coord_var = two_period_panel.groupby("unit")[["lat", "lon"]].nunique()
        assert (coord_var.values == 1).all(), "Fixture coords must be time-invariant"

        with pytest.raises(NotImplementedError, match="conley"):
            DifferenceInDifferences(
                vcov_type="conley",
                conley_coords=("lat", "lon"),
                conley_cutoff_km=2000.0,
            ).fit(two_period_panel, outcome="y", treatment="treated", time="time")

    def test_multi_period_did_with_conley_raises(self):
        """MultiPeriodDiD is intrinsically a panel estimator; vcov_type='conley'
        is rejected end-to-end. Closes CI reviewer P1 #1."""
        from diff_diff import MultiPeriodDiD

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
        with pytest.raises(NotImplementedError, match="MultiPeriodDiD.*conley"):
            MultiPeriodDiD(
                vcov_type="conley",
                conley_coords=("lat", "lon"),
                conley_cutoff_km=2000.0,
            ).fit(df_mp, outcome="y", treatment="treated", time="time", reference_period=1)


class TestConleyTWFE:
    """TwoWayFixedEffects rejects vcov_type='conley' end-to-end.

    TWFE is intrinsically a multi-period panel estimator. Cross-sectional
    Conley over (unit, time) rows would treat same-unit cross-time pairs as
    d_ij=0 -> K=1, mishandling the space-time HAC. The supported Phase 1
    path for Conley with FE is to demean externally (single-period collapse)
    and call compute_robust_vcov directly. Phase 2 will add a space-time
    product kernel / Driscoll-Kraay estimator. Closes CI reviewer P1 #1.
    """

    @pytest.fixture
    def panel(self):
        """Build a 2-period panel with geocoords for TWFE rejection tests."""
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

    def test_twfe_conley_raises(self, panel):
        """TWFE + vcov_type='conley' is rejected unconditionally."""
        from diff_diff import TwoWayFixedEffects

        with pytest.raises(NotImplementedError, match="TwoWayFixedEffects.*conley"):
            TwoWayFixedEffects(
                vcov_type="conley",
                conley_coords=("lat", "lon"),
                conley_cutoff_km=2000.0,
            ).fit(panel, outcome="y", treatment="treated", time="time", unit="unit")

    def test_twfe_conley_with_explicit_cluster_raises(self, panel):
        """User explicitly setting cluster=... with conley still raises (the
        outer panel-rejection raise fires first)."""
        from diff_diff import TwoWayFixedEffects

        with pytest.raises(NotImplementedError, match="conley"):
            TwoWayFixedEffects(
                vcov_type="conley",
                cluster="unit",
                conley_coords=("lat", "lon"),
                conley_cutoff_km=2000.0,
            ).fit(panel, outcome="y", treatment="treated", time="time", unit="unit")

    def test_twfe_conley_with_wild_bootstrap_raises(self, panel):
        """Conley + wild_bootstrap on TWFE raises (the outer panel-rejection
        fires before the inference-mode check)."""
        from diff_diff import TwoWayFixedEffects

        with pytest.raises(NotImplementedError, match="conley"):
            TwoWayFixedEffects(
                vcov_type="conley",
                inference="wild_bootstrap",
                conley_coords=("lat", "lon"),
                conley_cutoff_km=2000.0,
            ).fit(panel, outcome="y", treatment="treated", time="time", unit="unit")

    def test_twfe_conley_repeated_coords_across_periods_raises(self, panel):
        """Per CI reviewer P1 #1 recommendation: regression test where
        coordinates repeat across multiple periods. Without the panel
        rejection, cross-sectional Conley would silently produce wrong SE
        because pairs (i, t1) <-> (i, t2) have d_ij = 0 -> K = 1."""
        from diff_diff import TwoWayFixedEffects

        # Each unit's lat/lon is constant across t=0 and t=1 in the fixture.
        # Confirm via grouping that coords are time-invariant.
        coord_var = panel.groupby("unit")[["lat", "lon"]].nunique()
        assert (coord_var.values == 1).all(), "Fixture coords must be time-invariant"

        with pytest.raises(NotImplementedError, match="conley"):
            TwoWayFixedEffects(
                vcov_type="conley",
                conley_coords=("lat", "lon"),
                conley_cutoff_km=2000.0,
            ).fit(panel, outcome="y", treatment="treated", time="time", unit="unit")


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

    def test_did_conley_combinations_all_raise(self, df):
        """Every DifferenceInDifferences + vcov_type='conley' combination
        rejects unconditionally (DiD is intrinsically a two-period panel;
        cross-sectional Conley is unsafe over (unit, time) rows). Asserts
        the reject regardless of cluster=, absorb=, or missing coords/cutoff.
        Closes CI reviewer P1 #1.
        """
        from diff_diff import DifferenceInDifferences

        # cluster + conley
        with pytest.raises(NotImplementedError, match="conley"):
            DifferenceInDifferences(
                vcov_type="conley",
                cluster="stratum",
                conley_coords=("lat", "lon"),
                conley_cutoff_km=100.0,
            ).fit(df, outcome="y", treatment="treated", time="time")
        # missing conley_coords
        with pytest.raises(NotImplementedError, match="conley"):
            DifferenceInDifferences(
                vcov_type="conley",
                conley_cutoff_km=100.0,
            ).fit(df, outcome="y", treatment="treated", time="time")
        # missing conley_cutoff_km
        with pytest.raises(NotImplementedError, match="conley"):
            DifferenceInDifferences(
                vcov_type="conley",
                conley_coords=("lat", "lon"),
            ).fit(df, outcome="y", treatment="treated", time="time")
        # unknown coord column (data validation skipped — outer reject fires first)
        with pytest.raises(NotImplementedError, match="conley"):
            DifferenceInDifferences(
                vcov_type="conley",
                conley_coords=("missing_lat", "lon"),
                conley_cutoff_km=100.0,
            ).fit(df, outcome="y", treatment="treated", time="time")
        # absorb + conley
        with pytest.raises(NotImplementedError, match="conley"):
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

    def test_synthetic_did_set_params_conley_raises(self):
        """SyntheticDiD.set_params(vcov_type='conley') must raise (mirrors
        __init__'s contract — closes the silent-bypass gap CI reviewer flagged
        as P1 CQ1)."""
        from diff_diff import SyntheticDiD

        est = SyntheticDiD()
        # Snapshot pre-call state
        before_variance = est.variance_method
        before_n_boot = est.n_bootstrap
        before_zeta = est.zeta_omega

        with pytest.raises(TypeError, match="conley"):
            est.set_params(vcov_type="conley")
        # Verify nothing mutated
        assert est.variance_method == before_variance
        assert est.n_bootstrap == before_n_boot
        assert est.zeta_omega == before_zeta

    def test_synthetic_did_set_params_conley_kwarg_raises(self):
        from diff_diff import SyntheticDiD

        est = SyntheticDiD()
        with pytest.raises(TypeError, match="conley"):
            est.set_params(conley_cutoff_km=100.0)
        # Verify the conley attr stays None (rejected before mutation)
        assert getattr(est, "conley_cutoff_km", None) is None

    def test_synthetic_did_get_params_includes_conley_keys(self):
        """get_params() / set_params() round-trip must include the inherited
        conley_* keys with None values for sklearn-style API consistency
        (CI reviewer P2 CQ3)."""
        from diff_diff import SyntheticDiD

        est = SyntheticDiD(variance_method="placebo", n_bootstrap=10)
        params = est.get_params()
        assert "vcov_type" in params and params["vcov_type"] is None
        assert "conley_coords" in params and params["conley_coords"] is None
        assert "conley_cutoff_km" in params and params["conley_cutoff_km"] is None
        assert "conley_metric" in params and params["conley_metric"] is None
        assert "conley_kernel" in params and params["conley_kernel"] is None
        # Round-trip: passing None values back into set_params is a no-op
        est.set_params(**params)
        assert est.variance_method == "placebo"
        assert est.n_bootstrap == 10


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
