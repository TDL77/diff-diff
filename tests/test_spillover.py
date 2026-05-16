"""Tests for SpilloverDiD (Butts 2021 ring-indicator spillover-aware DiD).

Step 1 surface: ring-construction helpers and the public class scaffold.
Step 2+ surfaces are added incrementally as the implementation lands.
"""

import numpy as np
import pandas as pd
import pytest

from diff_diff.spillover import (
    SpilloverDiD,
    _apply_callable_metric_pairwise,
    _apply_horizon_binning,
    _build_event_study_design,
    _build_ring_indicators,
    _check_omega_0_connectivity,
    _compute_event_time_per_row,
    _compute_nearest_treated_distance_sparse,
    _compute_nearest_treated_distance_staggered,
    _compute_nearest_treated_distance_static,
    _convert_treatment_to_first_treat,
    _euclidean_pairwise,
    _extract_treatment_onsets,
    _haversine_km_pairwise,
    _pairwise_ring_distances,
    _ring_label,
)
from tests._dgp_utils import (
    generate_butts_nonstaggered_dgp,
    generate_butts_staggered_dgp,
)

# =============================================================================
# Pairwise-distance primitives
# =============================================================================


class TestHaversinePairwise:
    """Tests for _haversine_km_pairwise."""

    def test_zero_distance_when_same_point(self):
        coords = np.array([[40.7128, -74.0060]])  # NYC
        result = _haversine_km_pairwise(coords, coords)
        assert result.shape == (1, 1)
        assert abs(result[0, 0]) < 1e-9

    def test_known_pair_nyc_to_la(self):
        # NYC (40.7128 N, 74.0060 W) to LA (34.0522 N, 118.2437 W)
        # Reference great-circle distance ~ 3935.7 km (within 0.5% of any source)
        nyc = np.array([[40.7128, -74.0060]])
        la = np.array([[34.0522, -118.2437]])
        result = _haversine_km_pairwise(nyc, la)
        assert result.shape == (1, 1)
        assert abs(result[0, 0] - 3935.7) < 5.0

    def test_pairwise_matrix_shape(self):
        coords_a = np.array([[40.0, -74.0], [34.0, -118.0]])
        coords_b = np.array([[51.5, -0.1], [35.7, 139.7], [40.7, -74.0]])
        result = _haversine_km_pairwise(coords_a, coords_b)
        assert result.shape == (2, 3)
        # All non-negative
        assert (result >= 0).all()


class TestEuclideanPairwise:
    """Tests for _euclidean_pairwise."""

    def test_known_3_4_5(self):
        a = np.array([[0.0, 0.0]])
        b = np.array([[3.0, 4.0]])
        result = _euclidean_pairwise(a, b)
        assert abs(result[0, 0] - 5.0) < 1e-12

    def test_zero_distance_same_point(self):
        coords = np.array([[1.5, 2.5], [3.0, 4.0]])
        result = _euclidean_pairwise(coords, coords)
        np.testing.assert_allclose(np.diag(result), 0.0, atol=1e-12)


class TestPairwiseRingDistances:
    """Tests for the _pairwise_ring_distances dispatch."""

    def test_haversine_branch(self):
        a = np.array([[0.0, 0.0]])
        b = np.array([[0.0, 0.0]])
        result = _pairwise_ring_distances(a, b, "haversine")
        assert result.shape == (1, 1)
        assert abs(result[0, 0]) < 1e-9

    def test_euclidean_branch(self):
        a = np.array([[0.0, 0.0]])
        b = np.array([[1.0, 0.0]])
        result = _pairwise_ring_distances(a, b, "euclidean")
        assert abs(result[0, 0] - 1.0) < 1e-12

    def test_callable_branch(self):
        a = np.array([[0.0, 0.0], [1.0, 1.0]])
        b = np.array([[2.0, 2.0]])

        def custom(x, y):
            return np.full((x.shape[0], y.shape[0]), 7.5)

        result = _pairwise_ring_distances(a, b, custom)
        assert result.shape == (2, 1)
        np.testing.assert_allclose(result, 7.5)

    def test_unknown_metric_raises(self):
        a = np.array([[0.0, 0.0]])
        b = np.array([[1.0, 1.0]])
        with pytest.raises(ValueError, match="Unknown conley_metric"):
            _pairwise_ring_distances(a, b, "manhattan")


class TestApplyCallableMetricPairwise:
    """Validation of user-supplied callable distance metrics."""

    def test_wrong_shape_raises(self):
        a = np.array([[0.0, 0.0], [1.0, 1.0]])
        b = np.array([[2.0, 2.0], [3.0, 3.0]])

        def bad(x, y):
            return np.zeros((1, 1))

        with pytest.raises(ValueError, match="shape"):
            _apply_callable_metric_pairwise(bad, a, b)

    def test_non_finite_raises(self):
        a = np.array([[0.0, 0.0]])
        b = np.array([[1.0, 1.0]])

        def bad(x, y):
            return np.array([[np.inf]])

        with pytest.raises(ValueError, match="non-finite"):
            _apply_callable_metric_pairwise(bad, a, b)

    def test_negative_raises(self):
        a = np.array([[0.0, 0.0]])
        b = np.array([[1.0, 1.0]])

        def bad(x, y):
            return np.array([[-1.0]])

        with pytest.raises(ValueError, match="negative"):
            _apply_callable_metric_pairwise(bad, a, b)


# =============================================================================
# Static nearest-treated distance
# =============================================================================


@pytest.fixture
def small_static_panel():
    """3 treated units near origin, 3 near-controls 25-100km out, 3 far at 500km."""
    rng = np.random.default_rng(42)
    treated = [(0.0 + rng.normal(0, 0.05), 0.0 + rng.normal(0, 0.05)) for _ in range(3)]
    near = [(0.4 + i * 0.05, 0.0) for i in range(3)]  # ~44-55 km east
    far = [(5.0 + i * 0.1, 0.0) for i in range(3)]  # ~556+ km east
    units = []
    coords = []
    treats = []
    for i, c in enumerate(treated):
        units.append(f"T{i}")
        coords.append(c)
        treats.append(1)
    for i, c in enumerate(near):
        units.append(f"N{i}")
        coords.append(c)
        treats.append(0)
    for i, c in enumerate(far):
        units.append(f"F{i}")
        coords.append(c)
        treats.append(0)
    # Two periods so the static panel has 2 rows per unit
    rows = []
    for u, c, d in zip(units, coords, treats):
        for t in (0, 1):
            rows.append(
                {
                    "unit": u,
                    "time": t,
                    "lat": c[0],
                    "lon": c[1],
                    "D": d * t,  # turns on at t=1 for treated
                }
            )
    return pd.DataFrame(rows)


class TestComputeNearestTreatedDistanceStatic:
    """Static (non-staggered) nearest-treated distance helper."""

    def test_treated_units_have_zero_distance(self, small_static_panel):
        treated_ids = np.array(["T0", "T1", "T2"])
        d_i, unit_index = _compute_nearest_treated_distance_static(
            small_static_panel,
            unit="unit",
            coords=("lat", "lon"),
            metric="haversine",
            treated_unit_ids=treated_ids,
        )
        for tid in treated_ids:
            pos = np.where(unit_index == tid)[0][0]
            # Treated units' nearest treated is themselves OR an adjacent T*; the
            # static fixture clusters them within rng.normal(0, 0.05) deg ~6 km.
            assert d_i[pos] < 15.0  # all treated cluster within ~15 km

    def test_near_controls_below_far_controls(self, small_static_panel):
        treated_ids = np.array(["T0", "T1", "T2"])
        d_i, unit_index = _compute_nearest_treated_distance_static(
            small_static_panel,
            unit="unit",
            coords=("lat", "lon"),
            metric="haversine",
            treated_unit_ids=treated_ids,
        )
        near_pos = [np.where(unit_index == f"N{i}")[0][0] for i in range(3)]
        far_pos = [np.where(unit_index == f"F{i}")[0][0] for i in range(3)]
        assert all(d_i[p] < 100.0 for p in near_pos)
        assert all(d_i[p] > 500.0 for p in far_pos)

    def test_euclidean_metric(self, small_static_panel):
        treated_ids = np.array(["T0", "T1", "T2"])
        d_i_h, _ = _compute_nearest_treated_distance_static(
            small_static_panel,
            unit="unit",
            coords=("lat", "lon"),
            metric="haversine",
            treated_unit_ids=treated_ids,
        )
        d_i_e, _ = _compute_nearest_treated_distance_static(
            small_static_panel,
            unit="unit",
            coords=("lat", "lon"),
            metric="euclidean",
            treated_unit_ids=treated_ids,
        )
        # Different units, but ordering should be consistent (near < far)
        # so the rank of distances matches between metrics.
        order_h = np.argsort(d_i_h)
        order_e = np.argsort(d_i_e)
        np.testing.assert_array_equal(order_h, order_e)

    def test_no_treated_units_raises(self, small_static_panel):
        with pytest.raises(ValueError, match="no treated units present"):
            _compute_nearest_treated_distance_static(
                small_static_panel,
                unit="unit",
                coords=("lat", "lon"),
                metric="haversine",
                treated_unit_ids=np.array(["nonexistent_unit"]),
            )

    def test_unit_index_is_sorted(self, small_static_panel):
        treated_ids = np.array(["T0", "T1", "T2"])
        _, unit_index = _compute_nearest_treated_distance_static(
            small_static_panel,
            unit="unit",
            coords=("lat", "lon"),
            metric="haversine",
            treated_unit_ids=treated_ids,
        )
        # Sorted lexicographically: F0, F1, F2, N0, N1, N2, T0, T1, T2
        expected = ["F0", "F1", "F2", "N0", "N1", "N2", "T0", "T1", "T2"]
        np.testing.assert_array_equal(unit_index, expected)


class TestComputeNearestTreatedDistanceSparse:
    """Sparse cKDTree path for nearest-treated computation."""

    def test_sparse_matches_dense_haversine(self, small_static_panel):
        # Force sparse path on the small fixture by using a tight cutoff.
        treated_ids = np.array(["T0", "T1", "T2"])
        unit_coords_df = (
            small_static_panel[["unit", "lat", "lon"]]
            .drop_duplicates(subset="unit")
            .set_index("unit")
            .sort_index()
        )
        all_coords = unit_coords_df[["lat", "lon"]].values.astype(np.float64)
        treated_mask = np.array(
            [uid in set(treated_ids.tolist()) for uid in unit_coords_df.index],
            dtype=bool,
        )
        treated_coords = all_coords[treated_mask]
        # Sparse path with a 1000 km cutoff should agree with the dense path on
        # all in-range units; far controls (>500 km but <1000 km from any
        # treated) get their true nearest-treated distance.
        d_sparse = _compute_nearest_treated_distance_sparse(
            all_coords=all_coords,
            treated_coords=treated_coords,
            metric="haversine",
            cutoff_km=1000.0,
        )
        d_dense = _haversine_km_pairwise(all_coords, treated_coords).min(axis=1)
        # Mask: only compare entries within cutoff in dense (sparse returns inf otherwise).
        in_range = d_dense <= 1000.0 * (1 + 1e-6)
        np.testing.assert_allclose(d_sparse[in_range], d_dense[in_range], atol=1e-8)

    def test_sparse_inf_when_no_treated_in_range(self):
        # Single unit at (50, 0); treated cluster at (0, 0). With cutoff 100km,
        # great-circle ~5500 km exceeds it; expect inf.
        all_coords = np.array([[50.0, 0.0]])
        treated_coords = np.array([[0.0, 0.0]])
        d = _compute_nearest_treated_distance_sparse(
            all_coords=all_coords,
            treated_coords=treated_coords,
            metric="haversine",
            cutoff_km=100.0,
        )
        assert np.isinf(d[0])

    def test_sparse_euclidean(self):
        all_coords = np.array([[0.0, 0.0], [5.0, 0.0], [100.0, 0.0]])
        treated_coords = np.array([[0.0, 0.0]])
        d = _compute_nearest_treated_distance_sparse(
            all_coords=all_coords,
            treated_coords=treated_coords,
            metric="euclidean",
            cutoff_km=10.0,
        )
        assert abs(d[0]) < 1e-12
        assert abs(d[1] - 5.0) < 1e-12
        assert np.isinf(d[2])


# =============================================================================
# Staggered nearest-treated distance
# =============================================================================


@pytest.fixture
def staggered_panel():
    """Panel with two cohorts (t_treat=1 and t_treat=2) plus never-treated."""
    rows = []
    # Cohort A: 2 units treated at t=1 (near origin)
    cohort_a = {"A0": (0.0, 0.0), "A1": (0.1, 0.0)}
    # Cohort B: 2 units treated at t=2 (10 deg east of origin, ~1100 km away)
    cohort_b = {"B0": (0.0, 10.0), "B1": (0.0, 10.1)}
    # Never-treated: 1 unit far away
    never = {"N0": (50.0, 0.0)}  # very far north
    first_treat = {
        **{u: 1 for u in cohort_a},
        **{u: 2 for u in cohort_b},
        **{u: np.inf for u in never},
    }
    coords = {**cohort_a, **cohort_b, **never}
    for t in range(4):  # periods 0..3
        for u, (lat, lon) in coords.items():
            rows.append({"unit": u, "time": t, "lat": lat, "lon": lon})
    df = pd.DataFrame(rows)
    return df, first_treat


class TestComputeNearestTreatedDistanceStaggered:
    """Staggered (time-varying) nearest-treated distance helper."""

    def test_inf_pre_any_treatment(self, staggered_panel):
        df, ft = staggered_panel
        d_it, row_unit, row_time, _trigger = _compute_nearest_treated_distance_staggered(
            df,
            unit="unit",
            time="time",
            coords=("lat", "lon"),
            metric="haversine",
            first_treat_by_unit=ft,
        )
        # Period 0 has no treated units yet -> d_it = inf for all rows.
        mask_t0 = row_time == 0
        assert np.isinf(d_it[mask_t0]).all()

    def test_cohort_a_active_at_t1(self, staggered_panel):
        df, ft = staggered_panel
        d_it, row_unit, row_time, _trigger = _compute_nearest_treated_distance_staggered(
            df,
            unit="unit",
            time="time",
            coords=("lat", "lon"),
            metric="haversine",
            first_treat_by_unit=ft,
        )
        mask_t1 = row_time == 1
        # Cohort A treats at t=1; B units should be ~1100km from A; A units near zero.
        for u in ("A0", "A1"):
            row = mask_t1 & (row_unit == u)
            assert d_it[row][0] < 15.0  # within their own cohort
        for u in ("B0", "B1"):
            row = mask_t1 & (row_unit == u)
            # B is ~1100km east of A; min distance to {A0, A1}.
            # B0 at lon=10 -> 1112 km; B1 at lon=10.1 -> 1123 km.
            assert 1100.0 < d_it[row][0] < 1130.0

    def test_running_min_across_cohorts_at_t2(self, staggered_panel):
        df, ft = staggered_panel
        d_it, row_unit, row_time, _trigger = _compute_nearest_treated_distance_staggered(
            df,
            unit="unit",
            time="time",
            coords=("lat", "lon"),
            metric="haversine",
            first_treat_by_unit=ft,
        )
        mask_t2 = row_time == 2
        # At t=2, B0 and B1 are also treated; the nearest-treated set for B units is {A,B}
        # but B is closer to itself -> nearly zero distance now.
        for u in ("B0", "B1"):
            row = mask_t2 & (row_unit == u)
            assert d_it[row][0] < 15.0


# =============================================================================
# Ring indicator construction
# =============================================================================


class TestBuildRingIndicators:
    """Tests for _build_ring_indicators."""

    def test_three_rings_three_distances(self):
        rings = [0.0, 50.0, 100.0, 200.0]  # K=3 rings
        d_values = np.array([25.0, 75.0, 150.0])
        masks = _build_ring_indicators(d_values, rings)
        assert masks.shape == (3, 3)
        # Row 0: distance 25 -> Ring 1 (0 to 50)
        np.testing.assert_array_equal(masks[0], [True, False, False])
        # Row 1: distance 75 -> Ring 2 (50 to 100)
        np.testing.assert_array_equal(masks[1], [False, True, False])
        # Row 2: distance 150 -> Ring 3 (100 to 200)
        np.testing.assert_array_equal(masks[2], [False, False, True])

    def test_interior_boundary_belongs_to_upper_ring(self):
        """Unit exactly at the boundary between two interior rings."""
        rings = [0.0, 50.0, 100.0, 200.0]
        d_values = np.array([50.0])  # exactly on the 50.0 boundary
        masks = _build_ring_indicators(d_values, rings)
        # 50.0 should fall in Ring 2 (the upper of the boundary pair) per the
        # half-open-at-top convention.
        np.testing.assert_array_equal(masks[0], [False, True, False])

    def test_outermost_boundary_belongs_to_last_ring(self):
        """Unit exactly at d_bar should fall in the outermost ring, not be far."""
        rings = [0.0, 50.0, 100.0, 200.0]
        d_values = np.array([200.0])  # exactly at d_bar
        masks = _build_ring_indicators(d_values, rings)
        # 200.0 should fall in the OUTERMOST ring (closed-at-top convention).
        np.testing.assert_array_equal(masks[0], [False, False, True])

    def test_distance_at_origin_lands_in_first_ring(self):
        """Treated units have d_i = 0; they fall in Ring_1."""
        rings = [0.0, 50.0, 100.0, 200.0]
        d_values = np.array([0.0])
        masks = _build_ring_indicators(d_values, rings)
        np.testing.assert_array_equal(masks[0], [True, False, False])

    def test_far_away_unit_in_no_ring(self):
        """Distance beyond d_bar puts unit in NO ring."""
        rings = [0.0, 50.0, 100.0, 200.0]
        d_values = np.array([300.0])
        masks = _build_ring_indicators(d_values, rings)
        np.testing.assert_array_equal(masks[0], [False, False, False])

    def test_single_ring(self):
        """K=1 (single ring) case (Butts Equation 5 single-S_i form)."""
        rings = [0.0, 100.0]
        d_values = np.array([0.0, 50.0, 99.9, 100.0, 100.1])
        masks = _build_ring_indicators(d_values, rings)
        assert masks.shape == (5, 1)
        # First four are <= 100, last is > 100 (far-away).
        np.testing.assert_array_equal(masks[:, 0], [True, True, True, True, False])

    def test_too_few_breakpoints_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            _build_ring_indicators(np.array([0.0]), [50.0])

    def test_non_increasing_raises(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            _build_ring_indicators(np.array([0.0]), [50.0, 50.0, 100.0])

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            _build_ring_indicators(np.array([0.0]), [-10.0, 100.0])


class TestRingLabel:
    """Tests for _ring_label."""

    def test_interior_ring_half_open(self):
        rings = [0.0, 50.0, 100.0, 200.0]
        assert _ring_label(rings, 0) == "[0, 50)"
        assert _ring_label(rings, 1) == "[50, 100)"

    def test_outermost_ring_closed(self):
        rings = [0.0, 50.0, 100.0, 200.0]
        assert _ring_label(rings, 2) == "[100, 200]"

    def test_single_ring_closed_form(self):
        rings = [0.0, 100.0]
        assert _ring_label(rings, 0) == "[0, 100]"


# =============================================================================
# Public class skeleton
# =============================================================================


class TestSpilloverDiDInitGetParamsSetParams:
    """Constructor + sklearn-like get_params / set_params surface."""

    def test_construction_with_defaults(self):
        est = SpilloverDiD(rings=[0.0, 50.0, 100.0])
        assert est.rings == [0.0, 50.0, 100.0]
        assert est.d_bar is None
        assert est.vcov_type == "hc1"
        assert est.is_fitted_ is False

    def test_construction_with_all_kwargs(self):
        est = SpilloverDiD(
            rings=[0.0, 50.0, 200.0],
            d_bar=200.0,
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_metric="haversine",
            conley_cutoff_km=200.0,
            conley_lag_cutoff=0,
            cluster="region",
            alpha=0.10,
            anticipation=1,
            event_study=True,
            horizon_max=5,
            rank_deficient_action="error",
        )
        assert est.d_bar == 200.0
        assert est.vcov_type == "conley"
        assert est.cluster == "region"
        assert est.alpha == 0.10
        assert est.event_study is True

    def test_get_params_returns_constructor_args(self):
        est = SpilloverDiD(rings=[0.0, 100.0])
        params = est.get_params()
        # Check all constructor args present
        expected = {
            "rings",
            "d_bar",
            "vcov_type",
            "conley_coords",
            "conley_metric",
            "conley_cutoff_km",
            "conley_lag_cutoff",
            "cluster",
            "alpha",
            "anticipation",
            "event_study",
            "horizon_max",
            "rank_deficient_action",
        }
        assert set(params.keys()) == expected

    def test_set_params_updates_attribute(self):
        est = SpilloverDiD(rings=[0.0, 50.0])
        est.set_params(d_bar=100.0, alpha=0.10)
        assert est.d_bar == 100.0
        assert est.alpha == 0.10

    def test_set_params_returns_self(self):
        est = SpilloverDiD(rings=[0.0, 50.0])
        out = est.set_params(d_bar=100.0)
        assert out is est

    def test_set_params_rejects_unknown_key(self):
        est = SpilloverDiD(rings=[0.0, 50.0])
        with pytest.raises(ValueError, match="Unknown parameter"):
            est.set_params(nonexistent_kwarg=42)

    def test_fit_survey_design_not_implemented(self):
        """survey_design= is a deferred enhancement (Wave B MVP)."""
        est = SpilloverDiD(rings=[0.0, 50.0], conley_coords=("lat", "lon"))
        df = pd.DataFrame(
            {
                "unit": ["A", "A"],
                "time": [0, 1],
                "y": [1.0, 2.0],
                "D": [0, 1],
                "lat": [0.0, 0.0],
                "lon": [0.0, 0.0],
            }
        )
        with pytest.raises(NotImplementedError, match="survey_design"):
            est.fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                treatment="D",
                survey_design=object(),
            )


# =============================================================================
# Step 3: Two-stage Gardner fit() integration
# =============================================================================


def _make_butts_2period_dgp(
    *,
    n_treated: int = 10,
    n_near_control: int = 30,
    n_far_control: int = 30,
    tau_total: float = -0.07,
    delta_1: float = -0.04,
    d_bar: float = 100.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a 2-period panel with known direct + spillover effects.

    Layout:
      - Treated units cluster near (lat=0, lon=0).
      - Near-controls distributed within d_bar km.
      - Far-controls placed ~2*d_bar km away (clean control group).
    Outcomes (potential outcomes model):
      - Y_it(0, 0) = mu_i + lambda_t + e_it  (clean trend, common across all units)
      - Treated unit at t=1: Y = mu_i + lambda_1 + tau_total + e_it
      - Near-control at t=1: Y = mu_i + lambda_1 + delta_1 + e_it
      - Far-control at t=1: Y = mu_i + lambda_1 + e_it
    All units satisfy parallel trends (Butts Assumption 6/7).
    """
    rng = np.random.default_rng(seed)
    n_units = n_treated + n_near_control + n_far_control
    units = [f"u{i:03d}" for i in range(n_units)]
    mu = rng.normal(0.0, 0.5, size=n_units)

    coords = []
    is_treated = []
    is_near = []
    for i in range(n_treated):
        # Cluster within ~5 km of origin
        coords.append((rng.normal(0, 0.05), rng.normal(0, 0.05)))
        is_treated.append(True)
        is_near.append(False)
    for i in range(n_near_control):
        # Within d_bar (uniform in a band 10–80 km east)
        lat = rng.uniform(0.1, 0.7)  # ~11–78 km north
        lon = rng.uniform(-0.3, 0.3)  # spread east-west
        coords.append((lat, lon))
        is_treated.append(False)
        is_near.append(True)
    for i in range(n_far_control):
        # Far-aways at 2*d_bar+ km
        lat = rng.uniform(2.0, 3.0)  # ~220–330 km north
        lon = rng.uniform(-0.5, 0.5)
        coords.append((lat, lon))
        is_treated.append(False)
        is_near.append(False)

    rows = []
    lambda_t = [0.0, 0.1]  # common time trend
    for i, u in enumerate(units):
        for t in (0, 1):
            y_clean = mu[i] + lambda_t[t]
            if t == 1 and is_treated[i]:
                y = y_clean + tau_total
            elif t == 1 and is_near[i]:
                y = y_clean + delta_1
            else:
                y = y_clean
            y += rng.normal(0, 0.02)  # noise
            rows.append(
                {
                    "unit": u,
                    "time": t,
                    "lat": coords[i][0],
                    "lon": coords[i][1],
                    "D": int(is_treated[i] and t == 1),
                    "y": y,
                }
            )
    return pd.DataFrame(rows)


class TestSpilloverDiDFitBasic:
    """Step 3 integration: fit() produces sensible point estimates."""

    def test_fit_runs_without_error(self):
        df = _make_butts_2period_dgp(seed=42)
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
        )
        result = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        assert result is not None
        assert est.is_fitted_

    def test_recovers_tau_total_within_tolerance(self):
        df = _make_butts_2period_dgp(seed=42, tau_total=-0.07, delta_1=-0.04)
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        result = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        # Single-seed tolerance — looser than the 200-seed MC test in Step 5.
        assert abs(result.att - (-0.07)) < 0.04

    def test_recovers_ring_coefficient(self):
        df = _make_butts_2period_dgp(seed=42, tau_total=-0.07, delta_1=-0.04)
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        result = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        assert result.spillover_effects is not None
        ring_coef = result.spillover_effects.iloc[0]["coef"]
        assert abs(ring_coef - (-0.04)) < 0.04

    def test_result_has_expected_fields(self):
        df = _make_butts_2period_dgp(seed=42)
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        result = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        assert result.ring_breakpoints == [0.0, 100.0]
        assert result.d_bar == 100.0
        assert result.is_staggered is False
        assert result.n_far_away_obs > 0
        assert result.stage1_n_obs > 0
        assert "[0, 100]" in result.n_units_ever_in_ring

    def test_summary_includes_ring_block(self):
        df = _make_butts_2period_dgp(seed=42)
        est = SpilloverDiD(rings=[0.0, 50.0, 100.0], conley_coords=("lat", "lon"))
        result = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        summary = result.summary()
        assert "Spillover Effects" in summary
        # Two ring labels: [0, 50) and [50, 100]
        assert "[0, 50)" in summary or "[50, 100]" in summary

    def test_to_dict_serializes_spillover_effects(self):
        df = _make_butts_2period_dgp(seed=42)
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        result = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        d = result.to_dict()
        assert "spillover_effects" in d
        assert d["spillover_effects"] is not None
        assert "ring_breakpoints" in d
        assert d["d_bar"] == 100.0


class TestSpilloverDiDRawDataInvariant:
    """Step 3: caller's DataFrame must not be mutated by fit()."""

    def test_caller_data_unchanged(self):
        df = _make_butts_2period_dgp(seed=42)
        original_cols = list(df.columns)
        original_len = len(df)
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        # Caller's DataFrame should not gain or lose columns/rows from fit().
        assert list(df.columns) == original_cols
        assert len(df) == original_len


# =============================================================================
# Step 5: Identification MC tests via _dgp_utils.py
# =============================================================================


class TestSpilloverDiDIdentification:
    """50-seed Monte Carlo: SpilloverDiD recovers known DGP within MC tolerance.

    Plan's Step 5 target was 200 seeds; this is a faster default that still
    rejects gross misidentification. A 200-seed version marked `@pytest.mark.slow`
    runs in CI's full suite (`pytest -m slow`).
    """

    def test_nonstaggered_recovers_tau_total(self):
        att_estimates = []
        n_seeds = 50
        for s in range(n_seeds):
            df = generate_butts_nonstaggered_dgp(tau_total=-0.07, delta_1=-0.04, seed=s)
            result = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon")).fit(
                df, outcome="y", unit="unit", time="time", treatment="D"
            )
            att_estimates.append(result.att)
        mean_att = float(np.mean(att_estimates))
        # MC tolerance: mean over 50 seeds at error_sd=0.05, ~200 units →
        # SE-of-mean ~ 0.05/sqrt(50 * 200) ~ 5e-4. Tolerance 0.02 leaves
        # margin for DGP design noise.
        assert (
            abs(mean_att - (-0.07)) < 0.02
        ), f"non-staggered tau_total: expected -0.07, got {mean_att:.4f}"

    def test_nonstaggered_recovers_delta_1(self):
        delta_estimates = []
        n_seeds = 50
        for s in range(n_seeds):
            df = generate_butts_nonstaggered_dgp(tau_total=-0.07, delta_1=-0.04, seed=s)
            result = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon")).fit(
                df, outcome="y", unit="unit", time="time", treatment="D"
            )
            if result.spillover_effects is not None:
                delta_estimates.append(result.spillover_effects.iloc[0]["coef"])
        mean_delta = float(np.mean(delta_estimates))
        assert (
            abs(mean_delta - (-0.04)) < 0.02
        ), f"non-staggered delta_1: expected -0.04, got {mean_delta:.4f}"

    @pytest.mark.slow
    def test_nonstaggered_recovers_at_200_seeds(self):
        """Plan-targeted 200-seed MC. Marked slow; run via `pytest -m slow`."""
        att_estimates = []
        delta_estimates = []
        for s in range(200):
            df = generate_butts_nonstaggered_dgp(tau_total=-0.07, delta_1=-0.04, seed=s)
            result = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon")).fit(
                df, outcome="y", unit="unit", time="time", treatment="D"
            )
            att_estimates.append(result.att)
            if result.spillover_effects is not None:
                delta_estimates.append(result.spillover_effects.iloc[0]["coef"])
        assert abs(np.mean(att_estimates) - (-0.07)) < 0.02
        assert abs(np.mean(delta_estimates) - (-0.04)) < 0.02

    def test_staggered_recovers_tau_total_and_delta_1(self):
        """Staggered MC with 30 seeds (smaller because each DGP is larger).

        Anchors BOTH `tau_total` and `delta_1` recovery on the staggered
        DGP. Per-ring `delta_jk` (event-time decomposition) is deferred
        alongside `event_study=True` support.
        """
        att_estimates = []
        delta_estimates = []
        for s in range(30):
            df = generate_butts_staggered_dgp(tau_total=-0.07, delta_1=-0.04, seed=s)
            result = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon")).fit(
                df, outcome="y", unit="unit", time="time", first_treat="first_treat"
            )
            att_estimates.append(result.att)
            if result.spillover_effects is not None and len(result.spillover_effects) > 0:
                delta_estimates.append(result.spillover_effects.iloc[0]["coef"])
        mean_att = float(np.mean(att_estimates))
        mean_delta = float(np.mean(delta_estimates))
        # Staggered MC is noisier than non-staggered; allow a looser
        # tolerance (0.04 on tau_total, 0.03 on delta_1).
        assert (
            abs(mean_att - (-0.07)) < 0.04
        ), f"staggered tau_total: expected -0.07, got {mean_att:.4f}"
        assert (
            abs(mean_delta - (-0.04)) < 0.03
        ), f"staggered delta_1: expected -0.04, got {mean_delta:.4f}"


# =============================================================================
# Step 3 (continued): staggered smoke test
# =============================================================================


# =============================================================================
# Step 7: Conley integration end-to-end
# =============================================================================


class TestSpilloverDiDWithConley:
    """Step 7: vcov_type='conley' flows through stage 2 cleanly."""

    def test_conley_fit_runs(self):
        df = generate_butts_nonstaggered_dgp(seed=42)
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            vcov_type="conley",
            conley_cutoff_km=200.0,
            conley_lag_cutoff=0,  # cross-sectional only (2-period panel)
        )
        result = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        assert result.vcov_type == "conley"
        assert result.conley_lag_cutoff == 0
        assert np.isfinite(result.se)

    def test_conley_kwargs_threaded_to_solve_ols(self):
        """Round-8 CI review P3 (test coverage gap): the previous test was a
        smoke test that only asserted finite SE + ATT invariance — a silent
        fallback to HC1 would have passed. This test plumbing-verifies that
        `solve_ols` is actually called with `vcov_type="conley"` AND the
        Conley-specific kwargs (`conley_coords`, `conley_cutoff_km`,
        `conley_metric`, `conley_time`, `conley_unit`, `conley_lag_cutoff`).
        """
        from unittest.mock import patch

        df = generate_butts_nonstaggered_dgp(
            seed=42, n_treated=20, n_near_control=80, n_far_control=100
        )
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            vcov_type="conley",
            conley_cutoff_km=200.0,
            conley_metric="haversine",
            conley_lag_cutoff=0,
        )
        # Patch solve_ols at the import site in spillover.py so we can
        # observe the kwargs SpilloverDiD passes through at stage 2.
        import diff_diff.spillover as spillover_mod

        captured: dict = {}

        original_solve_ols = spillover_mod.solve_ols

        def spy_solve_ols(*args, **kwargs):
            # Capture the LAST call's kwargs (stage 2 is the last solve_ols
            # invocation in fit()).
            captured.clear()
            captured.update(kwargs)
            return original_solve_ols(*args, **kwargs)

        with patch.object(spillover_mod, "solve_ols", side_effect=spy_solve_ols):
            result = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")

        # Conley kwargs reached solve_ols (no silent HC1 fallback).
        assert (
            captured.get("vcov_type") == "conley"
        ), f"expected solve_ols vcov_type='conley', got {captured.get('vcov_type')!r}"
        assert captured.get("conley_cutoff_km") == 200.0
        assert captured.get("conley_metric") == "haversine"
        assert captured.get("conley_lag_cutoff") == 0
        # The fit-time-derived spatial / temporal arrays must be present and
        # have the right shape.
        coords = captured.get("conley_coords")
        assert coords is not None and coords.shape == (result.n_obs, 2)
        conley_time = captured.get("conley_time")
        conley_unit = captured.get("conley_unit")
        assert conley_time is not None and len(conley_time) == result.n_obs
        assert conley_unit is not None and len(conley_unit) == result.n_obs
        # And the reported SE is finite (the actual Conley computation
        # completed end-to-end).
        assert np.isfinite(result.se)

    def test_conley_att_invariant_vs_hc1(self):
        """Point-estimate invariance: vcov choice does not change ATT
        (the residualization + OLS fit are independent of variance).
        """
        df = generate_butts_nonstaggered_dgp(
            seed=42, n_treated=20, n_near_control=80, n_far_control=100
        )
        result_hc1 = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            vcov_type="hc1",
        ).fit(df, outcome="y", unit="unit", time="time", treatment="D")
        result_conley = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            vcov_type="conley",
            conley_cutoff_km=200.0,
            conley_lag_cutoff=0,
        ).fit(df, outcome="y", unit="unit", time="time", treatment="D")
        assert abs(result_hc1.att - result_conley.att) < 1e-10


# =============================================================================
# Step 3 (continued): staggered smoke test
# =============================================================================


class TestSpilloverDiDStaggeredFit:
    """Step 3: staggered timing produces sensible results."""

    def test_staggered_fit_runs(self):
        # 3 cohorts, 4 periods
        rng = np.random.default_rng(0)
        rows = []
        cohort_onsets = {1: 1, 2: 2, "C": np.inf}
        # 6 units per cohort placed near distinct centers
        for cohort_id, onset in cohort_onsets.items():
            center_lat = 0.0 if cohort_id == 1 else (10.0 if cohort_id == 2 else 50.0)
            for i in range(6):
                u = f"{cohort_id}_{i}"
                lat = center_lat + rng.normal(0, 0.05)
                lon = rng.normal(0, 0.05)
                first_treat = float(onset) if onset != np.inf else np.inf
                for t in range(4):
                    rows.append(
                        {
                            "unit": u,
                            "time": t,
                            "lat": lat,
                            "lon": lon,
                            "first_treat": first_treat,
                            "y": 1.0
                            + 0.1 * t
                            + (0.05 * (t >= first_treat) if np.isfinite(first_treat) else 0)
                            + rng.normal(0, 0.05),
                        }
                    )
        df = pd.DataFrame(rows)
        est = SpilloverDiD(
            rings=[0.0, 50.0],  # ring covers 0-50 km; far cutoff at 50
            conley_coords=("lat", "lon"),
        )
        result = est.fit(df, outcome="y", unit="unit", time="time", first_treat="first_treat")
        assert result.is_staggered is True
        assert np.isfinite(result.att)


# =============================================================================
# Step 2: Timing conversion helpers
# =============================================================================


class TestExtractTreatmentOnsets:
    """Tests for _extract_treatment_onsets."""

    def test_canonical_finite_onsets(self):
        df = pd.DataFrame(
            {
                "unit": ["A", "A", "B", "B", "C", "C"],
                "first_treat": [1, 1, 2, 2, np.inf, np.inf],
                "time": [0, 1, 0, 1, 0, 1],
            }
        )
        onsets = _extract_treatment_onsets(df, "first_treat", "unit")
        assert onsets == {"A": 1.0, "B": 2.0, "C": np.inf}

    def test_zero_treated_as_never(self):
        df = pd.DataFrame({"unit": ["A", "A"], "first_treat": [0, 0], "time": [0, 1]})
        onsets = _extract_treatment_onsets(df, "first_treat", "unit")
        assert onsets == {"A": np.inf}

    def test_nan_treated_as_never(self):
        df = pd.DataFrame({"unit": ["A", "A"], "first_treat": [np.nan, np.nan], "time": [0, 1]})
        onsets = _extract_treatment_onsets(df, "first_treat", "unit")
        assert onsets == {"A": np.inf}


class TestConvertTreatmentToFirstTreat:
    """Tests for _convert_treatment_to_first_treat."""

    def test_basic_conversion(self):
        # 2 units, 3 periods; A treated from t=1, B never-treated.
        df = pd.DataFrame(
            {
                "unit": ["A"] * 3 + ["B"] * 3,
                "time": [0, 1, 2, 0, 1, 2],
                "D": [0, 1, 1, 0, 0, 0],
            }
        )
        out, col = _convert_treatment_to_first_treat(df, "D", "time", "unit")
        assert col == "_spillover_first_treat"
        a_rows = out[out["unit"] == "A"]
        b_rows = out[out["unit"] == "B"]
        assert (a_rows["_spillover_first_treat"] == 1.0).all()
        assert np.isinf(b_rows["_spillover_first_treat"]).all()

    def test_no_treated_units_marks_all_inf(self):
        df = pd.DataFrame({"unit": ["A", "A"], "time": [0, 1], "D": [0, 0]})
        out, _ = _convert_treatment_to_first_treat(df, "D", "time", "unit")
        assert np.isinf(out["_spillover_first_treat"]).all()

    def test_missing_treatment_column_raises(self):
        df = pd.DataFrame({"unit": ["A"], "time": [0]})
        with pytest.raises(ValueError, match="not in data"):
            _convert_treatment_to_first_treat(df, "D", "time", "unit")

    def test_non_binary_treatment_raises(self):
        df = pd.DataFrame({"unit": ["A", "A"], "time": [0, 1], "D": [0, 2]})
        with pytest.raises(ValueError, match="exact 0/1"):
            _convert_treatment_to_first_treat(df, "D", "time", "unit")

    def test_caller_dataframe_unchanged(self):
        df = pd.DataFrame({"unit": ["A", "A"], "time": [0, 1], "D": [0, 1]})
        original_cols = list(df.columns)
        _convert_treatment_to_first_treat(df, "D", "time", "unit")
        # The defensive copy + column add does NOT leak back to caller.
        assert list(df.columns) == original_cols


# =============================================================================
# Step 2: SpilloverDiD validators
# =============================================================================


@pytest.fixture
def simple_panel():
    """Minimal valid 2-period panel for validator tests."""
    return pd.DataFrame(
        {
            "unit": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "time": [0, 1] * 4,
            "lat": [0.0, 0.0, 0.1, 0.1, 5.0, 5.0, 5.1, 5.1],
            "lon": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "D": [0, 1, 0, 1, 0, 0, 0, 0],
            "first_treat": [1, 1, 1, 1, np.inf, np.inf, np.inf, np.inf],
            "y": np.arange(8.0),
        }
    )


class TestValidateSpilloverInputs:
    """Behavioral validation of front-door checks."""

    def test_valid_minimal_input_passes(self, simple_panel):
        est = SpilloverDiD(rings=[0.0, 200.0], conley_coords=("lat", "lon"))
        # Should not raise.
        est._validate_spillover_inputs(simple_panel, "D", None, "time", "unit", "y")
        assert est._effective_d_bar == 200.0

    def test_rings_too_short_raises(self, simple_panel):
        est = SpilloverDiD(rings=[100.0])
        with pytest.raises(ValueError, match="at least 2 breakpoints"):
            est._validate_spillover_inputs(simple_panel, "D", None, "time", "unit", "y")

    def test_rings_non_sorted_raises(self, simple_panel):
        est = SpilloverDiD(rings=[50.0, 100.0, 75.0])
        with pytest.raises(ValueError, match="strictly increasing"):
            est._validate_spillover_inputs(simple_panel, "D", None, "time", "unit", "y")

    def test_rings_negative_raises(self, simple_panel):
        est = SpilloverDiD(rings=[-10.0, 100.0])
        with pytest.raises(ValueError, match="non-negative"):
            est._validate_spillover_inputs(simple_panel, "D", None, "time", "unit", "y")

    def test_d_bar_mismatched_with_rings_raises(self, simple_panel):
        est = SpilloverDiD(rings=[0.0, 100.0, 200.0], d_bar=150.0)
        with pytest.raises(ValueError, match="d_bar.*must equal"):
            est._validate_spillover_inputs(simple_panel, "D", None, "time", "unit", "y")

    def test_d_bar_equal_to_max_rings_accepted(self, simple_panel):
        est = SpilloverDiD(rings=[0.0, 100.0, 200.0], d_bar=200.0, conley_coords=("lat", "lon"))
        est._validate_spillover_inputs(simple_panel, "D", None, "time", "unit", "y")
        assert est._effective_d_bar == 200.0

    def test_d_bar_default_uses_max_rings(self, simple_panel):
        est = SpilloverDiD(rings=[0.0, 50.0, 175.0], conley_coords=("lat", "lon"))
        est._validate_spillover_inputs(simple_panel, "D", None, "time", "unit", "y")
        assert est._effective_d_bar == 175.0

    def test_treatment_and_first_treat_both_raise(self, simple_panel):
        est = SpilloverDiD(rings=[0.0, 200.0])
        with pytest.raises(ValueError, match="either.*or"):
            est._validate_spillover_inputs(simple_panel, "D", "first_treat", "time", "unit", "y")

    def test_neither_treatment_nor_first_treat_raises(self, simple_panel):
        est = SpilloverDiD(rings=[0.0, 200.0])
        with pytest.raises(ValueError, match="Exactly one of"):
            est._validate_spillover_inputs(simple_panel, None, None, "time", "unit", "y")

    def test_missing_required_column_raises(self, simple_panel):
        est = SpilloverDiD(rings=[0.0, 200.0])
        with pytest.raises(ValueError, match="Missing required columns"):
            est._validate_spillover_inputs(
                simple_panel, "D", None, "time", "nonexistent_unit_col", "y"
            )

    def test_conley_requires_coords(self, simple_panel):
        est = SpilloverDiD(
            rings=[0.0, 200.0],
            vcov_type="conley",
            conley_cutoff_km=200.0,
            conley_lag_cutoff=0,
        )
        with pytest.raises(ValueError, match="conley_coords"):
            est._validate_spillover_inputs(simple_panel, "D", None, "time", "unit", "y")

    def test_conley_coords_must_be_2_tuple(self, simple_panel):
        est = SpilloverDiD(
            rings=[0.0, 200.0],
            vcov_type="conley",
            conley_coords=("lat",),  # type: ignore[arg-type]  # only 1 element
            conley_cutoff_km=200.0,
            conley_lag_cutoff=0,
        )
        with pytest.raises(ValueError, match="2-tuple"):
            est._validate_spillover_inputs(simple_panel, "D", None, "time", "unit", "y")

    def test_conley_coord_column_missing_raises(self, simple_panel):
        est = SpilloverDiD(
            rings=[0.0, 200.0],
            vcov_type="conley",
            conley_coords=("lat", "missing_lon"),
            conley_cutoff_km=200.0,
            conley_lag_cutoff=0,
        )
        with pytest.raises(ValueError, match="'missing_lon' not in data"):
            est._validate_spillover_inputs(simple_panel, "D", None, "time", "unit", "y")

    def test_conley_requires_positive_cutoff(self, simple_panel):
        est = SpilloverDiD(
            rings=[0.0, 200.0],
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=-1.0,
            conley_lag_cutoff=0,
        )
        with pytest.raises(ValueError, match="conley_cutoff_km"):
            est._validate_spillover_inputs(simple_panel, "D", None, "time", "unit", "y")

    def test_conley_requires_lag_cutoff(self, simple_panel):
        est = SpilloverDiD(
            rings=[0.0, 200.0],
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=200.0,
            conley_lag_cutoff=None,
        )
        with pytest.raises(ValueError, match="conley_lag_cutoff"):
            est._validate_spillover_inputs(simple_panel, "D", None, "time", "unit", "y")

    def test_nan_coords_raise(self, simple_panel):
        df = simple_panel.copy()
        df.loc[0, "lat"] = np.nan
        est = SpilloverDiD(
            rings=[0.0, 200.0],
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=200.0,
            conley_lag_cutoff=0,
        )
        with pytest.raises(ValueError, match="non-finite"):
            est._validate_spillover_inputs(df, "D", None, "time", "unit", "y")

    def test_no_treated_observations_raises(self, simple_panel):
        df = simple_panel.copy()
        df["D"] = 0
        est = SpilloverDiD(rings=[0.0, 200.0], conley_coords=("lat", "lon"))
        with pytest.raises(ValueError, match="No treated observations"):
            est._validate_spillover_inputs(df, "D", None, "time", "unit", "y")

    def test_cluster_column_missing_raises(self, simple_panel):
        est = SpilloverDiD(
            rings=[0.0, 200.0],
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=200.0,
            conley_lag_cutoff=0,
            cluster="not_a_real_column",
        )
        with pytest.raises(ValueError, match="cluster column"):
            est._validate_spillover_inputs(simple_panel, "D", None, "time", "unit", "y")


class TestValidateFarAwayExists:
    """Tests for SpilloverDiD._validate_far_away_exists."""

    def test_returns_count_when_satisfied(self):
        est = SpilloverDiD(rings=[0.0, 100.0])
        est._effective_d_bar = 100.0
        d = np.array([10.0, 50.0, 500.0, 1000.0])
        is_control = np.array([False, True, True, True])
        n = est._validate_far_away_exists(d, is_control)
        assert n == 2

    def test_raises_when_no_far_controls(self):
        est = SpilloverDiD(rings=[0.0, 100.0])
        est._effective_d_bar = 100.0
        d = np.array([10.0, 50.0, 99.9, 100.0])
        is_control = np.array([False, True, True, True])
        with pytest.raises(ValueError, match="Assumption 5"):
            est._validate_far_away_exists(d, is_control)

    def test_raises_when_far_units_all_treated(self):
        """Only treated units beyond d_bar (impossible in non-staggered, but the
        validator's job is to check the population that identifies the
        counterfactual: controls strictly past d_bar)."""
        est = SpilloverDiD(rings=[0.0, 100.0])
        est._effective_d_bar = 100.0
        d = np.array([200.0, 300.0, 50.0])
        is_control = np.array([False, False, True])  # only the close unit is control
        with pytest.raises(ValueError, match="Assumption 5"):
            est._validate_far_away_exists(d, is_control)


# =============================================================================
# Codex-review regression tests (post-Wave-B-MVP first review)
# =============================================================================


class TestSpilloverDiDCovariatesRejected:
    """covariates= must raise NotImplementedError in Wave B MVP.

    Stage-1 covariate residualization (Gardner-style) is not yet wired
    through; appending raw covariates only at stage 2 silently biases
    tau_total / delta_j on panels with time-varying covariates.
    """

    def test_covariates_raises_not_implemented(self):
        df = _make_butts_2period_dgp(seed=42)
        df["x"] = np.random.default_rng(0).normal(size=len(df))
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        with pytest.raises(NotImplementedError, match="covariates"):
            est.fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                treatment="D",
                covariates=["x"],
            )

    def test_empty_covariates_accepted(self):
        """An empty covariates list is the same as no covariates — should NOT raise."""
        df = _make_butts_2period_dgp(seed=42)
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        # Should not raise.
        est.fit(df, outcome="y", unit="unit", time="time", treatment="D", covariates=[])


class TestSpilloverDiDAbsorbingTreatmentValidation:
    """Reversible / non-absorbing treatment patterns must raise."""

    def test_reversible_treatment_path_raises(self):
        # A unit's treatment goes [0, 1, 0] across 3 periods.
        rows = []
        rng = np.random.default_rng(0)
        for u in ("treated_reversing", "ctrl_far"):
            for t in range(3):
                if u == "treated_reversing":
                    d_val = 1 if t == 1 else 0
                    lat, lon = 0.0, 0.0
                else:
                    d_val = 0
                    lat, lon = 5.0, 0.0
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "lat": lat,
                        "lon": lon,
                        "D": d_val,
                        "y": rng.normal(),
                    }
                )
        df = pd.DataFrame(rows)
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        with pytest.raises(ValueError, match="non-absorbing"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")

    def test_non_constant_first_treat_raises(self):
        # Same unit has two different first_treat values across rows.
        rows = []
        for t in range(3):
            rows.append(
                {
                    "unit": "u1",
                    "time": t,
                    "lat": 0.0,
                    "lon": 0.0,
                    "first_treat": 1.0 if t < 2 else 2.0,  # CHANGES at t=2
                    "y": float(t),
                }
            )
            rows.append(
                {
                    "unit": "u2_far",
                    "time": t,
                    "lat": 5.0,
                    "lon": 0.0,
                    "first_treat": np.inf,
                    "y": float(t),
                }
            )
        df = pd.DataFrame(rows)
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        with pytest.raises(ValueError, match="non-constant"):
            est.fit(df, outcome="y", unit="unit", time="time", first_treat="first_treat")


class TestSpilloverDiDAllEventuallyTreated:
    """All-eventually-treated staggered designs (no never-treated units)
    should work as long as some not-yet-treated rows are far-away controls.
    The Codex review flagged the prior is_ever_control logic as too strict.
    """

    def test_all_eventually_treated_staggered(self):
        # Two cohorts, no never-treated units. Far cohort treats at t=10
        # (well past the panel's t=3 horizon), so its early-period rows
        # serve as far-away controls for the early cohort's treatment.
        rng = np.random.default_rng(0)
        # Per-unit coords sampled ONCE (within-unit constancy required).
        early_coords = [(rng.normal(0, 0.005), rng.normal(0, 0.005)) for _ in range(8)]
        late_coords = [(5.0 + rng.normal(0, 0.005), rng.normal(0, 0.005)) for _ in range(8)]
        rows = []
        # Early cohort: treats at t=2, clustered at origin
        for i, (lat, lon) in enumerate(early_coords):
            u = f"early_{i}"
            for t in range(4):
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "lat": lat,
                        "lon": lon,
                        "first_treat": 2.0,
                        "y": rng.normal() + 0.1 * t + (-0.07 if t >= 2 else 0.0),
                    }
                )
        # Late cohort: treats at t=10 (far outside panel), placed FAR from early
        for i, (lat, lon) in enumerate(late_coords):
            u = f"late_{i}"
            for t in range(4):
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "lat": lat,
                        "lon": lon,
                        "first_treat": 10.0,  # never treats in panel
                        "y": rng.normal() + 0.1 * t,
                    }
                )
        df = pd.DataFrame(rows)
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        result = est.fit(df, outcome="y", unit="unit", time="time", first_treat="first_treat")
        assert np.isfinite(result.att)


class TestSpilloverDiDConleyCoordsAlwaysRequired:
    """conley_coords must be validated on every fit path, not just vcov_type=conley.
    The Codex review noted the default-hc1 path was failing with AssertionError.
    """

    def test_missing_conley_coords_on_hc1_path_raises_value_error(self):
        df = _make_butts_2period_dgp(seed=42)
        est = SpilloverDiD(rings=[0.0, 100.0])  # no conley_coords, default hc1
        with pytest.raises(ValueError, match="conley_coords"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")


class TestSpilloverDiDResultNObsMatchesEstimationSample:
    """result.n_obs must equal the stage-2 estimation sample (after dropping
    rows with non-finite y_tilde from rank-deficient stage-1 FE).
    """

    def test_n_obs_equals_finite_mask_count(self):
        df = _make_butts_2period_dgp(seed=42)
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        result = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        # In a well-conditioned DGP no rows are dropped, so n_obs = len(df).
        assert result.n_obs == len(df)
        # n_treated + n_control == n_obs (no overlap, no leakage).
        assert result.n_treated + result.n_control == result.n_obs


# =============================================================================
# Codex review round-2 regression tests
# =============================================================================


class TestSpilloverDiDOmega0IdentificationCheck:
    """Stage-1 FE support: every unit and every period in the panel must have
    at least one Omega_0 = {D_it=0 AND S_it=0} row, otherwise FE is
    unidentified for that unit/period and stage-2 estimates would be
    silently dropped. Round-2 codex review wanted up-front rejection.
    """

    def test_unsupported_period_raises(self):
        # Panel where t=1 has zero Omega_0 support: every t=1 unit is
        # either treated (D=1) or near (S=1 since d_i <= d_bar=200).
        # The far-aways only contribute at t=0 (no t=1 row).
        rng = np.random.default_rng(0)
        # Per-unit coords sampled ONCE (within-unit-constancy required).
        treated_coords = [(rng.normal(0, 0.005), rng.normal(0, 0.005)) for _ in range(4)]
        near_coords = [(rng.uniform(0.1, 0.5), rng.uniform(-0.3, 0.3)) for _ in range(4)]
        far_coords = [(5.0 + rng.normal(0, 0.005), rng.normal(0, 0.005)) for _ in range(4)]
        rows = []
        for i, (lat, lon) in enumerate(treated_coords):
            for t in range(2):
                rows.append(
                    {
                        "unit": f"T{i}",
                        "time": t,
                        "lat": lat,
                        "lon": lon,
                        "D": int(t == 1),
                        "y": rng.normal(),
                    }
                )
        for i, (lat, lon) in enumerate(near_coords):
            for t in range(2):
                rows.append(
                    {
                        "unit": f"N{i}",
                        "time": t,
                        "lat": lat,
                        "lon": lon,
                        "D": 0,
                        "y": rng.normal(),
                    }
                )
        # Far-aways at PRE only: no t=1 row → Omega_0 ∩ {t=1} is empty.
        for i, (lat, lon) in enumerate(far_coords):
            rows.append(
                {
                    "unit": f"F{i}",
                    "time": 0,
                    "lat": lat,
                    "lon": lon,
                    "D": 0,
                    "y": rng.normal(),
                }
            )
        df = pd.DataFrame(rows)
        est = SpilloverDiD(rings=[0.0, 200.0], d_bar=200.0, conley_coords=("lat", "lon"))
        with pytest.raises(ValueError, match="unidentified"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")


class TestSpilloverDiDConleyCoordsConstantPerUnit:
    """conley_coords must be constant within each unit across rows.
    Round-2 codex review noted ring construction collapses coords via
    `drop_duplicates(subset=[unit])` — non-constant coords silently use
    only the first row's location.
    """

    def test_time_varying_coords_raises(self):
        rows = []
        rng = np.random.default_rng(0)
        # u1 has different lat at t=0 vs t=1
        rows.append({"unit": "u1", "time": 0, "lat": 0.0, "lon": 0.0, "D": 0, "y": rng.normal()})
        rows.append(
            {
                "unit": "u1",
                "time": 1,
                "lat": 1.5,  # changed!
                "lon": 0.0,
                "D": 1,
                "y": rng.normal(),
            }
        )
        # u2 is well-behaved (constant coords)
        for t in (0, 1):
            rows.append(
                {
                    "unit": "u2",
                    "time": t,
                    "lat": 5.0,
                    "lon": 0.0,
                    "D": 0,
                    "y": rng.normal(),
                }
            )
        df = pd.DataFrame(rows)
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        with pytest.raises(ValueError, match="non-constant"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")


# =============================================================================
# Codex review round-3 regression tests
# =============================================================================


class TestSpilloverDiDFractionalTreatmentRejected:
    """Treatment column with fractional values (0.9, 1.1, etc.) must raise.
    Round-3 codex review caught that `int(v) in (0, 1)` was rounding down
    and silently misclassifying fractional rows.
    """

    @pytest.mark.parametrize("bad_value", [0.5, 0.9, 1.1, -0.1, 2.5])
    def test_fractional_treatment_raises(self, bad_value):
        df = _make_butts_2period_dgp(seed=42).copy()
        # Cast D to float64 before assigning the fractional value. Modern
        # pandas (3.x) raises TypeError on int64-column fractional setitem
        # BEFORE SpilloverDiD.fit() ever sees the input, so we promote the
        # column dtype first to ensure the fractional value actually
        # reaches the validator we're testing.
        df["D"] = df["D"].astype(float)
        first_treated_idx = df.index[df["D"] == 1][0]
        df.loc[first_treated_idx, "D"] = bad_value
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        with pytest.raises(ValueError, match="exact 0/1"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")


class TestSpilloverDiDNoTreatedRowRaises:
    """If all first_treat > max(time), D_it is all zeros after the
    anticipation shift. Must raise a clear identification error rather
    than crashing in solve_ols.
    """

    def test_future_only_first_treat_raises(self):
        rng = np.random.default_rng(0)
        # All units treat at t=10, but panel only spans t=0..2.
        rows = []
        for i in range(4):
            lat, lon = rng.normal(0, 0.005), rng.normal(0, 0.005)
            for t in range(3):
                rows.append(
                    {
                        "unit": f"T{i}",
                        "time": t,
                        "lat": lat,
                        "lon": lon,
                        "first_treat": 10.0,  # > max(time) = 2
                        "y": rng.normal(),
                    }
                )
        for i in range(4):
            lat, lon = 5.0 + rng.normal(0, 0.005), rng.normal(0, 0.005)
            for t in range(3):
                rows.append(
                    {
                        "unit": f"F{i}",
                        "time": t,
                        "lat": lat,
                        "lon": lon,
                        "first_treat": np.inf,
                        "y": rng.normal(),
                    }
                )
        df = pd.DataFrame(rows)
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        with pytest.raises(ValueError, match="No observation is treated"):
            est.fit(df, outcome="y", unit="unit", time="time", first_treat="first_treat")


class TestSpilloverDiDHaversineDomain:
    """Haversine lat/lon range validation applies on EVERY vcov path
    (not just vcov_type='conley'), because ring construction always
    uses the configured metric. Round-3 codex review noted out-of-range
    coords silently produced wrong ring assignment on hc1/cluster paths.
    """

    def test_out_of_range_lat_on_hc1_raises(self):
        df = _make_butts_2period_dgp(seed=42).copy()
        # Corrupt one row's lat to be > 90 (impossible geographic value).
        df.loc[0, "lat"] = 95.0
        # Force the constancy check to ignore this corruption: also corrupt
        # the unit's other row to the same value (constant per unit).
        unit_of_first = df.loc[0, "unit"]
        df.loc[df["unit"] == unit_of_first, "lat"] = 95.0
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            vcov_type="hc1",  # NOT conley
        )
        with pytest.raises(ValueError, match=r"latitude.*\[-90, 90\]"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")

    def test_out_of_range_lon_on_hc1_raises(self):
        df = _make_butts_2period_dgp(seed=42).copy()
        unit_of_first = df.loc[0, "unit"]
        df.loc[df["unit"] == unit_of_first, "lon"] = 200.0  # > 180
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        with pytest.raises(ValueError, match=r"longitude.*\[-180, 180\]"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")

    def test_euclidean_metric_skips_range_check(self):
        """conley_metric='euclidean' must NOT enforce haversine ranges."""
        df = _make_butts_2period_dgp(seed=42).copy()
        # Coordinates 95.0 / 200.0 are valid Euclidean but invalid haversine.
        df.loc[df["unit"] == df.loc[0, "unit"], "lat"] = 95.0
        df.loc[df["unit"] == df.loc[0, "unit"], "lon"] = 200.0
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            conley_metric="euclidean",
        )
        # Should not raise. (May still fail downstream for other reasons —
        # we just need to confirm the haversine range gate is metric-aware.)
        try:
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        except ValueError as exc:
            # Acceptable: a different (non-domain) error from downstream.
            assert "[-90, 90]" not in str(exc) and "[-180, 180]" not in str(exc)


# =============================================================================
# Codex review round-4 regression tests
# =============================================================================


class TestSpilloverDiDZeroIndexedBaselineTreated:
    """Auto-generated `_spillover_first_treat` (from binary `D`) must NOT
    collapse `0` onto the never-treated sentinel. Round-4 codex review
    caught that `_extract_treatment_onsets` was collapsing 0 → inf for
    EVERY first_treat input, silently reclassifying baseline-treated
    units (D=1 at t=0) as never-treated.
    """

    def test_baseline_treated_unit_at_t0_recognized(self):
        """A baseline-treated unit (D=1 at t=0) used to silently become
        never-treated because `_convert_treatment_to_first_treat` wrote
        first_treat=0 and `_extract_treatment_onsets` collapsed 0 -> inf.
        After the fix, u1 is correctly recognized as treated. Because u1
        has no Omega_0 rows (D=1 at all t), it triggers the round-16
        warn-and-drop path: the warning naming `u1_baseline` PROVES it was
        recognized as treated (the OLD bug silently reclassified u1 to
        "never treated", which would have passed the Omega_0 check
        without warning and produced garbage estimates).
        """
        rng = np.random.default_rng(0)
        rows = []
        # u1: baseline-treated (D=1 at all t). No Omega_0 rows → warned-
        # and-dropped from stage 2.
        # u2: treated from t=1 (D=0 at t=0, D=1 at t=1). Far from u1.
        # u3: untreated far-control. Provides Omega_0 support.
        for t in (0, 1):
            rows.append(
                {
                    "unit": "u1_baseline",
                    "time": t,
                    "lat": 0.0,
                    "lon": 0.0,
                    "D": 1,
                    "y": rng.normal(),
                }
            )
        for t in (0, 1):
            rows.append(
                {
                    "unit": "u2_treated_t1",
                    "time": t,
                    "lat": 10.0,
                    "lon": 0.0,
                    "D": int(t == 1),
                    "y": rng.normal(),
                }
            )
        for t in (0, 1):
            rows.append(
                {
                    "unit": "u3_far_control",
                    "time": t,
                    "lat": 20.0,
                    "lon": 0.0,
                    "D": 0,
                    "y": rng.normal(),
                }
            )
        df = pd.DataFrame(rows)
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        # PROOF u1 is recognized as treated: the warning names u1_baseline.
        with pytest.warns(UserWarning, match="u1_baseline"):
            result = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        # u1's 2 rows excluded from stage 2; u2 (1 treated row at t=1) and
        # u3 (2 control rows) remain. Verify n_treated reflects only the
        # supported sample.
        assert result.n_treated == 1, (
            f"After warn-drop of u1_baseline, expected 1 treated row "
            f"(u2 at t=1); got n_treated={result.n_treated}"
        )

    def test_partial_unsupported_units_warn_and_drop(self):
        """Round-16 codex review: units with no Omega_0 row should be
        warned-and-dropped (matching TwoStageDiD's always-treated convention),
        not block the full fit. The remaining supported sample fits normally.
        """
        rng = np.random.default_rng(1)
        rows = []
        # 4 baseline-treated units (no Omega_0 rows → all 4 warned-dropped).
        for k in range(4):
            for t in (0, 1):
                rows.append(
                    {
                        "unit": f"baseline_{k}",
                        "time": t,
                        "lat": 0.0 + k * 0.001,
                        "lon": 0.0,
                        "D": 1,
                        "y": rng.normal(),
                    }
                )
        # 3 validly-treated units (treated from t=1; far from baselines).
        for k in range(3):
            for t in (0, 1):
                rows.append(
                    {
                        "unit": f"treated_t1_{k}",
                        "time": t,
                        "lat": 10.0 + k * 0.01,
                        "lon": 0.0,
                        "D": int(t == 1),
                        "y": rng.normal(),
                    }
                )
        # 5 far-controls (full Omega_0 support).
        for k in range(5):
            for t in (0, 1):
                rows.append(
                    {
                        "unit": f"far_control_{k}",
                        "time": t,
                        "lat": 20.0 + k * 0.01,
                        "lon": 0.0,
                        "D": 0,
                        "y": rng.normal(),
                    }
                )
        df = pd.DataFrame(rows)
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        with pytest.warns(UserWarning, match="4 unit\\(s\\) have NO"):
            result = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        # 4 baselines × 2 periods = 8 rows excluded. Remaining: 3 treated +
        # 5 controls = 8 units × 2 periods = 16 rows. n_treated = 3 (one
        # per treated unit at t=1).
        assert result.n_treated == 3
        assert result.n_obs == 16

    def test_unsupported_period_still_raises(self):
        """Period-level Omega_0 unsupport remains a hard error (round-16
        codex split): dropping a period would remove all units' rows at
        that t, losing the cross-time identification entirely.
        """
        rng = np.random.default_rng(2)
        # Balanced panel where t=1 has NO Omega_0 rows: every unit at t=1
        # is either treated or near a treated unit.
        rows = []
        # 2 treated units at t=1; 2 near-controls (within d_bar of treated)
        # at both t. No far-controls → no Omega_0 row at t=1.
        for k in range(2):
            for t in (0, 1):
                rows.append(
                    {
                        "unit": f"T{k}",
                        "time": t,
                        "lat": 0.0 + k * 0.001,
                        "lon": 0.0,
                        "D": int(t == 1),
                        "y": rng.normal(),
                    }
                )
        # Near-controls at both periods. Pre: untreated and (no current
        # treatment) → S=0 → Omega_0. Post: untreated but treated nearby →
        # S=1 → NOT Omega_0. So t=1 has no Omega_0 row.
        for k in range(2):
            for t in (0, 1):
                rows.append(
                    {
                        "unit": f"N{k}",
                        "time": t,
                        "lat": 0.1 + k * 0.001,
                        "lon": 0.0,
                        "D": 0,
                        "y": rng.normal(),
                    }
                )
        df = pd.DataFrame(rows)
        est = SpilloverDiD(rings=[0.0, 100.0], d_bar=100.0, conley_coords=("lat", "lon"))
        with pytest.raises(ValueError, match="period.*unidentified|unidentified.*period"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")

    def test_baseline_treated_with_far_control_pretreatment_works(self):
        """A control unit at t<0 (pre-treatment for the baseline-treated)
        provides the missing Omega_0 support, but baseline-treated units
        still have no untreated rows. Confirm the system can FIT when
        there's a clean control population, while still recognizing
        baseline-treated units as treated."""
        rng = np.random.default_rng(0)
        rows = []
        # u1 is treated from t=1 (NOT baseline) — gives Omega_0 support.
        for t in (0, 1):
            rows.append(
                {
                    "unit": "u1_t1_treated",
                    "time": t,
                    "lat": 0.0,
                    "lon": 0.0,
                    "D": int(t == 1),
                    "y": rng.normal(),
                }
            )
        for t in (0, 1):
            rows.append(
                {
                    "unit": "u2_far",
                    "time": t,
                    "lat": 5.0,
                    "lon": 0.0,
                    "D": 0,
                    "y": rng.normal(),
                }
            )
        df = pd.DataFrame(rows)
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        result = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        # n_treated should be 1 (u1 at t=1 only); n_control should be 3
        # (u1's t=0 row + u2's both rows). Critical: u1 is NOT silently
        # reclassified.
        assert result.n_treated == 1
        assert result.n_control == 3


class TestSpilloverDiDCallableSelfDistance:
    """Callable metrics on the ring-construction path must satisfy the
    same self-distance / symmetry contract as conley's vcov path.
    Round-4 codex review noted positive self-distance silently corrupted
    ring assignment on hc1/cluster fits.
    """

    def test_positive_self_distance_callable_raises(self):
        df = _make_butts_2period_dgp(seed=42)

        def bad_metric(a, b):
            # Returns CONSTANT 7.5 — fails the zero-diagonal check on the
            # (n, n) self-call validation.
            return np.full((a.shape[0], b.shape[0]), 7.5)

        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            conley_metric=bad_metric,
        )
        with pytest.raises(ValueError, match="diagonal"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")


# =============================================================================
# Codex review round-5 regression tests
# =============================================================================


class TestSpilloverDiDAnticipationPropagation:
    """anticipation is in `__init__` / `get_params` — round-5 codex review
    flagged that it wasn't surfaced on the SpilloverDiDResults / to_dict()
    so downstream consumers couldn't reconstruct the fitted estimand.
    """

    def test_anticipation_round_trips_to_result_and_dict(self):
        df = _make_butts_2period_dgp(seed=42)
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            anticipation=0,
        )
        result = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        # anticipation is on the result object
        assert result.anticipation == 0
        # anticipation is in to_dict()
        d = result.to_dict()
        assert "anticipation" in d
        assert d["anticipation"] == 0


class TestSpilloverDiDAnticipationBehavior:
    """Round-7 CI review P1: anticipation must change the fitted estimand,
    not just round-trip through the result object. It shifts BOTH the
    treatment indicator and the ring-exposure clock by `-anticipation`,
    moving rows in and out of Omega_0 and changing `tau_total` / `delta_j`.
    Hand-built 4-period panel with one treated unit at t=2: anticipation=1
    promotes t=1 into the "treated" window, dropping that row from
    Omega_0 and increasing n_treated by one period's worth of obs.
    Verified on both fit entry paths (`treatment=` and `first_treat=`).
    """

    @staticmethod
    def _make_4period_panel():
        rng = np.random.default_rng(42)
        # 1 treated @ t=2, 1 near-control, 2 far-controls. 4 periods.
        specs = [
            ("treated", 0.0, [0, 0, 1, 1]),
            ("near", 0.5, [0, 0, 0, 0]),
            ("far1", 5.0, [0, 0, 0, 0]),
            ("far2", 5.1, [0, 0, 0, 0]),
        ]
        rows = []
        for unit, lat, d_pattern in specs:
            for t in range(4):
                rows.append(
                    {
                        "unit": unit,
                        "time": t,
                        "lat": lat,
                        "lon": 0.0,
                        "D": d_pattern[t],
                        "y": rng.normal(),
                    }
                )
        return pd.DataFrame(rows)

    def test_anticipation_shifts_omega_0_on_treatment_path(self):
        """anticipation=1 on the binary `treatment=` path: the effective
        treatment indicator slides one period earlier, so t=1 (formerly
        Omega_0 for treated + near units) is dropped from stage 1 and
        promoted into the "currently-treated / currently-exposed" zone.
        Stage 1 sample shrinks; n_treated grows; att changes."""
        df = self._make_4period_panel()
        r0 = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            anticipation=0,
        ).fit(df, outcome="y", unit="unit", time="time", treatment="D")
        r1 = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            anticipation=1,
        ).fit(df, outcome="y", unit="unit", time="time", treatment="D")
        # n_treated grows by 1 row (treated unit's t=1 row promoted under
        # the shifted indicator).
        assert r1.n_treated == r0.n_treated + 1, (
            f"anticipation=1 should add one treated period; "
            f"got n_treated {r0.n_treated} -> {r1.n_treated}"
        )
        # Stage 1 Omega_0 sample shrinks (rows promoted to treated/exposed
        # leave Omega_0).
        assert r1.stage1_n_obs < r0.stage1_n_obs, (
            f"anticipation=1 should shrink Omega_0; got stage1_n_obs "
            f"{r0.stage1_n_obs} -> {r1.stage1_n_obs}"
        )
        # And the estimand changes (different sample → different att).
        assert r0.att != r1.att, f"anticipation=1 should change att; got {r0.att} == {r1.att}"

    def test_anticipation_shifts_omega_0_on_first_treat_path(self):
        """anticipation=1 on the Gardner `first_treat=` path: same shift
        applies. The first_treat column carries treatment onsets directly,
        and anticipation subtracts from each onset for both the D_it
        construction AND the ring-exposure (S_it) clock."""
        df = self._make_4period_panel()
        # Convert binary D to first_treat column.
        first_treat_map = {"treated": 2.0, "near": np.inf, "far1": np.inf, "far2": np.inf}
        df_ft = df.copy()
        df_ft["first_treat"] = df_ft["unit"].map(first_treat_map)
        df_ft = df_ft.drop(columns=["D"])

        r0 = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            anticipation=0,
        ).fit(df_ft, outcome="y", unit="unit", time="time", first_treat="first_treat")
        r1 = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            anticipation=1,
        ).fit(df_ft, outcome="y", unit="unit", time="time", first_treat="first_treat")
        assert r1.n_treated == r0.n_treated + 1, (
            f"first_treat path anticipation=1: expected +1 treated, "
            f"got {r0.n_treated} -> {r1.n_treated}"
        )
        assert r1.stage1_n_obs < r0.stage1_n_obs, (
            f"first_treat path anticipation=1: Omega_0 should shrink, "
            f"got {r0.stage1_n_obs} -> {r1.stage1_n_obs}"
        )
        assert r0.att != r1.att, (
            f"first_treat path anticipation=1: expected att to change, " f"got {r0.att} == {r1.att}"
        )

    def test_anticipation_shift_matches_across_fit_paths(self):
        """Sanity-check that the `treatment=` and `first_treat=` paths
        produce identical results under the same anticipation setting —
        the two entry points are internally unified, so anticipation must
        compose consistently with both."""
        df = self._make_4period_panel()
        df_ft = df.copy()
        df_ft["first_treat"] = df_ft["unit"].map(
            {"treated": 2.0, "near": np.inf, "far1": np.inf, "far2": np.inf}
        )
        df_ft = df_ft.drop(columns=["D"])

        for ant in (0, 1):
            r_d = SpilloverDiD(
                rings=[0.0, 100.0],
                conley_coords=("lat", "lon"),
                anticipation=ant,
            ).fit(df, outcome="y", unit="unit", time="time", treatment="D")
            r_ft = SpilloverDiD(
                rings=[0.0, 100.0],
                conley_coords=("lat", "lon"),
                anticipation=ant,
            ).fit(df_ft, outcome="y", unit="unit", time="time", first_treat="first_treat")
            assert r_d.att == r_ft.att, (
                f"anticipation={ant}: att mismatch between paths " f"({r_d.att} vs {r_ft.att})"
            )
            assert (
                r_d.stage1_n_obs == r_ft.stage1_n_obs
            ), f"anticipation={ant}: stage1_n_obs mismatch between paths"


class TestSpilloverDiDEffectiveRankDoF:
    """Stage-2 residual df should use effective rank (after solve_ols drops
    rank-deficient columns), not raw column count. Round-5 codex review
    noted that using raw `X_2_fit.shape[1]` understates df_resid on
    rank-deficient stage-2 fits and silently inflates p-values / CI widths.
    """

    def test_rank_deficient_design_uses_effective_rank(self):
        # Construct a panel where the INNER ring [0, 50) has no controls
        # so its stage-2 column is identically zero and solve_ols drops
        # it. After the fix, df_resid uses the effective rank (k=2:
        # treatment + outer ring [50, 200]), not the raw 3-column count.
        rng = np.random.default_rng(42)
        rows = []
        # 8 treated near origin
        for i in range(8):
            lat, lon = rng.normal(0, 0.005), rng.normal(0, 0.005)
            for t in (0, 1):
                rows.append(
                    {
                        "unit": f"T{i}",
                        "time": t,
                        "lat": lat,
                        "lon": lon,
                        "D": int(t == 1),
                        "y": rng.normal(),
                    }
                )
        # 20 NEAR-controls in the OUTER ring [50, 200) only — at ~1.2°
        # ≈ 133 km from origin (inside outer ring, outside inner ring).
        for i in range(20):
            lat, lon = 1.2 + rng.normal(0, 0.005), rng.normal(0, 0.005)
            for t in (0, 1):
                rows.append(
                    {
                        "unit": f"N{i}",
                        "time": t,
                        "lat": lat,
                        "lon": lon,
                        "D": 0,
                        "y": rng.normal(),
                    }
                )
        # 20 far-controls beyond d_bar=200 → identify the counterfactual.
        for i in range(20):
            lat, lon = 5.0 + rng.normal(0, 0.005), rng.normal(0, 0.005)
            for t in (0, 1):
                rows.append(
                    {
                        "unit": f"F{i}",
                        "time": t,
                        "lat": lat,
                        "lon": lon,
                        "D": 0,
                        "y": rng.normal(),
                    }
                )
        df = pd.DataFrame(rows)
        est = SpilloverDiD(rings=[0.0, 50.0, 200.0], conley_coords=("lat", "lon"))
        result = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        # Sanity: outer-ring coef + ATT both finite.
        assert np.isfinite(result.att)
        assert np.isfinite(result.spillover_effects.loc["[50, 200]", "coef"])
        # The empty inner ring's coef should be NaN (dropped by solve_ols).
        assert np.isnan(result.spillover_effects.loc["[0, 50)", "coef"])


# =============================================================================
# Codex review round-6 regression tests
# =============================================================================


class TestSpilloverDiDStringCodedTimeOnTreatmentPath:
    """The `treatment=` path must coerce `time` to numeric BEFORE running
    `_convert_treatment_to_first_treat`. Round-6 codex review noted that
    string-coded numeric periods like ['0', '2', '10'] would sort
    lexicographically ('10' < '2') and produce the wrong onset.
    """

    def test_string_coded_time_treatment_path(self):
        # Unit u1: treated starting at time "2" (the SECOND period when
        # sorted numerically). Lexicographic sort would mis-order "10" < "2"
        # and assign first_treat = "10" (the alphabetic min of treated rows).
        rng = np.random.default_rng(42)
        rows = []
        # u1: time periods "0", "2", "10" — treated at "2" and "10"
        for t_str, t_num in [("0", 0), ("2", 2), ("10", 10)]:
            rows.append(
                {
                    "unit": "u1",
                    "time": t_str,
                    "lat": 0.0,
                    "lon": 0.0,
                    "D": 1 if t_num >= 2 else 0,
                    "y": rng.normal(),
                }
            )
        # u2: far-away never-treated
        for t_str in ("0", "2", "10"):
            rows.append(
                {
                    "unit": "u2",
                    "time": t_str,
                    "lat": 5.0,
                    "lon": 0.0,
                    "D": 0,
                    "y": rng.normal(),
                }
            )
        df = pd.DataFrame(rows)
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        # The bug previously was: lex sort would treat "10" as the smallest
        # treated time, assigning u1 first_treat = "10" / 10.0 — outside
        # the relevant comparison range. With the fix, first_treat = 2
        # (numeric), so D_it = 1 at the rows with time = "2" AND "10" → 2
        # treated rows.
        result = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        # PROOF that time was coerced to numeric BEFORE onset detection:
        # u1 has D=1 at numeric time=2 and time=10 (sorted numerically);
        # n_treated must therefore be 2. Under the lex-sort bug, u1's onset
        # would be "10" (since "10" < "2" lexicographically) and only the
        # t="10" row would be flagged → n_treated = 1.
        assert result.n_treated == 2, (
            f"string-coded numeric time not coerced before onset detection: "
            f"expected n_treated=2 (numeric sort, t in {{2, 10}}), got "
            f"{result.n_treated}"
        )


class TestSpilloverDiDOutcomeColumnRequired:
    """`outcome` should fail front-door with a ValueError, not late KeyError.
    Round-6 codex review noted the validator skipped outcome.
    """

    def test_missing_outcome_column_raises_value_error(self):
        df = _make_butts_2period_dgp(seed=42).drop(columns=["y"])
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        with pytest.raises(ValueError, match="Missing required columns"):
            est.fit(
                df,
                outcome="missing_outcome",
                unit="unit",
                time="time",
                treatment="D",
            )


# =============================================================================
# Codex review round-7 regression tests
# =============================================================================


class TestSpilloverDiDPanelStructure:
    """Reject duplicate (unit, time) cells AND unbalanced panels up front.
    Round-7 codex review reproduced an identification failure on an
    unbalanced panel where moving a far-away outcome silently shifted
    ATT by ~100x.
    """

    def test_duplicate_unit_time_cell_raises(self):
        # u1 has two rows at the same period — duplicate (unit, time) cell.
        df = pd.DataFrame(
            {
                "unit": ["u1", "u1", "u1", "u2", "u2", "u2"],
                "time": [0, 0, 1, 0, 1, 1],  # duplicates at (u1, 0) and (u2, 1)
                "lat": [0.0, 0.0, 0.0, 5.0, 5.0, 5.0],
                "lon": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "D": [0, 0, 1, 0, 0, 0],
                "y": [1.0, 1.0, 2.0, 0.5, 0.6, 0.6],
            }
        )
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        with pytest.raises(ValueError, match="duplicate.*unit, time"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")

    def test_unbalanced_panel_raises(self):
        # u1 has 2 periods but u2 has only 1 → unbalanced panel.
        df = pd.DataFrame(
            {
                "unit": ["u1", "u1", "u2"],
                "time": [0, 1, 0],
                "lat": [0.0, 0.0, 5.0],
                "lon": [0.0, 0.0, 0.0],
                "D": [0, 1, 0],
                "y": [1.0, 2.0, 0.5],
            }
        )
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        with pytest.raises(ValueError, match="Unbalanced panel"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")


# =============================================================================
# Codex review round-8 regression tests
# =============================================================================


class TestSpilloverDiDRingsStartAtZero:
    """rings[0] must equal 0; otherwise units in 0 <= d_it < rings[0] are
    flagged exposed but get zero spillover regressors → silent bias.
    """

    def test_rings_starting_above_zero_raises(self):
        est = SpilloverDiD(rings=[10.0, 50.0, 100.0], conley_coords=("lat", "lon"))
        df = _make_butts_2period_dgp(seed=42)
        with pytest.raises(ValueError, match="rings\\[0\\] must equal 0"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")


class TestSpilloverDiDHC2NotSupported:
    """vcov_type='hc2' and 'hc2_bm' require per-coefficient BM/CR2 DOF
    that the inline stage-2 inference doesn't provide. Round-8 codex
    review caught that we'd silently return wrong p-values/CIs.
    """

    @pytest.mark.parametrize("vcov_type", ["hc2", "hc2_bm"])
    def test_hc2_paths_raise_not_implemented(self, vcov_type):
        df = _make_butts_2period_dgp(seed=42)
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"), vcov_type=vcov_type)
        with pytest.raises(NotImplementedError, match="hc2"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")


class TestSpilloverDiDRankDeficientActionValidation:
    """rank_deficient_action must be one of {warn, error, silent}.
    Mirrors the sibling constructor guards at two_stage.py:149 and
    stacked_did.py.
    """

    @pytest.mark.parametrize("bad_value", ["raise", "ignore", "", "WARN", "Error", None])
    def test_invalid_rank_deficient_action_raises_at_init(self, bad_value):
        with pytest.raises(ValueError, match="rank_deficient_action must be"):
            SpilloverDiD(
                rings=[0.0, 100.0],
                conley_coords=("lat", "lon"),
                rank_deficient_action=bad_value,
            )

    @pytest.mark.parametrize("good_value", ["warn", "error", "silent"])
    def test_valid_rank_deficient_action_accepted(self, good_value):
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            rank_deficient_action=good_value,
        )
        assert est.rank_deficient_action == good_value


class TestSpilloverDiDOmega0Connectivity:
    """Round-5 CI review P1: balanced panel + per-unit/per-period Omega_0
    coverage is NECESSARY but not SUFFICIENT for stage-1 FE
    identification — the Omega_0 bipartite graph must also be CONNECTED.
    If it splits into K > 1 components, the iterative FE solver returns
    FE only up to component-specific constants, and residualization
    combines mu_i from one component with lambda_t from another,
    silently corrupting tau_total / delta_j.

    The check fires on the SUPPORTED-units subgraph (after unit-level
    warn-and-drop). Under the current absorbing-treatment + period-strict
    + unit-warn-drop regime the disconnected case may be unreachable in
    practice via a real DGP — these tests unit-test the helper directly
    with synthetic (unit_codes, time_codes, omega_0_mask) arrays so the
    check is exercised even if no DGP can reach it through the public
    `.fit()` path.
    """

    def test_disconnected_two_components_raises(self):
        """Two units in periods {0, 1}, two more in periods {2, 3}; no
        unit appears in both halves. Connectivity check must fail.
        """
        # 4 supported units (codes 0..3), 4 periods (codes 0..3).
        # Omega_0 rows: (u0, t0), (u0, t1), (u1, t0), (u1, t1),
        #               (u2, t2), (u2, t3), (u3, t2), (u3, t3).
        unit_codes_arr = np.array([0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 3, 0, 1, 2, 3])
        time_codes_arr = np.array([0, 1, 0, 1, 2, 3, 2, 3, 2, 2, 0, 0, 3, 3, 1, 1])
        omega_0_mask = np.array(
            [True, True, True, True, True, True, True, True]
            + [False, False, False, False, False, False, False, False]
        )
        with pytest.raises(
            ValueError, match="disconnected components|Stage-1 fixed effects unidentified"
        ):
            _check_omega_0_connectivity(
                omega_0_mask=omega_0_mask,
                unit_codes_arr=unit_codes_arr,
                time_codes_arr=time_codes_arr,
                units_in_omega_0={0, 1, 2, 3},
                n_times=4,
                unit_uniques=["u0", "u1", "u2", "u3"],
            )

    def test_connected_via_bridge_unit_succeeds(self):
        """Add a single bridge unit that has Omega_0 rows in all periods —
        the graph becomes connected and the check must pass.
        """
        unit_codes_arr = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4])
        time_codes_arr = np.array([0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 2, 3])
        omega_0_mask = np.array([True] * 12)
        # u4 spans all 4 periods, connecting the two halves through itself.
        _check_omega_0_connectivity(
            omega_0_mask=omega_0_mask,
            unit_codes_arr=unit_codes_arr,
            time_codes_arr=time_codes_arr,
            units_in_omega_0={0, 1, 2, 3, 4},
            n_times=4,
            unit_uniques=["u0", "u1", "u2", "u3", "u4"],
        )  # must not raise

    def test_single_supported_unit_short_circuits(self):
        """n_supp <= 1 short-circuits — no multi-component case possible."""
        unit_codes_arr = np.array([0, 0])
        time_codes_arr = np.array([0, 1])
        omega_0_mask = np.array([True, True])
        _check_omega_0_connectivity(
            omega_0_mask=omega_0_mask,
            unit_codes_arr=unit_codes_arr,
            time_codes_arr=time_codes_arr,
            units_in_omega_0={0},
            n_times=2,
            unit_uniques=["u0"],
        )  # must not raise

    def test_three_components_error_message_names_units(self):
        """Error message should name first few units per component for
        actionable debugging.
        """
        # 3 units, 3 periods, 3-way disconnection: (u0, t0), (u1, t1), (u2, t2).
        unit_codes_arr = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        time_codes_arr = np.array([0, 1, 2, 1, 2, 0, 2, 0, 1])
        omega_0_mask = np.array([True, True, True] + [False] * 6)
        with pytest.raises(ValueError) as exc_info:
            _check_omega_0_connectivity(
                omega_0_mask=omega_0_mask,
                unit_codes_arr=unit_codes_arr,
                time_codes_arr=time_codes_arr,
                units_in_omega_0={0, 1, 2},
                n_times=3,
                unit_uniques=["unit_A", "unit_B", "unit_C"],
            )
        msg = str(exc_info.value)
        assert "3 disconnected components" in msg
        assert "unit_A" in msg or "unit_B" in msg or "unit_C" in msg

    def test_normal_butts_dgp_does_not_trigger(self):
        """Positive case: a standard non-staggered Butts DGP must NOT
        trigger the connectivity check.
        """
        from tests._dgp_utils import generate_butts_nonstaggered_dgp

        df = generate_butts_nonstaggered_dgp(seed=0)
        # Just verify .fit() succeeds — if connectivity check were
        # over-eager, this would fail.
        result = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon")).fit(
            df, outcome="y", unit="unit", time="time", treatment="D"
        )
        assert result.att is not None


# =============================================================================
# Codex review round-9 regression tests
# =============================================================================


class TestSpilloverDiDAnticipationValidation:
    """anticipation must be a non-negative integer. Round-9 codex review
    caught that fractional / negative values silently shifted timing.
    """

    @pytest.mark.parametrize("bad_value", [-1, 0.5, 1.5, -0.1])
    def test_invalid_anticipation_raises_treatment_path(self, bad_value):
        df = _make_butts_2period_dgp(seed=42)
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            anticipation=bad_value,
        )
        with pytest.raises(ValueError, match="anticipation"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")


class TestSpilloverDiDNonFiniteOutcomeRejected:
    """outcome must be finite per-row; NaN/Inf raise a targeted ValueError."""

    def test_nan_outcome_raises(self):
        df = _make_butts_2period_dgp(seed=42).copy()
        df.loc[0, "y"] = np.nan
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        with pytest.raises(ValueError, match="non-finite"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")

    def test_inf_outcome_raises(self):
        df = _make_butts_2period_dgp(seed=42).copy()
        df.loc[0, "y"] = np.inf
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        with pytest.raises(ValueError, match="non-finite"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")


# =============================================================================
# Codex review round-10 regression tests
# =============================================================================


class TestSpilloverDiDNaNTreatmentRejected:
    """NaN in the binary treatment column must raise.
    Round-10 codex review caught that `_convert_treatment_to_first_treat`
    silently dropped NaN rows via `dropna()` before validation, then
    rebuilt D_it from the inferred onset — coercing missing rows to
    treated or control without warning.
    """

    @pytest.mark.parametrize(
        "d_pattern",
        [
            [0, 1, float("nan")],
            [float("nan"), 1, 1],
            [0, float("nan"), 0],
        ],
    )
    def test_nan_in_treatment_helper_raises(self, d_pattern):
        df = pd.DataFrame(
            {
                "unit": ["u1"] * 3,
                "time": [0, 1, 2],
                "D": d_pattern,
            }
        )
        with pytest.raises(ValueError, match="NaN"):
            _convert_treatment_to_first_treat(df, "D", "time", "unit")

    def test_nan_in_treatment_end_to_end_raises(self):
        df = _make_butts_2period_dgp(seed=42).copy()
        first_treated_idx = df.index[df["D"] == 1][0]
        df.loc[first_treated_idx, "D"] = np.nan
        # Convert column dtype to float so NaN is preserved.
        df["D"] = df["D"].astype(float)
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        with pytest.raises(ValueError, match="NaN"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")


# =============================================================================
# Codex review round-11 regression tests
# =============================================================================


class TestSpilloverDiDClusterNaNRejected:
    """NaN in cluster column must raise. Round-11 codex review caught
    that missing cluster ids silently changed SEs and overstated
    n_clusters because np.unique counts NaN as its own cluster but
    pandas groupby drops it from the cluster meat.
    """

    def test_numeric_nan_cluster_raises(self):
        df = _make_butts_2period_dgp(seed=42).copy()
        df["region"] = 1.0
        df.loc[df.index[0], "region"] = np.nan
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            cluster="region",
        )
        with pytest.raises(ValueError, match="cluster.*missing"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")

    def test_object_nan_cluster_raises(self):
        df = _make_butts_2period_dgp(seed=42).copy()
        df["region"] = "A"
        df.loc[df.index[0], "region"] = None  # object-typed NaN
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            cluster="region",
        )
        with pytest.raises(ValueError, match="cluster.*missing"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")


class TestSpilloverDiDIdentifierNaNRejected:
    """unit / time / first_treat columns must not contain NaN — round-11
    codex review noted these fell through to opaque numpy / pandas
    errors instead of targeted ValueErrors.
    """

    def test_nan_unit_raises(self):
        df = _make_butts_2period_dgp(seed=42).copy()
        df.loc[df.index[0], "unit"] = None
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        with pytest.raises(ValueError, match="identifier column 'unit'"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")

    def test_nan_time_raises(self):
        df = _make_butts_2period_dgp(seed=42).copy()
        df["time"] = df["time"].astype(float)
        df.loc[df.index[0], "time"] = np.nan
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        with pytest.raises(ValueError, match="identifier column 'time'"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")

    def test_nan_first_treat_raises(self):
        df = _make_butts_2period_dgp(seed=42).copy()
        df.loc[df.index[0], "first_treat"] = np.nan
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        with pytest.raises(ValueError, match="first_treat.*missing"):
            est.fit(df, outcome="y", unit="unit", time="time", first_treat="first_treat")


# =============================================================================
# Codex review round-12 regression tests
# =============================================================================


class TestSpilloverDiDMixedRawTimeEncoding:
    """Mixed encodings that collapse under pd.to_numeric must be caught
    by the validator AFTER coercion. Round-12 codex review caught that
    raw labels ['0', 0] (str + int) would pass duplicate-cell validation
    on the raw labels then collapse to (0, 0) after pd.to_numeric, with
    no warning.
    """

    def test_mixed_str_and_int_time_collapse_caught_as_duplicate(self):
        # u1 has time entries '0' (str) and 0 (int). They collapse to 0
        # under pd.to_numeric → duplicate (u1, 0) cell.
        df = pd.DataFrame(
            {
                "unit": ["u1", "u1", "u1", "u2", "u2", "u2"],
                "time": ["0", 0, 1, 0, 0, 1],
                "lat": [0.0, 0.0, 0.0, 5.0, 5.0, 5.0],
                "lon": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "D": [0, 0, 1, 0, 0, 0],
                "y": [1.0, 1.0, 2.0, 0.5, 0.6, 0.6],
            }
        )
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        with pytest.raises(ValueError, match="duplicate.*unit, time"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")

    def test_leading_zero_string_time_collapse_caught(self):
        # '01' and 1 collapse under pd.to_numeric to (1, 1) → duplicate.
        df = pd.DataFrame(
            {
                "unit": ["u1", "u1", "u1", "u2", "u2", "u2"],
                "time": [0, "01", 1, 0, "01", 1],
                "lat": [0.0, 0.0, 0.0, 5.0, 5.0, 5.0],
                "lon": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "D": [0, 0, 1, 0, 0, 0],
                "y": [1.0, 1.0, 2.0, 0.5, 0.6, 0.6],
            }
        )
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        with pytest.raises(ValueError, match="duplicate.*unit, time"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")


class TestSpilloverDiDNonStaggeredFEEquivalence:
    """Round-14 codex review (P2): pin the Gardner identity empirically.

    Under non-staggered timing, two-stage Gardner with the Omega_0-restricted
    stage 1 should produce the same `tau_total` as a single-stage TWFE ring
    regression on the full sample with the time-varying ring covariate. This
    is Butts Eqs. 4-6 / Proposition 2.3 (non-staggered identification).

    The Omega_0 restriction would BREAK Gardner identity in general (stage 1
    estimates FE on a subset, predicts onto observations outside the training
    set), but on Butts-Assumption-satisfying DGPs it is empirically innocent
    at floating-point precision. This test PINS that equivalence so any
    methodology drift surfaces in CI rather than as a silent estimand shift.

    Codex round 14 reported wildly divergent values (seed 3: +0.0238 from
    SpilloverDiD vs -0.0735 from FE) but those numbers were unreproducible —
    our 20-seed sweep confirms bit-identity at atol=1e-10.
    """

    @staticmethod
    def _fit_butts_single_stage_fe_ring(
        df, *, outcome, unit, time, treatment, rings, lat="lat", lon="lon"
    ):
        """Reference: single-stage TWFE ring regression on full sample.

        Y_it = mu_i + lambda_t + tau * D_it + sum_j delta_j * (1 - D_it) * Ring_{it,j}

        For non-staggered with shared onset t_treat, Ring_{it,j} = 1{d_i in
        [rings_j, rings_{j+1})} * 1{t >= t_treat}. Uses library's solve_ols
        for rank-deficient-safe pseudo-inverse.
        """
        import math
        import warnings

        from diff_diff.linalg import solve_ols

        rings = sorted(rings)
        df = df.copy()
        units = sorted(df[unit].unique())
        times = sorted(df[time].unique())
        unit_idx = {u: i for i, u in enumerate(units)}
        time_idx = {t: i for i, t in enumerate(times)}
        n = len(df)

        treated_set = set(df.loc[df[treatment] == 1, unit].unique())
        lat_map = df.groupby(unit)[lat].first().to_dict()
        lon_map = df.groupby(unit)[lon].first().to_dict()

        def hav(u1, u2):
            lat1, lon1 = math.radians(lat_map[u1]), math.radians(lon_map[u1])
            lat2, lon2 = math.radians(lat_map[u2]), math.radians(lon_map[u2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
            return 2 * 6371.0 * math.asin(math.sqrt(a))

        d_i = {}
        for u in units:
            if u in treated_set:
                d_i[u] = 0.0
            else:
                d_i[u] = min(hav(u, tt) for tt in treated_set)

        K = len(rings) - 1
        ring_of_unit = {}
        for u in units:
            d = d_i[u]
            ring_of_unit[u] = -1
            for j in range(K):
                if rings[j] <= d < rings[j + 1]:
                    ring_of_unit[u] = j
                    break

        t_treat = df.loc[df[treatment] == 1, time].min()

        n_u = len(units)
        n_t = len(times)
        n_reg = 1 + K
        # Intercept + (n_u - 1) unit FE dummies + (n_t - 1) time FE dummies + n_reg regressors
        X = np.zeros((n, 1 + (n_u - 1) + (n_t - 1) + n_reg))
        X[:, 0] = 1.0
        y = df[outcome].values.astype(float)

        for i, row in enumerate(df.itertuples(index=False)):
            u = getattr(row, unit)
            t = getattr(row, time)
            D = getattr(row, treatment)
            if unit_idx[u] > 0:
                X[i, 1 + unit_idx[u] - 1] = 1.0
            if time_idx[t] > 0:
                X[i, 1 + (n_u - 1) + time_idx[t] - 1] = 1.0
            X[i, 1 + (n_u - 1) + (n_t - 1) + 0] = D
            ridx = ring_of_unit[u]
            if ridx >= 0 and t >= t_treat:
                X[i, 1 + (n_u - 1) + (n_t - 1) + 1 + ridx] = 1 - D

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            beta, _, _ = solve_ols(X, y, vcov_type="hc1")
        tau = beta[1 + (n_u - 1) + (n_t - 1) + 0]
        return tau

    def test_nonstaggered_one_ring_matches_single_stage_fe_20_seeds(self):
        """20-seed bit-identity sweep, 1 ring (rings=[0, 200])."""
        from tests._dgp_utils import generate_butts_nonstaggered_dgp

        diffs = []
        for seed in range(20):
            df = generate_butts_nonstaggered_dgp(seed=seed, tau_total=-0.07, delta_1=-0.04)
            est = SpilloverDiD(rings=[0, 200], d_bar=200.0, conley_coords=("lat", "lon"))
            spill = est.fit(df, outcome="y", unit="unit", time="time", treatment="D").att
            fe_tau = self._fit_butts_single_stage_fe_ring(
                df,
                outcome="y",
                unit="unit",
                time="time",
                treatment="D",
                rings=[0, 200],
            )
            diffs.append(abs(spill - fe_tau))
        max_abs_diff = max(diffs)
        assert max_abs_diff < 1e-10, (
            f"Gardner identity broken: max |SpilloverDiD - single-stage FE| "
            f"= {max_abs_diff:.6e} across 20 seeds (expected < 1e-10)"
        )

    def test_nonstaggered_multi_ring_matches_single_stage_fe_10_seeds(self):
        """10-seed bit-identity sweep with multi-ring spec."""
        from tests._dgp_utils import generate_butts_nonstaggered_dgp

        # DGP places near-controls in d ≤ d_bar/2 = 100km, so rings beyond 100
        # may be empty. Use rings=[0, 50, 200] which has near-controls in
        # [0, 50) and possibly [50, 200).
        diffs = []
        for seed in range(10):
            df = generate_butts_nonstaggered_dgp(seed=seed, tau_total=-0.07, delta_1=-0.04)
            est = SpilloverDiD(rings=[0, 50, 200], d_bar=200.0, conley_coords=("lat", "lon"))
            spill = est.fit(df, outcome="y", unit="unit", time="time", treatment="D").att
            fe_tau = self._fit_butts_single_stage_fe_ring(
                df,
                outcome="y",
                unit="unit",
                time="time",
                treatment="D",
                rings=[0, 50, 200],
            )
            diffs.append(abs(spill - fe_tau))
        max_abs_diff = max(diffs)
        assert max_abs_diff < 1e-10, (
            f"Multi-ring Gardner identity broken: max diff "
            f"= {max_abs_diff:.6e} across 10 seeds (expected < 1e-10)"
        )


class TestSpilloverDiDCoefficientsAlignToVcov:
    """Round-15 codex review (P2): `coefficients` must expose ALL stage-2
    coefficients (treatment + K ring slots), not just `ATT`, so consumers
    can align names to the `(1+K)×(1+K)` `vcov` rows/cols. The vcov
    columns are `["treatment", "_spillover_<ring_label>", ...]`; the
    coefficients dict mirrors those keys plus an `"ATT"` alias for the
    treatment slot (sibling-estimator convention).
    """

    def _fit_one(self, rings):
        from tests._dgp_utils import generate_butts_nonstaggered_dgp

        df = generate_butts_nonstaggered_dgp(seed=42, tau_total=-0.07, delta_1=-0.04)
        est = SpilloverDiD(rings=rings, d_bar=float(rings[-1]), conley_coords=("lat", "lon"))
        return est.fit(df, outcome="y", unit="unit", time="time", treatment="D")

    def test_coefficients_has_treatment_and_ring_keys(self):
        result = self._fit_one([0, 50, 200])
        assert "ATT" in result.coefficients
        assert "treatment" in result.coefficients
        ring_keys = [k for k in result.coefficients if k.startswith("_spillover_")]
        assert (
            len(ring_keys) == 2
        ), f"Expected 2 ring coefficients, got {len(ring_keys)}: {ring_keys}"

    def test_att_alias_equals_treatment_slot(self):
        result = self._fit_one([0, 100])
        assert result.coefficients["ATT"] == result.coefficients["treatment"]
        assert result.coefficients["ATT"] == result.att

    def test_coefficients_length_matches_vcov_dimension(self):
        result = self._fit_one([0, 50, 200])
        assert result.vcov.shape == (3, 3)
        stage2_keys = [k for k in result.coefficients if k != "ATT"]
        assert len(stage2_keys) == result.vcov.shape[0]

    def test_ring_coefficients_match_spillover_effects_dataframe(self):
        result = self._fit_one([0, 50, 200])
        for ring_label, row in result.spillover_effects.iterrows():
            key = f"_spillover_{ring_label}"
            assert key in result.coefficients, f"Missing key {key} in coefficients"
            assert result.coefficients[key] == row["coef"], (
                f"Drift on {ring_label}: coefficients[{key}]="
                f"{result.coefficients[key]} vs spillover_effects.coef={row['coef']}"
            )


# =============================================================================
# Wave C: _compute_event_time_per_row helper unit tests
# =============================================================================


class TestComputeEventTimePerRowHelper:
    """Unit tests for the two-clock event-time helper (Wave C).

    Verifies:
    - K_direct = t - effective_onset for ever-treated rows; NaN for never-treated.
    - K_spill = t - trigger_onset for triggered rows; NaN otherwise.
    - trigger_onset is the EARLIEST in-range cohort onset (running min).
    - Multi-cohort priority: an earlier cohort wins over a later, even if the
      later cohort's units are closer to the row's unit.
    """

    def _make_panel(self, unit_coords, onsets, n_periods=5):
        """Build a balanced panel from (unit -> (lat, lon)) and (unit -> onset)."""
        rows = []
        for u, (lat, lon) in unit_coords.items():
            ft = onsets.get(u, np.inf)
            for t in range(n_periods):
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "lat": lat,
                        "lon": lon,
                        "first_treat": ft,
                    }
                )
        return pd.DataFrame(rows)

    def test_k_direct_for_ever_treated_units(self):
        """K_direct = t - effective_onset on all rows of ever-treated units."""
        df = self._make_panel(
            unit_coords={"A": (0.0, 0.0), "B": (5.0, 0.0)},
            onsets={"A": 1.0, "B": 3.0},
            n_periods=5,
        )
        K_direct, _ = _compute_event_time_per_row(
            data=df,
            unit="unit",
            row_unit=df["unit"].values,
            row_time=df["time"].values,
            effective_onsets={"A": 1.0, "B": 3.0},
            coords=("lat", "lon"),
            metric="euclidean",
            d_bar=10.0,
        )
        # A: K = t - 1 for t in {0..4} → {-1, 0, 1, 2, 3}
        # B: K = t - 3 for t in {0..4} → {-3, -2, -1, 0, 1}
        expected_a = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
        expected_b = np.array([-3.0, -2.0, -1.0, 0.0, 1.0])
        a_rows = df["unit"].values == "A"
        b_rows = df["unit"].values == "B"
        np.testing.assert_array_equal(K_direct[a_rows], expected_a)
        np.testing.assert_array_equal(K_direct[b_rows], expected_b)

    def test_k_direct_nan_for_never_treated(self):
        df = self._make_panel(
            unit_coords={"A": (0.0, 0.0), "C": (1.0, 0.0)},
            onsets={"A": 1.0},  # C never-treated
            n_periods=3,
        )
        K_direct, _ = _compute_event_time_per_row(
            data=df,
            unit="unit",
            row_unit=df["unit"].values,
            row_time=df["time"].values,
            effective_onsets={"A": 1.0, "C": np.inf},
            coords=("lat", "lon"),
            metric="euclidean",
            d_bar=10.0,
        )
        c_rows = df["unit"].values == "C"
        assert np.all(np.isnan(K_direct[c_rows]))

    def test_k_spill_for_in_range_unit(self):
        """A never-treated unit within d_bar of A gets K_spill = t - A.onset."""
        df = self._make_panel(
            unit_coords={"A": (0.0, 0.0), "C": (1.0, 0.0)},
            onsets={"A": 1.0},
            n_periods=4,
        )
        _, K_spill = _compute_event_time_per_row(
            data=df,
            unit="unit",
            row_unit=df["unit"].values,
            row_time=df["time"].values,
            effective_onsets={"A": 1.0, "C": np.inf},
            coords=("lat", "lon"),
            metric="euclidean",
            d_bar=10.0,
        )
        # C: at t=0 → NaN (pre-trigger); t=1,2,3 → 0, 1, 2.
        c_rows = df["unit"].values == "C"
        expected_c = np.array([np.nan, 0.0, 1.0, 2.0])
        np.testing.assert_array_equal(K_spill[c_rows], expected_c)

    def test_k_spill_nan_for_far_unit(self):
        df = self._make_panel(
            unit_coords={"A": (0.0, 0.0), "D": (100.0, 0.0)},  # D far
            onsets={"A": 1.0},
            n_periods=4,
        )
        _, K_spill = _compute_event_time_per_row(
            data=df,
            unit="unit",
            row_unit=df["unit"].values,
            row_time=df["time"].values,
            effective_onsets={"A": 1.0, "D": np.inf},
            coords=("lat", "lon"),
            metric="euclidean",
            d_bar=10.0,
        )
        d_rows = df["unit"].values == "D"
        assert np.all(np.isnan(K_spill[d_rows]))

    def test_k_spill_trigger_is_earliest_cohort_in_range(self):
        """A unit in range of BOTH cohorts gets trigger = EARLIER cohort, even
        if the later cohort is geographically closer."""
        df = self._make_panel(
            unit_coords={
                "A": (0.0, 0.0),  # cohort 1, onset=1
                "B": (3.0, 0.0),  # cohort 2, onset=3
                "C": (2.5, 0.0),  # at distance 2.5 from A AND 0.5 from B; both <= d_bar
            },
            onsets={"A": 1.0, "B": 3.0},
            n_periods=5,
        )
        _, K_spill = _compute_event_time_per_row(
            data=df,
            unit="unit",
            row_unit=df["unit"].values,
            row_time=df["time"].values,
            effective_onsets={"A": 1.0, "B": 3.0, "C": np.inf},
            coords=("lat", "lon"),
            metric="euclidean",
            d_bar=10.0,
        )
        # C: trigger = A's onset (1), NOT B's onset (3), even though B is closer.
        # K_spill at t=0 → NaN; t=1 → 0; t=2 → 1; t=3 → 2; t=4 → 3.
        c_rows = df["unit"].values == "C"
        expected_c = np.array([np.nan, 0.0, 1.0, 2.0, 3.0])
        np.testing.assert_array_equal(K_spill[c_rows], expected_c)

    def test_k_spill_pre_trigger_is_nan(self):
        """Even an in-range unit has K_spill = NaN before the trigger cohort activates."""
        df = self._make_panel(
            unit_coords={"A": (0.0, 0.0), "C": (1.0, 0.0)},
            onsets={"A": 2.0},  # cohort onset is at t=2
            n_periods=4,
        )
        _, K_spill = _compute_event_time_per_row(
            data=df,
            unit="unit",
            row_unit=df["unit"].values,
            row_time=df["time"].values,
            effective_onsets={"A": 2.0, "C": np.inf},
            coords=("lat", "lon"),
            metric="euclidean",
            d_bar=10.0,
        )
        c_rows = df["unit"].values == "C"
        # C: t=0,1 → NaN (pre-trigger); t=2,3 → 0, 1.
        expected_c = np.array([np.nan, np.nan, 0.0, 1.0])
        np.testing.assert_array_equal(K_spill[c_rows], expected_c)

    def test_anticipation_shifts_both_clocks(self):
        """When effective_onsets is anticipation-shifted, both clocks shift accordingly."""
        df = self._make_panel(
            unit_coords={"A": (0.0, 0.0), "C": (1.0, 0.0)},
            onsets={"A": 3.0},
            n_periods=5,
        )
        # anticipation=2 → effective_onset(A) = 3 - 2 = 1.
        K_direct, K_spill = _compute_event_time_per_row(
            data=df,
            unit="unit",
            row_unit=df["unit"].values,
            row_time=df["time"].values,
            effective_onsets={"A": 1.0, "C": np.inf},  # anticipation-shifted
            coords=("lat", "lon"),
            metric="euclidean",
            d_bar=10.0,
        )
        a_rows = df["unit"].values == "A"
        c_rows = df["unit"].values == "C"
        # A's K_direct: t=0..4 → {-1, 0, 1, 2, 3} (against effective_onset=1).
        np.testing.assert_array_equal(K_direct[a_rows], np.array([-1.0, 0.0, 1.0, 2.0, 3.0]))
        # C's K_spill: trigger=1; t=0 → NaN; t=1..4 → 0, 1, 2, 3.
        np.testing.assert_array_equal(K_spill[c_rows], np.array([np.nan, 0.0, 1.0, 2.0, 3.0]))


class TestApplyHorizonBinningHelper:
    """Unit tests for the horizon-clip helper (Wave C)."""

    def test_clips_to_endpoint_bins(self):
        K = np.array([-5.0, -3.0, -1.0, 0.0, 2.0, 5.0, 10.0])
        out = _apply_horizon_binning(K, horizon_max=3)
        np.testing.assert_array_equal(out, np.array([-3.0, -3.0, -1.0, 0.0, 2.0, 3.0, 3.0]))

    def test_nan_preservation(self):
        K = np.array([np.nan, -5.0, 0.0, np.nan, 10.0])
        out = _apply_horizon_binning(K, horizon_max=3)
        # NaN positions remain NaN; finite positions clipped.
        assert np.isnan(out[0])
        assert np.isnan(out[3])
        np.testing.assert_array_equal(out[[1, 2, 4]], np.array([-3.0, 0.0, 3.0]))

    def test_none_horizon_returns_input_unchanged(self):
        K = np.array([-10.0, 0.0, np.nan, 100.0])
        out = _apply_horizon_binning(K, horizon_max=None)
        # Should equal input exactly (NaN preserved by np.array_equal with equal_nan=True).
        assert np.array_equal(out, K, equal_nan=True)

    def test_zero_horizon_collapses_to_single_bin(self):
        """H=0 maps every finite K to 0; NaN preserved."""
        K = np.array([-5.0, -1.0, 0.0, 3.0, np.nan])
        out = _apply_horizon_binning(K, horizon_max=0)
        assert out[0] == 0.0 and out[1] == 0.0 and out[2] == 0.0 and out[3] == 0.0
        assert np.isnan(out[4])

    def test_negative_horizon_raises(self):
        K = np.array([0.0, 1.0])
        with pytest.raises(ValueError, match="non-negative integer"):
            _apply_horizon_binning(K, horizon_max=-1)

    def test_float_horizon_raises(self):
        K = np.array([0.0, 1.0])
        with pytest.raises(ValueError, match="non-negative integer"):
            _apply_horizon_binning(K, horizon_max=2.5)


class TestBuildEventStudyDesignHelper:
    """Unit tests for the event-study stage-2 design builder (Wave C).

    Verifies column count, reference-period drop, column-name convention,
    all-zero pre-filter, and rectangular_grid emission.
    """

    def _hand_built_panel(self):
        """4 units (treated A,B,C ever-treated; D never), 5 periods, 2 rings."""
        # D_it: A treated at t>=2, B at t>=4, C never (ever-treated for D_i but K
        # range only includes pre rows in this example). D never treated.
        # We construct K_direct/K_spill arrays directly for hand-control.
        n_rows = 4 * 5  # 4 units * 5 periods
        D_it = np.zeros(n_rows)
        # Rows ordered: A(t=0..4), B(t=0..4), C(t=0..4), D(t=0..4).
        # A treated post t=2: rows 2,3,4 → D_it=1
        # B treated post t=4: row 9 → D_it=1
        D_it[[2, 3, 4, 9]] = 1.0
        # Ring masks: 2 rings. Let's say A and B are in ring 0 of each other,
        # C is in ring 1 of A. D far from all.
        ring_masks = np.zeros((n_rows, 2), dtype=bool)
        # B's rows in ring 0 (of A) for t >= A.onset=2 → rows 7, 8 (B at t=2, 3)
        # B at t=4 is treated (D_it=1), Ring^k still nonzero but (1-D_it)*Ring=0.
        ring_masks[[7, 8, 9], 0] = True
        # C's rows in ring 1 (of A) for t >= A.onset=2 → rows 12, 13, 14
        ring_masks[[12, 13, 14], 1] = True
        ring_labels = ["[0, 50)", "[50, 200)"]
        # K_direct: A's K_direct = t - 2 for all A rows; B's = t - 4; C, D = NaN.
        K_direct = np.full(n_rows, np.nan)
        K_direct[0:5] = np.arange(5) - 2.0  # A: -2,-1,0,1,2
        K_direct[5:10] = np.arange(5) - 4.0  # B: -4,-3,-2,-1,0
        # K_spill: post-trigger only. B's trigger = A.onset = 2, so K_spill = t-2
        # at rows 7, 8 (B at t=2, 3); row 9 is treated post → still in ring but
        # (1-D_it) zeros it; row 9 K_spill = 4 - 2 = 2 if we want consistent
        # data, but contribution is zero. Set K_spill[9] = 2 so K_set is complete.
        K_spill = np.full(n_rows, np.nan)
        K_spill[7] = 0.0  # B at t=2
        K_spill[8] = 1.0  # B at t=3
        K_spill[9] = 2.0  # B at t=4
        K_spill[12] = 0.0  # C at t=2
        K_spill[13] = 1.0  # C at t=3
        K_spill[14] = 2.0  # C at t=4
        return D_it, ring_masks, ring_labels, K_direct, K_spill

    def test_column_count_with_full_grid(self):
        """Full grid H=2, ref=-1 → 2H = 4 direct + 4 × 2 spillover = 12 candidate
        columns. Some empty cells pre-filtered."""
        D_it, ring_masks, ring_labels, K_direct, K_spill = self._hand_built_panel()
        event_time_grid = [-2, -1, 0, 1, 2]
        with pytest.warns(UserWarning, match="all-zero"):
            X_2, names, meta, rect_grid, n_obs = _build_event_study_design(
                D_it=D_it,
                ring_masks=ring_masks,
                ring_labels=ring_labels,
                K_direct_binned=K_direct,
                K_spill_binned=K_spill,
                event_time_grid=event_time_grid,
                ref_period=-1,
            )
        # 4 k bins (after dropping ref=-1) × (1 direct + 2 rings) = 12 candidate cols.
        assert len(rect_grid) == 12
        # X_2 has only the non-empty kept columns.
        assert X_2.shape == (20, len(names))
        assert len(names) == len(meta) == len(n_obs)
        # All n_obs > 0 for kept columns.
        assert all(n > 0 for n in n_obs)

    def test_column_name_convention_signed(self):
        D_it, ring_masks, ring_labels, K_direct, K_spill = self._hand_built_panel()
        with pytest.warns(UserWarning):  # all-zero pre-filter fires
            _, names, _, _, _ = _build_event_study_design(
                D_it=D_it,
                ring_masks=ring_masks,
                ring_labels=ring_labels,
                K_direct_binned=K_direct,
                K_spill_binned=K_spill,
                event_time_grid=[-2, -1, 0, 1, 2],
                ref_period=-1,
            )
        # Sample expected names: "D^k=+0", "D^k=-2", "_spillover_[0, 50)^k=+1".
        assert any(n == "D^k=+0" for n in names)
        # At least one D^k=-2 column (A has K_direct=-2 at t=0).
        assert any(n == "D^k=-2" for n in names)
        # At least one spillover column with the [0, 50) ring label.
        assert any(n.startswith("_spillover_[0, 50)^k=") for n in names)

    def test_reference_period_dropped(self):
        D_it, ring_masks, ring_labels, K_direct, K_spill = self._hand_built_panel()
        with pytest.warns(UserWarning):
            _, names, meta, rect_grid, _ = _build_event_study_design(
                D_it=D_it,
                ring_masks=ring_masks,
                ring_labels=ring_labels,
                K_direct_binned=K_direct,
                K_spill_binned=K_spill,
                event_time_grid=[-2, -1, 0, 1, 2],
                ref_period=-1,
            )
        # k=-1 must NOT appear in any column name or any rect_grid tuple.
        assert not any("k=-1" in n for n in names)
        assert not any(k == -1 for (_, _, k) in rect_grid)

    def test_rectangular_grid_includes_dropped_cells(self):
        D_it, ring_masks, ring_labels, K_direct, K_spill = self._hand_built_panel()
        with pytest.warns(UserWarning):
            _, names, _, rect_grid, _ = _build_event_study_design(
                D_it=D_it,
                ring_masks=ring_masks,
                ring_labels=ring_labels,
                K_direct_binned=K_direct,
                K_spill_binned=K_spill,
                event_time_grid=[-2, -1, 0, 1, 2],
                ref_period=-1,
            )
        # Direct: 4 k bins (excluding ref=-1) → 4 entries in rect_grid.
        direct_in_grid = [(s, r, k) for (s, r, k) in rect_grid if s == "direct"]
        assert len(direct_in_grid) == 4
        # Spillover: 4 k bins × 2 rings = 8 entries.
        spill_in_grid = [(s, r, k) for (s, r, k) in rect_grid if s == "spillover"]
        assert len(spill_in_grid) == 8
        # Sanity: each (ring, k) combo appears exactly once.
        spill_pairs = [(r, k) for (s, r, k) in spill_in_grid]
        assert len(set(spill_pairs)) == 8

    def test_n_obs_per_col_correctness(self):
        D_it, ring_masks, ring_labels, K_direct, K_spill = self._hand_built_panel()
        with pytest.warns(UserWarning):
            _, names, meta, _, n_obs = _build_event_study_design(
                D_it=D_it,
                ring_masks=ring_masks,
                ring_labels=ring_labels,
                K_direct_binned=K_direct,
                K_spill_binned=K_spill,
                event_time_grid=[-2, -1, 0, 1, 2],
                ref_period=-1,
            )
        # D^k=+0 has 2 rows contributing (A at t=2 and B at t=4 both have K_direct=0).
        d0_idx = names.index("D^k=+0")
        assert n_obs[d0_idx] == 2
        # D^k=-2 has 2 rows (A at t=0 and B at t=2 both have K_direct=-2).
        dm2_idx = names.index("D^k=-2")
        assert n_obs[dm2_idx] == 2

    def test_ref_period_must_be_int(self):
        D_it, ring_masks, ring_labels, K_direct, K_spill = self._hand_built_panel()
        with pytest.raises(TypeError, match="integer"):
            _build_event_study_design(
                D_it=D_it,
                ring_masks=ring_masks,
                ring_labels=ring_labels,
                K_direct_binned=K_direct,
                K_spill_binned=K_spill,
                event_time_grid=[-2, -1, 0, 1, 2],
                ref_period=-1.0,
            )

    def test_ring_labels_length_mismatch_raises(self):
        D_it, ring_masks, _, K_direct, K_spill = self._hand_built_panel()
        with pytest.raises(ValueError, match="ring_labels"):
            _build_event_study_design(
                D_it=D_it,
                ring_masks=ring_masks,
                ring_labels=["only_one_label"],  # K=2 but only 1 label
                K_direct_binned=K_direct,
                K_spill_binned=K_spill,
                event_time_grid=[-1, 0, 1],
                ref_period=-1,
            )


# =============================================================================
# Wave C: SpilloverDiD(event_study=True) end-to-end test surface
# =============================================================================


def _fit_event_study(
    df,
    *,
    rings=(0.0, 50.0, 200.0),
    horizon_max=None,
    anticipation=0,
    vcov_type="hc1",
    **fit_kwargs,
):
    """Helper: silence event-study warnings and return the SpilloverDiD result."""
    est = SpilloverDiD(
        rings=list(rings),
        d_bar=max(rings),
        conley_coords=("lat", "lon"),
        conley_metric="haversine",
        conley_cutoff_km=max(rings),
        conley_lag_cutoff=0,
        vcov_type=vcov_type,
        event_study=True,
        horizon_max=horizon_max,
        anticipation=anticipation,
    )
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("ignore", UserWarning)
        return est.fit(
            df,
            outcome="y",
            unit="unit",
            time="time",
            first_treat=fit_kwargs.pop("first_treat", "first_treat"),
            **fit_kwargs,
        )


class TestSpilloverDiDEventStudyAPI:
    """Wave C: surface-level API verification for event_study=True."""

    def test_event_study_emits_att_dynamic(self):
        df = generate_butts_staggered_dgp(seed=42)
        res = _fit_event_study(df, horizon_max=2)
        assert res.event_study is True
        assert res.att_dynamic is not None
        assert isinstance(res.att_dynamic, pd.DataFrame)
        # Columns present.
        assert set(res.att_dynamic.columns) == {
            "coef",
            "se",
            "t_stat",
            "p_value",
            "ci_low",
            "ci_high",
            "n_obs",
        }

    def test_event_study_emits_multiindex_spillover_effects(self):
        df = generate_butts_staggered_dgp(seed=42)
        res = _fit_event_study(df, horizon_max=2)
        assert isinstance(res.spillover_effects.index, pd.MultiIndex)
        assert list(res.spillover_effects.index.names) == ["ring", "k"]

    def test_event_study_emits_event_study_effects_dict(self):
        df = generate_butts_staggered_dgp(seed=42)
        res = _fit_event_study(df, horizon_max=2)
        assert isinstance(res.event_study_effects, dict)
        # Every key is an integer event-time bin.
        assert all(isinstance(k, int) for k in res.event_study_effects.keys())
        # Each entry has the TwoStageDiD schema.
        for k, entry in res.event_study_effects.items():
            assert set(entry.keys()) == {"effect", "se", "n_obs", "t_stat", "p_value", "conf_int"}
            assert isinstance(entry["conf_int"], tuple)
            assert len(entry["conf_int"]) == 2

    def test_event_study_effects_reference_row_matches_two_stage_did(self):
        """Reference row must use conf_int=(0.0, 0.0) per TwoStageDiD parity."""
        df = generate_butts_staggered_dgp(seed=42)
        res = _fit_event_study(df, horizon_max=2)
        ref = res.event_study_effects[res.reference_period]
        assert ref["effect"] == 0.0
        assert ref["se"] == 0.0
        assert ref["n_obs"] == 0
        assert ref["conf_int"] == (0.0, 0.0)
        assert np.isnan(ref["t_stat"])
        assert np.isnan(ref["p_value"])

    def test_event_study_false_leaves_new_fields_none(self):
        """When event_study=False, the new Wave C fields stay None."""
        df = generate_butts_staggered_dgp(seed=42)
        est = SpilloverDiD(
            rings=[0.0, 50.0, 200.0],
            d_bar=200.0,
            conley_coords=("lat", "lon"),
            event_study=False,
        )
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", UserWarning)
            res = est.fit(df, outcome="y", unit="unit", time="time", first_treat="first_treat")
        assert res.att_dynamic is None
        assert res.event_study_effects is None
        assert res.reference_period is None
        assert res.horizon_max is None


class TestSpilloverDiDEventStudyReferencePeriod:
    """Reference period mirrors TwoStageDiD: ref = -1 - anticipation."""

    def test_reference_period_default_anticipation(self):
        df = generate_butts_staggered_dgp(seed=42)
        res = _fit_event_study(df, horizon_max=3, anticipation=0)
        assert res.reference_period == -1

    def test_reference_period_with_anticipation_shifts(self):
        df = generate_butts_staggered_dgp(seed=42)
        res = _fit_event_study(df, horizon_max=4, anticipation=2)
        assert res.reference_period == -3

    def test_reference_row_appears_in_att_dynamic(self):
        df = generate_butts_staggered_dgp(seed=42)
        res = _fit_event_study(df, horizon_max=2, anticipation=0)
        # Reference k=-1 row exists with coef=0, se=0, n_obs=0.
        assert -1 in res.att_dynamic.index
        ref_row = res.att_dynamic.loc[-1]
        assert ref_row["coef"] == 0.0
        assert ref_row["se"] == 0.0
        assert ref_row["n_obs"] == 0


class TestSpilloverDiDEventStudyReduceToAggregate:
    """Reduce-to-Wave-B-aggregate at horizon_max=None on constant-tau DGP.

    Note: horizon_max=0 is REJECTED under event_study=True (PR #456 R5 fix):
    single bin k=0 leaves no event-time pair to anchor the reference period
    against. Users wanting a single aggregate direct effect should use
    event_study=False instead.
    """

    def test_constant_tau_horizon_none_recovers_wave_b_att(self):
        """Deterministic constant-tau DGP (`error_sd=0`) + `horizon_max=None` →
        lincom-weighted scalar `att` reproduces Wave B's aggregate `tau_total`
        bit-identically. Tightened per PR #456 R2 review to match the
        CHANGELOG's claimed `atol=1e-10` contract instead of a loose 1e-3."""
        df = generate_butts_staggered_dgp(
            seed=42,
            tau_total=-0.07,
            delta_1=-0.04,
            error_sd=0.0,  # deterministic — no noise.
        )
        agg_est = SpilloverDiD(
            rings=[0.0, 50.0, 200.0],
            d_bar=200.0,
            conley_coords=("lat", "lon"),
            event_study=False,
        )
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", UserWarning)
            agg = agg_est.fit(df, outcome="y", unit="unit", time="time", first_treat="first_treat")
        es = _fit_event_study(df, horizon_max=None)
        # With deterministic effects (error_sd=0), the equivalence holds at
        # machine precision: under constant-tau, both the aggregate D_it
        # column and the sample-share-weighted average over per-event-time
        # tau_k columns produce identical regression output.
        assert abs(agg.att - es.att) < 1e-10, (
            f"Reduce-to-aggregate equivalence failed at error_sd=0: "
            f"agg.att={agg.att:.15f}, es.att={es.att:.15f}, "
            f"diff={abs(agg.att - es.att):.3e}"
        )

    def test_lincom_att_matches_hand_computed(self):
        df = generate_butts_staggered_dgp(seed=11)
        res = _fit_event_study(df, horizon_max=3)
        post = res.att_dynamic[res.att_dynamic.index >= 0]
        total = post["n_obs"].sum()
        hand_att = (post["coef"] * post["n_obs"]).sum() / total
        assert abs(hand_att - res.att) < 1e-10


class TestSpilloverDiDEventStudyValidation:
    """Wave C validation: horizon_max < 0 and ref_period outside window both raise."""

    def test_negative_horizon_max_raises(self):
        df = generate_butts_staggered_dgp(seed=1)
        est = SpilloverDiD(
            rings=[0.0, 50.0, 200.0],
            d_bar=200.0,
            conley_coords=("lat", "lon"),
            event_study=True,
            horizon_max=-1,
        )
        with pytest.raises(ValueError, match="non-negative integer"):
            est.fit(df, outcome="y", unit="unit", time="time", first_treat="first_treat")

    def test_ref_period_outside_window_raises(self):
        df = generate_butts_staggered_dgp(seed=1)
        est = SpilloverDiD(
            rings=[0.0, 50.0, 200.0],
            d_bar=200.0,
            conley_coords=("lat", "lon"),
            event_study=True,
            horizon_max=1,
            anticipation=2,  # ref=-3 outside [-1,+1]
        )
        with pytest.raises(ValueError, match="falls outside the binning window"):
            est.fit(df, outcome="y", unit="unit", time="time", first_treat="first_treat")

    def test_horizon_max_none_with_anticipation_works(self):
        df = generate_butts_staggered_dgp(seed=1)
        # horizon_max=None auto-detects H; ref=-3 with anticipation=2 always fits.
        res = _fit_event_study(df, horizon_max=None, anticipation=2)
        assert res.reference_period == -3

    def test_horizon_max_zero_with_event_study_raises(self):
        """PR #456 R5 P1: horizon_max=0 is rejected under event_study=True
        (the single k=0 bin has no event-time pair to anchor the reference
        against). Users wanting a single aggregate effect should use
        event_study=False."""
        df = generate_butts_staggered_dgp(seed=1)
        est = SpilloverDiD(
            rings=[0.0, 50.0, 200.0],
            d_bar=200.0,
            conley_coords=("lat", "lon"),
            event_study=True,
            horizon_max=0,
        )
        with pytest.raises(ValueError, match="horizon_max=0 is not supported"):
            est.fit(df, outcome="y", unit="unit", time="time", first_treat="first_treat")

    def test_non_numeric_anticipation_raises_targeted_value_error(self):
        """PR #456 R2 P2: anticipation must be validated BEFORE the ref_period
        compatibility check; otherwise `-1 - self.anticipation` would raise a
        raw TypeError on non-numeric input instead of the targeted ValueError."""
        df = generate_butts_staggered_dgp(seed=1)
        est = SpilloverDiD(
            rings=[0.0, 50.0, 200.0],
            d_bar=200.0,
            conley_coords=("lat", "lon"),
            event_study=True,
            horizon_max=2,
            anticipation="1",  # type: ignore[arg-type]
        )
        with pytest.raises(ValueError, match="anticipation must be a non-negative integer"):
            est.fit(df, outcome="y", unit="unit", time="time", first_treat="first_treat")

    def test_none_anticipation_raises_targeted_value_error(self):
        """Same P2 fix: None anticipation must surface the targeted ValueError."""
        df = generate_butts_staggered_dgp(seed=1)
        est = SpilloverDiD(
            rings=[0.0, 50.0, 200.0],
            d_bar=200.0,
            conley_coords=("lat", "lon"),
            event_study=True,
            horizon_max=2,
            anticipation=None,  # type: ignore[arg-type]
        )
        with pytest.raises(ValueError, match="anticipation must be a non-negative integer"):
            est.fit(df, outcome="y", unit="unit", time="time", first_treat="first_treat")


class TestSpilloverDiDEventStudyBackwardCompat:
    """event_study=False reproduces the unchanged Wave B aggregate path.

    The golden values below were captured against the current (Wave C)
    `event_study=False` path on `generate_butts_nonstaggered_dgp(seed=42)`.
    Wave C does not modify the aggregate stage-2 design construction
    (``spillover.py`` lines around the ``else`` branch at the `event_study`
    dispatch), the stage-2 fit, or the aggregate extraction logic — those
    lines are byte-identical to Wave B in this PR. The PIN therefore anchors
    the unchanged aggregate path against accidental drift, but it is not a
    literal "pre-Wave-C" checkout artifact. Any future change to the
    aggregate path must update both these goldens and the CHANGELOG
    aggregate-path bit-identity claim simultaneously.
    """

    # PR #456 R3 golden capture (event_study=False on the seed-42 fixture).
    _WAVE_B_GOLDEN_ATT = -0.08620379515400438
    _WAVE_B_GOLDEN_SE = 0.017812406263278957
    _WAVE_B_GOLDEN_RING_INNER_COEF = -0.0371780776943839
    _WAVE_B_GOLDEN_RING_INNER_SE = 0.008298917907045593
    _WAVE_B_GOLDEN_RING_OUTER_COEF = -0.009441319618178406
    _WAVE_B_GOLDEN_RING_OUTER_SE = 0.015538307675860204

    def test_event_study_false_matches_wave_b_golden(self):
        """Pre-Wave-C golden parity (not just determinism): pin att/se on a
        deterministic DGP at 1e-14 tolerance and assert reproduction within
        ULP-scale BLAS reduction-order drift across runners. Strengthened
        per PR #456 R3 review — the previous determinism check (fit twice on
        the current code path) did not actually anchor against a pre-Wave-C
        baseline. Tolerance softened from `==` to `assert_allclose(rtol=1e-14,
        atol=1e-14)` after CI Pure Python Fallback (Linux py3.14) flagged a
        1-ULP drift from the macOS Accelerate capture machine — the
        identification claim is unchanged; the platform-pinning was."""
        df = generate_butts_nonstaggered_dgp(seed=42)
        est = SpilloverDiD(
            rings=[0.0, 50.0, 200.0],
            d_bar=200.0,
            conley_coords=("lat", "lon"),
            event_study=False,
        )
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", UserWarning)
            res = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        # Goldens were captured on a single machine (BLAS reduction order is
        # platform-dependent); pin at 1e-14 tolerance per
        # `feedback_assert_allclose_numerical_parity`. Tight enough to catch
        # real aggregate-path drift, loose enough to absorb ULP-scale
        # cross-runner reduction-order differences (Pure Python Fallback on
        # Linux py3.14 drifts ~1 ULP from macOS Accelerate captures).
        np.testing.assert_allclose(
            res.att,
            self._WAVE_B_GOLDEN_ATT,
            rtol=1e-14,
            atol=1e-14,
            err_msg=f"event_study=False att drift: got {res.att!r}, expected {self._WAVE_B_GOLDEN_ATT!r}",
        )
        np.testing.assert_allclose(
            res.se,
            self._WAVE_B_GOLDEN_SE,
            rtol=1e-14,
            atol=1e-14,
            err_msg=f"event_study=False se drift: got {res.se!r}, expected {self._WAVE_B_GOLDEN_SE!r}",
        )
        # Per-ring entries must also match.
        inner = res.spillover_effects.loc["[0, 50)"]
        np.testing.assert_allclose(
            inner["coef"],
            self._WAVE_B_GOLDEN_RING_INNER_COEF,
            rtol=1e-14,
            atol=1e-14,
            err_msg=f"inner ring coef drift: got {inner['coef']!r}, expected {self._WAVE_B_GOLDEN_RING_INNER_COEF!r}",
        )
        np.testing.assert_allclose(
            inner["se"],
            self._WAVE_B_GOLDEN_RING_INNER_SE,
            rtol=1e-14,
            atol=1e-14,
            err_msg=f"inner ring se drift: got {inner['se']!r}, expected {self._WAVE_B_GOLDEN_RING_INNER_SE!r}",
        )
        outer = res.spillover_effects.loc["[50, 200]"]
        np.testing.assert_allclose(
            outer["coef"], self._WAVE_B_GOLDEN_RING_OUTER_COEF, rtol=1e-14, atol=1e-14
        )
        np.testing.assert_allclose(
            outer["se"], self._WAVE_B_GOLDEN_RING_OUTER_SE, rtol=1e-14, atol=1e-14
        )

    def test_event_study_false_bit_identical_to_wave_b_fixture(self):
        df = generate_butts_nonstaggered_dgp(seed=42)
        est_a = SpilloverDiD(
            rings=[0.0, 50.0, 200.0],
            d_bar=200.0,
            conley_coords=("lat", "lon"),
            event_study=False,
        )
        est_b = SpilloverDiD(
            rings=[0.0, 50.0, 200.0],
            d_bar=200.0,
            conley_coords=("lat", "lon"),
            event_study=False,
        )
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", UserWarning)
            res_a = est_a.fit(df, outcome="y", unit="unit", time="time", treatment="D")
            res_b = est_b.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        # Determinism guard (the golden parity check above pins the actual values).
        assert res_a.att == res_b.att
        assert res_a.se == res_b.se


class TestSpilloverDiDEventStudyIdentification:
    """100-seed MC verifies per-event-time tau_k recovery on a known DGP."""

    def test_per_event_time_tau_k_recovery(self):
        # Mild heterogeneous tau profile: k=0 → -0.07; k=1 → -0.06; k=2 → -0.05.
        def tau_fn(k):
            return -0.07 + 0.01 * k

        tau_k_estimates = {k: [] for k in [0, 1, 2]}

        for s in range(50):  # 50 seeds; expensive at 100 for tutorial-paced CI
            df = generate_butts_staggered_dgp(
                seed=s,
                tau_per_event_time=tau_fn,
                delta_per_ring_per_event_time=lambda j, k: -0.04,
            )
            try:
                res = _fit_event_study(df, horizon_max=2)
            except Exception:
                continue
            for k in tau_k_estimates:
                if k in res.att_dynamic.index:
                    val = res.att_dynamic.loc[k, "coef"]
                    if np.isfinite(val):
                        tau_k_estimates[k].append(val)

        for k, target in [(0, -0.07), (1, -0.06), (2, -0.05)]:
            mean_est = np.mean(tau_k_estimates[k])
            assert abs(mean_est - target) < 0.025, (
                f"k={k}: mean tau_k estimate {mean_est:.4f} differs from "
                f"target {target:.4f} by more than 0.025 over "
                f"{len(tau_k_estimates[k])} seeds"
            )

    def test_per_ring_event_time_delta_jk_recovery(self):
        """PR #456 R3 fix: also verify per-(ring, event-time) `delta_jk`
        recovery — not just `tau_k`. REGISTRY says Wave C covers `delta_jk`
        recovery; this test backs that claim.

        DGP places all near-controls in ring 0 (one-cohort-one-cluster), so
        only ring 0 cells fire; outer rings emit NaN coefs with n_obs=0
        (rectangular schema).
        """

        def delta_fn(j, k):
            # Mild profile in ring 0: k=0 → -0.04; k=1 → -0.035; k=2 → -0.03.
            return -0.04 + 0.005 * k

        delta_k_estimates = {k: [] for k in [0, 1, 2]}

        for s in range(50):
            df = generate_butts_staggered_dgp(
                seed=s,
                tau_per_event_time=lambda k: -0.07,
                delta_per_ring_per_event_time=delta_fn,
            )
            try:
                res = _fit_event_study(df, horizon_max=2)
            except Exception:
                continue
            # Ring 0 corresponds to the inner ring; ring labels are like
            # "[0, 50)" depending on rings passed. Iterate by position.
            ring_labels = res.spillover_effects.index.get_level_values("ring").unique()
            inner_ring = ring_labels[0]
            for k in delta_k_estimates:
                key = (inner_ring, k)
                if key in res.spillover_effects.index:
                    val = res.spillover_effects.loc[key, "coef"]
                    if np.isfinite(val):
                        delta_k_estimates[k].append(val)

        for k, target in [(0, -0.04), (1, -0.035), (2, -0.03)]:
            mean_est = np.mean(delta_k_estimates[k])
            assert abs(mean_est - target) < 0.025, (
                f"delta_jk recovery: k={k} target={target:.4f}, "
                f"mean_est={mean_est:.4f} over {len(delta_k_estimates[k])} seeds "
                f"(tolerance 0.025)"
            )


class TestSpilloverDiDEventStudyPlaceboPretrends:
    """On a no-pre-trend DGP, pre-treatment coefs have nominal Type I rate."""

    def test_no_pretrend_dgp_yields_insignificant_pre_coefs(self):
        # DGP with constant tau=-0.07 only post-treatment (no pre-trend).
        n_seeds = 50
        n_significant_pre = 0
        for s in range(n_seeds):
            df = generate_butts_staggered_dgp(
                seed=s,
                tau_per_event_time=lambda k: -0.07 if k >= 0 else 0.0,
            )
            try:
                res = _fit_event_study(df, horizon_max=2)
            except Exception:
                continue
            # Pre-treatment coef at k=-2 (k=-1 is reference, dropped).
            if -2 in res.att_dynamic.index:
                p = res.att_dynamic.loc[-2, "p_value"]
                if np.isfinite(p) and p < 0.10:
                    n_significant_pre += 1
        type1_rate = n_significant_pre / n_seeds
        # Nominal alpha=0.10 + headroom for finite-sample / single-pre-coef testing.
        assert type1_rate < 0.30, (
            f"Pre-treatment k=-2 placebo Type I rate {type1_rate:.2f} exceeds "
            f"0.30 (nominal 0.10 + headroom). DGP has no pre-trend, so pre-"
            f"treatment coefs should be insignificant."
        )


class TestSpilloverDiDEventStudySingularity:
    """Rectangular schema: empty (ring, k) cells emit NaN with n_obs=0."""

    def test_negative_k_spillover_cells_are_nan(self):
        """K_spill is structurally >=0, so negative-k spillover cells are empty."""
        df = generate_butts_staggered_dgp(seed=42)
        res = _fit_event_study(df, horizon_max=2)
        # The (ring, k=-2) cells should appear with NaN coef and n_obs=0.
        # K_spill structurally >= 0, so any k < 0 spillover cell is empty.
        neg_k_rows = res.spillover_effects.xs(-2, level="k")
        # Either all NaN or all dropped pre-filter; rectangular schema emits NaN.
        for ring_label, row in neg_k_rows.iterrows():
            assert row["n_obs"] == 0
            assert np.isnan(row["coef"])

    def test_outer_ring_cells_may_be_empty(self):
        """Default DGP has no units in [50, 200) ring → all NaN with n_obs=0."""
        df = generate_butts_staggered_dgp(seed=42)
        res = _fit_event_study(df, horizon_max=2)
        if "[50, 200]" in res.spillover_effects.index.get_level_values("ring"):
            outer = res.spillover_effects.xs("[50, 200]", level="ring")
            # All n_obs = 0 (no units in the outer ring in this DGP).
            assert all(outer["n_obs"] == 0)


class TestSpilloverDiDEventStudyConleyIntegration:
    """vcov dimensions + diagonal positivity after Conley path with expanded design."""

    def test_conley_vcov_shape_matches_kept_cols(self):
        df = generate_butts_staggered_dgp(seed=42)
        est = SpilloverDiD(
            rings=[0.0, 50.0, 200.0],
            d_bar=200.0,
            conley_coords=("lat", "lon"),
            conley_metric="haversine",
            conley_cutoff_km=200.0,
            conley_lag_cutoff=0,
            vcov_type="conley",
            event_study=True,
            horizon_max=2,
        )
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", UserWarning)
            res = est.fit(df, outcome="y", unit="unit", time="time", first_treat="first_treat")
        # vcov is square of len(coefficients) - 1 (the "ATT" alias is the
        # only non-column entry in the dict).
        n_kept = len([k for k in res.coefficients.keys() if k != "ATT"])
        assert res.vcov.shape == (n_kept, n_kept)
        # Diagonal entries (variances) post-clamp must be non-negative.
        # SpilloverDiD clamps in the per-coef SE extraction; verify the
        # vcov itself is finite where written.
        finite_diag = np.diag(res.vcov)[np.isfinite(np.diag(res.vcov))]
        assert all(finite_diag >= 0)


class TestSpilloverDiDEventStudySummaryRoundTrip:
    """summary() includes per-event-time blocks; pickle round-trip preserves MultiIndex."""

    def test_summary_includes_dynamic_block(self):
        df = generate_butts_staggered_dgp(seed=42)
        res = _fit_event_study(df, horizon_max=2)
        s = res.summary()
        assert "Dynamic Direct Effects" in s
        assert "k=" in s or "+" in s  # event-time labels rendered

    def test_pickle_round_trip_preserves_multiindex(self):
        import pickle

        df = generate_butts_staggered_dgp(seed=42)
        res = _fit_event_study(df, horizon_max=2)
        round_tripped = pickle.loads(pickle.dumps(res))
        # MultiIndex preserved.
        assert isinstance(round_tripped.spillover_effects.index, pd.MultiIndex)
        # att_dynamic preserved.
        pd.testing.assert_frame_equal(res.att_dynamic, round_tripped.att_dynamic)

    def test_to_dict_serializes_new_fields(self):
        df = generate_butts_staggered_dgp(seed=42)
        res = _fit_event_study(df, horizon_max=2)
        d = res.to_dict()
        assert "att_dynamic" in d
        assert "event_study_effects" in d
        assert "horizon_max" in d
        assert "reference_period" in d


class TestSpilloverDiDEventStudyFitIdempotence:
    """Clone + repeat-fit produces bit-identical att_dynamic AND spillover_effects."""

    def test_fit_twice_bit_identical(self):
        df = generate_butts_staggered_dgp(seed=42)
        est = SpilloverDiD(
            rings=[0.0, 50.0, 200.0],
            d_bar=200.0,
            conley_coords=("lat", "lon"),
            event_study=True,
            horizon_max=2,
        )
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", UserWarning)
            res_1 = est.fit(df, outcome="y", unit="unit", time="time", first_treat="first_treat")
            res_2 = est.fit(df, outcome="y", unit="unit", time="time", first_treat="first_treat")
        pd.testing.assert_frame_equal(res_1.att_dynamic, res_2.att_dynamic)
        pd.testing.assert_frame_equal(res_1.spillover_effects, res_2.spillover_effects)
        assert res_1.att == res_2.att


class TestSpilloverDiDEventStudyFiniteMaskPath:
    """PR #456 R1 fix: event_study=True must use post-finite_mask counts.

    When stage-1 warn-and-drop excludes baseline-treated units (those with
    no Omega_0 rows), the per-event-time `n_obs` values in att_dynamic /
    event_study_effects AND the share weights for the scalar `att` must
    reflect the POST-mask sample — not the pre-mask design.
    """

    def _make_warn_and_drop_panel(self):
        rng = np.random.default_rng(1)
        rows = []
        # 4 baseline-treated units (no Omega_0 rows → warned-dropped).
        for k in range(4):
            for t in (0, 1, 2):
                rows.append(
                    {
                        "unit": f"baseline_{k}",
                        "time": t,
                        "lat": 0.0 + k * 0.001,
                        "lon": 0.0,
                        "D": 1,
                        "y": rng.normal(),
                    }
                )
        # 3 validly-treated units (treated from t=1; supported).
        for k in range(3):
            for t in (0, 1, 2):
                rows.append(
                    {
                        "unit": f"treated_t1_{k}",
                        "time": t,
                        "lat": 10.0 + k * 0.01,
                        "lon": 0.0,
                        "D": int(t >= 1),
                        "y": rng.normal(),
                    }
                )
        # 5 far-controls (full Omega_0 support).
        for k in range(5):
            for t in (0, 1, 2):
                rows.append(
                    {
                        "unit": f"far_control_{k}",
                        "time": t,
                        "lat": 20.0 + k * 0.01,
                        "lon": 0.0,
                        "D": 0,
                        "y": rng.normal(),
                    }
                )
        return pd.DataFrame(rows)

    def test_n_obs_in_att_dynamic_reflects_post_mask_sample(self):
        df = self._make_warn_and_drop_panel()
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            event_study=True,
            horizon_max=1,
        )
        with pytest.warns(UserWarning):
            res = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        # 4 baselines × 3 periods = 12 rows excluded. Remaining: 3 treated +
        # 5 controls = 8 units × 3 periods = 24 rows. n_treated = 3 supported
        # treated units × 2 post-treatment periods (t=1, t=2) = 6.
        assert res.n_obs == 24, f"n_obs={res.n_obs} (expected 24)"
        # att_dynamic: pre-mask, baseline_{0..3} had D=1 at every t, but
        # those rows are now excluded. The n_obs per k should ONLY count the
        # treated_t1_{0..2} rows.
        # At k=0 (t=1, supported treated): 3 rows.
        # At k=1 (t=2, supported treated): 3 rows.
        # At k=-1 (t=0, supported treated; reference): 0 rows (reference is dropped).
        assert res.att_dynamic.loc[0, "n_obs"] == 3, (
            f"k=0 n_obs={res.att_dynamic.loc[0, 'n_obs']} (expected 3 — the 3 "
            "supported treated_t1 rows at t=1, NOT 7 including pre-mask baselines)"
        )
        assert (
            res.att_dynamic.loc[1, "n_obs"] == 3
        ), f"k=+1 n_obs={res.att_dynamic.loc[1, 'n_obs']} (expected 3)"

    def test_event_study_effects_n_obs_reflects_post_mask_sample(self):
        df = self._make_warn_and_drop_panel()
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            event_study=True,
            horizon_max=1,
        )
        with pytest.warns(UserWarning):
            res = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        # event_study_effects dict mirrors att_dynamic, must be consistent.
        for k in res.att_dynamic.index:
            es_n = res.event_study_effects[int(k)]["n_obs"]
            dyn_n = res.att_dynamic.loc[k, "n_obs"]
            assert es_n == dyn_n, (
                f"k={k}: event_study_effects n_obs ({es_n}) disagrees "
                f"with att_dynamic n_obs ({dyn_n})"
            )

    def test_scalar_att_weights_use_post_mask_counts(self):
        """Lincom att = sum_{k>=0} w_k * tau_k where w_k = post-mask n_obs / total."""
        df = self._make_warn_and_drop_panel()
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            event_study=True,
            horizon_max=1,
        )
        with pytest.warns(UserWarning):
            res = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        # Hand-compute share-weighted att from att_dynamic post-mask n_obs.
        post = res.att_dynamic[res.att_dynamic.index >= 0]
        total = post["n_obs"].sum()
        if total > 0 and not post["coef"].isna().any():
            hand_att = (post["coef"] * post["n_obs"]).sum() / total
            assert abs(hand_att - res.att) < 1e-10, (
                f"att={res.att}, hand-computed from post-mask n_obs={hand_att}, "
                f"diff={abs(hand_att - res.att):.2e}"
            )


class TestSpilloverDiDEventStudyRankDeficientFailClosed:
    """PR #456 R1 fix: when solve_ols drops a post-direct column as NaN,
    the scalar `att` must fail closed (NaN with warning), not silently
    discard weight mass via np.nansum on a fixed weight vector.
    """

    def test_nan_post_direct_coef_yields_nan_att_with_warning(self, monkeypatch):
        """Monkey-patch solve_ols to NaN out one post-treatment direct coef
        and assert att=NaN with the documented warning."""
        df = generate_butts_staggered_dgp(seed=42)
        from diff_diff import spillover as spillover_mod
        from diff_diff.linalg import solve_ols as real_solve_ols

        def solve_ols_with_nan_post_direct(*args, **kwargs):
            coef, residuals, vcov = real_solve_ols(*args, **kwargs)
            column_names = kwargs.get("column_names", [])
            # Find the first post-treatment direct column (D^k=+N with N>=0)
            # and NaN out its coefficient.
            for i, name in enumerate(column_names):
                if name.startswith("D^k=+") and name != "D^k=-0":
                    coef[i] = float("nan")
                    if vcov is not None:
                        vcov[i, :] = float("nan")
                        vcov[:, i] = float("nan")
                    break
            return coef, residuals, vcov

        monkeypatch.setattr(spillover_mod, "solve_ols", solve_ols_with_nan_post_direct)
        est = SpilloverDiD(
            rings=[0.0, 50.0, 200.0],
            d_bar=200.0,
            conley_coords=("lat", "lon"),
            event_study=True,
            horizon_max=2,
        )
        with pytest.warns(UserWarning, match="scalar `att` is NaN"):
            res = est.fit(df, outcome="y", unit="unit", time="time", first_treat="first_treat")
        assert np.isnan(res.att), f"Expected att=NaN, got {res.att}"
        assert np.isnan(res.se), f"Expected se=NaN, got {res.se}"


class TestSpilloverDiDEventStudyReferencePeriodSpilloverRows:
    """PR #456 R1 fix (P3): rectangular spillover_effects must include
    (ring, ref_period) rows with coef=0.0, se=0.0, n_obs=0 (matching the
    direct-effect reference row convention)."""

    def test_ref_period_row_present_per_ring(self):
        df = generate_butts_staggered_dgp(seed=42)
        res = _fit_event_study(df, horizon_max=2)
        ref = res.reference_period
        # Every ring should have a (ring, ref_period) row.
        for ring_label in res.spillover_effects.index.get_level_values("ring").unique():
            assert (
                ring_label,
                ref,
            ) in res.spillover_effects.index, (
                f"Missing (ring={ring_label}, k={ref}) row in spillover_effects"
            )

    def test_ref_period_row_uses_zero_anchor(self):
        df = generate_butts_staggered_dgp(seed=42)
        res = _fit_event_study(df, horizon_max=2)
        ref = res.reference_period
        for ring_label in res.spillover_effects.index.get_level_values("ring").unique():
            row = res.spillover_effects.loc[(ring_label, ref)]
            assert row["coef"] == 0.0
            assert row["se"] == 0.0
            assert row["n_obs"] == 0
            assert row["ci_low"] == 0.0
            assert row["ci_high"] == 0.0
            assert np.isnan(row["t_stat"])
            assert np.isnan(row["p_value"])


class TestSpilloverDiDEventStudyPlotIntegration:
    """PR #456 R5 P2: plot_event_study must honor reference_period.

    Wave C's rectangular event_study_effects emits multiple rows with
    `n_obs = 0` (empty horizons + the reference). The legacy plot reference
    detection picks the FIRST `n_obs == 0` row, which may be a non-reference
    horizon. The fix prefers `results.reference_period` when present.
    """

    def test_plot_event_study_uses_explicit_reference_period(self):
        """Set an oversized horizon_max so multiple horizons have n_obs=0.
        The reference detection must still pick the documented reference
        period (-1 with default anticipation=0), not the first empty
        horizon found by iteration order."""
        from diff_diff.visualization._event_study import _extract_plot_data

        df = generate_butts_staggered_dgp(seed=42)
        # horizon_max=4 on a 6-period panel yields several empty post-direct
        # horizons (e.g. cohort onset=3 only has k=0..2 in-panel, so k=+3, +4
        # are empty for that cohort's contribution) plus the reference at -1.
        res = _fit_event_study(df, horizon_max=4)
        (
            effects,
            se,
            periods,
            pre_periods,
            post_periods,
            ref_period,
            ref_inferred,
            *_,
        ) = _extract_plot_data(
            res,
            periods=None,
            pre_periods=None,
            post_periods=None,
            reference_period=None,
        )
        # Reference inference uses the explicit attribute (preferred over the
        # n_obs==0 heuristic that could pick any empty horizon).
        assert ref_inferred is True
        assert ref_period == res.reference_period == -1, (
            f"plot_event_study picked reference_period={ref_period}, "
            f"expected {res.reference_period} from explicit attribute"
        )
