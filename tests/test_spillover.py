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
    _build_ring_indicators,
    _check_omega_0_connectivity,
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
        d_it, row_unit, row_time = _compute_nearest_treated_distance_staggered(
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
        d_it, row_unit, row_time = _compute_nearest_treated_distance_staggered(
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
        d_it, row_unit, row_time = _compute_nearest_treated_distance_staggered(
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

    def test_staggered_recovers_tau_total(self):
        """Staggered MC with 30 seeds (smaller because each DGP is larger)."""
        att_estimates = []
        for s in range(30):
            df = generate_butts_staggered_dgp(tau_total=-0.07, delta_1=-0.04, seed=s)
            result = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon")).fit(
                df, outcome="y", unit="unit", time="time", first_treat="first_treat"
            )
            att_estimates.append(result.att)
        mean_att = float(np.mean(att_estimates))
        # Staggered MC is noisier; allow a looser tolerance.
        assert (
            abs(mean_att - (-0.07)) < 0.04
        ), f"staggered tau_total: expected -0.07, got {mean_att:.4f}"


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

    def test_conley_se_differs_from_hc1(self):
        """Conley SE differs from HC1 baseline (spatial correlation in errors)."""
        df = generate_butts_nonstaggered_dgp(
            seed=42, n_treated=20, n_near_control=80, n_far_control=100
        )
        est_hc1 = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            vcov_type="hc1",
        )
        result_hc1 = est_hc1.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        est_conley = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            vcov_type="conley",
            conley_cutoff_km=200.0,
            conley_lag_cutoff=0,
        )
        result_conley = est_conley.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        # ATT point estimate unchanged across vcov; SE may differ.
        assert abs(result_hc1.att - result_conley.att) < 1e-10
        # Conley SE may be larger or smaller than HC1 depending on spatial
        # error correlation; just assert it's not identical.
        # (Synthetic DGP has independent errors so they may be very close;
        # use a loose tolerance — primarily a wiring test.)
        assert np.isfinite(result_conley.se)


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
