"""Drift detection for Tutorial 23 (``docs/tutorials/23_spillover_tva.ipynb``).

The tutorial narrative quotes seed-specific numbers (naive vs.
SpilloverDiD comparison, sensitivity-grid endpoints, HC1 vs. Conley
SE). If library numerics drift (estimator changes, RNG path changes,
BLAS path changes), the prose can go stale silently while
``pytest --nbmake`` still passes — it only checks that the cells
execute without error.

These asserts re-derive the same numbers using the locked T23 DGP
duplicated below (verbatim from the notebook §2 code cell), then check
them against the values quoted in the tutorial markdown. If a future
change moves any number outside its tolerance band, this test fails
and a maintainer is forced to either update the prose or investigate
the methodology shift before merge.

T23 is the first SpilloverDiD tutorial. It demonstrates the
``SpilloverDiD`` estimator on a TVA-style synthetic panel reproducing
the Butts (2021) §4 Table 1 Panel A ~40% understatement direction
(naive multi-period TWFE significantly understates the direct effect
when near-controls absorb spillover). The DGP-builder constants below
MUST stay in sync with the corresponding constants in the notebook §2
code cell; the ``test_dgp_true_parameters_match_quoted`` test catches
silent drift on those values.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import MultiPeriodDiD, SpilloverDiD

# Locked DGP parameters — must stay in sync with the notebook §2 cell.
MAIN_SEED = 23
N_TREATED = 25
N_NEAR = 120
N_FAR = 55
T_PERIODS = 4
FIRST_TREAT = 3
TAU_TOTAL = -7.4
DELTA_1 = -4.5
D_BAR_KM = 100.0
NOISE_SD = 0.5


def _build_t23_panel(seed: int = MAIN_SEED) -> pd.DataFrame:
    """Duplicated verbatim from the notebook §2 code cell. Keep in sync."""
    rng = np.random.default_rng(seed)
    n_units = N_TREATED + N_NEAR + N_FAR
    units = [f"u{i:04d}" for i in range(n_units)]
    alpha = rng.normal(0.0, 1.0, size=n_units)
    lambda_t = np.array([0.0, 0.5, 1.0, 1.5])[:T_PERIODS]

    coords = np.empty((n_units, 2))
    is_treated_unit = np.zeros(n_units, dtype=bool)
    is_near_unit = np.zeros(n_units, dtype=bool)
    for i in range(N_TREATED):
        coords[i] = (rng.normal(0, 0.05), rng.normal(0, 0.05))
        is_treated_unit[i] = True
    for i in range(N_TREATED, N_TREATED + N_NEAR):
        coords[i] = (rng.uniform(0.1, 0.7), rng.uniform(-0.3, 0.3))
        is_near_unit[i] = True
    for i in range(N_TREATED + N_NEAR, n_units):
        coords[i] = (rng.uniform(2.0, 3.0), rng.uniform(-0.5, 0.5))

    rows = []
    for i, u in enumerate(units):
        for t in range(1, T_PERIODS + 1):
            D_it = int(is_treated_unit[i] and t >= FIRST_TREAT)
            Ring1_it = int(is_near_unit[i] and t >= FIRST_TREAT)
            y = (
                alpha[i]
                + lambda_t[t - 1]
                + TAU_TOTAL * D_it
                + DELTA_1 * Ring1_it * (1 - D_it)
                + rng.normal(0, NOISE_SD)
            )
            rows.append(
                {
                    "unit": u,
                    "time": t,
                    "lat": coords[i, 0],
                    "lon": coords[i, 1],
                    "ever_treated": int(is_treated_unit[i]),
                    "D": D_it,
                    "y": y,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return _build_t23_panel()


@pytest.fixture(scope="module")
def naive_fit(panel):
    est = MultiPeriodDiD()
    with warnings.catch_warnings():
        # absorb=['unit'] makes the unit-invariant 'ever_treated' indicator
        # perfectly collinear with the unit FE; MultiPeriodDiD drops it
        # (with a UserWarning) and identifies the ATT through the
        # ever_treated x post interaction columns. This is the expected
        # TWFE specification; the rank-deficient drop is benign.
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="Rank-deficient design matrix"
        )
        return est.fit(
            panel,
            outcome="y",
            treatment="ever_treated",
            time="time",
            post_periods=[3, 4],
            unit="unit",
            absorb=["unit"],
            reference_period=2,  # explicit pre-period; matches the current MPD default
        )


def _silence_spillover_matmul_warnings():
    """Apply the notebook's narrow ``.*encountered in matmul``
    ``RuntimeWarning`` filter. The three matmul warnings ("divide by
    zero" / "overflow" / "invalid value") are an Apple Silicon M4 +
    macOS Sequoia + numpy<2.3 Accelerate BLAS artifact documented at
    ``TODO.md`` under "RuntimeWarnings in Linear Algebra Operations"
    (root cause: Apple BLAS SME kernels corrupt the FP status register;
    tracked as numpy#28687, fixed in numpy>=2.3). They DO NOT fire on
    M3 / Intel / Linux or numpy>=2.3 — so this filter is a no-op there,
    and any platform-specific noise it does silence does not affect
    result correctness.

    The post-filter warning surface (zero remaining warnings on the
    T23 DGP) is pinned by ``test_spillover_fit_warning_policy_post_filter_clean``
    and ``test_spillover_conley_fit_warning_policy_post_filter_clean``.
    A new RuntimeWarning with a different message, or any UserWarning /
    FutureWarning, fails those tests."""
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=r".*encountered in matmul")


@pytest.fixture(scope="module")
def spillover_fit(panel):
    est = SpilloverDiD(rings=[0.0, D_BAR_KM], conley_coords=("lat", "lon"))
    with warnings.catch_warnings():
        _silence_spillover_matmul_warnings()
        return est.fit(panel, outcome="y", unit="unit", time="time", treatment="D")


@pytest.fixture(scope="module")
def spillover_conley_lag0_fit(panel):
    est = SpilloverDiD(
        rings=[0.0, D_BAR_KM],
        conley_coords=("lat", "lon"),
        vcov_type="conley",
        conley_cutoff_km=D_BAR_KM,
        conley_lag_cutoff=0,
    )
    with warnings.catch_warnings():
        _silence_spillover_matmul_warnings()
        return est.fit(panel, outcome="y", unit="unit", time="time", treatment="D")


@pytest.fixture(scope="module")
def spillover_conley_lag1_fit(panel):
    est = SpilloverDiD(
        rings=[0.0, D_BAR_KM],
        conley_coords=("lat", "lon"),
        vcov_type="conley",
        conley_cutoff_km=D_BAR_KM,
        conley_lag_cutoff=1,
    )
    with warnings.catch_warnings():
        _silence_spillover_matmul_warnings()
        return est.fit(panel, outcome="y", unit="unit", time="time", treatment="D")


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------


def test_panel_composition(panel):
    """800 rows = 200 units × 4 periods; 25 treated, 120 near, 55 far."""
    assert len(panel) == (N_TREATED + N_NEAR + N_FAR) * T_PERIODS == 800
    assert panel["unit"].nunique() == N_TREATED + N_NEAR + N_FAR == 200
    assert panel["time"].nunique() == T_PERIODS == 4
    treated_units = panel.loc[panel["ever_treated"] == 1, "unit"].nunique()
    assert treated_units == N_TREATED == 25


def test_panel_geographic_bands(panel):
    """Treated cluster within ~10 km of origin (5 sigma * 2 lat/lon),
    near-controls in [0.1, 0.7] lat degrees, far-controls in [2.0, 3.0]."""
    units = panel.drop_duplicates("unit")
    treated = units[units["ever_treated"] == 1]
    untreated = units[units["ever_treated"] == 0]
    near = untreated[untreated["lat"] <= 1.0]
    far = untreated[untreated["lat"] > 1.0]

    assert len(near) == N_NEAR == 120
    assert len(far) == N_FAR == 55

    # Treated cluster geometry
    assert treated["lat"].abs().max() < 0.3
    assert treated["lon"].abs().max() < 0.3
    # Near-control band geometry
    assert near["lat"].min() >= 0.1
    assert near["lat"].max() <= 0.7
    # Far-control band geometry
    assert far["lat"].min() >= 2.0
    assert far["lat"].max() <= 3.0


def test_dgp_true_parameters_match_quoted():
    """True parameters quoted in the tutorial narrative (§2)."""
    assert TAU_TOTAL == -7.4
    assert DELTA_1 == -4.5
    assert D_BAR_KM == 100.0
    assert NOISE_SD == 0.5
    assert MAIN_SEED == 23


def test_estimator_construction_matches_quoted():
    """The §5 fit instantiation parameters must match the docstring narrative."""
    est = SpilloverDiD(rings=[0.0, D_BAR_KM], conley_coords=("lat", "lon"))
    params = est.get_params()
    assert params["rings"] == [0.0, 100.0]
    assert params["d_bar"] is None  # auto-default to max(rings)
    assert params["conley_coords"] == ("lat", "lon")
    assert params["vcov_type"] == "hc1"


def test_naive_twfe_understates_tau_total(naive_fit):
    """§3 quoted: naive ATT ≈ -4.29, ~58% of true tau_total (~42% understatement)."""
    ratio = naive_fit.att / TAU_TOTAL
    assert 0.55 <= ratio <= 0.62, f"naive ratio={ratio:.3f} outside [0.55, 0.62] band"


def test_naive_att_endpoint_matches_quoted(naive_fit):
    """§3 quoted endpoint: round-to-1 pin (looser than round-to-2 for BLAS safety)."""
    assert round(naive_fit.att, 1) == -4.3


def test_spillover_did_recovers_tau_total(spillover_fit):
    """§5 quoted: SpilloverDiD tau_total ≈ -7.34 ± 0.12, recovers true -7.4."""
    assert abs(spillover_fit.att - TAU_TOTAL) < 0.5
    assert round(spillover_fit.att, 1) == -7.3


def test_spillover_did_recovers_delta_1(spillover_fit):
    """§5 quoted: SpilloverDiD delta_1 ≈ -4.53 ± 0.07, recovers true -4.5."""
    delta_1 = float(spillover_fit.spillover_effects.iloc[0]["coef"])
    assert abs(delta_1 - DELTA_1) < 0.5
    assert round(delta_1, 1) == -4.5


def test_rings_sensitivity_grid_endpoints(panel):
    """§4 quoted: d_bar=50 → tau=-5.4, others (100/150/200) → tau=-7.3.

    Per the plan and reviewer guidance, round-to-1 tolerance is safer
    than round-to-2 against BLAS divergence on the borderline-rank-deficient
    smallest grid point.
    """
    expected_tau = {50.0: -5.4, 100.0: -7.3, 150.0: -7.3, 200.0: -7.3}
    expected_delta = {50.0: -2.6, 100.0: -4.5, 150.0: -4.5, 200.0: -4.5}
    for outer in (50.0, 100.0, 150.0, 200.0):
        est = SpilloverDiD(rings=[0.0, outer], conley_coords=("lat", "lon"))
        with warnings.catch_warnings():
            _silence_spillover_matmul_warnings()
            res = est.fit(panel, outcome="y", unit="unit", time="time", treatment="D")
        assert res.spillover_effects is not None
        delta_1 = float(res.spillover_effects.iloc[0]["coef"])
        assert round(res.att, 1) == expected_tau[outer], (
            f"d_bar={outer}: tau={res.att:.4f} (rounded {round(res.att, 1)}) "
            f"vs expected {expected_tau[outer]}"
        )
        assert round(delta_1, 1) == expected_delta[outer], (
            f"d_bar={outer}: delta_1={delta_1:.4f} (rounded {round(delta_1, 1)}) "
            f"vs expected {expected_delta[outer]}"
        )


def test_rings_grid_d_bar_100_to_200_identical(panel):
    """§4 narrative claim: once d_bar covers the true spillover horizon
    (which here ends at ~78 km), widening past 100 km adds zero
    observations to the ring and the estimates are identical."""
    results = []
    for outer in (100.0, 150.0, 200.0):
        est = SpilloverDiD(rings=[0.0, outer], conley_coords=("lat", "lon"))
        with warnings.catch_warnings():
            _silence_spillover_matmul_warnings()
            res = est.fit(panel, outcome="y", unit="unit", time="time", treatment="D")
        results.append(res.att)
    np.testing.assert_allclose(results, results[0] * np.ones(3), atol=1e-10)


def test_conley_se_differs_from_hc1(spillover_fit, spillover_conley_lag0_fit):
    """§6 sanity: Conley vcov produces a different SE than HC1 by more than
    floating-point noise. Pairs with `test_conley_se_less_than_hc1` which
    pins the direction of the difference for this specific DGP."""
    assert abs(spillover_conley_lag0_fit.se - spillover_fit.se) > 1e-6


def test_conley_se_less_than_hc1(spillover_fit, spillover_conley_lag0_fit):
    """§6 prose claim: 'on this DGP, the Conley spatial-HAC SE comes in
    *lower* than HC1'. Pin the direction so the narrative doesn't go
    stale if a future library change flips the sign of the per-pair
    score covariance and reverses the inequality."""
    assert spillover_conley_lag0_fit.se < spillover_fit.se


def test_conley_se_point_estimates_invariant(
    spillover_fit, spillover_conley_lag0_fit, spillover_conley_lag1_fit
):
    """§6 narrative claim: variance-type choice doesn't move the point
    estimates. tau_total is bit-identical across HC1 / Conley lag=0 /
    Conley lag=1 (all paths use the same OLS solve; only the meat
    differs)."""
    np.testing.assert_allclose(
        [spillover_conley_lag0_fit.att, spillover_conley_lag1_fit.att],
        spillover_fit.att,
        atol=1e-10,
    )


def test_conley_lag_cutoff_changes_se_vs_lag_zero(
    spillover_conley_lag0_fit, spillover_conley_lag1_fit
):
    """§6 sanity: adding the serial term (lag=1) changes the SE relative
    to spatial-only (lag=0). Direction-agnostic — on this DGP it
    shrinks, on others it can grow."""
    assert abs(spillover_conley_lag1_fit.se - spillover_conley_lag0_fit.se) > 1e-6


def test_summary_renders_without_warning(spillover_fit):
    """§5 smoke: the summary() call runs clean on the headline fit."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = spillover_fit.summary()
    assert isinstance(out, str)
    assert len(out) > 0


def _assert_post_filter_warning_surface_is_clean(captured) -> None:
    """Shared T19-style platform-agnostic warning-policy assertion.

    The notebook's narrow ``.*encountered in matmul`` filter (see
    `_silence_spillover_matmul_warnings`) silences three Apple Silicon
    M4 + numpy<2.3 Accelerate BLAS warnings that are emitted on the
    affected platform but DO NOT fire on M3 / Intel / Linux or
    numpy>=2.3 (per ``TODO.md`` "RuntimeWarnings in Linear Algebra
    Operations"). The drift contract this assertion locks is
    platform-agnostic:

    - on platforms where the matmul warnings fire, they get filtered
      and never reach the captured list;
    - on platforms where they don't fire, the filter is a no-op;

    EITHER WAY the post-filter captured list must be empty. Any
    UserWarning, FutureWarning, DeprecationWarning, or RuntimeWarning
    with a non-matmul message will fail this assertion and force the
    maintainer to either update the notebook narrative or fix the
    underlying cause."""
    if not captured:
        return
    details = [(msg.category.__name__, str(msg.message)) for msg in captured]
    assert False, (
        f"Unexpected post-filter warnings on the T23 DGP: {details}. "
        f"If a new warning is genuinely expected, broaden "
        f"`_silence_spillover_matmul_warnings()` and update the §5/§6 "
        f"notebook narrative accordingly."
    )


def test_spillover_fit_warning_policy_post_filter_clean(panel):
    """§5 warning-policy guard (T19-pattern, platform-agnostic).

    Mirrors the notebook's narrow ``.*encountered in matmul`` filter
    inside the capture block, then asserts the post-filter warning
    surface is empty on the T23 DGP. On Apple Silicon M4 + numpy<2.3
    the three known BLAS matmul warnings fire and are filtered; on
    M3 / Intel / Linux or numpy>=2.3 the filter is a no-op. EITHER
    WAY a fresh ``UserWarning`` / ``FutureWarning`` or any non-matmul
    ``RuntimeWarning`` will fail this guard."""
    est = SpilloverDiD(rings=[0.0, D_BAR_KM], conley_coords=("lat", "lon"))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _silence_spillover_matmul_warnings()  # mirror notebook §5 filter
        est.fit(panel, outcome="y", unit="unit", time="time", treatment="D")
    _assert_post_filter_warning_surface_is_clean(w)


def test_spillover_conley_fit_warning_policy_post_filter_clean(panel):
    """§6 warning-policy guard, parallel to §5 but on the Conley path
    (vcov_type="conley", conley_cutoff_km=d_bar, conley_lag_cutoff in {0, 1}).
    Same T19-style platform-agnostic contract: mirror the notebook
    filter inside the capture, assert no remaining warning escaped."""
    for lag in (0, 1):
        est = SpilloverDiD(
            rings=[0.0, D_BAR_KM],
            conley_coords=("lat", "lon"),
            vcov_type="conley",
            conley_cutoff_km=D_BAR_KM,
            conley_lag_cutoff=lag,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _silence_spillover_matmul_warnings()  # mirror notebook §6 filter
            est.fit(panel, outcome="y", unit="unit", time="time", treatment="D")
        _assert_post_filter_warning_surface_is_clean(w)
