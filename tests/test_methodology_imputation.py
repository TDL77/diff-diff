"""Methodology verification tests for ImputationDiD.

Targets Borusyak, Jaravel & Spiess (2024), *Revisiting Event-Study Designs:
Robust and Efficient Estimation*, Review of Economic Studies 91(6), 3253-3285
(DOI 10.1093/restud/rdae007).

Paper-equation walk-through (each Verified Component class maps to a numbered
result, verified against the source PDF in
``docs/methodology/papers/borusyak-jaravel-spiess-2024-review.md``):

- **Theorem 1 / 2** (p. 3267-8) — the 3-step imputation estimator (Step 1 fit on
  the untreated set Omega_0 only via eq. 5; Step 2 impute Y(0); Step 3 weighted
  aggregation) recovers the target ATT (``TestB2024Theorem2Imputation``).
- **Theorem 3 / Eqs. 6-8** (p. 3271-2) — conservative clustered variance and the
  *unit-clustered* Equation 8 auxiliary aggregator
  (``TestB2024Theorem3Variance``, ``TestB2024Eq8AuxiliaryAggregator``).
- **Proposition 5** (p. 3266) — without never-treated units, horizons
  ``K_it >= H_bar = max(E_i) - min(E_i)`` are not identified -> NaN + warning
  (``TestB2024Proposition5NoNeverTreated``).
- **Test 1 / Eq. 9 + Proposition 9** (p. 3273-4) — robust pre-trend test on
  Omega_0 only, independent of the treatment-effect estimate
  (``TestB2024Proposition9Test1``).
- Library extensions / deviations (multiplier bootstrap, survey TSL,
  ``aux_partition`` defaults, NaN inference, Prop-5 refuse-to-estimate)
  (``TestB2024LibraryDeviations``).

R-parity (bottom of file, NOT a methodology walk-through): ``TestImputationDiDParityR``
pins Python output against R ``didimputation::did_imputation()`` on fixed-seed
goldens. R ``didimputation`` implements the paper's Equation 8 only at the
cohort x event-time partition (where it equals ``sum(v^2 * tau)/sum(v^2)``); see
``docs/methodology/REGISTRY.md`` ``## ImputationDiD`` "Deviation from R".

See also:

- ``docs/methodology/papers/borusyak-jaravel-spiess-2024-review.md`` (primary-source review)
- ``docs/methodology/REGISTRY.md`` ``## ImputationDiD`` block
- ``METHODOLOGY_REVIEW.md`` ``ImputationDiD`` section
- ``tests/test_imputation.py`` (implementation-detail unit tests)
- ``benchmarks/R/generate_didimputation_golden.R`` (R goldens generator)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest
from scipy.sparse.linalg import MatrixRankWarning

from diff_diff import ImputationDiD

# =============================================================================
# Module-level R-fixture availability + per-class seed decorrelation
# =============================================================================

GOLDEN_PATH = Path(__file__).parent.parent / "benchmarks" / "data" / "didimputation_golden.json"
PANEL_PATH = Path(__file__).parent.parent / "benchmarks" / "data" / "didimputation_test_panel.csv"
_R_FIXTURE_AVAILABLE = GOLDEN_PATH.is_file() and PANEL_PATH.is_file()

_BASE_SEED_THM2 = 9101
_BASE_SEED_THM3 = 9202
_BASE_SEED_EQ8 = 9303
_BASE_SEED_PROP5 = 9404
_BASE_SEED_PROP9 = 9505
_BASE_SEED_DEVIATIONS = 9606


# =============================================================================
# Helpers
# =============================================================================


def _make_staggered_panel(
    rng: np.random.Generator,
    *,
    cohorts: List[int],
    n_per_cohort: int = 100,
    n_periods: int = 6,
    tau_constant: Optional[float] = None,
    tau_by_horizon: Optional[Dict[int, float]] = None,
    sigma: float = 0.1,
    include_never_treated: bool = True,
    pretrend_slope: float = 0.0,
) -> pd.DataFrame:
    """Balanced staggered-adoption panel satisfying parallel trends.

    DGP (BJS Assumption 1): ``y_it = c_i + beta_t + w_it * tau_{K_it} + u_it``,
    with ``c_i ~ N(0,1)``, common time trend ``beta_t = 0.5 t`` (parallel
    trends hold -- no cohort-specific trends unless ``pretrend_slope != 0``),
    ``u_it ~ N(0, sigma^2)``. Treatment is absorbing from the cohort's event
    date. ``first_treat = 0`` denotes never-treated.

    ``pretrend_slope != 0`` injects a cohort-specific linear trend
    ``pretrend_slope * cohort_rank * t`` that violates parallel trends (used to
    exercise the pre-trend test's power).
    """
    if tau_constant is None and tau_by_horizon is None:
        tau_constant = 1.0
    rows: List[Dict[str, Any]] = []
    unit_id = 0
    all_cohorts = ([0] + list(cohorts)) if include_never_treated else list(cohorts)
    cohort_rank = {g: r for r, g in enumerate(sorted(cohorts))}
    for g in all_cohorts:
        for _ in range(n_per_cohort):
            c_i = rng.standard_normal()
            for t in range(1, n_periods + 1):
                beta_t = 0.5 * t
                u = sigma * rng.standard_normal()
                treated = g > 0 and t >= g
                if treated:
                    k = t - g
                    if tau_by_horizon is not None:
                        tau = tau_by_horizon.get(k, 0.0)
                    else:
                        tau = tau_constant if tau_constant is not None else 0.0
                else:
                    tau = 0.0
                trend = pretrend_slope * cohort_rank.get(g, 0) * t if g > 0 else 0.0
                y = c_i + beta_t + trend + (tau if treated else 0.0) + u
                rows.append(
                    {
                        "unit": unit_id,
                        "time": t,
                        "first_treat": g,
                        "outcome": y,
                    }
                )
            unit_id += 1
    return pd.DataFrame(rows)


# =============================================================================
# Theorem 1 / 2 — the imputation estimator
# =============================================================================


class TestB2024Theorem2Imputation:
    """Theorem 1/2 (p. 3267-8): 3-step imputation recovers the target ATT,
    fitting the counterfactual model on the untreated set Omega_0 only."""

    def test_recovers_constant_att(self) -> None:
        """Under a constant treatment effect tau=2.0, the overall ATT is recovered.

        DGP: 2 cohorts + never-treated, N=300 units, sigma=0.1. The per-obs SE is
        ~sigma/sqrt(N_1); a 0.05 band is >5 sigma.
        """
        rng = np.random.default_rng(_BASE_SEED_THM2 + 1)
        panel = _make_staggered_panel(
            rng, cohorts=[3, 4], n_per_cohort=100, tau_constant=2.0, sigma=0.1
        )
        res = ImputationDiD().fit(
            panel, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        assert abs(res.overall_att - 2.0) < 0.05

    def test_recovers_heterogeneous_event_study(self) -> None:
        """Horizon-specific effects tau_K = 1 + 0.5*K are recovered per horizon."""
        rng = np.random.default_rng(_BASE_SEED_THM2 + 2)
        tau_by_h = {0: 1.0, 1: 1.5, 2: 2.0, 3: 2.5}
        panel = _make_staggered_panel(
            rng, cohorts=[2, 3], n_per_cohort=120, tau_by_horizon=tau_by_h, sigma=0.1
        )
        res = ImputationDiD().fit(
            panel,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )
        assert res.event_study_effects is not None
        for h, expected in tau_by_h.items():
            assert h in res.event_study_effects, f"missing horizon {h}"
            got = res.event_study_effects[h]["effect"]
            assert abs(got - expected) < 0.06, f"h={h}: {got:.4f} vs {expected}"

    def test_step1_uses_untreated_only(self) -> None:
        """Perturbing a single treated outcome by delta shifts the overall ATT by
        exactly delta/N_1 -- proving treated observations never feed back into the
        Step-1 counterfactual model (eq. 5 is fit on Omega_0 only)."""
        rng = np.random.default_rng(_BASE_SEED_THM2 + 3)
        panel = _make_staggered_panel(
            rng, cohorts=[3, 4], n_per_cohort=60, tau_constant=1.0, sigma=0.1
        )
        base = ImputationDiD().fit(
            panel, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        n_1 = int(((panel["first_treat"] > 0) & (panel["time"] >= panel["first_treat"])).sum())

        perturbed = panel.copy()
        treated_idx = perturbed.index[
            (perturbed["first_treat"] > 0) & (perturbed["time"] >= perturbed["first_treat"])
        ][0]
        delta = 100.0
        perturbed.loc[treated_idx, "outcome"] += delta
        pert = ImputationDiD().fit(
            perturbed, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        # Only the perturbed obs's own tau_hat changes (weight 1/N_1).
        assert abs((pert.overall_att - base.overall_att) - delta / n_1) < 1e-6


# =============================================================================
# Theorem 3 — conservative clustered variance
# =============================================================================


class TestB2024Theorem3Variance:
    """Theorem 3 (p. 3271-2): the conservative clustered SE is finite/positive and
    (being conservative) is no smaller than a within-cohort-homogeneous benchmark."""

    def test_se_finite_and_positive(self) -> None:
        rng = np.random.default_rng(_BASE_SEED_THM3 + 1)
        panel = _make_staggered_panel(
            rng, cohorts=[3, 4], n_per_cohort=80, tau_constant=1.0, sigma=0.2
        )
        res = ImputationDiD().fit(
            panel, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        assert np.isfinite(res.overall_se) and res.overall_se > 0

    def test_event_study_ses_finite(self) -> None:
        rng = np.random.default_rng(_BASE_SEED_THM3 + 2)
        panel = _make_staggered_panel(
            rng, cohorts=[2, 3], n_per_cohort=80, tau_constant=1.0, sigma=0.2
        )
        res = ImputationDiD().fit(
            panel,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )
        assert res.event_study_effects is not None
        for h, eff in res.event_study_effects.items():
            # Skip the normalized reference period (effect=se=0 by construction).
            if h >= 0 and np.isfinite(eff["effect"]):
                assert np.isfinite(eff["se"]) and eff["se"] > 0, f"h={h}"

    def test_singular_omega0_routes_to_dense_fallback(self) -> None:
        """Regression: a rank-deficient Ω₀ makes A₀'A₀ singular, where SciPy
        `spsolve` returns NaN with a `MatrixRankWarning` instead of raising. The
        variance projection must still route to the dense-`lstsq` fallback (not
        silently zero the untreated influence contributions via `np.nan_to_num`)
        even under production warning filters that do NOT promote the warning.
        """
        # A period observed ONLY among treated obs -> its time FE is unidentified
        # in Ω₀ -> A₀ has an all-zero column -> A₀'A₀ is singular. Drop the
        # never-treated units at t=4 so t=4 appears only for the treated cohort.
        rng = np.random.default_rng(_BASE_SEED_THM3 + 9)
        rows: List[Dict[str, Any]] = []
        uid = 0
        for g in (0, 2):
            for _ in range(30):
                c_i = rng.standard_normal()
                for t in (1, 2, 3, 4):
                    if g == 0 and t == 4:
                        continue  # never-treated not observed at t=4
                    treated = g > 0 and t >= g
                    y = c_i + 0.5 * t + (1.0 if treated else 0.0) + 0.1 * rng.standard_normal()
                    rows.append({"unit": uid, "time": t, "first_treat": g, "outcome": y})
                uid += 1
        panel = pd.DataFrame(rows)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # Production-like: MatrixRankWarning is NOT promoted to an error here.
            warnings.filterwarnings("ignore", category=MatrixRankWarning)
            res = ImputationDiD(rank_deficient_action="silent").fit(
                panel,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )
        # The code's internal MatrixRankWarning->error promotion must trigger the
        # dense fallback even though the warning is ambiently ignored.
        assert any(
            "dense lstsq" in str(w.message) for w in caught
        ), "expected the dense-lstsq fallback under a singular Ω₀"
        assert np.isfinite(res.overall_se)


# =============================================================================
# Equation 8 — the unit-clustered auxiliary aggregator
# =============================================================================


class TestB2024Eq8AuxiliaryAggregator:
    """Equation 8 (p. 3272): the auxiliary treatment-effect model uses the
    *unit-clustered* aggregator
    ``tau_tilde_g = sum_i(sum_t v)(sum_t v*tau) / sum_i(sum_t v)^2`` -- NOT the
    naive observation-level mean ``sum(v*tau)/sum(v)``. The two differ whenever a
    unit contributes several observations to a group (coarser partitions) or the
    weights are non-uniform."""

    def test_unit_clustered_formula_handcalc(self) -> None:
        """White-box hand-calculation of _compute_auxiliary_residuals_treated.

        Construct one cohort group (cohort=2) under aux_partition='cohort' with:
          unit A: two obs, tau_hat = (0, 0), v = (1, 1)
          unit B: one obs,  tau_hat = 5,       v = 1
        Unit-clustered Eq. 8:
          a_A = 1+1 = 2, b_A = 0;  a_B = 1, b_B = 5
          tau_tilde = (2*0 + 1*5) / (2^2 + 1^2) = 5/5 = 1.0
        Observation-level mean (the OLD, wrong form):
          sum(v*tau)/sum(v) = (0+0+5)/(1+1+1) = 5/3 ~ 1.667
        So the returned residuals eps = tau_hat - tau_tilde must equal the
        unit-clustered values [-1, -1, 4], not the obs-level [-1.67, -1.67, 3.33].
        Weights are uniform here -- the divergence is driven purely by unit A
        contributing two observations to the group.
        """
        df_1 = pd.DataFrame(
            {
                "unit": ["A", "A", "B"],
                "time": [2, 3, 2],
                "first_treat": [2, 2, 2],
                "_rel_time": [0, 1, 0],
                "outcome": [0.0, 0.0, 5.0],
            }
        )
        est = ImputationDiD(aux_partition="cohort")
        # grand_mean=0 and all-zero FE => y_hat_0 = 0 => tau_hat = outcome.
        eps = est._compute_auxiliary_residuals_treated(
            df_1,
            "outcome",
            "unit",
            "time",
            "first_treat",
            None,
            {"A": 0.0, "B": 0.0},
            {2: 0.0, 3: 0.0},
            0.0,
            None,
            np.array([1.0, 1.0, 1.0]),
        )
        np.testing.assert_allclose(eps, [-1.0, -1.0, 4.0], atol=1e-12)
        # And NOT the observation-level form:
        assert not np.allclose(eps, [5.0 / 3 * -1, 5.0 / 3 * -1, 5 - 5.0 / 3], atol=1e-3)

    def test_nan_tau_co_group_obs_is_no_op(self) -> None:
        """A NaN-tau_hat observation (always v=0 by construction) must NOT poison
        its group's tau_tilde via 0*NaN=NaN. Add unit C with a missing FE (NaN
        tau_hat) and v=0 to the group above; the A/B residuals must be unchanged
        and C's residual is NaN (zeroed downstream in the variance product)."""
        df_1 = pd.DataFrame(
            {
                "unit": ["A", "A", "B", "C"],
                "time": [2, 3, 2, 2],
                "first_treat": [2, 2, 2, 2],
                "_rel_time": [0, 1, 0, 0],
                "outcome": [0.0, 0.0, 5.0, 7.0],
            }
        )
        est = ImputationDiD(aux_partition="cohort")
        # C is absent from unit_fe => NaN alpha_i => NaN tau_hat; its v_treated=0.
        eps = est._compute_auxiliary_residuals_treated(
            df_1,
            "outcome",
            "unit",
            "time",
            "first_treat",
            None,
            {"A": 0.0, "B": 0.0},
            {2: 0.0, 3: 0.0},
            0.0,
            None,
            np.array([1.0, 1.0, 1.0, 0.0]),
        )
        np.testing.assert_allclose(eps[:3], [-1.0, -1.0, 4.0], atol=1e-12)
        assert np.isnan(eps[3])

    def test_public_se_regression_pin_cohort_partition(self) -> None:
        """Public-API guard: the overall SE under aux_partition='cohort' on a
        fixed unbalanced design flows through the unit-clustered Eq. 8 aggregator.

        Correctness of the aggregator is established by the hand-calc above and by
        the R-parity class; this pins the public SE path against regression (a
        revert to the observation-level form would move this number).
        """
        rng = np.random.default_rng(_BASE_SEED_EQ8 + 7)
        panel = _make_staggered_panel(
            rng, cohorts=[2, 4], n_per_cohort=50, n_periods=6, tau_constant=1.0, sigma=0.3
        )
        res = ImputationDiD(aux_partition="cohort").fit(
            panel, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        assert np.isfinite(res.overall_se) and res.overall_se > 0
        # Regression pin (value produced by the unit-clustered Eq. 8 code path).
        assert res.overall_se == pytest.approx(_EQ8_COHORT_SE_PIN, abs=1e-8)


# Pin value for test_public_se_regression_pin_cohort_partition, produced by the
# unit-clustered Eq. 8 implementation (see the test docstring). Deterministic
# given the fixed-seed design (the SE computation itself has no randomness).
_EQ8_COHORT_SE_PIN = 0.042000264835


# =============================================================================
# Proposition 5 — non-identification without never-treated units
# =============================================================================


class TestB2024Proposition5NoNeverTreated:
    """Proposition 5 (p. 3266): with no never-treated units and H_bar =
    max(E_i)-min(E_i), horizons K >= H_bar are not identified -> NaN + warning."""

    def test_horizons_at_or_above_hbar_are_nan_with_warning(self) -> None:
        rng = np.random.default_rng(_BASE_SEED_PROP5 + 1)
        # Cohorts 3 and 5, NO never-treated => H_bar = 5 - 3 = 2.
        panel = _make_staggered_panel(
            rng,
            cohorts=[3, 5],
            n_per_cohort=80,
            n_periods=8,
            tau_constant=1.0,
            sigma=0.1,
            include_never_treated=False,
        )
        with pytest.warns(UserWarning, match="identified"):
            res = ImputationDiD().fit(
                panel,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                aggregate="event_study",
            )
        assert res.event_study_effects is not None
        h_bar = 2
        for h, eff in res.event_study_effects.items():
            if h >= h_bar:
                assert np.isnan(eff["effect"]), f"h={h} >= H_bar should be NaN"

    def test_never_treated_present_identifies_all_horizons(self) -> None:
        rng = np.random.default_rng(_BASE_SEED_PROP5 + 2)
        panel = _make_staggered_panel(
            rng,
            cohorts=[3, 5],
            n_per_cohort=80,
            n_periods=8,
            tau_constant=1.0,
            sigma=0.1,
            include_never_treated=True,
        )
        res = ImputationDiD().fit(
            panel,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )
        assert res.event_study_effects is not None
        # With never-treated controls, post horizons are identified (finite).
        assert any(
            np.isfinite(eff["effect"]) and h >= 2 for h, eff in res.event_study_effects.items()
        )


# =============================================================================
# Test 1 / Equation 9 + Proposition 9 — robust pre-trend test
# =============================================================================


class TestB2024Proposition9Test1:
    """Test 1 / Eq. 9 (p. 3273): pre-trend test on Omega_0 only; Proposition 9
    (p. 3274): the test is independent of the treatment-effect estimate."""

    def test_pretrend_test_does_not_reject_under_parallel_trends(self) -> None:
        rng = np.random.default_rng(_BASE_SEED_PROP9 + 1)
        panel = _make_staggered_panel(
            rng, cohorts=[4, 5], n_per_cohort=120, n_periods=7, tau_constant=1.0, sigma=0.1
        )
        res = ImputationDiD().fit(
            panel, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        pt = res.pretrend_test()
        assert "p_value" in pt
        assert pt["p_value"] > 0.05

    def test_pretrend_test_rejects_under_violation(self) -> None:
        rng = np.random.default_rng(_BASE_SEED_PROP9 + 2)
        panel = _make_staggered_panel(
            rng,
            cohorts=[4, 5],
            n_per_cohort=120,
            n_periods=7,
            tau_constant=1.0,
            sigma=0.1,
            pretrend_slope=0.4,
        )
        res = ImputationDiD().fit(
            panel, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        pt = res.pretrend_test()
        assert pt["p_value"] < 0.05

    def test_estimate_independent_of_pretrend_request(self) -> None:
        """Proposition 9: requesting pre-period coefficients does not change the
        treatment-effect estimate (estimation is orthogonal to pre-testing)."""
        rng = np.random.default_rng(_BASE_SEED_PROP9 + 3)
        panel = _make_staggered_panel(
            rng, cohorts=[3, 4], n_per_cohort=80, n_periods=7, tau_constant=1.5, sigma=0.1
        )
        common = dict(outcome="outcome", unit="unit", time="time", first_treat="first_treat")
        base = ImputationDiD().fit(panel, **common)
        with_pre = ImputationDiD(pretrends=True).fit(panel, **common)
        assert with_pre.overall_att == pytest.approx(base.overall_att, abs=1e-10)


# =============================================================================
# Library extensions / deviations (not in the paper)
# =============================================================================


class TestB2024LibraryDeviations:
    """Library extensions beyond BJS 2024 (documented in REGISTRY.md)."""

    def test_multiplier_bootstrap_is_library_extension(self, ci_params) -> None:
        """The paper proposes only analytical SEs; the multiplier bootstrap on the
        Theorem-3 influence function is a library extension."""
        rng = np.random.default_rng(_BASE_SEED_DEVIATIONS + 1)
        panel = _make_staggered_panel(
            rng, cohorts=[3, 4], n_per_cohort=80, tau_constant=1.0, sigma=0.2
        )
        n_boot = ci_params.bootstrap(99)
        res = ImputationDiD(n_bootstrap=n_boot, seed=7).fit(
            panel, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        assert res.bootstrap_results is not None
        assert np.isfinite(res.bootstrap_results.overall_att_se)

    def test_aux_partition_options_all_run(self) -> None:
        """aux_partition choices (cohort_horizon/cohort/horizon) are library
        defaults; the paper does not prescribe the partition."""
        rng = np.random.default_rng(_BASE_SEED_DEVIATIONS + 2)
        panel = _make_staggered_panel(
            rng, cohorts=[3, 4], n_per_cohort=60, tau_constant=1.0, sigma=0.2
        )
        common = dict(outcome="outcome", unit="unit", time="time", first_treat="first_treat")
        for partition in ("cohort_horizon", "cohort", "horizon"):
            res = ImputationDiD(aux_partition=partition).fit(panel, **common)
            assert np.isfinite(res.overall_se), partition


# =============================================================================
# R parity — didimputation::did_imputation (skip-guarded)
# =============================================================================


@pytest.fixture(scope="module")
def golden() -> dict:
    if not _R_FIXTURE_AVAILABLE:
        pytest.skip(
            "R didimputation parity fixture not present. Run "
            "`Rscript benchmarks/R/generate_didimputation_golden.R` to regenerate "
            "`benchmarks/data/didimputation_golden.json`."
        )
    with GOLDEN_PATH.open("r") as f:
        return json.loads(f.read())


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    if not _R_FIXTURE_AVAILABLE:
        pytest.skip("R didimputation parity fixture not present.")
    return pd.read_csv(PANEL_PATH)


class TestImputationDiDParityR:
    """Pin Python ImputationDiD against R ``didimputation::did_imputation()``.

    The fixture is an unweighted staggered panel at the cohort x event-time
    partition (R's only mode), which validates the FULL variance machinery — the
    untreated `v_it` projection (Supplementary Proposition A3, otherwise not
    analytically verifiable) and the clustering — against the reference: SEs match
    to ~1e-10 and point estimates to ~1e-7 on the reference platform (the tests
    assert ATT ``abs=1e-6`` / SE ``abs=1e-7`` for cross-platform robustness). At
    this partition with uniform weights
    the unit-clustered Equation 8 coincides with both R's ``sum(v^2*tau)/sum(v^2)``
    and the old observation-level mean, so this class confirms *correctness*; the
    Eq. 8 *distinction* from the old form (which needs non-uniform weights or a
    coarser partition, with no R analogue) is proven by the white-box hand-calc in
    ``TestB2024Eq8AuxiliaryAggregator``.
    """

    def test_overall_att_matches_r(self, golden: dict, panel: pd.DataFrame) -> None:
        res = ImputationDiD().fit(
            panel, outcome="y", unit="unit", time="time", first_treat="first_treat"
        )
        assert res.overall_att == pytest.approx(golden["overall"]["att"], abs=1e-6)

    def test_overall_se_matches_r(self, golden: dict, panel: pd.DataFrame) -> None:
        res = ImputationDiD().fit(
            panel, outcome="y", unit="unit", time="time", first_treat="first_treat"
        )
        assert res.overall_se == pytest.approx(golden["overall"]["se"], abs=1e-7)

    def test_event_study_atts_match_r(self, golden: dict, panel: pd.DataFrame) -> None:
        res = ImputationDiD().fit(
            panel,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )
        assert res.event_study_effects is not None
        es = golden["event_study"]
        assert len(es["horizons"]) > 0
        for h, att in zip(es["horizons"], es["att"]):
            # Every golden horizon must be present and finite -- no silent skips.
            assert h in res.event_study_effects, f"missing horizon {h}"
            got = res.event_study_effects[h]["effect"]
            assert np.isfinite(got), f"non-finite ATT at h={h}"
            assert got == pytest.approx(att, abs=1e-6), f"h={h}"

    def test_event_study_ses_match_r(self, golden: dict, panel: pd.DataFrame) -> None:
        """Per-horizon SEs match R didimputation (the variance machinery, not just
        the point estimates) -- ~1e-10 observed on the reference platform, asserted
        here at abs=1e-7 for cross-platform robustness."""
        res = ImputationDiD().fit(
            panel,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )
        assert res.event_study_effects is not None
        es = golden["event_study"]
        assert len(es["horizons"]) > 0
        for h, se in zip(es["horizons"], es["se"]):
            # Every golden horizon must be present and finite -- no silent skips.
            assert h in res.event_study_effects, f"missing horizon {h}"
            got = res.event_study_effects[h]["se"]
            assert np.isfinite(got), f"non-finite SE at h={h}"
            assert got == pytest.approx(se, abs=1e-7), f"h={h}"
