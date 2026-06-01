"""Methodology + R-parity tests for the classic Synthetic Control estimator.

Covers Abadie-Diamond-Hainmueller (2010) ``SyntheticControl``:

* **Validation gates** (10 baked-in checks): predictor-period leakage, absorbing
  post-period suffix + no-anticipation cross-check, post canonicalization, donor
  filtering, empty windows, poor-fit warning, duplicate predictor labels,
  inner-solve non-convergence warning, order-independent gap path, and the
  ``standardize="none"`` deviation.
* **custom_v cross-field** + degenerate ``J==1`` / ``T0`` paths + ``get_params``
  round-trip + the NaN-inference contract.
* **Two-tier R `Synth` parity** on the Basque dataset (Abadie-Gardeazabal 2003):
  Tier-1 feeds R's ``solution.v`` through ``custom_v`` and asserts the donor
  weights match deterministically (optimizer-independent); Tier-2 checks the
  data-driven nested fit lands in a tolerance band (the nested V legitimately
  differs because our outer objective uses all pre periods, not R's
  ``time.optimize.ssr`` window).

The Basque fixtures live in ``tests/data/`` (not ``benchmarks/data/``) so the
deterministic Tier-1 test runs in isolated-install CI without R; regenerate via
``Rscript benchmarks/R/generate_synth_basque_golden.R``.
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from diff_diff import (
    DiagnosticReport,
    SyntheticControl,
    SyntheticControlResults,
    synthetic_control,
)
from tests.conftest import assert_nan_inference

DATA_DIR = Path(__file__).parent / "data"
GOLDEN_PATH = DATA_DIR / "synth_basque_golden.json"
PANEL_PATH = DATA_DIR / "synth_basque_panel.csv"

PREDICTORS = [
    "school.illit",
    "school.prim",
    "school.med",
    "school.high",
    "school.post.high",
    "invest",
]


# ---------------------------------------------------------------------------
# Cheap optimizer settings for behavior tests (pure-Python CI speed)
# ---------------------------------------------------------------------------
# Behavior tests only need a VALID, cleanly-converged fit, not data-driven V quality.
# The production nested defaults (n_starts=4, inner_max_iter=10000, inner_min_decrease=1e-5)
# cost 30-150s per *pure-Python* fit because the inner Frank-Wolfe solve grinds its slow
# sublinear tail to hit the tight tolerance on every objective evaluation. Loosening the
# inner tolerance + a single start + a small outer cap gives a clean ~0.1s fit without
# changing what these tests assert. Pure-Python coverage of the production-default nested
# path (n_starts=4 with the _v_starts heuristic candidates + the tight inner_min_decrease=1e-5)
# is kept by the dedicated non-slow ``test_nested_production_defaults_smoke`` (a 2-donor panel
# whose inner FW simplex is ~1-D, so defaults stay <0.1s). The @slow Tier-2 Basque test
# additionally covers the defaults in the Rust matrix, and the Rust<->numpy Frank-Wolfe kernel
# equivalence is locked by tests/test_rust_backend.py::test_sc_weight_fw_matches_numpy.
#
# NB: inner_max_iter is deliberately LEFT AT DEFAULT here — the speedup comes from the
# looser tolerance letting FW terminate on *convergence* (not on an iteration cap), so the
# solve stays clean (no non-convergence warning). Do NOT fold inner_max_iter into _FAST or
# the inner-non-convergence warning starts firing spuriously.
_FAST = dict(n_starts=1, optimizer_options={"maxiter": 50}, inner_min_decrease=1e-3)
# Churn tests deliberately force inner non-convergence (inner_max_iter=1); KEEP that and only
# cap the outer optimizer so it does not iterate to maxiter on the flat penalty landscape.
_FAST_CHURN = dict(n_starts=1, optimizer_options={"maxiter": 5})


# ---------------------------------------------------------------------------
# Synthetic panel builders (fast; no R needed)
# ---------------------------------------------------------------------------


def _make_panel(n_donors=4, T=8, T0=6, effect=3.0, seed=0):
    """Balanced panel where the treated unit is a convex mix of two donors."""
    rng = np.random.default_rng(seed)
    years = list(range(2000, 2000 + T))
    donors = {}
    for j in range(n_donors):
        base = rng.normal(10, 2)
        trend = rng.normal(0, 0.3)
        donors[j] = base + trend * np.arange(T) + rng.normal(0, 0.15, T)
    if n_donors >= 2:
        treated = 0.6 * donors[0] + 0.4 * donors[1] + rng.normal(0, 0.08, T)
    else:
        treated = donors[0] + rng.normal(0, 0.08, T)
    treated = treated.copy()
    treated[T0:] += effect
    rows = []
    for j in range(n_donors):
        for t in range(T):
            rows.append(
                {"unit": f"d{j}", "year": years[t], "y": donors[j][t], "treated": 0, "x": float(j)}
            )
    for t in range(T):
        rows.append(
            {
                "unit": "treated",
                "year": years[t],
                "y": treated[t],
                "treated": int(t >= T0),
                "x": 0.5,
            }
        )
    return pd.DataFrame(rows), years, T0


# ---------------------------------------------------------------------------
# Validation 1: predictor periods must be within the pre window (no leakage)
# ---------------------------------------------------------------------------


def test_predictor_window_outside_pre_rejected():
    df, years, T0 = _make_panel()
    post_year = years[T0]
    with pytest.raises(ValueError, match="outside the pre-treatment window"):
        SyntheticControl(seed=0).fit(
            df,
            "y",
            "treated",
            "unit",
            "year",
            predictors=["y"],
            predictor_window=[years[0], post_year],
        )


def test_special_predictor_period_outside_pre_rejected():
    df, years, T0 = _make_panel()
    with pytest.raises(ValueError, match="outside the pre-treatment window"):
        SyntheticControl(seed=0).fit(
            df,
            "y",
            "treated",
            "unit",
            "year",
            special_predictors=[("y", [years[T0]], "mean")],
        )


def test_pre_period_outcomes_outside_pre_rejected():
    df, years, T0 = _make_panel()
    with pytest.raises(ValueError, match="outside the pre-treatment window"):
        SyntheticControl(seed=0).fit(
            df, "y", "treated", "unit", "year", pre_period_outcomes=[years[T0]]
        )


# ---------------------------------------------------------------------------
# Validation 2: post must be a contiguous suffix + no-anticipation
# ---------------------------------------------------------------------------


def test_non_contiguous_post_rejected():
    df, years, T0 = _make_panel()
    # Drop a MIDDLE post period -> the remaining set is not a suffix of the axis.
    bad_post = [years[T0]] + years[T0 + 2 :]
    with pytest.raises(ValueError, match="contiguous suffix"):
        SyntheticControl(seed=0).fit(df, "y", "treated", "unit", "year", post_periods=bad_post)


def test_anticipation_in_pre_rejected():
    df, years, T0 = _make_panel()
    # Mark a pre period as treated for the treated unit, but declare the standard
    # post window -> D==1 appears inside the pre window (anticipation).
    df = df.copy()
    mask = (df["unit"] == "treated") & (df["year"] == years[T0 - 1])
    df.loc[mask, "treated"] = 1
    with pytest.raises(ValueError, match="no-anticipation"):
        SyntheticControl(seed=0).fit(df, "y", "treated", "unit", "year", post_periods=years[T0:])


def test_untreated_period_in_post_rejected():
    # Absorbing exposure: a D==0 period inside the (contiguous) post suffix must be
    # rejected, not averaged into the ATT (treated path 0,...,1,0 with post=[T0:]).
    df, years, T0 = _make_panel()
    df = df.copy()
    df.loc[(df["unit"] == "treated") & (df["year"] == years[-1]), "treated"] = 0
    with pytest.raises(ValueError, match="uninterrupted exposure|D==0 in post"):
        SyntheticControl(seed=0).fit(df, "y", "treated", "unit", "year", post_periods=years[T0:])


def test_non_binary_treatment_rejected():
    # A non-{0,1} treatment code must fail closed (else the unit is silently dropped
    # from both treated and donor sets, changing the donor pool / weights / ATT).
    df, years, T0 = _make_panel()
    df = df.copy()
    df.loc[(df["unit"] == "d0") & (df["year"] == years[0]), "treated"] = 2
    with pytest.raises(ValueError, match="binary"):
        synthetic_control(df, "y", "treated", "unit", "year", seed=0)


def test_missing_treatment_value_rejected():
    # A donor with a missing treatment cell would be silently classified by
    # groupby(...).max() (NaN dropped) — must fail closed before classification.
    df, years, T0 = _make_panel()
    df = df.copy()
    df.loc[(df["unit"] == "d0") & (df["year"] == years[0]), "treated"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        synthetic_control(df, "y", "treated", "unit", "year", seed=0)


def test_estimators_module_reexport():
    # Backward-compat import surface (mirrors SyntheticDiD / TwoWayFixedEffects).
    from diff_diff.estimators import SyntheticControl as SC

    assert SC is SyntheticControl


# ---------------------------------------------------------------------------
# Validation 3 + 9: explicit post canonicalized; gap path order-independent
# ---------------------------------------------------------------------------


def test_post_periods_canonicalized_and_gap_order_independent():
    df, years, T0 = _make_panel()
    ordered = years[T0:]
    scrambled = list(reversed(ordered)) + [ordered[-1]]  # unsorted + duplicate
    r1 = synthetic_control(
        df, "y", "treated", "unit", "year", post_periods=ordered, seed=0, **_FAST
    )
    r2 = synthetic_control(
        df, "y", "treated", "unit", "year", post_periods=scrambled, seed=0, **_FAST
    )
    assert r1.post_periods == r2.post_periods == ordered
    assert abs(r1.att - r2.att) < 1e-12
    gdf = r2.get_gap_df()
    # Calendar-sorted regardless of input order.
    assert list(gdf["period"]) == sorted(gdf["period"])
    assert (gdf[gdf["phase"] == "post"]["period"].tolist()) == ordered


# ---------------------------------------------------------------------------
# Validation 4: donor pool filtering
# ---------------------------------------------------------------------------


def test_donor_pool_restricts_donors():
    df, years, T0 = _make_panel(n_donors=4)
    res = synthetic_control(
        df, "y", "treated", "unit", "year", donor_pool=["d0", "d1"], seed=0, **_FAST
    )
    assert res.n_donors == 2
    assert set(res.get_weights_df()["unit"]) <= {"d0", "d1"}


def test_contaminated_donor_pool_rejected():
    df, years, T0 = _make_panel()
    # The treated unit itself must never appear in the donor pool.
    with pytest.raises(ValueError, match="treated unit|ever-treated|never-treated"):
        synthetic_control(df, "y", "treated", "unit", "year", donor_pool=["d0", "treated"], seed=0)


def test_ever_treated_donor_rejected():
    # A second ever-treated unit (not the designated treated) cannot be a donor.
    df, years, T0 = _make_panel()
    df = df.copy()
    df.loc[(df["unit"] == "d0") & (df["year"] >= years[T0]), "treated"] = 1
    with pytest.raises(ValueError, match="ever-treated|never-treated"):
        synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            treated_unit="treated",
            donor_pool=["d0", "d1"],
            seed=0,
        )


# ---------------------------------------------------------------------------
# Validation 5: empty windows rejected
# ---------------------------------------------------------------------------


def test_empty_predictor_window_rejected():
    df, _, _ = _make_panel()
    with pytest.raises(ValueError, match="must not be empty"):
        SyntheticControl(seed=0).fit(
            df, "y", "treated", "unit", "year", predictors=["y"], predictor_window=[]
        )


def test_empty_special_period_list_rejected():
    df, _, _ = _make_panel()
    with pytest.raises(ValueError, match="must not be empty"):
        SyntheticControl(seed=0).fit(
            df, "y", "treated", "unit", "year", special_predictors=[("y", [], "mean")]
        )


# ---------------------------------------------------------------------------
# Fail-closed on non-finite data entering the matching problem
# ---------------------------------------------------------------------------


def test_non_finite_predictor_rejected():
    # PARTIAL missingness in a predictor window: fail closed (deliberate deviation
    # from R Synth's na.rm=TRUE — see REGISTRY). All-NA windows behave identically.
    df, years, T0 = _make_panel()
    df = df.copy()
    df.loc[(df["unit"] == "d0") & (df["year"] == years[0]), "x"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        SyntheticControl(seed=0).fit(
            df,
            "y",
            "treated",
            "unit",
            "year",
            predictors=["x"],
            predictor_window=[years[0], years[1]],
        )


def test_all_na_predictor_window_rejected():
    # FULLY-missing predictor window: same fail-closed contract as partial (no na.rm).
    df, years, T0 = _make_panel()
    df = df.copy()
    df.loc[(df["unit"] == "d0") & (df["year"].isin([years[0], years[1]])), "x"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        SyntheticControl(seed=0).fit(
            df,
            "y",
            "treated",
            "unit",
            "year",
            predictors=["x"],
            predictor_window=[years[0], years[1]],
        )


def test_outer_v_nonconvergence_warning():
    # Outer V-search non-convergence must not be silent (optimizer capped at 1 iter).
    df, _, _ = _make_panel()
    with pytest.warns(UserWarning, match="Outer V-search"):
        # maxiter=1 forces the OUTER non-convergence; n_starts=1 + a loose inner tolerance
        # keep the (still-real) inner solves cheap. Loosening inner_min_decrease does not
        # affect whether the outer optimizer hits its 1-iteration cap.
        synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            seed=0,
            n_starts=1,
            optimizer_options={"maxiter": 1},
            inner_min_decrease=1e-3,
        )


def test_inner_v_search_nonconvergence_warning():
    # Intermediate inner solves during the nested V search must not be silent: forcing
    # inner_max_iter=1 makes them truncate, and the estimator emits an aggregated warning.
    df, _, _ = _make_panel()
    with pytest.warns(UserWarning, match="during nested V selection"):
        synthetic_control(
            df, "y", "treated", "unit", "year", seed=0, inner_max_iter=1, **_FAST_CHURN
        )


def test_single_inner_nonconvergence_excluded_from_v_ranking(monkeypatch):
    # A single LOW-RATE non-converged objective evaluation must be EXCLUDED from V
    # ranking (penalized out of the argmin), not merely warned about: force exactly one
    # objective eval (the uniform-start eval, max(v) < 0.9) to report conv=False and
    # assert (a) the any-occurrence warning fires, and (b) the selected V is a genuine
    # small-MSPE fit (mspe_v << penalty), i.e. the truncated candidate did not win.
    import importlib

    # NB: ``diff_diff.synthetic_control`` the attribute is the convenience *function*
    # (it shadows the submodule, same as ``diff_diff.trop``), so reach the module via
    # importlib to monkeypatch its module-global _inner_solve_W.
    sc = importlib.import_module("diff_diff.synthetic_control")

    df, _, _ = _make_panel()
    real_solve = sc._inner_solve_W
    state = {"failed": False}

    def patched(X1s, X0s, v, max_iter, min_decrease):
        w, conv = real_solve(X1s, X0s, v, max_iter, min_decrease)
        if not state["failed"] and float(np.max(v)) < 0.9:  # a spread V => an objective eval
            state["failed"] = True
            return w, False
        return w, conv

    monkeypatch.setattr(sc, "_inner_solve_W", patched)
    with pytest.warns(UserWarning, match="during nested V selection"):
        res = synthetic_control(df, "y", "treated", "unit", "year", seed=0, **_FAST)

    assert state["failed"]  # the patch actually fired on an objective evaluation
    assert np.isfinite(res.att)
    # Exclusion proof: the chosen V's outer-objective MSPE is a real (small) value, not
    # the large penalty a truncated candidate would have carried.
    assert res.mspe_v is not None and res.mspe_v < 1.0


def test_n_starts_one_runs():
    # n_starts=1 uses only the uniform start (short-circuits the heuristic candidates)
    # and still produces a valid nested fit.
    df, _, _ = _make_panel()
    res = synthetic_control(
        df,
        "y",
        "treated",
        "unit",
        "year",
        seed=0,
        n_starts=1,
        optimizer_options={"maxiter": 50},
        inner_min_decrease=1e-3,
    )
    assert np.isfinite(res.att)
    assert abs(sum(res.donor_weights.values()) - 1.0) < 1e-6


def test_nested_production_defaults_smoke():
    # Coverage anchor: exercise the FULL production-default nested path end-to-end in
    # pure-Python — n_starts=4 (so the _v_starts heuristic candidates: inverse-variance,
    # univariate-fit and Dirichlet starts are generated, which the n_starts=1 _FAST tests
    # skip) and the tight inner_min_decrease=1e-5. A 2-donor panel keeps the inner
    # Frank-Wolfe simplex effectively 1-D, so the default settings still run in <0.1s and
    # this stays non-slow. The @slow Tier-2 Basque test covers the defaults only in the Rust
    # matrix; this is the pure-Python complement.
    df, _, _ = _make_panel(n_donors=2)
    res = synthetic_control(df, "y", "treated", "unit", "year", seed=0)  # production defaults
    assert np.isfinite(res.att)
    assert abs(sum(res.donor_weights.values()) - 1.0) < 1e-6
    assert res.n_donors == 2
    assert res.mspe_v is not None  # nested V was selected by minimizing pre-period MSPE


def test_non_finite_outcome_rejected():
    df, years, T0 = _make_panel()
    df = df.copy()
    df.loc[(df["unit"] == "d1") & (df["year"] == years[2]), "y"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        synthetic_control(df, "y", "treated", "unit", "year", seed=0)


def test_distinct_special_period_sets_not_duplicate():
    # Same var/op, same endpoints + length, different intermediate period -> distinct
    # predictors, must NOT be rejected as duplicates.
    df, years, T0 = _make_panel(T=8, T0=6)
    res = SyntheticControl(seed=0, **_FAST).fit(
        df,
        "y",
        "treated",
        "unit",
        "year",
        special_predictors=[
            ("y", [years[0], years[2], years[4]], "mean"),
            ("y", [years[0], years[3], years[4]], "mean"),
        ],
    )
    assert len(res.v_weights) == 2
    assert len(set(res.v_weights)) == 2  # two distinct labels


def test_reordered_special_periods_are_duplicates():
    # Same var/op with reordered periods canonicalize to the same spec -> duplicate.
    df, years, T0 = _make_panel(T=8, T0=6)
    with pytest.raises(ValueError, match="Duplicate predictor label"):
        SyntheticControl(seed=0).fit(
            df,
            "y",
            "treated",
            "unit",
            "year",
            special_predictors=[
                ("y", [years[0], years[1], years[2]], "mean"),
                ("y", [years[2], years[1], years[0]], "mean"),
            ],
        )


def test_duplicate_predictor_window_periods_deduped():
    # A repeated period in predictor_window must not re-weight the mean: the
    # deduped window [y0,y0,y1] matches the explicit [y0,y1].
    df, years, T0 = _make_panel()
    r_dup = synthetic_control(
        df,
        "y",
        "treated",
        "unit",
        "year",
        predictors=["y"],
        predictor_window=[years[0], years[0], years[1]],
        seed=0,
        **_FAST,
    )
    r_uniq = synthetic_control(
        df,
        "y",
        "treated",
        "unit",
        "year",
        predictors=["y"],
        predictor_window=[years[0], years[1]],
        seed=0,
        **_FAST,
    )
    assert abs(r_dup.att - r_uniq.att) < 1e-9


def test_median_op_rejected():
    # median is a non-linear aggregation, not an ADH linear combination.
    df, _, _ = _make_panel()
    with pytest.raises(ValueError, match="must be one of"):
        SyntheticControl(seed=0).fit(
            df, "y", "treated", "unit", "year", predictors=["x"], predictors_op="median"
        )


# ---------------------------------------------------------------------------
# Validation 6: poor pre-fit warning
# ---------------------------------------------------------------------------


def test_poor_fit_warning():
    # Donors are all ~constant near 10; treated is centred near 50 with a trend,
    # so no convex combination can reproduce it -> RMSPE >> SD(treated pre).
    rng = np.random.default_rng(1)
    years = list(range(2000, 2010))
    T0 = 7
    rows = []
    for j in range(4):
        for t, yr in enumerate(years):
            rows.append({"unit": f"d{j}", "year": yr, "y": 10 + rng.normal(0, 0.1), "treated": 0})
    for t, yr in enumerate(years):
        rows.append({"unit": "treated", "year": yr, "y": 50 + 2.0 * t, "treated": int(t >= T0)})
    df = pd.DataFrame(rows)
    with pytest.warns(UserWarning, match="Pre-treatment fit is poor"):
        synthetic_control(df, "y", "treated", "unit", "year", seed=0, **_FAST)


def test_poor_fit_warning_flat_treated_pre_path():
    # Flat treated pre-path (SD == 0) that donors near 10 cannot reproduce: RMSPE > 0
    # must still warn (the former `pre_sd > 0` gate suppressed this case).
    rng = np.random.default_rng(2)
    years = list(range(2000, 2010))
    T0 = 7
    rows = []
    for j in range(4):
        for yr in years:
            rows.append({"unit": f"d{j}", "year": yr, "y": 10 + rng.normal(0, 0.1), "treated": 0})
    for i, yr in enumerate(years):
        rows.append(
            {"unit": "treated", "year": yr, "y": (5.0 if i < T0 else 8.0), "treated": int(i >= T0)}
        )
    df = pd.DataFrame(rows)
    with pytest.warns(UserWarning, match="Pre-treatment fit is poor"):
        synthetic_control(df, "y", "treated", "unit", "year", seed=0, **_FAST)


# ---------------------------------------------------------------------------
# Validation 7: duplicate predictor labels rejected
# ---------------------------------------------------------------------------


def test_duplicate_predictor_label_rejected():
    df, years, T0 = _make_panel()
    pre = years[:T0]
    with pytest.raises(ValueError, match="Duplicate predictor label"):
        SyntheticControl(seed=0).fit(
            df,
            "y",
            "treated",
            "unit",
            "year",
            special_predictors=[("y", pre, "mean"), ("y", pre, "mean")],
        )


def test_duplicate_regular_predictor_rejected():
    df, _, _ = _make_panel()
    with pytest.raises(ValueError, match="Duplicate predictor label"):
        SyntheticControl(seed=0).fit(df, "y", "treated", "unit", "year", predictors=["x", "x"])


# ---------------------------------------------------------------------------
# Validation 8: inner-solve non-convergence warning
# ---------------------------------------------------------------------------


def test_inner_nonconvergence_warning():
    df, _, _ = _make_panel(n_donors=4)
    with pytest.warns(UserWarning, match="did not converge"):
        SyntheticControl(seed=0, v_method="nested", inner_max_iter=1, **_FAST_CHURN).fit(
            df, "y", "treated", "unit", "year"
        )


# ---------------------------------------------------------------------------
# Validation 10: standardize="none" deviation runs
# ---------------------------------------------------------------------------


def test_standardize_none_runs():
    df, _, _ = _make_panel()
    res = synthetic_control(df, "y", "treated", "unit", "year", standardize="none", seed=0, **_FAST)
    assert res.standardize == "none"
    assert np.isfinite(res.att)


# ---------------------------------------------------------------------------
# custom_v cross-field rules (fail-closed)
# ---------------------------------------------------------------------------


def test_custom_v_required_when_method_custom():
    with pytest.raises(ValueError, match="custom_v is required"):
        SyntheticControl(v_method="custom")


def test_custom_v_rejected_when_method_nested():
    with pytest.raises(ValueError, match="must be None when v_method='nested'"):
        SyntheticControl(v_method="nested", custom_v=[1.0, 1.0])


def test_custom_v_negative_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        SyntheticControl(v_method="custom", custom_v=[1.0, -1.0])


def test_custom_v_wrong_length_rejected():
    df, _, _ = _make_panel()
    # 3 entries but the default (all-pre-outcomes) predictor count differs.
    with pytest.raises(ValueError, match="custom_v has length"):
        SyntheticControl(v_method="custom", custom_v=[1.0, 1.0, 1.0]).fit(
            df, "y", "treated", "unit", "year"
        )


# ---------------------------------------------------------------------------
# Degenerate paths: J==1, T0==0, T0==1
# ---------------------------------------------------------------------------


def test_single_donor_degenerate_warns():
    df, _, _ = _make_panel(n_donors=1)
    with pytest.warns(UserWarning, match="single donor"):
        res = synthetic_control(df, "y", "treated", "unit", "year", seed=0)
    assert res.n_donors == 1
    assert abs(sum(res.donor_weights.values()) - 1.0) < 1e-9


def test_no_pre_period_rejected():
    # All periods treated for the treated unit -> no pre period.
    rows = []
    years = [2000, 2001]
    for j in range(3):
        for yr in years:
            rows.append({"unit": f"d{j}", "year": yr, "y": 10.0 + yr, "treated": 0})
    for yr in years:
        rows.append({"unit": "treated", "year": yr, "y": 12.0 + yr, "treated": 1})
    df = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="No pre-treatment periods|Cannot infer"):
        synthetic_control(df, "y", "treated", "unit", "year", seed=0)


def test_single_pre_period_nested_warns():
    rows = []
    years = [2000, 2001, 2002]
    rng = np.random.default_rng(0)
    for j in range(3):
        for yr in years:
            rows.append({"unit": f"d{j}", "year": yr, "y": 10.0 + rng.normal(), "treated": 0})
    for i, yr in enumerate(years):
        rows.append({"unit": "treated", "year": yr, "y": 11.0 + i, "treated": int(i >= 1)})
    df = pd.DataFrame(rows)
    with pytest.warns(UserWarning, match="single pre period"):
        synthetic_control(df, "y", "treated", "unit", "year", seed=0)


def test_multiple_treated_units_rejected():
    df, _, _ = _make_panel()
    df = df.copy()
    df.loc[(df["unit"] == "d0") & (df["year"] >= 2006), "treated"] = 1
    with pytest.raises(ValueError, match="exactly one"):
        synthetic_control(df, "y", "treated", "unit", "year", seed=0)


# ---------------------------------------------------------------------------
# sklearn-like API: get_params round-trip + transactional set_params
# ---------------------------------------------------------------------------


def test_get_set_params_roundtrip():
    est = SyntheticControl(n_starts=3, standardize="none", alpha=0.1, seed=7)
    params = est.get_params()
    assert set(params) == {
        "v_method",
        "custom_v",
        "optimizer_options",
        "n_starts",
        "inner_max_iter",
        "inner_min_decrease",
        "standardize",
        "alpha",
        "seed",
    }
    est2 = SyntheticControl().set_params(**params)
    assert est2.get_params() == params


def test_set_params_rolls_back_on_invalid():
    est = SyntheticControl(alpha=0.05)
    with pytest.raises(ValueError):
        est.set_params(alpha=1.5)
    assert est.alpha == 0.05  # unchanged after failed update


# ---------------------------------------------------------------------------
# NaN-inference contract + result accessors
# ---------------------------------------------------------------------------


def test_nan_inference_contract():
    df, _, _ = _make_panel()
    res = synthetic_control(df, "y", "treated", "unit", "year", seed=0, **_FAST)
    assert_nan_inference(
        {"se": res.se, "t_stat": res.t_stat, "p_value": res.p_value, "conf_int": res.conf_int}
    )
    assert np.isfinite(res.att)


def test_result_accessors_render():
    df, _, _ = _make_panel()
    res = synthetic_control(df, "y", "treated", "unit", "year", seed=0, **_FAST)
    assert isinstance(res, SyntheticControlResults)
    assert isinstance(res.summary(), str) and "Synthetic Control" in res.summary()
    assert "att" in res.to_dict()
    assert res.to_dataframe().shape[0] == 1
    gdf = res.get_gap_df()
    assert set(gdf.columns) == {"period", "gap", "phase"}
    wdf = res.get_weights_df()
    assert list(wdf.columns) == ["unit", "weight"]
    # PR-2: fit() populates the placebo refit snapshot and the treated unit's
    # RMSPE ratio; the placebo reference distribution is not computed until
    # in_space_placebo() runs (placebo_p_value stays NaN, gaps/df unset).
    assert res._fit_snapshot is not None
    assert res._placebo_gaps is None and res._placebo_df is None
    assert np.isfinite(res.rmspe_ratio)
    assert np.isnan(res.placebo_p_value) and res.n_placebos == 0


def test_inferred_post_matches_explicit():
    df, years, T0 = _make_panel()
    r_inf = synthetic_control(df, "y", "treated", "unit", "year", seed=0, **_FAST)
    r_exp = synthetic_control(
        df, "y", "treated", "unit", "year", post_periods=years[T0:], seed=0, **_FAST
    )
    assert r_inf.post_periods == r_exp.post_periods == years[T0:]
    assert abs(r_inf.att - r_exp.att) < 1e-12


# ---------------------------------------------------------------------------
# R-parity (Basque / Abadie-Gardeazabal 2003 via R `Synth`)
# ---------------------------------------------------------------------------


def _load_golden():
    if not GOLDEN_PATH.exists() or not PANEL_PATH.exists():
        pytest.skip(
            "Basque golden fixtures missing — regenerate via "
            "`Rscript benchmarks/R/generate_synth_basque_golden.R`."
        )
    return json.load(open(GOLDEN_PATH)), pd.read_csv(PANEL_PATH)


def _basque_kwargs(golden):
    special = [
        (
            s["var"],
            list(s["periods"]) if isinstance(s["periods"], list) else [s["periods"]],
            s["op"],
        )
        for s in golden["config"]["special"]
    ]
    return dict(
        treated_unit=golden["config"]["treated_regionno"],
        donor_pool=list(golden["config"]["controls"]),
        predictors=PREDICTORS,
        predictors_op="mean",
        predictor_window=list(range(1964, 1970)),
        special_predictors=special,
    )


def test_basque_tier1_custom_v_parity():
    """Tier-1 (hard gate): given R's solution.v, donor weights match R deterministically."""
    golden, df = _load_golden()
    custom_v = np.asarray(golden["solution_v"], dtype=float)
    res = SyntheticControl(v_method="custom", custom_v=custom_v).fit(
        df, "gdpcap", "treated", "regionno", "year", **_basque_kwargs(golden)
    )
    # Predictor matrix + ordering: X1 matches R's dataprep exactly.
    X1_py = res.predictor_balance["treated"].to_numpy(dtype=float)
    X1_r = np.asarray(golden["X1"], dtype=float)
    np.testing.assert_allclose(X1_py, X1_r, atol=1e-6)

    # Donor weights match R's solution.w (the published Cataluna/Madrid mix).
    controls = sorted(int(c) for c in golden["config"]["controls"])
    w_r = {int(k): v for k, v in golden["solution_w"].items()}
    w_py = {int(k): v for k, v in res.donor_weights.items()}
    wr = np.array([w_r.get(c, 0.0) for c in controls])
    wp = np.array([w_py.get(c, 0.0) for c in controls])
    np.testing.assert_allclose(wp, wr, atol=1e-3)
    # Published anchor: region 10 ~ 0.85, region 14 ~ 0.15.
    assert w_py.get(10, 0) > 0.80 and w_py.get(14, 0) > 0.10


@pytest.mark.slow
def test_basque_tier2_nested_band():
    """Tier-2 (band): the data-driven nested fit lands near R's solution.

    Loose by design — our outer objective minimizes MSPE over all pre periods,
    while R uses the ``time.optimize.ssr`` (1960-1969) window, so the nested V
    legitimately differs; multistart Nelder-Mead/Powell is also BLAS/platform
    sensitive. We check fit quality and that the dominant donors agree.
    """
    golden, df = _load_golden()
    res = SyntheticControl(v_method="nested", seed=0).fit(
        df, "gdpcap", "treated", "regionno", "year", **_basque_kwargs(golden)
    )
    r_sqrt_loss = golden["loss_v"] ** 0.5
    assert res.pre_rmspe <= r_sqrt_loss * 1.5  # comparable pre-fit quality

    years = np.asarray(golden["years"])
    r_att = float(np.asarray(golden["gap"])[years >= 1970].mean())
    assert abs(res.att - r_att) < 0.2  # avg post-gap within band

    # Dominant donors agree with R (Cataluna region 10, Madrid region 14).
    top2 = [u for u, _ in sorted(res.donor_weights.items(), key=lambda kv: -kv[1])[:2]]
    assert set(top2) == {10, 14}
    assert res.donor_weights.get(10, 0) + res.donor_weights.get(14, 0) > 0.7


def test_basque_tier1_leave_one_out_parity():
    """Tier-1 LOO (deterministic): dropping the dominant donor (region 10) with R's
    ``solution.v`` held fixed, the reduced-pool refit's ATT and gap path match R's
    drop-donor ``synth`` exactly (a direct R anchor on the reduced-pool W-solve;
    ``leave_one_out()`` on a custom-V fit reuses that fixed V on the donor pool minus
    the dropped unit). Region 10 carries ~85% of the full-pool weight, so dropping it
    swings the synthetic onto regions 7+14 — the single-donor-dependence signal LOO
    exists to surface."""
    golden, df = _load_golden()
    if "leave_one_out" not in golden:
        pytest.skip("LOO golden missing — regenerate via the R script.")
    loo_g = golden["leave_one_out"]
    dropped = int(loo_g["dropped_regionno"])
    custom_v = np.asarray(golden["solution_v"], dtype=float)
    res = SyntheticControl(v_method="custom", custom_v=custom_v).fit(
        df, "gdpcap", "treated", "regionno", "year", **_basque_kwargs(golden)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loo = res.leave_one_out()
    row = loo[(loo["status"] == "loo") & (loo["dropped_unit"] == dropped)]
    assert len(row) == 1
    assert float(row["att"].iloc[0]) == pytest.approx(float(loo_g["att"]), abs=1e-2)
    # Full reduced-pool gap trajectory (1955-1997) matches R's drop-donor synth.
    gaps = res.get_leave_one_out_gaps()
    gap_py = gaps[gaps["dropped_unit"] == dropped].sort_values("period")["gap"].to_numpy()
    np.testing.assert_allclose(gap_py, np.asarray(loo_g["gap"], dtype=float), atol=2e-2)


# ---------------------------------------------------------------------------
# In-space placebo permutation inference (Abadie-Diamond-Hainmueller 2010 §2.4)
# ---------------------------------------------------------------------------


def _fit_for_placebo(n_donors=4, effect=3.0, **kw):
    """Fit with cheap settings on a panel carrying a strong post-treatment effect."""
    df, _, _ = _make_panel(n_donors=n_donors, effect=effect)
    opts = dict(_FAST)
    opts.update(kw)
    with warnings.catch_warnings():  # single-donor / poor-fit fit warnings are not under test
        warnings.simplefilter("ignore")
        return synthetic_control(df, "y", "treated", "unit", "year", seed=0, **opts)


def test_in_space_placebo_strong_effect_ranks_treated_first():
    # A 3.0-unit post effect on a treated unit that is a clean convex mix of two
    # donors -> treated RMSPE ratio is the most extreme -> rank 1 -> p = 1/(J+1).
    res = _fit_for_placebo(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pdf = res.in_space_placebo()
    assert res.n_placebos == 4 and res.n_failed == 0
    treated_ratio = pdf.loc[pdf["is_treated"], "rmspe_ratio"].iloc[0]
    placebo_ratios = pdf.loc[~pdf["is_treated"], "rmspe_ratio"]
    assert (treated_ratio > placebo_ratios).all()  # treated is the most extreme unit
    assert res.placebo_p_value == pytest.approx(1 / (res.n_placebos + 1))
    # Exactly one treated row; the placebo rows are exactly the donor units.
    assert int(pdf["is_treated"].sum()) == 1
    assert pdf.loc[pdf["is_treated"], "unit"].iloc[0] == "treated"
    assert set(pdf.loc[~pdf["is_treated"], "unit"]) == {"d0", "d1", "d2", "d3"}


def test_in_space_placebo_excludes_real_treated_from_donor_pools():
    # The real treated unit is never in the donor universe, so it cannot serve as
    # a donor for any placebo (ADH 2010 contamination guard; SCtools convention).
    res = _fit_for_placebo(n_donors=4)
    snap = res._fit_snapshot
    assert snap.treated_id == "treated" and "treated" not in snap.donor_ids
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res.in_space_placebo()
    # Each donor became a placebo exactly once; the treated unit is not a placebo.
    assert "treated" not in res._placebo_gaps
    assert set(res._placebo_gaps) == set(snap.donor_ids)


def test_in_space_placebo_p_in_valid_discrete_set():
    res = _fit_for_placebo(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res.in_space_placebo()
    valid = {(k + 1) / (res.n_placebos + 1) for k in range(res.n_placebos + 1)}
    assert any(res.placebo_p_value == pytest.approx(v) for v in valid)


def test_in_space_placebo_does_not_touch_analytical_inference():
    # The permutation p-value is SEPARATE from the analytical fields, which stay
    # NaN; is_significant stays bound to the (NaN) p_value, not placebo_p_value.
    res = _fit_for_placebo(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res.in_space_placebo()
    assert np.isfinite(res.placebo_p_value)
    assert_nan_inference(
        {"se": res.se, "t_stat": res.t_stat, "p_value": res.p_value, "conf_int": res.conf_int}
    )
    assert res.is_significant is False


def test_in_space_placebo_deterministic():
    res = _fit_for_placebo(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p1 = res.in_space_placebo()
        first_p = res.placebo_p_value
        p2 = res.in_space_placebo()
    assert res.placebo_p_value == first_p  # bit-equal p-value across runs
    pd.testing.assert_frame_equal(p1, p2)  # identical rows AND row order


def test_in_space_placebo_requires_two_donors():
    res = _fit_for_placebo(n_donors=1)
    with pytest.warns(UserWarning, match="at least 2 donors"):
        pdf = res.in_space_placebo()
    assert np.isnan(res.placebo_p_value) and res.n_placebos == 0
    assert len(pdf) == 1 and bool(pdf["is_treated"].iloc[0])


def test_in_space_placebo_two_donors_warns_coarse():
    res = _fit_for_placebo(n_donors=2)
    with pytest.warns(UserWarning, match="coarse"):
        res.in_space_placebo()
    # 2 placebos -> reference set of 3 -> p in {1/3, 2/3, 1}.
    assert res.n_placebos == 2
    assert any(res.placebo_p_value == pytest.approx(v) for v in (1 / 3, 2 / 3, 1.0))


def test_in_space_placebo_fails_closed_on_nonconverged_treated_fit():
    # inner_max_iter=1 truncates the treated unit's own Frank-Wolfe solve, so its
    # RMSPE ratio is not a valid optimum. in_space_placebo() must fail closed
    # (NaN p-value + warning), NOT rank a truncated treated statistic.
    df, _, _ = _make_panel(n_donors=4, effect=3.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            seed=0,
            n_starts=1,
            inner_max_iter=1,
            optimizer_options={"maxiter": 5},
        )
    assert res._fit_converged is False  # treated fit was truncated
    with pytest.warns(UserWarning, match="did not converge at fit time"):
        pdf = res.in_space_placebo()
    assert np.isnan(res.placebo_p_value)
    assert res.n_placebos == 0 and res.n_failed == 0  # the placebo loop never ran
    assert len(pdf) == 1 and bool(pdf["is_treated"].iloc[0])  # treated row only


def test_in_space_placebo_pickle_drops_snapshot_keeps_scalars():
    res = _fit_for_placebo(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res.in_space_placebo()
    restored = pickle.loads(pickle.dumps(res))
    # Scalars survive; panel-derived state is dropped.
    assert restored.placebo_p_value == res.placebo_p_value
    assert restored.rmspe_ratio == res.rmspe_ratio
    assert restored.n_placebos == res.n_placebos and restored.n_failed == res.n_failed
    assert restored._fit_snapshot is None and restored._placebo_gaps is None
    # The small aggregate table survives, so get_placebo_df still works...
    assert len(restored.get_placebo_df()) == len(res.get_placebo_df())
    # ...but a re-run of the refit raises (the snapshot is gone).
    with pytest.raises(ValueError, match="requires the fit snapshot"):
        restored.in_space_placebo()


def test_in_space_placebo_custom_v_path():
    df, _, _ = _make_panel(n_donors=4)
    # Default predictors = all pre-period outcomes -> k = number of pre periods (T0).
    k = 6
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            v_method="custom",
            custom_v=np.ones(k),
            inner_min_decrease=1e-3,
        )
        pdf = res.in_space_placebo()
    assert res.n_placebos == 4 and np.isfinite(res.placebo_p_value)
    assert len(pdf) == 5


def test_get_placebo_df_before_run_returns_treated_row_only():
    res = _fit_for_placebo(n_donors=4)
    pdf = res.get_placebo_df()
    assert len(pdf) == 1
    assert bool(pdf["is_treated"].iloc[0]) and pdf["status"].iloc[0] == "treated"
    assert set(pdf.columns) == {
        "unit",
        "pre_mspe",
        "post_mspe",
        "rmspe_ratio",
        "is_treated",
        "status",
    }


def test_rmspe_ratio_floors_zero_pre_mspe():
    # Perfect pre-fit (pre-MSPE == 0) must yield a large-but-finite ratio, not
    # inf/nan (which would corrupt the permutation rank).
    from diff_diff.synthetic_control import _rmspe_ratio

    pre = np.zeros(5)
    assert np.isfinite(_rmspe_ratio(pre, np.array([1.0, 2.0, 3.0]), scale=10.0))
    # A zero-effect (post all zero) placebo has ratio 0 — the least extreme.
    assert _rmspe_ratio(pre, np.zeros(3), scale=10.0) == 0.0


def test_in_space_placebo_perfect_treated_fit_finite_ratio():
    # 2-donor panel where the treated unit EQUALS d0 in the pre-period -> the inner
    # FW solve lands on w=[1, 0], so the treated pre-MSPE is (bit-)exactly 0. The
    # RMSPE ratio must stay FINITE (scale-aware floor), never inf/nan.
    rng = np.random.default_rng(2)
    T, T0 = 8, 6
    years = list(range(2000, 2000 + T))
    d0 = rng.normal(10, 2, T)
    d1 = rng.normal(5, 2, T)
    treated = d0.copy()
    treated[T0:] += 5.0  # identical to d0 in the pre-period, clean post effect
    rows = []
    for name, series in (("d0", d0), ("d1", d1)):
        for t in range(T):
            rows.append({"unit": name, "year": years[t], "y": series[t], "treated": 0})
    for t in range(T):
        rows.append({"unit": "treated", "year": years[t], "y": treated[t], "treated": int(t >= T0)})
    df = pd.DataFrame(rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            v_method="custom",
            custom_v=np.ones(T0),
            inner_min_decrease=1e-3,
        )
    assert res.pre_rmspe == pytest.approx(0.0, abs=1e-9)
    assert np.isfinite(res.rmspe_ratio) and res.rmspe_ratio > 0


def test_in_space_placebo_immune_to_post_fit_mutation():
    # The fit snapshot must COPY caller-owned mutable inputs (custom_v,
    # optimizer_options), so mutating them after fit() cannot silently change
    # in_space_placebo() output on an already-returned results object.
    df, _, _ = _make_panel(n_donors=4)
    cv = np.ones(6)  # k = 6 default pre-period-outcome predictors
    opts = {"maxiter": 50}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            v_method="custom",
            custom_v=cv,
            optimizer_options=opts,
            inner_min_decrease=1e-3,
        )
        p1 = res.in_space_placebo().copy()
        pval1 = res.placebo_p_value
    snap = res._fit_snapshot
    assert snap.custom_v is not cv and snap.optimizer_options is not opts
    # Mutate the caller-owned originals AFTER fit -> placebo output must not change.
    cv[:] = [1e6, 1.0, 1.0, 1.0, 1.0, 1.0]
    opts["maxiter"] = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p2 = res.in_space_placebo().copy()
    assert res.placebo_p_value == pval1
    pd.testing.assert_frame_equal(p1, p2)


def test_get_placebo_df_includes_failed_donors(monkeypatch):
    # When the treated fit IS valid but some per-donor placebo refits fail to
    # converge, get_placebo_df() must still list EVERY unit (treated + each donor)
    # so callers can tell which donors failed -> exactly n_donors + 1 rows.
    # (A truncated treated fit instead fails the whole placebo run closed, tested
    # separately; here we simulate isolated donor failures with a converged treated
    # fit by monkeypatching the per-donor refit to fail for the first two donors.)
    import importlib

    # diff_diff.synthetic_control the SUBMODULE is shadowed by the re-exported
    # synthetic_control FUNCTION on the package, so import the module explicitly.
    sc = importlib.import_module("diff_diff.synthetic_control")

    res = _fit_for_placebo(n_donors=4)  # treated fit converges (normal settings)
    real_fit_unit = sc._placebo_fit_unit
    calls = {"n": 0}

    def flaky_fit_unit(snap, unit, donor_pool, n_starts):
        calls["n"] += 1
        if calls["n"] <= 2:  # first two donor refits "fail"
            return None
        return real_fit_unit(snap, unit, donor_pool, n_starts)

    monkeypatch.setattr(sc, "_placebo_fit_unit", flaky_fit_unit)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pdf = res.in_space_placebo()
    assert len(pdf) == res.n_donors + 1  # treated + every donor, regardless of failures
    assert res.n_failed == 2 and res.n_placebos == res.n_donors - 2
    failed = pdf[pdf["status"] == "failed"]
    assert len(failed) == 2 and failed["rmspe_ratio"].isna().all()  # NaN metrics


def test_in_space_placebo_fails_closed_on_underoptimized_outer_v():
    # An under-optimized OUTER V search (optimizer maxiter=1) leaves the treated
    # fit's V non-optimal even though the inner solve converges. Its RMSPE ratio is
    # therefore not a valid optimum, so in_space_placebo() must FAIL CLOSED rather
    # than silently rank an anti-conservatively under-optimized statistic.
    df, _, _ = _make_panel(n_donors=4, effect=3.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            seed=0,
            n_starts=1,
            optimizer_options={"maxiter": 1},  # outer V search cannot converge
            inner_min_decrease=1e-3,  # inner still converges -> isolates the outer path
        )
    assert res._fit_converged is False  # outer V non-convergence -> invalid fit
    with pytest.warns(UserWarning, match="did not converge at fit time"):
        res.in_space_placebo()
    assert np.isnan(res.placebo_p_value)
    assert res.n_placebos == 0 and res.n_failed == 0  # placebo loop never ran


def test_outer_v_convergence_tracks_selected_incumbent(monkeypatch):
    # _outer_solve_V must report convergence of the SELECTED (lowest-objective)
    # incumbent, NOT "any start converged". Here the first multistart succeeds with a
    # HIGH objective while the winning (lowest-objective) start reports success=False;
    # the fit must be flagged non-converged so in_space_placebo() fails closed.
    import importlib

    from scipy.optimize import OptimizeResult

    sc = importlib.import_module("diff_diff.synthetic_control")
    calls = {"n": 0}

    def fake_minimize(fun, x0, **kwargs):
        calls["n"] += 1
        x0 = np.asarray(x0, dtype=float)
        if kwargs.get("method") == "Nelder-Mead":
            # 1st start: high objective but converged; later: low objective (wins) but NOT.
            if calls["n"] == 1:
                return OptimizeResult(x=x0, fun=10.0, success=True)
            return OptimizeResult(x=x0, fun=1.0, success=False)
        # Powell polish: neither improves on nor converges at the incumbent.
        return OptimizeResult(x=x0, fun=5.0, success=False)

    monkeypatch.setattr(sc, "minimize", fake_minimize)
    df, _, _ = _make_panel(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df, "y", "treated", "unit", "year", seed=0, n_starts=2, inner_min_decrease=1e-3
        )
    # The winning incumbent came from a success=False run -> selected V is not a
    # validated optimum, so the fit must not be marked converged.
    assert res._fit_converged is False
    with pytest.warns(UserWarning, match="did not converge at fit time"):
        res.in_space_placebo()
    assert np.isnan(res.placebo_p_value)


def test_outer_v_powell_success_at_worse_point_does_not_validate(monkeypatch):
    # The Powell polish must validate the SELECTED incumbent only when it converges
    # back AT the incumbent's objective level. Here the winning (lowest-objective)
    # Nelder-Mead start reports success=False, and Powell "succeeds" but at a STRICTLY
    # WORSE objective (it ended elsewhere). Powell's success says nothing about the
    # selected incumbent, so the fit must stay non-converged and fail closed -- a flag
    # of "converged" here would silently admit an under-optimized V into the placebo
    # ranking and produce wrong permutation inference.
    import importlib

    from scipy.optimize import OptimizeResult

    sc = importlib.import_module("diff_diff.synthetic_control")
    calls = {"n": 0}

    def fake_minimize(fun, x0, **kwargs):
        calls["n"] += 1
        x0 = np.asarray(x0, dtype=float)
        if kwargs.get("method") == "Nelder-Mead":
            # Single start: lowest objective wins but reports success=False.
            return OptimizeResult(x=x0, fun=1.0, success=False)
        # Powell polish: SUCCEEDS, but at a strictly worse objective than the incumbent.
        return OptimizeResult(x=x0, fun=5.0, success=True)

    monkeypatch.setattr(sc, "minimize", fake_minimize)
    df, _, _ = _make_panel(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df, "y", "treated", "unit", "year", seed=0, n_starts=1, inner_min_decrease=1e-3
        )
    # Powell's success at a worse point must NOT flip the selected incumbent to converged.
    assert res._fit_converged is False
    with pytest.warns(UserWarning, match="did not converge at fit time"):
        res.in_space_placebo()
    assert np.isnan(res.placebo_p_value)


def test_to_dict_includes_placebo_scalars():
    res = _fit_for_placebo(n_donors=4)
    d = res.to_dict()
    for key in ("placebo_p_value", "rmspe_ratio", "n_placebos", "n_failed"):
        assert key in d
    # Before the placebo run: rmspe_ratio is finite (fit-time), placebo_p_value NaN.
    assert np.isfinite(d["rmspe_ratio"]) and np.isnan(d["placebo_p_value"])
    assert d["n_placebos"] == 0 and d["n_failed"] == 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res.in_space_placebo()
    d2 = res.to_dict()
    assert np.isfinite(d2["placebo_p_value"]) and d2["n_placebos"] == 4


def test_summary_distinguishes_infeasible_placebo_from_not_run():
    # summary() must tell "placebo never run" apart from "placebo run but produced
    # no valid reference set" (J<2 here -> placebo_p_value NaN but it WAS attempted),
    # and name the SPECIFIC infeasibility reason (too few donors).
    df, _, _ = _make_panel(n_donors=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(df, "y", "treated", "unit", "year", seed=0, **_FAST)
        before = res.summary()
        res.in_space_placebo()  # infeasible: single donor -> no placebo distribution
        after = res.summary()
    assert "Run in_space_placebo()" in before  # never run
    assert np.isnan(res.placebo_p_value) and res._placebo_df is not None  # attempted
    assert res._placebo_status == "too_few_donors"
    assert "requires at least 2 donors" in after  # specific reason, not "not run"
    assert "Run in_space_placebo()" not in after  # not mislabeled as "not run"


def test_summary_treated_fit_failure_names_specific_reason():
    # When the treated unit's OWN fit fails to converge, in_space_placebo() fails
    # closed with n_placebos=0, n_failed=0 -- the SAME counts as the J<2 case. The
    # CI codex P2: summary() must not reconstruct the reason from those counts and
    # narrate "too few donors or all donor refits failed" (false here); it must name
    # the treated-fit non-convergence recorded in _placebo_status.
    df, _, _ = _make_panel(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            seed=0,
            n_starts=1,
            optimizer_options={"maxiter": 1},  # outer V cannot converge -> fail closed
            inner_min_decrease=1e-3,
        )
        assert res._fit_converged is False
        with pytest.warns(UserWarning, match="did not converge at fit time"):
            res.in_space_placebo()
        after = res.summary()
    assert res._placebo_status == "treated_fit_nonconverged"
    assert res.n_placebos == 0 and res.n_failed == 0  # same counts as J<2
    assert "treated unit's own SCM fit" in after and "did not converge" in after
    # Must NOT misdiagnose as the donor-side reason.
    assert "too few" not in after.lower()
    assert "all donor refits" not in after.lower()


def test_in_space_placebo_rejects_invalid_n_starts():
    # CI codex P2: the n_starts override must fail fast on non-positive / non-integer
    # values (mirroring the estimator constructor) rather than silently coercing via
    # int(...) into a degenerate one-start (or invalid) permutation procedure.
    res = _fit_for_placebo(n_donors=4)
    for bad in (0, -1, -5):
        with pytest.raises(ValueError, match="n_starts override must be a positive integer"):
            res.in_space_placebo(n_starts=bad)
    for bad in (2.5, "3"):
        with pytest.raises(ValueError, match="n_starts override must be a positive integer"):
            res.in_space_placebo(n_starts=bad)  # type: ignore[arg-type]
    # The placebo state must be untouched by a rejected override.
    assert res._placebo_status is None and res._placebo_df is None


def test_rmspe_ratio_is_root_scale():
    # The reported statistic is the ROOT-scale ratio RMSPE_post/RMSPE_pre =
    # sqrt(MSPE_post/MSPE_pre), NOT the MSPE ratio. Hand-worked: pre-MSPE = 4,
    # post-MSPE = 9 -> sqrt(9/4) = 1.5 (the MSPE ratio would be 9/4 = 2.25).
    from diff_diff.synthetic_control import _rmspe_ratio

    pre = np.array([2.0, 2.0])  # MSPE = 4
    post = np.array([3.0, 3.0])  # MSPE = 9
    assert _rmspe_ratio(pre, post, scale=10.0) == pytest.approx(1.5)
    # Zero post-effect -> ratio 0; perfect pre-fit -> finite (floored), not inf.
    assert _rmspe_ratio(pre, np.zeros(2), scale=10.0) == pytest.approx(0.0)
    # Perfect pre-fit (zero pre-gaps) -> floored denominator -> finite, not inf.
    assert np.isfinite(_rmspe_ratio(np.zeros(2), post, scale=10.0))


# ---------------------------------------------------------------------------
# Leave-one-out donor robustness (ADH 2015 §4)
# ---------------------------------------------------------------------------


def _equal_mix_panel(n_donors=5, T=8, T0=6, effect=3.0, seed=1):
    """Near-identical donors -> equal-ish weights -> dropping any one barely moves
    the synthetic (the LOO-stable regime)."""
    rng = np.random.default_rng(seed)
    years = list(range(2000, 2000 + T))
    base = rng.normal(10, 0.4, n_donors)
    common = np.cumsum(rng.normal(0, 0.2, T))  # shared trend
    donors = {j: base[j] + common + rng.normal(0, 0.08, T) for j in range(n_donors)}
    treated = np.mean([donors[j] for j in range(n_donors)], axis=0) + rng.normal(0, 0.04, T)
    treated = treated.copy()
    treated[T0:] += effect
    rows = []
    for j in range(n_donors):
        for t in range(T):
            rows.append({"unit": f"d{j}", "year": years[t], "y": donors[j][t], "treated": 0})
    for t in range(T):
        rows.append({"unit": "treated", "year": years[t], "y": treated[t], "treated": int(t >= T0)})
    return pd.DataFrame(rows)


def _single_donor_panel(n_donors=4, T=8, T0=6, effect=3.0, seed=2):
    """One donor (d0) tracks the treated unit; the rest are far away -> weight
    concentrates on d0 -> dropping d0 swings the result (the LOO-fragile regime)."""
    rng = np.random.default_rng(seed)
    years = list(range(2000, 2000 + T))
    d0_path = 10 + np.cumsum(rng.normal(0, 0.3, T))
    donors = {0: d0_path + rng.normal(0, 0.03, T)}
    for j in range(1, n_donors):
        donors[j] = (25.0 + 6.0 * j) + np.cumsum(rng.normal(0, 0.3, T))  # far from treated
    treated = d0_path + rng.normal(0, 0.03, T)
    treated = treated.copy()
    treated[T0:] += effect
    rows = []
    for j in range(n_donors):
        for t in range(T):
            rows.append({"unit": f"d{j}", "year": years[t], "y": donors[j][t], "treated": 0})
    for t in range(T):
        rows.append({"unit": "treated", "year": years[t], "y": treated[t], "treated": int(t >= T0)})
    return pd.DataFrame(rows)


def _fit_cheap(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return synthetic_control(df, "y", "treated", "unit", "year", seed=0, **_FAST)


_LOO_COLS = ["dropped_unit", "att", "pre_rmspe", "post_rmspe", "rmspe_ratio", "delta_att", "status"]


def test_leave_one_out_baseline_row_and_structure():
    res = _fit_for_placebo(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loo = res.leave_one_out()
    assert list(loo.columns) == _LOO_COLS
    # Exactly one baseline row, first, reading directly from the full fit.
    base = loo.iloc[0]
    # dropped_unit is "not applicable" for the baseline row (pandas renders the
    # None as NA in the donor-id column).
    assert base["status"] == "baseline" and pd.isna(base["dropped_unit"])
    assert base["att"] == pytest.approx(res.att) and base["delta_att"] == 0.0
    assert base["pre_rmspe"] == pytest.approx(res.pre_rmspe)
    assert base["rmspe_ratio"] == pytest.approx(res.rmspe_ratio)
    # One LOO row per positively-weighted donor (no failures on this clean panel).
    pos = [d for d in res._fit_snapshot.donor_ids if d in res.donor_weights]
    loo_rows = loo[loo["status"] == "loo"]
    assert set(loo_rows["dropped_unit"]) == set(pos)
    assert res._loo_n_failed == 0 and res._loo_status == "ran"
    # delta_att == att - full att, exactly.
    for _, r in loo_rows.iterrows():
        assert r["delta_att"] == pytest.approx(r["att"] - res.att)
    # Sorted by |delta_att| descending.
    deltas = loo_rows["delta_att"].abs().to_numpy()
    assert np.all(np.diff(deltas) <= 1e-12)
    # att_range spans the LOO refits.
    lo, hi = res._loo_att_range
    assert lo <= hi and lo == pytest.approx(loo_rows["att"].min())
    assert hi == pytest.approx(loo_rows["att"].max())


def test_leave_one_out_stable_when_no_donor_dominates():
    res = _fit_cheap(_equal_mix_panel(n_donors=5, effect=3.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loo = res.leave_one_out()
    loo_rows = loo[loo["status"] == "loo"]
    # Near-identical donors -> dropping any one barely moves the ATT (well under the
    # 3.0 effect). att_range is correspondingly tight.
    assert loo_rows["delta_att"].abs().max() < 1.0
    lo, hi = res._loo_att_range
    assert (hi - lo) < 1.0


def test_leave_one_out_swings_when_one_donor_dominates():
    res = _fit_cheap(_single_donor_panel(n_donors=4, effect=3.0))
    # Weight concentrates on d0.
    assert res.donor_weights.get("d0", 0.0) > 0.5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loo = res.leave_one_out()
    loo_rows = loo[loo["status"] == "loo"]
    # Dropping the dominant donor is the most influential drop (top finite row) and
    # moves the ATT by a non-trivial amount.
    top = loo_rows.iloc[0]
    assert top["dropped_unit"] == "d0"
    assert abs(top["delta_att"]) > 0.2


def test_leave_one_out_deterministic():
    res = _fit_for_placebo(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loo1 = res.leave_one_out()
        loo2 = res.leave_one_out()
    pd.testing.assert_frame_equal(loo1, loo2)


def test_leave_one_out_requires_two_donors():
    res = _fit_for_placebo(n_donors=1)
    with pytest.warns(UserWarning, match="at least 2 donors"):
        loo = res.leave_one_out()
    assert len(loo) == 1 and loo.iloc[0]["status"] == "baseline"
    assert res._loo_status == "too_few_donors" and res._loo_att_range is None


def test_leave_one_out_fails_closed_on_nonconverged_treated_fit():
    df, _, _ = _make_panel(n_donors=4, effect=3.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df, "y", "treated", "unit", "year", seed=0, inner_max_iter=1, **_FAST_CHURN
        )
    assert res._fit_converged is False
    with pytest.warns(UserWarning, match="did not converge at fit time"):
        loo = res.leave_one_out()
    assert len(loo) == 1 and loo.iloc[0]["status"] == "baseline"
    assert res._loo_status == "treated_fit_nonconverged"


def test_leave_one_out_refit_failure_tallied(monkeypatch):
    import importlib

    sc = importlib.import_module("diff_diff.synthetic_control")
    res = _fit_for_placebo(n_donors=4)
    real_fit_unit = sc._placebo_fit_unit
    calls = {"n": 0}

    def flaky_fit_unit(snap, unit, donor_pool, n_starts):
        calls["n"] += 1
        if calls["n"] == 1:  # first leave-one-out refit "fails"
            return None
        return real_fit_unit(snap, unit, donor_pool, n_starts)

    monkeypatch.setattr(sc, "_placebo_fit_unit", flaky_fit_unit)
    with pytest.warns(UserWarning, match="failed to converge"):
        loo = res.leave_one_out()
    assert res._loo_n_failed == 1
    failed = loo[loo["status"] == "failed"]
    assert len(failed) == 1
    assert failed[["att", "pre_rmspe", "rmspe_ratio", "delta_att"]].isna().all().all()
    # Failed rows sort last (after the baseline + the converged LOO rows).
    assert loo.iloc[-1]["status"] == "failed"


def test_leave_one_out_pickle_drops_gaps_keeps_table():
    res = _fit_for_placebo(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res.leave_one_out()
    restored = pickle.loads(pickle.dumps(res))
    # The summary table + scalars survive; panel-derived gap paths do not.
    pd.testing.assert_frame_equal(restored.get_leave_one_out_df(), res.get_leave_one_out_df())
    assert restored._loo_gaps is None
    assert restored._loo_att_range == res._loo_att_range
    with pytest.raises(ValueError, match="not retained after pickling"):
        restored.get_leave_one_out_gaps()


def test_leave_one_out_gaps_long_form():
    res = _fit_for_placebo(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res.leave_one_out()
    gaps = res.get_leave_one_out_gaps()
    assert list(gaps.columns) == ["dropped_unit", "period", "gap", "phase"]
    pos = [d for d in res._fit_snapshot.donor_ids if d in res.donor_weights]
    assert set(gaps["dropped_unit"]) == set(pos)
    # Every dropped donor has a full pre+post trajectory.
    n_periods = len(res.pre_periods) + len(res.post_periods)
    assert (gaps.groupby("dropped_unit").size() == n_periods).all()
    assert set(gaps["phase"]) == {"pre", "post"}


def test_leave_one_out_accessor_before_run_raises():
    res = _fit_for_placebo(n_donors=4)
    with pytest.raises(ValueError, match="call leave_one_out"):
        res.get_leave_one_out_df()
    with pytest.raises(ValueError, match="call leave_one_out"):
        res.get_leave_one_out_gaps()


def test_leave_one_out_does_not_touch_analytical_inference():
    res = _fit_for_placebo(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res.leave_one_out()
    assert_nan_inference(
        {"se": res.se, "t_stat": res.t_stat, "p_value": res.p_value, "conf_int": res.conf_int}
    )
    assert res.is_significant is False


def test_leave_one_out_requires_snapshot():
    res = _fit_for_placebo(n_donors=4)
    restored = pickle.loads(pickle.dumps(res))
    with pytest.raises(ValueError, match="requires the fit snapshot"):
        restored.leave_one_out()


# ---------------------------------------------------------------------------
# In-time placebo: snapshot-truncation helper (ADH 2015 §4)
# ---------------------------------------------------------------------------


def _snap_for_in_time(**kw):
    return _fit_for_placebo(n_donors=4, **kw)._fit_snapshot


def test_truncate_snapshot_positional_split():
    from diff_diff.synthetic_control import _truncate_snapshot_in_time

    snap = _snap_for_in_time()
    assert list(snap.pre_periods) == [2000, 2001, 2002, 2003, 2004, 2005]
    mod, _ = _truncate_snapshot_in_time(snap, 2003)
    assert mod is not None
    assert mod.pre_periods == [2000, 2001, 2002]  # pre-fake = strictly before t_f
    assert mod.post_periods == [2003, 2004, 2005]  # post-fake = held-out pre, t_f first
    # all_periods EXCLUDES the true post periods (2006, 2007) -> airtight no-peeking.
    assert mod.all_periods == [2000, 2001, 2002, 2003, 2004, 2005]
    assert 2006 not in mod.all_periods and 2007 not in mod.all_periods


def test_truncate_snapshot_drops_specs_in_held_out_window():
    from diff_diff.synthetic_control import _truncate_snapshot_in_time

    snap = _snap_for_in_time()  # default pre_period_outcomes="all": one lag per pre period
    mod, dropped = _truncate_snapshot_in_time(snap, 2003)
    for spec in mod.specs:  # surviving specs reference only pre-fake periods
        assert all(p < 2003 for p in spec.periods)
    assert len(dropped) == 3  # lags at 2003/2004/2005 dropped
    assert len(mod.specs) == len(snap.specs) - 3


def test_truncate_snapshot_custom_v_lockstep():
    from diff_diff.synthetic_control import _truncate_snapshot_in_time

    df, _, _ = _make_panel(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            v_method="custom",
            custom_v=np.arange(1.0, 7.0),  # distinct entries to verify the subset
            inner_min_decrease=1e-3,
        )
    snap = res._fit_snapshot
    mod, _ = _truncate_snapshot_in_time(snap, 2003)
    # custom_v subset IN LOCKSTEP with the surviving specs (the default lag specs are
    # ordered by ascending pre period, so the first three entries survive).
    assert mod.custom_v is not None and len(mod.custom_v) == len(mod.specs)
    np.testing.assert_array_equal(mod.custom_v, np.array([1.0, 2.0, 3.0]))


def test_truncate_snapshot_straddling_window_partial_keep():
    from diff_diff.synthetic_control import _truncate_snapshot_in_time

    df, _, _ = _make_panel(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            special_predictors=[("y", [2002, 2003, 2004], "mean")],
            pre_period_outcomes=[2000, 2001],
            inner_min_decrease=1e-3,
        )
    snap = res._fit_snapshot
    mod, _ = _truncate_snapshot_in_time(snap, 2003)
    # The special predictor straddles t_f -> truncated to its pre-fake part [2002].
    special = [s for s in mod.specs if s.kind == "special"]
    assert len(special) == 1 and special[0].periods == [2002]


def test_truncate_snapshot_infeasible_too_few_pre_fake():
    from diff_diff.synthetic_control import _truncate_snapshot_in_time

    snap = _snap_for_in_time()
    # Fewer than 2 pre-fake periods -> infeasible (the deliberate >=2 rule; an
    # auto-swept single-pre-fake placebo is a non-credible pre-fit — documented Note).
    assert _truncate_snapshot_in_time(snap, 2000)[0] is None  # 0 pre-fake
    assert _truncate_snapshot_in_time(snap, 2001)[0] is None  # 1 pre-fake


def test_truncate_snapshot_infeasible_all_specs_dropped():
    from diff_diff.synthetic_control import _truncate_snapshot_in_time

    df, _, _ = _make_panel(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            special_predictors=[("y", [2004, 2005], "mean")],
            pre_period_outcomes=[2004, 2005],
            inner_min_decrease=1e-3,
        )
    snap = res._fit_snapshot
    # t_f=2003 leaves >=2 pre-fake periods, but every spec lives in [2004, 2005]
    # -> all dropped -> infeasible (cannot fit with zero predictors).
    mod, dropped = _truncate_snapshot_in_time(snap, 2003)
    assert mod is None and len(dropped) == len(snap.specs)


def test_truncate_snapshot_does_not_mutate_original():
    from diff_diff.synthetic_control import _truncate_snapshot_in_time

    snap = _snap_for_in_time()
    before = [list(s.periods) for s in snap.specs]
    _truncate_snapshot_in_time(snap, 2003)
    after = [list(s.periods) for s in snap.specs]
    assert before == after  # shared spec objects are never mutated in place


# ---------------------------------------------------------------------------
# In-time placebo: end-to-end (ADH 2015 §4)
# ---------------------------------------------------------------------------

_IN_TIME_COLS = [
    "placebo_period",
    "placebo_att",
    "pre_fit_rmspe",
    "rmspe_ratio",
    "n_pre_fake",
    "n_post_fake",
    "n_dropped_specs",
    "status",
]


def test_in_time_placebo_near_zero_when_effect_post_only():
    # The effect is only in the TRUE post window (>=2006); every backdated placebo
    # falls in the clean pre window, so the placebo "effect" should be ~0.
    res = _fit_for_placebo(n_donors=4, effect=3.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        itp = res.in_time_placebo()
    assert list(itp.columns) == _IN_TIME_COLS
    ran = itp[itp["status"] == "ran"]
    assert len(ran) > 0
    assert ran["placebo_att"].abs().max() < 1.0  # well below the 3.0 true effect


def test_in_time_placebo_sweep_feasibility():
    res = _fit_for_placebo(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        itp = res.in_time_placebo()
    # pre = [2000..2005] -> feasible dates = pre[2:] = [2002, 2003, 2004, 2005]
    # (>=2 pre-fake periods — the deliberate Note-documented restriction).
    assert list(itp["placebo_period"]) == [2002, 2003, 2004, 2005]
    assert (itp["status"] == "ran").all()
    # n_pre_fake + n_post_fake == n_pre for every row, with >=2 pre-fake + >=1 post-fake.
    assert ((itp["n_pre_fake"] + itp["n_post_fake"]) == len(res.pre_periods)).all()
    assert (itp["n_pre_fake"] >= 2).all() and (itp["n_post_fake"] >= 1).all()


def test_in_time_placebo_explicit_post_date_raises():
    res = _fit_for_placebo(n_donors=4)
    with pytest.raises(ValueError, match="true post-treatment period"):
        res.in_time_placebo([2006])


def test_in_time_placebo_date_not_in_pre_raises():
    res = _fit_for_placebo(n_donors=4)
    with pytest.raises(ValueError, match="not a pre-treatment period"):
        res.in_time_placebo([1999])


def test_in_time_placebo_empty_explicit_input_raises():
    # An explicit but EMPTY container is malformed (NOT "every date infeasible") -> raise
    # (codex R6 P1). None still means "sweep all feasible dates".
    res = _fit_for_placebo(n_donors=4)
    for empty in ([], (), pd.Index([]), np.array([])):
        with pytest.raises(ValueError, match="placebo_periods is empty"):
            res.in_time_placebo(empty)
    # The malformed call must not leave any in-time state behind.
    assert res._in_time_df is None and res._in_time_status is None


def test_in_time_placebo_dedups_and_canonicalizes_explicit_dates():
    # Duplicate / unordered explicit dates -> de-duplicated + pre-period-ordered, so no
    # duplicate refits and n_dates is not inflated (codex R7 P3).
    res = _fit_for_placebo(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        itp = res.in_time_placebo([2004, 2002, 2004])  # duplicate 2004, unordered
    assert list(itp["placebo_period"]) == [2002, 2004]  # unique, canonical pre-period order


def test_leave_one_out_immune_to_donor_weights_mutation():
    # Codex R8 P1: the LOO drop-set is FROZEN at fit time (snap.weighted_donor_ids =
    # the >1e-6 reportable support), NOT read from the mutable presentation-level
    # donor_weights dict. So mutating donor_weights after the fit must NOT change which
    # donors are dropped — the robustness result depends only on the fit.
    res = _fit_for_placebo(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        before = set(res.leave_one_out()[lambda d: d["status"] != "baseline"]["dropped_unit"])
    assert before == set(res._fit_snapshot.weighted_donor_ids)  # drops the frozen support
    # Mutate the public dict: drop a real donor, inject a bogus one.
    victim = next(iter(res.donor_weights))
    res.donor_weights = {k: v for k, v in res.donor_weights.items() if k != victim}
    res.donor_weights["bogus_donor"] = 0.99
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        after = set(res.leave_one_out()[lambda d: d["status"] != "baseline"]["dropped_unit"])
    assert after == before  # unchanged by the mutation
    assert "bogus_donor" not in after  # a donor not in the fit is never dropped
    assert victim in after  # still dropped despite removal from donor_weights


def test_in_time_placebo_early_date_infeasible_no_raise():
    res = _fit_for_placebo(n_donors=4)
    # A valid pre-date with too few (<2) pre-fake periods -> NaN infeasible row +
    # warning, NOT a raise.
    with pytest.warns(UserWarning, match="infeasible"):
        itp = res.in_time_placebo([2001])  # 1 pre-fake period
    assert len(itp) == 1 and itp.iloc[0]["status"] == "infeasible"
    assert np.isnan(itp.iloc[0]["placebo_att"])


def test_in_time_placebo_custom_v_zero_mass_is_infeasible_not_failed():
    # A custom_v whose mass lies entirely on specs that TRUNCATE drops leaves a
    # zero-mass surviving V -> the date is INFEASIBLE under the supplied custom_v,
    # NOT a convergence failure (codex R2 P1b: v/v.sum() would be 0/0).
    df, _, _ = _make_panel(n_donors=4)  # default: 6 lag specs (2000..2005)
    v = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])  # all mass on the 2003/2004/2005 lags
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            v_method="custom",
            custom_v=v,
            inner_min_decrease=1e-3,
        )
        itp = res.in_time_placebo([2003])  # keeps lags 2000/2001/2002 -> all zero weight
    row = itp[itp["placebo_period"] == 2003]
    assert len(row) == 1 and row.iloc[0]["status"] == "infeasible"  # NOT "failed"
    assert res._in_time_status == "all_dates_infeasible"


def test_leave_one_out_uniform_shift_surfaced_by_delta_not_range(monkeypatch):
    # Codex R3 P1b: when every donor-drop shifts the ATT the SAME way, the raw
    # att_range has ~zero width (looks stable) but the donor dependence is large.
    # The headline metric must be baseline-relative (max |delta_att|), not the range.
    import importlib

    sc = importlib.import_module("diff_diff.synthetic_control")
    res = _fit_for_placebo(n_donors=4)
    baseline = float(res.att)
    snap = res._fit_snapshot
    shift = 5.0  # same large shift for EVERY drop -> uniform

    def uniform_shift(snap_arg, unit, pool, n_starts):
        gp = {p: 0.0 for p in snap.pre_periods}
        gp.update({p: baseline + shift for p in snap.post_periods})
        return gp, 1.0

    monkeypatch.setattr(sc, "_placebo_fit_unit", uniform_shift)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res.leave_one_out()
    lo, hi = res._loo_att_range
    assert (hi - lo) == pytest.approx(0.0, abs=1e-9)  # raw range would hide the shift
    assert res._loo_max_abs_delta_att == pytest.approx(shift, abs=1e-9)  # delta reveals it
    native = DiagnosticReport(res).to_dict()["estimator_native_diagnostics"]
    assert native["leave_one_out"]["max_abs_delta_att"] == pytest.approx(shift, abs=1e-9)


def test_in_time_placebo_windowed_covariate_dropped_and_warns():
    # A special predictor measured over [2004, 2005] falls entirely in the held-out
    # window for t_f=2003 -> dropped (TRUNCATE) + warning + n_dropped_specs reflects it.
    df, _, _ = _make_panel(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            special_predictors=[("y", [2004, 2005], "mean")],
            pre_period_outcomes=[2000, 2001, 2002, 2003],
            inner_min_decrease=1e-3,
        )
    with pytest.warns(UserWarning, match="dropped"):
        itp = res.in_time_placebo([2003])
    row = itp.iloc[0]
    # The special predictor (and the lag at 2003) lie in [2003, 2005] -> dropped.
    assert row["n_dropped_specs"] >= 1 and row["status"] == "ran"


def test_in_time_placebo_all_specs_dropped_infeasible():
    df, _, _ = _make_panel(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            special_predictors=[("y", [2004, 2005], "mean")],
            pre_period_outcomes=[2004, 2005],
            inner_min_decrease=1e-3,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        itp = res.in_time_placebo([2003])  # every predictor is at 2004/2005
    assert itp.iloc[0]["status"] == "infeasible"


def test_in_time_placebo_custom_v_runs_without_shape_error():
    # End-to-end guard for the custom_v lockstep subset: without it the custom path
    # would raise a shape mismatch once specs are dropped.
    df, _, _ = _make_panel(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            v_method="custom",
            custom_v=np.ones(6),
            inner_min_decrease=1e-3,
        )
        itp = res.in_time_placebo()
    assert (itp["status"] == "ran").any()


def test_in_time_placebo_accepts_2d_custom_v():
    # fit() accepts an array-like custom_v (e.g. a (1, k) row vector, raveled during
    # validation); the in-time TRUNCATE subset must ravel before indexing or a 2D
    # custom_v raises IndexError (codex R5 P1). Must match the 1D result exactly.
    df, _, _ = _make_panel(n_donors=4)
    v1d = np.arange(1.0, 7.0)
    v2d = v1d.reshape(1, 6)  # row-vector form accepted at fit time
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res1 = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            v_method="custom",
            custom_v=v1d,
            inner_min_decrease=1e-3,
        )
        res2 = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            v_method="custom",
            custom_v=v2d,
            inner_min_decrease=1e-3,
        )
        itp1 = res1.in_time_placebo([2003])
        itp2 = res2.in_time_placebo([2003])  # would IndexError before the ravel fix
    pd.testing.assert_frame_equal(itp1, itp2)


def test_in_time_placebo_deterministic():
    res = _fit_for_placebo(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        itp1 = res.in_time_placebo()
        itp2 = res.in_time_placebo()
    pd.testing.assert_frame_equal(itp1, itp2)


def test_in_time_placebo_fails_closed_on_nonconverged_treated_fit():
    df, _, _ = _make_panel(n_donors=4, effect=3.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df, "y", "treated", "unit", "year", seed=0, inner_max_iter=1, **_FAST_CHURN
        )
    assert res._fit_converged is False
    with pytest.warns(UserWarning, match="did not converge"):
        itp = res.in_time_placebo()
    assert len(itp) == 0 and res._in_time_status == "treated_fit_nonconverged"


def test_in_time_placebo_pickle_drops_gaps_keeps_table():
    res = _fit_for_placebo(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res.in_time_placebo()
    restored = pickle.loads(pickle.dumps(res))
    pd.testing.assert_frame_equal(restored.get_in_time_placebo_df(), res.get_in_time_placebo_df())
    assert restored._in_time_gaps is None
    with pytest.raises(ValueError, match="not retained after pickling"):
        restored.get_in_time_placebo_gaps()
    with pytest.raises(ValueError, match="requires the fit snapshot"):
        restored.in_time_placebo()


def test_in_time_placebo_gaps_long_form():
    res = _fit_for_placebo(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res.in_time_placebo([2003])
    gaps = res.get_in_time_placebo_gaps()
    assert list(gaps.columns) == ["placebo_period", "period", "gap", "phase"]
    assert set(gaps["phase"]) == {"pre_fake", "post_fake"}
    # Periods before t_f=2003 are pre_fake; 2003+ are post_fake.
    assert set(gaps.loc[gaps["phase"] == "pre_fake", "period"]) == {2000, 2001, 2002}
    assert set(gaps.loc[gaps["phase"] == "post_fake", "period"]) == {2003, 2004, 2005}


def test_in_time_placebo_accessor_before_run_raises():
    res = _fit_for_placebo(n_donors=4)
    with pytest.raises(ValueError, match="call in_time_placebo"):
        res.get_in_time_placebo_df()
    with pytest.raises(ValueError, match="call in_time_placebo"):
        res.get_in_time_placebo_gaps()


def test_in_time_placebo_does_not_touch_analytical_inference():
    res = _fit_for_placebo(n_donors=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res.in_time_placebo()
    assert_nan_inference(
        {"se": res.se, "t_stat": res.t_stat, "p_value": res.p_value, "conf_int": res.conf_int}
    )
    assert res.is_significant is False


# ---------------------------------------------------------------------------
# Self-consistency parity: the ADH-2015 diagnostics are EXACT re-runs of the
# validated solver on the equivalent sub-problem.
#
# R `Synth` has NO in-time-placebo or leave-one-out function (verified against its
# full CRAN function index), so there is no canonical R *output* to match for these
# diagnostics specifically. Instead we prove (deterministically, via a fixed custom
# V) that leave_one_out() equals a from-scratch fit on the reduced donor pool, and
# in_time_placebo() equals a from-scratch fit on the backdated/truncated panel.
# Because the custom-V solver is itself R-anchored on Basque
# (test_basque_tier1_custom_v_parity), this transitively anchors the diagnostics to
# R while directly validating that the re-run mechanism is exact (not approximate).
# ---------------------------------------------------------------------------


def test_leave_one_out_matches_fresh_reduced_pool_fit():
    df, _, _ = _make_panel(n_donors=4)
    v = np.arange(1.0, 7.0)  # k = 6 default lag predictors; fixed V -> deterministic
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            v_method="custom",
            custom_v=v,
            inner_min_decrease=1e-3,
        )
        loo = res.leave_one_out()
    donor_ids = list(res._fit_snapshot.donor_ids)
    d = [x for x in donor_ids if x in res.donor_weights][0]  # a positively-weighted donor
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fresh = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            v_method="custom",
            custom_v=v,
            inner_min_decrease=1e-3,
            donor_pool=[x for x in donor_ids if x != d],
        )
    loo_att = loo.loc[loo["dropped_unit"] == d, "att"].iloc[0]
    assert loo_att == pytest.approx(fresh.att, abs=1e-7)


def test_in_time_placebo_matches_fresh_backdated_fit():
    df, _, _ = _make_panel(n_donors=4)  # years 2000-2007, T0=6 -> pre = 2000..2005
    v = np.arange(1.0, 7.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = synthetic_control(
            df,
            "y",
            "treated",
            "unit",
            "year",
            v_method="custom",
            custom_v=v,
            inner_min_decrease=1e-3,
        )
        itp = res.in_time_placebo([2003])
    placebo_att = itp.loc[itp["placebo_period"] == 2003, "placebo_att"].iloc[0]
    # Fresh backdated fit: drop the true post periods, treat 2003 as the intervention,
    # feed the pre-fake-subset V (lags at 2000/2001/2002 -> v[:3]).
    back = df[df["year"] <= 2005].copy()
    back["treated"] = ((back["unit"] == "treated") & (back["year"] >= 2003)).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fresh = synthetic_control(
            back,
            "y",
            "treated",
            "unit",
            "year",
            v_method="custom",
            custom_v=v[:3],
            inner_min_decrease=1e-3,
        )
    assert placebo_att == pytest.approx(fresh.att, abs=1e-7)


# ---------------------------------------------------------------------------
# All-refits-failed branches (codex R1 P1): when EVERY refit fails to converge,
# the status must NOT be reported as "ran" / mislabeled as dimensional infeasibility.
# ---------------------------------------------------------------------------


def test_leave_one_out_all_refits_failed_status(monkeypatch):
    import importlib

    sc = importlib.import_module("diff_diff.synthetic_control")
    res = _fit_for_placebo(n_donors=4)
    monkeypatch.setattr(sc, "_placebo_fit_unit", lambda *a, **k: None)  # every drop fails
    with pytest.warns(UserWarning, match="failed to converge"):
        loo = res.leave_one_out()
    # Distinct status (NOT "ran"); att_range is None; baseline + only failed rows.
    assert res._loo_status == "all_refits_failed"
    assert res._loo_att_range is None
    assert (loo["status"] != "loo").all()  # no successful drop
    assert (loo.iloc[1:]["status"] == "failed").all()
    # DiagnosticReport must surface it as NOT "ran", with the convergence reason.
    native = DiagnosticReport(res).to_dict()["estimator_native_diagnostics"]
    assert native["leave_one_out"]["status"] != "ran"
    # Machine-readable code distinguishes numerical failure from structural infeasibility.
    assert native["leave_one_out"]["reason_code"] == "all_refits_failed"
    assert "failed to converge" in native["leave_one_out"]["reason"]


def test_in_time_placebo_all_dates_failed_status(monkeypatch):
    import importlib

    sc = importlib.import_module("diff_diff.synthetic_control")
    res = _fit_for_placebo(n_donors=4)
    monkeypatch.setattr(sc, "_placebo_fit_unit", lambda *a, **k: None)  # every refit fails
    with pytest.warns(UserWarning, match="failed to converge"):
        itp = res.in_time_placebo()
    # Convergence failure must NOT be mislabeled as dimensional infeasibility.
    assert res._in_time_status == "all_dates_failed"
    assert (itp["status"] == "failed").all() and len(itp) > 0
    native = DiagnosticReport(res).to_dict()["estimator_native_diagnostics"]
    assert native["in_time_placebo"]["status"] != "ran"
    assert native["in_time_placebo"]["reason_code"] == "all_dates_failed"
    assert "failed to converge" in native["in_time_placebo"]["reason"]


def test_in_time_placebo_mixed_failed_and_infeasible_status(monkeypatch):
    # Codex R8 P2: a no-success run with BOTH a dimensionally-infeasible date AND a
    # convergence-failed date must report the mixed "all_dates_unusable" status with
    # both counts — NOT be mislabeled as exclusively failed (which would falsely claim
    # "none was dimensionally infeasible").
    import importlib

    sc = importlib.import_module("diff_diff.synthetic_control")
    res = _fit_for_placebo(n_donors=4)
    # Feasible dates "fail" to converge; 2001 (1 pre-fake) is dimensionally infeasible.
    monkeypatch.setattr(sc, "_placebo_fit_unit", lambda *a, **k: None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        itp = res.in_time_placebo([2001, 2003])  # 2001 infeasible, 2003 fails
    assert res._in_time_status == "all_dates_unusable"
    assert res._in_time_n_failed == 1 and res._in_time_n_infeasible == 1
    assert set(itp["status"]) == {"infeasible", "failed"}
    block = DiagnosticReport(res).to_dict()["estimator_native_diagnostics"]["in_time_placebo"]
    assert block["reason_code"] == "all_dates_unusable"
    assert block["n_failed"] == 1 and block["n_infeasible"] == 1
