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
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from diff_diff import SyntheticControl, SyntheticControlResults, synthetic_control
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
# changing what these tests assert. The full production defaults are still exercised by the
# @slow Tier-2 Basque test (which runs in the Rust matrix under `-m ''`), and the
# Rust<->numpy Frank-Wolfe kernel equivalence is locked by
# tests/test_rust_backend.py::test_sc_weight_fw_matches_numpy.
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
    # Reserved PR-2/3 attributes present and None.
    assert res._placebo_gaps is None and res._rmspe_ratio is None and res._fit_snapshot is None


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
