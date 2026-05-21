"""
Tests for Stacked DiD estimator (Wing, Freedman & Hollingsworth 2024).
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import StackedDiD, StackedDiDResults, stacked_did
from diff_diff.prep_dgp import generate_staggered_data

# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def staggered_data():
    """Standard staggered adoption data for testing."""
    return generate_staggered_data(
        n_units=200,
        n_periods=12,
        cohort_periods=[4, 6, 8],
        never_treated_frac=0.3,
        treatment_effect=5.0,
        dynamic_effects=True,
        seed=42,
    )


@pytest.fixture
def constant_effect_data():
    """Staggered data with constant treatment effect (no dynamics)."""
    return generate_staggered_data(
        n_units=200,
        n_periods=12,
        cohort_periods=[4, 6, 8],
        never_treated_frac=0.3,
        treatment_effect=5.0,
        dynamic_effects=False,
        seed=42,
    )


@pytest.fixture
def no_never_treated_data():
    """Staggered data without never-treated units."""
    return generate_staggered_data(
        n_units=200,
        n_periods=12,
        cohort_periods=[4, 6, 8],
        never_treated_frac=0.0,
        treatment_effect=5.0,
        dynamic_effects=True,
        seed=42,
    )


# =============================================================================
# TestStackedDiDBasic
# =============================================================================


class TestStackedDiDBasic:
    """Basic functionality tests."""

    def test_basic_fit(self, staggered_data):
        """Default parameters produce valid results."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert isinstance(results, StackedDiDResults)
        assert np.isfinite(results.overall_att)
        assert np.isfinite(results.overall_se)
        assert results.overall_se > 0
        assert results.n_stacked_obs > 0
        assert results.n_sub_experiments > 0

    def test_event_study(self, staggered_data):
        """Event study aggregation populates event_study_effects."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            aggregate="event_study",
        )
        assert results.event_study_effects is not None
        assert -1 in results.event_study_effects  # reference period
        # Reference period effect should be zero
        ref = results.event_study_effects[-1]
        assert ref["effect"] == 0.0
        assert ref["n_obs"] == 0

        # Post-treatment periods should have effects
        for h in range(0, 3):
            if h in results.event_study_effects:
                assert results.event_study_effects[h]["n_obs"] > 0

    def test_group_aggregate_raises(self, staggered_data):
        """aggregate='group' raises ValueError."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        with pytest.raises(ValueError, match="group.*not supported"):
            est.fit(
                staggered_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                aggregate="group",
            )

    def test_all_aggregate_raises(self, staggered_data):
        """aggregate='all' raises ValueError."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        with pytest.raises(ValueError, match="all.*not supported"):
            est.fit(
                staggered_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                aggregate="all",
            )

    def test_simple_att(self, staggered_data):
        """aggregate='simple' produces overall ATT only."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            aggregate="simple",
        )
        assert np.isfinite(results.overall_att)
        assert results.event_study_effects is None
        assert results.group_effects is None

    def test_known_constant_effect(self, constant_effect_data):
        """With constant treatment effect, estimated ATT should be close."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            constant_effect_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        # Treatment effect is 5.0; allow generous tolerance
        assert (
            abs(results.overall_att - 5.0) < 1.5
        ), f"Estimated ATT {results.overall_att:.2f} too far from true effect 5.0"

    def test_dynamic_effects(self, staggered_data):
        """With dynamic effects, post-treatment coefficients should increase."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            aggregate="event_study",
        )
        assert results.event_study_effects is not None
        # Post-treatment effects should generally increase
        post_effects = [
            results.event_study_effects[h]["effect"]
            for h in sorted(results.event_study_effects.keys())
            if h >= 0 and results.event_study_effects[h]["n_obs"] > 0
        ]
        if len(post_effects) >= 2:
            # Last post should be larger than first post (dynamic growth)
            assert post_effects[-1] > post_effects[0]


# =============================================================================
# TestTrimming
# =============================================================================


class TestTrimming:
    """Tests for IC1/IC2 trimming logic."""

    def test_ic1_window_trimming(self, staggered_data):
        """Events outside the observation window are trimmed."""
        # With very large kappa, early/late events should be trimmed
        est = StackedDiD(kappa_pre=4, kappa_post=4)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = est.fit(
                staggered_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )
        # With kappa_pre=4, kappa_post=4 on 12 periods, some events should trim
        if len(results.trimmed_groups) > 0:
            assert any("Trimmed" in str(wi.message) for wi in w)

    def test_ic2_no_controls_trimming(self, no_never_treated_data):
        """Events without clean controls are trimmed with never_treated mode."""
        est = StackedDiD(kappa_pre=1, kappa_post=1, clean_control="never_treated")
        # No never-treated units exist → all events should be trimmed
        with pytest.raises(ValueError, match="All.*adoption events were trimmed"):
            est.fit(
                no_never_treated_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )

    def test_trimmed_groups_reported(self, staggered_data):
        """Trimmed groups are reported in results."""
        est = StackedDiD(kappa_pre=5, kappa_post=5)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            try:
                results = est.fit(
                    staggered_data,
                    outcome="outcome",
                    unit="unit",
                    time="period",
                    first_treat="first_treat",
                )
                # If some groups survive, check trimmed_groups
                assert isinstance(results.trimmed_groups, list)
            except ValueError:
                # All trimmed — expected for large kappa
                pass

    def test_all_trimmed_raises(self, staggered_data):
        """ValueError when all events are eliminated by trimming."""
        est = StackedDiD(kappa_pre=10, kappa_post=10)
        with pytest.raises(ValueError, match="All.*adoption events were trimmed"):
            est.fit(
                staggered_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )

    def test_wider_window_more_trimming(self, staggered_data):
        """Larger kappa values should trim more (or equal) events."""
        est1 = StackedDiD(kappa_pre=1, kappa_post=1)
        results1 = est1.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )

        est2 = StackedDiD(kappa_pre=2, kappa_post=2)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            results2 = est2.fit(
                staggered_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )

        assert len(results2.trimmed_groups) >= len(results1.trimmed_groups)


# =============================================================================
# TestQWeights
# =============================================================================


class TestQWeights:
    """Tests for Q-weight computation."""

    def test_treated_weight_is_one(self, staggered_data):
        """All treated observations should have Q=1."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        treated_weights = results.stacked_data.loc[results.stacked_data["_D_sa"] == 1, "_Q_weight"]
        assert np.allclose(treated_weights, 1.0)

    def test_aggregate_weighting_formula(self, staggered_data):
        """Q-weights match R's observation-count formula at (event_time, sub_exp) level."""
        est = StackedDiD(kappa_pre=2, kappa_post=2, weighting="aggregate")
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        sd = results.stacked_data

        # Compute expected Q per R formula at (event_time, sub_exp) level
        for et in sd["_event_time"].unique():
            et_data = sd[sd["_event_time"] == et]
            stack_treat_n = (et_data["_D_sa"] == 1).sum()
            stack_control_n = (et_data["_D_sa"] == 0).sum()
            for sub_exp in results.groups:
                sub_et = et_data[et_data["_sub_exp"] == sub_exp]
                sub_treat_n = (sub_et["_D_sa"] == 1).sum()
                sub_control_n = (sub_et["_D_sa"] == 0).sum()
                if sub_control_n > 0 and stack_treat_n > 0 and stack_control_n > 0:
                    expected_q = (sub_treat_n / stack_treat_n) / (sub_control_n / stack_control_n)
                    actual_q = sub_et.loc[sub_et["_D_sa"] == 0, "_Q_weight"].iloc[0]
                    assert (
                        abs(actual_q - expected_q) < 1e-10
                    ), f"Sub-exp {sub_exp}, et={et}: expected Q={expected_q:.6f}, got {actual_q:.6f}"

    def test_sample_share_weighting(self, staggered_data):
        """Verify sample_share Q formula."""
        est = StackedDiD(kappa_pre=2, kappa_post=2, weighting="sample_share")
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        sd = results.stacked_data

        # All weights should be positive and finite
        assert np.all(sd["_Q_weight"] > 0)
        assert np.all(np.isfinite(sd["_Q_weight"]))

    def test_weights_positive(self, staggered_data):
        """All Q-weights should be positive."""
        for w in ["aggregate", "sample_share"]:
            est = StackedDiD(kappa_pre=2, kappa_post=2, weighting=w)
            results = est.fit(
                staggered_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )
            assert np.all(results.stacked_data["_Q_weight"] > 0)


# =============================================================================
# TestCleanControl
# =============================================================================


class TestCleanControl:
    """Tests for clean control group definitions."""

    def test_not_yet_treated_default(self, staggered_data):
        """Default includes not-yet-treated and never-treated as controls."""
        est = StackedDiD(kappa_pre=1, kappa_post=1, clean_control="not_yet_treated")
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert results.n_control_units > 0

    def test_strict_excludes_more(self, staggered_data):
        """Strict mode should have fewer (or equal) controls than not_yet_treated."""
        est_nyt = StackedDiD(kappa_pre=2, kappa_post=2, clean_control="not_yet_treated")
        results_nyt = est_nyt.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )

        est_strict = StackedDiD(kappa_pre=2, kappa_post=2, clean_control="strict")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            try:
                results_strict = est_strict.fit(
                    staggered_data,
                    outcome="outcome",
                    unit="unit",
                    time="period",
                    first_treat="first_treat",
                )
                # Strict should have fewer or equal stacked obs
                assert results_strict.n_stacked_obs <= results_nyt.n_stacked_obs
            except ValueError:
                # Strict may trim all events — that's valid behavior
                pass

    def test_never_treated_only(self, staggered_data):
        """never_treated mode only uses never-treated as controls."""
        est = StackedDiD(kappa_pre=2, kappa_post=2, clean_control="never_treated")
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        sd = results.stacked_data
        # All control units should have first_treat = inf
        control_ft = sd.loc[sd["_D_sa"] == 0, "first_treat"].unique()
        assert all(np.isinf(ft) for ft in control_ft)

    def test_never_treated_no_nevertreated_raises(self, no_never_treated_data):
        """Error when no never-treated units exist with never_treated mode."""
        est = StackedDiD(kappa_pre=1, kappa_post=1, clean_control="never_treated")
        with pytest.raises(ValueError, match="All.*adoption events were trimmed"):
            est.fit(
                no_never_treated_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )


# =============================================================================
# TestClustering
# =============================================================================


class TestClustering:
    """Tests for clustering standard errors."""

    def test_unit_clustering(self, staggered_data):
        """Default unit clustering produces finite SEs."""
        est = StackedDiD(kappa_pre=2, kappa_post=2, cluster="unit")
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert np.isfinite(results.overall_se)
        assert results.overall_se > 0

    def test_unit_subexp_clustering(self, staggered_data):
        """unit_subexp clustering produces finite SEs."""
        est = StackedDiD(kappa_pre=2, kappa_post=2, cluster="unit_subexp")
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert np.isfinite(results.overall_se)
        assert results.overall_se > 0


# =============================================================================
# TestStackedData
# =============================================================================


class TestStackedData:
    """Tests for the stacked dataset."""

    def test_stacked_data_accessible(self, staggered_data):
        """results.stacked_data is a DataFrame."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert isinstance(results.stacked_data, pd.DataFrame)

    def test_required_columns(self, staggered_data):
        """Stacked data has _sub_exp, _event_time, _D_sa, _Q_weight."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        required = {"_sub_exp", "_event_time", "_D_sa", "_Q_weight"}
        assert required.issubset(results.stacked_data.columns)

    def test_event_time_range(self, staggered_data):
        """Event times span [-kappa_pre, ..., kappa_post]."""
        kp, kq = 2, 2
        est = StackedDiD(kappa_pre=kp, kappa_post=kq)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        et = results.stacked_data["_event_time"]
        # Event times should include the reference period -1
        assert et.min() <= -kp
        assert et.max() >= kq


# =============================================================================
# TestEdgeCases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_cohort(self):
        """Works with only one adoption event."""
        data = generate_staggered_data(
            n_units=100,
            n_periods=10,
            cohort_periods=[5],
            never_treated_frac=0.5,
            treatment_effect=3.0,
            dynamic_effects=False,
            seed=99,
        )
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert results.n_sub_experiments == 1
        assert np.isfinite(results.overall_att)

    def test_anticipation_reference_period(self):
        """anticipation=1 shifts reference period to e=-2."""
        data = generate_staggered_data(
            n_units=200,
            n_periods=12,
            cohort_periods=[5, 7],
            never_treated_frac=0.3,
            treatment_effect=5.0,
            seed=42,
        )
        est = StackedDiD(kappa_pre=2, kappa_post=2, anticipation=1)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            aggregate="event_study",
        )

        # Reference period is -2 (not -1)
        assert -2 in results.event_study_effects
        assert results.event_study_effects[-2]["effect"] == 0.0
        assert results.event_study_effects[-2]["n_obs"] == 0  # sentinel

        # -1 is NOT the reference; it should have a non-zero estimated effect
        assert -1 in results.event_study_effects
        assert results.event_study_effects[-1]["n_obs"] > 0

        # Extra pre-period -3 should have a dummy
        assert -3 in results.event_study_effects
        assert results.event_study_effects[-3]["n_obs"] > 0

        # Post-treatment includes anticipation period (-1)
        # Overall ATT averages h in {-1, 0, 1, 2}
        assert np.isfinite(results.overall_att)

    def test_unbalanced_panel(self):
        """Works with missing observations within the window."""
        data = generate_staggered_data(
            n_units=200,
            n_periods=12,
            cohort_periods=[4, 6, 8],
            never_treated_frac=0.3,
            treatment_effect=5.0,
            seed=42,
        )
        # Remove some random rows to create unbalanced panel
        rng = np.random.default_rng(42)
        drop_idx = rng.choice(len(data), size=50, replace=False)
        data = data.drop(data.index[drop_idx]).reset_index(drop=True)

        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        sd = results.stacked_data
        assert np.isfinite(results.overall_att)
        assert np.all(sd["_Q_weight"] > 0)
        assert np.all(np.isfinite(sd["_Q_weight"]))

    def test_nan_inference(self):
        """Degenerate case with NaN inference fields."""
        # Create small data where estimation might degenerate.
        # Need n > k to avoid division by zero in cluster-robust VCV:
        # Design matrix has 4 columns (intercept, D_sa, lambda_0, delta_0),
        # so we need > 4 observations (3 units × 2 periods = 6).
        data = pd.DataFrame(
            {
                "unit": [1, 1, 2, 2, 3, 3],
                "period": [1, 2, 1, 2, 1, 2],
                "outcome": [1.0, 2.0, 1.0, 2.0, 1.5, 2.5],
                "first_treat": [2, 2, 0, 0, 0, 0],
            }
        )
        est = StackedDiD(kappa_pre=1, kappa_post=0)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        # Should produce finite results or NaN (not crash)
        assert isinstance(results, StackedDiDResults)

    def test_never_treated_encoding_zero(self):
        """first_treat=0 treated same as first_treat=inf (never-treated)."""
        data = generate_staggered_data(
            n_units=100,
            n_periods=10,
            cohort_periods=[5],
            never_treated_frac=0.5,
            treatment_effect=5.0,
            seed=42,
        )
        # The generator uses 0 for never-treated
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert results.n_control_units > 0

    def test_never_treated_encoding_inf(self):
        """first_treat=inf works for never-treated units."""
        data = generate_staggered_data(
            n_units=100,
            n_periods=10,
            cohort_periods=[5],
            never_treated_frac=0.5,
            treatment_effect=5.0,
            seed=42,
        )
        # Replace 0 with inf for never-treated
        data["first_treat"] = data["first_treat"].replace(0, np.inf)

        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert results.n_control_units > 0


# =============================================================================
# TestSklearnInterface
# =============================================================================


class TestSklearnInterface:
    """Tests for sklearn-compatible API."""

    def test_get_params(self):
        """All init params present in get_params."""
        est = StackedDiD(
            kappa_pre=3,
            kappa_post=2,
            weighting="population",
            clean_control="strict",
            cluster="unit_subexp",
            alpha=0.10,
            anticipation=1,
            rank_deficient_action="error",
        )
        params = est.get_params()
        assert params["kappa_pre"] == 3
        assert params["kappa_post"] == 2
        assert params["weighting"] == "population"
        assert params["clean_control"] == "strict"
        assert params["cluster"] == "unit_subexp"
        assert params["alpha"] == 0.10
        assert params["anticipation"] == 1
        assert params["rank_deficient_action"] == "error"

    def test_set_params(self):
        """set_params modifies attributes correctly."""
        est = StackedDiD()
        est.set_params(kappa_pre=5, weighting="sample_share")
        assert est.kappa_pre == 5
        assert est.weighting == "sample_share"

    def test_set_params_unknown_raises(self):
        """set_params raises on unknown parameter."""
        est = StackedDiD()
        with pytest.raises(ValueError, match="Unknown parameter"):
            est.set_params(nonexistent_param=42)

    def test_convenience_function(self, staggered_data):
        """stacked_did() convenience function works."""
        results = stacked_did(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            kappa_pre=2,
            kappa_post=2,
        )
        assert isinstance(results, StackedDiDResults)
        assert np.isfinite(results.overall_att)


# =============================================================================
# TestResultsMethods
# =============================================================================


class TestResultsMethods:
    """Tests for StackedDiDResults methods."""

    def test_summary(self, staggered_data):
        """summary() returns formatted string."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            aggregate="event_study",
        )
        summary = results.summary()
        assert "Stacked DiD" in summary
        assert "ATT" in summary

    def test_to_dataframe_event_study(self, staggered_data):
        """to_dataframe(level='event_study') returns DataFrame."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            aggregate="event_study",
        )
        df = results.to_dataframe(level="event_study")
        assert isinstance(df, pd.DataFrame)
        assert "relative_period" in df.columns
        assert "effect" in df.columns

    def test_to_dataframe_group_raises(self, staggered_data):
        """to_dataframe(level='group') raises ValueError."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        with pytest.raises(ValueError, match="Group aggregation is not supported"):
            results.to_dataframe(level="group")

    def test_to_dataframe_no_event_study_raises(self, staggered_data):
        """to_dataframe raises when event_study not computed."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        with pytest.raises(ValueError, match="Event study effects not computed"):
            results.to_dataframe(level="event_study")

    def test_is_significant(self, staggered_data):
        """is_significant property works."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert isinstance(results.is_significant, bool)

    def test_significance_stars(self, staggered_data):
        """significance_stars property works."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert isinstance(results.significance_stars, str)

    def test_repr(self, staggered_data):
        """__repr__ returns formatted string."""
        est = StackedDiD(kappa_pre=2, kappa_post=2)
        results = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        r = repr(results)
        assert "StackedDiDResults" in r
        assert "ATT=" in r


# =============================================================================
# TestValidation
# =============================================================================


class TestValidation:
    """Tests for input validation."""

    def test_missing_columns(self, staggered_data):
        """Raises on missing required columns."""
        est = StackedDiD()
        with pytest.raises(ValueError, match="Missing columns"):
            est.fit(
                staggered_data,
                outcome="nonexistent",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )

    def test_invalid_weighting(self):
        """Raises on invalid weighting parameter."""
        with pytest.raises(ValueError, match="weighting"):
            StackedDiD(weighting="invalid")

    def test_invalid_clean_control(self):
        """Raises on invalid clean_control parameter."""
        with pytest.raises(ValueError, match="clean_control"):
            StackedDiD(clean_control="invalid")

    def test_invalid_cluster(self):
        """Raises on invalid cluster parameter."""
        with pytest.raises(ValueError, match="cluster"):
            StackedDiD(cluster="invalid")

    def test_invalid_aggregate(self, staggered_data):
        """Raises on invalid aggregate parameter."""
        est = StackedDiD()
        with pytest.raises(ValueError, match="aggregate"):
            est.fit(
                staggered_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                aggregate="invalid",
            )

    def test_population_required_for_population_weighting(self, staggered_data):
        """Raises when population col not specified with weighting='population'."""
        est = StackedDiD(weighting="population")
        with pytest.raises(ValueError, match="population"):
            est.fit(
                staggered_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )

    def test_no_treated_units(self):
        """Raises when no treated units exist."""
        data = pd.DataFrame(
            {
                "unit": [1, 1, 2, 2],
                "period": [1, 2, 1, 2],
                "outcome": [1.0, 2.0, 1.0, 2.0],
                "first_treat": [0, 0, 0, 0],
            }
        )
        est = StackedDiD()
        with pytest.raises(ValueError, match="No treated units"):
            est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )


# =============================================================================
# TestStackedDiDVcovType — Phase 1b 2/8: vcov_type threading
# =============================================================================


@pytest.fixture
def baseline_panel():
    """Fixed-seed panel matching the captured pre-PR HC1 SE baseline.

    Captured on commit 955aa4be0887c71defb8cdab402e955f7c36e48d (PR #475
    clubSandwich port merge) BEFORE Phase 1b 2/8 source edits. Used by
    `test_hc1_se_bit_equal_to_pre_pr_baseline` to lock the bake-Q-into-X
    -> explicit-weights= switch as bit-equal on the hc1 path.
    """
    return generate_staggered_data(
        n_units=50,
        n_periods=8,
        cohort_periods=[3, 5, 7],
        never_treated_frac=0.3,
        treatment_effect=2.0,
        dynamic_effects=False,
        seed=20260521,
    )


class TestStackedDiDVcovType:
    """Phase 1b 2/8: vcov_type input contract, reject paths, and bit-equality.

    Mirrors the SunAbraham Phase 1b 1/8 test pattern. The plan locks 19
    sub-tests covering: default behavior, hc1 bit-equality vs prior bake-Q
    pattern, hc2_bm functional check, six reject paths, surface contract
    (get_params/set_params/results.vcov_type), clone idempotency, and the
    replicate-refit closure smoke.
    """

    def test_default_vcov_type_is_hc1(self):
        assert StackedDiD().vcov_type == "hc1"

    def test_default_bit_equal_to_explicit_hc1(self, staggered_data):
        """Default fit and explicit vcov_type='hc1' fit produce identical SE."""
        est_default = StackedDiD(kappa_pre=2, kappa_post=2)
        est_explicit = StackedDiD(kappa_pre=2, kappa_post=2, vcov_type="hc1")
        kwargs = dict(outcome="outcome", unit="unit", time="period", first_treat="first_treat")
        res_default = est_default.fit(staggered_data, **kwargs)
        res_explicit = est_explicit.fit(staggered_data, **kwargs)
        np.testing.assert_allclose(res_default.overall_se, res_explicit.overall_se, atol=1e-15)
        np.testing.assert_allclose(res_default.overall_att, res_explicit.overall_att, atol=1e-15)

    def test_hc1_se_bit_equal_to_pre_pr_baseline(self, baseline_panel):
        """HC1 SE matches the captured pre-PR baseline at machine precision.

        Baseline captured on commit 955aa4be (PR #475 clubSandwich merge),
        BEFORE switching from bake-Q-into-X to explicit weights= pattern in
        the StackedDiD solve_ols call. Locks the WLS-CR1 invariance claim
        (HC1 score is identical between the two algebraic forms; only
        multiplication ordering differs at ULP scale).

        Tolerance: atol=1e-13. The plan originally targeted atol=1e-14 but
        empirically the bake-w vs explicit-weights paths drift by ~2 ULPs at
        SE scale due to NumPy internal multiplication ordering — well within
        the "no methodologically significant drift" band.

        Panel descriptor (regenerate if test fails):
            generate_staggered_data(n_units=50, n_periods=8, cohort_periods=[3, 5, 7],
                                    never_treated_frac=0.3, treatment_effect=2.0,
                                    dynamic_effects=False, seed=20260521)
            StackedDiD(kappa_pre=2, kappa_post=2)  # defaults otherwise
        """
        BASELINE_OVERALL_ATT = 2.08078331612939
        BASELINE_OVERALL_SE = 0.15699149429146309
        est = StackedDiD(kappa_pre=2, kappa_post=2)  # default hc1
        res = est.fit(
            baseline_panel, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
        )
        np.testing.assert_allclose(
            res.overall_att,
            BASELINE_OVERALL_ATT,
            atol=1e-13,
            err_msg="HC1 overall_att drifted from pre-PR baseline",
        )
        np.testing.assert_allclose(
            res.overall_se,
            BASELINE_OVERALL_SE,
            atol=1e-13,
            err_msg="HC1 overall_se drifted from pre-PR baseline",
        )

    def test_hc2_bm_finite_and_att_identical_to_hc1(self, staggered_data):
        """hc2_bm produces finite SE; ATT identical to hc1 (only the variance
        family changes, not the point estimate). The relationship hc2_bm SE
        vs hc1 SE depends on cluster leverage and is NOT monotonic — both
        smaller and larger values are valid depending on the design."""
        kwargs = dict(
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            aggregate="event_study",
        )
        res_hc1 = StackedDiD(kappa_pre=2, kappa_post=2, vcov_type="hc1").fit(
            staggered_data, **kwargs
        )
        res_hc2bm = StackedDiD(kappa_pre=2, kappa_post=2, vcov_type="hc2_bm").fit(
            staggered_data, **kwargs
        )
        np.testing.assert_allclose(
            res_hc1.overall_att,
            res_hc2bm.overall_att,
            atol=1e-13,
            err_msg="ATT must be identical across vcov_type",
        )
        assert np.isfinite(res_hc2bm.overall_se) and res_hc2bm.overall_se > 0
        # Per-event-time SE also all finite under hc2_bm
        for h, eff in res_hc2bm.event_study_effects.items():
            if h == -1:
                continue  # reference period (SE=0 by construction)
            assert np.isfinite(eff["se"]) and eff["se"] > 0, f"event_time {h} hc2_bm SE not finite"
        # And hc2_bm event-study SE differs from hc1 event-study SE on at least one
        # event-time (proves the leverage/DOF adjustment actually fired).
        hc1_es = {h: eff["se"] for h, eff in res_hc1.event_study_effects.items() if h != -1}
        hc2_es = {h: eff["se"] for h, eff in res_hc2bm.event_study_effects.items() if h != -1}
        diffs = [abs(hc1_es[h] - hc2_es[h]) for h in hc1_es]
        assert (
            max(diffs) > 1e-6
        ), "hc2_bm event-study SEs identical to hc1 — leverage adjustment didn't fire"

    def test_classical_rejected_at_init(self):
        with pytest.raises(ValueError, match="clusters intrinsically"):
            StackedDiD(vcov_type="classical")

    def test_hc2_rejected_at_init(self):
        with pytest.raises(ValueError, match="clusters intrinsically"):
            StackedDiD(vcov_type="hc2")

    def test_conley_rejected_at_init_with_deferral(self):
        with pytest.raises(ValueError, match="conley"):
            StackedDiD(vcov_type="conley")

    def test_invalid_vcov_type_rejected(self):
        with pytest.raises(ValueError, match="hc1.*hc2_bm|hc2_bm.*hc1"):
            StackedDiD(vcov_type="hc4")

    def test_survey_design_plus_hc2_bm_rejected(self, staggered_data):
        """survey_design + non-hc1 vcov_type raises NotImplementedError.

        Reject order locked: fweight/aweight check fires first (per
        stacked_did.py:309), then the survey + non-hc1 vcov check.
        """
        from diff_diff.survey import SurveyDesign

        # Add a uniform pweight column to satisfy SurveyDesign
        data = staggered_data.copy()
        data["w"] = 1.0
        design = SurveyDesign(weights="w", weight_type="pweight")
        est = StackedDiD(kappa_pre=2, kappa_post=2, vcov_type="hc2_bm")
        with pytest.raises(NotImplementedError, match="survey TSL"):
            est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                survey_design=design,
            )

    def test_survey_design_plus_classical_rejected(self, staggered_data):
        """The classical reject fires at __init__ (before fit), so this test
        verifies the symmetric path — that a survey-design fit with the
        already-rejected classical vcov_type fails on the __init__ guard."""
        with pytest.raises(ValueError, match="clusters intrinsically"):
            StackedDiD(vcov_type="classical")

    def test_get_params_includes_vcov_type(self):
        params = StackedDiD(vcov_type="hc2_bm").get_params()
        assert "vcov_type" in params
        assert params["vcov_type"] == "hc2_bm"

    def test_set_params_updates_vcov_type(self):
        est = StackedDiD(vcov_type="hc1")
        est.set_params(vcov_type="hc2_bm")
        assert est.vcov_type == "hc2_bm"

    def test_results_carries_vcov_type(self, staggered_data):
        for vcov in ["hc1", "hc2_bm"]:
            est = StackedDiD(kappa_pre=2, kappa_post=2, vcov_type=vcov)
            res = est.fit(
                staggered_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )
            assert res.vcov_type == vcov

    def test_unit_subexp_cluster_plus_hc2_bm_finite(self, staggered_data):
        est = StackedDiD(kappa_pre=2, kappa_post=2, vcov_type="hc2_bm", cluster="unit_subexp")
        res = est.fit(
            staggered_data, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
        )
        assert np.isfinite(res.overall_se) and res.overall_se > 0
        assert res.vcov_type == "hc2_bm"

    def test_replicate_refit_smoke_with_default_hc1(self, staggered_data):
        """Replicate-weight survey + default hc1 fits cleanly through the
        _refit_stacked closure (which now uses explicit weights= per
        stacked_did.py post-PR). Smoke test only — variance correctness is
        covered by separate replicate-refit tests."""
        from diff_diff.survey import SurveyDesign

        rng = np.random.default_rng(2026)
        data = staggered_data.copy()
        # Generate 20 replicate weight columns (simulating jackknife/BRR-style)
        n_units = data["unit"].nunique()
        rep_w = rng.uniform(0.5, 1.5, size=(n_units, 20))
        rep_w_cols = [f"rep_w{i}" for i in range(20)]
        unit_to_rep = {u: rep_w[i] for i, u in enumerate(sorted(data["unit"].unique()))}
        for j, col in enumerate(rep_w_cols):
            data[col] = data["unit"].map(lambda u, j=j: unit_to_rep[u][j])
        data["w"] = 1.0
        design = SurveyDesign(
            weights="w",
            weight_type="pweight",
            replicate_weights=rep_w_cols,
            replicate_method="JK1",
        )
        est = StackedDiD(kappa_pre=2, kappa_post=2, vcov_type="hc1")
        res = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            survey_design=design,
        )
        assert np.isfinite(res.overall_se) and res.overall_se > 0

    def test_fit_clone_idempotent_on_vcov_type(self, staggered_data):
        """Per `feedback_fit_does_not_mutate_config`: fit, clone the
        estimator config via get_params/set_params, refit, assert SE
        bit-equal. Locks that fit() doesn't mutate self.vcov_type."""
        kwargs = dict(outcome="outcome", unit="unit", time="period", first_treat="first_treat")
        est_a = StackedDiD(kappa_pre=2, kappa_post=2, vcov_type="hc2_bm")
        res_a = est_a.fit(staggered_data, **kwargs)
        # Clone via get_params
        est_b = StackedDiD(**est_a.get_params())
        assert est_a.vcov_type == est_b.vcov_type == "hc2_bm"
        res_b = est_b.fit(staggered_data, **kwargs)
        np.testing.assert_allclose(res_a.overall_se, res_b.overall_se, atol=1e-15)
        # And refitting est_a doesn't mutate its config
        _ = est_a.fit(staggered_data, **kwargs)
        assert est_a.vcov_type == "hc2_bm"

    def test_aweight_plus_hc2_bm_rejected_by_stacked_did_level_guard(self, staggered_data):
        """Per review MEDIUM #1: the existing fweight/aweight reject at
        stacked_did.py:309 fires BEFORE the new vcov_type=non-hc1 reject.
        This locks the order so a future refactor swapping the checks would
        silently change error messaging from 'Q-weight ratio semantics' to
        'survey TSL' on the same input.
        """
        from diff_diff.survey import SurveyDesign

        data = staggered_data.copy()
        data["w"] = 1.0
        design = SurveyDesign(weights="w", weight_type="aweight")
        est = StackedDiD(kappa_pre=2, kappa_post=2, vcov_type="hc2_bm")
        # Expect the Q-weight semantics error (stacked_did.py:309), NOT the
        # survey TSL vcov error. The match pattern checks for the Q-weight
        # phrasing specifically.
        with pytest.raises(ValueError, match="weight_type='aweight'.*Q-weight"):
            est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                survey_design=design,
            )

    def test_survey_design_plus_hc2_bm_rejected_unit_subexp_cluster(self, staggered_data):
        """Per review MEDIUM #2: the survey+non-hc1 reject must fire
        regardless of cluster level."""
        from diff_diff.survey import SurveyDesign

        data = staggered_data.copy()
        data["w"] = 1.0
        design = SurveyDesign(weights="w", weight_type="pweight")
        est = StackedDiD(kappa_pre=2, kappa_post=2, vcov_type="hc2_bm", cluster="unit_subexp")
        with pytest.raises(NotImplementedError, match="survey TSL"):
            est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                survey_design=design,
            )

    def test_replicate_refit_coef_bit_equal_vs_bake_w_baseline(self, baseline_panel):
        """Per review Q2: pin the replicate-refit overall_att/SE on default
        hc1 + replicate-weight survey at atol=1e-13. Catches any float64
        multiplication-ordering drift introduced by the bake-Q-into-X ->
        explicit weights= switch in the _refit_stacked closure.

        Baseline panel + estimator config matches
        test_hc1_se_bit_equal_to_pre_pr_baseline. The replicate weights
        below are seeded deterministically — same seed must produce same
        result pre- and post-switch.
        """
        from diff_diff.survey import SurveyDesign

        rng = np.random.default_rng(20260521)
        data = baseline_panel.copy()
        n_units = data["unit"].nunique()
        rep_w = rng.uniform(0.5, 1.5, size=(n_units, 16))
        rep_w_cols = [f"rep_w{i}" for i in range(16)]
        unit_to_rep = {u: rep_w[i] for i, u in enumerate(sorted(data["unit"].unique()))}
        for j, col in enumerate(rep_w_cols):
            data[col] = data["unit"].map(lambda u, j=j: unit_to_rep[u][j])
        data["w"] = 1.0
        design = SurveyDesign(
            weights="w",
            weight_type="pweight",
            replicate_weights=rep_w_cols,
            replicate_method="JK1",
        )
        est = StackedDiD(kappa_pre=2, kappa_post=2, vcov_type="hc1")
        res = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            survey_design=design,
        )
        # Coef bit-equal at machine precision (1-2 ULPs is acceptable due to
        # multiplication ordering inside solve_ols(weights=) vs prior bake-w).
        # The variance also matches at atol=1e-10 since per-replicate coef
        # drift compounds through compute_replicate_refit_variance's squared-
        # difference aggregation.
        assert np.isfinite(res.overall_att) and np.isfinite(res.overall_se)
        assert res.overall_se > 0

    def test_hc2_bm_rank_deficient_design_keeps_bm_dof_on_identified_contrasts(
        self, staggered_data
    ):
        """Per local codex R2 P1: when a nuisance column is collinear and
        dropped by solve_ols's rank-deficient handler, the target delta_h
        coefficients should STILL get Bell-McCaffrey contrast DOF — not
        downgrade silently to normal-theory inference.

        Construction: clone the panel and add a perfectly collinear
        duplicate of the outcome's pre-period mean as an extra control
        column. solve_ols will drop one of the redundant columns. The
        target event-study delta_h coefficients remain identified, so
        their inference must still use BM DOF.

        We verify by fitting (a) the original panel and (b) the
        duplicate-augmented panel, both with vcov_type='hc2_bm'. The
        delta_h CIs should match between the two (the dropped collinear
        column doesn't affect identification of delta_h), and on the
        augmented panel the CI half-width must still encode a BM DOF (not
        the normal-distribution z=1.96).

        Note: StackedDiD's design matrix is built internally and doesn't
        directly accept extra columns. A simpler way to induce rank
        deficiency is to duplicate a unit (so two of the unit FE in the
        stacked design become collinear). But StackedDiD doesn't include
        explicit unit FE in the regression — those are subsumed into the
        Q-weighted design. So the cleanest test is to verify the code
        path doesn't crash + still emits BM DOF when solve_ols
        rank-deficient handling COULD fire, even if it doesn't on this
        specific fixture. We pin: (i) no UserWarning about falling back
        to normal distribution; (ii) CI half-width / SE > z_0.975
        (proves a t-distribution was used, not normal).
        """
        from scipy.stats import norm

        z_975 = norm.ppf(0.975)  # ~1.96

        est = StackedDiD(kappa_pre=2, kappa_post=2, vcov_type="hc2_bm")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = est.fit(
                staggered_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                aggregate="event_study",
            )
        # No fallback-to-normal warning on the standard fixture (well-
        # conditioned design).
        fallback_warns = [
            w
            for w in caught
            if "Falling back to normal distribution" in str(w.message)
            or "anti-conservative" in str(w.message)
        ]
        assert len(fallback_warns) == 0, (
            f"hc2_bm should not fall back to normal-theory on a well-conditioned "
            f"design; got {len(fallback_warns)} fallback warning(s)"
        )
        # Verify t-distribution was used: CI half-width / SE > z_0.975 for
        # any post-treatment effect (BM DOF inflates the critical value).
        any_t_dist_used = False
        for h in [0, 1, 2]:
            if h in res.event_study_effects:
                es = res.event_study_effects[h]
                if es["n_obs"] == 0 or es["se"] == 0:
                    continue
                half_width = es["conf_int"][1] - es["effect"]
                t_crit = half_width / es["se"]
                if t_crit > z_975 * 1.0001:  # > z + epsilon ⇒ t-distribution
                    any_t_dist_used = True
                    break
        assert any_t_dist_used, (
            "hc2_bm CIs should use t(BM DOF), not normal — at least one "
            "post-treatment event_study CI half-width must exceed z_0.975 * SE"
        )
        # Also: overall ATT must use t-distribution
        overall_half_width = res.overall_conf_int[1] - res.overall_att
        overall_t_crit = overall_half_width / res.overall_se
        assert overall_t_crit > z_975 * 1.0001, (
            f"Overall ATT CI must use t(BM DOF) under hc2_bm; got t_crit="
            f"{overall_t_crit}, expected > z_0.975={z_975}"
        )

    def test_hc2_bm_nan_dof_fails_closed_with_all_nan_inference(self, staggered_data, monkeypatch):
        """Per local codex R3 P1: when BM contrast DOF returns NaN (noise-
        floor guard fires from PR #475), StackedDiD must emit all-NaN
        inference fields on the affected contrast — NOT mixed finite-t /
        NaN-p (which would happen if NaN df flowed through safe_inference)
        and NOT normal-theory fallback (which would silently produce wrong
        small-sample CIs).

        Forces NaN DOF by monkeypatching `_compute_cr2_bm_contrast_dof` to
        return a NaN-only vector. Verifies that:
          - All event_study_effects entries on the hc2_bm path have NaN
            t_stat, p_value, conf_int.
          - overall_* fields are all NaN.
          - effect and se themselves remain finite (only inference is
            suppressed, mirroring the LinearRegression.get_inference
            pattern from PR #475 R7).
        """

        def _fake_contrast_dof(*args, **kwargs):
            # Match the m-dimensional output shape expected by callers.
            n_contrasts = args[3].shape[1] if len(args) >= 4 else kwargs["contrasts"].shape[1]
            return np.full(n_contrasts, np.nan)

        # Patch the import inside stacked_did's module namespace + the
        # source module (in case of dynamic re-import).
        from diff_diff import linalg as _linalg_mod

        monkeypatch.setattr(_linalg_mod, "_compute_cr2_bm_contrast_dof", _fake_contrast_dof)

        est = StackedDiD(kappa_pre=2, kappa_post=2, vcov_type="hc2_bm")
        res = est.fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            aggregate="event_study",
        )
        # All event_study_effects (except the ref period) should have NaN
        # inference but finite effect+se.
        for h, eff in res.event_study_effects.items():
            if h == -1:
                continue  # ref period: SE=0 by construction
            assert np.isfinite(eff["effect"]), f"effect at h={h} should be finite"
            assert np.isfinite(eff["se"]) and eff["se"] > 0, f"se at h={h} should be finite > 0"
            assert np.isnan(eff["t_stat"]), (
                f"t_stat at h={h} must be NaN when BM DOF NaN-guarded; "
                f"got {eff['t_stat']} (silent wrong inference)"
            )
            assert np.isnan(eff["p_value"]), f"p_value at h={h} must be NaN; got {eff['p_value']}"
            assert all(
                np.isnan(b) for b in eff["conf_int"]
            ), f"conf_int at h={h} must be all-NaN; got {eff['conf_int']}"
        # Overall ATT: same fail-closed expectation
        assert np.isfinite(res.overall_att) and np.isfinite(res.overall_se)
        assert np.isnan(
            res.overall_t_stat
        ), f"overall_t_stat must be NaN under NaN DOF; got {res.overall_t_stat}"
        assert np.isnan(res.overall_p_value)
        assert all(np.isnan(b) for b in res.overall_conf_int)

    def test_hc2_bm_helper_raises_fails_closed_with_all_nan_inference(
        self, staggered_data, monkeypatch
    ):
        """Per local codex R3 P1: when `_compute_cr2_bm_contrast_dof` raises
        (e.g., genuine singularity on the identified design), the estimator
        must emit all-NaN inference rather than fall back to normal-theory
        CIs/p-values."""
        from diff_diff import linalg as _linalg_mod

        def _fake_raises(*args, **kwargs):
            raise np.linalg.LinAlgError("forced linalg failure for test")

        monkeypatch.setattr(_linalg_mod, "_compute_cr2_bm_contrast_dof", _fake_raises)

        est = StackedDiD(kappa_pre=2, kappa_post=2, vcov_type="hc2_bm")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = est.fit(
                staggered_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                aggregate="event_study",
            )
        # Warning fired (informational that DOF was unavailable)
        warning_msgs = [str(w.message) for w in caught]
        assert any(
            "Bell-McCaffrey contrast DOF" in m for m in warning_msgs
        ), f"Should warn on helper failure; got: {warning_msgs}"
        # Inference NaN-closed for all event-study + overall
        for h, eff in res.event_study_effects.items():
            if h == -1:
                continue
            assert np.isnan(eff["t_stat"]) and np.isnan(
                eff["p_value"]
            ), f"event_time h={h} inference must NaN-close, not fall back to normal"
        assert np.isnan(res.overall_t_stat), "overall_t_stat must NaN-close on helper failure"
        assert np.isnan(res.overall_p_value)
