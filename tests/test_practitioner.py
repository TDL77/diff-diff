"""Tests for the practitioner guidance module."""

import numpy as np
import pytest

from diff_diff import (
    BaconDecomposition,
    CallawaySantAnna,
    DifferenceInDifferences,
    HeterogeneousAdoptionDiDEventStudyResults,
    HeterogeneousAdoptionDiDResults,
    MultiPeriodDiD,
    generate_did_data,
    generate_staggered_data,
)
from diff_diff.continuous_did_results import ContinuousDiDResults
from diff_diff.efficient_did_results import EfficientDiDResults
from diff_diff.imputation_results import ImputationDiDResults
from diff_diff.practitioner import STEPS, practitioner_next_steps
from diff_diff.results import DiDResults, SyntheticDiDResults
from diff_diff.stacked_did_results import StackedDiDResults
from diff_diff.sun_abraham import SunAbrahamResults
from diff_diff.triple_diff import TripleDifferenceResults
from diff_diff.trop_results import TROPResults
from diff_diff.two_stage_results import TwoStageDiDResults


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def did_data():
    return generate_did_data(n_units=50, treatment_effect=3.0, seed=42)


@pytest.fixture(scope="session")
def staggered_data():
    return generate_staggered_data(n_units=60, n_periods=8, treatment_effect=2.0, seed=42)


@pytest.fixture(scope="session")
def did_results(did_data):
    did = DifferenceInDifferences()
    return did.fit(did_data, outcome="outcome", treatment="treated", time="post")


@pytest.fixture(scope="session")
def multi_period_results(did_data):
    es = MultiPeriodDiD()
    return es.fit(did_data, outcome="outcome", unit="unit", time="period", treatment="treated")


@pytest.fixture(scope="session")
def cs_results(staggered_data):
    cs = CallawaySantAnna()
    return cs.fit(
        staggered_data,
        outcome="outcome",
        unit="unit",
        time="period",
        first_treat="first_treat",
    )


@pytest.fixture(scope="session")
def bacon_results(staggered_data):
    bacon = BaconDecomposition()
    return bacon.fit(
        staggered_data,
        outcome="outcome",
        unit="unit",
        time="period",
        first_treat="first_treat",
    )


# ---------------------------------------------------------------------------
# Mock result fixtures for expensive estimators
# ---------------------------------------------------------------------------
def _mock_result(cls, **overrides):
    """Create a minimal mock of a results dataclass."""
    # Provide default fields that most result types share
    defaults = dict(
        att=0.5,
        se=0.1,
        t_stat=5.0,
        p_value=0.001,
        conf_int=(0.3, 0.7),
        n_obs=100,
        n_treated=50,
        n_control=50,
    )
    defaults.update(overrides)
    try:
        return cls(**defaults)
    except TypeError:
        # Some result classes have different required fields
        return cls.__new__(cls)


@pytest.fixture
def mock_synth_results():
    r = SyntheticDiDResults.__new__(SyntheticDiDResults)
    r.att = 1.0
    r.se = 0.3
    return r


@pytest.fixture
def mock_trop_results():
    r = TROPResults.__new__(TROPResults)
    r.att = 0.8
    r.se = 0.2
    return r


@pytest.fixture
def mock_efficient_results():
    r = EfficientDiDResults.__new__(EfficientDiDResults)
    r.overall_att = 0.6
    r.overall_se = 0.15
    return r


@pytest.fixture
def mock_continuous_results():
    r = ContinuousDiDResults.__new__(ContinuousDiDResults)
    r.overall_att = 0.4
    r.overall_se = 0.1
    return r


@pytest.fixture
def mock_triple_results():
    r = TripleDifferenceResults.__new__(TripleDifferenceResults)
    r.att = 0.7
    r.se = 0.2
    return r


@pytest.fixture
def mock_sa_results():
    r = SunAbrahamResults.__new__(SunAbrahamResults)
    r.overall_att = 0.5
    r.overall_se = 0.1
    return r


@pytest.fixture
def mock_imputation_results():
    r = ImputationDiDResults.__new__(ImputationDiDResults)
    r.overall_att = 0.5
    r.overall_se = 0.1
    return r


@pytest.fixture
def mock_two_stage_results():
    r = TwoStageDiDResults.__new__(TwoStageDiDResults)
    r.overall_att = 0.5
    r.overall_se = 0.1
    return r


@pytest.fixture
def mock_stacked_results():
    r = StackedDiDResults.__new__(StackedDiDResults)
    r.overall_att = 0.5
    r.overall_se = 0.1
    return r


@pytest.fixture
def mock_had_results():
    r = HeterogeneousAdoptionDiDResults.__new__(HeterogeneousAdoptionDiDResults)
    r.att = 0.5
    return r


@pytest.fixture
def mock_had_event_study_results():
    r = HeterogeneousAdoptionDiDEventStudyResults.__new__(HeterogeneousAdoptionDiDEventStudyResults)
    # 5 horizons: e in {-3, -2, 0, 1, 2}
    r.att = np.array([0.01, -0.02, 0.30, 0.45, 0.50])
    r.event_times = np.array([-3, -2, 0, 1, 2])
    return r


@pytest.fixture
def mock_had_results_nan_att():
    r = HeterogeneousAdoptionDiDResults.__new__(HeterogeneousAdoptionDiDResults)
    r.att = float("nan")
    return r


@pytest.fixture
def mock_had_event_study_results_all_nan():
    r = HeterogeneousAdoptionDiDEventStudyResults.__new__(HeterogeneousAdoptionDiDEventStudyResults)
    r.att = np.full(5, np.nan)
    return r


@pytest.fixture
def mock_had_event_study_results_partial_nan():
    r = HeterogeneousAdoptionDiDEventStudyResults.__new__(HeterogeneousAdoptionDiDEventStudyResults)
    r.att = np.array([0.5, np.nan, 0.3])
    return r


# ---------------------------------------------------------------------------
# Tests: return schema
# ---------------------------------------------------------------------------
class TestReturnSchema:
    def test_has_expected_keys(self, did_results):
        output = practitioner_next_steps(did_results, verbose=False)
        assert "estimator" in output
        assert "completed" in output
        assert "next_steps" in output
        assert "warnings" in output

    def test_estimator_name(self, did_results):
        output = practitioner_next_steps(did_results, verbose=False)
        assert output["estimator"] == "DifferenceInDifferences"

    def test_estimation_always_completed(self, did_results):
        output = practitioner_next_steps(did_results, verbose=False)
        assert "estimation" in output["completed"]

    def test_steps_1_and_2_always_emitted(self, did_results):
        """Steps 1 (target parameter) and 2 (assumptions) should always appear."""
        output = practitioner_next_steps(did_results, verbose=False)
        baker_steps = [s["baker_step"] for s in output["next_steps"]]
        assert 1 in baker_steps, "Step 1 (target parameter) missing"
        assert 2 in baker_steps, "Step 2 (assumptions) missing"

    def test_steps_1_and_2_filterable(self, did_results):
        """Agents can filter Steps 1-2 via completed_steps."""
        output = practitioner_next_steps(
            did_results,
            completed_steps=["target_parameter", "assumptions"],
            verbose=False,
        )
        baker_steps = [s["baker_step"] for s in output["next_steps"]]
        assert 1 not in baker_steps
        assert 2 not in baker_steps

    def test_next_steps_are_dicts(self, did_results):
        output = practitioner_next_steps(did_results, verbose=False)
        for step in output["next_steps"]:
            assert "baker_step" in step
            assert "label" in step
            assert "why" in step
            assert "code" in step
            assert "priority" in step

    def test_warnings_are_strings(self, did_results):
        output = practitioner_next_steps(did_results, verbose=False)
        for w in output["warnings"]:
            assert isinstance(w, str)


# ---------------------------------------------------------------------------
# Tests: each result type produces guidance
# ---------------------------------------------------------------------------
class TestResultTypeDispatch:
    def test_did_results(self, did_results):
        output = practitioner_next_steps(did_results, verbose=False)
        assert len(output["next_steps"]) > 0

    def test_multi_period_results(self, multi_period_results):
        output = practitioner_next_steps(multi_period_results, verbose=False)
        assert len(output["next_steps"]) > 0
        assert output["estimator"] == "MultiPeriodDiD (Event Study)"

    def test_cs_results(self, cs_results):
        output = practitioner_next_steps(cs_results, verbose=False)
        assert len(output["next_steps"]) > 0
        assert output["estimator"] == "CallawaySantAnna"

    def test_bacon_results(self, bacon_results):
        output = practitioner_next_steps(bacon_results, verbose=False)
        assert len(output["next_steps"]) > 0
        assert output["estimator"] == "BaconDecomposition"
        # Bacon should suggest switching to a robust estimator
        labels = [s["label"] for s in output["next_steps"]]
        assert any("heterogeneity-robust" in lbl for lbl in labels)

    def test_sa_results(self, mock_sa_results):
        output = practitioner_next_steps(mock_sa_results, verbose=False)
        assert len(output["next_steps"]) > 0
        assert output["estimator"] == "SunAbraham"
        # SA guidance should use to_dataframe, NOT aggregate='group'
        all_code = " ".join(s.get("code", "") for s in output["next_steps"])
        assert "aggregate=" not in all_code or "to_dataframe" in all_code

    def test_imputation_results(self, mock_imputation_results):
        output = practitioner_next_steps(mock_imputation_results, verbose=False)
        assert len(output["next_steps"]) > 0
        # ImputationDiD has no control_group parameter — code snippets must not use it
        all_code = " ".join(s.get("code", "") for s in output["next_steps"])
        assert "control_group" not in all_code

    def test_two_stage_results(self, mock_two_stage_results):
        output = practitioner_next_steps(mock_two_stage_results, verbose=False)
        assert len(output["next_steps"]) > 0
        # TwoStageDiD has no control_group parameter — code snippets must not use it
        all_code = " ".join(s.get("code", "") for s in output["next_steps"])
        assert "control_group" not in all_code

    def test_stacked_results(self, mock_stacked_results):
        output = practitioner_next_steps(mock_stacked_results, verbose=False)
        assert len(output["next_steps"]) > 0
        # StackedDiD uses clean_control, not control_group
        all_text = " ".join(s.get("code", "") + s.get("why", "") for s in output["next_steps"])
        assert "not_yet_treated" not in all_text or "clean_control" in all_text

    def test_synth_results(self, mock_synth_results):
        output = practitioner_next_steps(mock_synth_results, verbose=False)
        assert len(output["next_steps"]) > 0
        assert output["estimator"] == "SyntheticDiD"
        # SDiD handler steps (exclude generic Steps 1-2) should NOT use staggered knobs
        handler_steps = [s for s in output["next_steps"] if s["baker_step"] > 2]
        all_code = " ".join(s.get("code", "") for s in handler_steps)
        assert "control_group" not in all_code
        assert "anticipation" not in all_code

    def test_trop_results(self, mock_trop_results):
        output = practitioner_next_steps(mock_trop_results, verbose=False)
        assert len(output["next_steps"]) > 0
        # TROP handler steps (exclude generic Steps 1-2) should NOT use staggered knobs
        handler_steps = [s for s in output["next_steps"] if s["baker_step"] > 2]
        all_code = " ".join(s.get("code", "") for s in handler_steps)
        assert "control_group" not in all_code
        assert "anticipation" not in all_code

    def test_efficient_results(self, mock_efficient_results):
        output = practitioner_next_steps(mock_efficient_results, verbose=False)
        assert len(output["next_steps"]) > 0
        # EfficientDiD uses never_treated/last_cohort — code must not suggest not_yet_treated
        all_code = " ".join(s.get("code", "") for s in output["next_steps"])
        assert "not_yet_treated" not in all_code

    def test_continuous_results(self, mock_continuous_results):
        output = practitioner_next_steps(mock_continuous_results, verbose=False)
        assert len(output["next_steps"]) > 0
        # ContinuousDiD should NOT emit check_parallel_trends
        all_text = " ".join(s.get("code", "") for s in output["next_steps"])
        assert "check_parallel_trends" not in all_text

    def test_triple_results(self, mock_triple_results):
        output = practitioner_next_steps(mock_triple_results, verbose=False)
        assert len(output["next_steps"]) > 0
        # DDD should NOT claim "requires PT along two dimensions"
        all_text = " ".join(s.get("why", "") for s in output["next_steps"])
        assert "two dimensions" not in all_text
        assert "check_parallel_trends" not in " ".join(
            s.get("code", "") for s in output["next_steps"]
        )


# ---------------------------------------------------------------------------
# Tests: completed_steps filtering
# ---------------------------------------------------------------------------
class TestCompletedSteps:
    def test_filter_parallel_trends(self, cs_results):
        full = practitioner_next_steps(cs_results, verbose=False)
        filtered = practitioner_next_steps(
            cs_results, completed_steps=["parallel_trends"], verbose=False
        )
        assert len(filtered["next_steps"]) < len(full["next_steps"])
        # No step should have baker_step 3 about parallel trends
        for s in filtered["next_steps"]:
            if s["baker_step"] == 3:
                assert "parallel trends" not in s["label"].lower()

    def test_filter_sensitivity(self, cs_results):
        full = practitioner_next_steps(cs_results, verbose=False)
        filtered = practitioner_next_steps(
            cs_results, completed_steps=["sensitivity"], verbose=False
        )
        assert len(filtered["next_steps"]) < len(full["next_steps"])

    def test_filter_all_steps(self, cs_results):
        output = practitioner_next_steps(cs_results, completed_steps=list(STEPS), verbose=False)
        assert len(output["next_steps"]) == 0

    def test_invalid_step_name_raises(self, did_results):
        with pytest.raises(ValueError, match="Unknown step names"):
            practitioner_next_steps(did_results, completed_steps=["invalid_step"], verbose=False)


# ---------------------------------------------------------------------------
# Tests: verbose output
# ---------------------------------------------------------------------------
class TestVerboseOutput:
    def test_verbose_prints(self, did_results, capsys):
        practitioner_next_steps(did_results, verbose=True)
        captured = capsys.readouterr()
        assert "Practitioner Guidance" in captured.out
        assert "Baker et al." in captured.out
        assert "DifferenceInDifferences" in captured.out

    def test_no_print_when_silent(self, did_results, capsys):
        practitioner_next_steps(did_results, verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""


# ---------------------------------------------------------------------------
# Tests: NaN handling
# ---------------------------------------------------------------------------
class TestNaNHandling:
    def test_nan_att_produces_warning(self):
        r = DiDResults(
            att=float("nan"),
            se=float("nan"),
            t_stat=float("nan"),
            p_value=float("nan"),
            conf_int=(float("nan"), float("nan")),
            n_obs=100,
            n_treated=50,
            n_control=50,
        )
        output = practitioner_next_steps(r, verbose=False)
        assert len(output["warnings"]) > 0
        assert any("NaN" in w for w in output["warnings"])

    def test_nan_avg_att_multi_period(self):
        """MultiPeriodDiDResults uses avg_att, not att."""
        from diff_diff.results import MultiPeriodDiDResults

        r = MultiPeriodDiDResults.__new__(MultiPeriodDiDResults)
        r.avg_att = float("nan")
        output = practitioner_next_steps(r, verbose=False)
        assert any("NaN" in w for w in output["warnings"])


# ---------------------------------------------------------------------------
# Tests: Bacon handler warnings
# ---------------------------------------------------------------------------
class TestBaconWarnings:
    def test_forbidden_comparison_warning(self, bacon_results):
        output = practitioner_next_steps(bacon_results, verbose=False)
        # Real Bacon results from staggered data should have forbidden comparisons
        weight = getattr(bacon_results, "total_weight_later_vs_earlier", 0)
        if weight > 0.01:
            assert any("contaminated" in w for w in output["warnings"])

    def test_bacon_with_high_forbidden_weight(self):
        """Mock Bacon results with high forbidden comparison weight."""
        from diff_diff.bacon import BaconDecompositionResults

        r = BaconDecompositionResults.__new__(BaconDecompositionResults)
        r.overall_att = 0.5
        r.total_weight_later_vs_earlier = 0.4
        r.comparisons = []
        output = practitioner_next_steps(r, verbose=False)
        assert any("contaminated" in w for w in output["warnings"])
        assert any("40%" in w for w in output["warnings"])


# ---------------------------------------------------------------------------
# Tests: EfficientDiD handler path
# ---------------------------------------------------------------------------
class TestEfficientDiDHandler:
    def test_hausman_pretest_in_guidance(self, mock_efficient_results):
        output = practitioner_next_steps(mock_efficient_results, verbose=False)
        labels = [s["label"] for s in output["next_steps"]]
        assert any("hausman" in lbl.lower() or "Hausman" in lbl for lbl in labels)

    def test_hausman_snippet_uses_classmethod(self, mock_efficient_results):
        output = practitioner_next_steps(mock_efficient_results, verbose=False)
        hausman_steps = [
            s
            for s in output["next_steps"]
            if "hausman" in s["label"].lower() or "Hausman" in s["label"]
        ]
        assert len(hausman_steps) > 0
        assert "hausman_pretest" in hausman_steps[0]["code"]


# ---------------------------------------------------------------------------
# Tests: unknown result type fallback
# ---------------------------------------------------------------------------
class TestFallback:
    def test_unknown_type(self):
        class FakeResults:
            att = 1.0
            se = 0.5

        output = practitioner_next_steps(FakeResults(), verbose=False)
        assert len(output["next_steps"]) > 0
        assert output["estimator"] == "FakeResults"


# ---------------------------------------------------------------------------
# Tests: HeterogeneousAdoptionDiD (HAD) handler dispatch
# ---------------------------------------------------------------------------
class TestHADDispatch:
    def test_had_results_dispatch(self, mock_had_results):
        output = practitioner_next_steps(mock_had_results, verbose=False)
        assert len(output["next_steps"]) > 0
        assert output["estimator"] == "HeterogeneousAdoptionDiD (HAD)"

    def test_had_event_study_dispatch(self, mock_had_event_study_results):
        output = practitioner_next_steps(mock_had_event_study_results, verbose=False)
        assert len(output["next_steps"]) > 0
        assert output["estimator"] == "HeterogeneousAdoptionDiD (Event Study)"

    def test_had_pretest_workflow_referenced(self, mock_had_results):
        output = practitioner_next_steps(mock_had_results, verbose=False)
        all_code = " ".join(s.get("code", "") for s in output["next_steps"])
        assert "did_had_pretest_workflow" in all_code

    def test_had_event_study_pretest_workflow_referenced(self, mock_had_event_study_results):
        output = practitioner_next_steps(mock_had_event_study_results, verbose=False)
        all_code = " ".join(s.get("code", "") for s in output["next_steps"])
        assert "did_had_pretest_workflow" in all_code
        assert "aggregate='event_study'" in all_code

    def test_had_bandwidth_diagnostics_referenced(self, mock_had_results):
        output = practitioner_next_steps(mock_had_results, verbose=False)
        all_text = " ".join(
            (s.get("code", "") + " " + s.get("why", "")) for s in output["next_steps"]
        )
        assert "bandwidth_diagnostics" in all_text

    def test_had_event_study_simultaneous_bands_referenced(self, mock_had_event_study_results):
        output = practitioner_next_steps(mock_had_event_study_results, verbose=False)
        all_text = " ".join(
            (s.get("code", "") + " " + s.get("why", "")) for s in output["next_steps"]
        )
        assert "cband" in all_text
        # Either "sup-t" wording or "simultaneous" wording is acceptable.
        assert ("sup-t" in all_text) or ("simultaneous" in all_text)

    def test_had_no_comparison_group_framing(self, mock_had_results, mock_had_event_study_results):
        for fixture in (mock_had_results, mock_had_event_study_results):
            output = practitioner_next_steps(fixture, verbose=False)
            all_text = " ".join(
                (s.get("code", "") + " " + s.get("why", "") + " " + s.get("label", ""))
                for s in output["next_steps"]
            )
            all_text += " ".join(output["warnings"])
            assert "no comparison group" not in all_text.lower()
            assert "missing comparison" not in all_text.lower()

    def test_had_nan_warning_scalar(self, mock_had_results_nan_att):
        output = practitioner_next_steps(mock_had_results_nan_att, verbose=False)
        warnings = " ".join(output["warnings"])
        assert "NaN" in warnings or "nan" in warnings.lower()

    def test_had_event_study_nan_warning_array(self, mock_had_event_study_results_all_nan):
        output = practitioner_next_steps(mock_had_event_study_results_all_nan, verbose=False)
        warnings = " ".join(output["warnings"])
        assert "per-horizon" in warnings or "All" in warnings

    def test_had_partial_nan_array_no_warning(self, mock_had_event_study_results_partial_nan):
        # Partial-NaN arrays are legitimate event-study output (some
        # horizons may collapse on degenerate-design grounds while others
        # remain finite). The all-NaN warning must NOT fire here.
        output = practitioner_next_steps(mock_had_event_study_results_partial_nan, verbose=False)
        # No "per-horizon" or "All ... NaN" warning string should appear.
        warnings = " ".join(output["warnings"])
        assert "per-horizon" not in warnings
        assert "All " not in warnings

    def test_had_step_4_estimator_selection_present(
        self, mock_had_results, mock_had_event_study_results
    ):
        # Step-4 must surface the WAS-vs-ATT(d) estimand difference (not
        # a blanket "if untreated → not HAD" rule which would contradict
        # REGISTRY § HeterogeneousAdoptionDiD edge cases lines ~2403/2408).
        for fixture in (mock_had_results, mock_had_event_study_results):
            output = practitioner_next_steps(fixture, verbose=False)
            step_4_steps = [s for s in output["next_steps"] if s["baker_step"] == 4]
            assert len(step_4_steps) >= 1
            all_text = " ".join(
                (s.get("code", "") + " " + s.get("why", "") + " " + s.get("label", ""))
                for s in step_4_steps
            )
            # Routing nudge must name ContinuousDiD as the estimand
            # alternative; framing must center on WAS vs ATT(d) (the
            # actual estimand differentiator), NOT on whether untreated
            # units exist.
            assert "ContinuousDiD" in all_text
            assert "WAS" in all_text
            assert "ATT(d)" in all_text

    def test_had_step_4_does_not_misframe_untreated_unit_routing(
        self, mock_had_results, mock_had_event_study_results
    ):
        # Per REGISTRY: HAD is compatible with a small share of
        # never-treated units (paper edge case), and on staggered
        # event-study panels never-treated units are explicitly RETAINED
        # (Appendix B.2 / had.py:1325). The Step-4 routing must NOT
        # carry the wrong "if untreated → not HAD" framing.
        for fixture in (mock_had_results, mock_had_event_study_results):
            output = practitioner_next_steps(fixture, verbose=False)
            step_4_steps = [s for s in output["next_steps"] if s["baker_step"] == 4]
            all_text = " ".join(
                (s.get("code", "") + " " + s.get("why", "") + " " + s.get("label", ""))
                for s in step_4_steps
            ).lower()
            forbidden_phrases = (
                "switch away from had",
                "had's was divisor under-weights",
                "drop untreated",
                "must drop never-treated",
            )
            for phrase in forbidden_phrases:
                assert phrase not in all_text, (
                    f"HAD Step-4 must not carry the phrase {phrase!r}: "
                    f"per REGISTRY § HeterogeneousAdoptionDiD edge cases, "
                    f"HAD is compatible with a small share of never-treated "
                    f"units and explicitly retains them on staggered "
                    f"event-study panels."
                )

    def test_handle_continuous_step_4_routes_to_had(self, mock_continuous_results):
        # Symmetric pair: ContinuousDiD users with no untreated units
        # should be routed to HeterogeneousAdoptionDiD.
        output = practitioner_next_steps(mock_continuous_results, verbose=False)
        step_4_steps = [s for s in output["next_steps"] if s["baker_step"] == 4]
        assert len(step_4_steps) >= 1
        all_text = " ".join((s.get("code", "") + " " + s.get("why", "")) for s in step_4_steps)
        assert "HeterogeneousAdoptionDiD" in all_text

    def test_handle_generic_ndarray_att_triggers_warning(self):
        # Cross-handler regression: a future estimator that returns
        # ndarray att and falls through to _handle_generic must produce
        # the same all-NaN warning as the dedicated HAD event-study path.
        class FutureNdarrayAttResults:
            att: np.ndarray

        r = FutureNdarrayAttResults()
        r.att = np.full(3, np.nan)
        output = practitioner_next_steps(r, verbose=False)
        warnings = " ".join(output["warnings"])
        assert "per-horizon" in warnings or "All" in warnings

    def test_had_handlers_string_only_no_attribute_reads(
        self, mock_had_results, mock_had_event_study_results
    ):
        # Stability invariant #7: handlers are STRING-ONLY at runtime.
        # The fixtures construct results with ONLY .att (and event_times
        # on the event-study fixture); confirm no AttributeError is
        # raised when the handlers run. Protects against a future
        # refactor that starts reading result.<some_field> inside a
        # handler and silently breaks the minimal-fixture contract.
        for fixture in (mock_had_results, mock_had_event_study_results):
            output = practitioner_next_steps(fixture, verbose=False)
            assert isinstance(output, dict)
            assert "next_steps" in output

    def test_had_handler_snippets_are_valid_python_syntax(
        self, mock_had_results, mock_had_event_study_results
    ):
        # Snippet smoke test: every code block emitted by the HAD
        # handlers must parse as valid Python. Catches the failure mode
        # where snippets reference undefined names with placeholder
        # syntax that doesn't compile (e.g. `survey_design=design` with
        # no `design` defined in scope, or attribute typos that break
        # copy/paste).
        import ast

        for fixture in (mock_had_results, mock_had_event_study_results):
            output = practitioner_next_steps(fixture, verbose=False)
            for step in output["next_steps"]:
                code = step.get("code", "")
                if not code.strip():
                    continue
                try:
                    ast.parse(code)
                except SyntaxError as e:
                    pytest.fail(
                        f"Step {step['baker_step']} ({step['label']!r}) "
                        f"emits a code snippet that does not parse as "
                        f"valid Python: {e}\n\nSnippet:\n{code}"
                    )

    def test_handle_continuous_step_4_snippet_is_valid_python(self, mock_continuous_results):
        # Same syntax check on the symmetric Step-4 in _handle_continuous.
        import ast

        output = practitioner_next_steps(mock_continuous_results, verbose=False)
        step_4_steps = [s for s in output["next_steps"] if s["baker_step"] == 4]
        for step in step_4_steps:
            code = step.get("code", "")
            if code.strip():
                ast.parse(code)  # raises SyntaxError on failure

    def test_had_step_3_pretest_assumption_labels_correct(self, mock_had_results):
        # Per docs/methodology/REGISTRY.md and diff_diff/had_pretests.py
        # docstrings:
        #   - did_had_pretest_workflow(aggregate="overall") covers paper
        #     Section 4.2 steps 1 + 3 ONLY; step 2 (Assumption 7
        #     pre-trends) is explicitly NOT covered on the overall path.
        #   - qug_test = support-infimum test (H0: d_lower = 0),
        #     NOT "Assumption 5" (Design 1 sign identification, which is
        #     not testable per registry).
        #   - stute_test = Assumption 8 linearity, NOT Assumption 7
        #     mean-independence.
        # The single-period Step-3 guidance must not mislabel these.
        output = practitioner_next_steps(mock_had_results, verbose=False)
        step_3_steps = [s for s in output["next_steps"] if s["baker_step"] == 3]
        assert len(step_3_steps) == 1
        why = step_3_steps[0].get("why", "")
        # Must NOT call QUG an "Assumption 5" test.
        assert "QUG (Assumption 5" not in why, (
            "Step-3 why-text must not call QUG an 'Assumption 5' test - "
            "QUG tests H_0: d_lower = 0 (paper Theorem 4); Assumption 5 "
            "is the Design 1 sign-identification condition and is NOT "
            "testable per registry."
        )
        # Must NOT claim Stute is Assumption 7 mean-independence.
        forbidden = (
            "Stute (Assumption 7",
            "Stute / Yatchew-HR Assumption 7",
            "Assumption 7 mean-independence",
        )
        for phrase in forbidden:
            assert phrase not in why, (
                f"Step-3 why-text must not carry the phrase {phrase!r} - "
                f"stute_test / yatchew_hr_test are Assumption 8 linearity "
                f"tests (paper Section 4.2 step 3); Assumption 7 (pre-trends) "
                f"is paper step 2 and is NOT covered on the overall workflow "
                f"path - the workflow's verdict explicitly flags that gap."
            )
        # Must positively acknowledge the Assumption 7 / step 2 gap on
        # the overall path (not silently imply it's covered).
        assert "Assumption 7" in why or "step 2" in why, (
            "Step-3 why-text must explicitly mention Assumption 7 / step 2 "
            "to acknowledge the gap on the overall workflow path - "
            "agents reading the guidance must not assume the workflow "
            "covers what it does not cover."
        )
