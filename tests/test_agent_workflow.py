"""Tests for the stateless agent_workflow() orchestrator.

Mirrors the content-stability pattern from tests/test_guides.py: assert
fingerprint strings appear in the output rather than pinning exact
formatting. The orchestrator's stability contract is that it names the
five canonical workflow primitives in a copy-pasteable script.
"""

import pandas as pd
import pytest

import diff_diff


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "firm_id": [1, 1, 2, 2],
            "year": [0, 1, 0, 1],
            "treated": [0, 0, 1, 1],
            "logwage": [0.1, 0.2, 0.1, 0.9],
        }
    )


def test_returns_dict_with_canonical_keys(df):
    out = diff_diff.agent_workflow(
        df,
        unit="firm_id",
        time="year",
        treatment="treated",
        outcome="logwage",
        verbose=False,
    )
    expected = {
        "profile_call",
        "guide_call",
        "fit_candidates",
        "validation_calls",
        "reporting_call",
        "script",
    }
    assert expected <= set(out.keys())


def test_script_names_canonical_workflow(df):
    out = diff_diff.agent_workflow(
        df,
        unit="firm_id",
        time="year",
        treatment="treated",
        outcome="logwage",
        verbose=False,
    )
    script = out["script"]
    for name in (
        "profile_panel",
        "get_llm_guide",
        "practitioner_next_steps",
        "BusinessReport",
    ):
        assert name in script, f"{name!r} missing from script"


def test_templates_column_names(df):
    out = diff_diff.agent_workflow(
        df,
        unit="firm_id",
        time="year",
        treatment="treated",
        outcome="logwage",
        first_treat="cohort",
        verbose=False,
    )
    script = out["script"]
    for col in ("firm_id", "year", "treated", "logwage", "cohort"):
        assert col in script, f"column {col!r} missing from templated script"


def test_first_treat_omitted_when_none(df):
    out = diff_diff.agent_workflow(
        df,
        unit="firm_id",
        time="year",
        treatment="treated",
        outcome="logwage",
        verbose=False,
    )
    # Without first_treat=, the templated Step 3 should NOT mention it.
    assert "first_treat=" not in out["script"]


def test_first_treat_appears_when_provided(df):
    out = diff_diff.agent_workflow(
        df,
        unit="firm_id",
        time="year",
        treatment="treated",
        outcome="logwage",
        first_treat="cohort_year",
        verbose=False,
    )
    assert 'first_treat="cohort_year"' in out["script"]


def test_does_not_inspect_df():
    # Pure orchestrator: a structurally-empty DataFrame must still produce
    # the templated script (no df inspection happens).
    out = diff_diff.agent_workflow(
        pd.DataFrame(),
        unit="a",
        time="b",
        treatment="c",
        outcome="d",
        verbose=False,
    )
    assert "profile_panel" in out["script"]
    assert 'unit="a"' in out["script"]


def test_fit_candidates_all_importable(df):
    """Every estimator name in fit_candidates must remain importable.

    Catches the drift case where an estimator is renamed but the
    orchestrator's candidates list still references the old name.
    """
    out = diff_diff.agent_workflow(
        df,
        unit="firm_id",
        time="year",
        treatment="treated",
        outcome="logwage",
        verbose=False,
    )
    for name in out["fit_candidates"]:
        assert hasattr(diff_diff, name), (
            f"agent_workflow advertises {name!r} but it's not on the "
            f"public surface — rename detected without orchestrator update."
        )


def test_verbose_true_prints_script(df, capsys):
    out = diff_diff.agent_workflow(
        df,
        unit="firm_id",
        time="year",
        treatment="treated",
        outcome="logwage",
        verbose=True,
    )
    captured = capsys.readouterr()
    assert "profile_panel" in captured.out
    assert out["script"] in captured.out


def test_verbose_false_silent(df, capsys):
    diff_diff.agent_workflow(
        df,
        unit="firm_id",
        time="year",
        treatment="treated",
        outcome="logwage",
        verbose=False,
    )
    captured = capsys.readouterr()
    assert captured.out == ""
