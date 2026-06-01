"""Pure-logic tests for the reviewer-eval engine (no codex, no network).

Covers what survives the carve-back to the minimal comparison harness: the run
store round-trip + content-hash keying, the model JSON round-trip, and the
side-by-side comparison bundle (rendered from each run's case snapshot). These
run in normal CI; skipped only when the harness isn't on disk.
"""

import pathlib
import sys

import pytest

_EVAL_ROOT = pathlib.Path(__file__).resolve().parent.parent / "tools" / "reviewer-eval"

pytestmark = pytest.mark.skipif(
    not _EVAL_ROOT.exists(),
    reason="reviewer-eval eval harness not present (isolated install)",
)

if _EVAL_ROOT.exists() and str(_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVAL_ROOT))


# --------------------------------------------------------------------------- #
# Modules import + model JSON round-trip.
# --------------------------------------------------------------------------- #


def test_engine_modules_import():
    import engine.compare  # noqa: F401
    import engine.models  # noqa: F401
    import engine.runner  # noqa: F401
    import engine.store  # noqa: F401


def test_models_json_roundtrip():
    from engine.models import RunResult, run_result_from_dict, to_jsonable

    rr = RunResult(
        case_id="c",
        config_id="A",
        repeat_idx=0,
        review_markdown="hi",
        model="gpt-5.4",
        run_id="c.A.r0.deadbeef",
        case_snapshot={"stratum": "s1_synthetic", "ground_truth": []},
    )
    assert run_result_from_dict(to_jsonable(rr)) == rr


# --------------------------------------------------------------------------- #
# Run store: round-trip + content-hash keying.
# --------------------------------------------------------------------------- #


def test_store_roundtrip(tmp_path):
    from engine.models import RunResult
    from engine.store import RunStore, run_key

    store = RunStore(str(tmp_path / "runs"))
    key = run_key("c", "A", 0, "tag1")
    rr = RunResult(case_id="c", config_id="A", repeat_idx=0, review_markdown="x", run_id=key)
    store.save(key, rr)
    assert store.has(key)
    assert store.load(key) == rr
    assert [r.run_id for r in store.load_all()] == [key]


def test_run_key_distinct_by_experiment_tag():
    from engine.store import run_key

    # Same case/config/repeat but different experiment identity must not collide.
    assert run_key("c", "A", 0, "tag-gpt54") != run_key("c", "A", 0, "tag-gpt55")
    # ...and the key is stable for a fixed identity.
    assert run_key("c", "A", 0, "t") == run_key("c", "A", 0, "t")


def test_run_key_distinct_by_case_tag():
    from engine.store import run_key

    base = ("c", "A", 0, "exp")
    # Same case/config/repeat/experiment but different case identity must NOT collide.
    assert run_key(*base, "casetag1") != run_key(*base, "casetag2")
    # Stable for a fixed case identity.
    assert run_key(*base, "ct") == run_key(*base, "ct")
    # case_tag genuinely participates (a case edit can't alias the no-case-tag key).
    assert run_key(*base) != run_key(*base, "ct")


# --------------------------------------------------------------------------- #
# Comparison bundle: rendered from each run's case_snapshot (not the corpus).
# --------------------------------------------------------------------------- #


def _bug(**kw):
    d = {
        "id": "c1:b1",
        "file": "f.py",
        "line_window": [10, 20],
        "bug_class": "x",
        "expected_severity": "P1",
        "rationale": "removed guard",
    }
    d.update(kw)
    return d


def _snap(stratum="s1_synthetic", **kw):
    d = {
        "title": "T",
        "stratum": stratum,
        "ground_truth": [_bug()],
        "expect_no_blockers": False,
        "allow_severities": ["P2", "P3"],
        "known_fp_topics": [],
    }
    d.update(kw)
    return d


def _run(case_id, config_id, review, snap, **kw):
    from engine.models import RunResult

    return RunResult(
        case_id=case_id,
        config_id=config_id,
        repeat_idx=0,
        review_markdown=review,
        model=kw.pop("model", "m"),
        case_snapshot=snap,
        **kw,
    )


def test_build_bundle_has_ground_truth_and_both_reviews():
    from engine.compare import build_bundle

    snap = _snap()
    runs = [
        _run("c1", "A", "A says: bug at f.py", snap, model="gpt-5.4"),
        _run("c1", "B", "B says: looks fine", snap, model="gpt-5.5"),
    ]
    out = build_bundle(runs)
    assert "c1:b1" in out
    assert "removed guard" in out
    assert "A says: bug at f.py" in out
    assert "B says: looks fine" in out
    # The grading instruction must point readers at the real rubric.
    assert "pr_review.md" in out


def test_build_bundle_marks_negative_control():
    from engine.compare import build_bundle

    snap = _snap(stratum="s3_negative", title="", ground_truth=[], expect_no_blockers=True)
    out = build_bundle([_run("cl", "A", "ok", snap, model="gpt-5.4")])
    assert "NO known bugs" in out


def test_build_bundle_infra_error_surfaced_not_as_review():
    from engine.compare import build_bundle

    snap = _snap(ground_truth=[])
    out = build_bundle([_run("c", "A", "", snap, model="gpt-5.4", infra_error="codex timeout")])
    assert "INFRA_ERROR" in out and "codex timeout" in out


def test_build_bundle_fence_survives_embedded_code_fences():
    """A review containing ``` fences must not break the bundle's own fence."""
    from engine.compare import build_bundle

    review = "Here is code:\n```python\nx = 1\n```\nthat's the bug."
    out = build_bundle([_run("c", "A", review, _snap(), model="gpt-5.4")])
    # The whole review (including its embedded fence) must appear verbatim.
    assert "```python\nx = 1\n```" in out


def test_build_bundle_renders_only_cases_with_runs():
    """A subset run renders only its own cases — no empty placeholder sections."""
    from engine.compare import build_bundle

    out = build_bundle([_run("only", "A", "R", _snap(title="Only"))])
    assert "## only" in out
    assert "_(no runs for this case)_" not in out
