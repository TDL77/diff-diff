"""Runtime tests for the reviewer-eval harness (no codex, no network).

These exercise the real ``CodexReviewer.review()`` return contract and the
experiment-identity keying that resume relies on — the two bugs an earlier local
AI review caught (the ``ReviewOutput`` kwarg mismatch and run-artifact aliasing
across models) live here, so this is where they get a regression test.

``call_codex`` and the git worktree are stubbed.
"""

import pathlib
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parent.parent
_EVAL_ROOT = _REPO / "tools" / "reviewer-eval"

pytestmark = pytest.mark.skipif(
    not _EVAL_ROOT.exists(),
    reason="reviewer-eval harness not present (isolated install)",
)

if _EVAL_ROOT.exists() and str(_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVAL_ROOT))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_reviewer(monkeypatch, review_md="## Overall Assessment\n✅ Looks good\n"):
    """A CodexReviewer with call_codex + worktree stubbed (no codex, no git)."""
    from adapters import codex_reviewer as cr
    from adapters import worktree

    r = cr.CodexReviewer(
        repo_root=str(_REPO), runs_root="/tmp/reviewer-eval-test", prompt_text="BASE PROMPT BODY"
    )

    # Stub the codex call and the worktree materialize/cleanup so review() is
    # exercised end-to-end without spawning codex or touching git.
    monkeypatch.setattr(
        r._mod,
        "call_codex",
        lambda prompt, model, repo_root: (review_md, {"backend": "codex"}),
        raising=True,
    )
    monkeypatch.setattr(
        r,
        "build_prompt_for_case",
        lambda case, worktree_key=None: ("PROMPT", "/tmp/reviewer-eval-test/wt", "deadbeef"),
        raising=True,
    )
    monkeypatch.setattr(worktree, "cleanup", lambda *a, **k: None, raising=True)
    return r


def _case():
    from engine.models import STRATUM_HISTORICAL, Case

    return Case(id="c1", stratum=STRATUM_HISTORICAL)


# --------------------------------------------------------------------------- #
# Bug 1 (was P1): CodexReviewer.review() must return a valid ReviewOutput.
# --------------------------------------------------------------------------- #


def test_codex_reviewer_review_returns_ok(monkeypatch):
    from engine.models import Config, ReviewOutput

    r = _make_reviewer(monkeypatch)
    out = r.review(_case(), Config(id="B", model="gpt-5.5"), 0)
    assert isinstance(out, ReviewOutput)
    assert out.review_markdown.startswith("## Overall Assessment")
    assert out.cli_version  # recorded
    assert out.latency_s >= 0.0


def test_run_matrix_produces_ok_runresult(monkeypatch):
    """A successful review must yield an ok RunResult, not an INFRA_ERROR."""
    from engine.models import Config
    from engine.runner import run_matrix
    from engine.store import RunStore

    r = _make_reviewer(monkeypatch)
    store = RunStore("/tmp/reviewer-eval-test/runs-ok")
    # fresh store each run
    for f in pathlib.Path(store.root).glob("*.json"):
        f.unlink()
    results = run_matrix(
        [_case()],
        [Config(id="B", model="gpt-5.5")],
        r,
        store,
        k=1,
        max_parallel=1,
    )
    assert len(results) == 1
    assert results[0].ok, f"expected ok RunResult, got infra_error={results[0].infra_error}"
    assert results[0].review_markdown


# --------------------------------------------------------------------------- #
# Bug 2 (was P0): experiment identity must not alias across different models
# sharing the same config id.
# --------------------------------------------------------------------------- #


def test_experiment_tag_differs_by_model(monkeypatch):
    from engine.models import Config

    r = _make_reviewer(monkeypatch)
    tag_a = r.experiment_tag(Config(id="B", model="gpt-5.4"))
    tag_b = r.experiment_tag(Config(id="B", model="gpt-5.5"))
    assert tag_a != tag_b, "same config id + different model must yield distinct tags"


def test_run_key_no_alias_across_models(monkeypatch):
    from engine.models import Config
    from engine.store import run_key

    r = _make_reviewer(monkeypatch)
    k4 = run_key("c1", "B", 0, r.experiment_tag(Config(id="B", model="gpt-5.4")))
    k5 = run_key("c1", "B", 0, r.experiment_tag(Config(id="B", model="gpt-5.5")))
    assert k4 != k5, "run files for different models must not collide under one config id"


def test_runresult_carries_run_id_and_prompt_sha(monkeypatch):
    """The run artifact must record its own identity so compare can key on it."""
    from engine.models import Config
    from engine.runner import run_matrix
    from engine.store import RunStore

    r = _make_reviewer(monkeypatch)
    store = RunStore("/tmp/reviewer-eval-test/runs-id")
    for f in pathlib.Path(store.root).glob("*.json"):
        f.unlink()
    results = run_matrix(
        [_case()],
        [Config(id="B", model="gpt-5.5")],
        r,
        store,
        k=1,
        max_parallel=1,
    )
    rr = results[0]
    assert rr.run_id, "RunResult must carry a stable run_id"
    assert rr.prompt_sha, "RunResult must record the prompt_sha it reviewed"


def test_resume_reruns_when_model_changes(monkeypatch):
    """Changing the model under the same config id must NOT resume stale runs."""
    from engine.models import Config
    from engine.runner import run_matrix
    from engine.store import RunStore

    r = _make_reviewer(monkeypatch, review_md="## A\n✅ first\n")
    store = RunStore("/tmp/reviewer-eval-test/runs-resume")
    for f in pathlib.Path(store.root).glob("*.json"):
        f.unlink()

    run_matrix(
        [_case()],
        [Config(id="B", model="gpt-5.4")],
        r,
        store,
        k=1,
        max_parallel=1,
    )
    # Now rerun the SAME config id but a DIFFERENT model. Must not reuse the
    # gpt-5.4 artifact; a new run file must appear.
    r2 = _make_reviewer(monkeypatch, review_md="## B\n✅ second\n")
    run_matrix(
        [_case()],
        [Config(id="B", model="gpt-5.5")],
        r2,
        store,
        k=1,
        max_parallel=1,
    )
    files = sorted(pathlib.Path(store.root).glob("*.json"))
    assert len(files) == 2, f"expected 2 distinct run files (one per model), got {len(files)}"


def _ns(**kw):
    import argparse

    return argparse.Namespace(**kw)


# --------------------------------------------------------------------------- #
# Case-aware run identity (P1 #1): editing a case must invalidate its cache.
# --------------------------------------------------------------------------- #


def test_case_tag_changes_with_case_content():
    from adapters.codex_reviewer import CodexReviewer
    from engine.models import STRATUM_HISTORICAL, Case

    r = CodexReviewer(repo_root=str(_REPO), runs_root="/tmp/reviewer-eval-test", prompt_text="X")
    fx = {"kind": "git_range", "base_sha": "aaa"}
    base = Case(id="c", stratum=STRATUM_HISTORICAL, fixture=dict(fx))
    same = Case(id="c", stratum=STRATUM_HISTORICAL, fixture=dict(fx))
    edited = Case(
        id="c", stratum=STRATUM_HISTORICAL, fixture={"kind": "git_range", "base_sha": "bbb"}
    )
    assert r.case_tag(base) == r.case_tag(same)  # stable; no patch read (git_range)
    assert r.case_tag(base) != r.case_tag(edited)  # base_sha edit -> new tag
    # the machine-local _case_dir must NOT affect the tag
    with_dir = Case(id="c", stratum=STRATUM_HISTORICAL, fixture={**fx, "_case_dir": "/wherever"})
    assert r.case_tag(base) == r.case_tag(with_dir)


def test_case_tag_reads_patch_bytes_and_fails_loud(tmp_path):
    from adapters.codex_reviewer import CodexReviewer
    from engine.models import STRATUM_SYNTHETIC, Case

    r = CodexReviewer(repo_root=str(_REPO), runs_root=str(tmp_path / "runs"), prompt_text="X")
    patch = tmp_path / "inject.diff"
    patch.write_text("AAA")
    fx = {
        "kind": "stored_patch",
        "base_sha": "x",
        "patch": "inject.diff",
        "_case_dir": str(tmp_path),
    }
    t1 = r.case_tag(Case(id="c", stratum=STRATUM_SYNTHETIC, fixture=dict(fx)))
    patch.write_text("BBB")  # editing the patch bytes must change the tag
    assert r.case_tag(Case(id="c", stratum=STRATUM_SYNTHETIC, fixture=dict(fx))) != t1
    patch.unlink()  # a declared-but-missing patch must fail loud, not hash-around it
    with pytest.raises(FileNotFoundError):
        r.case_tag(Case(id="c", stratum=STRATUM_SYNTHETIC, fixture=dict(fx)))


def test_resume_reruns_when_case_changes(monkeypatch):
    from engine.models import STRATUM_HISTORICAL, Case, Config
    from engine.runner import run_matrix
    from engine.store import RunStore

    r = _make_reviewer(monkeypatch)
    store = RunStore("/tmp/reviewer-eval-test/runs-case")
    for f in pathlib.Path(store.root).glob("*.json"):
        f.unlink()
    cfg = [Config(id="A", model="gpt-5.4")]
    run_matrix(
        [Case(id="x", stratum=STRATUM_HISTORICAL, fixture={"base_sha": "aaa"})],
        cfg,
        r,
        store,
        k=1,
        max_parallel=1,
    )
    # Same case id, edited content -> must NOT resume the stale run.
    run_matrix(
        [Case(id="x", stratum=STRATUM_HISTORICAL, fixture={"base_sha": "bbb"})],
        cfg,
        r,
        store,
        k=1,
        max_parallel=1,
    )
    files = sorted(pathlib.Path(store.root).glob("*.json"))
    assert len(files) == 2, f"editing the case must rerun, not resume; got {len(files)}"


# --------------------------------------------------------------------------- #
# compare (P1 #2): the per-run manifest isolates one experiment.
# --------------------------------------------------------------------------- #


def test_compare_honors_manifest(tmp_path, monkeypatch):
    import run_eval
    from engine.models import RunResult
    from engine.store import RunStore, write_json

    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path / "runs"))
    store = RunStore(str(tmp_path / "runs" / "full"))
    cid = "s1-coef-dict-collision"  # a real corpus case so build_bundle renders it
    store.save(
        "keep",
        RunResult(
            case_id=cid,
            config_id="A",
            repeat_idx=0,
            review_markdown="KEEP THIS REVIEW",
            model="gpt-5.5",
            run_id="keep",
        ),
    )
    store.save(
        "drop",
        RunResult(
            case_id=cid,
            config_id="A",
            repeat_idx=0,
            review_markdown="STALE DROP REVIEW",
            model="gpt-5.4",
            run_id="drop",
        ),
    )
    write_json(
        str(tmp_path / "runs" / "full-manifest.json"), {"run_ids": ["keep"], "configs": ["A"]}
    )
    assert run_eval.cmd_compare(_ns(subdir="full")) == 0
    out = (tmp_path / "runs" / "full" / "comparison.md").read_text()
    assert "KEEP THIS REVIEW" in out
    assert "STALE DROP REVIEW" not in out, "manifest must exclude the stale experiment's run"


def test_compare_without_manifest_warns_but_succeeds(tmp_path, monkeypatch, capsys):
    import run_eval
    from engine.models import RunResult
    from engine.store import RunStore

    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path / "runs"))
    store = RunStore(str(tmp_path / "runs" / "full"))
    store.save(
        "only",
        RunResult(
            case_id="s1-coef-dict-collision",
            config_id="A",
            repeat_idx=0,
            review_markdown="SOLO REVIEW",
            model="gpt-5.4",
            run_id="only",
        ),
    )
    # no manifest written -> fall back to all runs, but warn
    assert run_eval.cmd_compare(_ns(subdir="full")) == 0
    assert "no manifest" in capsys.readouterr().err.lower()
    assert "SOLO REVIEW" in (tmp_path / "runs" / "full" / "comparison.md").read_text()


def test_compare_renders_from_run_snapshot_not_corpus(tmp_path, monkeypatch):
    import run_eval
    from engine.models import RunResult
    from engine.store import RunStore, write_json

    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path / "runs"))
    store = RunStore(str(tmp_path / "runs" / "full"))
    # Ground truth whose marker exists ONLY in the artifact's snapshot — and a
    # case_id that is NOT in the corpus — so a corpus reload could not produce it.
    snap = {
        "title": "Snapshot Case",
        "stratum": "s2_historical",
        "ground_truth": [
            {
                "id": "snap:b1",
                "file": "z.py",
                "line_window": [1, 2],
                "bug_class": "x",
                "expected_severity": "P1",
                "rationale": "SNAPSHOT-ONLY-MARKER",
            }
        ],
        "expect_no_blockers": False,
        "allow_severities": ["P2", "P3"],
        "known_fp_topics": [],
    }
    store.save(
        "k",
        RunResult(
            case_id="not-a-corpus-case",
            config_id="A",
            repeat_idx=0,
            review_markdown="rev",
            model="gpt-5.4",
            run_id="k",
            case_snapshot=snap,
        ),
    )
    write_json(str(tmp_path / "runs" / "full-manifest.json"), {"run_ids": ["k"], "configs": ["A"]})
    assert run_eval.cmd_compare(_ns(subdir="full")) == 0
    out = (tmp_path / "runs" / "full" / "comparison.md").read_text()
    # Ground truth comes from the run's snapshot — compare never reads the corpus.
    assert "SNAPSHOT-ONLY-MARKER" in out
    assert "snap:b1" in out
    assert "Snapshot Case" in out


def test_run_rejects_unknown_configs(tmp_path, monkeypatch):
    import run_eval

    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path / "runs"))
    # A typo'd config id must fail closed BEFORE any codex call (no reviewer built),
    # rather than silently running 0/0 and writing an empty manifest.
    rc = run_eval.cmd_run(_ns(configs="Z", strata=None, subdir="full", k=1, max_parallel=1))
    assert rc == 1
    assert not (tmp_path / "runs" / "full-manifest.json").exists()


def test_case_tag_changes_with_scoring_metadata():
    """A metadata-only case edit (ground truth, NOT the fixture) must bust the cache.

    Regression for PR #510 P1: case_tag previously hashed only the fixture+patch, so
    editing ground_truth/severity/negative-control flags left the run key unchanged
    and `compare` graded against a stale snapshot.
    """
    from adapters.codex_reviewer import CodexReviewer
    from engine.models import STRATUM_HISTORICAL, Case, GroundTruthBug

    r = CodexReviewer(repo_root=str(_REPO), runs_root="/tmp/reviewer-eval-test", prompt_text="X")
    fx = {"kind": "git_range", "base_sha": "aaa"}  # identical fixture across all three
    base = Case(
        id="c",
        stratum=STRATUM_HISTORICAL,
        fixture=dict(fx),
        ground_truth=[
            GroundTruthBug(
                id="c:b1", file="f.py", line_window=(1, 5), bug_class="x", expected_severity="P1"
            )
        ],
    )
    sev = Case(
        id="c",
        stratum=STRATUM_HISTORICAL,
        fixture=dict(fx),
        ground_truth=[
            GroundTruthBug(
                id="c:b1", file="f.py", line_window=(1, 5), bug_class="x", expected_severity="P0"
            )
        ],
    )
    neg = Case(
        id="c",
        stratum=STRATUM_HISTORICAL,
        fixture=dict(fx),
        ground_truth=[],
        expect_no_blockers=True,
    )
    assert r.case_tag(base) != r.case_tag(sev), "editing expected_severity must bust the cache"
    assert r.case_tag(base) != r.case_tag(neg), "editing expect_no_blockers must bust the cache"


def test_run_and_smoke_fail_closed_on_empty_corpus(tmp_path, monkeypatch):
    """run/smoke must NOT report success (or write a manifest) on zero selected cases."""
    import run_eval

    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path / "runs"))
    # A stratum that matches no corpus directory -> zero cases (no codex reached).
    rc_run = run_eval.cmd_run(
        _ns(configs="A,B", strata=["no_such_stratum"], subdir="full", k=1, max_parallel=1)
    )
    assert rc_run == 1
    assert not (tmp_path / "runs" / "full-manifest.json").exists(), "no manifest for a no-op run"
    rc_smoke = run_eval.cmd_smoke(
        _ns(configs="A", strata=["no_such_stratum"], k=1, limit=0, max_parallel=1)
    )
    assert rc_smoke == 1


def test_build_prompt_cleans_worktree_on_build_failure(monkeypatch):
    """A prompt-build failure after materialize (e.g. the notebook guard) must not
    leak a detached worktree."""
    from adapters import ci_prompt, worktree
    from adapters import codex_reviewer as cr
    from engine.models import STRATUM_SYNTHETIC, Case

    r = cr.CodexReviewer(repo_root=str(_REPO), runs_root="/tmp/reviewer-eval-test", prompt_text="X")

    class _Mat:
        worktree_dir = "/tmp/reviewer-eval-test/wt-leaktest"
        base_sha = "b"
        head_sha = "h"

    def _raise(**_kw):
        raise NotImplementedError("notebook case unsupported")

    cleaned = []
    monkeypatch.setattr(worktree, "materialize", lambda *a, **k: _Mat(), raising=True)
    monkeypatch.setattr(ci_prompt, "build_ci_prompt", _raise, raising=True)
    monkeypatch.setattr(worktree, "cleanup", lambda wt, root: cleaned.append(wt), raising=True)

    case = Case(id="c", stratum=STRATUM_SYNTHETIC, fixture={"_case_dir": "/x"})
    with pytest.raises(NotImplementedError):
        r.build_prompt_for_case(case, worktree_key="c.A.r0")
    assert cleaned == ["/tmp/reviewer-eval-test/wt-leaktest"], "worktree must be cleaned on failure"
