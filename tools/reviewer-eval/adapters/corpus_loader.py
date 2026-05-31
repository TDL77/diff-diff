"""Load the JSON corpus into engine ``Case`` objects.

Resolves each ground-truth bug's ``bug_class`` to keyword lists via
``corpus/bug_class_synonyms.json`` so the rendered comparison bundle carries the
right vocabulary. Threads each case's on-disk directory into
``fixture["_case_dir"]`` so the worktree adapter can find ``inject.diff``.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from engine.models import Case, GroundTruthBug

from adapters import worktree


def _load_synonyms(corpus_dir: str) -> dict[str, list[str]]:
    path = os.path.join(corpus_dir, "bug_class_synonyms.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _bug_from_dict(d: dict, synonyms: dict[str, list[str]]) -> GroundTruthBug:
    lw = d.get("line_window", [0, 0])
    bug_class = d.get("bug_class", "")
    keywords = list(d.get("class_keywords", [])) or synonyms.get(bug_class, [])
    return GroundTruthBug(
        id=d["id"],
        file=d["file"],
        line_window=(int(lw[0]), int(lw[1])),
        bug_class=bug_class,
        expected_severity=d.get("expected_severity", "P1"),
        must_catch=d.get("must_catch", True),
        anchor_symbol=d.get("anchor_symbol", ""),
        class_keywords=keywords,
        rationale=d.get("rationale", ""),
        provenance=d.get("provenance", {}),
    )


def _case_from_dict(d: dict, case_dir: str, synonyms: dict[str, list[str]]) -> Case:
    fixture = dict(d.get("fixture", {}))
    fixture["_case_dir"] = case_dir
    # Carry pr_context through the fixture so the reviewer can read it.
    if "pr_context" in d:
        fixture["pr_context"] = d["pr_context"]
    return Case(
        id=d["id"],
        stratum=d["stratum"],
        title=d.get("title", ""),
        fixture=fixture,
        ground_truth=[_bug_from_dict(b, synonyms) for b in d.get("ground_truth", [])],
        expect_no_blockers=d.get("expect_no_blockers", False),
        allow_severities=d.get("allow_severities", ["P2", "P3"]),
        known_fp_topics=d.get("known_fp_topics", []),
        weight=float(d.get("weight", 1.0)),
        notes=d.get("notes", ""),
    )


class CorpusLoader:
    def __init__(self, corpus_dir: str, repo_root: str):
        self.corpus_dir = corpus_dir
        self.repo_root = repo_root
        self.cases_dir = os.path.join(corpus_dir, "cases")
        self._synonyms = _load_synonyms(corpus_dir)

    def load_cases(self, strata: Optional[list[str]] = None) -> list[Case]:
        cases: list[Case] = []
        if not os.path.isdir(self.cases_dir):
            return cases
        for stratum in sorted(os.listdir(self.cases_dir)):
            if strata and stratum not in strata:
                continue
            stratum_dir = os.path.join(self.cases_dir, stratum)
            if not os.path.isdir(stratum_dir):
                continue
            for case_id in sorted(os.listdir(stratum_dir)):
                case_dir = os.path.join(stratum_dir, case_id)
                case_json = os.path.join(case_dir, "case.json")
                if not os.path.exists(case_json):
                    continue
                with open(case_json, encoding="utf-8") as fh:
                    d = json.load(fh)
                cases.append(_case_from_dict(d, case_dir, self._synonyms))
        return cases

    def verify(self, case: Case) -> Optional[str]:
        """Materialize the case and assert the diff touches expected files.

        Returns an error string, or None if OK. Worktree is always cleaned up.
        """
        runs_root = os.path.join(self.corpus_dir, "..", "runs")
        worktrees_root = os.path.join(os.path.abspath(runs_root), ".worktrees")
        fixture = dict(case.fixture)
        case_dir = fixture.get("_case_dir", "")
        try:
            mat = worktree.materialize(
                case.id, fixture, self.repo_root, worktrees_root, case_dir=case_dir
            )
        except worktree.MaterializeError as exc:
            return f"materialize failed: {exc}"
        try:
            from adapters.ci_prompt import git_name_status, touches_notebook

            name_status = git_name_status(mat.worktree_dir, mat.base_sha, mat.head_sha)
            if not name_status.strip():
                return "empty diff (base==head or patch was a no-op)"
            if touches_notebook(name_status):
                return (
                    "notebook case unsupported: ci_prompt omits the CI workflow's "
                    "<notebook-prose> block; port that extraction before adding "
                    "a notebook-touching case."
                )
            touched = {
                line.split("\t", 1)[1].strip() for line in name_status.splitlines() if "\t" in line
            }
            expected = {b.file for b in case.ground_truth}
            missing = {
                f for f in expected if not any(t.endswith(f) or f.endswith(t) for t in touched)
            }
            if expected and missing:
                return (
                    f"diff does not touch expected file(s) {sorted(missing)}; "
                    f"touched {sorted(touched)}"
                )
            return None
        finally:
            worktree.cleanup(mat.worktree_dir, self.repo_root)


__all__ = ["CorpusLoader"]
