"""Build the side-by-side A/B comparison bundle for an LLM (or human) to grade.

This replaces the old programmatic grader + metrics + gates engine. Instead of
parsing free-form review prose with regex, we emit ONE markdown document: per
case, the ground-truth bugs followed by each arm's RAW review output, plus
grading instructions that point at the real `pr_review.md` rubric. An LLM reads
it top-to-bottom and fills the caught/missed/false-positive table. Stdlib-only.
"""

from __future__ import annotations

import re
from typing import Iterable

from engine.models import RunResult

# Order strata for a stable, sensible read order; unknown strata sort last.
_STRATUM_ORDER = {
    "s1_synthetic": 0,
    "s2_historical": 1,
    "s3_negative": 2,
    "s4_missed": 3,
}

_HEADER = """# Codex reviewer A/B comparison

Two configs reviewed each case below: **A** (control) and **B** (candidate).
Each case shows its ground-truth bugs (what a faithful review SHOULD surface)
followed by every config's RAW review output.

## How to grade
Read each case's ground truth, then both raw reviews, and fill this table — one
row per (case, ground-truth bug), plus a row for any false positive on a
negative-control case:

| case | bug id | expected sev | A caught? | A sev | B caught? | B sev | notes |
|------|--------|--------------|-----------|-------|-----------|-------|-------|

Rules:
- Severity definitions (P0–P3) and what counts as a "real issue" (P0/P1) live in
  `.github/codex/prompts/pr_review.md` — the exact rubric both models were given.
- A finding "catches" a bug if it names the same defect at the same
  location/symbol, regardless of wording or output format.
- On a **negative-control** case (marked "NO known bugs" below), any P0/P1
  finding is a FALSE POSITIVE; P2/P3 and the listed known-FP topics are allowed.
- Note any bug that BOTH arms miss (a shared blind spot) and any bug only one arm
  catches (the signal that matters for the upgrade decision).
"""


def _fence_for(text: str) -> str:
    """A backtick fence longer than the longest backtick run in ``text``.

    Review markdown routinely contains ``` code fences; wrapping it in a fence of
    (max-run + 1) backticks keeps the embedded fences from terminating ours.
    """
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def _render_ground_truth(snap: dict) -> str:
    """Render a case's ground truth from its run-time snapshot (a plain dict)."""
    if snap.get("expect_no_blockers"):
        lines = [
            "**NO known bugs** (negative control) — any P0/P1 finding is a FALSE " "POSITIVE.",
            f"- Allowed severities: {', '.join(snap.get('allow_severities') or []) or 'none'}",
        ]
        for topic in snap.get("known_fp_topics") or []:
            desc = topic.get("topic") or topic.get("description") or str(topic)
            lines.append(f"- Known-FP topic (do NOT flag): {desc}")
        return "\n".join(lines)

    gt = snap.get("ground_truth") or []
    if not gt:
        return "_(no ground-truth bugs recorded)_"

    out = ["**Ground-truth bugs** (a faithful review should surface these):"]
    for bug in gt:
        lw = list(bug.get("line_window") or [0, 0]) + [0, 0]
        lo, hi = lw[0], lw[1]
        loc = f"`{bug.get('file', '?')}`"
        if lo or hi:
            loc += f" lines {lo}–{hi}"
        if bug.get("anchor_symbol"):
            loc += f" (symbol `{bug['anchor_symbol']}`)"
        out.append(
            f"- **{bug.get('id', '?')}** — expected **{bug.get('expected_severity', '?')}**, "
            f"class `{bug.get('bug_class', '?')}` — {loc}"
        )
        if bug.get("class_keywords"):
            out.append(f"  - keywords: {', '.join(bug['class_keywords'])}")
        if bug.get("rationale"):
            out.append(f"  - why: {bug['rationale']}")
    return "\n".join(out)


def _render_review(rr: RunResult) -> str:
    label = f"### {rr.config_id} ({rr.model or 'model?'}) — review"
    if rr.repeat_idx:
        label += f" (repeat {rr.repeat_idx})"
    if not rr.ok:
        return f"{label}\n\n> INFRA_ERROR — excluded: `{rr.infra_error}`"
    md = rr.review_markdown.strip() or "_(empty review)_"
    meta = f"latency {rr.latency_s:.0f}s · cli {rr.cli_version or '?'}"
    fence = _fence_for(md)
    return f"{label}\n\n_{meta}_\n\n{fence}markdown\n{md}\n{fence}"


def build_bundle(runs: Iterable[RunResult]) -> str:
    """Render the full comparison bundle from the stored run artifacts.

    Each run carries its own ``case_snapshot`` (the case AS REVIEWED), so the
    bundle reflects exactly what ran — never the current corpus. Only cases that
    actually have runs are rendered (so a subset run shows only its cases), and a
    later edit to the corpus can't change how an old run is presented. Cases are
    grouped by ``case_id`` and ordered by stratum.
    """
    runs_by_case: dict[str, list[RunResult]] = {}
    for rr in runs:
        runs_by_case.setdefault(rr.case_id, []).append(rr)

    def _sort_key(case_id: str):
        snap = runs_by_case[case_id][0].case_snapshot or {}
        return (_STRATUM_ORDER.get(snap.get("stratum", ""), 99), case_id)

    case_ids = sorted(runs_by_case, key=_sort_key)

    parts = [_HEADER, f"\n_{len(case_ids)} cases._\n"]
    for case_id in case_ids:
        case_runs = sorted(runs_by_case[case_id], key=lambda r: (r.config_id, r.repeat_idx))
        snap = case_runs[0].case_snapshot or {}
        parts.append("\n---\n")
        title = f" — {snap['title']}" if snap.get("title") else ""
        parts.append(f"## {case_id}{title}\n")
        parts.append(f"_stratum: {snap.get('stratum', '?')}_\n")
        parts.append(_render_ground_truth(snap))
        parts.append("")
        for rr in case_runs:
            parts.append(_render_review(rr))
            parts.append("")

    return "\n".join(parts).rstrip() + "\n"


__all__ = ["build_bundle"]
