#!/usr/bin/env python3
"""CLI for the minimal Codex-reviewer A/B comparison harness.

Pipeline:

  verify-corpus   Materialize every case and assert its diff touches the
                  expected files (no codex; cheap; run in CI-lite).
  smoke           Tiny matrix (default: control arm, k=1) end-to-end to
                  validate plumbing -- the FIRST command that calls codex.
  run             Full A/B matrix; saves each arm's RAW review markdown.
  compare         Emit one side-by-side bundle (ground truth + both arms' raw
                  reviews) for an LLM (or you) to read into a caught/missed/
                  false-positive table.

Usage:
  python tools/reviewer-eval/run_eval.py verify-corpus
  python tools/reviewer-eval/run_eval.py smoke --configs A
  python tools/reviewer-eval/run_eval.py run --configs A,B
  python tools/reviewer-eval/run_eval.py compare --subdir full
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# Make `engine` and `adapters` importable (the eval dir is the package root).
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from adapters.corpus_loader import CorpusLoader  # noqa: E402
from adapters.openai_review_loader import find_repo_root  # noqa: E402
from engine.store import RunStore, read_json, write_json  # noqa: E402

CONFIG_DIR = os.path.join(HERE, "config")
CORPUS_DIR = os.path.join(HERE, "corpus")
RUNS_DIR = os.path.join(HERE, "runs")


def _manifest_path(subdir: str) -> str:
    """Per-invocation run manifest — a SIBLING of runs/<subdir>/ so the RunStore's
    ``*.json`` glob in ``load_all`` never parses it as a RunResult."""
    return os.path.join(RUNS_DIR, f"{subdir}-manifest.json")


def _load_configs() -> dict:
    return read_json(os.path.join(CONFIG_DIR, "configs.json"))  # type: ignore[return-value]


def _make_configs(which: list) -> list:
    from engine.models import Config

    raw = _load_configs()
    by_id = {}
    for key in ("control", "candidate"):
        c = raw[key]
        by_id[c["id"]] = Config(
            id=c["id"],
            model=c["model"],
            effort=c.get("effort", "xhigh"),
            sandbox=c.get("sandbox", "read-only"),
            action_version=c.get("action_version", "v1"),
            cli_version=c.get("cli_version"),
            label=c.get("label", ""),
        )
    return [by_id[i] for i in which if i in by_id]


def _resolve_configs(arg: str):
    """Parse --configs, returning the Config list or None if any id is unknown.

    Fail-closed: a typo like ``--configs Z`` (or ``A,Z``) returns None so the
    caller can abort, rather than silently running a partial/empty matrix.
    """
    requested = [c for c in (arg or "").split(",") if c]
    configs = _make_configs(requested)
    if not configs or len(configs) != len(requested):
        return None
    return configs


def _bad_configs_msg(arg: str) -> str:
    return (
        f"invalid --configs {arg!r}: unknown or empty config id(s) "
        "(valid ids come from config/configs.json, e.g. A,B)"
    )


def cmd_verify_corpus(args: argparse.Namespace) -> int:
    repo_root = find_repo_root()
    loader = CorpusLoader(CORPUS_DIR, repo_root)
    cases = loader.load_cases(args.strata)
    if not cases:
        print("no cases found", file=sys.stderr)
        return 1
    failures = 0
    for case in cases:
        err = loader.verify(case)
        status = "OK" if err is None else f"FAIL: {err}"
        if err is not None:
            failures += 1
        print(f"[{status}] {case.id} ({case.stratum})")
    print(f"\n{len(cases) - failures}/{len(cases)} cases verified.")
    return 1 if failures else 0


def _build_reviewer(repo_root: str):
    from adapters.codex_reviewer import CodexReviewer

    return CodexReviewer(repo_root=repo_root, runs_root=RUNS_DIR)


def cmd_smoke(args: argparse.Namespace) -> int:
    from engine.runner import run_matrix

    repo_root = find_repo_root()
    loader = CorpusLoader(CORPUS_DIR, repo_root)
    cases = loader.load_cases(args.strata)
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print(
            f"no cases selected (strata={args.strata}, limit={args.limit}); nothing to run",
            file=sys.stderr,
        )
        return 1
    configs = _resolve_configs(args.configs)
    if configs is None:
        print(_bad_configs_msg(args.configs), file=sys.stderr)
        return 1
    reviewer = _build_reviewer(repo_root)
    print(f"codex CLI version: {reviewer.cli_version()}")
    store = RunStore(os.path.join(RUNS_DIR, "smoke"))
    results = run_matrix(
        cases,
        configs,
        reviewer,
        store,
        k=args.k,
        max_parallel=args.max_parallel,
        progress=lambda m: print(f"  {m}"),
        assert_cli_equal=len(configs) > 1,
    )
    ok = sum(1 for r in results if r.ok)
    print(f"\nsmoke: {ok}/{len(results)} runs ok")
    for r in results:
        tag = "OK" if r.ok else f"INFRA_ERROR ({r.infra_error})"
        print(f"  {r.case_id} {r.config_id} r{r.repeat_idx}: {r.latency_s:.0f}s [{tag}]")
    return 0 if ok == len(results) else 2


def cmd_run(args: argparse.Namespace) -> int:
    from engine.runner import run_matrix

    repo_root = find_repo_root()
    loader = CorpusLoader(CORPUS_DIR, repo_root)
    cases = loader.load_cases(args.strata)
    if not cases:
        print(f"no cases selected (strata={args.strata}); nothing to run", file=sys.stderr)
        return 1
    configs = _resolve_configs(args.configs)
    if configs is None:
        print(_bad_configs_msg(args.configs), file=sys.stderr)
        return 1
    reviewer = _build_reviewer(repo_root)
    store = RunStore(os.path.join(RUNS_DIR, args.subdir))
    results = run_matrix(
        cases,
        configs,
        reviewer,
        store,
        k=args.k,
        max_parallel=args.max_parallel,
        progress=lambda m: print(f"  {m}"),
    )
    ok = sum(1 for r in results if r.ok)
    # Manifest scopes `compare` to THIS invocation's runs (resumed + infra-errored
    # included — they carry run_ids), so a later rerun into the same subdir can't
    # silently mix experiments into one bundle. Empty run_ids are dropped (M1).
    run_ids = [r.run_id for r in results if r.run_id]
    if len(run_ids) != len(results):
        print(
            f"  warning: {len(results) - len(run_ids)} run(s) had no run_id; "
            "omitted from the manifest",
            file=sys.stderr,
        )
    write_json(
        _manifest_path(args.subdir),
        {"run_ids": run_ids, "configs": [c.id for c in configs]},
    )
    print(f"\n{ok}/{len(results)} runs ok. Next: compare --subdir {args.subdir}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Emit the side-by-side bundle for an LLM (or human) to grade.

    Renders ONLY from the stored run artifacts under runs/<subdir> (each carries
    its own as-reviewed case snapshot), scoped to the last `run` invocation via its
    manifest. Writes runs/<subdir>/comparison.md. No corpus reload, no scoring.
    """
    from engine.compare import build_bundle

    runs = RunStore(os.path.join(RUNS_DIR, args.subdir)).load_all()
    if not runs:
        print(f"no runs found under runs/{args.subdir}", file=sys.stderr)
        return 1

    # Scope to the last `run` invocation via its manifest, so a subdir that
    # accumulated runs from multiple experiments yields ONE clean A/B bundle.
    manifest_path = _manifest_path(args.subdir)
    try:
        manifest = read_json(manifest_path)
        wanted = set(manifest.get("run_ids", []))  # type: ignore[union-attr]
    except (OSError, ValueError):
        wanted = None
    if wanted is None:
        print(
            f"  WARNING: no manifest at {os.path.basename(manifest_path)}; comparing "
            f"ALL runs under runs/{args.subdir} (may mix experiments). Re-run `run` "
            "to regenerate it.",
            file=sys.stderr,
        )
    else:
        no_id = [r for r in runs if not r.run_id]
        if no_id:
            print(
                f"  warning: {len(no_id)} run(s) have no run_id; excluded from the "
                "manifest-scoped bundle",
                file=sys.stderr,
            )
        runs = [r for r in runs if r.run_id in wanted]
        if not runs:
            print(
                f"manifest matched no runs under runs/{args.subdir}; re-run `run`.",
                file=sys.stderr,
            )
            return 1
    bundle = build_bundle(runs)
    out_dir = os.path.join(RUNS_DIR, args.subdir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "comparison.md")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(bundle)
    ok = sum(1 for r in runs if r.ok)
    n_cases = len({r.case_id for r in runs})
    print(f"wrote {out_path} ({ok}/{len(runs)} runs ok across {n_cases} cases)")
    print(
        "Next: have an LLM (a subagent or in-conversation) read it into the "
        "caught/missed/false-positive table, then decide."
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Codex reviewer A/B comparison harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser("verify-corpus", help="materialize + check every case")
    pv.add_argument("--strata", nargs="*", default=None)
    pv.set_defaults(func=cmd_verify_corpus)

    ps = sub.add_parser("smoke", help="tiny end-to-end run (first codex call)")
    ps.add_argument("--configs", default="A")
    ps.add_argument("--strata", nargs="*", default=None)
    ps.add_argument("--k", type=int, default=1)
    ps.add_argument("--limit", type=int, default=0)
    ps.add_argument("--max-parallel", type=int, default=2)
    ps.set_defaults(func=cmd_smoke)

    pr = sub.add_parser("run", help="full A/B matrix; saves raw reviews")
    pr.add_argument("--configs", default="A,B")
    pr.add_argument("--strata", nargs="*", default=None)
    pr.add_argument("--subdir", default="full")
    pr.add_argument("--k", type=int, default=1, help="repeats per (case, config)")
    pr.add_argument("--max-parallel", type=int, default=5)
    pr.set_defaults(func=cmd_run)

    pc = sub.add_parser("compare", help="emit the side-by-side bundle to grade")
    pc.add_argument("--subdir", default="full")
    pc.set_defaults(func=cmd_compare)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
