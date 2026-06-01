"""Generic glue for the minimal Codex-reviewer A/B comparison harness.

Plain data (``engine.models``), a resumable run store (``engine.store``), a
run-matrix executor (``engine.runner``), and the side-by-side bundle builder
(``engine.compare``). The diff-diff-specific bindings (corpus, codex
invocation, worktrees) live in ``adapters/``. See
``tools/reviewer-eval/README.md`` for the flow.
"""
