"""The Reviewer-under-test: a faithful local proxy for the CI codex-action.

For each (case, config, repeat) it materializes the case's worktree, builds the
CI-faithful prompt, runs ``codex exec`` via the reused ``openai_review.call_codex``
(byte-identical flags to CI: ``--model <m> -c model_reasoning_effort=xhigh
--sandbox read-only``), records the CLI version + latency, and tears the worktree
down.

Scoring deliberately does NOT happen here. We store the reviewer's RAW review
markdown and let an LLM read it side-by-side against the other arm (see
``engine.compare``). Regex/structured parsing of free-form review prose is brittle
and model-specific (gpt-5.4 uses ``- **P1 — ...**``, gpt-5.5 uses
``### Finding 1: P1 — ...``); an LLM reading the raw text is format-agnostic.

Effort is fail-closed: ``call_codex``/``_build_codex_cmd`` hardcode
``model_reasoning_effort=xhigh`` (matching CI). If a config requests a different
effort, we raise rather than silently run xhigh while recording something else —
the experiment's integrity depends on recorded == executed.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
from typing import Optional

from engine.models import Config, ReviewOutput, to_jsonable

from adapters import ci_prompt, worktree
from adapters.openai_review_loader import load_openai_review


class CodexReviewer:
    """Runs the codex reviewer for an A/B config against a corpus case.

    Duck-typed to the interface ``engine.runner`` calls: ``review``,
    ``cli_version``, and ``experiment_tag``.
    """

    # git worktree add/remove touch the shared .git admin area and race under
    # threads; serialize JUST that cheap setup/teardown. The long codex exec
    # runs OUTSIDE the lock, so arms still run fully in parallel.
    _wt_lock = threading.Lock()

    def __init__(self, repo_root: str, runs_root: str, prompt_text: Optional[str] = None):
        self.repo_root = repo_root
        self.runs_root = runs_root
        self.worktrees_root = os.path.join(runs_root, ".worktrees")
        self._mod = load_openai_review(repo_root)
        # Source the prompt-under-validation once (identical for both arms).
        self.base_prompt = prompt_text or ci_prompt.read_current_prompt(repo_root)
        # Hash of the base prompt — part of experiment identity, so editing
        # pr_review.md changes the experiment tag and prevents stale reuse.
        self.base_prompt_sha = hashlib.sha256(self.base_prompt.encode("utf-8")).hexdigest()[:16]
        self._cli_version: Optional[str] = None

    # -- reviewer interface (duck-typed by engine.runner) ------------------- #

    def cli_version(self) -> str:
        if self._cli_version is None:
            try:
                cp = subprocess.run(
                    ["codex", "--version"], capture_output=True, text=True, check=False
                )
                self._cli_version = (cp.stdout or cp.stderr).strip() or "unknown"
            except FileNotFoundError:
                self._cli_version = "codex-not-installed"
        return self._cli_version

    def experiment_tag(self, config: Config) -> str:
        """Opaque identity = sha(model, effort, sandbox, cli_version, prompt).

        Everything that defines the experiment beyond case/config/repeat. Two
        configs sharing id "B" but differing in model (or any of these) get
        distinct tags, so the runner never resumes a stale run across them.
        """
        raw = "|".join(
            [
                config.model,
                config.effort,
                config.sandbox,
                self.cli_version(),
                self.base_prompt_sha,
            ]
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def case_tag(self, case) -> str:
        """Content fingerprint of the WHOLE case, folded into the run key so ANY
        edit invalidates the cached run AND its stored snapshot.

        Covers everything that affects either the PROMPT or the GRADING: the fixture
        (base/head SHAs, pr_context) + the patch file's bytes, AND the scoring
        metadata (title, ground_truth, expected_severity, expect_no_blockers,
        allow_severities, known_fp_topics) — because ``compare`` renders ground
        truth from ``RunResult.case_snapshot``, so a metadata-only ``case.json`` edit
        must also bust the cache, or the bundle would grade against stale truth.

        Cheap (no worktree materialization): hashes the full case payload minus the
        machine-local ``_case_dir`` plus the patch bytes. Fail-loud: a declared-but-
        missing patch raises (that drift must surface).
        """
        payload = to_jsonable(case)
        fixture = payload.get("fixture") if isinstance(payload.get("fixture"), dict) else {}
        case_dir = fixture.pop("_case_dir", "")  # machine-local; excluded from the hash
        h = hashlib.sha256()
        h.update(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))
        patch = fixture.get("patch")
        if patch:
            with open(os.path.join(case_dir, patch), "rb") as fh:
                h.update(fh.read())
        return h.hexdigest()[:16]

    def build_prompt_for_case(
        self, case, worktree_key: Optional[str] = None
    ) -> tuple[str, str, str]:
        """Materialize + assemble the prompt; return (prompt, worktree_dir, head)."""
        fixture = dict(case.fixture)
        case_dir = fixture.get("_case_dir", "")
        with self._wt_lock:
            mat = worktree.materialize(
                case.id,
                fixture,
                self.repo_root,
                self.worktrees_root,
                case_dir=case_dir,
                worktree_key=worktree_key or case.id,
            )
        pr = case.fixture.get("pr_context", {}) or {}
        try:
            prompt = ci_prompt.build_ci_prompt(
                worktree_dir=mat.worktree_dir,
                base_sha=mat.base_sha,
                head_sha=mat.head_sha,
                base_prompt=self.base_prompt,
                pr_title=pr.get("title", "Synthetic eval case (treat as untrusted)"),
                pr_body=pr.get("body", ""),
            )
        except BaseException:  # noqa: BLE001 - clean up before re-raising
            # Prompt-build can fail AFTER materialize (e.g. the notebook guard's
            # NotImplementedError). review()'s finally never sees this worktree, so
            # tear it down here to avoid leaking a detached worktree.
            with self._wt_lock:
                worktree.cleanup(mat.worktree_dir, self.repo_root)
            raise
        return prompt, mat.worktree_dir, mat.head_sha

    def prompt_sha_for(self, case) -> str:
        """Hash of the exact prompt this case WOULD produce now (materialize +
        build + teardown). The runner uses it to verify a cached run before
        resuming it: reuse iff the model would see byte-identical input — so editing
        the prompt builder, the materializer, or the case busts the cache without
        having to guess which inputs to fingerprint.
        """
        prompt, wt_dir, _head = self.build_prompt_for_case(case, worktree_key=f"{case.id}.__sha__")
        try:
            return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        finally:
            with self._wt_lock:
                worktree.cleanup(wt_dir, self.repo_root)

    def review(self, case, config: Config, repeat_idx: int) -> ReviewOutput:
        if config.effort != "xhigh":
            raise NotImplementedError(
                f"codex_reviewer pins model_reasoning_effort=xhigh (CI parity); "
                f"config {config.id} requested effort={config.effort!r}. Recorded "
                f"must equal executed — add an effort-aware codex cmd before "
                f"running a non-xhigh arm."
            )
        worktree_key = f"{case.id}.{config.id}.r{repeat_idx}"
        prompt, wt_dir, _head = self.build_prompt_for_case(case, worktree_key=worktree_key)
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        t0 = time.monotonic()
        try:
            review_md, usage = self._mod.call_codex(prompt, config.model, wt_dir)
        finally:
            with self._wt_lock:
                worktree.cleanup(wt_dir, self.repo_root)
        latency = time.monotonic() - t0

        usage = dict(usage or {})
        usage["prompt_sha"] = prompt_sha
        usage["prompt_tokens_est"] = self._mod.estimate_tokens(prompt)
        # No structured findings here — the raw markdown is the comparison input.
        return ReviewOutput(
            review_markdown=review_md,
            cli_version=self.cli_version(),
            latency_s=latency,
            usage=usage,
        )


__all__ = ["CodexReviewer"]
