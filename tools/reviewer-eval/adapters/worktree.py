"""Materialize a case's diff state in a throwaway detached git worktree.

Never touches the user's primary checkout or current branch. Worktrees live
under ``runs/.worktrees/`` (gitignored). Supports three fixture kinds:

  * ``git_range``     — checkout pinned ``head_sha``; diff is ``base_sha..head_sha``
                        (S2/S3/S4: real historical PR states).
  * ``stored_patch``  — checkout ``base_sha``; ``git apply`` a frozen ``inject.diff``;
                        commit locally; diff is ``base_sha..HEAD`` (S1, preferred —
                        survives HEAD drift, unlike a live revert).
  * ``git_revert``    — checkout ``base_sha``; ``git revert --no-commit`` a fix
                        commit; commit locally (S1 alternative; brittle on drift).

A materialization failure raises ``MaterializeError``; the runner turns that
into an INFRA_ERROR RunResult — NEVER a missed bug (plan: infra noise must not
trip a recall floor).
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Optional


class MaterializeError(RuntimeError):
    """A case could not be faithfully materialized."""


@dataclass
class Materialized:
    worktree_dir: str
    base_sha: str
    head_sha: str


def _git(repo: str, args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo, check=check, capture_output=True, text=True)


def _resolve(repo: str, rev: str) -> str:
    cp = _git(repo, ["rev-parse", "--verify", f"{rev}^{{commit}}"], check=False)
    if cp.returncode != 0:
        raise MaterializeError(f"cannot resolve revision {rev!r}: {cp.stderr.strip()}")
    return cp.stdout.strip()


def _ensure_present(repo: str, sha: str) -> None:
    if _git(repo, ["cat-file", "-e", f"{sha}^{{commit}}"], check=False).returncode != 0:
        raise MaterializeError(
            f"commit {sha} not present in repo {repo}; fetch it before materializing."
        )


def materialize(
    case_id: str,
    fixture: dict,
    repo_root: str,
    worktrees_root: str,
    case_dir: Optional[str] = None,
    worktree_key: Optional[str] = None,
) -> Materialized:
    """Create the worktree for ``case_id`` and return its location + SHAs.

    ``case_dir`` is the directory holding the case's ``inject.diff`` (for
    stored_patch). ``worktrees_root`` is created if absent.

    ``worktree_key`` names the worktree directory; it MUST be unique per
    concurrently-running invocation. Parallel A/B runs of the same case share a
    ``case_id`` but need distinct worktrees (else one arm's setup removes the
    other's checkout mid-review). Callers pass e.g. ``"<case>.<config>.r<rep>"``;
    it defaults to ``case_id`` for sequential callers (verify-corpus).
    """
    kind = fixture.get("kind")
    base = fixture.get("base_sha")
    if not base:
        raise MaterializeError(f"{case_id}: fixture missing base_sha")
    _ensure_present(repo_root, base)
    base = _resolve(repo_root, base)

    os.makedirs(worktrees_root, exist_ok=True)
    wt = os.path.join(worktrees_root, (worktree_key or case_id).replace("/", "_"))
    # Clean any stale worktree from a previous crashed run.
    if os.path.exists(wt):
        cleanup(wt, repo_root)

    if kind == "git_range":
        head = fixture.get("head_sha")
        if not head:
            raise MaterializeError(f"{case_id}: git_range fixture missing head_sha")
        _ensure_present(repo_root, head)
        head = _resolve(repo_root, head)
        cp = _git(repo_root, ["worktree", "add", "--detach", wt, head], check=False)
        if cp.returncode != 0:
            raise MaterializeError(f"{case_id}: worktree add failed: {cp.stderr.strip()}")
        return Materialized(worktree_dir=wt, base_sha=base, head_sha=head)

    if kind in ("stored_patch", "git_revert"):
        cp = _git(repo_root, ["worktree", "add", "--detach", wt, base], check=False)
        if cp.returncode != 0:
            raise MaterializeError(f"{case_id}: worktree add failed: {cp.stderr.strip()}")
        try:
            if kind == "stored_patch":
                patch_rel = fixture.get("patch")
                if not patch_rel:
                    raise MaterializeError(f"{case_id}: stored_patch missing 'patch'")
                patch_path = os.path.join(case_dir or "", patch_rel)
                if not os.path.exists(patch_path):
                    raise MaterializeError(f"{case_id}: patch not found at {patch_path}")
                ap = _git(wt, ["apply", "--whitespace=nowarn", patch_path], check=False)
                if ap.returncode != 0:
                    raise MaterializeError(
                        f"{case_id}: git apply failed (HEAD likely drifted from the "
                        f"frozen patch): {ap.stderr.strip()}"
                    )
            else:  # git_revert
                rc = fixture.get("revert_commit")
                if not rc:
                    raise MaterializeError(f"{case_id}: git_revert missing revert_commit")
                _ensure_present(repo_root, rc)
                rv = _git(wt, ["revert", "--no-commit", rc], check=False)
                if rv.returncode != 0:
                    raise MaterializeError(
                        f"{case_id}: git revert failed (conflict on drift): " f"{rv.stderr.strip()}"
                    )
            _git(wt, ["add", "-A"])
            msg = fixture.get("commit_message", f"eval: inject bug for {case_id}")
            # Identity is provided inline so the eval never depends on global git config.
            commit = _git(
                wt,
                [
                    "-c",
                    "user.name=codex-eval",
                    "-c",
                    "user.email=codex-eval@local",
                    "commit",
                    "--no-verify",
                    "-m",
                    msg,
                ],
                check=False,
            )
            if commit.returncode != 0:
                raise MaterializeError(
                    f"{case_id}: commit failed (empty patch?): {commit.stderr.strip()}"
                )
            head = _resolve(wt, "HEAD")
            return Materialized(worktree_dir=wt, base_sha=base, head_sha=head)
        except MaterializeError:
            cleanup(wt, repo_root)
            raise

    raise MaterializeError(f"{case_id}: unknown fixture kind {kind!r}")


def cleanup(worktree_dir: str, repo_root: str) -> None:
    """Remove a worktree (best-effort; prune dangling admin files)."""
    _git(repo_root, ["worktree", "remove", "--force", worktree_dir], check=False)
    _git(repo_root, ["worktree", "prune"], check=False)
    # Belt-and-suspenders: if the dir somehow survived, leave it (a later run
    # cleans it) rather than rm -rf'ing an unexpected path.


__all__ = ["materialize", "cleanup", "Materialized", "MaterializeError"]
