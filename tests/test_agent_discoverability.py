"""Contract test for the agent-discoverability surface (issue #461).

This is a static snapshot test of the four contract surfaces that PR
#464 introduced for LLM-agent discovery:

1. ``__all__`` membership of agent-facing primitives
2. ``dir(diff_diff)`` head-first ordering (via the ``_OrderedName`` trick)
3. Top-level ``__doc__`` content (first paragraph names the recommended
   call; the 5-step workflow primitives all appear)
4. ``agent_workflow()`` output references the canonical downstream
   primitives by name

It also locks the ``__dir__()`` invariants (head matches
``_AGENT_FACING_ORDER``, tail is alphabetic by ``str``, module dunders
are preserved, ``inspect.getmembers`` parity).

Closes the ``__dir__`` contract-test deferral row from PR #464's
``TODO.md``.

No live API calls, no subprocess, no live agents — purely string/identity
assertions runnable in the default ``pytest`` suite.
"""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

import diff_diff
from diff_diff import _AGENT_FACING_ORDER

# ---------------------------------------------------------------------------
# __all__ membership
# ---------------------------------------------------------------------------


def test_agent_facing_names_in_all():
    """The named primitives must remain in the public API surface.

    Catches an export pruning that would silently remove an agent-facing
    name from ``from diff_diff import *``.
    """
    required = {
        "agent_workflow",
        "profile_panel",
        "get_llm_guide",
        "practitioner_next_steps",
        "BusinessReport",
    }
    assert required <= set(
        diff_diff.__all__
    ), f"missing from __all__: {required - set(diff_diff.__all__)}"


def test_estimator_class_names_importable():
    """Class-name renames silently break agent recognition.

    The canonical staggered estimators + the simple-2x2 case must remain
    importable under their documented names; the orchestrator's Step 3
    examples and ``llms-autonomous.txt`` routing matrix reference them
    by these literal identifiers.
    """
    from diff_diff import (  # noqa: F401
        CallawaySantAnna,
        ChaisemartinDHaultfoeuille,
        ContinuousDiD,
        DifferenceInDifferences,
        HeterogeneousAdoptionDiD,
        HonestDiD,
        ImputationDiD,
        PreTrendsPower,
        SunAbraham,
        TwoWayFixedEffects,
        WooldridgeDiD,
    )


# ---------------------------------------------------------------------------
# __dir__() head-first ordering + _OrderedName invariants
# ---------------------------------------------------------------------------


def test_dir_head_matches_agent_facing_order():
    """``dir(diff_diff)`` must surface ``_AGENT_FACING_ORDER`` at the
    head, IN THE DECLARED ORDER.

    Anchors to the contract (the override's curated tuple) rather than
    a fixed slice length: if a future change adds or trims the head
    tuple, this test follows it. Catches the failure mode where
    ``__dir__()`` is dropped, mis-ordered, or where the
    ``_OrderedName`` ``__lt__`` is broken.
    """
    names = dir(diff_diff)
    head_size = len(_AGENT_FACING_ORDER)
    assert names[:head_size] == list(_AGENT_FACING_ORDER), (
        f"dir() head does not match _AGENT_FACING_ORDER. "
        f"Got: {names[:head_size]!r}. "
        f"Expected: {list(_AGENT_FACING_ORDER)!r}."
    )


def test_dir_tail_alphabetic_by_str():
    """The non-head portion of ``dir()`` should stay alphabetic when
    keyed by ``str``.

    The ``_OrderedName`` head members compare with custom ``__lt__``
    (priority then alphabetic); tail elements are plain strings sorted
    by CPython's ``PyList_Sort``. ``sorted(tail, key=str)`` is the
    canonical recovery key in case any downstream tooling re-sorts.
    """
    names = dir(diff_diff)
    tail = names[len(_AGENT_FACING_ORDER) :]
    assert tail == sorted(tail, key=str)


def test_dir_returns_full_module_namespace():
    """``dir(diff_diff)`` must enumerate the full module namespace.

    Restricting to ``__all__`` would drop module dunders (``__doc__``,
    ``__name__``, ``__file__``) and break ``inspect.getmembers``
    consumers. The override returns ``[_OrderedName(n) for n in
    globals()]`` to preserve that compatibility.
    """
    names = dir(diff_diff)
    for dunder in ("__doc__", "__name__", "__file__", "__all__"):
        assert dunder in names, f"{dunder!r} missing from dir() output"


def test_getmembers_parity_with_default_module_dir():
    """``inspect.getmembers(diff_diff)`` should return the same set of
    names as ``dir(diff_diff)``, with ``__doc__`` accessible.

    Catches regressions where ``__dir__`` is reduced to ``__all__`` only.
    """
    dir_names = set(dir(diff_diff))
    gm_names = {name for name, _ in inspect.getmembers(diff_diff)}
    assert dir_names == gm_names, (
        f"dir() and inspect.getmembers() disagree by " f"{sorted(dir_names ^ gm_names)[:5]}"
    )
    # And the steering surface must be accessible.
    assert diff_diff.__doc__ is not None
    assert "agent_workflow" in diff_diff.__doc__.lower()


# ---------------------------------------------------------------------------
# _OrderedName subclass invariants
# ---------------------------------------------------------------------------


def test_ordered_name_isinstance_str():
    """Every ``dir()`` element must still be ``isinstance(..., str)`` so
    consumers that type-check don't break.
    """
    for name in dir(diff_diff):
        assert isinstance(
            name, str
        ), f"dir() element {name!r} is type {type(name).__name__}, not a str subclass"


def test_ordered_name_str_methods_work():
    """The head ``_OrderedName`` instances must support all the str
    operations downstream tooling relies on (upper, eq, hash for dict
    keys, ``in`` membership, f-string interpolation).
    """
    head = dir(diff_diff)[: len(_AGENT_FACING_ORDER)]
    for n in head:
        assert n.upper() == str(n).upper()
        assert n == str(n)
        assert {n: 1}.get(n) == 1
        assert n in [str(n)]
        assert f"{n}" == str(n)


# ---------------------------------------------------------------------------
# __doc__ first-paragraph contract
# ---------------------------------------------------------------------------


def test_doc_first_paragraph_names_agent_workflow():
    """``help(diff_diff)`` opens with ``__doc__``; the first non-blank
    paragraph must name ``agent_workflow``.

    Catches a docstring rewrite that drops the recommended-call hint
    from the top-of-help surface.
    """
    doc = diff_diff.__doc__
    assert doc is not None
    first_block = doc.strip().split("\n\n")[0]
    assert "agent_workflow" in first_block.lower()


def test_doc_names_canonical_workflow_helpers():
    """The full 5-step workflow's primitive names must remain reachable
    from ``help(diff_diff)``.

    Catches a docstring trim that removes references to the downstream
    helpers an agent following the doc would call next.
    """
    assert diff_diff.__doc__ is not None
    doc_lower = diff_diff.__doc__.lower()
    for name in (
        "profile_panel",
        "get_llm_guide",
        "practitioner_next_steps",
        "businessreport",
    ):
        assert name in doc_lower, f"{name!r} missing from __doc__"


# ---------------------------------------------------------------------------
# agent_workflow() output references the canonical primitives
# ---------------------------------------------------------------------------


def test_agent_workflow_output_names_canonical_helpers():
    """Calling ``agent_workflow()`` must still produce a script that
    names the four downstream primitives. Catches the orchestrator
    content drifting away from the helpers it advertises.
    """
    df = pd.DataFrame({"u": [1], "t": [0], "tr": [0], "y": [0.0]})
    out = diff_diff.agent_workflow(
        df,
        unit="u",
        time="t",
        treatment="tr",
        outcome="y",
        verbose=False,
    )
    for name in (
        "profile_panel",
        "get_llm_guide",
        "practitioner_next_steps",
        "BusinessReport",
    ):
        assert name in out["script"], f"{name!r} missing from agent_workflow script"


def test_agent_workflow_fit_candidates_resolve_on_diff_diff():
    """Every estimator advertised in ``agent_workflow().fit_candidates``
    must be a real attribute on the ``diff_diff`` namespace.

    Mirrors the per-PR test in ``test_agent_workflow.py``; here we
    re-assert as part of the discoverability contract so a rename
    that escapes the per-PR suite is still caught at the surface
    level.
    """
    df = pd.DataFrame({"u": [1], "t": [0], "tr": [0], "y": [0.0]})
    out = diff_diff.agent_workflow(
        df,
        unit="u",
        time="t",
        treatment="tr",
        outcome="y",
        verbose=False,
    )
    missing = [n for n in out["fit_candidates"] if not hasattr(diff_diff, n)]
    assert not missing, f"fit_candidates not on diff_diff namespace: {missing}"


# ---------------------------------------------------------------------------
# Cross-surface sanity (all four agent-facing entrypoints callable)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    sorted(
        {
            "agent_workflow",
            "profile_panel",
            "get_llm_guide",
            "practitioner_next_steps",
            "BusinessReport",
        }
    ),
)
def test_agent_facing_entrypoint_callable(name):
    """Each agent-facing primitive must remain a callable attribute on
    the top-level package.

    Catches an accidental replacement of one of these names with a
    module or constant (which would silently break the agent's
    ``help(name)`` follow-up).
    """
    obj = getattr(diff_diff, name)
    assert callable(obj), f"{name!r} is not callable on the diff_diff namespace"
