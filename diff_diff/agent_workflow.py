"""Stateless orchestrator: print the recommended diff-diff workflow with
the caller's column names wired in.

This module exists to give LLM agents a single, recognizable entrypoint
that names the rest of the agent-facing workflow (`profile_panel`,
`get_llm_guide`, `practitioner_next_steps`, `BusinessReport`). The
function does not fit, inspect, or recommend — it templates a copy-
pasteable script.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Pattern → candidate estimator names. Flat union below is the
# `fit_candidates` field of the returned dict; each name must remain a
# valid `hasattr(diff_diff, name)` (locked by the contract test in
# tests/test_agent_discoverability.py and tests/test_agent_workflow.py).
_WORKFLOW_PATTERNS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "Staggered adoption + binary treatment + has_never_treated control",
        ("CallawaySantAnna", "SunAbraham", "ImputationDiD"),
    ),
    (
        "Continuous treatment dose (non-binary numeric intensity)",
        ("ContinuousDiD",),
    ),
    (
        "Heterogeneous adoption intensity across treated units",
        ("HeterogeneousAdoptionDiD",),
    ),
    (
        "Simple 2x2 DiD (binary treatment, two periods, no staggering)",
        ("DifferenceInDifferences",),
    ),
    (
        "Parallel trends in doubt — diagnose before fitting",
        ("PreTrendsPower", "HonestDiD"),
    ),
)


def _format_kwargs(**kwargs: Optional[str]) -> str:
    parts = [f'{k}="{v}"' for k, v in kwargs.items() if v is not None]
    return ", ".join(parts)


def agent_workflow(
    df: Any,
    *,
    unit: str,
    time: str,
    treatment: str,
    outcome: str,
    first_treat: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Print the recommended diff-diff workflow with your column names wired in.

    Stateless orchestrator. Calls nothing internally. Returns a dict;
    optionally prints a copy-pasteable script (``verbose=True``, the
    default). ``df`` is not inspected — column names are templated
    verbatim into the output.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format panel data. Not inspected; included so the agent
        can pass the same handle along to the next call.
    unit : str
        Column identifying the cross-sectional unit.
    time : str
        Column identifying the time period.
    treatment : str
        Column holding the treatment indicator or dose.
    outcome : str
        Column holding the outcome variable.
    first_treat : str, optional
        Column with each unit's first-treatment period (or NaN for
        never-treated controls). When supplied, the templated fit
        snippet adds ``first_treat="<colname>"`` to the call so the
        agent doesn't have to remember which staggered estimators
        accept it.
    verbose : bool, default True
        If True, print the script to stdout. The dict is always
        returned regardless.

    Returns
    -------
    dict
        Keys:

        - ``"profile_call"`` (str): call signature for
          :func:`diff_diff.profile_panel`.
        - ``"guide_call"`` (str): call signature for
          :func:`diff_diff.get_llm_guide`.
        - ``"fit_candidates"`` (list of str): flat union of estimator /
          diagnostic class names referenced in the workflow patterns.
          Every name resolves on the top-level ``diff_diff`` namespace.
        - ``"validation_calls"`` (list of str): call signatures for the
          post-fit validation step.
        - ``"reporting_call"`` (str): call signature for
          :class:`diff_diff.BusinessReport`.
        - ``"script"`` (str): printable multi-line workflow.

    Examples
    --------
    >>> import pandas as pd
    >>> import diff_diff
    >>> df = pd.DataFrame({
    ...     "firm_id": [1, 1, 2, 2],
    ...     "year": [0, 1, 0, 1],
    ...     "treated": [0, 0, 1, 1],
    ...     "logwage": [0.1, 0.2, 0.1, 0.9],
    ... })
    >>> out = diff_diff.agent_workflow(df, unit="firm_id", time="year",
    ...                                treatment="treated", outcome="logwage",
    ...                                verbose=False)
    >>> "profile_panel" in out["script"]
    True
    """
    del df  # intentionally unused: orchestrator templates from column names only
    profile_kwargs = _format_kwargs(unit=unit, time=time, treatment=treatment, outcome=outcome)
    profile_call = f"diff_diff.profile_panel(df, {profile_kwargs})"
    guide_call = 'diff_diff.get_llm_guide("autonomous")'

    fit_kwargs = _format_kwargs(
        unit=unit,
        time=time,
        treatment=treatment,
        outcome=outcome,
        first_treat=first_treat,
    )
    fit_example_call = f"diff_diff.CallawaySantAnna(...).fit(df, {fit_kwargs})"

    validation_calls = [
        "diff_diff.practitioner_next_steps(result)",
    ]
    reporting_call = "diff_diff.BusinessReport(result).full_report()"

    fit_candidates: List[str] = []
    pattern_lines: List[str] = []
    for label, names in _WORKFLOW_PATTERNS:
        pattern_lines.append(f"    - {label}")
        pattern_lines.append(f"        candidates: {', '.join(names)}")
        for n in names:
            if n not in fit_candidates:
                fit_candidates.append(n)

    pattern_block = "\n".join(pattern_lines)

    script = f"""diff_diff workflow for your data
=================================

Step 1 - Describe the panel:
    profile = {profile_call}
    print(profile)

Step 2 - Choose an estimator. Consult the routing matrix:
    print({guide_call})

    Routing patterns to look up in the matrix:
{pattern_block}

Step 3 - Fit (CallawaySantAnna shown; substitute the matching candidate):
    result = {fit_example_call}

Step 4 - Validate:
    {validation_calls[0]}

Step 5 - Report:
    {reporting_call}

Full reference: diff_diff.get_llm_guide("full")
Practitioner recipe: diff_diff.get_llm_guide("practitioner")
"""

    if verbose:
        print(script)

    return {
        "profile_call": profile_call,
        "guide_call": guide_call,
        "fit_candidates": fit_candidates,
        "validation_calls": validation_calls,
        "reporting_call": reporting_call,
        "script": script,
    }
