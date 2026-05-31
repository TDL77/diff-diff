"""
Result container for the classic Synthetic Control Method (SCM) estimator.

This module contains the ``SyntheticControlResults`` dataclass, extracted from
``synthetic_control.py`` to mirror the TROP estimator/results split.

The classic synthetic control of Abadie, Diamond & Hainmueller (2010) produces a
gap path and donor/predictor weights but **no analytical standard error**.
Accordingly ``se``/``t_stat``/``p_value``/``conf_int`` are always NaN on this
object; the point estimate ``att`` (average post-period gap) is the reported
quantity. Significance comes from in-space placebo permutation inference via
:meth:`SyntheticControlResults.in_space_placebo` (a separate ``placebo_p_value``
field, not the NaN ``p_value``).
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff.results import _format_survey_block, _get_significance_stars

__all__ = ["SyntheticControlResults"]


@dataclass
class _SyntheticControlFitSnapshot:
    """Panel state retained for post-hoc in-space placebo refits.

    Holds everything ``SyntheticControlResults.in_space_placebo()`` needs to
    refit ANY donor as the pseudo-treated unit without re-reading the original
    DataFrame. Built in ``SyntheticControl.fit()`` and excluded from pickling by
    ``SyntheticControlResults.__getstate__`` (it retains the full treated+donor
    outcome/predictor panel — a privacy/size hazard if serialized).

    ``specs`` is annotated ``List[Any]`` rather than ``List[_PredictorSpec]`` to
    avoid an import cycle (``_PredictorSpec`` lives in ``synthetic_control.py``,
    which imports this module). ``donor_ids`` is an ORDERED list so the placebo
    iteration order — and therefore the rank / p-value — is deterministic.
    """

    pivots: Dict[str, pd.DataFrame]
    specs: List[Any]
    outcome: str
    all_periods: List[Any]
    pre_periods: List[Any]
    post_periods: List[Any]
    donor_ids: List[Any]
    treated_id: Any
    standardize: str
    v_method: str
    custom_v: Optional[Any]
    n_starts: int
    seed: Optional[int]
    optimizer_options: Optional[Dict[str, Any]]
    inner_max_iter: int
    inner_min_decrease: float


@dataclass
class SyntheticControlResults:
    """
    Results from a classic Synthetic Control Method (SCM) estimation.

    Implements Abadie, Diamond & Hainmueller (2010), "Synthetic Control Methods
    for Comparative Case Studies." A single treated unit's counterfactual is the
    convex combination ``Σ_j w_j · Y_jt`` of donor units chosen to match the
    treated unit's pre-period outcomes and predictors; the treatment effect path
    is the gap ``α̂_1t = Y_1t − Σ_j w_j · Y_jt`` over the post periods.

    Attributes
    ----------
    att : float
        Average post-period gap (the reported point estimate). The per-period
        gaps are in ``gap_path``.
    se : float
        Always NaN — classic SCM has no analytical standard error (inference is
        permutation/placebo based; see Abadie-Diamond-Hainmueller 2010 §2.4).
    t_stat, p_value : float
        Always NaN (no analytical SE).
    conf_int : tuple[float, float]
        Always (NaN, NaN) (no analytical SE).
    n_obs : int
        Number of observations (treated + donor rows over all periods) used.
    n_donors : int
        Number of donor units in the (post-filter) donor pool.
    n_pre_periods : int
        Number of pre-treatment periods.
    n_post_periods : int
        Number of post-treatment periods.
    donor_weights : dict
        Mapping ``{donor_unit_id: weight}`` on the unit simplex. Weights below
        the interpretability floor (1e-6) are dropped.
    v_weights : dict
        Mapping ``{predictor_label: v}`` — the diagonal predictor-importance
        matrix V, trace-normalized to sum to 1.
    predictor_balance : pandas.DataFrame
        Predictor-balance table: for each predictor, the treated value, the
        synthetic value (donor-weighted), and the donor-pool mean.
    gap_path : dict
        Mapping ``{period: gap}`` for ALL periods (pre periods carry the fit
        residual used for ``pre_rmspe``; post periods carry the effect path).
    pre_rmspe : float
        Root mean squared prediction error over the pre-treatment periods (the
        primary fit diagnostic).
    mspe_v : float, optional
        The outer-objective value (pre-period outcome MSPE of ``W*(V*)``),
        populated only when the nested outer search actually runs; None on the
        ``v_method="custom"`` path and on the degenerate single-donor path.
    treated_unit : Any
        The treated unit's identifier.
    pre_periods, post_periods : list
        Calendar-sorted pre / post period values.
    v_method : str
        ``"nested"`` (data-driven V) or ``"custom"`` (user-supplied V).
    standardize : str
        ``"std"`` (per-row SD scaling) or ``"none"``.
    alpha : float
        Significance level recorded for downstream (placebo) inference.
    rmspe_ratio : float
        The treated unit's post/pre RMSPE ratio = ``sqrt(MSPE_post / MSPE_pre)`` —
        the in-space placebo test statistic (ADH 2010 §2.4), computed at fit time.
    placebo_p_value : float
        In-space placebo permutation p-value (``rank / (n_placebos + 1)``), NaN
        until :meth:`in_space_placebo` is run. SEPARATE from the (always-NaN)
        analytical ``p_value``; ``is_significant`` stays bound to ``p_value``.
    n_placebos, n_failed : int
        Donor placebos that entered the permutation reference set / were excluded
        for non-convergence. Both 0 until :meth:`in_space_placebo` is run.
    survey_metadata : Any, optional
        Reserved; always None in this release.

    Significance for classic SCM comes from :meth:`in_space_placebo` (opt-in
    in-space placebo permutation inference); :meth:`get_placebo_df` returns the
    per-unit RMSPE-ratio table used for the rank.
    """

    att: float
    se: float
    t_stat: float
    p_value: float
    conf_int: Tuple[float, float]
    n_obs: int
    n_donors: int
    n_pre_periods: int
    n_post_periods: int
    donor_weights: Dict[Any, float]
    v_weights: Dict[str, float]
    predictor_balance: pd.DataFrame
    gap_path: Dict[Any, float]
    pre_rmspe: float
    treated_unit: Any
    pre_periods: List[Any]
    post_periods: List[Any]
    v_method: str
    standardize: str
    alpha: float = 0.05
    mspe_v: Optional[float] = None
    survey_metadata: Optional[Any] = field(default=None)
    # In-space placebo permutation inference (Abadie-Diamond-Hainmueller 2010
    # Section 2.4), populated by ``in_space_placebo()``. ``rmspe_ratio`` (the
    # treated unit's post/pre RMSPE ratio) is computed at fit time; the rest stay
    # at their no-inference defaults until a placebo run. NOTE: the permutation
    # ``placebo_p_value`` is deliberately SEPARATE from ``p_value`` (which stays
    # NaN) — it is not an analytical p-value, has no SE / t-stat, and does not
    # flow through ``safe_inference``. ``is_significant`` likewise stays bound to
    # the (NaN) ``p_value``, NOT ``placebo_p_value``.
    placebo_p_value: float = np.nan
    rmspe_ratio: float = np.nan
    n_placebos: int = 0
    n_failed: int = 0

    def __post_init__(self) -> None:
        # Internal state set per instance by ``fit()`` / ``in_space_placebo()``.
        # Declared here (not as dataclass fields) so ``dataclasses.fields()`` /
        # ``dataclasses.asdict()`` cannot reach the retained panel state.
        # ``_fit_snapshot`` (full panel) and ``_placebo_gaps`` (per-unit gap paths)
        # are panel-derived and nulled on pickle by ``__getstate__``; ``_placebo_df``
        # holds the small per-unit aggregate table returned by ``get_placebo_df()``.
        self._fit_snapshot: Optional[_SyntheticControlFitSnapshot] = None
        self._placebo_gaps: Optional[Dict[Any, Dict[Any, float]]] = None
        self._placebo_df: Optional[pd.DataFrame] = None
        # Whether the treated unit's own inner Frank-Wolfe weight solve converged.
        # in_space_placebo() fails closed when this is False: a truncated treated
        # fit makes the ranked statistic (rmspe_ratio) not a valid SCM optimum.
        self._fit_converged: bool = True

    def __getstate__(self) -> Dict[str, Any]:
        """Exclude panel-derived internal state from pickling.

        ``_fit_snapshot`` retains the full treated+donor panel and ``_placebo_gaps``
        the per-unit gap paths — both panel-derived, a privacy/size hazard if the
        pickle is sent elsewhere. The scalar placebo fields (``placebo_p_value``,
        ``rmspe_ratio``, ``n_placebos``, ``n_failed``) and the small ``_placebo_df``
        aggregate table survive. An unpickled result keeps all public fields; a
        diagnostic call that needs the snapshot (``in_space_placebo``) then raises a
        ValueError directing the user to re-fit. Mirrors ``SyntheticDiDResults``.
        """
        state = self.__dict__.copy()
        state["_fit_snapshot"] = None
        state["_placebo_gaps"] = None
        return state

    def __repr__(self) -> str:
        """Concise string representation."""
        return (
            f"SyntheticControlResults(ATT={self.att:.4f}, "
            f"pre_RMSPE={self.pre_rmspe:.4f}, "
            f"n_donors={self.n_donors}, "
            f"v_method={self.v_method!r})"
        )

    @property
    def coef_var(self) -> float:
        """Coefficient of variation: SE / abs(ATT). NaN here (SE is always NaN)."""
        if not (np.isfinite(self.se) and self.se >= 0):
            return np.nan
        if not np.isfinite(self.att) or self.att == 0:
            return np.nan
        return self.se / abs(self.att)

    @property
    def is_significant(self) -> bool:
        """Always False — classic SCM produces no analytical p-value."""
        return bool(np.isfinite(self.p_value) and self.p_value < self.alpha)

    @property
    def significance_stars(self) -> str:
        """Significance stars based on p-value (empty here — p_value is NaN)."""
        return _get_significance_stars(self.p_value)

    def summary(self, alpha: Optional[float] = None) -> str:
        """
        Generate a formatted summary of the estimation results.

        Parameters
        ----------
        alpha : float, optional
            Significance level; defaults to the alpha used during estimation.

        Returns
        -------
        str
            Formatted summary table.
        """
        alpha = alpha or self.alpha

        n_top = min(5, len(self.donor_weights))
        top_donors = sorted(self.donor_weights.items(), key=lambda kv: kv[1], reverse=True)[:n_top]

        lines = [
            "=" * 75,
            "Synthetic Control Method (SCM) Estimation Results".center(75),
            "Abadie, Diamond & Hainmueller (2010)".center(75),
            "=" * 75,
            "",
            f"{'Observations:':<28} {self.n_obs:>10}",
            f"{'Donor units:':<28} {self.n_donors:>10}",
            f"{'Pre-treatment periods:':<28} {self.n_pre_periods:>10}",
            f"{'Post-treatment periods:':<28} {self.n_post_periods:>10}",
            f"{'Treated unit:':<28} {str(self.treated_unit):>10}",
            "",
            "-" * 75,
            "Fit Diagnostics".center(75),
            "-" * 75,
            f"{'Pre-treatment RMSPE:':<28} {self.pre_rmspe:>10.4f}",
            f"{'V selection:':<28} {self.v_method:>10}",
            f"{'Standardization:':<28} {self.standardize:>10}",
        ]
        if self.mspe_v is not None and np.isfinite(self.mspe_v):
            lines.append(f"{'Outer-objective MSPE:':<28} {self.mspe_v:>10.6f}")

        if self.survey_metadata is not None:
            lines.extend(_format_survey_block(self.survey_metadata, 75))

        lines.extend(
            [
                "",
                "-" * 75,
                f"{'Top donor weights (w_j)':<40}",
                "-" * 75,
            ]
        )
        for unit_id, w in top_donors:
            lines.append(f"{'  ' + str(unit_id):<40} {w:>10.4f}")

        lines.extend(
            [
                "",
                "-" * 75,
                f"{'Parameter':<15} {'Estimate':>12} {'Std. Err.':>12} "
                f"{'t-stat':>10} {'P>|t|':>10}",
                "-" * 75,
                f"{'ATT (avg gap)':<15} {self.att:>12.4f} {'n/a':>12} " f"{'n/a':>10} {'n/a':>10}",
                "-" * 75,
                "",
            ]
        )
        # Three states: (1) placebo never run -> point to in_space_placebo();
        # (2) run with a valid reference set -> show the permutation p-value;
        # (3) run but infeasible (no placebo entered the rank, e.g. J<2 or all
        # donors failed) -> say so explicitly rather than implying it was not run.
        # ``_placebo_df is not None`` is the "attempted" signal (survives pickling).
        placebo_attempted = self._placebo_df is not None
        if placebo_attempted and np.isfinite(self.placebo_p_value):
            # The classic analytical fields above stay n/a (no SE); this is the
            # permutation p-value of the post/pre RMSPE ratio, p = rank/(n_placebos+1).
            lines.extend(
                [
                    "In-space placebo permutation inference "
                    "(Abadie-Diamond-Hainmueller 2010, Section 2.4):",
                    f"{'  RMSPE ratio (post/pre):':<34} {self.rmspe_ratio:>10.4f}",
                    f"{'  Permutation p-value:':<34} {self.placebo_p_value:>10.4f}",
                    f"{'  Placebos in reference set:':<34} {self.n_placebos:>10d}"
                    + (f"  ({self.n_failed} excluded)" if self.n_failed else ""),
                    "",
                    "(Analytical SE is still undefined for classic SCM; the "
                    "p-value above is permutation-based.)",
                    "=" * 75,
                ]
            )
        elif placebo_attempted:
            lines.extend(
                [
                    "In-space placebo permutation inference was attempted but "
                    "produced no valid reference set",
                    f"(0 placebos entered the rank; {self.n_failed} failed to "
                    "converge). placebo_p_value is undefined — too few donors or "
                    "all donor refits failed. Inspect get_placebo_df().",
                    "=" * 75,
                ]
            )
        else:
            lines.extend(
                [
                    "Inference: classic SCM has no analytical standard error.",
                    "Run in_space_placebo() for in-space permutation inference",
                    "(Abadie-Diamond-Hainmueller 2010, Section 2.4).",
                    "=" * 75,
                ]
            )

        return "\n".join(lines)

    def print_summary(self, alpha: Optional[float] = None) -> None:
        """Print the summary to stdout."""
        print(self.summary(alpha))

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert scalar results to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary of the scalar estimation results (weights/balance/gaps
            are available via the ``get_*_df`` accessors).
        """
        result = {
            "att": self.att,
            "se": self.se,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "conf_int_lower": self.conf_int[0],
            "conf_int_upper": self.conf_int[1],
            "n_obs": self.n_obs,
            "n_donors": self.n_donors,
            "n_pre_periods": self.n_pre_periods,
            "n_post_periods": self.n_post_periods,
            "pre_rmspe": self.pre_rmspe,
            "mspe_v": self.mspe_v,
            "treated_unit": self.treated_unit,
            "v_method": self.v_method,
            "standardize": self.standardize,
            # In-space placebo permutation inference. rmspe_ratio is set at fit;
            # placebo_p_value / n_placebos / n_failed stay at their no-inference
            # defaults (NaN / 0) until in_space_placebo() runs.
            "rmspe_ratio": self.rmspe_ratio,
            "placebo_p_value": self.placebo_p_value,
            "n_placebos": self.n_placebos,
            "n_failed": self.n_failed,
        }
        if self.survey_metadata is not None:
            sm = self.survey_metadata
            result["weight_type"] = sm.weight_type
            result["effective_n"] = sm.effective_n
            result["design_effect"] = sm.design_effect
        return result

    def to_dataframe(self) -> pd.DataFrame:
        """Convert scalar results to a single-row pandas DataFrame."""
        return pd.DataFrame([self.to_dict()])

    def get_gap_df(self) -> pd.DataFrame:
        """
        Get the gap (effect) path as a DataFrame, in calendar order.

        Rebuilt period-keyed from ``gap_path`` using the canonical
        ``pre_periods + post_periods`` order so the row order is independent of
        any dict-insertion order. Columns: ``period``, ``gap``, ``phase``.

        Returns
        -------
        pandas.DataFrame
        """
        rows = []
        for period in list(self.pre_periods) + list(self.post_periods):
            if period in self.gap_path:
                phase = "post" if period in self.post_periods else "pre"
                rows.append({"period": period, "gap": self.gap_path[period], "phase": phase})
        return pd.DataFrame(rows, columns=["period", "gap", "phase"])

    def get_weights_df(self) -> pd.DataFrame:
        """
        Get donor weights as a DataFrame, sorted by weight descending.

        Returns
        -------
        pandas.DataFrame
            Columns: ``unit``, ``weight``.
        """
        items = sorted(self.donor_weights.items(), key=lambda kv: kv[1], reverse=True)
        return pd.DataFrame(
            [{"unit": unit, "weight": w} for unit, w in items],
            columns=["unit", "weight"],
        )

    _PLACEBO_COLS = ["unit", "pre_mspe", "post_mspe", "rmspe_ratio", "is_treated", "status"]

    def get_placebo_df(self) -> pd.DataFrame:
        """
        Get the in-space placebo distribution as a DataFrame (one row per unit).

        This is a per-unit SUMMARY table (one row per unit), enough to reproduce
        the permutation rank and a ratio-distribution plot — NOT the per-period
        placebo gap paths needed for the classic "spaghetti" plot (those are
        retained internally on ``_placebo_gaps`` for the successful placebos).
        Columns: ``unit``, ``pre_mspe``, ``post_mspe``, ``rmspe_ratio``,
        ``is_treated``, ``status`` (``"treated"`` / ``"placebo"`` / ``"failed"``).
        The treated unit is always present as a single ``is_treated=True,
        status="treated"`` row (its ratio is the original J-donor fit). After a
        placebo run **that produced a reference set** (``>= 2`` donors AND a
        converged treated fit), the table has ``n_donors + 1`` rows — every donor
        appears, including those whose refit did not converge (``status="failed"``
        with NaN metrics, excluded from the rank). In the degenerate / fail-closed
        cases (fewer than 2 donors, or a treated fit that did not converge) the
        placebo loop does not run, so only the treated row is returned.

        Populated by :meth:`in_space_placebo`; the summary table is retained on
        pickling, so it is still returned after a round-trip. Before any placebo
        run — including on an unpickled result that never ran one — only the
        treated row is returned.

        Returns
        -------
        pandas.DataFrame
        """
        if self._placebo_df is not None:
            return self._placebo_df.copy()
        from diff_diff.synthetic_control import _mspe

        pre = _mspe(self.gap_path, self.pre_periods)
        post = _mspe(self.gap_path, self.post_periods)
        return pd.DataFrame(
            [
                {
                    "unit": self.treated_unit,
                    "pre_mspe": pre,
                    "post_mspe": post,
                    "rmspe_ratio": self.rmspe_ratio,
                    "is_treated": True,
                    "status": "treated",
                }
            ],
            columns=self._PLACEBO_COLS,
        )

    def in_space_placebo(
        self,
        n_starts: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        In-space placebo permutation inference (Abadie-Diamond-Hainmueller 2010,
        Section 2.4).

        Reassigns the treatment to each donor in turn, re-estimates a synthetic
        control for that pseudo-treated donor against the OTHER donors, and ranks
        the real treated unit's post/pre RMSPE ratio among all units. Populates
        ``placebo_p_value``, ``n_placebos`` and ``n_failed`` on this object
        (``rmspe_ratio`` — the treated unit's own ratio — is set at fit time) and
        returns the placebo distribution via :meth:`get_placebo_df`.

        The real treated unit is **excluded from every placebo's donor pool**: its
        post-period outcome is treatment-contaminated, so allowing a placebo to
        load weight on it would bias the placebo gap. The ranking set is therefore
        the ``J+1`` units ``{treated} ∪ {J placebos}``, with each placebo fit
        against the other ``J-1`` donors (this matches the standard
        ``SCtools::generate.placebos`` construction). The post/pre RMSPE ratio
        normalizes by pre-treatment fit, which obviates the pre-fit-cutoff
        filtering of ADH Figures 5-7 (journal p. 502), so no pre-fit filter is
        offered — every converged placebo enters the rank.

        The permutation ``placebo_p_value`` is intentionally distinct from
        ``p_value`` (which stays NaN — classic SCM has no analytical SE) and from
        ``is_significant`` (which also stays bound to the NaN ``p_value``).

        A placebo is **excluded** from the reference set (counted in ``n_failed``)
        when its fit is not a valid optimum — EITHER its inner Frank-Wolfe weight
        solve did not converge (a truncated ``W`` is unusable) OR its outer ``V``
        search did not converge (an under-optimized ``V`` fits the pre-period worse,
        shrinking its RMSPE ratio and biasing the permutation p-value
        anti-conservatively). Each placebo refit **inherits the original fit's
        ``optimizer_options`` / ``n_starts``**, so valid inference requires settings
        adequate for the outer ``V`` search to converge: production defaults do;
        with cheap settings, raise ``n_starts`` here or re-fit with a larger
        ``optimizer_options['maxiter']`` (otherwise placebos are dropped as failed).
        The treated unit's own fit is held to the same standard — if its inner OR
        outer search did not converge, the whole run fails closed (see below).

        Parameters
        ----------
        n_starts : int, optional
            Override the multistart count for each placebo's nested V search.
            Default None inherits the original fit's ``n_starts``. The placebo
            loop is the cost driver (one outer V search per donor); lower it for a
            faster, coarser scan.

        Returns
        -------
        pandas.DataFrame
            The placebo distribution (see :meth:`get_placebo_df`).

        Raises
        ------
        ValueError
            If the fit snapshot is unavailable (e.g. this result was unpickled).
        """
        if self._fit_snapshot is None:
            raise ValueError(
                "in_space_placebo() requires the fit snapshot on the results "
                "object. This result appears to have been loaded from "
                "serialization (which excludes the snapshot) or produced by an "
                "older estimator version. Re-fit to enable in-space placebo "
                "inference."
            )
        from diff_diff.synthetic_control import _mspe, _placebo_fit_unit

        snap = self._fit_snapshot
        donors = list(snap.donor_ids)
        n_donors = len(donors)
        n_starts_eff = snap.n_starts if n_starts is None else int(n_starts)

        treated_pre = _mspe(self.gap_path, snap.pre_periods)
        treated_post = _mspe(self.gap_path, snap.post_periods)
        treated_ratio = self.rmspe_ratio

        rows: List[Dict[str, Any]] = [
            {
                "unit": snap.treated_id,
                "pre_mspe": treated_pre,
                "post_mspe": treated_post,
                "rmspe_ratio": treated_ratio,
                "is_treated": True,
                "status": "treated",
            }
        ]

        # Fail closed when the treated unit's OWN fit did not converge at fit time
        # (inner Frank-Wolfe weight solve OR outer V search): ranking a statistic
        # from a truncated / under-optimized treated fit would not be a valid ADH
        # 2010 §2.4 permutation (placebos already fail-closed on non-convergence, so
        # the treated unit must too). ``_fit_converged`` folds both failure modes, so
        # the remediation names the knobs for each.
        if not self._fit_converged:
            warnings.warn(
                "In-space placebo skipped: the treated unit's own SCM fit did not "
                "converge at fit time (inner Frank-Wolfe weight solve and/or outer V "
                "search), so its RMSPE ratio is not a valid optimum to rank against "
                "placebos. placebo_p_value is NaN — re-fit with a larger "
                "inner_max_iter / looser inner_min_decrease (inner) and/or a larger "
                "optimizer_options['maxiter'] / more n_starts (outer V search).",
                UserWarning,
                stacklevel=2,
            )
            self.placebo_p_value = np.nan
            self.n_placebos = 0
            self.n_failed = 0
            self._placebo_gaps = {}
            self._placebo_df = pd.DataFrame(rows, columns=self._PLACEBO_COLS)
            return self._placebo_df.copy()

        if n_donors < 2:
            warnings.warn(
                "In-space placebo inference requires at least 2 donors (each "
                f"placebo is fit against the other donors); only {n_donors} "
                "available. placebo_p_value is NaN.",
                UserWarning,
                stacklevel=2,
            )
            self.placebo_p_value = np.nan
            self.n_placebos = 0
            self.n_failed = 0
            self._placebo_gaps = {}
            self._placebo_df = pd.DataFrame(rows, columns=self._PLACEBO_COLS)
            return self._placebo_df.copy()

        if n_donors == 2:
            warnings.warn(
                "In-space placebo with 2 donors: each placebo is fit against a "
                "single donor (degenerate weight w=[1]) with no V search, so the "
                "permutation p-value is coarse (only 2 placebos enter the "
                "reference set; the smallest attainable p-value is 1/3).",
                UserWarning,
                stacklevel=2,
            )

        placebo_gaps: Dict[Any, Dict[Any, float]] = {}
        ranked_ratios: List[float] = []
        n_failed = 0

        for j in donors:
            pool = [d for d in donors if d != j]
            fitted = _placebo_fit_unit(snap, j, pool, n_starts_eff)
            if fitted is None:
                # Non-converged inner Frank-Wolfe weight solve (a truncated W is
                # unusable for ranking): exclude from BOTH the numerator and the
                # denominator (never penalize a truncated solve into the rank).
                # Still record the donor with NaN metrics so get_placebo_df()
                # returns the full treated + every-donor unit set.
                n_failed += 1
                rows.append(
                    {
                        "unit": j,
                        "pre_mspe": np.nan,
                        "post_mspe": np.nan,
                        "rmspe_ratio": np.nan,
                        "is_treated": False,
                        "status": "failed",
                    }
                )
                continue
            gap_path_j, ratio_j = fitted
            placebo_gaps[j] = gap_path_j
            pre_j = _mspe(gap_path_j, snap.pre_periods)
            post_j = _mspe(gap_path_j, snap.post_periods)
            ranked_ratios.append(ratio_j)
            rows.append(
                {
                    "unit": j,
                    "pre_mspe": pre_j,
                    "post_mspe": post_j,
                    "rmspe_ratio": ratio_j,
                    "is_treated": False,
                    "status": "placebo",
                }
            )

        n_placebos = len(ranked_ratios)
        if n_placebos == 0:
            warnings.warn(
                "No in-space placebo entered the reference set (all donors "
                f"failed to converge or were filtered out of {n_donors}); "
                "placebo_p_value is NaN.",
                UserWarning,
                stacklevel=2,
            )
            p_value = np.nan
        else:
            # One-sided rank, treated unit included as the "+1". Ties counted via
            # ``>=`` so the p-value is conservative.
            rank = 1 + sum(1 for r in ranked_ratios if r >= treated_ratio)
            p_value = rank / (n_placebos + 1)

        if n_failed > 0:
            warnings.warn(
                f"{n_failed} of {n_donors} in-space placebos failed to converge "
                "and were excluded from the permutation distribution; "
                f"placebo_p_value uses the remaining {n_placebos}.",
                UserWarning,
                stacklevel=2,
            )

        self.placebo_p_value = float(p_value)
        self.n_placebos = int(n_placebos)
        self.n_failed = int(n_failed)
        self._placebo_gaps = placebo_gaps
        self._placebo_df = pd.DataFrame(rows, columns=self._PLACEBO_COLS)
        return self._placebo_df.copy()
