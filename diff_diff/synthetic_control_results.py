"""
Result container for the classic Synthetic Control Method (SCM) estimator.

This module contains the ``SyntheticControlResults`` dataclass, extracted from
``synthetic_control.py`` to mirror the TROP estimator/results split.

The classic synthetic control of Abadie, Diamond & Hainmueller (2010) produces a
gap path and donor/predictor weights but **no analytical standard error** — the
paper proposes permutation/placebo inference instead (a later PR). Accordingly
``se``/``t_stat``/``p_value``/``conf_int`` are always NaN on this object; the
point estimate ``att`` (average post-period gap) is the reported quantity.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff.results import _format_survey_block, _get_significance_stars

__all__ = ["SyntheticControlResults"]


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
    survey_metadata : Any, optional
        Reserved; always None in this release.
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

    # Reserved for PR-2 (placebo inference) / PR-3 (conformal). These are plain
    # (un-annotated) class attributes, NOT dataclass fields, so dataclasses.fields()
    # and dataclasses.asdict() cannot reach them; fit() sets them per instance.
    _placebo_gaps = None
    _rmspe_ratio = None
    _fit_snapshot = None

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
                "Inference: classic SCM has no analytical standard error.",
                "Use permutation / placebo inference for significance testing",
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
