"""Results class for WooldridgeDiD (ETWFE) estimator."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff.utils import safe_inference


@dataclass
class WooldridgeDiDResults:
    """Results from WooldridgeDiD.fit().

    Core output is ``group_time_effects``: a dict keyed by (cohort_g, time_t)
    with per-cell ATT estimates and inference.  Call ``.aggregate(type)`` to
    compute any of the four jwdid_estat aggregation types.
    """

    # ------------------------------------------------------------------ #
    # Core cohort×time estimates                                          #
    # ------------------------------------------------------------------ #
    group_time_effects: Dict[Tuple[Any, Any], Dict[str, Any]]
    """key=(g,t), value={att, se, t_stat, p_value, conf_int}"""

    # ------------------------------------------------------------------ #
    # Simple (overall) aggregation — always populated at fit time         #
    # ------------------------------------------------------------------ #
    overall_att: float
    overall_se: float
    overall_t_stat: float
    overall_p_value: float
    overall_conf_int: Tuple[float, float]

    # ------------------------------------------------------------------ #
    # Other aggregations — populated by .aggregate()                      #
    # ------------------------------------------------------------------ #
    group_effects: Optional[Dict[Any, Dict]] = field(default=None, repr=False)
    calendar_effects: Optional[Dict[Any, Dict]] = field(default=None, repr=False)
    event_study_effects: Optional[Dict[int, Dict]] = field(default=None, repr=False)

    # ------------------------------------------------------------------ #
    # Metadata                                                            #
    # ------------------------------------------------------------------ #
    method: str = "ols"
    control_group: str = "not_yet_treated"
    groups: List[Any] = field(default_factory=list)
    time_periods: List[Any] = field(default_factory=list)
    n_obs: int = 0
    n_treated_units: int = 0
    n_control_units: int = 0
    alpha: float = 0.05
    anticipation: int = 0
    survey_metadata: Optional[Any] = field(default=None, repr=False)

    # Variance-family metadata. ``vcov_type`` records the configured analytical
    # family ("classical", "hc1", "hc2", or "hc2_bm"); when ``survey_design=``
    # is supplied the survey TSL (or replicate-weight refit) variance overrides
    # this — the field still records the configured value and
    # ``survey_metadata`` indicates the survey path was active. On bootstrap
    # fits (``n_bootstrap > 0``) the SE comes from the multiplier bootstrap,
    # not the analytical family. ``cluster_name`` / ``n_clusters`` are
    # populated when the fit was clustered (default unit cluster, or
    # user-set ``cluster=X``); both are ``None`` on explicit one-way
    # (``vcov_type in {"classical","hc2"}`` + no user cluster) fits where
    # the auto-cluster was dropped.
    vcov_type: str = "hc1"
    cluster_name: Optional[str] = None
    n_clusters: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Internal — used by aggregate() for delta-method SEs                 #
    # ------------------------------------------------------------------ #
    _gt_weights: Dict[Tuple[Any, Any], int] = field(default_factory=dict, repr=False)
    _gt_vcov: Optional[np.ndarray] = field(default=None, repr=False)
    """Full vcov of all β_{g,t} coefficients (ordered same as sorted group_time_effects keys)."""
    _gt_keys: List[Tuple[Any, Any]] = field(default_factory=list, repr=False)
    """Ordered list of (g,t) keys corresponding to _gt_vcov columns."""
    _df_survey: Optional[int] = field(default=None, repr=False)
    """Survey degrees of freedom for t-distribution inference."""
    _bm_per_cell_dof: Dict[Tuple[Any, Any], float] = field(default_factory=dict, repr=False)
    """Per-cell Bell-McCaffrey Satterthwaite DOF (only populated for vcov_type='hc2_bm').
    Used by group_time_effects[(g, t)] inference fields at fit time."""
    _bm_artifacts: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[Tuple[Any, Any], int]]] = field(
        default=None, repr=False
    )
    """(X_red, cluster_ids, bread_red, coef_idx_map) for hc2_bm; enables
    lazy BM contrast-DOF computation in aggregate().

    ``X_red`` / ``bread_red`` are the REDUCED (kept-column) design and bread
    matrix produced by ``_fit_ols`` after rank-deficient column drops — the
    same subspace ``solve_ols`` returned non-NaN coefficients in.
    ``coef_idx_map`` maps each ``(g, t)`` cell present in
    ``group_time_effects`` to its column index in ``X_red``. Storing reduced
    artifacts avoids the singular full-design bread that
    ``_compute_cr2_bm_contrast_dof`` would otherwise reject."""

    # ------------------------------------------------------------------ #
    # Public methods                                                      #
    # ------------------------------------------------------------------ #

    def aggregate(self, type: str) -> "WooldridgeDiDResults":  # noqa: A002
        """Compute and store one of the four jwdid_estat aggregation types.

        Parameters
        ----------
        type : "simple" | "group" | "calendar" | "event"

        Returns self for chaining.

        Notes
        -----
        When ``vcov_type == "hc2_bm"``, aggregated inference (t_stat / p_value /
        conf_int) uses Bell-McCaffrey Satterthwaite contrast-specific DOFs
        rather than the survey/None default. The BM DOFs are computed lazily
        from ``_bm_artifacts`` via ``_compute_cr2_bm_contrast_dof`` and
        fail-closed (NaN inference) when the helper raises or returns NaN —
        per ``feedback_bm_contrast_dof_fail_closed``.
        """
        valid = ("simple", "group", "calendar", "event")
        if type not in valid:
            raise ValueError(f"type must be one of {valid}, got {type!r}")

        gt = self.group_time_effects
        weights = self._gt_weights
        vcov = self._gt_vcov
        keys_ordered = self._gt_keys if self._gt_keys else sorted(gt.keys())

        def _agg_se(w_vec: np.ndarray) -> float:
            """Delta-method SE for a linear combination w'β given full vcov."""
            if vcov is None or len(w_vec) != vcov.shape[0]:
                return float("nan")
            return float(np.sqrt(max(w_vec @ vcov @ w_vec, 0.0)))

        # Compute BM contrast DOFs lazily for hc2_bm. ``cells_by_key`` is an
        # ordered mapping of aggregation_key -> list of (g, t) cells; the
        # contrast for each key sums the per-cell one-hot vectors weighted
        # by ``weights[(g, t)] / w_total``. Returns a dict mapping
        # aggregation_key -> df (or NaN on fail-closed). For non-hc2_bm,
        # returns an empty dict (caller falls back to ``self._df_survey``).
        def _bm_contrast_dofs_for(
            cells_by_key: Dict[Any, List[Tuple[Any, Any]]],
        ) -> Dict[Any, float]:
            if self.vcov_type != "hc2_bm" or self._bm_artifacts is None:
                return {}
            # ``X_red`` / ``bread_red`` are the REDUCED kept-column artifacts
            # from ``_fit_ols`` (post rank-deficient drops). ``coef_idx_map``
            # maps (g, t) → column index in ``X_red``. See
            # ``_bm_artifacts`` docstring above for the rationale.
            X_red, cluster_ids_full, bread_red, coef_idx_map = self._bm_artifacts
            n_red = X_red.shape[1]
            contrast_cols: List[np.ndarray] = []
            agg_keys: List[Any] = []
            for agg_key, cells in cells_by_key.items():
                if not cells:
                    continue
                w_total = sum(weights.get(c, 0) for c in cells)
                if w_total == 0:
                    continue
                col = np.zeros(n_red)
                contributed = False
                for c in cells:
                    if c not in coef_idx_map:
                        continue
                    col[coef_idx_map[c]] = weights.get(c, 0) / w_total
                    contributed = True
                if not contributed:
                    continue
                contrast_cols.append(col)
                agg_keys.append(agg_key)
            if not contrast_cols:
                return {k: float("nan") for k in cells_by_key}
            from diff_diff.linalg import _compute_cr2_bm_contrast_dof

            contrasts_matrix = np.column_stack(contrast_cols)
            dof_map: Dict[Any, float] = {}
            try:
                dof_vec = _compute_cr2_bm_contrast_dof(
                    X_red, cluster_ids_full, bread_red, contrasts_matrix
                )
                for i, k in enumerate(agg_keys):
                    candidate = float(dof_vec[i])
                    dof_map[k] = candidate if np.isfinite(candidate) else float("nan")
            except (ValueError, np.linalg.LinAlgError) as exc:
                warnings.warn(
                    f"WooldridgeDiDResults.aggregate({type!r}) could not "
                    f"compute Bell-McCaffrey contrast DOF "
                    f"({exc.__class__.__name__}: {exc}). "
                    "Affected aggregated inference (t_stat / p_value / "
                    "conf_int) will be NaN to preserve the hc2_bm contract.",
                    UserWarning,
                    stacklevel=3,
                )
                for k in agg_keys:
                    dof_map[k] = float("nan")
            # Fill non-computed keys with NaN to fail-closed.
            for k in cells_by_key:
                dof_map.setdefault(k, float("nan"))
            return dof_map

        def _build_effect(att: float, se: float, df_for_inference: Optional[float]) -> Dict[str, Any]:
            """Build an effect dict using ``df_for_inference`` for the t-distribution.

            When ``self.vcov_type == "hc2_bm"``, ``df_for_inference`` should be
            the BM contrast DOF (NaN → fail-closed). Otherwise it falls back
            to ``self._df_survey`` (None → normal-theory).
            """
            if self.vcov_type == "hc2_bm":
                if df_for_inference is None or not np.isfinite(df_for_inference):
                    return {
                        "att": att,
                        "se": se,
                        "t_stat": float("nan"),
                        "p_value": float("nan"),
                        "conf_int": (float("nan"), float("nan")),
                    }
                t_stat, p_value, conf_int = safe_inference(
                    att, se, alpha=self.alpha, df=df_for_inference
                )
            else:
                t_stat, p_value, conf_int = safe_inference(
                    att, se, alpha=self.alpha, df=self._df_survey
                )
            return {
                "att": att,
                "se": se,
                "t_stat": t_stat,
                "p_value": p_value,
                "conf_int": conf_int,
            }

        if type == "simple":
            # Re-compute overall using delta method (already stored in overall_* fields)
            # This is a no-op but keeps the method callable.
            pass

        elif type == "group":
            cells_by_g: Dict[Any, List[Tuple[Any, Any]]] = {}
            for g in self.groups:
                cells_by_g[g] = [
                    (g2, t) for (g2, t) in keys_ordered if g2 == g and t >= g
                ]
            dofs = _bm_contrast_dofs_for(cells_by_g)
            result: Dict[Any, Dict] = {}
            for g, cells in cells_by_g.items():
                if not cells:
                    continue
                w_total = sum(weights.get(c, 0) for c in cells)
                if w_total == 0:
                    continue
                att = sum(weights.get(c, 0) * gt[c]["att"] for c in cells) / w_total
                w_vec = np.array(
                    [weights.get(c, 0) / w_total if c in cells else 0.0 for c in keys_ordered]
                )
                se = _agg_se(w_vec)
                result[g] = _build_effect(att, se, dofs.get(g))
            self.group_effects = result

        elif type == "calendar":
            cells_by_t: Dict[Any, List[Tuple[Any, Any]]] = {}
            for t in self.time_periods:
                cells_by_t[t] = [
                    (g, t2) for (g, t2) in keys_ordered if t2 == t and t >= g
                ]
            dofs = _bm_contrast_dofs_for(cells_by_t)
            result = {}
            for t, cells in cells_by_t.items():
                if not cells:
                    continue
                w_total = sum(weights.get(c, 0) for c in cells)
                if w_total == 0:
                    continue
                att = sum(weights.get(c, 0) * gt[c]["att"] for c in cells) / w_total
                w_vec = np.array(
                    [weights.get(c, 0) / w_total if c in cells else 0.0 for c in keys_ordered]
                )
                se = _agg_se(w_vec)
                result[t] = _build_effect(att, se, dofs.get(t))
            self.calendar_effects = result

        elif type == "event":
            all_k = sorted({t - g for (g, t) in keys_ordered})
            cells_by_k: Dict[int, List[Tuple[Any, Any]]] = {}
            for k in all_k:
                cells_by_k[k] = [(g, t) for (g, t) in keys_ordered if t - g == k]
            dofs = _bm_contrast_dofs_for(cells_by_k)
            result = {}
            for k, cells in cells_by_k.items():
                if not cells:
                    continue
                w_total = sum(weights.get(c, 0) for c in cells)
                if w_total == 0:
                    continue
                att = sum(weights.get(c, 0) * gt[c]["att"] for c in cells) / w_total
                w_vec = np.array(
                    [weights.get(c, 0) / w_total if c in cells else 0.0 for c in keys_ordered]
                )
                se = _agg_se(w_vec)
                result[k] = _build_effect(att, se, dofs.get(k))
            self.event_study_effects = result

        return self

    def summary(self, aggregation: str = "simple") -> str:
        """Print formatted summary table.

        Parameters
        ----------
        aggregation : which aggregation to display ("simple", "group", "calendar", "event")
        """
        lines = [
            "=" * 70,
            "    Wooldridge Extended Two-Way Fixed Effects (ETWFE) Results",
            "=" * 70,
            f"Method:          {self.method}",
            f"Control group:   {self.control_group}",
            f"Observations:    {self.n_obs}",
            f"Treated units:   {self.n_treated_units}",
            f"Control units:   {self.n_control_units}",
            "-" * 70,
        ]

        if self.survey_metadata is not None:
            from diff_diff.results import _format_survey_block

            lines.extend(_format_survey_block(self.survey_metadata, 70))
            lines.append("-" * 70)

        def _fmt_row(label: str, att: float, se: float, t: float, p: float, ci: Tuple) -> str:
            from diff_diff.results import _get_significance_stars  # type: ignore

            stars = _get_significance_stars(p) if not np.isnan(p) else ""
            ci_lo = f"{ci[0]:.4f}" if not np.isnan(ci[0]) else "NaN"
            ci_hi = f"{ci[1]:.4f}" if not np.isnan(ci[1]) else "NaN"
            return (
                f"{label:<22} {att:>10.4f}  {se:>10.4f}  {t:>8.3f}  "
                f"{p:>8.4f}{stars}   [{ci_lo}, {ci_hi}]"
            )

        ci_pct = f"{(1 - self.alpha) * 100:.0f}%"
        header = (
            f"{'Parameter':<22} {'Estimate':>10}  {'Std. Err.':>10}  "
            f"{'t-stat':>8}  {'P>|t|':>8}   [{ci_pct} CI]"
        )
        lines.append(header)
        lines.append("-" * 70)

        if aggregation == "simple":
            lines.append(
                _fmt_row(
                    "ATT (simple)",
                    self.overall_att,
                    self.overall_se,
                    self.overall_t_stat,
                    self.overall_p_value,
                    self.overall_conf_int,
                )
            )
        elif aggregation == "group" and self.group_effects:
            for g, eff in sorted(self.group_effects.items()):
                lines.append(
                    _fmt_row(
                        f"ATT(g={g})",
                        eff["att"],
                        eff["se"],
                        eff["t_stat"],
                        eff["p_value"],
                        eff["conf_int"],
                    )
                )
        elif aggregation == "calendar" and self.calendar_effects:
            for t, eff in sorted(self.calendar_effects.items()):
                lines.append(
                    _fmt_row(
                        f"ATT(t={t})",
                        eff["att"],
                        eff["se"],
                        eff["t_stat"],
                        eff["p_value"],
                        eff["conf_int"],
                    )
                )
        elif aggregation == "event" and self.event_study_effects:
            for k, eff in sorted(self.event_study_effects.items()):
                if k < -self.anticipation:
                    suffix = " [pre]"
                elif k < 0:
                    suffix = " [antic]"
                else:
                    suffix = ""
                label = f"ATT(k={k})" + suffix
                lines.append(
                    _fmt_row(
                        label,
                        eff["att"],
                        eff["se"],
                        eff["t_stat"],
                        eff["p_value"],
                        eff["conf_int"],
                    )
                )
        else:
            lines.append(f"  (call .aggregate({aggregation!r}) first)")

        lines.append("=" * 70)
        return "\n".join(lines)

    def to_dataframe(self, aggregation: str = "event") -> pd.DataFrame:
        """Export aggregated effects to a DataFrame.

        Parameters
        ----------
        aggregation : "simple" | "group" | "calendar" | "event" | "gt"
            Use "gt" to export raw group-time effects.
        """
        if aggregation == "gt":
            rows = []
            for (g, t), eff in sorted(self.group_time_effects.items()):
                row = {"cohort": g, "time": t, "relative_period": t - g}
                row.update(eff)
                rows.append(row)
            return pd.DataFrame(rows)

        mapping = {
            "simple": [
                {
                    "label": "ATT",
                    "att": self.overall_att,
                    "se": self.overall_se,
                    "t_stat": self.overall_t_stat,
                    "p_value": self.overall_p_value,
                    "conf_int_lo": self.overall_conf_int[0],
                    "conf_int_hi": self.overall_conf_int[1],
                }
            ],
            "group": [
                {
                    "cohort": g,
                    **{k: v for k, v in eff.items() if k != "conf_int"},
                    "conf_int_lo": eff["conf_int"][0],
                    "conf_int_hi": eff["conf_int"][1],
                }
                for g, eff in sorted((self.group_effects or {}).items())
            ],
            "calendar": [
                {
                    "time": t,
                    **{k: v for k, v in eff.items() if k != "conf_int"},
                    "conf_int_lo": eff["conf_int"][0],
                    "conf_int_hi": eff["conf_int"][1],
                }
                for t, eff in sorted((self.calendar_effects or {}).items())
            ],
            "event": [
                {
                    "relative_period": k,
                    **{kk: vv for kk, vv in eff.items() if kk != "conf_int"},
                    "conf_int_lo": eff["conf_int"][0],
                    "conf_int_hi": eff["conf_int"][1],
                }
                for k, eff in sorted((self.event_study_effects or {}).items())
            ],
        }
        rows = mapping.get(aggregation, [])
        return pd.DataFrame(rows)

    def plot_event_study(self, **kwargs) -> None:
        """Event study plot. Calls aggregate('event') if needed."""
        if self.event_study_effects is None:
            self.aggregate("event")
        from diff_diff.visualization import plot_event_study  # type: ignore

        effects = {k: v["att"] for k, v in (self.event_study_effects or {}).items()}
        se = {k: v["se"] for k, v in (self.event_study_effects or {}).items()}
        plot_event_study(effects=effects, se=se, alpha=self.alpha, **kwargs)

    # --- Inference-field aliases (balance/external-adapter compatibility) ---
    @property
    def att(self) -> float:
        return self.overall_att

    @property
    def se(self) -> float:
        return self.overall_se

    @property
    def conf_int(self) -> Tuple[float, float]:
        return self.overall_conf_int

    @property
    def p_value(self) -> float:
        return self.overall_p_value

    @property
    def t_stat(self) -> float:
        return self.overall_t_stat

    def __repr__(self) -> str:
        n_gt = len(self.group_time_effects)
        att_str = f"{self.overall_att:.4f}" if not np.isnan(self.overall_att) else "NaN"
        se_str = f"{self.overall_se:.4f}" if not np.isnan(self.overall_se) else "NaN"
        p_str = f"{self.overall_p_value:.4f}" if not np.isnan(self.overall_p_value) else "NaN"
        return (
            f"WooldridgeDiDResults("
            f"ATT={att_str}, SE={se_str}, p={p_str}, "
            f"n_gt={n_gt}, method={self.method!r})"
        )
