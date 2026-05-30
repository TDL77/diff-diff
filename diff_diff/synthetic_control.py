"""
Classic Synthetic Control Method (SCM) estimator.

Implements Abadie, Diamond & Hainmueller (2010), "Synthetic Control Methods for
Comparative Case Studies: Estimating the Effect of California's Tobacco Control
Program," *JASA* 105(490):493-505. The method originates in Abadie &
Gardeazabal (2003).

A single treated unit's counterfactual is built as a convex combination of
"donor" (never-treated) units. Donor weights ``W*(V)`` solve a simplex-constrained
weighted least-squares fit of the treated unit's predictors; the predictor-importance
matrix ``V`` (diagonal, PSD) is either chosen data-driven by minimizing pre-period
outcome MSPE ("nested") or supplied by the user ("custom"). The treatment-effect
path is the gap ``α̂_1t = Y_1t − Σ_j w_j · Y_jt`` over the post periods.

Distinct from :class:`~diff_diff.SyntheticDiD` (Arkhangelsky et al. 2021), which adds
time weights and ridge regularization: classic SCM uses **donor weights only** and a
level-matching estimator, plus the outer ``V`` search SyntheticDiD has no analog for.

Inference: classic SCM has **no analytical standard error** — the paper proposes
permutation/placebo inference (a later PR). ``se``/``t_stat``/``p_value``/``conf_int``
are always NaN here; ``att`` (mean post-period gap) is the reported estimate.

Numerics provenance: the standardization divisor and the inner/outer optimization
scheme are NOT specified in ADH (2010) — they are pinned from the R ``Synth`` package
source / Abadie-Gardeazabal (2003) App. B. See ``docs/methodology/REGISTRY.md``
§SyntheticControl for the deviation/Note labels.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from diff_diff.synthetic_control_results import SyntheticControlResults
from diff_diff.utils import _sc_weight_fw, safe_inference, warn_if_not_converged

__all__ = ["SyntheticControl", "synthetic_control"]

# Aggregation operators allowed for predictor / special-predictor rows. Restricted to
# LINEAR combinations of pre-period values, matching ADH (2010) §2.3
# (`Ȳ_i^{K_m} = Σ_s k_s Y_is`): mean (k_s = 1/T0) and sum (k_s = 1). Non-linear
# aggregations (e.g. median) are intentionally NOT supported.
_OP_FUNCS: Dict[str, Any] = {"mean": np.mean, "sum": np.sum}

# Interpretability floor: donor weights below this are dropped from the reported
# ``donor_weights`` dict (mirrors prep.py's 1e-6 sparsification). The full weight
# vector is still used for the gap path / ATT.
_MIN_REPORT_WEIGHT = 1e-6


@dataclass
class _PredictorSpec:
    """Normalized predictor specification (one matrix row of X1/X0)."""

    label: str
    kind: str  # "predictor_avg" | "special" | "lag"
    var: str
    periods: List[Any]
    op: str  # "mean" | "sum" | "identity"


def _softmax(theta: np.ndarray) -> np.ndarray:
    """Map an unconstrained vector to the unit simplex (v >= 0, sum v = 1)."""
    e = np.exp(theta - np.max(theta))
    return e / np.sum(e)


class SyntheticControl:
    """
    Classic Synthetic Control Method estimator (Abadie-Diamond-Hainmueller 2010).

    Parameters
    ----------
    v_method : {"nested", "custom"}, default "nested"
        How the predictor-importance matrix V is chosen. ``"nested"`` selects V
        data-driven by minimizing the pre-period outcome MSPE of ``W*(V)``
        (ADH 2010 §2.3). ``"custom"`` uses the user-supplied ``custom_v`` and
        skips the outer search.
    custom_v : array-like, optional
        Diagonal of V (length = number of predictors). Required iff
        ``v_method="custom"``; must be None when ``v_method="nested"``. Must be
        finite and non-negative; trace-normalized internally.
    optimizer_options : dict, optional
        Extra options merged into every ``scipy.optimize.minimize`` call in the
        outer V search (e.g. ``maxiter``, ``xatol``, ``fatol``).
    n_starts : int, default 4
        Number of starting points for the multistart outer V search.
    inner_max_iter : int, default 10000
        Max iterations for the inner Frank-Wolfe simplex solve.
    inner_min_decrease : float, default 1e-5
        Inner-solve convergence scale (matches the SDID/Frank-Wolfe precedent in
        ``prep.py``). The Frank-Wolfe stop threshold is
        ``(inner_min_decrease * max(||b||, 1e-12))**2`` where ``b`` is the
        V^½-scaled treated predictor vector — scale-aware so convergence is
        meaningful at any data magnitude. 1e-5 reproduces R ``Synth``'s donor
        weights to ~1e-4 on the Basque benchmark while still signalling
        convergence; tighter values (e.g. 1e-6) can exhaust ``inner_max_iter``.
    standardize : {"std", "none"}, default "std"
        Predictor standardization. ``"std"`` divides each predictor row by its
        standard deviation across donors+treated (ddof=1), matching R ``Synth``.
        ``"none"`` is a deviation from R (see REGISTRY).
    alpha : float, default 0.05
        Significance level recorded for downstream (placebo) inference.
    seed : int, optional
        Seed for the multistart random (Dirichlet) starting points.
    """

    def __init__(
        self,
        v_method: str = "nested",
        custom_v: Optional[Any] = None,
        optimizer_options: Optional[Dict[str, Any]] = None,
        n_starts: int = 4,
        inner_max_iter: int = 10000,
        inner_min_decrease: float = 1e-5,
        standardize: str = "std",
        alpha: float = 0.05,
        seed: Optional[int] = None,
    ):
        self.v_method = v_method
        self.custom_v = custom_v
        self.optimizer_options = optimizer_options
        self.n_starts = n_starts
        self.inner_max_iter = inner_max_iter
        self.inner_min_decrease = inner_min_decrease
        self.standardize = standardize
        self.alpha = alpha
        self.seed = seed

        self._validate_config()

        # Internal state
        self.results_: Optional[SyntheticControlResults] = None
        self.is_fitted_: bool = False

    # =========================================================================
    # sklearn-like API
    # =========================================================================

    def _validate_config(self) -> None:
        """Validate hyperparameters; shared by ``__init__`` and ``set_params``."""
        if self.v_method not in ("nested", "custom"):
            raise ValueError(f"v_method must be one of ('nested', 'custom'), got {self.v_method!r}")
        if self.standardize not in ("std", "none"):
            raise ValueError(
                f"standardize must be one of ('std', 'none'), got {self.standardize!r}"
            )
        # custom_v cross-field rules — fail-closed, never silently ignore.
        if self.v_method == "custom" and self.custom_v is None:
            raise ValueError("custom_v is required when v_method='custom'.")
        if self.v_method == "nested" and self.custom_v is not None:
            raise ValueError(
                "custom_v must be None when v_method='nested' "
                "(it would be silently ignored otherwise)."
            )
        if self.custom_v is not None:
            v = np.asarray(self.custom_v, dtype=float).ravel()
            if v.size == 0:
                raise ValueError("custom_v must be non-empty.")
            if not np.all(np.isfinite(v)):
                raise ValueError("custom_v must be finite.")
            if np.any(v < 0):
                raise ValueError("custom_v must be non-negative.")
            if v.sum() <= 0:
                raise ValueError("custom_v must have a positive sum.")
        if not isinstance(self.n_starts, (int, np.integer)) or self.n_starts < 1:
            raise ValueError(f"n_starts must be a positive integer, got {self.n_starts!r}")
        if not isinstance(self.inner_max_iter, (int, np.integer)) or self.inner_max_iter < 1:
            raise ValueError(
                f"inner_max_iter must be a positive integer, got {self.inner_max_iter!r}"
            )
        if not (np.isfinite(self.inner_min_decrease) and self.inner_min_decrease > 0):
            raise ValueError(
                f"inner_min_decrease must be a positive float, got {self.inner_min_decrease!r}"
            )
        if not (0 < self.alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha!r}")

    def get_params(self) -> Dict[str, Any]:
        """Get estimator parameters."""
        return {
            "v_method": self.v_method,
            "custom_v": self.custom_v,
            "optimizer_options": self.optimizer_options,
            "n_starts": self.n_starts,
            "inner_max_iter": self.inner_max_iter,
            "inner_min_decrease": self.inner_min_decrease,
            "standardize": self.standardize,
            "alpha": self.alpha,
            "seed": self.seed,
        }

    def set_params(self, **params) -> "SyntheticControl":
        """Set estimator parameters.

        Applies updates transactionally: if ``_validate_config()`` rejects the
        post-update state, the instance is rolled back to its pre-call values so
        a raised ``ValueError`` leaves the object consistent.
        """
        _rollback: Dict[str, Any] = {}
        for key in params:
            if hasattr(self, key):
                _rollback[key] = getattr(self, key)
        try:
            for key, value in params.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    raise ValueError(f"Unknown parameter: {key}")
            self._validate_config()
        except (ValueError, TypeError):
            for key, prev in _rollback.items():
                setattr(self, key, prev)
            raise
        return self

    # =========================================================================
    # fit
    # =========================================================================

    def fit(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        unit: str,
        time: str,
        *,
        post_periods: Optional[List[Any]] = None,
        treated_unit: Optional[Any] = None,
        predictors: Optional[List[str]] = None,
        predictors_op: str = "mean",
        predictor_window: Optional[List[Any]] = None,
        special_predictors: Optional[List[Tuple[str, List[Any], str]]] = None,
        pre_period_outcomes: Optional[Any] = None,
        donor_pool: Optional[List[Any]] = None,
        survey_design: Any = None,
    ) -> SyntheticControlResults:
        """
        Fit the classic synthetic control model.

        Parameters
        ----------
        data : pandas.DataFrame
            Balanced panel.
        outcome, treatment, unit, time : str
            Column names. ``treatment`` is the ABSORBING treatment indicator
            (0/1): 1 for the treated unit in its treated periods, 0 otherwise.
        post_periods : list, optional
            Explicit post-treatment period values. If None, inferred from the
            treated unit's treatment column (the D==1 periods).
        treated_unit : Any, optional
            Identifier of the treated unit. If None, inferred as the single
            ever-treated unit.
        predictors : list of str, optional
            Columns averaged over ``predictor_window`` (using ``predictors_op``)
            to form predictor rows.
        predictors_op : {"mean", "sum"}, default "mean"
            Aggregation operator for ``predictors`` (linear combinations only, per
            ADH 2010 §2.3).
        predictor_window : list, optional
            Pre-periods over which ``predictors`` are averaged. Defaults to all
            pre periods. Must be a non-empty subset of the pre periods.
        special_predictors : list of (var, periods, op), optional
            Per-variable special predictors, each averaged over its own periods
            with its own operator (mirrors R ``Synth`` ``special.predictors``).
        pre_period_outcomes : "all" or list, optional
            Use individual pre-period outcomes as predictor rows ("all" = every
            pre period). When no predictor arguments are given at all, defaults
            to all pre-period outcomes.
        donor_pool : list, optional
            Explicit donor unit identifiers (must be never-treated). Defaults to
            all never-treated units.
        survey_design : optional
            Not yet supported — raises ``NotImplementedError`` if provided.

        Returns
        -------
        SyntheticControlResults
        """
        if survey_design is not None:
            raise NotImplementedError(
                "SyntheticControl does not yet support survey_design. "
                "Survey integration is planned for a later release."
            )

        # --- column validation ---
        required = [outcome, treatment, unit, time]
        if predictors:
            required += list(predictors)
        if special_predictors:
            required += [sp[0] for sp in special_predictors]
        missing = [c for c in dict.fromkeys(required) if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Reject missing structural values up front: a NaN treatment status would be
        # silently dropped by groupby(...).max() (a partially-missing donor history
        # could be misclassified as never-treated), and NaN unit/time break the panel
        # structure. Done BEFORE classification so donor-pool composition is honest.
        for col in (treatment, unit, time):
            if bool(data[col].isna().to_numpy().any()):
                raise ValueError(
                    f"Column {col!r} contains missing (NaN) values; treatment, unit, and "
                    "time must be fully observed (a missing treatment status would "
                    "silently change donor classification)."
                )

        # Validate the treatment indicator on the FULL input BEFORE classifying
        # units: a non-{0,1} history would otherwise be silently dropped from both
        # the treated and never-treated sets by _resolve_treated_and_donors, quietly
        # changing the donor pool / weights / ATT.
        _check_binary(data[treatment].to_numpy(), treatment)

        # --- treated unit + donor pool ---
        treated_id, donor_ids = _resolve_treated_and_donors(
            data, treatment, unit, treated_unit, donor_pool
        )

        # --- restrict to treated + donors; all_periods derived AFTER donor filter ---
        keep_ids = [treated_id] + donor_ids
        sub = cast(pd.DataFrame, data[data[unit].isin(keep_ids)])
        all_periods = sorted(sub[time].unique())

        # balanced panel check (on the analysis subset)
        _check_balanced(sub, unit, time, keep_ids, all_periods)

        # --- pre/post period resolution + canonicalization + cross-checks ---
        pre_periods, post_periods = _resolve_periods(
            sub, time, treatment, unit, treated_id, all_periods, post_periods, self.v_method
        )

        # --- predictor specs (validations 1, 5, 7) ---
        specs = _validate_predictors(
            predictors,
            predictors_op,
            predictor_window,
            special_predictors,
            pre_period_outcomes,
            outcome,
            pre_periods,
        )

        # --- pivots, predictor matrices, outcome matrices ---
        needed_vars = {outcome} | {s.var for s in specs}
        pivots = {
            v: sub.pivot(index=time, columns=unit, values=v).reindex(index=all_periods)
            for v in needed_vars
        }
        Y = pivots[outcome]
        Z1 = Y.loc[pre_periods, treated_id].to_numpy(dtype=float)
        Z0 = Y.loc[pre_periods, donor_ids].to_numpy(dtype=float)  # (n_pre, J)

        # Fail closed on non-finite outcomes for the treated/donor panel: NaN/inf in
        # the outcome would silently propagate into the gap path / ATT (and into Z for
        # the nested objective).
        outcome_block = Y.loc[all_periods, [treated_id] + donor_ids].to_numpy(dtype=float)
        if not np.all(np.isfinite(outcome_block)):
            raise ValueError(
                f"Outcome {outcome!r} contains non-finite (NaN/inf) values for the "
                "treated unit or a donor over the analysis periods; synthetic control "
                "requires a complete outcome panel."
            )

        X1, X0, labels = _build_predictor_matrix(pivots, specs, treated_id, donor_ids)
        # Fail closed on non-finite predictor cells (e.g. a covariate that is only
        # observed on a subset of the pre periods averaged over the full pre window).
        bad = [
            labels[i]
            for i in range(len(labels))
            if not (np.isfinite(X1[i]) and np.all(np.isfinite(X0[i, :])))
        ]
        if bad:
            raise ValueError(
                f"Predictor(s) {bad} have non-finite (NaN/inf) values for the treated "
                "unit or a donor over their selected periods. Restrict predictor_window / "
                "special_predictors periods to where the variable is observed."
            )
        X1s, X0s, _ = _standardize(X1, X0, self.standardize)

        k = X1s.shape[0]
        J = len(donor_ids)

        # --- solve for V and donor weights ---
        # mspe_v is the OUTER-objective value; it is populated only when a nested V
        # search actually ran (None on the custom and single-donor paths).
        mspe_v: Optional[float] = None
        if self.v_method == "custom":
            v = self._prepare_custom_v(k)
            w, converged = _inner_solve_W(X1s, X0s, v, self.inner_max_iter, self.inner_min_decrease)
        elif J == 1:
            # Degenerate: a single donor forces w = [1.0]; V is irrelevant.
            warnings.warn(
                "Only one donor unit is available; the synthetic control is that "
                "single donor (w = 1) and the V search is skipped. SCM is degenerate "
                "with a single donor.",
                UserWarning,
                stacklevel=2,
            )
            v = np.ones(k) / k
            w = np.array([1.0])
            converged = True
        else:
            v, w, converged, mspe_v = _outer_solve_V(
                X1s,
                X0s,
                Z1,
                Z0,
                self.n_starts,
                self.seed,
                self.optimizer_options,
                self.inner_max_iter,
                self.inner_min_decrease,
            )

        # --- gap path, fit diagnostics, ATT ---
        gap_path = _compute_gap_path(Y, w, treated_id, donor_ids, all_periods)
        pre_gaps = np.array([gap_path[p] for p in pre_periods], dtype=float)
        post_gaps = np.array([gap_path[p] for p in post_periods], dtype=float)
        pre_rmspe = float(np.sqrt(np.mean(pre_gaps**2)))
        att = float(np.mean(post_gaps))

        # Poor-fit warning (REGISTRY contract: warn when pre-RMSPE exceeds the SD of
        # the treated unit's pre-period outcomes). This includes a FLAT treated pre-path
        # (pre_sd == 0): any non-trivial RMSPE then means the synthetic cannot reproduce
        # a constant series. A scale-aware absolute floor (`_fit_tol`) guards against a
        # spurious warning on a near-perfect flat fit (RMSPE ~ roundoff).
        pre_sd = float(np.std(Z1, ddof=1)) if Z1.size > 1 else 0.0
        _fit_tol = 1e-8 * max(float(np.max(np.abs(Z1))) if Z1.size else 0.0, 1.0)
        if pre_rmspe > pre_sd and pre_rmspe > _fit_tol:
            warnings.warn(
                f"Pre-treatment fit is poor: RMSPE ({pre_rmspe:.4f}) exceeds the "
                f"standard deviation of treated pre-treatment outcomes ({pre_sd:.4f}). "
                f"The synthetic control may not adequately reproduce the treated unit's "
                f"pre-treatment trajectory; consider a different donor pool or predictors.",
                UserWarning,
                stacklevel=2,
            )

        # Inner-solve non-convergence warning — fires on nested / custom / k==1 paths.
        warn_if_not_converged(
            converged,
            "Synthetic control inner weight solver (Frank-Wolfe)",
            self.inner_max_iter,
            stacklevel=2,
        )

        # No analytical SE: all inference fields NaN.
        t_stat, p_value, conf_int = safe_inference(att, np.nan, self.alpha)

        # --- reporting structures ---
        donor_weights = {donor_ids[j]: float(w[j]) for j in range(J) if w[j] > _MIN_REPORT_WEIGHT}
        v_weights = {labels[i]: float(v[i]) for i in range(k)}
        synthetic_X = X0 @ w
        donor_mean = X0.mean(axis=1)
        predictor_balance = pd.DataFrame(
            {
                "predictor": labels,
                "treated": X1,
                "synthetic": synthetic_X,
                "donor_mean": donor_mean,
            }
        )

        results = SyntheticControlResults(
            att=att,
            se=np.nan,
            t_stat=t_stat,
            p_value=p_value,
            conf_int=conf_int,
            n_obs=int(len(sub)),
            n_donors=J,
            n_pre_periods=len(pre_periods),
            n_post_periods=len(post_periods),
            donor_weights=donor_weights,
            v_weights=v_weights,
            predictor_balance=predictor_balance,
            gap_path=gap_path,
            pre_rmspe=pre_rmspe,
            treated_unit=treated_id,
            pre_periods=list(pre_periods),
            post_periods=list(post_periods),
            v_method=self.v_method,
            standardize=self.standardize,
            alpha=self.alpha,
            mspe_v=mspe_v,
        )
        # Reserved for PR-2 (placebo inference) / PR-3 (conformal). Set as plain
        # attributes so dataclasses.asdict/fields cannot reach them.
        results._placebo_gaps = None
        results._rmspe_ratio = None
        results._fit_snapshot = None

        self.results_ = results
        self.is_fitted_ = True
        return results

    def _prepare_custom_v(self, k: int) -> np.ndarray:
        """Validate ``custom_v`` against the predictor count and trace-normalize."""
        v = np.asarray(self.custom_v, dtype=float).ravel()
        if v.shape[0] != k:
            raise ValueError(
                f"custom_v has length {v.shape[0]} but there are {k} predictors. "
                "custom_v must have exactly one entry per predictor row."
            )
        # Finiteness / non-negativity already enforced in _validate_config.
        return v / v.sum()


def synthetic_control(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    unit: str,
    time: str,
    **kwargs,
) -> SyntheticControlResults:
    """
    Convenience function for classic synthetic control estimation.

    Constructor-only keyword arguments (``v_method``, ``custom_v``, ``n_starts``,
    ``standardize``, ``alpha``, ``seed``, ``optimizer_options``,
    ``inner_max_iter``, ``inner_min_decrease``) and ``fit`` keyword arguments
    (``post_periods``, ``treated_unit``, ``predictors``, ``special_predictors``,
    ...) may both be passed via ``**kwargs``.

    Examples
    --------
    >>> from diff_diff import synthetic_control
    >>> res = synthetic_control(data, "y", "treated", "unit", "year",
    ...                         predictors=["x1", "x2"])
    >>> print(f"ATT: {res.att:.3f}, pre-RMSPE: {res.pre_rmspe:.3f}")
    """
    ctor_keys = set(SyntheticControl().get_params().keys())
    ctor_kwargs = {k: v for k, v in kwargs.items() if k in ctor_keys}
    fit_kwargs = {k: v for k, v in kwargs.items() if k not in ctor_keys}
    estimator = SyntheticControl(**ctor_kwargs)
    return estimator.fit(data, outcome, treatment, unit, time, **fit_kwargs)


# =============================================================================
# module-private helpers
# =============================================================================


def _check_binary(arr: np.ndarray, name: str) -> None:
    """Raise ValueError if ``arr`` contains values other than 0/1."""
    uniq = np.unique(arr[~np.isnan(arr)]) if arr.dtype.kind == "f" else np.unique(arr)
    if not np.all(np.isin(uniq, [0, 1])):
        raise ValueError(f"{name} must be binary (0 or 1). Found values: {uniq}")


def _check_balanced(
    sub: pd.DataFrame, unit: str, time: str, keep_ids: List[Any], all_periods: List[Any]
) -> None:
    """Raise ValueError if the analysis subset is not a balanced panel."""
    n_periods = len(all_periods)
    counts = sub.groupby(unit)[time].nunique()
    bad = [u for u in keep_ids if counts.get(u, 0) != n_periods]
    if bad:
        raise ValueError(
            f"Unbalanced panel: units {bad} are not observed at all {n_periods} "
            "periods. Synthetic control requires a balanced panel."
        )
    if len(sub) != len(keep_ids) * n_periods:
        raise ValueError(
            "Panel has duplicate (unit, time) rows; synthetic control requires "
            "exactly one observation per unit-period."
        )


def _resolve_treated_and_donors(
    data: pd.DataFrame,
    treatment: str,
    unit: str,
    treated_unit: Optional[Any],
    donor_pool: Optional[List[Any]],
) -> Tuple[Any, List[Any]]:
    """Identify the single treated unit and the (never-treated) donor pool."""
    ever_treated = cast(pd.Series, data.groupby(unit)[treatment].max())
    treated_units = [u for u, v in ever_treated.items() if v == 1]
    never_treated = [u for u, v in ever_treated.items() if v == 0]

    if treated_unit is not None:
        if treated_unit not in set(data[unit]):
            raise ValueError(f"treated_unit {treated_unit!r} not found in the data.")
        if treated_unit not in treated_units:
            raise ValueError(
                f"treated_unit {treated_unit!r} has no treated (D==1) periods. "
                "The treatment column must mark the treated unit's post periods."
            )
        treated_id = treated_unit
    else:
        if len(treated_units) == 0:
            raise ValueError(
                "No treated unit found (no unit has any D==1 period). Synthetic "
                "control requires exactly one treated unit."
            )
        if len(treated_units) > 1:
            raise ValueError(
                f"Found {len(treated_units)} ever-treated units {treated_units}; "
                "synthetic control requires exactly one. Pass treated_unit= to "
                "select one, and exclude the others from the donor pool."
            )
        treated_id = treated_units[0]

    if donor_pool is not None:
        donor_ids = list(dict.fromkeys(donor_pool))
        if treated_id in donor_ids:
            raise ValueError("donor_pool must not contain the treated unit.")
        missing = [u for u in donor_ids if u not in set(data[unit])]
        if missing:
            raise ValueError(f"donor_pool units not found in the data: {missing}")
        contaminated = [u for u in donor_ids if u not in set(never_treated)]
        if contaminated:
            raise ValueError(
                f"donor_pool contains ever-treated units {contaminated}; donors "
                "must be never-treated."
            )
    else:
        donor_ids = [u for u in never_treated if u != treated_id]

    if len(donor_ids) == 0:
        raise ValueError("No donor units available (the donor pool is empty).")

    return treated_id, donor_ids


def _resolve_periods(
    sub: pd.DataFrame,
    time: str,
    treatment: str,
    unit: str,
    treated_id: Any,
    all_periods: List[Any],
    post_periods: Optional[List[Any]],
    v_method: str,
) -> Tuple[List[Any], List[Any]]:
    """Resolve and validate pre/post periods (absorbing + no-anticipation)."""
    period_index = {p: i for i, p in enumerate(all_periods)}
    treated_rows = sub[sub[unit] == treated_id].set_index(time)[treatment]
    treated_d = {p: int(treated_rows.loc[p]) for p in all_periods}
    treated_active = [p for p in all_periods if treated_d[p] == 1]

    if post_periods is None:
        if not treated_active:
            raise ValueError(
                "Cannot infer post periods: the treated unit has no D==1 periods. "
                "Pass post_periods= explicitly."
            )
        post = sorted(treated_active, key=lambda p: period_index[p])
    else:
        # Canonicalize: unique + calendar-sorted (validation 3).
        seen = set()
        canon = []
        for p in post_periods:
            if p not in period_index:
                raise ValueError(f"post_periods value {p!r} is not a period in the data.")
            if p not in seen:
                seen.add(p)
                canon.append(p)
        if not canon:
            raise ValueError("post_periods must be non-empty.")
        post = sorted(canon, key=lambda p: period_index[p])

    # Contiguous suffix (absorbing assignment): post must be the last len(post)
    # periods of the calendar axis (validation 2).
    suffix = all_periods[len(all_periods) - len(post) :]
    if post != suffix:
        raise ValueError(
            "post_periods must be a contiguous suffix of the time axis (absorbing "
            f"treatment). Got {post}, expected the trailing periods {suffix}."
        )

    pre_periods = all_periods[: len(all_periods) - len(post)]

    # Absorbing-treatment cross-check vs the treated unit's D column (validation 2),
    # enforced on BOTH the inferred and explicit branches:
    #   - no anticipation: the treated unit must be untreated (D==0) in every pre period;
    #   - uninterrupted exposure: the treated unit must be treated (D==1) in every post
    #     period (an explicit suffix like [t2, t3] over a path 0,0,1,0 would otherwise
    #     average a treated period with an untreated one).
    anticipated = [p for p in pre_periods if treated_d[p] == 1]
    if anticipated:
        raise ValueError(
            f"Treated unit has D==1 in pre periods {anticipated}; this violates the "
            "no-anticipation / absorbing-treatment assumption. Redefine post_periods "
            "to begin at the first treated period."
        )
    untreated_post = [p for p in post if treated_d[p] != 1]
    if untreated_post:
        raise ValueError(
            f"Treated unit has D==0 in post periods {untreated_post}; absorbing "
            "treatment requires uninterrupted exposure (D==1) in every post period. "
            "Check post_periods against the treatment column."
        )

    if len(pre_periods) == 0:
        raise ValueError(
            "No pre-treatment periods: synthetic control cannot fit a counterfactual "
            "without at least one pre period."
        )
    if v_method == "nested" and len(pre_periods) == 1:
        warnings.warn(
            "Only one pre-treatment period: data-driven V selection (v_method='nested') "
            "is unreliable with a single pre period (the outer MSPE degenerates to one "
            "term). Consider v_method='custom' or more pre periods.",
            UserWarning,
            stacklevel=2,
        )

    return pre_periods, post


def _validate_predictors(
    predictors: Optional[List[str]],
    predictors_op: str,
    predictor_window: Optional[List[Any]],
    special_predictors: Optional[List[Tuple[str, List[Any], str]]],
    pre_period_outcomes: Optional[Any],
    outcome: str,
    pre_periods: List[Any],
) -> List[_PredictorSpec]:
    """Build the normalized, collision-checked predictor specification list.

    Canonical row ORDER: covariate/predictor averages -> special predictors ->
    per-period outcome lags (the row order matches R ``Synth::dataprep``). NOTE: the
    aggregation itself fails closed on any non-finite cell (handled in ``fit``),
    whereas R ``dataprep`` uses ``na.rm=TRUE`` — a documented deviation (see REGISTRY).
    """
    pre_set = set(pre_periods)
    pre_index = {p: i for i, p in enumerate(pre_periods)}
    specs: List[_PredictorSpec] = []
    labels: set = set()

    def _canon(periods: List[Any]) -> List[Any]:
        # Unique + calendar-sorted, so reordered/duplicated period lists are
        # equivalent (e.g. [c,b,a] == [a,b,c]; a repeated period does not
        # re-weight a "mean"). Assumes membership already checked by _check_periods.
        return sorted(dict.fromkeys(periods), key=lambda p: pre_index[p])

    def _add(label: str, kind: str, var: str, periods: List[Any], op: str) -> None:
        if label in labels:
            raise ValueError(
                f"Duplicate predictor label {label!r}. Each predictor (including "
                "per-period outcome lags) must have a unique label."
            )
        labels.add(label)
        specs.append(_PredictorSpec(label=label, kind=kind, var=var, periods=periods, op=op))

    def _check_periods(periods: List[Any], ctx: str) -> None:
        if len(periods) == 0:
            raise ValueError(f"{ctx} period list must not be empty.")
        leak = [p for p in periods if p not in pre_set]
        if leak:
            raise ValueError(
                f"{ctx} references periods {leak} outside the pre-treatment window "
                "(would leak post-treatment information)."
            )

    # 1) predictor averages
    if predictors:
        if predictors_op not in _OP_FUNCS:
            raise ValueError(
                f"predictors_op must be one of {sorted(_OP_FUNCS)}, got {predictors_op!r}"
            )
        if predictor_window is not None:
            window = list(predictor_window)
            _check_periods(window, "predictor_window")
            window = _canon(window)
        else:
            window = list(pre_periods)
        for var in predictors:
            _add(var, "predictor_avg", var, window, predictors_op)

    # 2) special predictors
    if special_predictors:
        for entry in special_predictors:
            if len(entry) != 3:
                raise ValueError(
                    "Each special predictor must be a (var, periods, op) tuple; " f"got {entry!r}."
                )
            var, periods, op = entry
            if op not in _OP_FUNCS:
                raise ValueError(
                    f"special predictor op must be one of {sorted(_OP_FUNCS)}, got {op!r}"
                )
            periods = list(periods)
            _check_periods(periods, f"special predictor {var!r}")
            periods = _canon(periods)  # reordered/duplicated -> same canonical spec
            # Full ordered period list in the label so distinct period sets sharing
            # the same endpoints/length (e.g. [2000,2002,2004] vs [2000,2003,2004])
            # do not collide — the label is also the v_weights / balance key.
            label = f"{var}@{op}[{','.join(str(p) for p in periods)}]"
            _add(label, "special", var, periods, op)

    # 3) per-period outcome lags
    if pre_period_outcomes is not None:
        if isinstance(pre_period_outcomes, str):
            if pre_period_outcomes != "all":
                raise ValueError(
                    f"pre_period_outcomes string must be 'all', got {pre_period_outcomes!r}"
                )
            lag_periods = list(pre_periods)
        else:
            lag_periods = list(pre_period_outcomes)
            _check_periods(lag_periods, "pre_period_outcomes")
            lag_periods = _canon(lag_periods)
        for p in lag_periods:
            _add(f"{outcome}_{p}", "lag", outcome, [p], "identity")

    # default: nothing specified -> use all pre-period outcomes as lag predictors
    if not specs:
        for p in pre_periods:
            _add(f"{outcome}_{p}", "lag", outcome, [p], "identity")

    return specs


def _build_predictor_matrix(
    pivots: Dict[str, pd.DataFrame],
    specs: List[_PredictorSpec],
    treated_id: Any,
    donor_ids: List[Any],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build X1 (k,) and X0 (k, J) from the normalized predictor specs."""
    k = len(specs)
    J = len(donor_ids)
    X1 = np.empty(k, dtype=float)
    X0 = np.empty((k, J), dtype=float)
    labels: List[str] = []

    for i, spec in enumerate(specs):
        pv = pivots[spec.var]
        block = pv.loc[spec.periods]  # rows: periods, cols: units
        if spec.op == "identity":
            X1[i] = float(block[treated_id].iloc[0])
            X0[i, :] = block[donor_ids].iloc[0].to_numpy(dtype=float)
        else:
            op_func = _OP_FUNCS[spec.op]
            X1[i] = float(op_func(block[treated_id].to_numpy(dtype=float)))
            X0[i, :] = op_func(block[donor_ids].to_numpy(dtype=float), axis=0)
        labels.append(spec.label)

    return X1, X0, labels


def _standardize(
    X1: np.ndarray, X0: np.ndarray, mode: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row-standardize predictors by SD across donors+treated (ddof=1).

    Matches R ``Synth``: ``divisor <- sqrt(apply(cbind(X0, X1), 1, var))``.
    Zero-variance rows keep divisor 1.0 (with a warning).
    """
    k = X1.shape[0]
    if mode == "none":
        return X1.copy(), X0.copy(), np.ones(k)
    combined = np.column_stack([X0, X1.reshape(-1, 1)])  # (k, J+1)
    divisor = np.sqrt(np.var(combined, axis=1, ddof=1))
    zero = divisor <= 0
    if np.any(zero):
        warnings.warn(
            f"{int(np.sum(zero))} predictor row(s) have zero variance across "
            "donors+treated; their standardization divisor is set to 1.0.",
            UserWarning,
            stacklevel=2,
        )
        divisor = divisor.copy()
        divisor[zero] = 1.0
    X1s = X1 / divisor
    X0s = X0 / divisor[:, None]
    return X1s, X0s, divisor


def _inner_solve_W(
    X1s: np.ndarray,
    X0s: np.ndarray,
    v: np.ndarray,
    inner_max_iter: int,
    inner_min_decrease: float,
) -> Tuple[np.ndarray, bool]:
    """Solve W*(V) = argmin_W (X1-X0 W)' diag(V) (X1-X0 W) on the unit simplex.

    Folds V^½ into the predictors and delegates to the Frank-Wolfe simplex solver.
    Returns the RAW simplex weights (no sparsification) so the outer objective is
    not perturbed; reporting-level cleanup happens in ``fit``.
    """
    J = X0s.shape[1]
    if J == 1:
        return np.array([1.0]), True
    vh = np.sqrt(v)
    A = vh[:, None] * X0s  # (k, J)
    b = vh * X1s  # (k,)
    packed = np.column_stack([A, b])  # (k, J+1); last column is the target
    min_decrease = inner_min_decrease * max(float(np.linalg.norm(b)), 1e-12)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*did not converge.*", category=UserWarning)
        w, converged = _sc_weight_fw(
            packed,
            zeta=0.0,
            intercept=False,
            min_decrease=min_decrease,
            max_iter=inner_max_iter,
            return_convergence=True,
        )
    return np.asarray(w, dtype=float), bool(converged)


def _v_starts(
    k: int,
    X1s: np.ndarray,
    X0s: np.ndarray,
    Z1: np.ndarray,
    Z0: np.ndarray,
    n_starts: int,
    rng: np.random.Generator,
    inner_max_iter: int,
    inner_min_decrease: float,
) -> List[np.ndarray]:
    """Build a list of starting ``theta`` vectors for the outer V search.

    Heuristic starts: uniform V; inverse-row-variance V; univariate-fit V
    (v_i ∝ 1/MSPE_i from solving with mass concentrated on predictor i). Remaining
    starts are random Dirichlet draws. Non-finite candidates are dropped
    (validation 10); uniform is always retained.
    """

    def _to_theta(v: np.ndarray) -> Optional[np.ndarray]:
        v = np.asarray(v, dtype=float)
        # Keep only finite, positive mass; reject candidates with no usable mass
        # (validation 10). Floor before log so a near-zero (or underflowed-to-0)
        # Dirichlet draw cannot yield log(0) = -inf and force an infinite retry.
        v = np.where(np.isfinite(v) & (v > 0), v, 0.0)
        total = float(np.sum(v))
        if total <= 0:
            return None
        v = np.maximum(v / total, 1e-12)
        theta = np.log(v)
        theta = theta - np.mean(theta)
        return theta if np.all(np.isfinite(theta)) else None

    # Candidates are generated lazily and we stop as soon as n_starts are collected,
    # so a small n_starts does not pay for heuristic starts it would only discard. In
    # particular n_starts=1 returns the uniform start without running the O(k) univariate
    # inner-solve loop below. The candidate ORDER (uniform -> inverse-variance ->
    # univariate-fit -> Dirichlet) is unchanged, so any given n_starts yields the same
    # set as before — only unused work is skipped.
    target = max(n_starts, 1)
    candidates: List[np.ndarray] = [np.zeros(k)]  # uniform V

    # inverse row variance of the standardized predictors over donors+treated
    if len(candidates) < target:
        combined = np.column_stack([X0s, X1s.reshape(-1, 1)])
        row_var = np.var(combined, axis=1, ddof=1)
        inv_var = np.where(row_var > 0, 1.0 / np.maximum(row_var, 1e-12), 0.0)
        if np.sum(inv_var) > 0:
            t = _to_theta(inv_var / np.sum(inv_var))
            if t is not None:
                candidates.append(t)

    # univariate-fit start: v_i ∝ 1 / (pre-outcome MSPE of W solved with V=e_i).
    # Skipped entirely when enough candidates are already collected (saves k inner solves).
    if len(candidates) < target:
        uni_mspe = np.empty(k)
        for i in range(k):
            e = np.zeros(k)
            e[i] = 1.0
            w_i, _ = _inner_solve_W(X1s, X0s, e, inner_max_iter, inner_min_decrease)
            uni_mspe[i] = float(np.mean((Z1 - Z0 @ w_i) ** 2))
        inv_mspe = np.where(uni_mspe > 0, 1.0 / np.maximum(uni_mspe, 1e-12), 0.0)
        if np.sum(inv_mspe) > 0:
            t = _to_theta(inv_mspe / np.sum(inv_mspe))
            if t is not None:
                candidates.append(t)

    # random Dirichlet draws to reach n_starts (bounded attempts as a backstop)
    attempts = 0
    max_attempts = 10 * n_starts + 20
    while len(candidates) < target and attempts < max_attempts:
        attempts += 1
        t = _to_theta(rng.dirichlet(np.ones(k)))
        if t is not None:
            candidates.append(t)

    return candidates[:target]


def _outer_solve_V(
    X1s: np.ndarray,
    X0s: np.ndarray,
    Z1: np.ndarray,
    Z0: np.ndarray,
    n_starts: int,
    seed: Optional[int],
    optimizer_options: Optional[Dict[str, Any]],
    inner_max_iter: int,
    inner_min_decrease: float,
) -> Tuple[np.ndarray, np.ndarray, bool, float]:
    """Data-driven V selection: minimize pre-period outcome MSPE of W*(V).

    Multistart Nelder-Mead over ``theta`` (V = softmax(theta)) plus a derivative-free
    Powell polish from the best point. Returns ``(v_star, w_star, converged, mspe)``.
    """
    k = X1s.shape[0]
    if k == 1:
        v = np.array([1.0])
        w, converged = _inner_solve_W(X1s, X0s, v, inner_max_iter, inner_min_decrease)
        return v, w, converged, float(np.mean((Z1 - Z0 @ w) ** 2))

    def objective(theta: np.ndarray) -> float:
        v = _softmax(theta)
        w, _ = _inner_solve_W(X1s, X0s, v, inner_max_iter, inner_min_decrease)
        return float(np.mean((Z1 - Z0 @ w) ** 2))

    nm_options = {"maxiter": 1000, "xatol": 1e-8, "fatol": 1e-8}
    if optimizer_options:
        nm_options.update(optimizer_options)
    # Powell uses xtol/ftol rather than Nelder-Mead's xatol/fatol; translate so the
    # same optimizer_options control both solvers without an OptimizeWarning.
    powell_options = dict(nm_options)
    if "xatol" in powell_options:
        powell_options["xtol"] = powell_options.pop("xatol")
    if "fatol" in powell_options:
        powell_options["ftol"] = powell_options.pop("fatol")

    rng = np.random.default_rng(seed)
    starts = _v_starts(k, X1s, X0s, Z1, Z0, n_starts, rng, inner_max_iter, inner_min_decrease)

    best_x: np.ndarray = starts[0]
    best_fun = np.inf
    outer_converged = False
    for theta0 in starts:
        res = minimize(objective, theta0, method="Nelder-Mead", options=nm_options)
        outer_converged = outer_converged or bool(res.success)
        if res.fun < best_fun:
            best_fun = float(res.fun)
            best_x = res.x

    # Derivative-free polish from the incumbent (best-of, mirrors R optimx).
    res_p = minimize(objective, best_x, method="Powell", options=powell_options)
    outer_converged = outer_converged or bool(res_p.success)
    if res_p.fun < best_fun:
        best_fun = float(res_p.fun)
        best_x = res_p.x

    # Surface a silent under-optimized V: if neither the multistart Nelder-Mead nor
    # the Powell polish reported success (e.g. optimizer_options={"maxiter": 1}), the
    # selected V / donor weights / ATT may be sub-optimal.
    if not outer_converged:
        warnings.warn(
            "Outer V-search (Nelder-Mead / Powell) did not converge; the selected "
            "predictor-importance matrix V (and the resulting donor weights / ATT) may "
            "be sub-optimal. Increase optimizer_options['maxiter'] or n_starts.",
            UserWarning,
            stacklevel=3,
        )

    v_star = _softmax(best_x)
    w_star, converged = _inner_solve_W(X1s, X0s, v_star, inner_max_iter, inner_min_decrease)
    mspe = float(np.mean((Z1 - Z0 @ w_star) ** 2))
    return v_star, w_star, converged, mspe


def _compute_gap_path(
    Y: pd.DataFrame,
    w: np.ndarray,
    treated_id: Any,
    donor_ids: List[Any],
    all_periods: List[Any],
) -> Dict[Any, float]:
    """Period-keyed gap path ``α̂_1t = Y_1t − Σ_j w_j Y_jt`` over all periods."""
    treated_series = Y.loc[all_periods, treated_id].to_numpy(dtype=float)
    donor_block = Y.loc[all_periods, donor_ids].to_numpy(dtype=float)  # (T, J)
    synthetic = donor_block @ w
    gaps = treated_series - synthetic
    return {period: float(g) for period, g in zip(all_periods, gaps)}
