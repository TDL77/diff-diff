Spillover-Aware DiD (Butts 2021)
=================================

Ring-indicator spillover-aware Difference-in-Differences estimator.

This module implements the methodology from Butts, K. (2023, originally 2021),
"Difference-in-Differences with Spatial Spillovers" (arXiv:2105.03737v3).
The estimator augments two-stage Gardner (2022) DiD with ring-indicator
covariates that identify, alongside the direct effect on treated units
(``tau_total``), per-ring spillover effects on near-control units
(``delta_j``).

The standard DiD estimator is biased for ``tau_total`` by
``-tau_spill(0)`` (paper Proposition 2.1 / Equations 1-2); the
ring-augmented regression removes this bias. The "far-away" cutoff
``d_bar`` defines the boundary beyond which spillovers vanish
(Assumption 5).

**When to use SpilloverDiD:**

- Spatial-policy DiD settings where treatment may spill over onto nearby
  control units (border-RD, place-based policy, neighborhood effects,
  geographic policy boundaries)
- When the canonical DiD coefficient is suspected to be attenuated or
  inflated by spillover bias and a far-away control group exists
- As a robustness check on TwoStageDiD / ImputationDiD when the
  treatment has plausibly local geographic externalities

**Reference:** Butts, K. (2023). Difference-in-Differences with Spatial
Spillovers. *arXiv:2105.03737v3*. Gardner, J. (2022). Two-stage
differences in differences. *arXiv:2207.05943*.

.. module:: diff_diff.spillover

SpilloverDiD
------------

Main estimator class.

.. autoclass:: diff_diff.SpilloverDiD
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

   .. rubric:: Methods

   .. autosummary::

      ~SpilloverDiD.fit
      ~SpilloverDiD.get_params
      ~SpilloverDiD.set_params

SpilloverDiDResults
-------------------

Results container with per-ring spillover-effect table.

.. autoclass:: diff_diff.SpilloverDiDResults
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::

      ~SpilloverDiDResults.summary
      ~SpilloverDiDResults.to_dict

Example Usage
-------------

Non-staggered 2-period panel with synthetic spillovers::

    from diff_diff import SpilloverDiD

    est = SpilloverDiD(
        rings=[0, 50, 100, 200],          # 3 rings: [0,50), [50,100), [100,200]
        conley_coords=("lat", "lon"),
        vcov_type="conley",
        conley_cutoff_km=200.0,
        conley_lag_cutoff=0,
    )
    result = est.fit(
        data=df,
        outcome="y",
        unit="unit",
        time="time",
        treatment="D",                    # binary; auto-converts to first_treat
    )
    print(result.summary())
    # Total effect on treated (Butts tau_total):
    print(f"tau_total = {result.att:.4f}")
    # Per-ring spillover-on-control:
    print(result.spillover_effects)

Staggered timing (Gardner-convention first_treat column)::

    result = est.fit(
        data=df,
        outcome="y",
        unit="unit",
        time="time",
        first_treat="first_treat",        # 0 / inf = never-treated
    )

Identification spec
-------------------

The stage-2 regressor for ring j is the time-varying form

.. math::

    (1 - D_{it}) \cdot \mathrm{Ring}_{it,j}

where ``Ring_{it,j}`` is the indicator that unit *i*'s nearest currently-
treated unit is in distance bin *j* at time *t*, and ``D_it`` is the
treatment indicator (1 if unit *i* is treated by time *t*). For
non-staggered timing, all treated units share one onset and ``Ring`` is
unit-static post-treatment / zero pre-treatment. For staggered timing,
``Ring_{it,j}`` is unit-time-varying (a unit's nearest treated neighbor
changes as more cohorts enter treatment).

Stage 1 fits unit + time fixed effects on Butts' subsample
``Omega_0 = {D_it = 0 AND S_it = 0}`` (untreated AND unexposed) — the
clean far-away control group. Reading the literal unit-static
``(1 - D_it) * S_i`` from paper Equation 5 yields a rank-deficient
design under TWFE; the diff-diff implementation uses the time-varying
form (paper page 12's ``S_it`` notation, made explicit in Section 5
Table 2).

Estimator Comparison
--------------------

.. list-table:: SpilloverDiD vs. TwoStageDiD vs. TwoWayFixedEffects
   :header-rows: 1
   :widths: 25 25 25 25

   * - Feature
     - SpilloverDiD
     - TwoStageDiD
     - TwoWayFixedEffects
   * - Spillover handling
     - Identifies per-ring delta_j
     - None (assumes SUTVA)
     - None (assumes SUTVA)
   * - Methodology
     - Two-stage Gardner + ring covariates
     - Two-stage Gardner
     - Single-stage TWFE
   * - Staggered timing
     - Yes
     - Yes
     - Biased under heterogeneity
   * - Stage-1 subsample
     - Butts: ``D=0 AND S=0`` (untreated AND unexposed)
     - ``D=0`` (untreated)
     - N/A (single stage)
   * - Conley spatial-HAC SE
     - Yes (Wave D GMM-corrected sandwich)
     - Not yet supported
     - Yes
   * - Cluster-robust SE
     - Yes (HC1 + CR1, Wave D GMM-corrected sandwich)
     - Yes (GMM sandwich + clusters)
     - Yes

Restrictions and follow-ups
---------------------------

The current implementation has the following documented restrictions
and planned follow-up enhancements:

- **Gardner GMM first-stage correction at stage 2** — SHIPPED in Wave D.
  Stage-2 variance now applies the influence-function-based correction
  for stage-1 FE estimation uncertainty across all three ``vcov_type``
  paths (HC1, Conley, cluster) on both ``event_study=False`` AND
  ``event_study=True``. The IF formula is
  ``psi_i = gamma_hat' * X_{10,i} * eps_{10,i} - X_{2,i} * eps_{2,i}``
  with ``gamma_hat = (X_10' X_10)^{-1} (X_1' X_2)``; the meat is
  ``Psi' K Psi`` where ``K`` is the path-dependent kernel matrix
  (identity for HC1, block-indicator for cluster, spatial kernel for
  Conley). Documented synthesis of Butts (2021) Section 3.1 + Gardner
  (2022) Section 4 + Conley (1999); no reference software combines all
  three ingredients. Point estimates unchanged; SE values shift upward
  by 1-few percent depending on first-stage residual variance.
- **Event-study mode** — ``event_study=True`` is SHIPPED in Wave C.
  The per-event-time × ring decomposition (Butts Section 5 / Table 2)
  emits per-event-time direct effects ``tau_k`` and per-(ring,
  event-time) spillover effects ``delta_jk`` as a ``att_dynamic``
  DataFrame plus MultiIndex ``spillover_effects``. The
  ``event_study_effects: Dict[int, Dict]`` alias mirrors
  ``TwoStageDiD``'s schema for ``plot_event_study`` consumption (the
  plotter prefers the new ``reference_period`` attribute over the
  legacy ``n_obs==0`` heuristic). ``DiagnosticReport`` routing for
  ``SpilloverDiDResults`` is queued as a follow-up. Reference
  period ``-1 - anticipation`` (TwoStageDiD parity). ``horizon_max``
  bins event-times into endpoint pools (no row drop — divergence
  from TwoStageDiD's filtering semantic, intentional per
  ``feedback_no_silent_failures``). ``horizon_max`` must be ``>=1`` or
  ``None`` under ``event_study=True``; ``horizon_max=0`` is rejected
  (the single bin ``k=0`` leaves no event-time pair to anchor the
  reference period — for a single aggregate effect, use
  ``event_study=False`` instead). Scalar ``att`` becomes a
  sample-share-weighted average of post-treatment ``tau_k`` with SE
  from linear-combination inference on the post-treatment vcov block.
  Per-event-time SEs apply the Wave D Gardner GMM first-stage
  uncertainty correction (see the "Gardner GMM first-stage correction"
  entry above).
- **Survey-design integration** — ``survey_design=`` raises
  ``NotImplementedError``.
- **Count-of-treated-in-ring** — only the "nearest-treated ring"
  specification is implemented. The "count" form re-introduces
  functional-form dependence (paper Section 3.2 end) and is queued.
- **Data-driven d_bar selection** — Butts (2021b) / Butts (2023) JUE
  Insight propose cross-validation under stronger parallel-trends
  assumptions. Not in this PR.
- **HC2 / HC2_BM (Bell-McCaffrey / CR2) variance** — current stage-2
  inference uses a generic residual-df (n - effective rank) for
  t-distribution lookups. ``vcov_type="hc2"`` / ``"hc2_bm"`` require
  per-coefficient BM / CR2 DOF and raise ``NotImplementedError``.
  Routing stage 2 through ``LinearRegression`` (which supplies the
  per-coefficient DOF metadata) is queued.
- **`vcov_type="classical"` (Wave D restriction)** — raises
  ``NotImplementedError``. The Wave D Gardner GMM first-stage
  uncertainty correction has not been derived for the classical
  homoskedastic variance (different meat structure
  ``sigma_hat^2 * (X_10' X_10)`` vs the Wave D IF outer product
  ``Psi' Psi``). Use ``vcov_type="hc1"`` for heteroskedasticity-robust
  SE with the GMM correction, or combine with ``cluster=<col>`` for
  CR1 with the GMM correction; both apply the Wave D synthesis
  (Butts §3.1 + Gardner §4 + Conley 1999) unconditionally.
- **Balanced panel required** — every unit must observe every period.
  An unbalanced (unit, time) Ω₀ bipartite graph can produce disconnected
  FE components and unidentified stage-1 residuals on treated rows.
  Validator rejects unbalanced inputs.
- **Omega_0 row-level identification (period strict, unit warn-drop, plus connectivity)** —
  Every period must have at least one Omega_0 row (else time FE for
  that period is structurally unidentified; hard ``ValueError``). Units
  lacking Omega_0 rows (e.g. baseline-treated units with ``D_it = 1`` at
  every observed ``t``) are warned-and-dropped: their unit FE is NaN
  and the downstream finite-mask path excludes them from stage 2.
  Additionally, the supported-units bipartite graph (units linked by
  shared Omega_0 periods) must form a single connected component;
  ``K > 1`` components raise ``ValueError`` because the FE solver would
  return only component-specific constants and residualization would
  silently mix them across components. Mirrors ``TwoStageDiD``'s
  always-treated unit handling on the warn-drop axis.
- **One row per (unit, time) cell required** — duplicate cells silently
  re-weight both stage-1 FE estimation and stage-2 OLS. Validator
  rejects duplicate cells; aggregate to unique cells before fitting.
- **rings[0] must equal 0** — the partition must cover treated
  locations (``d_it = 0`` belongs to Ring 1). Validator rejects rings
  that start at a nonzero inner edge to prevent a silent
  exposed-but-unmodeled population in ``0 <= d_it < rings[0]``.
- **conley_coords must be constant within unit** — ring construction
  uses a single (lat, lon) per unit (units are stationary point
  locations). Validator rejects coordinates that vary across rows of
  the same unit; aggregate to one location per unit (e.g.
  ``df.groupby(unit)[["lat","lon"]].first()``) before fitting if your
  raw data has moving coordinates. This requirement applies on every
  fit path, not just ``vcov_type="conley"``, because the rings
  themselves are always coordinate-derived.
