Synthetic Control Method (SCM)
==============================

Classic synthetic control estimator for a single treated unit (Abadie, Diamond &
Hainmueller 2010; originating in Abadie & Gardeazabal 2003).

The treated unit's counterfactual is a convex combination of "donor" (never-treated)
units. Donor weights ``W*(V)`` solve a simplex-constrained, predictor-importance-weighted
least-squares fit of the treated unit's pre-period predictors; the diagonal
predictor-importance matrix ``V`` is chosen either data-driven (minimizing pre-period
outcome MSPE, ``v_method="nested"``) or supplied by the user (``v_method="custom"``). The
treatment-effect path is the gap :math:`\hat{\alpha}_{1t} = Y_{1t} - \sum_j w_j Y_{jt}`
over the post periods.

**When to use SCM:**

- Exactly **one treated unit** with a long, well-fit pre-treatment period.
- A curated **donor pool** of comparable never-treated units.
- Aggregate / few-unit comparative case studies (states, regions, countries).

**Inference:** classic SCM has **no analytical standard error**. ``se``, ``t_stat``,
``p_value`` and ``conf_int`` are NaN; ``att`` (the mean post-period gap) is the reported
estimate. The paper's in-space placebo permutation inference (post/pre RMSPE-ratio,
``rank/(J+1)`` p-value) is planned for a follow-up release.

**Distinct from** :class:`~diff_diff.SyntheticDiD` (Arkhangelsky et al. 2021), which adds
time weights and ridge regularization; classic SCM uses **donor weights only** plus the
outer ``V`` search.

**Reference:** Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic Control
Methods for Comparative Case Studies: Estimating the Effect of California's Tobacco
Control Program. *Journal of the American Statistical Association*, 105(490), 493–505.
`doi:10.1198/jasa.2009.ap08746 <https://doi.org/10.1198/jasa.2009.ap08746>`_

SyntheticControl
----------------

Main estimator class for classic synthetic control estimation.

.. autoclass:: diff_diff.SyntheticControl
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

   .. rubric:: Methods

   .. autosummary::

      ~SyntheticControl.fit
      ~SyntheticControl.get_params
      ~SyntheticControl.set_params

SyntheticControlResults
-----------------------

Results container for synthetic control estimation.

.. autoclass:: diff_diff.SyntheticControlResults
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::

      ~SyntheticControlResults.summary
      ~SyntheticControlResults.print_summary
      ~SyntheticControlResults.to_dict
      ~SyntheticControlResults.to_dataframe
      ~SyntheticControlResults.get_gap_df
      ~SyntheticControlResults.get_weights_df

Convenience Function
--------------------

.. autofunction:: diff_diff.synthetic_control

Predictors and V selection
--------------------------

Predictor rows of ``X1`` (treated) / ``X0`` (donors) are built, in this canonical order
(matching R ``Synth::dataprep``), from:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Argument
     - Meaning
   * - ``predictors`` + ``predictor_window`` + ``predictors_op``
     - Columns averaged over a pre-period window (default: all pre periods).
   * - ``special_predictors``
     - ``(var, periods, op)`` triples, each averaged over its own periods/operator.
   * - ``pre_period_outcomes``
     - Individual pre-period outcomes as predictor rows (``"all"`` or a list). When
       no predictor arguments are given, defaults to all pre-period outcomes.

``v_method="nested"`` selects the diagonal predictor-importance matrix ``V`` by minimizing
the pre-period **outcome** MSPE of ``W*(V)`` over a multistart Nelder-Mead search with a
derivative-free Powell polish. ``v_method="custom"`` takes a user-supplied ``custom_v``
(one entry per predictor row, trace-normalized) and skips the outer search.

.. note::

   The predictor standardization (per-row SD over donors+treated, ddof=1) and the
   optimizer are pinned from the R ``Synth`` source — they are not specified in
   Abadie-Diamond-Hainmueller (2010). The outer objective uses all pre periods rather than
   R's ``time.optimize.ssr`` window, so the nested ``V`` differs from R by an
   efficiency-only choice. See ``docs/methodology/REGISTRY.md`` §SyntheticControl.

Example Usage
-------------

Basic usage with covariate and special predictors::

    from diff_diff import SyntheticControl

    scm = SyntheticControl(v_method="nested", seed=0)
    results = scm.fit(
        data,
        outcome="gdpcap",
        treatment="treated",   # absorbing 0/1 indicator
        unit="region",
        time="year",
        predictors=["invest", "school.high"],
        # Set predictor_window explicitly when a covariate is only observed on a
        # subset of the pre periods — the default averages over ALL pre periods and
        # fails closed if any selected cell is non-finite.
        predictor_window=[1964, 1965, 1966, 1967, 1968, 1969],
        special_predictors=[("gdpcap", [1960, 1965, 1969], "mean")],
    )
    results.print_summary()

    # Effect path and donor weights
    gap_df = results.get_gap_df()        # period, gap, phase
    weights_df = results.get_weights_df()  # unit, weight (descending)

Quick estimation with the convenience function::

    from diff_diff import synthetic_control

    results = synthetic_control(
        data, outcome="gdpcap", treatment="treated",
        unit="region", time="year",
    )
    print(f"ATT: {results.att:.3f}, pre-RMSPE: {results.pre_rmspe:.3f}")

Supplying a fixed predictor-importance matrix (skips the outer V search)::

    import numpy as np

    scm = SyntheticControl(v_method="custom", custom_v=np.ones(n_predictors))
    results = scm.fit(data, outcome="gdpcap", treatment="treated",
                      unit="region", time="year", predictors=["invest"])

Comparison with Synthetic DiD
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Feature
     - SyntheticControl
     - SyntheticDiD
   * - Unit (donor) weights
     - Simplex, predictor-importance weighted
     - Simplex, ridge-regularized
   * - Time weights
     - None (level matching)
     - Simplex (double difference)
   * - Predictor-importance ``V``
     - Nested / custom diagonal ``V``
     - No analog
   * - Inference
     - Placebo permutation (no analytical SE)
     - Bootstrap / jackknife / placebo

Use **SCM** for a single treated unit with a long pre-period and a curated donor pool;
use **SDID** when you have several treated units and parallel trends is plausible.
