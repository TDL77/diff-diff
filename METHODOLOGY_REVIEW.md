# Methodology Review

This document tracks the progress of reviewing each estimator's implementation against the Methodology Registry and academic references. It ensures that implementations are correct, consistent, and well-documented.

For the methodology registry with academic foundations and key equations, see [docs/methodology/REGISTRY.md](docs/methodology/REGISTRY.md).

---

## Overview

Each estimator in diff-diff should be periodically reviewed to ensure:
1. **Correctness**: Implementation matches the academic paper's equations
2. **Reference alignment**: Behavior matches reference implementations (R packages, Stata commands)
3. **Edge case handling**: Documented edge cases are handled correctly
4. **Standard errors**: SE formulas match the documented approach

### What "Complete" means in this tracker

A **Complete** entry has a documented review pass against the primary academic source captured in this file. The minimum content is:

- A "Corrections Made" block listing every implementation fix the review uncovered, or `(None — implementation verified correct)`.
- An explicit statement of deviations from the reference implementation, or `(None)`. Format varies — some entries use a dedicated "Deviations" / "Deviations from R" block, others surface deviations inline in "Corrections Made" or "Outstanding Concerns".
- Verification evidence: a "Verified Components" checklist, an "Edge Cases Verified" enumeration, an "R Comparison Results" table, or some combination of these.

The catalog grew incrementally over several quarters, so formats vary across the existing Complete entries; the consistent invariant is that someone walked through the implementation against the academic source and captured the result here. New reviews going forward should aim for the fuller structure (Verified Components + Corrections Made + Deviations + dedicated methodology test file) used by the more recent entries.

**In Progress** entries have a REGISTRY.md section and unit-test coverage, but no formal walk-through has been captured here yet. The In Progress band is wide — some entries also have some combination of a paper review (primary or companion), a dedicated methodology test file, and R parity fixtures; others have only the REGISTRY entry and unit tests (e.g., PowerAnalysis). The "Documentation in place" sub-section enumerates what each entry already has; the "Outstanding for promotion" sub-section enumerates what's still needed to flip it to Complete.

**Not Started** entries have neither a tracker walk-through nor an REGISTRY.md section. This tracker no longer carries any Not Started rows; new estimators are expected to enter as In Progress when their REGISTRY entry lands.

---

## Review Status Summary

### Core DiD Estimators

| Estimator | Module | R / Stata Reference | Status | Last Review |
|-----------|--------|---------------------|--------|-------------|
| DifferenceInDifferences | `estimators.py` | `fixest::feols()` | **Complete** | 2026-01-24 |
| MultiPeriodDiD | `estimators.py` | `fixest::feols()` | **Complete** | 2026-02-02 |
| TwoWayFixedEffects | `twfe.py` | `fixest::feols()` | **Complete** | 2026-02-08 |

### Staggered Treatment Estimators

| Estimator | Module | R / Stata Reference | Status | Last Review |
|-----------|--------|---------------------|--------|-------------|
| CallawaySantAnna | `staggered.py` | `did::att_gt()` | **Complete** | 2026-01-24 |
| SunAbraham | `sun_abraham.py` | `fixest::sunab()` | **Complete** | 2026-02-15 |
| StackedDiD | `stacked_did.py` | `stacked-did-weights` (Wing-Freedman-Hollingsworth code) | **Complete** | 2026-02-19 |
| ImputationDiD | `imputation.py` | `didimputation` | **In Progress** | — |
| TwoStageDiD | `two_stage.py` | `did2s` | **In Progress** | — |
| WooldridgeDiD (ETWFE) | `wooldridge.py` | `etwfe` (R) / `jwdid` (Stata) | **Complete** | 2026-05-22 |
| EfficientDiD | `efficient_did.py` | (no canonical R package) | **In Progress** | — |

### Continuous & Universal-Treatment Estimators

| Estimator | Module | R / Stata Reference | Status | Last Review |
|-----------|--------|---------------------|--------|-------------|
| ContinuousDiD | `continuous_did.py` | `contdid` v0.1.0 | **Complete** | 2026-05-20 |
| ChaisemartinDHaultfoeuille (DCDH) | `chaisemartin_dhaultfoeuille.py` | `DIDmultiplegtDYN` | **Complete** | 2026-05-21 |
| HeterogeneousAdoptionDiD (HAD) | `had.py`, `had_pretests.py` | `chaisemartin::did_had` (`Credible-Answers/did_had` v2.0.0); `nprobust` for bandwidth | **Complete** | 2026-05-20 |
| TROP | `trop.py`, `trop_local.py`, `trop_global.py` | (forthcoming; paper-author reference implementation) | **Complete** (paper `method="local"`) | 2026-05-24 |

### Triple-Difference Estimators

| Estimator | Module | R Reference | Status | Last Review |
|-----------|--------|-------------|--------|-------------|
| TripleDifference | `triple_diff.py` | `triplediff::ddd()` | **Complete** | 2026-02-18 |
| StaggeredTripleDifference | `staggered_triple_diff.py` | `triplediff::ddd(panel=TRUE)` + `agg_ddd()` | **In Progress** | — |

### Counterfactual / Synthetic Estimators

| Estimator | Module | R Reference | Status | Last Review |
|-----------|--------|-------------|--------|-------------|
| SyntheticDiD | `synthetic_did.py` | `synthdid::synthdid_estimate()` | **Complete** | 2026-04-23 |

### Diagnostics & Sensitivity

| Tool | Module | R Reference | Status | Last Review |
|------|--------|-------------|--------|-------------|
| BaconDecomposition | `bacon.py` | `bacondecomp::bacon()` | **Complete** | 2026-05-16 |
| HonestDiD | `honest_did.py` | `HonestDiD` package | **Complete** | 2026-04-01 |
| PreTrendsPower | `pretrends.py` | `pretrends` package | **Complete** | 2026-05-19 |
| PowerAnalysis | `power.py` | `pwr` / `DeclareDesign` | **In Progress** | — |
| PlaceboTests | `diagnostics.py` | (no canonical reference) | **In Progress** | — |

### Cross-Cutting Inference Features

| Feature | Module | Reference | Status | Last Review |
|---------|--------|-----------|--------|-------------|
| ConleySpatialHAC | `conley.py`, `linalg.py` | `conleyreg` (R) / `acreg` (Stata) | **In Progress** | — |
| Survey Data Support | `survey.py`, `bootstrap_utils.py` | `survey` package (R) | **In Progress** | — |

**Status legend** (matches the contract in [§ What "Complete" means in this tracker](#what-complete-means-in-this-tracker) above):
- **Not Started**: No REGISTRY.md entry yet. Reserved for future surfaces; this tracker currently carries no Not Started rows.
- **In Progress**: REGISTRY.md entry and unit-test coverage exist, but no formal walk-through has been captured in this document yet. The band is wide — see each entry's "Documentation in place" / "Outstanding for promotion" sub-sections for specifics.
- **Complete**: A documented review pass against the primary academic source is captured here (minimum: Corrections Made, Deviations or `(None)`, and Verified Components / Edge Cases Verified / R Comparison Results in some form).

---

## Detailed Review Notes

### Core DiD Estimators

#### DifferenceInDifferences

| Field | Value |
|-------|-------|
| Module | `estimators.py` |
| Primary Reference | Wooldridge (2010), Angrist & Pischke (2009) |
| R Reference | `fixest::feols()` |
| Status | **Complete** |
| Last Review | 2026-01-24 |

**Verified Components:**
- [x] ATT formula: Double-difference of cell means matches regression interaction coefficient
- [x] R comparison: ATT matches `fixest::feols()` within 1e-3 tolerance
- [x] R comparison: SE (HC1 robust) matches within 5%
- [x] R comparison: P-value matches within 0.01
- [x] R comparison: Confidence intervals overlap
- [x] R comparison: Cluster-robust SE matches within 10%
- [x] R comparison: Fixed effects (absorb) matches `feols(...|unit)` within 1%
- [x] Wild bootstrap inference (Rademacher, Mammen, Webb weights)
- [x] Formula interface (`y ~ treated * post`)
- [x] All REGISTRY.md edge cases tested

**Test Coverage:**
- 51 methodology verification tests in `tests/test_methodology_did.py`
- Existing unit-test coverage in `tests/test_estimators.py` (`TestDifferenceInDifferences` class plus shared estimator-API classes)
- R benchmark tests (skip if R not available)

**R Comparison Results:**
- ATT matches within 1e-3 (R JSON truncation limits precision)
- HC1 SE matches within 5%
- Cluster-robust SE matches within 10%
- Fixed effects results match within 1%

**Corrections Made:**
- (None — implementation verified correct)

**Outstanding Concerns:**
- R comparison precision limited by JSON output truncation (4 decimal places)
- Consider improving R script to output full precision for tighter tolerances

**Edge Cases Verified:**
1. Empty cells: Produces rank deficiency warning (expected behavior)
2. Singleton clusters: Included in variance estimation, contribute via residuals (corrected REGISTRY.md)
3. Rank deficiency: All three modes (warn/error/silent) working
4. Non-binary treatment/time: Raises ValueError as expected
5. No variation in treatment/time: Raises ValueError as expected
6. Missing values: Raises ValueError as expected

**Deviations from R's `fixest::feols()`:** (None — point estimates and SEs match within
documented tolerances; cluster-robust and absorbed-FE behavior verified.)

---

#### MultiPeriodDiD

| Field | Value |
|-------|-------|
| Module | `estimators.py` |
| Primary Reference | Freyaldenhoven et al. (2021), Wooldridge (2010), Angrist & Pischke (2009) |
| R Reference | `fixest::feols()` |
| Status | **Complete** |
| Last Review | 2026-02-02 |

**Verified Components:**
- [x] Full event-study specification: treatment × period interactions for ALL non-reference periods (pre and post)
- [x] Reference period coefficient is zero (normalized by omission from design matrix)
- [x] Default reference period is last pre-period (e=-1 convention, matches fixest/did)
- [x] Pre-period coefficients available for parallel trends assessment
- [x] Average ATT computed from post-treatment effects only, with covariance-aware SE
- [x] Returns PeriodEffect objects with confidence intervals for all periods
- [x] Supports balanced and unbalanced panels
- [x] NaN inference: t_stat/p_value/CI use NaN when SE is non-finite or zero
- [x] R-style NA propagation: avg_att is NaN if any post-period effect is unidentified
- [x] Rank-deficient design matrix: warns and sets NaN for dropped coefficients (R-style)
- [x] Staggered adoption detection warning (via `unit` parameter)
- [x] Treatment reversal detection warning
- [x] Time-varying D_it detection warning (advises creating ever-treated indicator)
- [x] Single pre-period warning (ATT valid but pre-trends assessment unavailable)
- [x] Post-period reference_period raises ValueError (would bias avg_att)
- [x] HonestDiD/PreTrendsPower integration uses interaction sub-VCV (not full regression VCV)
- [x] All REGISTRY.md edge cases tested

**Test Coverage:**
- 50 tests across `TestMultiPeriodDiD` and `TestMultiPeriodDiDEventStudy` in `tests/test_estimators.py`
- 18 new event-study specification tests added in PR #125

**Corrections Made:**
- **PR #125 (2026-02-02)**: Transformed from post-period-only estimator into full event-study
  specification with pre-period coefficients. Reference period default changed from first
  pre-period to last pre-period (e=-1 convention). HonestDiD/PreTrendsPower VCV extraction
  fixed to use interaction sub-VCV instead of full regression VCV.

**Outstanding Concerns:**
- R comparison benchmark via `benchmarks/R/benchmark_multiperiod.R` using
  `fixest::feols(outcome ~ treated * time_f | unit)`. ATT diff < 1e-11, SE diff 0.0%,
  period-effects correlation 1.0. Validated at small (200 units) and 1k scales.
- Endpoint binning for distant event times not yet implemented.
- FutureWarning for reference_period default change should eventually be removed once
  the transition is complete.

**Deviations from R's `fixest::feols()`:**
1. **Default SE is HC1**, not cluster-robust at unit level (the `fixest` default for panel
   data). Cluster-robust available via `cluster` parameter but not the default.
2. **Reference period default is last pre-period** (e=-1 convention, matches `fixest`/`did`);
   prior Python releases used first pre-period and the change is gated by a `FutureWarning`
   until the deprecation window closes.

---

#### TwoWayFixedEffects

| Field | Value |
|-------|-------|
| Module | `twfe.py` |
| Primary Reference | Wooldridge (2010), Ch. 10 |
| R Reference | `fixest::feols()` |
| Status | **Complete** |
| Last Review | 2026-02-08 |

**Verified Components:**
- [x] Within-transformation algebra: `y_it - ȳ_i - ȳ_t + ȳ` matches hand calculation (rtol=1e-12)
- [x] ATT matches manual demeaned OLS (rtol=1e-10)
- [x] ATT matches `DifferenceInDifferences` on 2-period data (rtol=1e-10)
- [x] Covariates are also within-transformed (sum to zero within unit/time groups)
- [x] R comparison: ATT matches `fixest::feols(y ~ treated:post | unit + post, cluster=~unit)` (rtol<0.1%)
- [x] R comparison: Cluster-robust SE match (rtol<1%)
- [x] R comparison: P-value match (atol<0.01)
- [x] R comparison: CI bounds match (rtol<1%)
- [x] R comparison: ATT and SE match with covariate (same tolerances)
- [x] Edge case: Staggered treatment triggers `UserWarning`
- [x] Edge case: Auto-clusters at unit level (SE matches explicit `cluster="unit"`)
- [x] Edge case: DF adjustment for absorbed FE matches manual `solve_ols()` with `df_adjustment`
- [x] Edge case: Covariate collinear with interaction raises `ValueError` ("cannot be identified")
- [x] Edge case: Covariate collinearity warns but ATT remains finite
- [x] Edge case: `rank_deficient_action="error"` raises `ValueError`
- [x] Edge case: `rank_deficient_action="silent"` emits no warnings
- [x] Edge case: Unbalanced panel produces valid results (finite ATT, positive SE)
- [x] Edge case: Missing unit column raises `ValueError`
- [x] Integration: `decompose()` returns `BaconDecompositionResults`
- [x] SE: Cluster-robust SE >= HC1 SE
- [x] SE: VCoV positive semi-definite
- [x] Wild bootstrap: Valid inference (finite SE, p-value in [0,1])
- [x] Wild bootstrap: All weight types (rademacher, mammen, webb) produce valid inference
- [x] Wild bootstrap: `inference="wild_bootstrap"` routes correctly
- [x] Params: `get_params()` returns all inherited parameters
- [x] Params: `set_params()` modifies attributes
- [x] Results: `summary()` contains "ATT"
- [x] Results: `to_dict()` contains att, se, t_stat, p_value, n_obs
- [x] Results: residuals + fitted = demeaned outcome (not raw)
- [x] Edge case: Multi-period time emits UserWarning advising binary post indicator
- [x] Edge case: Non-{0,1} binary time emits UserWarning (ATT still correct)
- [x] Edge case: ATT invariant to time encoding ({0,1} vs {2020,2021} produces identical results)

**Key Implementation Detail:**
The interaction term `D_i × Post_t` must be within-transformed (demeaned) alongside the outcome,
consistent with the Frisch-Waugh-Lovell (FWL) theorem: all regressors and the outcome must be
projected out of the fixed effects space. R's `fixest::feols()` does this automatically when
variables appear to the left of the `|` separator.

**Corrections Made:**
- **Bug fix: interaction term must be within-transformed** (found during review). The previous
  implementation used raw (un-demeaned) `D_i × Post_t` in the demeaned regression. This gave
  correct results only for 2-period panels where `post == period`. For multi-period panels
  (e.g., 4 periods with binary `post`), the raw interaction had incorrect correlation with
  demeaned Y, producing ATT approximately 1/3 of the true value. Fixed by applying the same
  within-transformation to the interaction term before regression. This matches R's
  `fixest::feols()` behavior. (`twfe.py` lines 99-113)

**Outstanding Concerns:**
- **Multi-period `time` parameter**: Multi-period time values (e.g., 1,2,3,4) produce
  `treated × period_number` instead of `treated × post_indicator`, which is not the standard
  D_it treatment indicator. A `UserWarning` is emitted when `time` has >2 unique values.
  For binary time with non-{0,1} values (e.g., {2020, 2021}), the ATT is mathematically
  correct (the within-transformation absorbs the scaling), but a warning recommends 0/1
  encoding for clarity. Users with multi-period data should create a binary `post` column.
- **Staggered treatment warning**: The warning only fires when `time` has >2 unique values
  (i.e., actual period numbers). With binary `time="post"`, all treated units appear to start
  treatment at `time=1`, making staggering undetectable. Users with staggered designs should
  use `decompose()` or `CallawaySantAnna` directly for proper diagnostics.

**Deviations from R's `fixest::feols()`:** (None — point estimates, cluster-robust SEs,
CI bounds, and absorbed-FE results all match within documented tolerances on both bare
and covariate-adjusted specifications.)

---

### Staggered Treatment Estimators

#### CallawaySantAnna

| Field | Value |
|-------|-------|
| Module | `staggered.py` |
| Primary Reference | Callaway & Sant'Anna (2021) |
| R Reference | `did::att_gt()` |
| Status | **Complete** |
| Last Review | 2026-01-24 |

**Verified Components:**
- [x] ATT(g,t) basic formula (hand-calculated exact match)
- [x] Doubly robust estimator
- [x] IPW estimator
- [x] Outcome regression
- [x] Base period selection (varying/universal)
- [x] Anticipation parameter handling
- [x] Simple/event-study/group aggregation
- [x] Analytical SE with weight influence function
- [x] Bootstrap SE (Rademacher/Mammen/Webb)
- [x] Control group composition (never_treated/not_yet_treated)
- [x] All documented edge cases from REGISTRY.md

**Test Coverage:**
- 61 methodology verification tests in `tests/test_methodology_callaway.py`
- Existing unit-test coverage in `tests/test_staggered.py`
- R benchmark tests (skip if R not available)

**R Comparison Results:**
- Overall ATT matches within 20% (difference due to dynamic effects in generated data)
- Post-treatment ATT(g,t) values match within 20%
- Pre-treatment effects may differ due to base_period handling differences

**Corrections Made:**
- (None — implementation verified correct)

**Outstanding Concerns:**
- R comparison shows ~20% difference in overall ATT with generated data
  - Likely due to differences in how dynamic effects are handled in data generation
  - Individual ATT(g,t) values match closely for post-treatment periods
  - Further investigation recommended with real-world data
- Pre-treatment ATT(g,t) may differ from R due to base_period="varying" semantics
  - Python uses t-1 as base for pre-treatment
  - R's behavior requires verification

**Deviations from R's did::att_gt():**
1. **NaN for invalid inference**: When SE is non-finite or zero, Python returns NaN for
   t_stat/p_value rather than potentially erroring. This is a defensive enhancement.

**Alignment with R's did::att_gt() (as of v2.1.5):**
1. **Webb weights**: Webb's 6-point distribution with values ±√(3/2), ±1, ±√(1/2)
   uses equal probabilities (1/6 each) matching R's `did` package. This gives
   E[w]=0, Var(w)=1.0, consistent with other bootstrap weight distributions.

   **Verification**: Our implementation matches the well-established `fwildclusterboot`
   R package (C++ source: [wildboottest.cpp](https://github.com/s3alfisc/fwildclusterboot/blob/master/src/wildboottest.cpp)).
   The implementation uses `sqrt(1.5)`, `1`, `sqrt(0.5)` (and negatives) with equal 1/6
   probabilities—identical to our values.

   **Note on documentation discrepancy**: Some documentation (e.g., fwildclusterboot
   vignette) describes Webb weights as "±1.5, ±1, ±0.5". This appears to be a
   simplification for readability. The actual implementations use ±√1.5, ±1, ±√0.5
   which provides the required unit variance (Var(w) = 1.0).

---

#### SunAbraham

| Field | Value |
|-------|-------|
| Module | `sun_abraham.py` |
| Primary Reference | Sun & Abraham (2021) |
| R Reference | `fixest::sunab()` |
| Status | **Complete** |
| Last Review | 2026-02-15 |

**Verified Components:**
- [x] Saturated TWFE regression with cohort × relative-time interactions
- [x] Within-transformation for unit and time fixed effects
- [x] Interaction-weighted event study effects (δ̂_e = Σ_g ŵ_{g,e} × δ̂_{g,e})
- [x] IW weights match event-time sample shares (n_{g,e} / Σ_g n_{g,e})
- [x] Overall ATT as weighted average of post-treatment effects
- [x] Delta method SE for aggregated effects (Var = w' Σ w)
- [x] Cluster-robust SEs at unit level
- [x] Reference period normalized to zero (e=-1 excluded from design matrix)
- [x] R comparison: ATT matches `fixest::sunab()` within machine precision (<1e-11)
- [x] R comparison: SE matches within 0.3% (small scale) / 0.1% (1k scale)
- [x] R comparison: Event study effects correlation = 1.000000
- [x] R comparison: Event study max diff < 1e-11
- [x] Bootstrap inference (pairs bootstrap)
- [x] Rank deficiency handling (warn/error/silent)
- [x] All REGISTRY.md edge cases tested

**Test Coverage:**
- Combined methodology + unit tests in `tests/test_sun_abraham.py` (the methodology verification block grew incrementally from the original 7 review tests as edge cases were added)
- R benchmark tests via `benchmarks/run_benchmarks.py --estimator sunab`

**R Comparison Results:**
- Overall ATT matches within machine precision (diff < 1e-11 at both scales)
- Cluster-robust SE matches within 0.3% (well within 1% threshold)
- Event study effects match perfectly (correlation 1.0, max diff < 1e-11)
- Validated at small (200 units) and 1k (1000 units) scales

**Corrections Made:**
1. **DF adjustment for absorbed FE** (`sun_abraham.py`, `_fit_saturated_regression()`):
   Added `df_adjustment = n_units + n_times - 1` to `LinearRegression.fit()` to account
   for absorbed unit and time fixed effects in degrees of freedom. Unlike TWFE (which uses
   `-2` plus an explicit intercept column), SunAbraham's saturated regression has no
   intercept, so all absorbed df must come from the adjustment. Affects t-distribution DoF
   for cohort-level p-values/CIs (slightly larger p-values, slightly wider CIs) but does
   NOT change VCV or SE values.

2. **NaN return for no post-treatment effects** (`sun_abraham.py`, `_compute_overall_att()`):
   Changed return from `(0.0, 0.0)` to `(np.nan, np.nan)` when no post-treatment effects
   exist. All downstream inference fields (t_stat, p_value, conf_int) correctly propagate
   NaN via existing guards in `fit()`.

3. **Deprecation warnings for unused parameters** (`sun_abraham.py`, `fit()`):
   Added `FutureWarning` for `min_pre_periods` and `min_post_periods` parameters that
   are accepted but never used (no-op). These will be removed in a future version.

4. **Removed event-time truncation at [-20, 20]** (`sun_abraham.py`):
   Removed the hardcoded cap `max(min(...), -20)` / `min(max(...), 20)` to match
   R's `fixest::sunab()` which has no such limit. All available relative times are
   now estimated.

5. **Warning for variance fallback path** (`sun_abraham.py`, `_compute_overall_att()`):
   Added `UserWarning` when the full weight vector cannot be constructed and a
   simplified variance (ignoring covariances between periods) is used as fallback.

6. **IW weights use event-time sample shares** (`sun_abraham.py`, `_compute_iw_effects()`):
   Changed IW weights from `n_g / Σ_g n_g` (cohort sizes) to `n_{g,e} / Σ_g n_{g,e}`
   (per-event-time observation counts) to match the REGISTRY.md formula. For balanced
   panels these are identical; for unbalanced panels the new formula correctly reflects
   actual sample composition at each event-time. Added unbalanced panel test.

7. **Normalize `np.inf` never-treated encoding** (`sun_abraham.py`, `fit()`):
   `first_treat=np.inf` (documented as valid for never-treated) was included in
   `treatment_groups` and `_rel_time` via `> 0` checks, producing `-inf` event times.
   Fixed by normalizing `np.inf` to `0` immediately after computing `_never_treated`.
   Same fix applied to `staggered.py` (`CallawaySantAnna`).

**Outstanding Concerns:**
- **Inference distribution**: Cohort-level p-values use t-distribution (via
  `LinearRegression.get_inference()`), while aggregated event study and overall ATT
  p-values use normal distribution (via `compute_p_value()`). This is asymptotically
  equivalent and standard for delta-method-aggregated quantities. R's fixest uses
  t-distribution at all levels, so aggregated p-values may differ slightly for small
  samples — this is a documented deviation.

**Deviations from R's fixest::sunab():**
1. **NaN for no post-treatment effects**: Python returns `(NaN, NaN)` for overall ATT/SE
   when no post-treatment effects exist. R would error.
2. **Normal distribution for aggregated inference**: Aggregated p-values use normal
   distribution (asymptotically equivalent). R uses t-distribution.

---

#### StackedDiD

| Field | Value |
|-------|-------|
| Module | `stacked_did.py` |
| Primary Reference | Wing, Freedman & Hollingsworth (2024), NBER WP 32054 |
| R Reference | `stacked-did-weights` (`create_sub_exp()` + `compute_weights()`) |
| Status | **Complete** |
| Last Review | 2026-02-19 |

**Verified Components:**
- [x] IC1 trimming: `a - kappa_pre >= T_min AND a + kappa_post <= T_max` (matches R reference)
- [x] IC2 trimming: Three clean control modes (not_yet_treated, strict, never_treated)
- [x] Sub-experiment construction: treated + clean controls within `[a - kappa_pre, a + kappa_post]`
- [x] Q-weights aggregate: treated Q=1, control `Q = (sub_treat_n/stack_treat_n) / (sub_control_n/stack_control_n)` per (event_time, sub_exp) — matches R `compute_weights()`
- [x] Q-weights population: `Q_a = (Pop_a^D / Pop^D) / (N_a^C / N^C)` (Table 1, Row 2)
- [x] Q-weights sample_share: `Q_a = ((N_a^D + N_a^C)/(N^D+N^C)) / (N_a^C / N^C)` (Table 1, Row 3)
- [x] WLS via sqrt(w) transformation (numerically equivalent to weighted regression)
- [x] Event study regression: `Y = α_0 + α_1·D_sa + Σ_{h≠-1}[λ_h·1(e=h) + δ_h·D_sa·1(e=h)] + U` (Eq. 3)
- [x] Reference period e=-1-anticipation normalized to zero (omitted from design matrix)
- [x] Delta-method SE for overall ATT: `SE = sqrt(ones' @ sub_vcv @ ones) / K`
- [x] Cluster-robust SEs at unit level (default) and unit×sub-experiment level
- [x] Anticipation parameter: reference period shifts to e=-1-anticipation, post-treatment includes anticipation periods
- [x] Rank deficiency handling (warn/error/silent via `solve_ols()`)
- [x] Never-treated encoding: both `first_treat=0` and `first_treat=inf` handled
- [x] R comparison: ATT matches within machine precision (diff < 2.1e-11)
- [x] R comparison: SE matches within machine precision (diff < 4.0e-10)
- [x] R comparison: Event study effects correlation = 1.000000, max diff < 4.5e-11
- [x] `safe_inference()` used for all inference fields
- [x] All REGISTRY.md edge cases tested

**Test Coverage:**
- `tests/test_stacked_did.py`: 10 test classes (basic, trimming, Q-weights, clean-control, clustering, stacked-data shape, edge cases, sklearn interface, results methods, validation)
- R benchmark tests via `benchmarks/run_benchmarks.py --estimator stacked`

**R Comparison Results (200 units, 8 periods, kappa_pre=2, kappa_post=2):**
| Metric | Python | R | Diff |
|--------|--------|---|------|
| Overall ATT | 2.277699574579 | 2.2776995746 | 2.1e-11 |
| Overall SE | 0.062045687626 | 0.062045688027 | 4.0e-10 |
| ES e=-2 ATT | 0.044517975379 | 0.044517975379 | <1e-12 |
| ES e=0 ATT | 2.104181683763 | 2.104181683800 | <1e-11 |
| ES e=1 ATT | 2.209990715130 | 2.209990715100 | <1e-11 |
| ES e=2 ATT | 2.518926324845 | 2.518926324800 | <1e-11 |
| Stacked obs | 1600 | 1600 | exact |
| Sub-experiments | 3 | 3 | exact |

**Corrections Made:**
1. **IC1 lower bound and time window aligned with R reference** (`stacked_did.py`,
   `_trim_adoption_events()` and `_build_sub_experiment()`): The paper text specifies
   time window `[a - kappa_pre - 1, a + kappa_post]` (including an extra pre-period),
   but the R reference implementation by co-author Hollingsworth uses
   `[a - kappa_pre, a + kappa_post]`. The extra period had no event-study dummy,
   altering the baseline regression. Fixed to match R: removed `-1` from both
   IC1 check (`a - kappa_pre >= T_min`) and time window start. Discrepancy documented
   in `docs/methodology/papers/wing-2024-review.md` Gaps section.

2. **Q-weight computation: event-time-specific for aggregate weighting** (`stacked_did.py`,
   `_compute_q_weights()`): Changed aggregate Q-weights from unit counts per sub-experiment
   to observation counts per (event_time, sub_exp), matching R reference `compute_weights()`.
   For balanced panels, results are unchanged. For unbalanced panels, weights now adjust for
   varying observation density. Population/sample_share retain unit-count formulas (paper notation).

3. **Anticipation parameter: reference period and dummies** (`stacked_did.py`, `fit()`):
   Reference period now shifts to `e = -1 - anticipation`. Event-time dummies cover the
   full window `[-kappa_pre - anticipation, ..., kappa_post]`. Post-treatment effects include
   anticipation periods. Consistent with ImputationDiD, TwoStageDiD, SunAbraham.

4. **Group aggregation removed** (`stacked_did.py`): `aggregate="group"` and `aggregate="all"`
   removed. The pooled stacked regression cannot produce cohort-specific effects without
   cohort×event-time interactions. Use CallawaySantAnna or ImputationDiD for cohort-level estimates.

5. **n_sub_experiments metadata** (`stacked_did.py`, `fit()`): Now tracks actual built
   sub-experiments, not all events in omega_kappa. Warns if any sub-experiments are empty
   after data filtering.

**Outstanding Concerns:**
- Population/sample_share Q-weights use paper's unit-count formulas (no R reference to validate)
- Anticipation not validated against R (R reference doesn't test anticipation > 0)

**Deviations from R's stacked-did-weights:**
1. **NaN for invalid inference**: Python returns NaN for t_stat/p_value/conf_int when
   SE is non-finite or zero. R would propagate through `fixest::feols()` error handling.

---

#### ImputationDiD

| Field | Value |
|-------|-------|
| Module | `imputation.py`, `imputation_bootstrap.py` |
| Primary Reference | Borusyak, Jaravel & Spiess (2024), *Revisiting Event-Study Designs: Robust and Efficient Estimation*, REStud 91(6) |
| R Reference | `didimputation` |
| Status | **In Progress** |
| Last Review | — |

**Documentation in place:**
- REGISTRY.md section: `## ImputationDiD` (paper-direct equations, edge cases, three-step algorithm)
- Implementation: 87 unit tests in `tests/test_imputation.py` (basic fit, event study, group aggregation, conservative variance, auxiliary partition, unidentified-estimand handling, balanced/unbalanced panels)
- Bootstrap path: `imputation_bootstrap.py` with multiplier-weight resampling
- Survey support: pweight + strata/PSU/FPC via TSL (Phase 6) with PSU-bootstrap path

**Outstanding for promotion:**
- Dedicated `tests/test_methodology_imputation.py` with paper-equation-numbered Verified Components walk-through
- R parity benchmark against `didimputation` (none on file)
- Formal enumeration of deviations from `didimputation` (NaN inference, refused-to-estimate behavior for unidentified estimands per Proposition 5)
- "Corrections Made" listing for any implementation fixes uncovered during the walk-through

---

#### TwoStageDiD

| Field | Value |
|-------|-------|
| Module | `two_stage.py`, `two_stage_bootstrap.py` |
| Primary Reference | Gardner (2022), *Two-stage differences in differences*, arXiv:2207.05943 |
| R Reference | `did2s` |
| Status | **In Progress** |
| Last Review | — |

**Documentation in place:**
- REGISTRY.md section: `## TwoStageDiD` (Stage 1 unit+time FE on untreated, Stage 2 OLS on residualized outcomes, GMM sandwich variance per Newey-McFadden Theorem 6.1)
- Implementation: 76 unit tests in `tests/test_two_stage.py` (matches ImputationDiD point estimates, R `did2s` global `(D'D)^{-1}` variance, always-treated unit exclusion, multiplier bootstrap)
- Documented R alignment: uses global `(D'D)^{-1}` matching `did2s` (not paper Eq. 6)

**Outstanding for promotion:**
- Dedicated `tests/test_methodology_two_stage.py` with paper-equation-numbered Verified Components walk-through
- R parity benchmark fixture against `did2s` (none on file)
- Documented deviation: Newey-McFadden Theorem 6.1 sandwich vs paper's Eq. 6 (already noted in REGISTRY but not formalized in this tracker)
- "Corrections Made" listing

---

#### WooldridgeDiD (ETWFE)

| Field | Value |
|-------|-------|
| Module | `wooldridge.py`, `wooldridge_results.py` |
| Primary Reference | Wooldridge (2025), *Two-way fixed effects, the two-way Mundlak regression, and difference-in-differences estimators*, Empirical Economics 69(5), 2545–2587 |
| Companion Reference | Wooldridge (2023), *Simple approaches to nonlinear difference-in-differences with panel data*, Econometrics Journal 26(3) (nonlinear extensions for logit/Poisson paths) |
| R Reference | `etwfe` (McDermott 2023); Stata `jwdid` (Rios-Avila 2021) |
| Status | **Complete** |
| Last Review | 2026-05-22 |

**Verified Components:**
- **Theorem 3.1 (Mundlak ≡ TWFE):** equivalence under non-singularity Eq. 3.3 — `tests/test_methodology_wooldridge.py::TestW2025Theorem31MundlakTWFEEquivalence`
- **Proposition 5.1 / 5.2 (Imputation ≡ POLS five-way chain):** `TestW2025Proposition51ImputationPOLSEquivalence`
- **Section 6 / Eqs. 6.1-6.5 event-study:** `TestW2025Section6EventStudy`
- **Section 7 aggregation paths (Eqs. 7.2-7.4 + 7.6):** opt-in `weights="cohort_share"` on `aggregate()` recovers paper Eq. 7.4 simple-overall and Eq. 7.6 event-time hand-calc forms — `TestW2025Section7AggregationPaths`
- **Section 8 / Eq. 8.1 heterogeneous cohort-specific trends:** `cohort_trends=True` adds `dg_i · t` interactions; recovers `tau` under heterogeneous-trends DGP — `TestW2025Section8HeterogeneousTrends`
- **Section 10 unbalanced panels + time-varying covariates (Eq. 10.1-10.6):** `TestW2025Section10UnbalancedPanels`

**Test Coverage:**
- `tests/test_methodology_wooldridge.py` — 10 test classes (6 paper-equation-numbered Theorem/Proposition/Section walk-throughs + `TestW2025LibraryDeviations` consolidating 5 surviving deviations + `TestWooldridgeParityR` vcov_type R-parity from PR #483 + `TestWooldridgeParityRPoisson` / `TestWooldridgeParityRLogit` surface tests with log-link goldens; numerical R-parity for nonlinear paths deferred per TODO row)
- `tests/test_wooldridge.py` — unit-level test suite covering OLS / logit / Poisson + four aggregations + survey support + vcov_type variants + cluster/bootstrap interactions
- `benchmarks/R/generate_wooldridge_golden.R` — clubSandwich + sandwich + etwfe goldens at `benchmarks/data/wooldridge_golden.json`

**Corrections Made:**
- **PR #484 (PR-A):** Added primary-source review for Wooldridge (2025) at `docs/methodology/papers/wooldridge-2025-review.md` (771 lines). Documented the cohort-share aggregation deviation (Eqs. 7.2-7.4 simple-overall AND Eq. 7.6 event-time) and the Section 8 heterogeneous-trends gap. REGISTRY § Aggregations Note + TODO row 95 extended to cover both paths.
- **PR-B (this PR):** Closed two paper gaps documented in PR-A:
  - **Opt-in cohort-share aggregation weighting** via `aggregate(weights="cohort_share")` on `WooldridgeDiDResults` (paper Eq. 7.4 simple-overall + Eq. 7.6 event-time). Default stays `weights="cell"` for `jwdid_estat` back-compat.
  - **Heterogeneous cohort trends** via `WooldridgeDiD(cohort_trends=True)` (paper Eq. 8.1; OLS path only; auto-routes to full-dummy mode regardless of `vcov_type` to keep math closure verified against existing R-parity goldens).
  - Extended R goldens to include `etwfe(family="poisson")` and `etwfe(family="logit")` log-link coefficients (surface tests in Python; numerical response-scale parity deferred to follow-up).

**Deviations from the paper / from R / library extensions:** See REGISTRY.md `## WooldridgeDiD (ETWFE)` → `### Deviations from the paper / from R / library extensions` block for the consolidated list (HC1 finite-sample factor, QMLE sandwich `(n-1)/(n-k)` term, nonlinear-vs-fixest direct QMLE, logit cohort+time additive dummies, anticipation + aggregation, cell-count default with opt-in cohort-share).

**Outstanding Concerns:**
- **Stata `jwdid` golden values** (TODO "Stata `jwdid` golden value tests" row): Stata-side parity infrastructure deferred until Stata install is available; R `etwfe` side covered in PR-B Stage D.
- **Response-scale APE / log-link bridge for Poisson + logit R parity** (new TODO row added in PR-B): direct cell-level numerical parity between diff-diff's response-scale ATT and R `etwfe` log-link coefficients requires either `emfx()`-based APE extraction on the R side or link-function inversion with baseline-mean adjustment.
- **QMLE sandwich Stata-parity `qmle` weight type** (TODO row 94): diff-diff's `(G/(G-1)) × ((n-1)/(n-k))` is conservative vs Stata's `G/(G-1)` only; awaiting Stata golden values to confirm material difference.
- **Repeated cross-sections** (paper p. 2581 → Deb et al. 2024): not in 2025 paper's main body; future PR.
- **Treatment exit / non-absorbing treatment** (2023 paper Section 7.2 sketch): not in 2025 paper; future PR.
- **`cohort_trends` polynomial extension** (`"quadratic"`, `"cubic"`): PR-B ships binary `True/False` for linear `dg_i · t`; forward-extensibility deferred.

---

#### EfficientDiD

| Field | Value |
|-------|-------|
| Module | `efficient_did.py`, `efficient_did_bootstrap.py`, `efficient_did_covariates.py`, `efficient_did_weights.py` |
| Primary Reference | Chen, Sant'Anna & Xie (2025), *Efficient Difference-in-Differences and Event Study Estimators* |
| R Reference | (no canonical R package; paper compares against `did` / `DIDmultiplegt` / BJS / Gardner / Wooldridge as benchmarks rather than providing a reference implementation) |
| Status | **In Progress** |
| Last Review | — |

**Documentation in place:**
- REGISTRY.md section: `## EfficientDiD` (full Theorem 4.1 EIF, sieve-based propensity-ratio estimation with AIC/BIC, kernel-smoothed conditional covariance, Hausman pretest for PT-All vs PT-Post, survey support)
- Implementation: 130 unit tests in `tests/test_efficient_did.py` + 12 validation tests in `tests/test_efficient_did_validation.py`
- Hausman pretest: implemented per Theorem A.1 with Moore-Penrose pseudoinverse for finite-sample non-PSD variance-difference matrix
- Survey support: pweight + strata/PSU/FPC via TSL on EIF scores; covariates DR path with WLS outcome regression and weighted sieve normal equations

**Outstanding for promotion:**
- **No paper review on file** under `docs/methodology/papers/` — write one
- Dedicated `tests/test_methodology_efficient_did.py` with Theorem 3.2 / Equation 3.5 / Equation 4.3 numbered Verified Components walk-through
- Cross-language anchor: the paper's empirical replication uses HRS data following Sun-Abraham (2021); a same-data benchmark against the paper's reported numbers (or a same-DGP MC against R alternatives) would substantiate the EIF construction
- Documented deviations: linear OLS working models for outcome regressions vs. paper's general nonparametric specification (DR safety net acknowledged but not separately validated); fixed-weight bootstrap aggregation vs. WIF-corrected analytical aggregation

---

### Continuous & Universal-Treatment Estimators

#### ContinuousDiD

| Field | Value |
|-------|-------|
| Module | `continuous_did.py`, `continuous_did_bspline.py`, `continuous_did_results.py` |
| Primary Reference | Callaway, Goodman-Bacon & Sant'Anna (2024), *Difference-in-Differences with a Continuous Treatment*, NBER WP 32117 |
| R Reference | `contdid` v0.1.0 (CRAN) — two parity surfaces at relative tolerance: (a) **scalar overall ATT parity** with raw R `cont_did` / `pte_default` output at `< 0.01` (1%) on all 6 benchmarks; **scalar overall ACRT parity** with raw R `cont_did` at `< 0.01` (1%) on benchmarks 4-5; (b) **harmonized boundary-knot-normalized curve parity** with R-side ATT(d)/ACRT(d) reconstructed under `Boundary.knots = range(treated_doses)` (matching the library) at `< 0.01` max ATT(d) and `< 0.02` max ACRT(d) on benchmarks 1-3 via the benchmark harness (`_run_r_contdid` rebuilds the R-side basis under `Boundary.knots = range(treated_doses)` at `tests/test_methodology_continuous_did.py:333-367`; `_compare_with_r` orchestrates the Python-vs-R comparison at `:395-459`); benchmark 6 is event-study, scalar `overall_att` only (binarized ATT, no curve comparison and no ACRT in event-study mode). Surface (a) is direct raw-package parity; surface (b) is reconstructed-basis parity because raw `contdid` curves use `range(dvals)` instead of `range(dose)`. NOT bit-exact (`atol=1e-8`) like HAD because of the boundary-knots deviation documented below. See `tests/test_methodology_continuous_did.py::TestRBenchmark` |
| Status | **Complete** |
| Last Review | 2026-05-20 |

**Verified Components:**
- [x] **PT and SPT identification** (CGBS 2024 Assumptions 1-2) — two-level parallel trends with explicit untreated-and-doses conditioning; estimands `ATT(d|d)`, `ATT(d)`, `ACRT(d)`, `ATT^{loc}`, `ATT^{glob}`, `ACRT^{glob}` defined in `docs/methodology/continuous-did.md` § 4 + REGISTRY `## ContinuousDiD` Identification block. Hand-calc coverage: `tests/test_methodology_continuous_did.py::TestLinearDoseResponse` (4 tests at `atol=1e-10` / `atol=1e-6` on no-noise linear DGP — locks the `ATT^{glob}` binarization formula `E[ΔY | D > 0] − E[ΔY | D = 0]`, the `ACRT^{glob}` plug-in average, and the `ATT(d) = 2d`, `ACRT(d) = 2` closed forms).
- [x] **B-spline basis matching `splines2::bSpline`** (cubic and linear degrees, `num_knots=0` default; global boundary knots from the training-dose range, NOT per-cell) — `tests/test_methodology_continuous_did.py::TestQuadraticWithCubicBasis::test_quadratic_recovery` recovers `ATT(d) = d²` at `atol=1e-6` via a degree-3 basis (cubic spline can represent quadratic exactly). The matching basis algorithm lives in `diff_diff/continuous_did_bspline.py` (216 LoC); the boundary-knots deviation from R `contdid` is documented in the Deviations block below.
- [x] **Multi-period (g,t) cell iteration with base period selection** — `TestMultiPeriodAggregation::test_multiple_groups` and `test_gt_cell_count` exercise the cohort iteration on 2-cohort staggered panels; cell counts agree with the R `ptetools`-style convention. Scalar parity with raw R `cont_did` at 1% relative further locks the staggered-aggregation surface via `TestRBenchmark::test_benchmark_4_staggered_dose` and `test_benchmark_5_not_yet_treated` (both assert overall ATT AND overall ACRT at `< 0.01`).
- [x] **Dose-response (`aggregate="dose"`) and event-study (`aggregate="eventstudy"`) aggregation** with group-proportional weights (`n_treated/n_total` per group, divided among post-treatment cells; matches R `ptetools` convention). Two R-side surfaces are exercised: (a) **scalar `overall_att`** via `TestRBenchmark::test_benchmark_1_basic_cubic` / `_2_linear` / `_3_interior_knots` / `_4_staggered_dose` / `_5_not_yet_treated` (dose mode) and `_6_event_study` (event-study mode — binarized ATT only; benchmark 6 validates the event-study code path through the scalar surface, NOT per-horizon `event_study_effects`); (b) **harmonized boundary-knot-normalized ATT(d) / ACRT(d) curves** on benchmarks 1-3 via the benchmark harness — `_run_r_contdid` at `tests/test_methodology_continuous_did.py:333-367` rebuilds the R-side basis under `Boundary.knots = range(treated_doses)` (raw `contdid` curves use `range(dvals)`, so this is reconstructed-basis parity not raw-package parity), and `_compare_with_r` orchestrates the comparison at `:395-459`. Per-benchmark tolerances: all 6 assert overall ATT at `< 0.01` (1%); benchmarks 1-3 additionally assert max ATT(d) at `< 0.01` and max ACRT(d) at `< 0.02` via the helper; benchmarks 4-5 assert overall ACRT at `< 0.01` inline. Per-horizon `event_study_effects` estimates and inference are exercised by Python-side tests at `tests/test_continuous_did.py:557-690` and `:1500-1528` (no R cross-language comparison on the per-horizon surface). Skipped if R / `contdid` not installed via `_check_r_contdid()`; benchmarks use R's `dvals` for exact evaluation-grid alignment between Python and R outputs (boundary knots are harmonized separately under surface (b) — see the `_run_r_contdid` helper's `Boundary.knots = range(treated_doses)` block at `tests/test_methodology_continuous_did.py:333-367`).
- [x] **Multiplier bootstrap for inference** (PSU-level multiplier weights on the survey path per Phase 6) — implementation in `diff_diff/continuous_did.py`; bootstrap SE invariant on rank-deficient cells locked in `TestEdgeCasesMethodology::test_all_same_dose` (verifies `dose_response_att.se` is finite on a heterogeneous-outcome / identical-dose DGP); 80 unit tests in `tests/test_continuous_did.py` exercise the rest of the bootstrap path.
- [x] **Analytical SEs via influence functions** (NOT delta method; corrected post-v3.0.0, see Corrections Made) — IF-based variance with `safe_inference()` joint-NaN consistency on all six estimand fields (`overall_att`, `overall_acrt`, dose-response, event-study).
- [x] **Survey support**: weighted B-spline OLS, two-stage linearization (TSL) on influence functions, bootstrap + survey via PSU-level multiplier weights (Phase 3 + Phase 6). Boxed in REGISTRY `## ContinuousDiD` → Implementation Checklist → "Survey design support (Phase 3)" item.
- [x] **`+inf` → `0` never-treated recoding** with `UserWarning` reporting the affected row count (axis-E silent-coercion fix per Phase 2 audit) — the R-style convention of `first_treat = +inf` is normalized internally but no longer absorbed silently. **Any negative `first_treat` value (including `-inf`) raises `ValueError`** with the affected row count. Locked in `tests/test_continuous_did.py`.
- [x] **Zero-`first_treat` rows with nonzero `dose` force-zeroed** with `UserWarning` reporting the affected row count (axis-E silent-coercion fix per Phase 2 audit) — never-treated cells must have `D=0` for internal consistency; the previous silent zeroing is now signaled. Locked in `tests/test_continuous_did.py`.
- [x] **`bspline_derivative_design_matrix` derivative-construction failure warning** (Phase 2 axis-C #12 silent-failures audit fix) — aggregates failed basis indices into a single `UserWarning` naming them, instead of swallowing `scipy.interpolate.BSpline.ValueError` and leaving silently zeroed derivative columns. Both ACRT point estimates AND analytical/bootstrap inference read the same `dPsi` matrix (`continuous_did.py:1026-1046` and the bootstrap ACRT path at `continuous_did.py:1524-1561`), so both are biased on partial-derivative failure — the warning wording makes that explicit. The all-identical-knot degenerate case (single dose value) remains silently handled because derivatives are mathematically zero there. Locked in `tests/test_continuous_did.py::TestBSplineDerivativeDegenerateBasis` (3 tests: `test_single_dose_is_silent`, `test_valueerror_from_bspline_emits_aggregate_warning`, `test_clean_knots_emit_no_warning`); source-level aggregate-warning block at `diff_diff/continuous_did_bspline.py:150-187`.
- [x] **Edge cases**: all-same-dose (rank-deficient design, recovers only intercept = `ATT^{glob}`, ACRT = 0 everywhere), single-treated-unit (insufficient for OLS, raises `ValueError` "No valid"), discrete-treatment (detected and warned, saturated regression deferred), rank-deficiency per cell (cell skipped under `rank_deficient_action="silent"` / `"warn"`), balanced-panel-required (matches R `contdid` v0.1.0). Locked in `TestEdgeCasesMethodology` (2 methodology tests) + rank-deficient unit tests in `test_continuous_did.py`.
- [x] **Anticipation-aware not-yet-treated control mask**: when `anticipation > 0`, the not-yet-treated control mask uses `G > t + anticipation` (not just `G > t`) to exclude cohorts in the anticipation window from controls. When `anticipation=0` (default), behavior is unchanged. CHANGELOG `[3.0.x]`-era fix; locked in `test_continuous_did.py`.

**Test Coverage:**
- 15 methodology tests in `tests/test_methodology_continuous_did.py` (5 classes: 4 + 1 + 2 + 2 + 6); the R-benchmark class (6 tests) skips if R / `contdid` v0.1.0 is not installed via `_check_r_contdid()` guard.
- 80 core unit tests in `tests/test_continuous_did.py` (1,530 LoC) covering the B-spline basis (`TestBSplineBasis`, `TestBSplineDerivativeDegenerateBasis`), bootstrap, IF-based analytical SE, anticipation, rank-deficient cells, dose grid, dvals/grid validation, `+inf` recoding, zero-dose coercion, and result-class field contracts. **Survey-design coverage is NOT in this file** — it lives in the dedicated survey suites (next bullet).
- ContinuousDiD survey-design tests: `tests/test_survey_phase3.py::TestContinuousDiDSurvey` (`tests/test_survey_phase3.py:653-705` analytical SE + bootstrap; `:1368-1407` event-study aggregation + survey-design rejection paths) and `tests/test_survey_phase6.py` (`:1230-1244` replicate-weight + n_bootstrap rejection; `:1548-1610` positive-weight-gate cell skipping).
- R cross-language coverage at relative tolerance (NOT bit-exact — see Deviations § "Boundary knots") on 6 benchmark configurations across two surfaces: (a) **scalar parity with raw R `cont_did` / `pte_default`** — all 6 assert overall ATT at `< 0.01` (1%); benchmarks 4-5 also assert overall ACRT at `< 0.01` inline; benchmark 6 is event-study mode with scalar `overall_att` only (binarized ATT, no per-horizon and no ACRT comparison). Per-horizon `event_study_effects` is exercised by Python-side tests at `tests/test_continuous_did.py:557-690` and `:1500-1528`. (b) **harmonized boundary-knot-normalized curve parity** with R-side ATT(d) / ACRT(d) reconstructed under `Boundary.knots = range(treated_doses)` (matching the library) on benchmarks 1-3 via the benchmark harness (`_run_r_contdid` does the R-side rebuild at `tests/test_methodology_continuous_did.py:333-367`; `_compare_with_r` orchestrates at `:395-459`) — max ATT(d) at `< 0.01` and max ACRT(d) at `< 0.02`. Surface (a) is direct raw-package parity; surface (b) is reconstructed-basis parity because raw `contdid` curves use `range(dvals)`.
- Documentation: `docs/methodology/continuous-did.md` (14,885 bytes theory note covering PT vs SPT, estimands, B-spline OLS, multiplier bootstrap).

**Corrections Made:**
1. **SE method correction (early v3.0.x):** ContinuousDiD originally computed SEs via delta method; corrected to influence-function-based variance. See CHANGELOG entries "Fix ContinuousDiD SE method: influence function, not delta method" + "Fix methodology doc: influence functions, not delta method for ContinuousDiD SEs".
2. **Anticipation-aware control mask** (CHANGELOG `[3.0.x]`-era): not-yet-treated control mask now uses `G > t + anticipation` instead of `G > t`, excluding cohorts in the anticipation window from controls.
3. **Phase 2 silent-failures audit fixes** (axis-C + axis-E):
   - **axis-C #12:** `bspline_derivative_design_matrix` no longer swallows `scipy.interpolate.BSpline.ValueError` silently; aggregates failed basis indices into a single `UserWarning`. Both ACRT point estimates AND analytical/bootstrap inference are affected when this fires.
   - **axis-E (silent coercion):** `+inf` → `0` never-treated recoding now emits `UserWarning` with affected row count; negative `first_treat` (including `-inf`) raises `ValueError`. Zero-`first_treat` rows with nonzero `dose` force-zeroed now also emit `UserWarning`.
4. **Bread normalization, fweight TSL scaling, weighted-mass IF linearization** (CHANGELOG): three pieces of the IF-based variance machinery on the survey path corrected to match the analytical identities. Replicate-IF variance score scaling also fixed for EfficientDiD / TripleDifference / ContinuousDiD as part of the same sweep.
5. **Tracker-promotion consolidation (this PR, 2026-05-20):** formal Deviations block added to REGISTRY `## ContinuousDiD` consolidating the boundary-knots deviation, the `bspline_derivative` warning, and the two axis-E silent-coercion warnings into a single labeled surface. The original Edge Cases / Notes entries remain in place — Deviations is an additional canonical surface (per CLAUDE.md "Documenting Deviations (AI Review Compatibility)" labels).

**Deviations from the paper / from R / library extensions:**
1. **Deviation from R — boundary knots use `range(dose)` not `range(dvals)`** — knots are built once from all treated doses (global, not per-cell) to ensure a common basis across (g,t) cells for aggregation. The evaluation grid is clamped to training-dose boundary knots (`range(dose)`). R's `contdid` v0.1.0 has an inconsistency where `splines2::bSpline(dvals)` uses `range(dvals)` instead of `range(dose)`, which can produce extrapolation artifacts at dose-grid extremes. **Scope caveat:** R cross-language coverage therefore runs at **relative** tolerance bands across two surfaces, NOT bit-exact (`atol=1e-8`) like HAD — `contdid` and ContinuousDiD cannot bit-match on aggregated dose-response or ACRT curves because they use different knot placement; the agreement band reflects the boundary-knot divergence rather than algorithmic drift. (a) **Scalar parity with raw R `cont_did` / `pte_default`** at 1% relative on overall ATT for all 6 benchmarks and on overall ACRT for benchmarks 4-5 (benchmark 6 is event-study, scalar `overall_att` only). (b) **Harmonized boundary-knot-normalized curve parity** with R-side ATT(d) / ACRT(d) reconstructed under `Boundary.knots = range(treated_doses)` (matching the library) on benchmarks 1-3 via the benchmark harness (`_run_r_contdid` does the R-side rebuild at `tests/test_methodology_continuous_did.py:333-367`; `_compare_with_r` orchestrates at `:395-459`) — max ATT(d) at 1% and max ACRT(d) at 2%. The slightly wider 2% ACRT(d)-curve tolerance on benchmarks 1-3 reflects the tighter coupling between basis derivative numerics and the boundary-knot choice; benchmarks 4-5 use overall scalars (`overall_acrt`) where the boundary effect averages down to 1%. Library extension toward methodological soundness (avoids extrapolation).
2. **Library extension — `bspline_derivative_design_matrix` derivative-failure warning** — previously swallowed `scipy.interpolate.BSpline.ValueError` in the per-basis derivative loop, leaving affected derivative-matrix columns silently zero. Now aggregates the failed basis indices into a single `UserWarning` naming them. Both ACRT point estimates and analytical/bootstrap inference read the same `dPsi` matrix, so both are biased when this fires — the warning wording makes that explicit. The all-identical-knot degenerate case (single dose value) remains silently handled (mathematically-zero derivatives are correct there). Phase 2 axis-C #12 silent-failures audit fix. No R correspondence; `contdid` v0.1.0 does not implement an equivalent warning.
3. **Library extension — `+inf` → `0` never-treated recoding warns** — the R-style convention of coding never-treated units as `first_treat=+inf` is still accepted and normalized to `first_treat=0` internally, but the estimator now emits a `UserWarning` reporting the row count so the silent recategorization is surfaced. Only `+inf` is recoded (matching the R convention). Any **negative** `first_treat` value (including `-inf`) raises `ValueError` with the row count, since such units would otherwise silently fall out of both the treated (`g > 0`) and never-treated (`g == 0`) masks. Pass `0` directly for never-treated units to avoid the warning. Library extension toward stricter safety; matches the broader Phase 2 axis-E silent-coercion convention. No R correspondence; `contdid` v0.1.0 silently absorbs `+inf` without a signal.
4. **Library extension — zero-`first_treat` rows with nonzero `dose` force-zeroed with warning** — never-treated cells must have `D=0` for internal consistency in the dose-response. The estimator now emits a `UserWarning` with the affected row count before the zeroing, so unintended nonzero doses on never-treated rows are no longer absorbed without a signal. Library extension toward stricter safety with no R correspondence — `contdid` v0.1.0 has the same `first_treat = 0` → `D = 0` invariant requirement but silently coerces without a warning; same axis-E silent-coercion lineage as #3.

**Outstanding Concerns:**
- **Covariate support (deferred)** — `covariates=` kwarg is not implemented; matches R `contdid` v0.1.0 which also has no covariate support. Tracked as a future-work row in TODO.md (Low priority).
- **Discrete-treatment saturated regression (deferred)** — when `dose` is detected as integer-valued, the estimator currently warns; the saturated regression approach (one coefficient per discrete dose level instead of B-spline basis) is not implemented. Tracked as a future-work row.
- **Lowest-dose-as-control (Remark 3.1, deferred)** — CGBS 2024 Remark 3.1 outlines using the lowest non-zero dose as the comparison group when `P(D=0) = 0`. Not implemented; the estimator requires `P(D=0) > 0` (never-treated controls present). Tracked as a future-work row.

These three are feature deferrals (paper-supported extensions that the library has chosen not to implement yet), not tracker blockers — REGISTRY `## ContinuousDiD` → Implementation Checklist already marks them as `[ ]` deferred (the "Covariate support", "Discrete treatment saturated regression", and "Lowest-dose-as-control (Remark 3.1)" items). They mirror the same "future work" status that the ChaisemartinDHaultfoeuille and TROP tracker rows carry for analogous optional extensions.

---

#### ChaisemartinDHaultfoeuille (DCDH)

| Field | Value |
|-------|-------|
| Module | `chaisemartin_dhaultfoeuille.py`, `chaisemartin_dhaultfoeuille_bootstrap.py`, `chaisemartin_dhaultfoeuille_results.py` |
| Primary References | (a) de Chaisemartin & D'Haultfœuille (2020), *Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects*, AER 110(9), 2964-2996. (b) de Chaisemartin & D'Haultfœuille (2022, revised July 2023), *Difference-in-Differences Estimators of Intertemporal Treatment Effects*, NBER WP 29873 — Web Appendix Section 3.7.3 for cohort-recentered plug-in variance. (Matches `docs/methodology/REGISTRY.md` `## ChaisemartinDHaultfoeuille` § "Primary sources". The Knau et al. 2026 universal-rollout paper, while authored by overlapping authors, is the primary source for `HeterogeneousAdoptionDiD` and is treated as adjacent context for DCDH, not a primary reference — see "Outstanding Concerns" below.) |
| R Reference | `DIDmultiplegtDYN` |
| Status | **Complete** |
| Last Review | 2026-05-21 |

**Verified Components:**
- [x] **AER 2020 Theorem 3** — per-period `DID_{+,t}` / `DID_{-,t}` plus aggregate `DID_M`, `DID_+`, `DID_-` — `tests/test_methodology_chaisemartin_dhaultfoeuille.py::TestMethodologyWorkedExample` (hand-calculable 4-group example: `DID_M = 2.5`, `DID_+ = 2.0`, `DID_- = 3.0` exact); paper review at `docs/methodology/papers/dechaisemartin-dhaultfoeuille-2020-review.md`.
- [x] **AER 2020 single-lag placebo `DID_M^pl`** — same Theorem 3 logic applied to `Y_{g,t-1} - Y_{g,t-2}` on 3-period cells — covered by the worked example class and `tests/test_chaisemartin_dhaultfoeuille.py::TestA11Handling` for the placebo Assumption 11 zero-retention path.
- [x] **AER 2020 Theorem 1 TWFE-weights diagnostic** — negative-weight detection on binary treatment via `twfe_diagnostic=True` and standalone `twowayfeweights()` — `tests/test_methodology_chaisemartin_dhaultfoeuille.py::TestTWFEDiagnostic` + `tests/test_chaisemartin_dhaultfoeuille.py::TestTwowayFeweightsHelper`; binary-only contract documented (non-binary inputs trigger `UserWarning` from `fit()` and `ValueError` from the standalone helper).
- [x] **NBER WP 29873 dynamic event study `DID_l`** (Equation 3 / 5 of the dynamic paper) — `tests/test_methodology_chaisemartin_dhaultfoeuille.py::TestCohortRecenteringCritical` + `TestLargeNRecovery`; paper review at `docs/methodology/papers/dechaisemartin-dhaultfoeuille-2022-review.md`. The `TestLargeNRecovery` class verifies that the multi-horizon estimator recovers the true ATT at large G.
- [x] **NBER WP 29873 dynamic placebos `DID^{pl}_l`, normalized `DID^n_l`, cost-benefit `delta`** (Lemma 4) — `tests/test_chaisemartin_dhaultfoeuille.py::TestMultiHorizonPlacebos`, `TestNormalizedEffects`, `TestCostBenefitDelta`, `TestSupTBands` (simultaneous confidence bands).
- [x] **NBER WP 29873 Web Appendix Section 3.7.3 cohort-recentered plug-in variance** — locked by `tests/test_methodology_chaisemartin_dhaultfoeuille.py::TestCohortRecenteringCritical::test_cohort_recentering_not_grand_mean` (constructs a designed DGP where cohort recentering and grand-mean recentering produce materially different SE and asserts they diverge — guards against silent regression to a single-mean centering).
- [x] **R `DIDmultiplegtDYN` parity at documented tolerance bands** — `tests/test_chaisemartin_dhaultfoeuille_parity.py` (26 tests). Tolerance class constants: `POINT_RTOL = 1e-4` (pure-direction point estimates), `MIXED_POINT_RTOL = 0.025` (2.5% on mixed-direction panels), `PURE_DIRECTION_SE_RTOL = 0.05` (5% on pure-direction SE after the Round 2 full-IF fix), `SE_RTOL = 0.10` (10% on multi-horizon SE), and `se_rtol=0.15` on the `joiners_only_long_multi_horizon` L_max=5 scenario where cell-count-weighting compounds across horizons. Deviations from R are documented in the consolidated REGISTRY Deviations block (D2 period-vs-cohort + D4 SE-normalization explain the residual SE gap).
- [x] **Phase 3 covariate adjustment (`DID^X`)** — Web Appendix Section 1.2 residualization-style adjustment with first-stage OLS on first-differenced covariates with time FEs, restricted to not-yet-treated observations — `tests/test_chaisemartin_dhaultfoeuille.py::TestCovariateAdjustment`.
- [x] **Phase 3 group-specific linear trends (`DID^{fd}`)** — Web Appendix Section 1.3 / Lemma 6, Z_mat first-differencing — `tests/test_chaisemartin_dhaultfoeuille.py::TestLinearTrends`.
- [x] **Phase 3 state-set-specific trends** — Web Appendix Section 1.4 control-pool restriction — `tests/test_chaisemartin_dhaultfoeuille.py::TestStateSetTrends`.
- [x] **Phase 3 heterogeneity testing** — Web Appendix Section 1.5 / Lemma 7 saturated-OLS test for treatment-effect heterogeneity along an interaction variable — `tests/test_chaisemartin_dhaultfoeuille.py::TestHeterogeneityTesting`.
- [x] **Design-2 switch-in/switch-out descriptive wrapper** — Web Appendix Section 1.6 — `tests/test_chaisemartin_dhaultfoeuille.py::TestDesign2`.
- [x] **`by_path` per-path event-study disaggregation** — `tests/test_chaisemartin_dhaultfoeuille.py::TestByPathGates` / `TestByPathBehavior` / `TestByPathEdgeCases` / `TestByPathBootstrap` / `TestByPathPlacebo` / `TestByPathSupTBands` / `TestByPathControls` / `TestByPathTrendsLinear` (~8 classes covering API gates, point-estimate path, bootstrap composition, placebo composition, sup-t bands, covariates, and linear trends).
- [x] **HonestDiD (Rambachan-Roth 2023) integration** on placebo + event study surface — `tests/test_chaisemartin_dhaultfoeuille.py::TestHonestDiDIntegration`.
- [x] **Non-binary (ordinal or continuous) treatment** — paper Section 2 of the dynamic companion defines treatment as a general `D_{g,t}`; binary `{0, 1}` is a special case — `tests/test_chaisemartin_dhaultfoeuille.py::TestNonBinaryTreatment`.
- [x] **Survey design support: Taylor-series linearization + replicate weights + Hall-Mammen wild PSU bootstrap** — `tests/test_survey_dcdh.py` (Binder TSL on the main ATT, DID^X, heterogeneity, TWFE diagnostic, and HonestDiD surfaces), `tests/test_survey_dcdh_replicate_psu.py` (Rao-Wu rescaled replicate weights for BRR/Fay/JK1/JKn/SDR), and three cell-period coverage suites (`tests/test_dcdh_cell_period_coverage.py`, `tests/test_dcdh_bootstrap_cell_period_coverage.py`, `tests/test_dcdh_heterogeneity_cell_period_coverage.py`) — the cell-period allocator's per-cell IF expansion is what enables within-group-varying PSU.
- [x] **Two primary-source DCDH paper reviews on file**: `docs/methodology/papers/dechaisemartin-dhaultfoeuille-2020-review.md` (2020 AER) and `docs/methodology/papers/dechaisemartin-dhaultfoeuille-2022-review.md` (2022/2023 NBER WP 29873). The adjacent `docs/methodology/papers/dechaisemartin-2026-review.md` is on disk as the primary source for `HeterogeneousAdoptionDiD` (HAD's universal-rollout case) and is referenced from DCDH as context only; it is not DCDH primary-source coverage.

**Test Coverage:**
- 12 methodology tests in `tests/test_methodology_chaisemartin_dhaultfoeuille.py` (4 classes: `TestMethodologyWorkedExample`, `TestCohortRecenteringCritical`, `TestTWFEDiagnostic`, `TestLargeNRecovery`).
- 26 R-parity tests in `tests/test_chaisemartin_dhaultfoeuille_parity.py` against `DIDmultiplegtDYN`.
- 352 unit tests in `tests/test_chaisemartin_dhaultfoeuille.py` covering Phase 1 + Phase 2 + Phase 3 + survey-design + by-path + HonestDiD surfaces (37 test classes).
- Survey-specific: `tests/test_survey_dcdh.py`, `tests/test_survey_dcdh_replicate_psu.py`, plus three dCDH cell-period coverage suites (`test_dcdh_cell_period_coverage.py`, `test_dcdh_bootstrap_cell_period_coverage.py`, `test_dcdh_heterogeneity_cell_period_coverage.py`).
- Two primary-source DCDH paper reviews on disk: 2020 AER and 2022/2023 NBER WP 29873 (see Verified Components above). The adjacent `docs/methodology/papers/dechaisemartin-2026-review.md` is on disk but is HAD's primary source, not DCDH's — it does not count toward DCDH primary-source coverage.

**Corrections Made:**
1. **Round 2 full-IF fix** (pre-promotion): never-switching groups now participate in the variance via stable-control roles under the full `Lambda^G_{g,l=1}` influence function. The `n_groups_dropped_never_switching` results field is retained for backwards compatibility but no longer represents an actual exclusion. After this fix, SE parity vs R on pure-direction scenarios narrowed from ~18% to ~3% (documented in REGISTRY `## ChaisemartinDHaultfoeuille` § "Note (deviation from R DIDmultiplegtDYN):" on period-vs-cohort control sets).
2. **PR-A consolidation (PR #478, 2026-05-21):** REGISTRY `## ChaisemartinDHaultfoeuille` reframed to clarify that the library's equal-cell weighting is a documented deviation from BOTH the AER 2020 Equation 3 (`N_{d,d',t} = sum_g N_{g,t}` observation sums) AND R `DIDmultiplegtDYN` (cell-size weighting); the prior framing called the Python contract "paper-literal", which was incorrect against the main-text formula. The period-vs-cohort Note was tightened to use the AER 2020's "transition-state notation" language. `docs/references.rst:199`, `docs/methodology/REGISTRY.md:488`, code docstrings, and the new paper review file headers all align on the `(2022, revised July 2023)` revision string for NBER WP 29873.
3. **PR-B tracker-promotion consolidation (this PR):** formal `### Deviations from the paper / from R / library extensions` block added to REGISTRY `## ChaisemartinDHaultfoeuille` consolidating 7 documented deviations into a single AI-review-recognized labeled surface per CLAUDE.md "Documenting Deviations" labels. The original scattered `**Note (deviation from R...):**` entries remain in place — the new Deviations block is an additional canonical surface for AI-review consumption.

**Deviations from the paper / from R / library extensions:**

(Cross-references the consolidated REGISTRY Deviations block — see `docs/methodology/REGISTRY.md` `## ChaisemartinDHaultfoeuille` § Deviations for the same 7 entries with full mechanical detail. Listed here in summary form.)

1. **Equal-cell weighting (deviation from BOTH paper Equation 3 AND R `DIDmultiplegtDYN`).** Library: each `(g,t)` cell contributes once regardless of within-cell observation count. AER 2020 Equation 3 defines `N_{d,d',t} = sum_g N_{g,t}` (observation-sum weighting); R weights by cell size. Agreement is exact on one-observation-per-cell inputs (the parity test generator). Phase 2 estimands (`DID_l`, `DID^{pl}_l`, `DID^n_l`, delta cost-benefit) inherit the same contract. Locked by `tests/test_chaisemartin_dhaultfoeuille.py::test_cell_count_weighting_unbalanced_input` (in `TestDropLargerLower`).
2. **Period-based vs cohort-based stable controls (deviation from R `DIDmultiplegtDYN`).** Python: `stable_0(t)` is any cell with `D_{g,t-1} = D_{g,t} = 0` regardless of baseline `D_{g,1}` (matches AER 2020 Theorem 3 transition-state notation `N_{0,0,t}` and `N_{1,1,t}` literally). R: cohort-based control sets additionally require `D_{g,1}` to match the side. Agreement exact on pure-direction panels; ~1% point-estimate divergence on mixed-direction panels where joiners' post-switch cells could serve as leavers' controls (or vice versa). After the Round 2 full-IF fix, SE parity gap on pure-direction scenarios narrowed from ~18% to ~3%.
3. **Balanced-baseline panel required + terminal missingness retained (deviation from R `DIDmultiplegtDYN`)** — one composite deviation with four enforcement paths: (a) groups missing the first global period raise `ValueError`; (b) groups with interior period gaps are dropped with `UserWarning`; (c) groups with terminal missingness (observed at baseline but missing one or more later periods) are **retained** and contribute from their observed periods only; (d) cell-period allocator paths (Binder TSL with within-group-varying PSU, Rao-Wu replicate ATT, cell-level wild PSU bootstrap) emit a targeted `ValueError` when cohort recentering would leak nonzero centered IF mass onto cells with no positive-weight observations. R accepts unbalanced panels with documented missing-treatment-before-first-switch handling. The four paths share a single underlying contract — "the panel must be balanced at baseline; terminal missingness is the only allowed unbalance; downstream variance machinery refuses to silently leak IF mass past the cell-period boundary".
4. **SE normalization `N_l` vs R `G` (~4% smaller analytical SE).** Python implements the dynamic paper's Section 3.7.3 plug-in formula verbatim: `SE = sigma-hat / sqrt(N_l)` where `N_l` is the number of eligible switcher groups at horizon `l`. R normalizes the influence function by `G` (total number of groups including never-switchers and stable controls). Both converge to the same asymptotic variance as `G → ∞`. In finite samples Python's tighter SE remains conservative (paper formula is already an upper bound on the true variance via Jensen's inequality under Assumption 8). Gap is deterministic on identical data and ~3.5-5.1% across horizons and scenarios.
5. **Singleton-cohort degeneracy → NaN with warning (deviation from R `DIDmultiplegtDYN`).** When every variance-eligible group forms its own `(D_{g,1}, F_g, S_g)` cohort, cohort recentering collapses the centered IF vector to all zeros and the estimator returns `overall_se = NaN` with `UserWarning`. R returns a non-zero SE on degenerate small-panel cases via small-sample sandwich machinery that Python does not implement. Both responses are valid for a degenerate case; Python's `NaN` + warning is the safer default. Bootstrap inherits the same degeneracy.
6. **`<50%` switcher warning at far horizons (library extension).** When fewer than 50% of the `l=1` switchers contribute at a far horizon `l`, `fit()` emits a `UserWarning`. The dynamic paper (NBER WP 29873) recommends not reporting such horizons (Favara-Imbs application, footnote 14). Library convention is to warn but compute; not present in R.
7. **DID^X covariate first-stage equal-cell weights (deviation from R `DIDmultiplegtDYN`).** Phase 3 covariate adjustment (`controls=[...]`) residualizes outcomes via per-baseline first-stage OLS on first-differenced covariates with time FEs. Python uses equal cell weights consistent with the Phase 1 cell-count convention (Deviation #1); R weights by `N_{gt}`. On one-observation-per-cell panels results are identical. When baseline-specific first stages fail (`n_obs = 0` or `n_obs < n_params`), both Python and R drop the affected strata.

**Outstanding Concerns:**
- **Customer-supplied `cluster=<col>`** — currently `cluster=None` is the only supported value; passing any non-`None` value raises `NotImplementedError` at construction time (and the same gate fires from `set_params`). Reserved for a future phase. Auto-cluster at the group level is the default; `survey_design.psu` is the auto-cluster surface for hierarchical sampling, and PSU-within-group-constant regimes (including the default `psu=group` auto-inject and strictly-coarser PSU with within-group constancy) route through the legacy group-level allocator with bit-identical SE.
- **2024 NBER WP revision re-review** — the disk PDF is the "March 2022, revised July 2023" version; the 2024 revision was not re-reviewed for this PR. `docs/references.rst:199`, `docs/methodology/REGISTRY.md:488`, and the paper review file headers all align on the July 2023 revision string. If the 2024 revision introduces methodological changes, a re-review may be queued as a TODO follow-up post-promotion.
- **Methodology-test-file coverage of the universal-rollout extension (Knau et al. 2026)** is out of scope for DCDH. The 2026 paper's primary contribution is the `HeterogeneousAdoptionDiD` (HAD) estimator (already promoted in PR #473) and the universal-rollout case where no unit remains untreated. DCDH covers the reversible / mixed-direction designs from the 2020 AER and the dynamic event study from the 2022/2023 NBER WP 29873; the 2026 paper's universal-rollout contributions are documented in the companion review at `docs/methodology/papers/dechaisemartin-2026-review.md` without a dedicated DCDH methodology test file.

---

#### HeterogeneousAdoptionDiD (HAD)

| Field | Value |
|-------|-------|
| Module | `had.py`, `had_pretests.py` |
| Primary Reference | de Chaisemartin, Ciccia, D'Haultfœuille & Knau (2026), *Difference-in-Differences Estimators When No Unit Remains Untreated*, arXiv:2405.04465v6 |
| R Reference | `chaisemartin::did_had` (`Credible-Answers/did_had` v2.0.0, SHA `edc09197`) — R-parity-locked at `atol=1e-8` on 3 DGPs × 5 method combos via `tests/test_did_had_parity.py`; `nprobust` (Calonico-Cattaneo-Farrell) v0.5.0 used as auxiliary reference for bandwidth selection only (machine-precision port at `atol=1e-14`) |
| Status | **Complete** |
| Last Review | 2026-05-20 |

**Verified Components:**
- [x] Eq. 3 / Theorem 1 (Design 1' WAS identification: `WAS = [E(ΔY) − lim_{d↓0} E(ΔY | D ≤ d)] / E(D)`, the boundary-subtracted form; the library estimates the boundary intercept via bias-corrected local linear and computes `att = (mean(ΔY) − τ_bc) / mean(D)`) — `tests/test_methodology_had.py::TestHADTheorem1Design1Prime` (7 tests including MC recovery on the simple `ΔY = β·D + ε` DGP, MC recovery on a NONZERO-BOUNDARY-INTERCEPT DGP `ΔY = c + β·D + ε` with `c != 0` to exercise the `mean(ΔY) − τ_bc` subtraction explicitly, and N(0,1) coverage at `n_replicates=200`, G=1000)
- [x] Eq. 7 (local-linear with bias-corrected CI) — covered by `tests/test_bias_corrected_lprobust.py` (44 tests, hand-derived R reference at `atol=1e-12`) and `tests/test_nprobust_port.py` (~46 tests, machine-precision port at `atol=1e-14`)
- [x] Eq. 11 / Theorem 3 (`WAS_{d_lower}` under Assumption 6, mass-point path) — `tests/test_methodology_had.py::TestHADTheorem3MassPoint` (5 tests including Wald-IV closed-form equivalence at `atol=1e-9`)
- [x] Theorem 4 (QUG null test, limit law `T_λ = (λ + E_1) / E_2` under Exp(1)/Exp(1)) — `tests/test_methodology_had.py::TestHADTheorem4QUG` (6 tests; MC distributional match against closed-form `F(t) = t/(1+t)` at KS-stat ≤ 0.05, n_draws=5000)
- [x] Eq. 29 / Theorem 7 (Yatchew-HR linearity test, paper-literal `σ²_diff = 1/(2G)` normalization) — `tests/test_methodology_had.py::TestHADTheorem7YatchewHR` (6 tests; standard-normal limit, normalization lock, both `null="linearity"` and `null="mean_independence"` modes)
- [x] Eq. 18 joint Stute pre-trends + homogeneity (sum-of-CvMs + shared-η Mammen wild bootstrap; both mean-independence and linearity nulls) — `tests/test_methodology_had.py::TestHADJointStute` (5 tests). Coverage scope: H0 fail-to-reject on `joint_pretrends_test` (mean-independence) and `joint_homogeneity_test` (linearity); H1 rejection demonstrated on `joint_homogeneity_test` via a nonlinear DGP. **Out of scope for the new methodology file:** the `trends_lin=True` linear-trend-detrended variant is SHIPPED in the library (R-parity locked against `DIDHAD::did_had(..., trends_lin=TRUE)` v2.0.0; see REGISTRY § "Note (Phase 4 — Eq 17 / Eq 18 linear-trend detrending shipped)" and `tests/test_did_had_parity.py`) but its methodology-walk-through tests are NOT duplicated in `test_methodology_had.py`. Pierce-Schott NUMERICAL replication against the published p=0.51 anchor on the LBD-restricted panel is the waived item (REGISTRY Deviations Note #3).
- [x] R parity (`chaisemartin::did_had`) at `atol=1e-8` on 3 DGPs × 5 method combos (bit-exact, `rtol=0`) — `tests/test_did_had_parity.py::TestPointSEParity` + `TestYatchewParity` (5 direct parity tests; YatchewTest closed-form parity at `atol=1e-10`)
- [x] `nprobust` (Calonico-Cattaneo-Farrell) port at machine precision (`atol=1e-14`) — `tests/test_nprobust_port.py` (7 classes spanning kernel constants, QR-based `(X'X)^{-1}`, three-stage MSE-DPI bandwidth, clustered variance, weighted local-linear, single-eval-point parity)
- [x] Bandwidth selector (CCF MSE-DPI) at 1% tolerance — `tests/test_bandwidth_selector.py` (8 classes covering public-API wrapper, stage diagnostics)
- [x] Survey support: pweight + strata/PSU/FPC via TSL on the continuous and mass-point paths; PSU-level Mammen wild bootstrap on the Stute family; closed-form weighted variance components on Yatchew (Phase 4.5 A/B/C; QUG-under-survey permanently deferred per Phase 4.5 C0)
- [x] Tutorials T21 (`docs/tutorials/21_had_pretest_workflow.ipynb`, 17 drift tests) + T22 (`docs/tutorials/22_had_survey_design.ipynb`, 32 drift tests across groups A-G); plus T20 (`docs/tutorials/20_had_brand_campaign.ipynb`, 14 drift tests)
- [x] Assumption 5/6 non-testability documented in `HeterogeneousAdoptionDiD` class docstring + `qug_test`/`stute_test`/`yatchew_hr_test`/`did_had_pretest_workflow` Notes blocks; reinforced by a fit-time `UserWarning` emitted from the outer `HeterogeneousAdoptionDiD.fit()` dispatch on the overall and event-study paths when the resolved design is Design 1 family (search `diff_diff/had.py` for "---- Assumption 5/6 warning on Design 1 paths ----")

**Test Coverage:**
- 36 methodology tests in `tests/test_methodology_had.py` (3 are `@pytest.mark.slow` + gated by `ci_params.bootstrap(...)`: Theorem 1 N(0,1) coverage at `n_reps=200`/`min_n=25`, Theorem 4 QUG limit-law KS at `n_draws=5000`/`min_n=200`, and Theorem 7 Yatchew-HR standard-normal KS at `n_reps=200`/`min_n=25` — each carries an n-conditional tolerance band per `feedback_bootstrap_drift_tests_need_backend_tolerance`) (this PR)
- ~1,137 implementation-detail tests across `tests/test_had.py`, `tests/test_had_pretests.py`, `tests/test_had_mc.py`, `tests/test_had_dual_knob_deprecation.py`
- 5 R-direct parity tests at `atol=1e-8` in `tests/test_did_had_parity.py`
- ~46 + ~44 nprobust port + bias-corrected port tests
- ~45 bandwidth selector tests
- 17 + 32 tutorial drift tests (T21 + T22), plus 14 T20 drift tests

**Corrections Made:**
1. **Phase 4.5 B sup-t bootstrap (PR #432, 2026-05-14):** introduced the gated simultaneous-band bootstrap on the weighted event-study path with the explicit `cband=True` + `aggregate="event_study"` + `weights= or survey_design=` gate.
2. **Phase 4.5 C survey support for linearity family (PR #432):** PSU-level Mammen wild bootstrap for Stute + closed-form weighted variance for Yatchew. Replaced an earlier `NotImplementedError` stub.
3. **HAD survey-design API consolidation (PR #439, 2026-05-15):** unified `survey_design=` kwarg across all 8 HAD surfaces; `survey=` / `weights=` become deprecated aliases for one minor cycle.
4. **Tracker-promotion docstring hardening (this PR, 2026-05-20):** added explicit "Non-testable assumptions (paper Section 3.1.2)" Notes block to the `HeterogeneousAdoptionDiD` class docstring + "Scope (what this test does NOT cover)" clauses to `qug_test` / `stute_test` / `yatchew_hr_test` / `did_had_pretest_workflow` Notes sections. Boxed the REGISTRY HAD Implementation Checklist closures for Phase-4 items (Pierce-Schott Figure 2 + Table 1 coverage waivers, Assumption 5/6 non-testability docs, staggered-timing fail-closed `ValueError`).

**Deviations from the paper / from R / library extensions:**
1. **Equal-weighting on the continuous path** (paper does not prescribe a unit-weighting scheme; library uses per-unit `w_g = 1` matching `_nprobust_port.lprobust`'s default, NOT cell-size weights). Locked in `tests/test_methodology_had.py::TestHADDeviations::test_equal_weighting_is_per_row_not_per_dose_cell` (probes the deviation via selective low-dose-region replication on a nonlinear DGP: per-row equal weighting predicts the att shifts; cell-size weighting predicts invariance).
2. **Sup-t bootstrap gating** — runs only when `aggregate="event_study"` AND `(weights= or survey_design= supplied)` AND `cband=True`. Unweighted event-study bit-exactly preserves pre-Phase 4.5 B output. Locked in `TestHADDeviations::test_sup_t_bootstrap_skipped_*`.
3. **Pierce-Schott Figure 2 replication waived** — R parity at `atol=1e-8` is a stronger anchor; paper Section 5.2 self-acknowledges NP estimators are too noisy on LBD-restricted PNTR data. See REGISTRY Deviations § "Pierce-Schott (2016) Figure 2 replication harness deferred" for the full scope-caveat statement.
4. **Table 1 coverage-rate reproduction waived** — same R-parity-is-stronger rationale; R parity locks point estimate + SE + CI bounds bit-exactly, coverage-rate MC would re-verify the CCF asymptotic coverage already pinned. Paper Table 1 (89% / 93% / 95% under-coverage at G=100 / 500 / 2500) documents the asymptotic gap that BOTH R and Python inherit.
5. **Staggered-timing fail-closed `ValueError`** at `diff_diff/had.py:1511` (paper prescribes "Warn"; library raises). Library extension toward stricter safety — `UserWarning` would let the silent-misuse bug class through. Locked in `TestHADDeviations::test_staggered_timing_fail_closed_value_error`.
6. **Eq. 18 linear-trend-detrended joint Stute SHIPPED** (PR #389) and R-parity-locked against `DIDHAD::did_had(..., trends_lin=TRUE)` v2.0.0 in `tests/test_did_had_parity.py` (3 DGPs × 5 method combos at `atol=1e-8`). The `tests/test_methodology_had.py::TestHADJointStute` walkthrough deliberately covers only the un-detrended mean-independence and linearity variants (no coverage duplication with the R-parity surface). The Pierce-Schott (2016) NUMERICAL replication against the published p=0.51 anchor on the LBD-restricted PNTR panel is what's waived (Deviations Note #3).

**Outstanding Concerns:**
- Module split (`had.py` ~4593 LoC, `had_pretests.py` ~4951 LoC) — tracked in TODO.md as tech debt, not a methodology gap.
- Bandwidth selector multi-eval, cross-horizon covariance on joint event-study — tracked as Phase follow-ups in TODO.md.
- Replicate-weight designs (BRR / Fay / JK1 / JKn / SDR) on HAD continuous path remain `NotImplementedError` (Phase 4.5 D follow-up).
- `covariates=` kwarg with Theorem 6 multivariate-covariate extension not implemented; currently a Python `TypeError` (kwarg absent from the `fit()` signature). Adding an explicit `**kwargs`-trap with `NotImplementedError` and a Theorem 6 pointer is tracked as a Low-priority follow-up in TODO.md.

---

#### TROP

| Field | Value |
|-------|-------|
| Module | `trop.py`, `trop_local.py`, `trop_results.py` (paper-aligned local method); `trop_global.py` (library-side efficiency adaptation — see "Scope" below) |
| Primary Reference | Athey, Imbens, Qu & Viviano (2025), *Triply Robust Panel Estimators*, arXiv:2508.21536 |
| R Reference | Paper-author reference implementation (not yet released as CRAN package) |
| Status | **Complete** (paper `method="local"`, version-pinned to arXiv v2 — see Version Pinning below) |
| Last Review | 2026-05-24 |

**Version Pinning:** This methodology promotion is anchored on **arXiv:2508.21536v2** (the version covered by the paper review on file at `docs/methodology/papers/athey-2025-review.md`). The current arXiv version is **v3** (submitted 2026-02-09). A formal v2→v3 source delta-check against the v3 PDF has **NOT** been performed for any of the sections this PR promotes (Eqs. 2-3, Algorithms 1-3, Section 2.2, Section 5.2-5.3, Section 6.1-6.2, Theorem 5.1, Corollary 1, Appendix Theorem 8.1). **Action item:** before the next paper-author reference implementation or substantive v3 release, refresh the paper review against the most recent arXiv version and re-validate the verified-component checklist; until then the promotion stays v2-anchored.

**Scope:** This methodology promotion covers the paper-aligned `method="local"` path (paper Algorithm 2: per-(i, t) estimation with observation-specific weights). The library also exposes `method="global"`, documented in `REGISTRY.md` as a "computationally efficient adaptation using the (1-W) masking principle from Eq. 2" — a library-side adaptation, NOT the paper's full Algorithm 2 estimator. Defensive coverage of the global method lives in `tests/test_trop.py::TestTROPGlobalMethod` (704 lines, ~30 tests for the global-method-specific surface) and is not duplicated in the methodology walk-through. Methodology promotion of `method="global"` as a primary surface would require either (a) a paper-side derivation of the global adaptation's equivalence to Algorithm 2 under specific conditions, or (b) a separate library-extension methodology review; both are deferred.

**Verified Components:**

- [x] Eq. 2 weighted nuclear-norm-penalised L estimation: proximal-gradient inner solver (soft-threshold SVD) converges to ``prox_{λ/2}(R)`` under uniform weights; **plain** (non-accelerated) prox-gradient objective ``f(L) + λ‖L‖_*`` is non-increasing across iterations of a toy loop (this verifies the prox + gradient ingredient, NOT the shipped accelerated FISTA outer loop — Nesterov momentum gives the faster ``O(1/k^2)`` rate but does not guarantee per-step monotonicity); the shipped weighted-prox solver on non-uniform weights produces a final objective that is **at-or-below initialisation** (final-vs-initial check via `_weighted_nuclear_norm_solve`, NOT per-iteration monotonicity — the accelerated FISTA loop is allowed to have transient per-step increases) and reduces total singular-value mass below the residual. (Eq. 10 balancing representation / decomposition is the paper's identity built from these ingredients per Section 5.2; direct numerical reconstruction is out of scope — see "Outstanding Concerns".)
- [x] Eq. 3 per-(i, t) weights: unit distance excludes the target period (``1{u ≠ t}`` mask in the kernel) and uses only periods where both units are untreated (``(1 - W_iu)(1 - W_ju)`` mask).
- [x] Eqs. 4-5 + Algorithm 1 LOOCV: ``Q(λ)`` sums squared pseudo-treatment effects over ALL control observations where ``D_js = 0`` (including pre-treatment cells of eventually-treated units, paper Eq. 2 control set); two-stage coordinate-descent cycling (footnote 2) returns a tuple of values from the input grids.
- [x] Corollary 1 (paper p. 23) — **single-draw sanity checks consistent with the three unbiasedness conditions, not a repeated-MC mean-bias study**: each of the three balance conditions (a) unit balance, (b) time balance, (c) ``B = 0`` is exercised on a targeted DGP that makes one condition trivially hold while keeping the others sub-optimal. The assertion in each case is a single-realisation ``|att - τ| < 3 * se`` band using the estimator's own bootstrap SE — this is a smoke check, NOT a repeated-draw Monte Carlo bias study of the paper's conditional-unbiasedness statement under fixed weights. A stronger MC bias study at fixed λ values is deferred (would multiply test runtime by ~30x for marginal additional evidence given the existing 3-σ band already catches order-of-magnitude bias regressions).
- [x] Theorem 5.1 (paper p. 23) — **simulation sanity check, not a direct theorem lock**: the paper's bias bound ``|E[τ_hat - τ | L]| <= ||Δ_u|| · ||Δ_t|| · ||B||_*`` is stated for FIXED, non-data-dependent weights. The library's TROP fit uses data-dependent LOOCV-tuned λ values, so the direct conditional bias bound is not tested here. Instead, the methodology test verifies the bound's empirical realisation: TROP RMSE strictly below DID RMSE under a confounded factor DGP with ``true τ = 0`` (calibration measurement: TROP/DID RMSE ratio ≈ 0.34 at ``factor_strength = 1.0``). The direct fixed-weight bound test is deferred — would require exposing oracle Γ / Λ / B from a paper-aligned DGP and computing each component of the bound from instrumented internals.
- [x] Section 2.2 special-case reductions: **DiD benchmark sanity check** (not a direct algebraic-equivalence proof) — on a no-interactive-FE multi-period panel (additive unit + time effects only, no factor structure), TROP with ``λ_nn = ∞`` + uniform weights produces an ATT within 0.5 of `DifferenceInDifferences` fitted as `outcome ~ treat * post_flag` (basic 2×2 design with `[const, D, T, D×T]`, extended to repeated observations within each treat×post cell). This is **empirical numerical agreement on a friendly DGP**, NOT a proof of the paper Section 2.2 algebraic reduction (which would require either a true 2-period block-assignment panel where the basic-DiD comparator is the algebraic target, or a comparison against `TwoWayFixedEffects` — both deferred). **Matrix Completion code path exercised, not equivalence-checked** — TROP with uniform weights + finite ``λ_nn`` engages the nuclear-norm prox solver (effective_rank > 0) and recovers ATT better than the DiD-style baseline on a factor-confounded DGP; this verifies the code path activates but does NOT prove equivalence with an independent MC reference implementation (which would require either an external MC port or a hand-written reference solver). SC / SDID reductions deferred — see "Outstanding Concerns".
- [x] Eq. 13 + Algorithm 2 per-(i, t) estimation: ``treatment_effects`` dict contains one finite ``τ_hat_it`` per treated cell; the aggregate ATT equals the unweighted mean of per-cell effects (Eq. 1). **Tests cover block adoption with a constant treatment effect**; **absorbing-state staggered adoption** and **heterogeneous per-cell effects** (paper Remark 6.1) are SUPPORTED by the code path but not directly verified in this methodology surface. **Section 6.1 non-absorbing / on-off / switching assignment patterns are explicitly OUT OF SCOPE** — the implementation rejects non-absorbing D-matrices via `trop_local.py` absorbing-state validation, and the methodology test enforces the rejection contract via `TestTROPDeviations::test_event_style_d_rejected_with_value_error` (event-style D being one specific non-absorbing pattern; the same absorbing-state validator catches all 1→0 transitions). Cross-coverage of the staggered-cohort fit path is `tests/test_methodology_trop.py::TestTROPAlgorithm1LOOCV::test_control_set_includes_pretreat_of_eventually_treated`.
- [x] Algorithm 3 stratified pairs bootstrap: under an unbalanced (3 treated, 17 control) panel, the stratified sampler reliably produces ≥ 67% successful bootstrap draws and a positive finite SE.
- [x] Section 3 / Eq. 6 semi-synthetic factor DGP: five recovery tests verify limiting-case uniform weights, unit-weight bias reduction, time-weight bias reduction, factor-model bias reduction with effective_rank > 0, and null-DGP recovery centred near zero.
- [x] safe_inference contract: confidence interval uses the t-distribution with df = max(1, n_treated_obs - 1), consistent with p_value (matches REGISTRY `## TROP` "Inference CI distribution" note, post safe_inference migration).

**Test Coverage:**

- 36 methodology tests (10 classes) in `tests/test_methodology_trop.py`.
- Defensive guards (107 tests in `tests/test_trop.py`): D-matrix absorbing-state validation, silent-warning audit, FISTA convergence warnings, bootstrap-failure-rate proportional warning, bootstrap NaN-SE propagation, module-split smoke tests.

**Deviations from paper:**

- **Gap #5 (weight normalisation)**: paper Section 5 (p. 20) states weights sum to one (``1ᵀω = 1ᵀθ = 1``), but Eq. 3 (p. 7) writes unnormalised exponential weights. The shipped implementation matches Eq. 2 (unnormalised). Locked by `tests/test_methodology_trop.py::TestTROPDeviations::test_unnormalized_weights_match_eq2`.
- **Gap #9 (unbalanced panels)**: paper assumes a balanced ``N × T`` panel; the library accepts unbalanced panels with missing control / pre-treatment cells. **The methodology test exercises 10% random drops on the control + pre-treatment subset** (TROP fit completes and returns finite ATT). Three additional unbalanced-panel regressions are in `tests/test_trop.py::TestPR110FeedbackRound8`. Missing-treated-cell handling and thinner pre-period donor support are NOT directly covered by a dedicated regression — those edge cases lean on the absorbing-state monotonicity validation in `trop_local.py` and the validator-side tests in `tests/test_trop.py::TestDMatrixValidation`. Locked by `TestTROPDeviations::test_unbalanced_panels_supported`.
- **Rank selection**: the library follows the paper's implicit rank-selection via nuclear-norm soft-thresholding (paper review Gap #8). `TROP.__init__` does NOT expose a discrete `rank_selection` parameter; effective rank is reported via `TROPResults.effective_rank` (sum of singular values divided by largest) as a diagnostic, not as a user-selectable mode. Earlier REGISTRY prose mentioning "cv / ic / elbow" rank-selection methods was an overclaim — corrected in this PR. Locked by `TestTROPDeviations::test_rank_selection_is_implicit_via_nuclear_norm`.
- **`λ_nn = ∞` → 1e10 internal conversion**: results metadata stores the original ``inf`` value (REGISTRY `## TROP` "λ_nn=∞ implementation" edge-case note) while computations use 1e10 as a numerical sentinel. Locked by `TestTROPDeviations::test_lambda_nn_inf_stored_unchanged`.
- **Bootstrap proportional 5% failure-rate warning** and **FISTA / outer-loop convergence warnings**: defensive surfaces introduced under the Phase 2 silent-failure audit (REGISTRY `## TROP` "Bootstrap minimum" and "alternating-minimization convergence" notes). Covered in `tests/test_trop.py::TestTROPBootstrapFailureRateGuard` and `TestTROPConvergenceWarnings`.

**Outstanding Concerns:**

- **Equation 14 covariate extension** (paper Section 6.2): the library does NOT implement covariates. `TROP.fit()` has no ``covariates`` keyword argument, and the corresponding Theorem 8.1 (paper Appendix, pp. 36-37) covariate-triple-robustness result is correspondingly out of scope. Non-support is locked by `TestTROPDeviations::test_covariates_not_supported`. Deferred until use cases motivate the X threading through `trop_local.py` / `trop_global.py` / LOOCV / bootstrap.
- **SC / SDID special-case reductions** (paper Section 2.2 third bullet): the paper claims TROP reduces to SC and SDID under ``λ_nn = ∞`` and "specific choices of unit and time weights", but the exact ``(ω, θ)`` maps are not provided in the paper text. The library does not expose an SC- or SDID-matching weight setter (only the Eq. 3 ``λ_unit`` / ``λ_time`` decay rates). Cross-language anchor against `SyntheticDiD` is deferred until paper-author code clarifies the weight map.
- **Rust backend survey scope**: the Rust accelerated paths remain pweight-only; full survey-design (strata, PSU, FPC via Rao-Wu) falls back to the Python bootstrap loop. Cross-backend parity is covered by `tests/test_trop.py` defensive surfaces.
- **Eq. 10 balancing decomposition** (paper Section 5.2 + Eq. 10): numerical reconstruction of the four-term identity ``Y_NT_hat = L_NT + θ·(Y_pre_N - L_pre_N) + ω·(Y_T_co - L_T_co) - Σ θ_t ω_i (Y_it_co - L_it_co)`` requires the internal per-(i, t) weight vectors ``θ_s^{i,t}`` / ``ω_j^{i,t}``, which are not exposed on the public TROP API. The Eq. 2 ingredients that the Eq. 10 derivation builds on (soft-threshold SVD, **plain prox-gradient monotonicity** — NOT the shipped accelerated FISTA outer loop, which uses Nesterov momentum and does not guarantee per-step monotonicity — weighted-prox solver) are independently verified in `TestTROPNuclearNormProx`. Direct Eq. 10 lock deferred — would require exposing the internal weight vectors as a results-side diagnostic or instrumenting the test against the solver's intermediate state.

**R Parity:** Deferred until the paper-author reference implementation is released ("forthcoming" per the paper). Tracker entry will be reopened when it lands.

---

### Triple-Difference Estimators

#### TripleDifference

| Field | Value |
|-------|-------|
| Module | `triple_diff.py` |
| Primary Reference | Ortiz-Villavicencio & Sant'Anna (2025), *Better Understanding Triple Differences Estimators*, arXiv:2505.09942 |
| R Reference | `triplediff::ddd()` (v0.2.1, CRAN) |
| Status | **Complete** |
| Last Review | 2026-02-18 |

**Verified Components:**
- [x] ATT matches R `triplediff::ddd()` for all 3 methods (DR, RA, IPW) — <0.001% relative difference
- [x] SE matches R `triplediff::ddd()` for all 3 methods — <0.001% relative difference
- [x] With-covariates ATT matches R — <0.001% relative difference
- [x] With-covariates SE matches R — <0.001% relative difference
- [x] Verified across all 4 DGP types from `gen_dgp_2periods()` (different model misspecification scenarios)
- [x] Influence function-based SE: `SE = std(w3*IF_3 + w2*IF_2 - w1*IF_1, ddof=1) / sqrt(n)`
- [x] Three-DiD decomposition: `DDD = DiD_3 + DiD_2 - DiD_1` matching R's approach
- [x] `safe_inference()` used for all inference fields (t_stat, p_value, conf_int)

**Test Coverage:**
- 45 methodology tests in `tests/test_methodology_triple_diff.py`

**Corrections Made:**
1. **Complete rewrite of estimation methods** (was naive cell-mean approach, now three-DiD
   decomposition). The original implementation computed DDD directly from 8 cell means with
   a naive cell-variance SE. Replaced with R's decomposition into three pairwise DiD
   comparisons (subgroup j vs reference subgroup 4), each using DR/IPW/RA methodology
   from Callaway & Sant'Anna. This fixed:
   - DR SE: was off by >100% (naive cell variance vs influence function)
   - IPW SE: was off by >200% (incorrect cell-probability-ratio weights)
   - With-covariates ATT: was off by >1000% for all methods (incorrect cell-by-cell regression)
2. **Influence function SE** replaces naive cell variance for all methods:
   `SE = std(w3*IF_3 + w2*IF_2 - w1*IF_1, ddof=1) / sqrt(n)` where
   `w_j = n / n_j` and `IF_j` is the per-observation influence function for pairwise DiD j.
3. **Propensity score estimation** now runs per-pairwise-comparison (P(subgroup=4|X) within
   {j, 4} subset) instead of global P(G=1|X).
4. **Outcome regression** now fits separate OLS per subgroup-time cell within each pairwise
   comparison, matching R's `compute_outcome_regression_rc()`.

**Outstanding Concerns:**
- Panel mode (`panel=TRUE`) with differenced outcomes not yet implemented (see Deviations).

**Deviations from R's `triplediff::ddd()`:**
1. **Repeated cross-section mode only**: Implementation uses `panel=FALSE`. Panel mode with
   differenced outcomes is not yet implemented; users with balanced panel data and
   time-invariant covariates should compute first differences manually before fitting.

**R Comparison Results (panel=FALSE, n=500 per DGP):**
| DGP | Method | Covariates | ATT Diff | SE Diff |
|-----|--------|-----------|----------|---------|
| 1 | DR | No | <0.001% | <0.001% |
| 1 | DR | Yes | <0.001% | <0.001% |
| 1 | REG | No | <0.001% | <0.001% |
| 1 | REG | Yes | <0.001% | <0.001% |
| 1 | IPW | No | <0.001% | <0.001% |
| 1 | IPW | Yes | <0.001% | <0.001% |
| 2-4 | All | Both | <0.001% | <0.001% |

---

#### StaggeredTripleDifference

| Field | Value |
|-------|-------|
| Module | `staggered_triple_diff.py`, `staggered_triple_diff_results.py` |
| Primary Reference | Ortiz-Villavicencio & Sant'Anna (2025) — same paper as TripleDifference, staggered case |
| R Reference | `triplediff::ddd(panel=TRUE)` + `agg_ddd()` (per `benchmarks/R/benchmark_staggered_triplediff.R`) |
| Status | **In Progress** |
| Last Review | — |

**Documentation in place:**
- REGISTRY.md section: `## StaggeredTripleDifference` (per-cohort comparisons against three sub-groups, DR/RA/IPW per component, GMM-optimal closed-form inverse-variance weighting, event-study via CS mixin, IF-based SEs, multiplier bootstrap for simultaneous bands, survey support)
- `tests/test_methodology_staggered_triple_diff.py`: 6 tests across 3 classes (never-treated comparison, not-yet-treated comparison, aggregation)
- Dedicated unit-test suite: `tests/test_staggered_triple_diff.py` (~680 lines, full coverage of DR/RA/IPW paths, both control-group modes, GMM weighting, event-study aggregation, edge cases)
- Survey-specific: `tests/test_survey_staggered_ddd.py`

**Outstanding for promotion:**
- Paper review under `docs/methodology/papers/` covering Ortiz-Villavicencio & Sant'Anna (2025) for the staggered case (the primary paper is shared with TripleDifference, but no dedicated review file exists on disk yet)
- R parity validation against `triplediff::ddd(panel=TRUE)` + `agg_ddd()` (per `benchmarks/R/benchmark_staggered_triplediff.R`) — CSV fixtures not committed (gitignored); tests skip without local R + `triplediff` (tracked in TODO.md row, PR #245)
- Per-cohort group-effect SE convention: implementation includes WIF (conservative vs R's `wif=NULL`); documented in REGISTRY, deferred decision on whether to add an opt-in WIF-disable path (tracked in TODO.md row, PR #245)
- Formal Verified Components walk-through here
- Cluster-robust analytical SEs accepted but not wired (deferred per REGISTRY)

---

### Counterfactual / Synthetic Estimators

#### SyntheticDiD

| Field | Value |
|-------|-------|
| Module | `synthetic_did.py` |
| Primary Reference | Arkhangelsky et al. (2021) |
| R Reference | `synthdid::synthdid_estimate()` |
| Status | **Complete** |
| Last Review | 2026-04-23 |

**Verified Components:**
- [x] Frank-Wolfe on the collapsed (N_co × T_pre) problem (Algorithm 1 of Arkhangelsky et al. 2021), matching R's `synthdid::fw.step()`
- [x] Unit weights: Frank-Wolfe with two-pass sparsification, matching R's `synthdid::sc.weight.fw()` and `sparsify_function()`
- [x] Time weights: Frank-Wolfe on collapsed form, matching R's `fw.step()`
- [x] Auto-computed `zeta_omega` / `zeta_lambda` from data noise level `N_tr × σ²` (Appendix D), matching R's default behavior
- [x] Pairs-bootstrap refit per Algorithm 2 step 2, warm-started from fit-time ω/λ via the new `init_weights=` kwargs on `compute_sdid_unit_weights` / `compute_time_weights`, matching R's `bootstrap_sample` which rebinds `attr(estimate, "opts")` per `update.omega=TRUE` / `update.lambda=TRUE`
- [x] Placebo variance (library default) and jackknife variance methods
- [x] Same-library validation: placebo-SE tracking vs. bootstrap-SE, AER §6.3 Monte Carlo truth
- [x] All REGISTRY.md SyntheticDiD edge cases tested

**Test Coverage:**
- 157 methodology tests in `tests/test_methodology_sdid.py`

**Corrections Made:**
1. **Time weights: Frank-Wolfe on collapsed form** (was heuristic inverse-distance).
   Replaced ad-hoc inverse-distance weighting with the Frank-Wolfe algorithm operating
   on the collapsed (N_co x T_pre) problem as specified in Algorithm 1 of
   Arkhangelsky et al. (2021), matching R's `synthdid::fw.step()`.
2. **Unit weights: Frank-Wolfe with two-pass sparsification** (was projected gradient
   descent with wrong penalty). Replaced projected gradient descent (which used an
   incorrect penalty formulation) with Frank-Wolfe optimization followed by two-pass
   sparsification, matching R's `synthdid::sc.weight.fw()` and `sparsify_function()`.
3. **Auto-computed regularization from data noise level** (was `lambda_reg=0.0`,
   `zeta=1.0`). Regularization parameters `zeta_omega` and `zeta_lambda` are now
   computed automatically from the data noise level (N_tr * sigma^2) as specified in
   Appendix D of Arkhangelsky et al. (2021), matching R's default behavior.
4. **Bootstrap SE is paper-faithful refit (Algorithm 2 step 2), matching R's default
   `synthdid::vcov(method="bootstrap")` including its warm-start shape.** On each
   pairs-bootstrap draw, ω and λ are re-estimated via Frank-Wolfe on the resampled
   panel using the fit-time normalized-scale zeta. The Frank-Wolfe first pass is
   warm-started from the fit-time ω (renormalized over the resampled controls via
   `_sum_normalize`) and the fit-time λ (unchanged), matching R's `bootstrap_sample`
   which rebinds `attr(estimate, "opts")` so those weights serve as the FW
   initialization per `update.omega=TRUE` / `update.lambda=TRUE`.
   *(Historical note: an earlier release shipped a fixed-weight shortcut here
   that matched neither the paper nor R's default vcov; that path was removed
   in PR #351 along with its R-parity fixture, which had also been mis-anchored.
   The same PR added the warm-start plumbing to `compute_sdid_unit_weights` /
   `compute_time_weights` via new `init_weights=` kwargs.)*
5. **Default `variance_method` changed to `"placebo"`** — intentional deviation from
   R's default (R's `synthdid::vcov()` defaults to `"bootstrap"`). The library default
   is placebo for two reasons: (a) placebo is unconditionally available on pweight-only
   survey designs, whereas refit bootstrap rejects every survey design in this release;
   (b) placebo sidesteps the ~5–30× slowdown of per-draw Frank-Wolfe re-estimation in
   refit bootstrap. See REGISTRY.md §SyntheticDiD `Note (default variance_method
   deviation from R)` for details.
6. **Deprecated `lambda_reg` and `zeta` params; new params are `zeta_omega` and
   `zeta_lambda`**. The old parameters had unclear semantics and did not correspond to
   the paper's notation. The new parameters directly match the paper and R package
   naming conventions. `lambda_reg` and `zeta` are deprecated with warnings and will
   be removed in a future release.

**Outstanding Concerns:**
- Cross-language parity anchor against R's default `synthdid::vcov(method="bootstrap")`
  or Julia `Synthdid.jl::src/vcov.jl::bootstrap_se` is desirable to bolster the
  methodology contract. Same-library validation (placebo-SE tracking, AER §6.3 MC truth)
  is in place; cross-language anchor tracked in TODO.md. The R-parity fixture from the
  previous release was deleted because it pinned the now-removed fixed-weight path.

**Deviations from R's synthdid::synthdid_estimate():**
1. **Default `variance_method` is `"placebo"`** (R defaults to `"bootstrap"`). Rationale:
   (a) placebo is unconditionally available on pweight-only survey designs, whereas refit
   bootstrap rejects every survey design in this release; (b) placebo sidesteps the
   ~5–30× slowdown of per-draw Frank-Wolfe re-estimation in refit bootstrap. Documented
   in REGISTRY.md §SyntheticDiD `Note (default variance_method deviation from R)`.
2. **Parameter names**: `zeta_omega` / `zeta_lambda` (matching the paper's notation);
   R uses `eta.omega` / `eta.lambda`. The deprecated Python aliases `lambda_reg` / `zeta`
   from prior releases emit `DeprecationWarning` and will be removed in a future release.

---

### Diagnostics & Sensitivity

#### BaconDecomposition

| Field | Value |
|-------|-------|
| Module | `bacon.py` |
| Primary Reference | Goodman-Bacon (2021), *Difference-in-differences with variation in treatment timing*, J. Econometrics 225(2), 254-277 |
| R Reference | `bacondecomp::bacon()` |
| Status | **Complete** |
| Last Review | 2026-05-16 |

**Verified Components:**
- [x] Theorem 1 decomposition identity: `β̂^DD = Σ s · β̂^{2x2}` at `atol=1e-10` (hand-calculable + noisy DGPs)
- [x] Weight sum-to-1: `Σ s = 1.0` at `atol=1e-10` under `weights="exact"`
- [x] Three comparison types correctly classified: `treated_vs_never`, `earlier_vs_later`, `later_vs_earlier`
- [x] Eq. 7 hand-checked: `V̂_{kU}^D = n_{kU}(1-n_{kU}) · D̄_k(1-D̄_k)` (via weight-ratio test, `atol=1e-10`)
- [x] Eq. 8 hand-checked: `V̂_{kℓ}^{D,k} = n_{kℓ}(1-n_{kℓ}) · (D̄_k-D̄_ℓ)/(1-D̄_ℓ) · (1-D̄_k)/(1-D̄_ℓ)`
- [x] Eq. 9 hand-checked: `V̂_{kℓ}^{D,ℓ} = n_{kℓ}(1-n_{kℓ}) · D̄_ℓ/D̄_k · (D̄_k-D̄_ℓ)/D̄_k`
- [x] Eq. 10b 2x2 estimator value: hand-calculable panel → β̂_{kU}^{2x2} = ATT exactly
- [x] Always-treated remap to U (paper footnote 11): `first_treat <= min(time)` (excluding never-treated sentinels `0` and `np.inf`) units auto-remapped via internal column, user's data preserved, count exposed on result
- [x] `weights="exact"` is the default (PR-B 2026-05-16); `weights="approximate"` retained as opt-in
- [x] Unbalanced panel: accepted with `UserWarning` (paper assumes balanced; library extension)
- [x] No untreated group: `s_{kU}` terms drop, weights renormalize, sum-to-1 still holds
- [x] Single timing group with U: only `treated_vs_never` comparisons
- [x] Survey design composes cleanly with exact mode and warn+remap
- [x] R `bacondecomp::bacon()` parity at `atol=1e-6` — 3 fixtures (`uniform_3groups_with_never_treated`, `two_groups_no_never_treated`, `always_treated_remapped`); TWFE coefficient + weights-sum match across all 3 fixtures; per-component estimate + weight parity locked on the 2 non-remap fixtures **and on the 6 timing-vs-timing rows of `always_treated_remapped`** (carve-out narrowed to U-bucket rows only); R→Python U-bucket fold-back asserted by a dedicated `test_always_treated_remapped_fold_back_matches_r` test that aggregates R's split `Later vs Always Treated` + `Treated vs Untreated` rows per cohort and compares to Python's single `treated_vs_never` cell at `atol=1e-6`. See `benchmarks/data/r_bacondecomp_golden.json` + `TestBaconParityR`.

**Test Coverage:**
- 34 methodology tests in `tests/test_methodology_bacon.py` across 6 classes — all active, including the 4 R-parity tests (3 aggregate/per-component + 1 always-treated fold-back; goldens committed at `benchmarks/data/r_bacondecomp_golden.json`)
- 32 existing tests in `tests/test_bacon.py` (basic decomposition, weight properties, weights-parameter API, TWFE integration, visualization, balanced-panel warnings, edge cases)

**R Comparison Results:**
- **Validated** at `atol=1e-6` against `bacondecomp::bacon()` (version 0.1.1, R 4.5.2). Goldens at `benchmarks/data/r_bacondecomp_golden.json`; generator at `benchmarks/R/generate_bacon_golden.R`. Three DGP fixtures:
  - `uniform_3groups_with_never_treated`: 9 components covering all three comparison types — full per-component parity (estimate + weight at `atol=1e-6`).
  - `two_groups_no_never_treated`: 2 components, timing-only decomposition — full per-component parity.
  - `always_treated_remapped`: TWFE coefficient + weights-sum match at `atol=1e-6`; the 6 timing-vs-timing rows (between cohorts 3/4/5) also satisfy direct per-component parity at `atol=1e-6` (carve-out narrowed to U-bucket rows only). The U-bucket breakdown diverges by convention (Python's paper-footnote-11 U-remap vs R's distinct `Later vs Always Treated` cohort decomposition); the aggregate is invariant to the re-bucketing per Theorem 1, and the R→Python fold-back is pinned by `test_always_treated_remapped_fold_back_matches_r` which aggregates R's split `Later vs Always Treated` + `Treated vs Untreated` rows per cohort and compares to Python's single `treated_vs_never` cell.

**Corrections Made:**
1. **Theorem 1 exact-weights rewrite** (`bacon.py:_recompute_exact_weights`, lines ~740-880). The previous "exact" mode implementation did not actually compute Eqs. 7-9 / 10e-g — it was missing the `(1 - n_kU)` factor in the within-subsample treatment variance, did not square the sample share, and added an extraneous `unit_share` factor not present in the paper. The post-hoc sum-to-1 normalization masked the relative-weight error but produced a decomposition error of ~0.3% (0.007 absolute) against TWFE on a 3-cohort + never-treated DGP. **Rewrote** the function to compute the exact numerators of Eqs. 10e/f/g (with proper Eqs. 7-9 variances) and let the post-hoc normalization handle the `V̂^D` denominator (Theorem 1 identity guarantees `V̂^D = Σ numerators`). Now matches TWFE at `atol=1e-10`. The existing `test_weighted_sum_equals_twfe` tolerance was tightened from `< 0.1` to `< 1e-10` to lock the contract.
2. **Default `weights` flipped from `"approximate"` to `"exact"`** at three entry points: `BaconDecomposition.__init__()` (`bacon.py:397`), `bacon_decompose()` convenience function (`bacon.py:1064`), `TwoWayFixedEffects.decompose()` (`twfe.py:684`). The paper-faithful Theorem 1 weights are now the default; the simplified approximate path remains opt-in via explicit `weights="approximate"`. `diff_diff/diagnostic_report.py:1740` (production diagnostic surface) was updated to pass explicit `weights="exact"`.
3. **Always-treated warn+remap via internal column** (`bacon.py:fit()`, lines ~487-525). Paper footnote 11 puts units with `t_i < 1` in `U`, but `bacon.py` previously only mapped `first_treat ∈ {0, np.inf}` into U. Added detection using ordered-time logic on the **time axis** (`first_treat <= min(time)` while excluding the never-treated sentinels `0` and `np.inf`) with `UserWarning` and automatic remap via an internal column (`__bacon_first_treat_internal__`), preserving the user's `first_treat` column unchanged. Detection handles event-time-encoded panels (`time ∈ [-2,..,3]`) correctly; the `0` sentinel restriction applies only to `first_treat`. Count exposed via new `BaconDecompositionResults.n_always_treated_remapped` field.

**Deviations from R's `bacondecomp::bacon()` and from the paper:**
1. **First-period boundary extension on always-treated remap** (library convention, deviation from paper footnote 11 strict rule and from R): Goodman-Bacon (2021) footnote 11 uses strict `t_i < 1` for the always-treated bucket (units treated *before* the first observable period). The library applies the **inclusive** `first_treat <= min(time)` rule, additionally folding units treated *at* the first observable period (`first_treat == min(time)`) into `U`. Rationale: such units have no untreated cell in-panel and cannot contribute as a treated cohort, so folding them into U mirrors the always-treated handling rather than dropping them silently. R `bacondecomp::bacon()` does NOT apply this boundary fold-back — it keeps `first_treat == min(time)` cohorts in their own bucket and emits `Later vs Always Treated` comparisons. When `min(time) > 1` (no first-period-treated cohorts) the library rule reduces to the paper's strict rule. Documented in REGISTRY `**Deviation (first-period boundary extension on always-treated remap)**`.
2. **Unbalanced panel acceptance** (library extension): R errors on unbalanced panels; Python emits a `UserWarning` and decomposes. The paper's Appendix A proof assumes balanced panels — decomposition on unbalanced panels is approximate to Theorem 1.
3. **Approximate weight mode** (Python-only optimization): `weights="approximate"` is a library-only fast path with simplified variance computation, not present in R. Users who want Python-R numerical parity should pass `weights="exact"` (the new default).
4. **NaN for invalid inference fields not applicable**: the decomposition is deterministic; there are no SE/p-value fields on the comparison output. The `decomposition_error` field is a finite float (zero in well-conditioned cases).

---

#### HonestDiD

| Field | Value |
|-------|-------|
| Module | `honest_did.py` |
| Primary Reference | Rambachan & Roth (2023), *A More Credible Approach to Parallel Trends*, RES 90(5), 2555-2591 |
| R Reference | `HonestDiD` package |
| Status | **Complete** |
| Last Review | 2026-04-01 |

**Verified Components:**
- [x] Delta^SD: second-difference constraints [1,-2,1] with delta_0=0 boundary handling
- [x] Delta^SD: T+Tbar-1 constraint rows (bridge constraint at t=0)
- [x] Delta^RM: constrains first differences (not levels), union of polyhedra per Lemma 2.2
- [x] Identified set LP: pins delta_pre = beta_pre via equality constraints (Equations 5-6)
- [x] M=0 for Delta^SD: linear extrapolation gives finite point-identified bounds
- [x] Mbar=0 for Delta^RM: point identification (all post first-diffs = 0)
- [x] Optimal FLCI for Delta^SD: folded normal cv_alpha, Nelder-Mead over pre-period weights
- [x] Sensitivity grid: bounds computed for each M in grid, breakdown value via binary search
- [x] Survey variance (RM, M=0 smoothness): t-distribution critical values from df_survey
- [ ] Survey variance (M>0 smoothness): optimal FLCI uses asymptotic normal only; df_survey=0 → NaN
- [x] CallawaySantAnna integration: universal base period, reference period filtering
- [x] Three-period analytical case matches paper Section 2.3
- [ ] ARP hybrid for Delta^RM: infrastructure implemented, moment inequality transformation needs calibration
- [ ] R comparison: pending (benchmark scripts need updating)

**Test Coverage:**
- Comprehensive unit-test coverage in `tests/test_honest_did.py` (15 test classes spanning DeltaSD/DeltaRM/DeltaSDRM bounds, FLCI, ARP infrastructure, CS integration, edge cases) — all passing
- 27 methodology verification tests in `tests/test_methodology_honest_did.py`
- R benchmark tests (pending)
- Paper review on file: `docs/methodology/papers/rambachan-roth-2023-review.md`

**Corrections Made:**
1. **DeltaRM: first differences, not levels** (`honest_did.py`, `_construct_constraints_rm_component`):
   The paper's Delta^RM constrains `|delta_{t+1} - delta_t|` (consecutive first differences)
   bounded by Mbar × max pre-treatment first difference. The code constrained `|delta_post|`
   (absolute levels) bounded by Mbar × max `|beta_pre|`. Completely rewritten using
   union-of-polyhedra decomposition per Lemma 2.2.

2. **LP pins delta_pre = beta_pre** (`honest_did.py`, `_solve_bounds_lp`):
   The paper's identified set LP (Equations 5-6) fixes pre-treatment violations to the observed
   pre-treatment coefficients. The code had no equality constraint — delta_pre was unconstrained.
   For Delta^SD(M=0), this made the LP unbounded. Added A_eq/b_eq equality constraints.

3. **DeltaSD constraint matrix: delta_0=0 boundary** (`honest_did.py`, `_construct_A_sd`):
   The code built second-difference matrices treating [delta_{-T},...,delta_{-1},delta_1,...,delta_{Tbar}]
   as consecutive, missing delta_0=0 at the boundary. Three boundary rows were wrong:
   - t=-1: `d_{-2} - 2*d_{-1} + 0` (uses delta_0=0)
   - t=0: `d_{-1} + d_1` (bridge constraint, was missing)
   - t=1: `0 - 2*d_1 + d_2` (uses delta_0=0)
   Now produces T+Tbar-1 rows (was T+Tbar-2).

4. **Optimal FLCI for Delta^SD** (`honest_did.py`, `_compute_optimal_flci`):
   Replaced naive FLCI (`lb - z*se, ub + z*se`) with the paper's optimal FLCI (Section 4.1):
   jointly optimizes affine estimator direction v and half-length chi using folded normal
   critical values cv_alpha(bias/se). Significantly narrower CIs.

5. **REGISTRY.md equations** (`docs/methodology/REGISTRY.md`):
   DeltaSD equation was first differences (should be second differences). DeltaRM equation
   was absolute levels (should be first differences). Both corrected with full formulations.

6. **Performance** (`honest_did.py`):
   Sensitivity grid reduced from ~9 minutes to 0.1 seconds via: Newton's method for cv_alpha
   (5 iterations vs 100), centrosymmetric bias LP (1 solve vs 2), M=0 short-circuit,
   looser Nelder-Mead tolerances.

**Outstanding Concerns:**
- **Delta^RM CI**: uses naive FLCI (conservative) instead of the paper's ARP conditional/hybrid
  confidence sets. ARP infrastructure exists but moment inequality transformation needs
  calibration. Tracked in TODO.md.
- R benchmark comparison not yet run (Python benchmark needs API update)
- Combined method uses single M for both SD and RM (DeltaSDRM dataclass has separate M/Mbar)

**Deviations from R's HonestDiD:**
1. **Deviation from R:** Delta^RM CIs use naive FLCI (`lb - z*se, ub + z*se`) instead of ARP
   conditional/hybrid. Conservative (wider CIs, valid coverage). ARP deferred.
2. **Note:** Delta^SD optimal FLCI matches the paper's Section 4.1 methodology: first-difference
   reparameterization, slope weights with sum(w)=sum_j j*l_j constraint (Eq. 17), bias LP in
   fd-space, folded normal (or folded non-central t for survey df). Nelder-Mead optimizer vs
   R's custom solver may produce numerical differences at tolerance level.
3. **Note:** `method="combined"` (Delta^SDRM) uses naive FLCI on the intersection of SD and RM
   bounds. The paper proves FLCI is not consistent for Delta^SDRM (Proposition 4.2). A runtime
   UserWarning is emitted. Use `method="smoothness"` or `method="relative_magnitude"` separately
   for paper-supported inference.
4. **Note (deviation from R):** Python warns (doesn't error) when CallawaySantAnna results use
   `base_period != "universal"`. R's HonestDiD requires universal base period.

---

#### PreTrendsPower

| Field | Value |
|-------|-------|
| Module | `pretrends.py` |
| Primary Reference | Roth (2022), *Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends*, AER:I 4(3), 305-322 |
| R Reference | `pretrends` package |
| Status | **Complete** |
| Last Review | 2026-05-19 |

**Documentation in place:**
- REGISTRY.md section: `## PreTrendsPower` — NIS-framed audit per Roth (2022) Section II.A-B with full equation blocks for both NIS and Wald forms; paper-supported alternative + γ-unit MDV + full-Σ_22 routing all locked.
- Paper review on file: `docs/methodology/papers/roth-2022-review.md` (added 2026-05-17 via PR #463).
- Implementation: `tests/test_pretrends.py` (67 tests — point-estimator, MDV, power curve, sensitivity, plus the PR-A R18 silent-failure regression and the PR-B custom-weight persistence regression) + event-study coverage in `tests/test_pretrends_event_study.py` (27 tests).
- Dedicated `tests/test_methodology_pretrends.py` (added 2026-05-18 in PR-B Step 7; PR-C 2026-05-19 activated `TestPretrendsParityR` with 4 concrete tests) — Roth (2022) Section II.A-B paper-equation-numbered Verified Components walk-through (8 classes covering NIS box probability, Wald-vs-NIS, Propositions 1-4 simulation parity, linear-units γ-scale, custom-weight persistence, CS/SA full-VCV, helper API, R parity at commit `122731d082`).
- R parity goldens: `benchmarks/data/r_pretrends_golden.json` generated by `benchmarks/R/generate_pretrends_golden.R` against `jonathandroth/pretrends` commit `122731d082` (package version 0.1.0); 4 fixtures (regular K=3, irregular K=3 `[-5,-3,-1]`, anticipation-shifted K=4, K=1 closed form) × NIS power + γ_p MDV at `atol=1e-4`.

**Verified Components:**
- [x] NIS box probability implemented via `scipy.stats.multivariate_normal.cdf` (Roth Section II.A-B primary form)
- [x] Wald noncentral-χ² form retained as paper-supported alternative (Propositions 1+3+4 all apply — convex ellipsoid acceptance region)
- [x] Both forms produce form-consistent MDV via doubling + brentq bisection with 1000-cap non-convergence fallback
- [x] Non-bootstrap CS adapter consumes full `event_study_vcov` sub-block (not diag)
- [x] Non-bootstrap SA adapter consumes full `event_study_vcov` sub-block (W-matrix construction `event_study_vcov = W @ vcov_cohort @ W.T` added to `SunAbrahamResults`)
- [x] Bootstrap CS/SA and replicate-weight survey paths fall through to `diag(ses^2)` (analytical VCV cleared to prevent mixing with bootstrap/replicate SE overrides)
- [x] `_get_violation_weights('linear')` honors actual pre-period relative-time labels via `fit()` threading → reported MDV is in Roth's γ units on irregular and anticipation-shifted grids. For `MultiPeriodDiDResults`, supported label types are numeric (`int` / `float` / `np.int64`) and `pandas.Period` / `pandas.Timestamp` / `np.datetime64`; **genuinely non-numeric labels** (string period IDs, unranked categoricals) emit an explicit `UserWarning` and fall through to the legacy count-based normalized direction (MDV is NOT in γ units in that case — re-fit with numeric labels)
- [x] `PreTrendsPowerResults` persists fitted `violation_weights` + `pretest_form` + `nis_box_probability`; `power_at(M)` works for all four violation types on fresh fits
- [x] Helper API (`compute_pretrends_power`, `compute_mdv`) accepts `violation_weights` and `pretest_form`; closes the PR-A R18 helper/class API gap
- [x] Summary, `to_dict`, `to_dataframe` dispatch on `pretest_form` (NIS prints box probability; Wald prints noncentrality)
- [x] R `pretrends` parity at commit `122731d082` (PR-C, 2026-05-19) — 4 fixtures × NIS power + γ_p MDV at `atol=1e-4`; `tests/test_methodology_pretrends.py::TestPretrendsParityR` active

---

#### PowerAnalysis

| Field | Value |
|-------|-------|
| Module | `power.py` |
| Primary References | Bloom (1995); Burlig, Preonas & Woerman (2020) — clustered DiD power (both listed in REGISTRY) |
| R Reference | `pwr` (basic) / `DeclareDesign` (design-based simulation) |
| Status | **In Progress** |
| Last Review | — |

**Documentation in place:**
- REGISTRY.md section: `## PowerAnalysis` (MDE / power / sample size / simulation-based power / cluster adjustment); primary sources Bloom (1995) and Burlig et al. (2020) listed
- Implementation: `tests/test_power.py` (MDE / power / sample-size / simulation paths plus cluster adjustment)

**Outstanding for promotion:**
- Paper review under `docs/methodology/papers/` (likely a combined review covering Bloom 1995 + Burlig et al. 2020)
- Dedicated `tests/test_methodology_power.py` with closed-form walk-through against `pwr::pwr.t.test()` and Burlig et al.'s clustered-DiD power formula
- Documented reference-validation harness against `pwr` / `DeclareDesign`
- Verify the REGISTRY Implementation Checklist (all five items currently unchecked)

---

#### PlaceboTests

| Field | Value |
|-------|-------|
| Module | `diagnostics.py` |
| Primary Reference | None canonical (general permutation / leave-one-out diagnostic) |
| R Reference | None canonical |
| Status | **In Progress** |
| Last Review | — |

**Documentation in place:**
- REGISTRY.md section: `## PlaceboTests` (NaN-inference edge cases for `permutation_test` and `leave_one_out_test`)
- Implementation: tests embedded in `tests/test_diagnostics.py`

**Outstanding for promotion:**
- Decide whether this surface warrants a standalone methodology review or whether the brief Verified Components walk-through + NaN-inference deviation log should live as a sub-section under each per-estimator diagnostic block instead
- If kept standalone: brief Verified Components block + Deviations block for the NaN-inference convention

---

### Cross-Cutting Inference Features

These are not estimators but variance/inference plumbing used across many estimators. They warrant their own methodology reviews because the implementation details (kernel choice, weight rescaling, df adjustment) are independently citable.

#### ConleySpatialHAC

| Field | Value |
|-------|-------|
| Module | `conley.py`, `linalg.py` (`_validate_vcov_args`, kernel construction) |
| Primary Reference | Conley (1999), *GMM Estimation with Cross-Sectional Dependence*, J. Econometrics 92(1), 1-45 |
| Secondary References | Andrews (1991) HAC theory; Colella, Lalive, Sakalli & Thoenig (2019) for the Stata `acreg` parallel; Düsterhöft (2021) `conleyreg` (CRAN) parity target |
| Status | **In Progress** |
| Last Review | — |

**Documentation in place:**
- REGISTRY.md section: `## ConleySpatialHAC` plus three sub-sections (combined spatial + cluster product kernel — Wave A #119; performance/scale — Wave A #120; callable `conley_metric` validation — Wave A #123)
- **Paper review on file**: `docs/methodology/papers/conley-1999-review.md` (review date 2026-05-09); plus four adjacent paper reviews for the spillover initiative: `butts-2021-review.md`, `butts-2023-review.md` (JUE Insight), `clarke-2017-review.md`, `colella-et-al-2019-review.md`
- Implementation: 162 tests in `tests/test_conley_vcov.py` (Phase 1 + Phase 2 space-time HAC)
- Wired through `DifferenceInDifferences`, `MultiPeriodDiD`, `TwoWayFixedEffects` via `vcov_type="conley"` enum

**Documentation in place (R parity):**
- R `conleyreg` goldens committed: `benchmarks/data/r_conleyreg_conley_golden.json`, generator `benchmarks/R/generate_conley_golden.R`
- Cross-sectional R parity at `atol=1e-6`: `tests/test_conley_vcov.py::TestConleyParityR`
- Panel (space-time) R parity at `atol=1e-6`: `TestConleyParitySpacetime` (dense path) and `TestConleySparseRParityForced` (sparse path forced)
- Internal block-decomposition cross-check at machine precision (matches `conleyreg::time_dist.cpp`): `TestConleyParitySpacetime::test_panel_matches_block_decomposed_reference` (inner tolerance `atol=1e-12`)

**Outstanding for promotion:**
- Dedicated `tests/test_methodology_conley.py` with paper-equation-numbered Verified Components walk-through (Equation 8 score-covariance, Bartlett kernel, Andrews-style truncation) consolidating the parity tests into a methodology checklist
- Summary R-parity table in this tracker (currently the parity results are scattered across class-level docstrings in `tests/test_conley_vcov.py`)
- Document deviation: indefiniteness guard applied to both spatial and cluster kernels (vs. Bartlett's PSD property)
- Resolution for the Phase 5 spillover-conley dependency on survey-weights interaction (currently raises `NotImplementedError` at the linalg validator)

---

#### Survey Data Support

| Field | Value |
|-------|-------|
| Module | `survey.py`, `bootstrap_utils.py` (plus per-estimator hooks) |
| Primary References | Binder (1983) for TSL variance; Lumley (2004) for the R `survey` package; Solon, Haider & Wooldridge (2015) for the "when to weight" framework |
| R Reference | `survey` R package |
| Status | **In Progress** |
| Last Review | — |

**Documentation in place:**
- REGISTRY.md sub-sections (under `## Survey Data Support`): Weighted Estimation, TSL Variance, Weight Type Effects on Inference, Absorbed FE with Survey Weights, Survey Degrees of Freedom, Survey Aggregation (`aggregate_survey`), Survey-Aware Bootstrap (Phase 6), Replicate Weight Variance (Phase 6), DEFF Diagnostics (Phase 6), Subpopulation Analysis (Phase 6), Survey DGP (`generate_survey_did_data`)
- **Theory document**: `docs/methodology/survey-theory.md` — full Binder-Lumley derivation of design-based variance for modern DiD estimators, including influence-function machinery
- 13 dedicated `tests/test_survey*.py` files: `test_survey.py`, `test_survey_dcdh.py`, `test_survey_dcdh_replicate_psu.py`, `test_survey_estimator_validation.py`, `test_survey_phase3.py`, `test_survey_phase4.py`, `test_survey_phase5.py`, `test_survey_phase6.py`, `test_survey_phase7a.py`, `test_survey_phase8.py`, `test_survey_r_crossvalidation.py`, `test_survey_real_data.py`, `test_survey_staggered_ddd.py`
- Per-estimator survey hooks documented in the REGISTRY sections of every estimator that supports survey design (DiD/TWFE/MultiPeriodDiD, CS, SunAbraham, StackedDiD, ImputationDiD, TwoStageDiD, WooldridgeDiD, EfficientDiD, ContinuousDiD, DCDH, HAD, TripleDifference, StaggeredTripleDifference, TROP, SyntheticDiD). Scope is *estimators*; survey-capable diagnostics (e.g., `BaconDecomposition` Phase 3, `HonestDiD` survey-df handling) are tracked in their own sections.

**Outstanding for promotion:**
- Dedicated `tests/test_methodology_survey.py` (or split between TSL and replicate-weight surfaces) with Binder-equation-numbered Verified Components walk-through
- R parity benchmark against `survey::svyglm` / `survey::svycontrast` for the linear DiD case (`tests/test_survey_r_crossvalidation.py` exists; needs to be wired into a documented "Reference results" table here)
- Document deviations: PSU-level Hall-Mammen wild clustering as the bootstrap path when survey design is present (vs. R `survey`'s default analytical TSL); strata-vs-no-strata bit-equality not achievable due to RNG-path divergence between the per-stratum numpy loop and the batched `generate_survey_multiplier_weights_batch` call (see `docs/methodology/REGISTRY.md` HAD Stute survey-bootstrap section, "Distributional parity, NOT bit-exact" note, for the documented impossibility — distributional parity holds at large B, exact agreement at `atol=1e-10` does not)
- Consolidated "Outstanding cross-estimator gaps" enumerating which estimators still raise `NotImplementedError` on which survey-design combinations (e.g., Conley + survey, SyntheticDiD + Conley, HAD replicate weights on Stute family)

---

## Review Process Guidelines

### Review Checklist

For each estimator, complete the following steps:

- [ ] **Read primary academic source** - Review the key paper(s) cited in REGISTRY.md and write a `docs/methodology/papers/<name>-review.md` review if one doesn't exist
- [ ] **Compare key equations** - Verify implementation matches equations in REGISTRY.md
- [ ] **Run benchmark against reference implementation** - Execute `benchmarks/run_benchmarks.py --estimator <name>` if available; otherwise generate fixtures and document parity tolerances
- [ ] **Verify edge case handling** - Check behavior matches REGISTRY.md documentation
- [ ] **Check standard error formula** - Confirm SE computation matches reference (analytical, bootstrap, cluster-robust, survey-aware)
- [ ] **Write dedicated methodology test file** - `tests/test_methodology_<name>.py` with paper-equation-numbered assertions that correspond 1:1 to the Verified Components list
- [ ] **Document deviations** - Add notes explaining intentional differences with rationale, using one of the REGISTRY.md labels (`- **Note:**`, `- **Deviation from R:**`, `**Note (deviation from R):**`)

### When to Update This Document

1. **After completing a review**: Update status to "Complete" and add date, populate Verified Components / Corrections Made / Deviations sections
2. **When making corrections**: Document what was fixed in the "Corrections Made" section with file path and line number
3. **When identifying issues**: Add to "Outstanding Concerns" for future investigation
4. **When deviating from reference**: Document the deviation and rationale; cross-reference the REGISTRY.md `Note (deviation from R)` block
5. **When promoting from In Progress to Complete**: Replace the "Documentation in place" / "Outstanding for promotion" pair with the full Verified Components / Corrections Made / Deviations structure used by Complete entries
6. **When adding a new estimator to the library**: Add a row to the appropriate Status Summary table marked **In Progress** and a stub section under the matching category in Detailed Review Notes (Documentation in place / Outstanding for promotion) — same PR that introduces the estimator. New surfaces enter as In Progress because they ship with a REGISTRY.md entry and unit tests by definition.

### Deviation Documentation

When our implementation intentionally differs from the reference implementation, document:

1. **What differs**: Specific behavior or formula that differs
2. **Why**: Rationale (e.g., "defensive enhancement", "bug in R package", "follows updated paper")
3. **Impact**: Whether results differ in practice
4. **Cross-reference**: Update REGISTRY.md edge cases section using one of the recognized labels

Example:
```
**Deviation (2025-01-15)**: CallawaySantAnna returns NaN for t_stat when SE is non-finite,
whereas R's `did::att_gt` would error. This is a defensive enhancement that provides
more graceful handling of edge cases while still signaling invalid inference to users.
```

### Priority Order (2026-05-15)

Promotion priority for the **In Progress** entries, ordered by what's blocked on substantive review work (top of list = needs review next) vs. consolidation pass (bottom of list = mostly tracker walk-through):

**Substantive-review-blocked (no methodology test file, no paper review, no R parity):**

1. **PreTrendsPower** — small surface, established R package (`pretrends`), Roth (2022) is short.
2. **PowerAnalysis** — larger surface (MDE / power / sample size / simulation paths); REGISTRY already lists Bloom (1995) and Burlig et al. (2020) as primary sources; least urgent if the library's power-analysis utilities are not heavily used.
3. **PlaceboTests** — decide first whether to keep standalone or absorb into per-estimator diagnostic sections; methodologically lightweight either way.
4. **EfficientDiD** — no paper review on file; substantial implementation work (`tests/test_efficient_did.py` + validation tests) needs paper-vs-code audit against Chen, Sant'Anna & Xie (2025).
5. **ImputationDiD / TwoStageDiD** — natural pair (both single-treatment-effect-imputation methods). Each needs paper review, methodology file, R parity fixture against `didimputation` / `did2s`.

**Consolidation-pass-blocked (already has paper review or methodology file or R parity; mostly Verified Components walk-through):**

6. **StaggeredTripleDifference** — shares the primary paper (Ortiz-Villavicencio & Sant'Anna 2025) with TripleDifference, but no dedicated paper review on file yet; needs R parity (R fixtures gitignored — tracked in TODO.md, PR #245).
7. **ConleySpatialHAC** — paper review + committed R `conleyreg` goldens; needs dedicated methodology test file + summary R-parity table in this tracker.
8. **Survey Data Support** — cross-cutting feature; promotion requires the per-estimator integration paths to be locked down first.

---

## Related Documents

- [REGISTRY.md](docs/methodology/REGISTRY.md) — Academic foundations and key equations
- [docs/methodology/papers/](docs/methodology/papers/) — Per-paper retrospective reviews (Athey 2025, Butts 2021/2023, Clarke 2017, Colella et al. 2019, Conley 1999, de Chaisemartin 2026, Rambachan-Roth 2023, Wooldridge 2023)
- [docs/methodology/continuous-did.md](docs/methodology/continuous-did.md) — ContinuousDiD theory note
- [docs/methodology/survey-theory.md](docs/methodology/survey-theory.md) — Design-based variance estimation for modern DiD estimators
- [docs/methodology/REPORTING.md](docs/methodology/REPORTING.md) — Reporting conventions across estimators
- [ROADMAP.md](ROADMAP.md) — Feature roadmap
- [TODO.md](TODO.md) — Technical debt tracking, including deferred methodology items from code reviews
- [CLAUDE.md](CLAUDE.md) — Development guidelines
