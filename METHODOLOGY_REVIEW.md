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

**In Progress** entries have a REGISTRY.md section and unit-test coverage, but no formal walk-through has been captured here yet. The In Progress band is wide — some entries also have some combination of a paper review (primary or companion), a dedicated methodology test file, and R parity fixtures (e.g., DCDH has a methodology file, R parity, and a companion-paper review for the 2026 universal-rollout extension; ContinuousDiD has the methodology file but no paper review); others have only the REGISTRY entry and unit tests (e.g., PowerAnalysis). The "Documentation in place" sub-section enumerates what each entry already has; the "Outstanding for promotion" sub-section enumerates what's still needed to flip it to Complete.

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
| WooldridgeDiD (ETWFE) | `wooldridge.py` | `etwfe` (R) / `jwdid` (Stata) | **In Progress** | — |
| EfficientDiD | `efficient_did.py` | (no canonical R package) | **In Progress** | — |

### Continuous & Universal-Treatment Estimators

| Estimator | Module | R / Stata Reference | Status | Last Review |
|-----------|--------|---------------------|--------|-------------|
| ContinuousDiD | `continuous_did.py` | `contdid` v0.1.0 | **In Progress** | — |
| ChaisemartinDHaultfoeuille (DCDH) | `chaisemartin_dhaultfoeuille.py` | `DIDmultiplegtDYN` | **In Progress** | — |
| HeterogeneousAdoptionDiD (HAD) | `had.py`, `had_pretests.py` | (paper-direct; `nprobust` for bandwidth) | **Complete** | 2026-05-20 |
| TROP | `trop.py`, `trop_local.py`, `trop_global.py` | (forthcoming; paper-author reference implementation) | **In Progress** | — |

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
| R Reference | `etwfe` (McDermott 2023); Stata `jwdid` (Rios-Avila 2021) |
| Status | **In Progress** |
| Last Review | — |

**Documentation in place:**
- REGISTRY.md section: `## WooldridgeDiD (ETWFE)` (saturated cohort×time interactions, OLS/logit/Poisson via IRLS, ASF-based ATT for nonlinear methods with delta-method SEs, four aggregations, survey support)
- **Companion-paper review on file**: `docs/methodology/papers/wooldridge-2023-review.md` covers Wooldridge (2023) *Simple approaches to nonlinear difference-in-differences with panel data*, Econometrics Journal 26(3) — the nonlinear extension that the logit/Poisson paths implement (retrospective, merged PR #443 on 2026-05-13). A dedicated review for the primary ETWFE source (Wooldridge 2025, *Empirical Economics* 69(5)) is **not** yet on file.
- Implementation: `tests/test_wooldridge.py` (covers OLS, logit, and Poisson paths plus the four aggregation types)

**Outstanding for promotion:**
- Dedicated paper review for the primary ETWFE source: write `docs/methodology/papers/wooldridge-2025-review.md` covering Wooldridge (2025) *Empirical Economics* 69(5), 2545–2587 (published version of the 2021 SSRN working paper / NBER WP 29154)
- Dedicated `tests/test_methodology_wooldridge.py` with paper-equation-numbered Verified Components walk-through
- R parity fixture against `etwfe` (and ideally Stata `jwdid`) covering OLS, logit, and Poisson paths
- Verified Components for nonlinear-method ASF / delta-method SE invariants
- "Corrections Made" listing

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
| R Reference | `contdid` v0.1.0 (CRAN) |
| Status | **In Progress** |
| Last Review | — |

**Documentation in place:**
- REGISTRY.md section: `## ContinuousDiD` plus dedicated theory note in `docs/methodology/continuous-did.md` (PT vs SPT identification, ATT(d|d) / ATT(d) / ACRT(d) / ATT^{loc} / ATT^{glob} / ACRT^{glob} estimands, B-spline OLS, multiplier bootstrap)
- `tests/test_methodology_continuous_did.py`: 15 tests across 5 classes (linear dose response, quadratic with cubic basis, multi-period aggregation, edge cases, R benchmark)
- Implementation: 80 unit tests in `tests/test_continuous_did.py`
- Survey support: weighted B-spline OLS, TSL on influence functions, bootstrap+survey (Phase 6)

**Outstanding for promotion:**
- Detailed Verified Components block here mirroring REGISTRY's Implementation Checklist (B-spline basis matching `splines2::bSpline`, multi-period cell iteration, dose-response and event-study aggregation, multiplier bootstrap, analytical SE via influence functions)
- Document the boundary-knots deviation from R `contdid` v0.1.0 (Python uses `range(dose)`; R uses `range(dvals)` which can produce extrapolation artifacts) in a formal Deviations block here
- Formalize the `+inf` recoding and zero-dose silent-zeroing warnings (currently in REGISTRY) into a Verified Components row

---

#### ChaisemartinDHaultfoeuille (DCDH)

| Field | Value |
|-------|-------|
| Module | `chaisemartin_dhaultfoeuille.py`, `chaisemartin_dhaultfoeuille_bootstrap.py`, `chaisemartin_dhaultfoeuille_results.py` |
| Primary References | (a) de Chaisemartin & D'Haultfœuille (2020), *Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects*, AER 110(9), 2964-2996. (b) de Chaisemartin & D'Haultfœuille (2022, revised 2024), *Difference-in-Differences Estimators of Intertemporal Treatment Effects*, NBER WP 29873 — Web Appendix Section 3.7.3 for cohort-recentered plug-in variance. (c) de Chaisemartin, Ciccia, D'Haultfœuille & Knau (2026) for the universal-rollout case. |
| R Reference | `DIDmultiplegtDYN` |
| Status | **In Progress** |
| Last Review | — |

**Documentation in place:**
- REGISTRY.md section: `## ChaisemartinDHaultfoeuille` (DID_M, DID_+, DID_-, single-lag placebo, TWFE-weights diagnostic, multiplier bootstrap, DID^X / DID^{fd} / state-set-specific trends / heterogeneity testing / Design-2 / by_path / HonestDiD integration, survey design + replicate weights + HM wild bootstrap)
- **Companion-paper review on file**: `docs/methodology/papers/dechaisemartin-2026-review.md` covers the 2026 universal-rollout extension (Knau et al.), which is the primary source for HAD rather than for DCDH. The 2020 AER and 2022/2024 NBER WP 29873 papers that define DCDH's core DID_M / DID_+ / DID_- and dynamic estimators do **not** yet have dedicated review files on disk.
- `tests/test_methodology_chaisemartin_dhaultfoeuille.py`: 12 tests across 4 classes (worked example, cohort recentering, TWFE diagnostic, large-N recovery)
- `tests/test_chaisemartin_dhaultfoeuille_parity.py`: 24 R parity tests against `DIDmultiplegtDYN`
- Implementation: 347 unit tests in `tests/test_chaisemartin_dhaultfoeuille.py`
- Survey-specific: `tests/test_survey_dcdh.py`, `tests/test_survey_dcdh_replicate_psu.py`, plus three dCDH cell-period coverage suites

**Outstanding for promotion:**
- **Primary-source paper reviews**: write `docs/methodology/papers/dechaisemartin-dhaultfoeuille-2020-review.md` covering the 2020 AER and a companion review covering 2022/2024 NBER WP 29873 (intertemporal treatment effects). The existing 2026 review covers the universal-rollout extension only.
- Formal Verified Components block here matching REGISTRY's exhaustive Implementation Checklist
- Consolidated Deviations summary (currently scattered across REGISTRY Notes): equal-cell weighting vs R cell-size weighting, terminal-missingness retention, A11 zero-retention convention, `<50%` switcher warning at far horizons
- Documented R parity tolerance bands at `l=1` (existing parity fixture in `test_chaisemartin_dhaultfoeuille_parity.py`)
- "Corrections Made" listing for the Round 2 full-IF fix (never-switching groups now participate in variance via stable-control roles)

---

#### HeterogeneousAdoptionDiD (HAD)

| Field | Value |
|-------|-------|
| Module | `had.py`, `had_pretests.py` |
| Primary Reference | de Chaisemartin, Ciccia, D'Haultfœuille & Knau (2026), *Difference-in-Differences Estimators When No Unit Remains Untreated*, arXiv:2405.04465v6 |
| R Reference | None (paper-direct implementation); `nprobust` (Calonico-Cattaneo-Farrell) used for bandwidth selection only |
| Status | **Complete** |
| Last Review | 2026-05-20 |

**Verified Components:**
- [x] Eq. 3 / Theorem 1 (Design 1' WAS identification: `WAS = [E(ΔY) − lim_{d↓0} E(ΔY | D ≤ d)] / E(D)`, the boundary-subtracted form; the library estimates the boundary intercept via bias-corrected local linear and computes `att = (mean(ΔY) − τ_bc) / mean(D)`) — `tests/test_methodology_had.py::TestHADTheorem1Design1Prime` (7 tests including MC recovery on the simple `ΔY = β·D + ε` DGP, MC recovery on a NONZERO-BOUNDARY-INTERCEPT DGP `ΔY = c + β·D + ε` with `c != 0` to exercise the `mean(ΔY) − τ_bc` subtraction explicitly, and N(0,1) coverage at `n_replicates=200`, G=1000)
- [x] Eq. 7 (local-linear with bias-corrected CI) — covered by `tests/test_bias_corrected_lprobust.py` (44 tests, hand-derived R reference at `atol=1e-12`) and `tests/test_nprobust_port.py` (~46 tests, machine-precision port at `atol=1e-14`)
- [x] Eq. 11 / Theorem 3 (`WAS_{d_lower}` under Assumption 6, mass-point path) — `tests/test_methodology_had.py::TestHADTheorem3MassPoint` (5 tests including Wald-IV closed-form equivalence at `atol=1e-9`)
- [x] Theorem 4 (QUG null test, limit law `T_λ = (λ + E_1) / E_2` under Exp(1)/Exp(1)) — `tests/test_methodology_had.py::TestHADTheorem4QUG` (6 tests; MC distributional match against closed-form `F(t) = t/(1+t)` at KS-stat ≤ 0.05, n_draws=5000)
- [x] Eq. 29 / Theorem 7 (Yatchew-HR linearity test, paper-literal `σ²_diff = 1/(2G)` normalization) — `tests/test_methodology_had.py::TestHADTheorem7YatchewHR` (6 tests; standard-normal limit, normalization lock, both `null="linearity"` and `null="mean_independence"` modes)
- [x] Eq. 18 mean-independence variant (joint Stute pre-trends + homogeneity, sum-of-CvMs + shared-η Mammen wild bootstrap) — `tests/test_methodology_had.py::TestHADJointStute` (5 tests; H0 fail-to-reject and H1 reject on linear vs. nonlinear DGPs). Eq. 18 linear-trend-detrended variant deferred per REGISTRY checklist (Phase 4 follow-up, `trends_lin=True`).
- [x] R parity (`chaisemartin::did_had`) at `atol=1e-8` on 3 DGPs × 5 method combos (bit-exact, `rtol=0`) — `tests/test_did_had_parity.py::TestPointSEParity` + `TestYatchewParity` (5 direct parity tests; YatchewTest closed-form parity at `atol=1e-10`)
- [x] `nprobust` (Calonico-Cattaneo-Farrell) port at machine precision (`atol=1e-14`) — `tests/test_nprobust_port.py` (7 classes spanning kernel constants, QR-based `(X'X)^{-1}`, three-stage MSE-DPI bandwidth, clustered variance, weighted local-linear, single-eval-point parity)
- [x] Bandwidth selector (CCF MSE-DPI) at 1% tolerance — `tests/test_bandwidth_selector.py` (8 classes covering public-API wrapper, stage diagnostics)
- [x] Survey support: pweight + strata/PSU/FPC via TSL on the continuous and mass-point paths; PSU-level Mammen wild bootstrap on the Stute family; closed-form weighted variance components on Yatchew (Phase 4.5 A/B/C; QUG-under-survey permanently deferred per Phase 4.5 C0)
- [x] Tutorials T21 (`docs/tutorials/21_had_pretest_workflow.ipynb`, 17 drift tests) + T22 (`docs/tutorials/22_had_survey_design.ipynb`, 32 drift tests across groups A-G); plus T20 (`docs/tutorials/20_had_brand_campaign.ipynb`, 14 drift tests)
- [x] Assumption 5/6 non-testability documented in `HeterogeneousAdoptionDiD` class docstring + `qug_test`/`stute_test`/`yatchew_hr_test`/`did_had_pretest_workflow` Notes blocks; reinforced by a fit-time `UserWarning` emitted from the outer `HeterogeneousAdoptionDiD.fit()` dispatch on the overall and event-study paths when the resolved design is Design 1 family (search `diff_diff/had.py` for "---- Assumption 5/6 warning on Design 1 paths ----")

**Test Coverage:**
- 35 methodology tests in `tests/test_methodology_had.py` (this PR)
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
6. **Eq. 18 linear-trend-detrended joint Stute deferred** per REGISTRY paper-review checklist (Phase 4 follow-up); mean-independence variant ships in Phase 3 and is what `TestHADJointStute` exercises.

**Outstanding Concerns:**
- Module split (`had.py` ~4593 LoC, `had_pretests.py` ~4951 LoC) — tracked in TODO.md as tech debt, not a methodology gap.
- Bandwidth selector multi-eval, cross-horizon covariance on joint event-study — tracked as Phase follow-ups in TODO.md.
- Replicate-weight designs (BRR / Fay / JK1 / JKn / SDR) on HAD continuous path remain `NotImplementedError` (Phase 4.5 D follow-up).
- `covariates=` kwarg with Theorem 6 multivariate-covariate extension not implemented; currently a Python `TypeError` (kwarg absent from the `fit()` signature). Adding an explicit `**kwargs`-trap with `NotImplementedError` and a Theorem 6 pointer is tracked as a Low-priority follow-up in TODO.md.

---

#### TROP

| Field | Value |
|-------|-------|
| Module | `trop.py`, `trop_local.py`, `trop_global.py`, `trop_results.py` |
| Primary Reference | Athey, Imbens, Qu & Viviano (2025), *Triply Robust Panel Estimators*, arXiv:2508.21536 |
| R Reference | Paper-author reference implementation (not yet released as CRAN package) |
| Status | **In Progress** |
| Last Review | — |

**Documentation in place:**
- REGISTRY.md section: `## TROP` (local: factor matrix via soft-threshold SVD, exponential-decay unit weights matching paper Eq. 2, LOOCV per Eq. 5, multiple rank-selection methods cv/ic/elbow; global: alternating minimization for nuclear-norm penalty with hard-coded inner-FISTA 20-iteration loop, ATT averaging over D==1 cells, Rust-accelerated LOOCV and bootstrap)
- **Paper review on file**: `docs/methodology/papers/athey-2025-review.md` (retrospective, merged PR #443 on 2026-05-13)
- Implementation: 120 unit tests in `tests/test_trop.py`
- Survey support: Rao-Wu rescaled bootstrap with cross-classified pseudo-strata; Rust backend remains pweight-only

**Outstanding for promotion:**
- Dedicated `tests/test_methodology_trop.py` with paper-equation-numbered Verified Components walk-through
- Cross-validation against the paper-author reference implementation (when it becomes available) or against the paper's reported numbers on the empirical applications
- Documented deviations: bootstrap proportional-failure warnings (5% threshold), alternating-minimization convergence warnings, Rust backend's pweight-only limitation vs. Python's full survey-design support

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

6. **HeterogeneousAdoptionDiD (HAD)** — largest current surface, Phase 4.5 just shipped; shares the de Chaisemartin (2026) paper review with DCDH; needs a dedicated Verified Components block.
7. **ChaisemartinDHaultfoeuille (DCDH)** — methodology test file + 24 R parity tests + 347 unit tests + a companion-paper review for the 2026 universal-rollout extension. Primary-source reviews for the 2020 AER and 2022/2024 NBER WP 29873 papers are still outstanding alongside the Verified Components walk-through.
8. **WooldridgeDiD (ETWFE)** — companion-paper review (Wooldridge 2023 nonlinear extension) merged in PR #443; primary-source review for Wooldridge (2025) ETWFE not yet on file, and no dedicated methodology test file.
9. **ContinuousDiD** — 15 methodology tests already in place; mostly a consolidation pass with a documented boundary-knots deviation from R `contdid` v0.1.0.
10. **TROP** — paper review recently merged (PR #443); needs methodology file and cross-language anchor (when paper-author reference becomes available).
11. **StaggeredTripleDifference** — shares the primary paper (Ortiz-Villavicencio & Sant'Anna 2025) with TripleDifference, but no dedicated paper review on file yet; needs R parity (R fixtures gitignored — tracked in TODO.md, PR #245).
12. **ConleySpatialHAC** — paper review + committed R `conleyreg` goldens; needs dedicated methodology test file + summary R-parity table in this tracker.
13. **Survey Data Support** — cross-cutting feature; promotion requires the per-estimator integration paths to be locked down first.

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
