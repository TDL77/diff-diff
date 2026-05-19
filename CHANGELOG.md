# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **PreTrendsPower R `pretrends` parity goldens (PR-C closes PR-B's deferred R-parity row).** JSON goldens at `benchmarks/data/r_pretrends_golden.json` generated from the committed `benchmarks/R/generate_pretrends_golden.R` script against `jonathandroth/pretrends` commit `122731d082` (package version 0.1.0, R 4.5.2). 4 fixtures cover regular K=3 grid (`uniform_3_pre_periods_no_anticipation`), irregular K=3 grid `[-5,-3,-1]` (`irregular_pre_periods` — locks the PR-B Step 4 γ-unit linear-weight fix), anticipation-shifted K=4 grid (`anticipation_shifted`), and K=1 closed form (`single_pre_period_closed_form` — Roth Proposition 2 univariate truncated-normal). `TestPretrendsParityR` in `tests/test_methodology_pretrends.py` now active (4 tests): NIS power vs R `pretrends::pretrends()` at `atol=1e-4` across all 4 fixtures × 4 γ values; γ_p MDV vs R `slope_for_power()` at `atol=1e-4` across all 4 fixtures × 2 target_power values; end-to-end `fit()` on irregular grid vs R γ_p at `atol=1e-4` (locks the full `fit() → _extract_pre_period_params → _get_violation_weights → _compute_mdv_nis` chain through the public API); K=1 three-way cross-check (Python ≡ analytical truncated-normal closed form `1 - Φ(z - γ/σ) + Φ(-z - γ/σ)` at `atol=1e-7`; both within `atol=1e-4` of R). Tolerance rationale: R hardcodes `thresholdTstat.Pretest=1.96` while Python uses `scipy.stats.norm.ppf(0.975) = 1.959963984540054` (`dz ≈ 3.6e-5`); R `slope_for_power` uses `uniroot(tol = .Machine$double.eps^0.25 ≈ 1.22e-4)` versus Python `brentq(xtol=2e-12)`; the inverse-solver tolerance gap dominates γ_p, and `mvtnorm::pmvnorm` (R) vs `scipy.stats.multivariate_normal.cdf` (Python) Genz-Bretz randomized-lattice differences bound the K=4 NIS power gap at ~5e-5. `METHODOLOGY_REVIEW.md` PreTrendsPower row promoted `**Complete** (R parity pending)` → `**Complete**`. Roth (2022) paper review's `R \`pretrends\` package version pin (provisional)` Gaps bullet struck. Closes the PR-C TODO row.

## [3.4.0] - 2026-05-19

### Added
- **`TwoWayFixedEffects(vcov_type in {"hc2","hc2_bm"})` now supported** (`diff_diff/twfe.py:155`). Lifts Gate 1 of the six HC2/HC2-BM `NotImplementedError` gates — the last absorbed-FE gate (DiD-absorb shipped earlier, MPD-absorb shipped earlier, MPD cluster+contrast-DOF shipped earlier in this release). Unlike DiD / MPD, TWFE has no `absorb=` / `fixed_effects=` parameter to swap (unit + time FEs are baked into the estimator's identity), so the same auto-route trick isn't applicable. Instead, `TwoWayFixedEffects.fit()` bypasses the within-transform when `vcov_type in {"hc2","hc2_bm"}` and stacks the full-dummy design `[intercept, treated×post, covariates, factor(unit), factor(time)]` explicitly, then runs OLS through the standard `solve_ols` path so the leverage correction `h_ii = x_i' (X'X)^{-1} x_i` and CR2 Bell-McCaffrey adjustment `A_g = (I - H_gg)^{-1/2}` compute on the full FE projection (FWL preserves coefficients and residuals but NOT the hat matrix). Verified at `atol=1e-10` vs `lm(y ~ treat_post + factor(unit) + factor(post)) + sandwich::vcovHC(type="HC2")` for HC2, vs `clubSandwich::vcovCR(cluster=seq_len(n), type="CR2") + coef_test()$df_Satt` for the singleton-cluster one-way HC2-BM Satterthwaite DOF, and vs `vcovCR(cluster=unit, type="CR2")` for the auto-cluster CR2-BM path (new `twfe_two_period` scenario in `benchmarks/data/clubsandwich_cr2_golden.json`). **Auto-cluster default:** TWFE's unit auto-cluster is preserved on `hc2_bm` (routes to CR2-BM at unit) and on `hc2 + wild_bootstrap` (the bootstrap consumes the cluster structure for resampling regardless of the analytical sandwich choice); dropped on explicit `hc2 + analytical` to match the one-way contract (the linalg validator rejects `hc2 + cluster_ids`). **User-visible surface change** (matches the DiD-absorb / MPD-absorb disclosures above): under `vcov_type in {"hc2","hc2_bm"}`, `result.coefficients`, `result.vcov`, `result.residuals`, `result.fitted_values`, and `result.r_squared` reflect the full-dummy fit rather than the within-transformed reduced fit (FE-dummy entries are included alongside the `"ATT"` key; `r_squared` is computed on the un-demeaned outcome; residuals / fitted are on the original scale; `len(result.coefficients) == result.vcov.shape[0]` invariant upheld). `result.att`, its SE, and analytical inference are unchanged (FWL-equivalent). HC1 / CR1 / Conley / classical paths remain on the within-transform. **Survey-design scope** (mirrors DiD-absorb): when `survey_design=` is supplied, the existing survey variance path (Taylor-series linearization or replicate-weight variance) takes precedence over the analytical HC2/HC2-BM sandwich; the full-dummy build only changes FE handling. **Rejected combos:** `vcov_type in {"hc2","hc2_bm"}` + replicate-weight survey designs (BRR / Fay / JK1 / JKn / SDR) raises `NotImplementedError` at `twfe.py:~233` because the replicate path re-demeans per replicate, which doesn't compose with the full-dummy build (would require per-replicate full-dummy refit); workaround: use `vcov_type="hc1"` for replicate-weight CR1. `hc2_bm + weights` remains blocked at the linalg validator (same gate as Gates 4-5 — weighted CR2 variants). New tests: `tests/test_estimators_vcov_type.py::TestFitBehavior` (9 tests: rejection flip → behavioral; refactor regression vs `DifferenceInDifferences(fixed_effects=[unit, time])` at `atol=1e-12`; auto-cluster default coverage on `hc2_bm`; explicit `hc2 + analytical` no-auto-cluster; `hc2 + wild_bootstrap` auto-cluster preserved; `hc2 / hc2_bm + replicate` rejection; always-treated unit finite ATT; coefficients-vs-vcov alignment invariant); `tests/test_methodology_twfe.py::TestTWFEHC2RParity` (3 R-parity tests at `atol=1e-10`).
- **Agent-discoverability contract test (`tests/test_agent_discoverability.py`).** New static-snapshot test pinning the agent-facing surface introduced by PR #464: `__all__` membership of `agent_workflow` / `profile_panel` / `get_llm_guide` / `practitioner_next_steps` / `BusinessReport`; `dir(diff_diff)` head-first ordering against `_AGENT_FACING_ORDER` (catches drift in the `_OrderedName` `__lt__` ordering trick); `_OrderedName` `isinstance(_, str)` + str-method compatibility; `dir()` full-namespace + `inspect.getmembers` parity; top-level `__doc__` first-paragraph mention of `agent_workflow` + named references to the 5-step workflow primitives; `agent_workflow()` script content references each downstream helper by name; canonical estimator class names (CallawaySantAnna, ContinuousDiD, HeterogeneousAdoptionDiD, etc.) remain importable. No live API calls; runs in the default pytest suite. Closes [issue #461](https://github.com/igerber/diff-diff/issues/461) (snapshot variant — live-agent regression test deferred to a separate follow-up that depends on causal-llm-eval packaging its harness). Also closes the `__dir__()` contract-test row from `TODO.md` that PR #464 deferred here.
- **`diff_diff.agent_workflow(df, unit=..., time=..., treatment=..., outcome=...)` — stateless orchestrator for LLM-agent discoverability** (`diff_diff/agent_workflow.py`). Prints (and returns as dict) a copy-pasteable 5-step workflow with the caller's column names templated in: `profile_panel` → `get_llm_guide("autonomous")` → `<Estimator>(...).fit(df, ...)` → `practitioner_next_steps(result)` → `BusinessReport(result).full_report()`. The function calls nothing internally and does not inspect `df`; it is a guided tour, not a router. Surfaces the canonical workflow primitives (`profile_panel`, `get_llm_guide`, `practitioner_next_steps`, `BusinessReport`) that cold-start agent dry-passes at [igerber/causal-llm-eval](https://github.com/igerber/causal-llm-eval) showed agents practically never reach for on their own. Output structure: `{"profile_call", "guide_call", "fit_candidates", "validation_calls", "reporting_call", "script"}`; `fit_candidates` is a flat list of estimator/diagnostic class names referenced in the workflow patterns (each must remain importable on `diff_diff`, locked by `tests/test_agent_workflow.py::test_fit_candidates_all_importable`). Closes [issue #460](https://github.com/igerber/diff-diff/issues/460).
- **Top-level `__doc__` rewritten to lead with the agent workflow** (`diff_diff/__init__.py`). `help(diff_diff)` now opens with the `agent_workflow(df, ...)` recommendation as the first non-blank paragraph; `get_llm_guide("full")` and `get_llm_guide("practitioner")` pointers preserved for the existing `tests/test_guides.py::test_module_docstring_mentions_helper` guard.
- **`dir(diff_diff)` now surfaces agent-facing entrypoints first** via a module-level `__dir__()` override paired with a small `_OrderedName(str)` subclass that subverts CPython's unconditional alphabetic sort (PyList_Sort respects `__lt__` on the elements). Agent-facing names (`agent_workflow`, `profile_panel`, `get_llm_guide`, `practitioner_next_steps`, `BusinessReport`, `DiagnosticReport`) appear at the head of the list; the remainder stays alphabetic via the `str.__lt__` fallback. The underlying `__all__` membership is **unchanged** and `from diff_diff import *` semantics are unaffected (driven by `__all__`, not `dir()`). Elements are `isinstance(x, str)` and compatible with `inspect.getmembers`, dict-key lookup, f-strings, and standard `str` methods; tooling that re-sorts via `sorted(dir(diff_diff))` will see priority order (use `sorted(dir(diff_diff), key=str)` to recover plain alphabetic if needed). Internal: `_AGENT_FACING_ORDER` tuple is read by the new `tests/test_agent_discoverability.py` contract test (PR B). Addresses [issue #460](https://github.com/igerber/diff-diff/issues/460) item 3.
- **`MultiPeriodDiD(cluster=..., vcov_type="hc2_bm")` now supported** (`diff_diff/estimators.py:1657`). Pre-PR the combination raised `NotImplementedError` because the cluster-aware CR2 Bell-McCaffrey Satterthwaite DOF for the post-period-average ATT (`avg_att = (1/n_post) Σ_{t ≥ t_treat} β_t`) was not implemented — only the per-coefficient case existed in `_compute_cr2_bm`. New `_compute_cr2_bm_contrast_dof` helper in `diff_diff/linalg.py` generalizes the per-coefficient loop to arbitrary `(k, m)` contrast matrices using the identical Pustejovsky-Tipton 2018 Section 4 algebra; `_compute_cr2_bm` is refactored to call it with `contrasts=eye(k)` so the existing per-coefficient parity to clubSandwich's `coef_test$df_Satt` is preserved (refactor regression at atol=1e-10). `MultiPeriodDiD.fit()` extends its existing avg_att DOF block to branch on `effective_cluster_ids`: one-way `_compute_bm_dof_from_contrasts` when None, cluster-aware `_compute_cr2_bm_contrast_dof` otherwise. Cluster IDs are per-observation length `n` and are NOT subscripted by the rank-deficient column-drop mask. R parity verified at atol=1e-10 against clubSandwich's `Wald_test(constraints=matrix(c, 1), test="HTZ")$df_denom` on the new `mpd_clustered_avg_att_dof` fixture in `benchmarks/data/clubsandwich_cr2_golden.json` (Wald_test's HTZ on a 1-row constraint matrix yields the Satterthwaite t-test DOF). Per-coefficient `period_effects[t].p_value` / `conf_int` and `avg_att` `avg_p_value` / `avg_conf_int` now reflect the correct Satterthwaite DOF rather than the n-k fallback under cluster+hc2_bm. Weighted CR2-BM (`survey_design=` paths) remains a separate gate. New tests: `tests/test_linalg_hc2_bm.py::TestCR2BMContrastDOF` (4 tests: refactor regression, R-parity, shape validation, cluster-count validation); existing `test_multi_period_cluster_plus_hc2_bm_rejected` flipped to behavioral `test_multi_period_cluster_plus_hc2_bm_produces_finite_inference`.
- **PreTrendsPower: NIS box probability as the new primary test form (PR-B methodology audit, Roth 2022).** Implements Roth (2022) Section II.A-B no-individually-significant (NIS) box probability `P(β̂_pre ∈ B_NIS(Σ))` as the new default `pretest_form='nis'` on `PreTrendsPower`, `compute_pretrends_power`, and `compute_mdv`. The Wald noncentral-χ² form previously shipped as the implicit default is now opt-in via `pretest_form='wald'` and remains as a paper-supported alternative (Propositions 1+3+4 all apply — the Wald ellipsoid is convex). Computation uses `scipy.stats.multivariate_normal.cdf` with `lower_limit=` for the rectangular box probability on the centered change-of-variable `Y = β̂_pre - δ_pre ~ N(0, Σ_22)`; the MDV is solved via doubling expansion + `optimize.brentq` bisection with a 1000-cap non-convergence fallback returning `np.inf`. New private helpers `_compute_power_nis` and `_compute_mdv_nis`; the existing methods are renamed `_compute_power_wald` and `_compute_mdv_wald` with byte-identical math, and `_compute_power` / `_compute_mdv` become dispatchers on `self.pretest_form`. `power_curve()` and `PreTrendsPowerResults.power_at()` inherit the dispatch (power_at via the new persisted `pretest_form` field on the result). The `summary()` / `to_dict()` / `to_dataframe()` outputs dispatch on `pretest_form` — NIS fits print "NIS box probability: ..." instead of "Non-centrality parameter: ...".
- **PreTrendsPower: full Σ_22 routing on CS and SA event-study adapters (PR-B methodology audit, Σ_22 fidelity).** The shipped `compute_pretrends_power` adapter previously hard-coded `np.diag(ses**2)` for both `CallawaySantAnnaResults` and `SunAbrahamResults` regardless of whether the analytical event-study VCV was available, dropping the off-diagonal correlations Roth's framework relies on. PR-B routes non-bootstrap CS fits through the full `event_study_vcov` sub-block (already persisted at `staggered_results.py:126-128`) and extends `SunAbrahamResults` to also persist `event_study_vcov` + `event_study_vcov_index` constructed via the W-matrix aggregation `event_study_vcov = W @ vcov_cohort @ W.T` where W is the cohort-aggregation matrix (`|event_times| × n_interactions` sparse matrix with `W[i, j] = cohort_weights[e_i][g]` at column `j = coef_index_map[(g, e_i)]`). The new shared helper `_extract_event_study_vcov_subblock` at module level in `pretrends.py` consumes the full VCV when available with a `.index()` lookup on `event_study_vcov_index`; defensive ValueError on label mismatch. Bootstrap fits and replicate-weight survey fits clear `event_study_vcov` (mirroring the CS bootstrap-clear pattern at `staggered.py:2032-2036`) so they fall through to `diag(ses^2)` and the analytical VCV is never mixed with bootstrap/replicate SE overrides downstream. Diagonal-entry sanity check verifies that `event_study_vcov[i, i] = se(e_i)^2` matches the existing per-event-time SE computation in `_compute_iw_effects` at `atol=1e-10`. **Backwards-compatible field additions**: new `event_study_vcov` + `event_study_vcov_index` fields on `SunAbrahamResults` default to `None`, so existing consumers that don't read them see no change.
- **`PreTrendsPowerResults` now persists fitted `violation_weights` + `pretest_form` + `nis_box_probability` (PR-B Step 5).** New optional fields on the result dataclass enable `power_at(M)` to work for ALL four violation types (linear / constant / last_period / **custom**) on fresh fits, by reading the stored weights directly instead of reconstructing from `violation_type` alone. The PR-A R18 NotImplementedError silent-failure guard for `violation_type='custom'` is retained ONLY for legacy serialized results (`violation_weights=None`) — fresh fits no longer hit it.
- **Helper API: `compute_pretrends_power` and `compute_mdv` now accept `violation_weights` and `pretest_form` (PR-B Step 6).** Closes the PR-A R18 helper/class API gap that previously made `violation_type='custom'` unusable from the helper functions. Helpers now forward both new parameters to the underlying `PreTrendsPower` class. Default `pretest_form='nis'` matches the class default. All existing helper call sites in `test_pretrends.py` and `test_pretrends_event_study.py` continue to pass without changes because the form-invariance of most assertions allowed the default flip with only 3 tests needing targeted updates.
- **NEW `tests/test_methodology_pretrends.py` (PR-B Step 7).** Roth (2022) Section II.A-B paper-equation-numbered Verified Components walk-through. 8 classes, 30+ tests covering K=1 closed-form (Proposition 2 proof), NIS box probability via MC simulation cross-check, Propositions 1-4 simulation parity, linear-units γ-scale verification on regular / irregular / pandas.Period grids, custom-weight persistence regression, JSON-serializability of `to_dict`, CS/SA full-VCV adapter regression, helper API end-to-end, NIS-vs-Wald differentiation, and skip-gated `TestPretrendsParityR` stubs for PR-C R-package goldens.
- **`benchmarks/R/generate_pretrends_golden.R` (PR-B Step 12).** R generator script for the PR-C deferred goldens. Script committed with a `<PR-C-PIN>` placeholder commit reference; PR-C pins the audited `pretrends` revision, runs the script, commits the JSON goldens at `benchmarks/data/r_pretrends_golden.json`, and activates the parity tests.
- **`MultiPeriodDiD(absorb=..., vcov_type in {"hc2", "hc2_bm"})` now supported** (`diff_diff/estimators.py:1476`). Mirrors the DiD-absorb auto-route shipped earlier in this release: when `absorb=` is paired with `vcov_type in {"hc2","hc2_bm"}`, `MultiPeriodDiD.fit()` promotes the absorb columns to `fixed_effects=` internally so the existing full-dummy-design code path computes the algebraically correct vcov on the event-study design (`treated + period_X dummies + treated:period_X interactions + factor(unit)`). Verified at ~1e-10 vs `lm() + sandwich::vcovHC(type="HC2")` and `lm() + clubSandwich::vcovCR(cluster=1:n, type="CR2")` on a 5-cohort × 5-period event-study fixture (new `tests/test_estimators_vcov_type.py::TestMPDAbsorbedFERParity` against `benchmarks/data/clubsandwich_cr2_golden.json` scenario `mpd_absorbed_fe_did`). HC1/CR1 paths on `absorb=` are unchanged (no leverage term). (`TwoWayFixedEffects(vcov_type in {"hc2","hc2_bm"})` was lifted later in this same release via an inline full-dummy build — see the top entry; TWFE has no `fixed_effects=` equivalent inside the estimator, so it gets a separate full-dummy branch rather than the absorb→fixed_effects parameter swap used here.) **Behavioral note (full `MultiPeriodDiDResults` surface change under auto-route):** under the auto-route, the entire returned `MultiPeriodDiDResults` reflects the full-dummy fit rather than the within-transformed fit — `result.coefficients`, `result.vcov`, `result.residuals`, `result.fitted_values`, `result.r_squared` all include the FE-dummy entries / un-demeaned values. `result.period_effects[t].effect` / `.se` / `.p_value` / `.conf_int` and `result.avg_att` / `.avg_se` are invariant to this routing (FWL guarantee). MPD requires a time-invariant ever-treated indicator that lies in the span of the intercept and the post-auto-route unit FE dummies (the exact alias depends on the omitted FE reference category under `pd.get_dummies(drop_first=True)`, not just on "the sum of treated-cohort unit dummies"), so `solve_ols` drops one column from that collinear set under R-style rank-deficiency handling. Which specific column is dropped is pivot-order and dummy-coding dependent (in the shipped parity fixture it is a never-treated unit dummy, not the `treated` main effect itself). The per-period interaction coefficients (`treated:period_X`) and `avg_att` are identified and invariant to that choice; parity tests target those rather than the `treated` main effect. **Survey-design scope (replicate weights):** when `survey_design=` uses replicate weights, the auto-route short-circuits the absorb-refit branch at `estimators.py:1693` and routes through the standard `compute_replicate_vcov` path on the fixed full-dummy design — correct because the design does not depend on replicate weights so no per-replicate refit is needed. **Redundant time-FE skip:** when the routed (or directly-supplied) `fixed_effects` list contains the `time` column, MPD silently skips emitting `<time>_<X>` dummies for that entry because the design already absorbs the time dimension via the non-reference period dummies; without the skip, the two blocks would collide on dummy names and the `coefficients` dict would silently collapse duplicates under `var_names`-keyed construction, breaking the coefficients-vs-vcov alignment that downstream consumers rely on. This applies to both the new `absorb=` auto-route and the pre-existing `fixed_effects=[<time_col>]` invocation.
- **`DifferenceInDifferences(absorb=..., vcov_type in {"hc2", "hc2_bm"})` now supported** (`diff_diff/estimators.py:382`). Previously raised `NotImplementedError` because the HC2 leverage correction and CR2 Bell-McCaffrey DOF depend on the FULL FE hat matrix, while within-transformation (FWL) preserves coefficients and residuals but not the hat. Lift via internal auto-route: when `absorb=` is paired with `vcov_type in {"hc2","hc2_bm"}`, the fit promotes the absorb columns to `fixed_effects=` internally so the existing full-dummy-design code path computes the algebraically correct vcov. Empirically matches `lm() + sandwich::vcovHC(type="HC2")` and `lm() + clubSandwich::vcovCR(cluster=..., type="CR2")` at ~1e-10 (verified via new `tests/test_estimators_vcov_type.py::TestDiDAbsorbedFERParity` against `benchmarks/data/clubsandwich_cr2_golden.json` scenario `absorbed_fe_did`, with the R generator using the singleton-cluster CR2 trick for one-way HC2-BM Satterthwaite DOF). HC1/CR1 paths unchanged. (`MultiPeriodDiD(absorb=...)` and `TwoWayFixedEffects(vcov_type in {"hc2","hc2_bm"})` were both lifted later in this same release — see the top entries; both use the same algebra on different fit-path structures.) **Behavioral note (full `DiDResults` surface change under auto-route):** under the auto-route, the entire returned `DiDResults` reflects the full-dummy fit rather than the within-transformed fit. Specifically, `result.coefficients` and `result.vcov` include the FE-dummy entries (matching the `fixed_effects=` path), `result.residuals` and `result.fitted_values` are on the un-demeaned outcome scale, and `result.r_squared` is computed on the un-demeaned outcome (so it absorbs the FE variance and will typically be higher than the within-R²). `result.att` is invariant to this routing (FWL guarantee). Downstream consumers reading `result.att` are unaffected; consumers reading the broader result surface should expect the full-dummy values. **Survey-design scope:** the auto-route changes the FE handling (and removes the prior absorbed-FE rejection), but `survey_design=` continues to drive its own variance path (Taylor-series linearization or replicate-weight variance, per the existing survey contract) rather than the analytical HC2/HC2-BM sandwich. The auto-route is therefore methodologically meaningful for non-survey fits and for the FE-handling side of survey fits; analytical small-sample inference under `vcov_type in {"hc2","hc2_bm"}` is bypassed when a survey design is supplied.
- **`SpilloverDiD` Gardner GMM first-stage uncertainty correction across HC1 / Conley / cluster (Wave D).** Closes the documented Wave B/C "SEs biased downward by a few percent" caveat. **Documented synthesis** of Butts (2021) Section 3.1 (the IF construction for spillover-aware DiD) + Gardner (2022) Section 4 (the two-stage GMM sandwich) + Conley (1999) (the spatial kernel). No reference software combines all three — `did2s` (Butts & Gardner) implements the Gardner correction without rings or Conley; `conleyreg` and `acreg` implement Conley without the two-stage correction. Wave D is the synthesis. Applies unconditionally under `vcov_type ∈ {"hc1", "conley", "cluster"}` for both `event_study=False` AND `event_study=True`. **Formula** (Butts 2021 §3.1 + Gardner 2022 §4): `psi_i = gamma_hat' * X_{10,i} * eps_{10,i} - X_{2,i} * eps_{2,i}` where `gamma_hat = (X_10' X_10)^{-1} (X_1' X_2)` is the stage-1-projection-of-stage-2 cross-moment; meat = `Psi' K Psi` with `K` dispatched by `vcov_type` (identity for HC1, block-indicator for cluster, spatial kernel for Conley); vcov = `(X_2' X_2)^{-1} @ meat @ (X_2' X_2)^{-1}`. **Finite-sample multipliers:** `n/(n-p)` for HC1; `G/(G-1) * (n-1)/(n-p)` for cluster CR1; no multiplier for Conley (preserves `conleyreg` / Wave B convention). **Public surface:** `vcov_type="classical"` now raises `NotImplementedError` upfront (the Wave D synthesis has not been derived for the homoskedastic meat structure `sigma_hat^2 * (X_10' X_10)`); REGISTRY's "vcov_type restrictions" block updated accordingly. **Point estimates unchanged** (`tau_total`, `delta_j`, event-study `tau_k` / `delta_jk` are byte-identical to Wave B/C); SE values shift upward by 1-few percent depending on first-stage residual variance. **Implementation:** new module-level helper `_compute_gmm_corrected_meat` in `diff_diff/two_stage.py` (NOT a modification of the existing `_compute_gmm_variance` method — TwoStageDiD's path is unchanged); new module-level helper `_build_butts_fe_design_csr` in `diff_diff/spillover.py`; new module-level helper `_compute_conley_meat` in `diff_diff/conley.py` factored out of `_compute_conley_vcov` so the same kernel-application code path handles both standard sandwich (`X * residuals`) and Wave D IF outer product (`Psi`) cases. **No new public API kwarg** — the correction is unconditional. Wave D variance mode dispatch derives from the public contract: `vcov_type="conley"` → `"conley"`; `cluster=<col>` → `"cluster"` (CR1); otherwise `"hc1"`. **Wave B/C SE goldens re-pinned** at `tests/test_spillover.py::TestSpilloverDiDEventStudyBackwardCompat` (constants renamed `_WAVE_B_GOLDEN_*` → `_WAVE_D_GOLDEN_*`; pre-Wave-D references retained as commented baselines for the directional inflation invariant `_WAVE_B_UNCORRECTED_*`). **Tests:** new test classes `TestSpilloverDiDWaveDGmmCorrectedHc1Hand` (hand-derived `Psi` on a 4-unit × 3-period over-identified panel — matches at `atol=1e-12`), `TestSpilloverDiDWaveDGmmCorrectedEventStudy` (vcov shape on event-study path), `TestSpilloverDiDWaveDGmmCorrectedNanInferenceContract` (rank-deficient column propagation), `TestSpilloverDiDWaveDGmmCorrectedValidatorWiring` (Conley validator fires from the new helper), `TestSpilloverDiDWaveDGmmCorrectedFitIdempotence` (clone + repeat-fit bit-identity per `feedback_fit_does_not_mutate_config`), `TestSpilloverDiDWaveDPublicVarianceContract` (end-to-end public `cluster=<col>` CR1 routing, single-cluster rejection, classical NotImplementedError). Closes the Gardner-GMM follow-up row in `TODO.md`.
- **BaconDecomposition R parity goldens.** Closes the PR-B deferral row in `TODO.md`. JSON goldens at `benchmarks/data/r_bacondecomp_golden.json` generated from the committed `benchmarks/R/generate_bacon_golden.R` script (3 fixtures: `uniform_3groups_with_never_treated`, `two_groups_no_never_treated`, `always_treated_remapped`) against `bacondecomp 0.1.1` on R 4.5.2. `tests/test_methodology_bacon.py::TestBaconParityR` now active (4 tests, no skips): TWFE coefficient parity at `atol=1e-6` across all 3 fixtures; weights-sum parity at `atol=1e-6` across all 3 fixtures; per-component estimate + weight parity at `atol=1e-6` on the 2 non-remap fixtures **and on the 6 timing-vs-timing rows of `always_treated_remapped`** (carve-out narrowed to U-bucket rows only); plus a dedicated fold-back test (`test_always_treated_remapped_fold_back_matches_r`) that pins the **documented convention divergence** on `always_treated_remapped` (R keeps `first_treat=1` as a distinct timing cohort and emits `Later vs Always Treated` comparisons; Python's paper-footnote-11 convention remaps those units to `U` and folds them into a single `treated_vs_never` cell per treated cohort) by aggregating R's split rows per cohort and asserting they match Python's single fold at `atol=1e-6`. The aggregate is invariant per Theorem 1; the per-component breakdown differs structurally between conventions but the fold-back is now directly asserted. New `**Note (R parity convention divergence on always-treated)**` and `**Deviation (first-period boundary extension on always-treated remap)**` in `docs/methodology/REGISTRY.md`. **First-period boundary deviation:** the paper uses strict `t_i < 1` for the always-treated bucket; the library uses the inclusive `first_treat <= min(time)` rule and folds `first_treat == min(time)` cohorts into `U`. R does NOT apply this fold (it keeps such cohorts as their own bucket). When `min(time) > 1` the rules coincide. Explicitly labeled in REGISTRY's Deviations block and mirrored in `METHODOLOGY_REVIEW.md` and `bacon.py`. METHODOLOGY_REVIEW.md tracker row promoted `**Complete** (R parity goldens pending)` → `**Complete**`.
- **`generate_ddd_panel_data` — panel-structured DGP for Triple-Difference power analysis** (`diff_diff/prep_dgp.py`). New public function exported from `diff_diff` and `diff_diff.prep` for panel DDD simulations. Cross-sectional `generate_ddd_data` remains available unchanged. Produces a balanced panel of `n_units × n_periods` with two unit-level binary dimensions (`group`, `partition`) and a derived `post = 1[period >= treatment_period]` indicator; columns: `unit, period, outcome, group, partition, post, treated, true_effect` (+ `x1, x2` when `add_covariates=True`). DDD-CPT identification holds because the `group * partition` interaction enters as a unit-level (time-invariant) term, leaving the triple-interaction `treatment_effect * group * partition * post` as the sole source of differential group × partition trend. Compatible with `TripleDifference(cluster="unit").fit(..., time="post")` (the cluster kwarg is required because `TripleDifference` is the repeated-cross-section `panel=FALSE` estimator and unclustered SE on panel-generated rows understates variance under within-unit serial correlation; the point estimate `att` is invariant to clustering — see the new `TripleDifference` REGISTRY note on panel-shaped input). Users get panel-realistic unit fixed effects and within-unit serial correlation while the binary 2×2×2 estimator surface is unchanged. **Stratified allocation:** the partition split is drawn stratified-by-group at the requested `partition_frac` so every `(group, partition)` cell receives at least one unit; a targeted `ValueError` is raised at fit-time when the rounded cell counts (`n_units`, `group_frac`, `partition_frac`) would leave any cell empty. This guarantees the 2x2x2 DDD surface is populated for any valid input — independent marginal sampling (the cross-sectional `generate_ddd_data` convention) could collapse cells when marginals are small (e.g., `n_units=4, group_frac=partition_frac=0.25`). Validates `1 <= treatment_period < n_periods`, `group_frac` and `partition_frac` strictly in `(0, 1)`, and `n_units >= 4`. Deterministic recovery (`noise_sd=0`) matches `treatment_effect` to ~1e-15 (covered by `tests/test_prep.py::TestGenerateDddPanelData`, 16 tests including infeasible-config rejection and smallest-feasible-config round-trip through `TripleDifference.fit`). `power.simulate_power` is NOT yet auto-routed to the panel DGP for `TripleDifference` (the existing `_ddd_dgp_kwargs` registry entry still ignores `n_periods` and the existing `_check_ddd_dgp_compat` warning still fires on non-default kwargs) — that wiring is tracked as a follow-up in TODO.md.
- **BaconDecomposition: Goodman-Bacon (2021) methodology audit (PR-B).** Closes the BaconDecomposition row in `METHODOLOGY_REVIEW.md` (status flipped from **In Progress** → **Complete** — initially with an R-parity-goldens caveat that was closed by the parity-goldens bullet above in this same release). Builds on the PR #451 paper review at `docs/methodology/papers/goodman-bacon-2021-review.md`. **Audit outcomes:** (1) Rewrote `_recompute_exact_weights` in `bacon.py` to actually implement Theorem 1 (Eqs. 7-9 + 10e-g) — the prior "exact" implementation was missing the `(1-n_kU)` factor in the subsample variance, did not square the sample share, and added an extraneous `unit_share` factor not present in the paper; the post-hoc sum-to-1 normalization masked the relative-weight error but produced ~0.3% decomposition error vs TWFE on a 3-cohort + never-treated DGP. The rewrite computes the exact numerators of Eqs. 10e/f/g and lets the post-hoc normalization handle the `V̂^D` denominator (Theorem 1's identity guarantees `V̂^D = Σ numerators`). The TWFE-vs-weighted-sum identity now holds at `atol=1e-10` on both noisy and hand-calculable DGPs. (2) Added always-treated warn+remap per paper footnote 11: units whose `first_treat` is at or before the first observable period (`first_treat <= min(time)`, excluding the never-treated sentinels `0` and `np.inf`) are automatically remapped to the `U` (untreated) bucket via an internal column (`__bacon_first_treat_internal__`) with a `UserWarning`. Detection uses ordered-time logic on the **time axis**, so panels whose `time` column has negative or zero-crossing labels (event-time encodings) are handled correctly; the `0` sentinel restriction applies only to `first_treat`, not to `time`, and a real treatment cohort with `first_treat == 0` would still be folded into U today (re-label such cohorts to a non-sentinel value before fitting). The user's original `first_treat` column is preserved unchanged. The count is surfaced as a new `BaconDecompositionResults.n_always_treated_remapped` dataclass field, rendered in `summary()` output when nonzero. **`n_never_treated` reports TRUE never-treated only**, computed from the original user column before remap — remapped always-treated units appear separately as `n_always_treated_remapped`, no double-counting. (3) New methodology test file `tests/test_methodology_bacon.py` (34 tests across 6 classes post-release; the audit added ~24 tests and the R-parity-goldens bullet above expanded coverage: `TestBaconHandCalculation` hand-checks Eqs. 7-9 + 10b-d on a minimal balanced panel at `atol=1e-10`; `TestBaconParityR` (4 tests, all active post-release once the R parity goldens bullet above landed; skips cleanly with a regenerate-instructions pointer in partial-checkout scenarios where the JSON is unavailable); `TestBaconAlwaysTreatedRemap` regression-tests warn+remap mechanics including user-data-preservation; `TestBaconEdgeCases` exercises no-untreated, single-cohort, unbalanced panel, constant-ATT recovery; `TestBaconWeightModes` locks the new exact-is-default contract; `TestBaconSurveyDesignNarrowing` confirms survey_design composes with exact mode and warn+remap). (4) R `bacondecomp::bacon()` parity generator committed at `benchmarks/R/generate_bacon_golden.R` covering three DGP fixtures (3-groups-with-U, 2-groups-no-U, always-treated-remapped); the JSON goldens deferral at audit time was closed in this same release by the parity-goldens bullet above. (5) `docs/methodology/REGISTRY.md` `## BaconDecomposition` block replaced with the paper-review-sourced entry plus three new sub-notes: weight modes (exact vs approximate), always-treated remap, R parity status. **Explicit removal:** the prior REGISTRY block's "Weights may be negative for later-vs-earlier comparisons" claim was incorrect per Theorem 1 (decomposition weights are strictly positive and sum to 1; negative weights are an estimand-level phenomenon, not estimator-level) and is dropped from the new entry. Closes the BaconDecomposition follow-up tracked at `TODO.md` (the prior row added in PR #451 is replaced by a narrower R-parity-goldens deferral row).
- **`SpilloverDiD(event_study=True)` — per-event-time × ring decomposition (Butts 2021 Section 5 / Table 2).** Replaces the Wave B `NotImplementedError` gate with the full per-event-time × ring decomposition. Emits per-event-time direct effects `tau_k` and per-(ring, event-time) spillover effects `delta_jk` as `att_dynamic: pd.DataFrame` (indexed by event-time `k`) and a MultiIndex `spillover_effects: pd.DataFrame` (levels `(ring_label, event_time)`). A TwoStageDiD-compatible `event_study_effects: Dict[int, Dict[str, Any]]` alias (matching `two_stage.py:1355-1389` schema with `conf_int = (low, high)` tuple) is also emitted for consumption by `plot_event_study` (`SpilloverDiDResults` is wired into `_extract_plot_data` and prefers the new `reference_period` attribute over the legacy `n_obs==0` heuristic). `DiagnosticReport` integration is NOT wired in this PR — registering `SpilloverDiDResults` in `DiagnosticReport`'s applicability/method tables is queued as a follow-up. **Methodology spec:** the implementation operationalizes Butts Section 5's single `K_it` symbol as TWO event-time clocks — `K_direct = t - effective_first_treat(i)` for ever-treated unit rows, and `K_spill = t - earliest-in-range-cohort-onset(i)` for spillover rows (running min across activated cohorts; NaN for pre-trigger and far-away rows). `K_spill >= 0` structurally; negative-k spillover cells emit rectangularly with `coef = NaN, n_obs = 0`. **Reference period:** `ref_period = -1 - anticipation` (mirrors `TwoStageDiD` at `two_stage.py:486`); when `horizon_max` is set, `ref_period` must fall inside `[-horizon_max, +horizon_max]` or fit raises `ValueError` — silent floor-shift to `-horizon_max` would change identification (rejected per `feedback_no_silent_failures`). The reference row in `att_dynamic` / `event_study_effects` uses `coef = 0.0, se = 0.0, n_obs = 0, conf_int = (0.0, 0.0)` for TwoStageDiD parity. **`horizon_max` semantics (divergence from TwoStageDiD):** SpilloverDiD bins event-times outside `[-horizon_max, +horizon_max]` into endpoint pools (no observations dropped); TwoStageDiD filters those rows. The divergence is intentional and cross-documented. With `horizon_max=None`, the helper auto-detects the bin set from observed K values. **Scalar `att` aggregation:** when `event_study=True`, the top-level `att` is the **sample-share-weighted average** of post-treatment `tau_k` (`att = sum_{k >= 0} w_k * tau_k` with `w_k = n_treated_at_k / total`). SE comes from linear-combination inference `Var(att) = w' V_subset w` on the post-treatment block of the stage-2 vcov — no separate fit. **Reduce-to-aggregate equivalence:** under a constant-tau DGP with `horizon_max=None`, the lincom-weighted scalar `att` reproduces Wave B's aggregate `tau_total` bit-identically in the deterministic limit (verified by `TestSpilloverDiDEventStudyReduceToAggregate`). Note: `horizon_max=0` is **not supported** under `event_study=True` (rejected at validation): the single bin `k=0` leaves no event-time pair to anchor the reference period against. Use `event_study=False` for a single aggregate direct effect (Wave B static spec); event-study mode requires `horizon_max>=1` or `horizon_max=None`. **Post-finite_mask sample contract:** `att_dynamic["n_obs"]`, `event_study_effects[k]["n_obs"]`, AND the scalar `att` share weights all reflect the POST-`finite_mask` stage-2 estimation sample (not the pre-mask design). On warn-and-drop fits (baseline-treated units without Omega_0 rows excluded), the reported `n_obs` per cell counts only rows that actually entered `solve_ols`. **Fail-closed scalar `att`:** if any post-treatment direct-effect coefficient is NaN (rank-deficient drop by `solve_ols`), the scalar `att` is set to NaN with an explicit warning rather than silently zeroing the dropped column's contribution via `np.nansum` on a fixed weight vector — inspect `att_dynamic` for the per-event-time coefficients and re-aggregate manually if appropriate. **Backward compatibility:** `event_study=False` leaves all Wave C fields (`att_dynamic`, `event_study_effects`, `horizon_max`, `reference_period`) as `None`. The aggregate stage-2 design construction, fit, and extraction logic on this path are byte-identical to Wave B; `TestSpilloverDiDEventStudyBackwardCompat` pins att / se / per-ring goldens captured on the unchanged aggregate path so any future drift fails the regression. **Variance:** at original Wave C ship time per-event-time SEs used `solve_ols`'s standard variance (HC1 / Conley / cluster paths) WITHOUT the Gardner GMM first-stage uncertainty correction. **Superseded by the Wave D Gardner GMM first-stage correction in this same release** (see the Wave D bullet above): per-event-time SEs now apply the IF outer-product correction unconditionally and shift upward by 1-few percent relative to the original Wave C ship-time values. **Tests:** `tests/test_spillover.py` adds 30 new test methods across event-study API, two-clock K helper, horizon binning, design builder, reference period, reduce-to-aggregate, identification MC (50 seeds, per-event-time tau_k recovery within 0.025), placebo pre-trends (Type I rate ≤ 0.30 over 50 seeds at alpha=0.10), singularity (rectangular schema), Conley integration (vcov shape + non-negative diagonal), summary/to_dict/pickle round-trip, event_study_effects schema parity with TwoStageDiD, lincom-att hand-computed, validation (`horizon_max < 0`, `ref_period < -horizon_max`), and fit idempotence. DGP factory `generate_butts_staggered_dgp` extended with `tau_per_event_time` and `delta_per_ring_per_event_time` callable kwargs (backward-compatible — both default to `None`, producing the Wave B scalar DGP bit-identically; verified by `tests/test_dgp_utils.py` with pinned SHA-256 baselines).
- **`SpilloverDiD` — ring-indicator spillover-aware DiD (Butts 2021).** New standalone estimator at `diff_diff/spillover.py` implementing two-stage Gardner methodology with ring-indicator covariates that identify direct effect on treated (`tau_total`) alongside per-ring spillover effects on near-control units (`delta_j`). Documented synthesis of ingredients (no single published software covers the exact recipe — `did2s` implements Gardner two-stage without rings; the Butts ring estimator has no R/Stata package): Butts (2021) Section 5 / Table 2 identification, Gardner (2022) two-stage residualize-then-fit, and the Conley spatial-HAC vcov shipped in 3.3.3. Handles both panel non-staggered (Equations 5/6/8) and Section 5 staggered timing in one estimator — non-staggered is the special case where all treated units share an onset time. **API:** `SpilloverDiD(rings=[0, 50, 100, 200], conley_coords=("lat","lon"), ...).fit(data, outcome="y", unit="unit", time="t", treatment="D")` (binary D auto-converted to `first_treat`) or `.fit(..., first_treat="first_treat")` (Gardner convention). Result: `SpilloverDiDResults(DiDResults)` with `.att` = `tau_total`, `.spillover_effects` (per-ring `pd.DataFrame` with `coef`/`se`/`t_stat`/`p_value`/`ci_low`/`ci_high`), `.ring_breakpoints`, `.d_bar`, `.n_units_ever_in_ring`, `.n_far_away_obs`, `.is_staggered`. `.coefficients` exposes all `(1+K)` stage-2 entries (`"treatment"` + `"_spillover_<ring_label>"`) plus an `"ATT"` alias keyed to vcov columns. **Methodology spec (committed):** stage-2 regressor is the time-varying `(1 - D_it) * Ring_{it,j}` form (paper page 12's `S_it = S_i * 1{t >= t_treat}` notation; Section 5 Table 2's `S^k_{it}` / `Ring^k_{it,j}`). Reading the literal unit-static `(1 - D_it) * S_i` from Equation 5 is algebraically rank-deficient under TWFE (`(1-D_it) * S_i = S_i - D_it`, with `S_i` absorbed by `mu_i`, leaving `-D_it`); only the time-varying form supports the paper's identification (Proposition 2.3). Stage-1 subsample uses Butts' STRICTER `Omega_0 = {D_it = 0 AND S_it = 0}` (untreated AND unexposed), not TwoStageDiD's `{D_it = 0}` alone — this prevents spillover-contaminated near-controls in pre/post periods from biasing the time FE. **Gardner identity (non-staggered):** a 20-seed deterministic regression test pins `SpilloverDiD.att` against a direct single-stage TWFE ring regression on the full sample (`y ~ mu_i + lambda_t + tau * D_it + sum_j delta_j * (1 - D_it) * Ring_{it,j}`) at `atol=1e-10` — empirically bit-identical, so the reported non-staggered `tau_total` IS the Butts Eqs. 4-6 estimator. **Identification-check policy (period strict, unit warn-and-drop, plus connectivity):** every period must have at least one Omega_0 row (hard `ValueError` — dropping a period removes all units' cross-time identification). Units lacking Omega_0 rows (e.g. baseline-treated units with `D_it = 1` at every observed `t`) are warned-and-dropped: their unit FE is NaN, residualization writes NaN on their rows, and the downstream finite-mask path excludes them from stage 2 — mirrors `TwoStageDiD`'s always-treated convention. Additionally, the supported-units bipartite graph (units linked by shared Omega_0 periods) must form a single connected component; `K > 1` components raise `ValueError` because the FE solver would return only component-specific constants and residualization would silently mix them across components (defense-in-depth — under absorbing treatment the disconnected case may be unreachable through the upstream validators, but the check future-proofs Wave B follow-ups). **Public API restrictions (Wave B MVP):** `covariates=` raises `NotImplementedError` because Gardner-style two-stage requires covariate effects estimated on the untreated-and-unexposed subsample at stage 1 (appending raw covariates only at stage 2 silently biases `tau_total` / `delta_j` on panels with time-varying covariates); non-absorbing / reversible treatment patterns (e.g. `[0, 1, 0]`) raise `ValueError` rather than being silently coerced into "treated from first 1 onward"; non-constant `first_treat` values across rows of the same unit raise `ValueError`; `conley_coords` is required on every fit path (not just `vcov_type="conley"`) because ring construction always uses it. **Far-away control identification:** uses CURRENT-period untreated status (`D_it = 0`) rather than never-treated-only, so all-eventually-treated staggered designs (no never-treated units) can identify the counterfactual via not-yet-treated far-away rows. **Variance (Wave B MVP ship-time):** stage-2 OLS variance via `solve_ols` (HC1 / Conley / cluster paths all flow through) WITHOUT the Gardner GMM first-stage uncertainty correction. **Superseded by the Wave D Gardner GMM first-stage correction in this same release** (see the Wave D bullet above): the GMM correction now applies unconditionally across HC1 / Conley / CR1 (via `cluster=<col>`), shifting SE values upward by 1-few percent relative to the original Wave B ship-time values. **Deferred features (planned follow-ups, as of Wave B ship-time):** `event_study=True` per-event-time × ring coefficients (Butts Table 2), `survey_design=` integration, `ring_method="count"` (count-of-treated-in-ring), data-driven `d_bar` selection (Butts 2021b / Butts 2023 JUE Insight), Gardner GMM first-stage correction at stage 2, sparse staggered ring-distance path. **Shipped in same release:** `event_study=True` (Wave C bullet above) + Gardner GMM first-stage correction (Wave D bullet above); remaining items still queued. **Tests:** `tests/test_spillover.py` (157 tests across ring-construction primitives, validators, fit integration, raw-data invariant, identification MC — non-staggered DGP at 50 seeds + 200-seed `@pytest.mark.slow` variant recovers both `tau_total` and `delta_1`; staggered DGP at 30 seeds anchors both `tau_total` and `delta_1` — Conley plumbing (verifies `solve_ols` is called with `vcov_type="conley"` + Conley kwargs, no silent HC1 fallback), Gardner identity bit-identity, coefficients-vs-vcov alignment, warn-and-drop, rank_deficient_action validation, Omega_0 bipartite-graph connectivity, anticipation behavior on both fit paths). DGP factories `tests/_dgp_utils.py::generate_butts_nonstaggered_dgp` / `generate_butts_staggered_dgp` satisfy Butts Assumptions 1/3/5/7 by construction.
- **`ChaisemartinDHaultfoeuille.predict_het` × `placebo`: R-parity on both global and per-path surfaces.** R-verified — `did_multiplegt_dyn(predict_het, placebo)` emits heterogeneity OLS results on backward (placebo) horizons via R's `DIDmultiplegtDYN:::did_multiplegt_main` placebo block (`effect = matrix(-i, ...)` rbind site); the same block runs per-by_level under `did_multiplegt_dyn(by_path, predict_het, placebo)`, so both global `res$results$predict_het` and per-by_level `res$by_level_i$results$predict_het` slots emit backward rows. R's predict_het syntax with `placebo > 0` requires the `c(-1)` sentinel in the horizon vector to trigger "compute heterogeneity for ALL forward (1..effects) AND ALL placebo (1..placebo) positions" — passing positive-only horizons errors with "specified numbers in predict_het that exceed the number of placebos". Python mirrors via `_compute_heterogeneity_test(..., placebo=L_max)` (set automatically from `self.placebo` at both global and per-path call sites in `fit()`) — the function iterates forward (1..L_max) and backward (-1..-L_max) horizons in a single loop with an explicit `out_idx < 0` eligibility guard for backward horizons whose `F_g` is too small (would otherwise silently misread `N_mat` via numpy negative indexing). `results.heterogeneity_effects` uses negative-int keys for backward horizons; `path_heterogeneity_effects` does the same per path. Placebo rows in `to_dataframe(level="by_path")` have non-NaN `het_*` columns when `placebo=True` and `heterogeneity=` are both set. **Survey gate (warn + skip):** `survey_design + placebo + heterogeneity` emits a `UserWarning` at fit-time and falls back to forward-horizon-only heterogeneity on both surfaces — the Binder TSL cell-period allocator's REGISTRY justification is tied to **post-period** attribution; backward-horizon attribution puts ψ_g mass on a pre-period cell, a separate library-extension claim that needs its own derivation. Forward-horizon `predict_het + survey_design` continues to work unchanged on both global and per-path surfaces. The function-level `_compute_heterogeneity_test` keeps a per-iteration `NotImplementedError` backstop for direct callers that bypass fit(). Pre-period allocator derivation deferred to a follow-up methodology PR (tracked in TODO.md). R parity confirmed at `tests/test_chaisemartin_dhaultfoeuille_parity.py::TestDCDHDynRParityHeterogeneityWithPlacebo` (scenario 23, `multi_path_reversible_predict_het_with_placebo_global`, `placebo=2, effects=3, no by_path`) and `::TestDCDHDynRParityByPathHeterogeneityWithPlacebo` (scenario 22, same DGP plus `by_path=3`); pinned at `BETA_RTOL=1e-6` / `SE_RTOL=1e-5` for `beta` / `se` / `t_stat` / `n_obs` and `INFERENCE_RTOL=1e-4` for `p_value` / `conf_int` across 3 paths × (3 forward + 2 placebo) = 15 horizons + 1 global × 5 horizons. Cross-surface invariants regression-tested at `tests/test_chaisemartin_dhaultfoeuille.py::TestByPathPredictHetPlacebo` (placebo het column population, survey-gate warn+skip behavior, forward+survey anti-regression, `out_idx<0` eligibility guard, single-path telescope `path_heterogeneity_effects[(only_path,)] == heterogeneity_effects` bit-exactly, summary rendering, direct-call `NotImplementedError` backstop). Closes TODO #422.

### Changed
- **PreTrendsPower: default `pretest_form` flipped from implicit Wald to explicit `'nis'` (PR-B methodology audit, Roth 2022).** The new default uses the paper-analyzed NIS box probability — the form Roth (2022) actually tabulates in his Section I.C empirical exercise and the form the R `pretrends` package implements. `pretest_form='wald'` preserves the **acceptance-region form** (noncentral-χ² on the quadratic form `δ' Σ_22^{-1} δ`) byte-identically — the methods are renamed `_compute_power_wald` + `_compute_mdv_wald` with unchanged bodies, dispatched on `self.pretest_form`. **Caveat on bit-identity for fitted results**: the linear-weight contract changed independently in PR-B Step 4 (see the next bullet), so a Wald fit on an irregular pre-period grid produces γ-unit MDV via the new `relative_times`-threaded path, NOT the pre-PR-B count-based L2-normalized MDV. Pre-PR-B Wald numerics are bit-identical to post-PR-B Wald output only on the legacy `relative_times=None` callable path (callers that bypass `fit()` and call `_get_violation_weights(n_pre)` directly) and on the regular-grid case where `|t| ∝ [n_pre-1, ..., 0]`. All existing `tests/test_pretrends.py` numerical assertions (101 helper/class references; only 3 tests depended on the exact Wald size-at-null property and were pinned to `pretest_form='wald'`) continue to produce identical numerical output. The `docs/tutorials/07_pretrends_power.ipynb` walkthrough re-render to reflect the default flip is tracked as a follow-up (the existing tutorial does not exercise the irregular-grid regime).
- **PreTrendsPower: `_get_violation_weights('linear')` now honors actual pre-period relative-time labels and skips L2 normalization → reported MDV is in Roth's γ units (PR-B Step 4).** Pre-PR-B, the linear-violation direction was constructed as `[n_pre-1, ..., 1, 0] / ||·||_2` from `n_pre` count alone — irregular pre-period grids like `{-5, -3, -1}` were treated as if the periods were `{-3, -2, -1}`, and the L2-normalization meant the reported MDV equaled `γ · ||t||_2`, not γ. PR-B threads the actual `relative_times` array from `_extract_pre_period_params` into `_get_violation_weights` and, for `violation_type='linear'` with `relative_times not None`, uses `weights = |t|` directly with NO L2 normalization. Then `δ_pre = M · |t|` reflects Roth's `δ_t = γ · t` convention and the reported MDV equals γ exactly. Verified: regular grid `[-3, -2, -1]` → weights `[3, 2, 1]`; irregular grid `[-5, -3, -1]` → weights `[5, 3, 1]`; backwards-compat callers that bypass `fit()` and pass only `n_pre` retain the legacy normalized `[n_pre-1, ..., 0] / ||·||_2` behavior. The `_extract_pre_period_params` return type widened from a 4-tuple to a 6-tuple `(effects, ses, vcov, n_pre, relative_times, covariance_source)`; the `relative_times` element is populated by all three adapter branches from their respective sorted pre-period lists (MPD via `pandas.Period` / `Timestamp` / `np.datetime64` arithmetic when applicable, falling back to a warn + count-based normalized direction for genuinely non-numeric labels), and the new `covariance_source` element records the actual extraction path for downstream report-layer tier classification.
- **BaconDecomposition: default `weights` flipped from `"approximate"` to `"exact"` (PR-B methodology audit).** The new default uses Goodman-Bacon (2021) Theorem 1's exact Eqs. 7-9 + 10e-g weights, matching R `bacondecomp::bacon()` at `atol=1e-6` (validated via `tests/test_methodology_bacon.py::TestBaconParityR`; see the new Added entry above for the convention divergence on always-treated cohorts). Hand-calculation + TWFE-vs-weighted-sum identity also hold at `atol=1e-10`. The `weights="approximate"` path remains available as an opt-in fast diagnostic for speed-sensitive loops; its numerical output may differ from R. Three entry points were flipped: `BaconDecomposition(weights="exact")` (`bacon.py:397`), `bacon_decompose(weights="exact")` (`bacon.py:1064`), `TwoWayFixedEffects.decompose(weights="exact")` (`twfe.py:684`). **Behavior change for users not passing explicit `weights=`**: the decomposition weights are now paper-faithful by default. Users who depended on the previous `"approximate"` numerics for diagnostic plots or comparison-type weight shares can preserve the old behavior by passing `weights="approximate"` explicitly. **Survey-design behavior change**: `weights="exact"` (now the default) routes through `_validate_unit_constant_survey`, which rejects survey designs whose weights / strata / PSU / FPC columns vary within a unit across periods (the exact-mode path collapses to per-unit aggregation via `groupby().first()`). The previous `weights="approximate"` default tolerated time-varying within-unit survey weights via observation-level weighted means. Users whose survey-weighted Bacon calls used time-varying within-unit weights must now either (a) collapse their weights to be unit-constant or (b) pass explicit `weights="approximate"` to retain the legacy obs-level path. The production diagnostic surface (`diff_diff/diagnostic_report.py:1740`) was updated to pass explicit `weights="exact"`. Existing test assertions in `tests/test_bacon.py` continue to pass with the new default; the `test_weighted_sum_equals_twfe` tolerance was tightened from `< 0.1` to `< 1e-10` to lock the Theorem 1 algebraic-identity contract.

- **`ChaisemartinDHaultfoeuille.predict_het` inference: t-distribution df threading (closes TODO pilot-412).** `_compute_heterogeneity_test` now passes `df = n_obs - rank(design)` to `safe_inference` on the non-survey OLS path, matching R `did_multiplegt_dyn(predict_het=...)`'s t-distribution inference (`DIDmultiplegtDYN:::did_multiplegt_main` `t_stat <- qt(0.975, df.residual(model))` site). Pre-PR Python used `df=None` (normal Z critical), producing 0.1-2% rtol gaps on `p_value` and `conf_int` vs R. Parity tolerance tightened on the existing forward-horizon scenarios (`multi_path_reversible_predict_het`, `multi_path_reversible_by_path_predict_het`) from "unpinned" to `INFERENCE_RTOL=1e-4` on `p_value` and `conf_int`; `beta` / `se` / `t_stat` continue at `BETA_RTOL=1e-6` / `SE_RTOL=1e-5`. **Post-drop rank (post-2026-05-16 wrap-up):** the df denominator uses the post-drop numerical rank via `_detect_rank_deficiency`, which `solve_ols` already calls internally. For full-rank designs `rank == n_params` and behavior is bit-identical to the pre-PR `n_obs - n_params` path; for near-rank-deficient designs that `solve_ols` retains rather than NaN-out (e.g., cohort-collinearity at high horizons), the post-drop rank is strictly lower and the post-PR `df` is larger, matching R's `lm()` convention. The Z-vs-t REGISTRY deviation note is replaced with an "R parity (post-2026-05-15 df threading)" positive-claim note.

- **`ChaisemartinDHaultfoeuille.by_path` negative-baseline path regression coverage.** New `tests/test_chaisemartin_dhaultfoeuille.py::TestByPathNonBinary::test_negative_baseline_path_supported` exercises switchers with `D_{g,1} = -1` and asserts that `path_effects` correctly contains negative-baseline tuple keys (e.g., `(-1, 0, 0, 0)`, `(-1, 1, 1, 1)`). This closes the test-coverage gap from PR #419: the existing `test_negative_integer_D_supported` only covered paths with negative values in non-baseline positions (e.g., `(0, -1, -1, -1)`), which does not trigger R's documented `substr(path, 1, 1)` baseline-extraction bug. Python's tuple-key matching is correct under any baseline value; this test pins the contract. No R-parity fixture is added because R is the buggy side on this regime — the deviation is documented in the REGISTRY non-binary treatment Note.

### Fixed
- **PreTrendsPower: unit-consistent level-scale ratio for tier classification (PR-B R12 follow-up).** PR-B Step 4 made the linear MDV report Roth's γ units (a slope on relative time), but downstream tier-classification heuristics still divided the raw γ by level-scale quantities — `DiagnosticReport.pretrends_power` computed `mdv_share_of_att = mdv / abs(att)`, `is_informative` checked `mdv < 2 * max(pre_period_ses)`, and `sensitivity_to_honest_did` reported `mdv_in_ses = mdv / max_pre_se`. On irregular pre-period grids this silently mixed slope and level scales and could mis-tier the same fit as `well_powered` / `moderately_powered` / `underpowered`. Fix: new `PreTrendsPowerResults.max_abs_pre_violation` property exposes the level-scale scalar `mdv * max(|violation_weights|)` — the largest level-scale pre-period deviation under the MDV. `is_informative`, `sensitivity_to_honest_did`, `DiagnosticReport._check_pretrends_power`, and `_format_precomputed_pretrends_power` all switched to consume `max_abs_pre_violation` instead of raw `mdv` for level-scale comparisons. `mdv_share_of_att` is now defined as `max_abs_pre_violation / abs(att)`; the schema also surfaces the new `max_abs_pre_violation` field for inspection. Legacy serialized results without `violation_weights` fall back to raw `mdv` (preserves pre-PR-B count-based L2-normalized behavior where `mdv` was already roughly level-scale). On the live `cs_fit` fixture the ratio moves from `0.053` (slope/level mismatch) to `0.211` (level/level) — still `well_powered`, but now interpretable. New regressions: `test_max_abs_pre_violation_uses_weight_scale_on_irregular_grid` (γ * 5 on `[-5, -3, -1]`), `test_is_informative_uses_level_scale_not_raw_gamma` (level-scale check beats raw-γ check on a constructed mismatch), plus the updated BR `test_full_vcov_path_no_downgrade_on_real_cs_fit` which now pins `0.35 < max_abs_pre_violation < 0.40`.
- **PreTrendsPower: `PreTrendsPowerResults.power_at(M)` for `violation_type='custom'` (PR-B Step 5).** PR-A R18 added a `NotImplementedError` guard to prevent silent equal-weights output when `power_at()` couldn't reconstruct the fitted custom weights. PR-B Step 5 persists the normalized `violation_weights` on `PreTrendsPowerResults` at fit time, so `power_at(M)` now works correctly for all four violation types (linear / constant / last_period / custom) on fresh fits. The PR-A guard is retained only for legacy serialized results lacking the new `violation_weights` field (refit with current library version to lift). Verified by the new `test_power_at_works_for_custom_violation_type` regression test and the companion `test_power_at_raises_on_legacy_custom_result_without_weights` (simulates a legacy serialized result by clearing `violation_weights` to None).
- **`DiagnosticReport` / `BusinessReport` covariance-source provenance propagation (PR-B Step 3, R3 follow-up).** Before PR-B, `DiagnosticReport._infer_cov_source` flagged CS / SA fits with populated `event_study_vcov` as `"diag_fallback_available_full_vcov_unused"`, and `_apply_diag_fallback_downgrade` then conservatively downgraded the `well_powered` tier to `moderately_powered`. PR-B Step 3 routes those fits through the full `Σ_22` sub-block at the estimator layer — but the report layer kept the old type-based inference, so correctly-computed full-VCV power results were silently being downgraded. Fix: `PreTrendsPowerResults` gains a new `covariance_source` field that `pretrends.py:_extract_pre_period_params` populates with `"full_pre_period_vcov"` or `"diag_fallback"` based on the actual extraction path taken; `DiagnosticReport._check_pretrends_power` and `_format_precomputed_pretrends_power` prefer that persisted label and fall back to type-based inference only for legacy serialized results that lack the field. Two paths now coexist through the report layer: **new fits** (post-PR-B, `covariance_source` is persisted) consume the persisted label directly — non-bootstrap CS / SA report `"full_pre_period_vcov"` and are NOT downgraded; **legacy serialized results** (pre-PR-B, no `covariance_source` field on the object) fall through to `_infer_cov_source`, which STILL emits the conservative `"diag_fallback_available_full_vcov_unused"` sentinel for CS / SA + populated `event_study_vcov` because without the persisted label we cannot distinguish a pre-PR-B fit (which used `diag(ses^2)`) from a post-PR-B fit, and the PR-A conservative downgrade still applies to preserve backwards-compat. For `MultiPeriodDiDResults` without `interaction_indices`, the legacy fallback reports `"diag_fallback"` (a genuine fallback, no downgrade applies). Effect: non-bootstrap CS / SA pre-trends power blocks on fresh fits now keep their well_powered tier through the report layer (instead of being downgraded by the conservative sentinel); legacy serialized results are unchanged. Verified by `test_precomputed_pretrends_power_persisted_full_vcov_no_downgrade` (new fits), `test_precomputed_pretrends_power_legacy_missing_field_still_downgraded` (legacy fallback contract), `test_precomputed_pretrends_power_consumes_persisted_cov_source` (persisted label takes precedence over legacy inference), and `test_precomputed_pretrends_power_legacy_mpd_without_interaction_indices_reports_diag`.

## [3.3.3] - 2026-05-15

### Added
- **Tutorial 22: Survey-Weighted HAD** (`docs/tutorials/22_had_survey_design.ipynb`) — end-to-end walkthrough of `HeterogeneousAdoptionDiD` + `did_had_pretest_workflow` on a BRFSS-shape stratified household-survey panel (5 strata × 6 PSUs/stratum × 2 states/PSU = 60 states; post-stratification raking weights with CV ~ 0.30; FPC = 30 PSUs/stratum; PSU × period interaction shocks injected so cluster correlation survives DiD first-differencing). Demonstrates the `SurveyDesign(strata=...)` path through the Stute pretest family that the previous `[Unreleased]` entry unblocked. Eight numbered sections: motivation; panel + in-notebook helper for attaching survey columns to a HAD panel; naive vs survey-aware headline fit with a side-by-side ATT / SE / CI table (~10% SE inflation, sign-only direction asserted); a dedicated section explaining why the SE inflation is modest for HAD specifically (WAS-d_lower IF concentration at the boundary vs full-panel regression coefficients); event-study fit with sup-t cband under the survey design (per-horizon table + matplotlib gated plot); pretest workflow on both `aggregate="overall"` and `aggregate="event_study"` paths walking the Phase 4.5 C0 QUG-deferred verdict suffix and the now-supported stratified-clustered Stute multiplier bootstrap; "Communicating to Leadership" two-paragraph template (executive + methodologist); Extensions + Summary Checklist surfacing the still-deferred `lonely_psu='adjust'` + singleton-strata, replicate-weight designs, and the permanent QUG-under-survey C0 deferral. Companion drift-test file `tests/test_t22_had_survey_design_drift.py` (32 tests across 7 groups: panel + survey composition with deterministic exact pins; naive-vs-survey headline with sign-only SE-inflation anchor; event-study cband-vs-pointwise ordering and post/pre coverage; pretest overall path with `_QUG_DEFERRED_SUFFIX` lock and Yatchew `sigma2_*` deterministic pins; pretest event-study path with the SAME `_QUG_DEFERRED_SUFFIX` lock plus a SEPARATE substring lock on `report.summary()` for the L736 QUG-skip note; workflow-surface separation locking that overall has Stute+Yatchew while event-study has joint pretrends/joint linearity with `yatchew=None` and `stute=None`; and weighted point-estimation contract anchoring `survey.att != naive.att` plus the algebraic identity `att = (dy_mean_w - tau_bc) / den_w` from `_fit_continuous`). Bootstrap p-value pins use anchored windows of total width 0.30 (± 0.15 around seeded centers) per `feedback_strata_bootstrap_path_divergence` (stratified Mammen multiplier paths reduce effective dofs vs non-strata; PR #432 commit `aef07020` already had to relax bit-equality bands on this code path). T20 and T21 "Extensions" bullets updated with forward-pointers to T22; `docs/practitioner_decision_tree.rst` HAD universal-rollout and survey sections each gain a `.. tip::` cross-link to T22 (adjacent to T20 / T17, NOT displacing); `docs/api/had.rst` gains a "Survey-aware fit" cross-reference; `docs/survey-roadmap.md` gains a "Phase 4.5 C ✅ Shipped" entry; bundled `diff_diff/guides/llms.txt` and `llms-practitioner.txt` carry T22 inventory entries (the `llms-full.txt` reference guide is left as a follow-up to keep T22 PR scope tight); `docs/doc-deps.yaml` wires T22 as a dependent of both `had.py` and `had_pretests.py`. Closes the Phase 5 (wave 2 second slice) tutorial gap; the realistic survey-weighted HAD workflow on BRFSS / CPS / NHANES / ACS-shaped designs is now end-to-end documented for practitioners.
- **HAD pretest workflow: stratified survey-design support (Phase 4.5 C continuation).** Lifts the `NotImplementedError` gate on `SurveyDesign(strata=...)` in `stute_test` (`had_pretests.py:1927-1940` pre-PR) and `stute_joint_pretest` (`:3259-3271` pre-PR), and by inheritance in `joint_pretrends_test`, `joint_homogeneity_test`, and `did_had_pretest_workflow` (the wrappers delegate to the joint Stute helper). Implements a documented synthesis of clustered-wild-bootstrap ingredients (Cameron-Gelbach-Miller 2008 cluster-level multipliers; Davidson-Flachaire 2008 wild-bootstrap centering; Djogbenou-MacKinnon-Nielsen 2019 cluster-wild consistency for nonlinear functionals; Kreiss-Lahiri 2012 within-block centering analogy; Wu 1986 / Liu 1988 Bessel small-sample correction) — no single paper covers the exact composition for the stratified Stute CvM functional. The recipe: within-stratum demean + `sqrt(n_h/(n_h-1))` Bessel rescale applied to the PSU multipliers `psu_mults` BEFORE the per-obs broadcast `eta_obs = psu_mults[b, psu_col_idx]` in the wild-residual loop. Bootstrap CvM variance matches the analytical Binder-TSL stratified target `V_S = sum_h (1 - f_h) (n_h / (n_h - 1)) sum_j (psi_hj - psi_h_bar)²` exactly (the `(1 - f_h)` FPC factor was already baked in by `generate_survey_multiplier_weights_batch`; this PR bakes the remaining `(n_h / (n_h - 1))` factor and enforces within-stratum-mean-zero centering). New shared helper `bootstrap_utils.apply_stratum_centering(psu_mults, resolved_survey, psu_ids, psu_axis=...)` is called from both the new Stute path (psu_axis=1 on the multiplier matrix) AND the existing HAD sup-t event-study cband bootstrap (psu_axis=0 on the PSU-aggregated influence tensor; refactored bit-exactly from the inline block previously at `had.py:2172-2204`). Locks the algebraic identity architecturally instead of leaving parallel code blocks to drift. MC oracle consistency validated under a 4-stratum × 6-PSU/stratum stratified null DGP with weights+strata+PSU (200 seeded draws, empirical Type I at α=0.05 in `[0, 0.10]` — 3σ band; the FPC bake-in is covered separately by the helper-unit test `test_fpc_baked_in_helper_is_fpc_agnostic`); MC power validated under a known-alternative stratified DGP (rejection > 0.50). HAD sup-t event-study cband bit-parity preserved (`atol=1e-14, rtol=1e-14` on the refactored helper output + 29 existing cband tests passing post-refactor; that helper-level bit-parity test locks the axis-0 algebra). A separate wired-in regression at `tests/test_had_pretests.py::TestStuteStratifiedSurveyBootstrap::test_stute_call_sites_invoke_apply_stratum_centering` monkey-patches the helper and asserts both Stute call sites (`stute_test` at `had_pretests.py:1985` and `stute_joint_pretest` at `:3312`) invoke it with `psu_axis=1` — that test fails if either call site is disconnected (the axis-0 helper-parity test alone does not catch that case). See `docs/methodology/REGISTRY.md` § HeterogeneousAdoptionDiD — "Note (Stute stratified survey-bootstrap calibration)" for the full derivation. Remaining deferrals: `lonely_psu='adjust'` + singleton-strata (same pseudo-stratum centering gap as the HAD sup-t deviation at REGISTRY:2382) and replicate-weight designs (BRR/Fay/JK1/JKn/SDR — separate Rao-Wu / JKn bootstrap composition). Unblocks the realistic survey-weighted HAD workflow on BRFSS/CPS/NHANES/ACS-shaped designs.
- **Conley (1999) Wave A mechanical extensions** on top of the Phase 1+2 sandwich (`diff_diff/conley.py`, `diff_diff/linalg.py`, `diff_diff/estimators.py`, `diff_diff/twfe.py`). **(1) DiD support (#118):** `DifferenceInDifferences(vcov_type="conley").fit(..., unit="<col>")` is now supported. `unit` is a fit-time kwarg (NOT on `__init__`; unused unless Conley is set; not part of `get_params()` / `set_params()`) mirroring `MultiPeriodDiD.fit(unit=...)` / `TwoWayFixedEffects.fit(unit=...)`. DiD inherits the same panel block-decomposed sandwich as MPD/TWFE; on a 2-period panel it matches `MultiPeriodDiD(...).fit(..., post_periods=[1], reference_period=0)` bit-exactly. Missing `unit=`/`conley_lag_cutoff`/`conley_coords`/`conley_cutoff_km` raise `ValueError`; `survey_design=` + Conley raises `NotImplementedError` (Bertanha-Imbens 2014 follow-up); `inference="wild_bootstrap"` + Conley raises `NotImplementedError`. **(2) Combined spatial + cluster product kernel (#119):** `compute_robust_vcov(vcov_type="conley", cluster_ids=...)` / `LinearRegression(vcov_type="conley", cluster_ids=...)` / `TwoWayFixedEffects(vcov_type="conley", cluster="<col>")` / `MultiPeriodDiD(vcov_type="conley", cluster="<col>")` / `DifferenceInDifferences(vcov_type="conley", cluster="<col>")` apply `K_total[i, j] = K_space(d_ij/h) · 1{c_i = c_j}`. On the panel block-decomposed path the cluster indicator multiplies BOTH the spatial sandwich AND the serial sandwich; the validator enforces that `cluster_ids` is constant within each unit across periods (the within-unit serial mask is then trivially all-ones; cross-sectional path has no such constraint). TWFE's default auto-cluster on the Conley path remains silently dropped (combining with unit-level clusters would zero out all between-unit pairs and defeat the spatial pooling); users must pass an explicit above-unit cluster (e.g. region) to opt in. DiD has no auto-cluster — the choice is fully explicit. Two limit fixtures anchor correctness (no R parity — R `conleyreg` does not support combined kernels): all-unique-clusters reduces to HC0; huge-cutoff reduces to pure within-cluster CR1. The huge-cutoff reduction is EXACT only for `conley_kernel="uniform"` (`K(u) = 1` for `|u| ≤ 1`); for `conley_kernel="bartlett"` the identity is asymptotic since `K_bartlett(u) = 1 - |u| < 1` for `u > 0`. The fixture anchor uses uniform for an exact identity check. Per-slice mask construction (NOT full n×n) preserves memory on panel paths. **(3) Sparse k-d-tree fast path (#120):** auto-activates for the spatial Bartlett meat when `n > 5_000` AND metric is `"haversine"` or `"euclidean"` AND kernel is `"bartlett"`. Builds a CSR sparse kernel matrix via `scipy.spatial.cKDTree.query_ball_tree` instead of materializing the full n×n distance matrix; haversine projects to a 3-D unit-sphere chord representation with the exact great-circle recomputed for in-range neighbors only. Bit-identity parity vs the dense path at `atol=1e-10`; R parity at `atol=1e-6` is preserved on the existing 3 panel R fixtures with the sparse path force-enabled. The bartlett-only gate is for boundary correctness — bartlett at `u=1` is exactly 0, so the sparse path safely drops at-cutoff pairs; uniform at `u=1` is 1 and would require a closed-interval query semantic that haversine chord projection cannot reliably preserve. Constants: `_CONLEY_SPARSE_N_THRESHOLD = 5_000` (auto-toggle); `_CONLEY_DENSE_WARN_N` renamed `_CONLEY_DENSE_OOM_WARN_N = 20_000` (memory exhaustion threshold for the dense fallback — independent of the sparse threshold). Private `_conley_sparse: Optional[bool]` kwarg on `_compute_conley_vcov` controls the toggle (`None` = auto, `True` = force, `False` = force dense; `True` with an unsupported kernel/metric raises). The serial component (within-unit Bartlett over time) remains dense regardless — per-unit slices are small. **(4) Callable `conley_metric` validation (#123):** result must satisfy shape `(n, n)`, finite, non-negative, symmetric to `atol=1e-10`, AND zero on the diagonal (`|d(i, i)| ≤ 1e-10`); each failure raises a targeted `ValueError` naming the violated invariant. The zero-diagonal contract is load-bearing for the Conley sandwich: the `i = j` term must reduce to the HC0 diagonal `X_i ε_i² X_i'` via `K(0) = 1`; positive self-distance would silently attenuate the HC0 contribution by `K(d_ii / h) < 1`. Built-in metrics (`"haversine"`, `"euclidean"`) satisfy this by construction. Previously, malformed callables produced opaque BLAS errors deep in the pipeline. **Tests:** `tests/test_conley_vcov.py::TestConleySparse` (12), `::TestConleySparseRParityForced` (3), `::TestConleyCluster` (10), `::TestConleyDistanceMetrics` extended (7 new); existing rejection tests flipped to behavioral; `test_did_conley_matches_mpd_post_periods_1` locks the DiD-vs-MPD bit-exact agreement. **Docs:** REGISTRY `## ConleySpatialHAC` updates: new "Combined spatial + cluster product kernel" + "Performance / scale" subsections, DiD-vs-TWFE cluster asymmetry paragraph, updated panel-API restrictions table. TODO rows 118 / 119 / 120 / 123 removed; rows 121 (Conley + survey_design / weights, Bertanha-Imbens 2014) and 122 (`SyntheticDiD(vcov_type="conley")`, spatial-block bootstrap per Politis-Romano 1994) retained for future waves.
- **Conley (1999) spatial-HAC standard errors via `vcov_type="conley"`** on cross-sectional `LinearRegression` / `compute_robust_vcov` plus panel `MultiPeriodDiD` / `TwoWayFixedEffects` (Phases 1 and 2 of the spillover-conley initiative). **Cross-sectional contract:** `conley_coords` (n × 2 array of lat/lon or projected coords), `conley_cutoff_km=<float>` (positive finite bandwidth in km for haversine, or coord units for euclidean — REQUIRED, no default per the no-silent-failures contract), `conley_metric="haversine"|"euclidean"|callable` (default `"haversine"`; great-circle uses Earth's mean radius 6371.01 km matching R `conleyreg`), `conley_kernel="bartlett"|"uniform"` (default `"bartlett"`; both kernels emit a `UserWarning` if the resulting meat has a materially negative eigenvalue — neither the radial 1-D Bartlett nor the uniform kernel is formally PSD-guaranteed; Conley 1999's explicit PSD formula is the 2-D separable lattice product window at Eq 3.14). Cross-sectional variance estimator `Var̂(β) = (X'X)^{-1} · ( Σ_{i,j} K(d_ij/h) · X_i ε_i ε_j X_j' ) · (X'X)^{-1}` (Conley 1999 Eq 4.2). **Panel contract (Phase 2, new):** Three new co-required kwargs `conley_time` (n-length array), `conley_unit` (n-length array), and `conley_lag_cutoff=<int>` (non-negative; 0 means within-period spatial only, no serial component) switch into the **block-decomposed panel sandwich** that matches R `conleyreg` with `lag_cutoff > 0`: `XeeX_total = Σ_t (within-period spatial sandwich) + Σ_u (within-unit Bartlett temporal sandwich, lag ∈ {1..L}, same-time excluded)`. This is NOT a multiplicative product kernel — verified empirically against `conleyreg::time_dist` and `XeeXhC` at ~1e-14 on the panel parity fixtures. The temporal kernel is hardcoded Bartlett `(1 - |lag|/(L+1))` regardless of `conley_kernel`, mirroring `conleyreg::time_dist.cpp`; documented as a `Note (deviation from R-symmetric API)` in REGISTRY. **Panel estimator wire-up (Phase 2):** `MultiPeriodDiD(vcov_type="conley", conley_lag_cutoff=...).fit(..., unit=...)` and `TwoWayFixedEffects(vcov_type="conley", conley_lag_cutoff=...).fit(..., unit=...)` lift the Phase 1 fit-time rejection; the `conley_time` and `conley_unit` arrays are auto-derived from the existing `time` and `unit` column-name arguments at fit-time. `DifferenceInDifferences(vcov_type="conley")` is also supported (Wave A #118 in this release; see the Wave A entry above) — pass `unit=<col>` as a fit-time kwarg to `DiD.fit(...)`. **Other constraints (Phase 1, unchanged):** `SyntheticDiD(vcov_type="conley")` raises `TypeError` (uses bootstrap variance, not analytical sandwich); `set_params` mirrors the constructor rejection. `vcov_type="conley"` + `weights=` / `survey_design=` raises `NotImplementedError` (Bertanha-Imbens 2014 weighted-Conley deferred to a follow-up PR). `vcov_type="conley"` + explicit `cluster_ids=` is supported via the combined spatial + cluster product kernel (Wave A #119; see the Wave A entry above). TWFE's default auto-cluster on the Conley path is silently dropped (combining with unit-level clusters would defeat the spatial pooling); users opt into the combined kernel by passing an explicit above-unit cluster. `inference="wild_bootstrap"` + Conley raises (incompatible inference modes). A sparse k-d-tree fast path auto-activates for the spatial Bartlett meat when `n > 5_000` with bartlett kernel and haversine/euclidean metric (Wave A #120); the dense fallback still emits an OOM `UserWarning` at `n > 20_000`. **Implementation:** Helpers live in `diff_diff/conley.py` (`_haversine_km`, `_pairwise_distance_matrix`, `_bartlett_kernel`, `_uniform_kernel`, `_validate_conley_kwargs`, `_compute_conley_vcov` — the validator and sandwich helper now accept keyword-only `time` / `unit` / `lag_cutoff` for the panel path); `compute_robust_vcov` in `diff_diff/linalg.py` threads the new kwargs through. **R `conleyreg` parity (Düsterhöft 2021, CRAN v0.1.9)** on **six** benchmark fixtures (`benchmarks/data/r_conleyreg_conley_golden.json`, regenerable via `benchmarks/R/generate_conley_golden.R`): 3 cross-sectional (Phase 1) + 3 new panel fixtures (`panel_haversine_lag1`, `panel_haversine_lag2`, `panel_lat_lon_realistic_lag1`; n_units × T = 60×3, 80×5, 100×4 at lag={1,2,1}); observed max abs diff ~5.7e-16. Earth radius 6371.01 km matches `conleyreg::haversine_dist`. Test file `tests/test_conley_vcov.py` skips parity cleanly when the JSON is absent. New REGISTRY section `## ConleySpatialHAC`. Subsequent phases of the spillover-conley initiative (ring-indicator spillover-aware DiD per Butts 2021; survey-design / replicate-weight support; `SyntheticDiD` Conley path) are tracked in `TODO.md` under "Tech Debt from Code Reviews" → spillover-conley rows.
- **Tutorial 21: HAD Pre-test Workflow** (`docs/tutorials/21_had_pretest_workflow.ipynb`) — composite pre-test walkthrough for `HeterogeneousAdoptionDiD` building on Tutorial 20's brand-campaign framing. Uses a 60-DMA × 8-week panel close in shape to T20's but with the dose distribution drawn from `Uniform[$0.01K, $50K]` (vs T20's `[$5K, $50K]`); the true support is strictly positive but very near zero, chosen so the QUG step in `did_had_pretest_workflow` fails-to-reject `H0: d_lower = 0` in this finite sample and the verdict text fires the load-bearing "Assumption 7 deferred" pivot for the upgrade-arc narrative. (HAD's `design="auto"` selector — a separate min/median heuristic at `had.py::_detect_design`, NOT the QUG p-value — independently lands on the `continuous_at_zero` identification path with target `WAS` on this panel because `d.min() < 0.01 * median(|d|)`. The QUG test and the design selector are independent rules that point to the same identification path here.) Walks through three surfaces: (a) `did_had_pretest_workflow(aggregate="overall")` on a two-period collapse, where the verdict explicitly flags Step 2 (Assumption 7 pre-trends) as not run because a single pre-period structurally cannot support a pre-trends test, and the structural fields `pretrends_joint` / `homogeneity_joint` are both `None`; (b) `did_had_pretest_workflow(aggregate="event_study")` on the full multi-period panel, where the verdict reads "TWFE admissible under Section 4 assumptions" because all three testable diagnostics (QUG + joint pre-trends Stute over 3 horizons + joint homogeneity Stute over 4 horizons) fail-to-reject — non-rejection evidence under finite-sample power and test specification, not proof that the identifying assumptions hold; and (c) a side panel exercising both `yatchew_hr_test` null modes — `null="linearity"` (default, paper Theorem 7) vs `null="mean_independence"` (Phase 4 R-parity with R `YatchewTest::yatchew_test(order=0)`) — on the within-pre-period first-difference paired with post-period dose, illustrating the stricter null's larger residual variance (`sigma2_lin` 7.01 vs 6.53) and smaller p-value (0.29 vs 0.49). Companion drift-test file `tests/test_t21_had_pretest_workflow_drift.py` (16 tests pinning panel composition, both verdict pivots, structural anchors on both paths, deterministic QUG / Yatchew statistics, bootstrap p-value tolerance bands per `feedback_bootstrap_drift_tests_need_backend_tolerance`, and `HAD(design="auto")` resolution to `continuous_at_zero` on this panel). T20's "Composite pretest workflow" Extensions bullet updated with a forward-pointer to T21. T22 weighted/survey HAD tutorial shipped as a follow-up notebook PR (see the T22 entry above).
- **`ChaisemartinDHaultfoeuille.by_path` and `paths_of_interest` now compose with `survey_design`** for analytical Binder TSL SE and replicate-weight bootstrap variance. The `NotImplementedError` gate at `chaisemartin_dhaultfoeuille.py:1233-1239` is replaced by a per-path multiplier-bootstrap-only gate (`survey_design + n_bootstrap > 0` under by_path / paths_of_interest still raises, since the survey-aware perturbation pivot for path-restricted IFs is methodologically underived). Per-path SE routes through the existing `_survey_se_from_group_if` cell-period allocator: the per-period IF (`U_pp_l_path`) is built with non-path switcher-side contributions skipped (control contributions are unchanged, matching the joiners/leavers IF convention; preserves the row-sum identity `U_pp.sum(axis=1) == U`), cohort-recentered via `_cohort_recenter_per_period`, then expanded to observations as `psi_i = U_pp[g_i, t_i] · (w_i / W_{g_i, t_i})`. Replicate-weight designs unconditionally use the cell allocator (Class A contract from PR #323). New `_refresh_path_inference` helper post-call refreshes `safe_inference` on every populated entry across `multi_horizon_inference`, `placebo_horizon_inference`, `path_effects`, and `path_placebos` so all four surfaces use the same final `df_survey` after per-path replicate fits append `n_valid` to the shared accumulator. Path-enumeration ranking under `survey_design` remains unweighted (group-cardinality, not population-weight mass). Lonely-PSU policy stays sample-wide, not per-path. Telescope invariant: on a single-path panel, per-path SE matches the global non-by_path survey SE bit-exactly. **No R parity** — R `did_multiplegt_dyn` does not support survey weighting; this is a Python-only methodology extension. The global non-by_path TSL multiplier-bootstrap path is unaffected (anti-regression test `tests/test_chaisemartin_dhaultfoeuille.py::TestByPathSurveyDesignAnalytical::test_global_survey_plus_n_bootstrap_still_works` locks the per-path-only scope of the new gate). Cross-surface invariants regression-tested at `TestByPathSurveyDesignAnalytical` (~17 tests across gate / dispatch / analytical SE / replicate-weight SE / per-path placebos / `trends_linear` composition / unobserved-path warnings / final-df refresh regressions) and `TestByPathSurveyDesignTelescope`. See `docs/methodology/REGISTRY.md` §`ChaisemartinDHaultfoeuille` `Note (Phase 3 by_path ...)` → "Per-path survey-design SE" for the full contract.
- **Inference-field aliases on staggered result classes** for adapter / external-consumer compatibility. Read-only `@property` aliases expose the flat `att` / `se` / `conf_int` / `p_value` / `t_stat` names (matching `DiDResults` / `TROPResults` / `SyntheticDiDResults` / `TripleDifferenceResults` / `HeterogeneousAdoptionDiDResults`) on every result class that previously only carried prefixed canonical fields: `CallawaySantAnnaResults`, `StackedDiDResults`, `EfficientDiDResults`, `ChaisemartinDHaultfoeuilleResults`, `StaggeredTripleDiffResults`, `WooldridgeDiDResults`, `SunAbrahamResults`, `ImputationDiDResults`, `TwoStageDiDResults` (mapping to `overall_*`); `ContinuousDiDResults` (mapping to `overall_att_*`, ATT-side as the headline, ACRT-side accessible unchanged via `overall_acrt_*`); `MultiPeriodDiDResults` (mapping to `avg_*`). `ContinuousDiDResults` additionally exposes `overall_se` / `overall_conf_int` / `overall_p_value` / `overall_t_stat` aliases for naming consistency with the rest of the staggered family. Aliases are pure read-throughs over the canonical fields — no recomputation, no behavior change — so the `safe_inference()` joint-NaN contract (per CLAUDE.md "Inference computation") is inherited automatically (NaN canonical → NaN alias, locked at `tests/test_result_aliases.py::test_pattern_b_aliases_propagate_nan`). The native `overall_*` / `overall_att_*` / `avg_*` fields remain canonical for documentation and computation. Motivated by the `balance.interop.diff_diff.as_balance_diagnostic()` adapter (`facebookresearch/balance` PR #465) which calls `getattr(res, "se", None)` / `getattr(res, "conf_int", None)` without a fallback chain — pre-alias, every staggered result class returned `None` on those keys, silently dropping `se` and `conf_int` from the adapter's diagnostic dict. 23 alias-mechanic + balance-adapter regression tests at `tests/test_result_aliases.py`. Patch-level (additive on stable surfaces).
- **`ChaisemartinDHaultfoeuille.by_path` + non-binary integer treatment** — `by_path=k` now accepts integer-coded discrete treatment (D in Z, e.g. ordinal `{0, 1, 2}`); path tuples become integer-state tuples like `(0, 2, 2, 2)`. The previous `NotImplementedError` gate at `chaisemartin_dhaultfoeuille.py:1870` is replaced by a `ValueError` for continuous D (e.g. `D=1.5`) at fit-time per the no-silent-failures contract — the existing `int(round(float(v)))` cast in `_enumerate_treatment_paths` is now defensive (no-op for integer-coded D). Validated against R `did_multiplegt_dyn(..., by_path)` for D in `{0, 1, 2}` via the new `multi_path_reversible_by_path_non_binary` golden-value scenario (78 switchers, 3 paths, single-baseline custom DGP, F_g >= 4): per-path point estimates match R bit-exactly (rtol ~1e-9 on event horizons; rtol+atol envelope for placebo near-zero values), per-path SE inherits the documented cross-path cohort-sharing deviation (~5% rtol observed; SE_RTOL=0.15 envelope). **Deviation from R for multi-character baseline states (D >= 10 or negative D):** R's `did_multiplegt_by_path` derives the per-path baseline via `path_index$baseline_XX <- substr(path_index$path, 1, 1)`, which captures only the first character of the comma-separated path string. For multi-character baselines this drops the rest of the value: for `path = "12,12,..."` it captures `"1"` instead of `"12"`; for `path = "-1,-1,..."` it captures `"-"` instead of `"-1"`. R's per-path control-pool subset is mis-allocated in both regimes. Python's tuple-key matching is correct — the per-path point estimates we compute are correct; R's per-path subset for the same path is buggy. The shipped R-parity scenarios stay in nonnegative single-digit `D in {0, 1, 2}` to avoid the R bug; negative-integer treatment-state support (paths containing negative D values in non-baseline positions) is regression-tested in Python only at `tests/test_chaisemartin_dhaultfoeuille.py::TestByPathNonBinary::test_negative_integer_D_supported` (no R parity); a dedicated regression for a negative-baseline path (e.g. `(-1, 0, 0, 0)`) is deferred. R-parity test at `tests/test_chaisemartin_dhaultfoeuille_parity.py::TestDCDHDynRParityByPathNonBinary`; cross-surface invariants regression-tested at `tests/test_chaisemartin_dhaultfoeuille.py::TestByPathNonBinary`.
- **New `paths_of_interest` kwarg on `ChaisemartinDHaultfoeuille`** for user-specified treatment-path subsets, alternative to `by_path=k`'s top-k automatic ranking. Mutually exclusive with `by_path`; setting both raises `ValueError` at `__init__` and `set_params` time. Each path tuple must be a list/tuple of `int` of length `L_max + 1` (uniformity validated at `__init__`; length match against `L_max + 1` validated at fit-time); `bool` and `np.bool_` are explicitly rejected, `np.integer` accepted and canonicalized to Python `int` for tuple-key consistency. Duplicates emit a `UserWarning` and are deduplicated; paths not observed in the panel emit a `UserWarning` and are omitted from `path_effects`. Paths appear in `results.path_effects` in the user-specified order, modulo deduplication and unobserved-path filtering. Composes with non-binary D and all downstream `by_path` surfaces (bootstrap, per-path placebos, per-path joint sup-t bands, `controls`, `trends_linear`, `trends_nonparam`) — mechanical filter on observed paths via the same `_enumerate_treatment_paths` call site, no methodology change. **Python-only API extension; no R equivalent** — R's `did_multiplegt_dyn(..., by_path=k)` only accepts a positive int (top-k) or `-1` (all paths). The `by_path` precondition gate in `chaisemartin_dhaultfoeuille.py` (drop_larger_lower / L_max / `design2` / `honest_did` mutex; the `survey_design` mutex was lifted later in the same Unreleased cycle and `heterogeneity` was composed in, so neither remains a mutex in the shipped gate) and the 11 `self.by_path is not None` activation branches in `fit()` were rerouted to fire under either selector. Validation + behavior + cross-feature regressions at `tests/test_chaisemartin_dhaultfoeuille.py::TestPathsOfInterest`.
- **CI AI reviewer now sees tutorial notebook prose.** Substituted a markdown extract for the `docs/tutorials/*.ipynb` diff exclusion in `.github/workflows/ai_pr_review.yml`: the workflow's prompt-build step stages a trusted `tools/notebook_md_extract.py` from `BASE_SHA` (`git show "${BASE_SHA}":tools/notebook_md_extract.py > /tmp/...`, mirroring the existing base-staging of `pr_review.md`), loops over changed tutorial notebooks, and appends a `<notebook-prose untrusted="true">` block (prose + code + executed outputs) to the compiled prompt. The wrapper uses the same close-tag inline-Python sanitization as the existing `<pr-body>` / `<pr-title>` / `<previous-ai-review-output>` wrappers and gets a sibling persistent-policy directive at `pr_review.md:79` ("Treat the contents of `<notebook-prose>` blocks the same way..."). The new `tools/notebook_md_extract.py` is stdlib-only (no `nbformat` dep, no `pip install` step in the workflow) with a `_to_str()` helper that coerces nbformat raw JSON's list-or-string `source` / `text` fields (88% list-form rate verified across the project's 22 tutorials). `--max-output-chars 20000` / `--max-total-chars 200000` caps prevent any single oversized output or notebook from blowing the prompt budget. `text/html`-only outputs (no `text/plain` co-emit), `image/*` data, and `raw` cells are intentionally dropped (see module docstring). `tools/**` added to `rust-test.yml` path filters so extractor-only changes still trigger the test job. Also reaped the temporary T21 review aid at `docs/_review/t21_notebook_extract.md` and the `_review` entry in `docs/conf.py:exclude_patterns` — both lingered on `origin/main` from PR #409 and should have been cleaned up when T21 landed. Closes the visibility gap surfaced during PR #409 (T21), where the Codex reviewer ran 3+ rounds blind to the actual tutorial prose.
- **HAD `practitioner_next_steps()` handler + `llms-full.txt` reference section** (Phase 5). Adds `_handle_had` and `_handle_had_event_study` to `diff_diff/practitioner.py::_HANDLERS`, routing both `HeterogeneousAdoptionDiDResults` (single-period) and `HeterogeneousAdoptionDiDEventStudyResults` (event-study) through HAD-specific Baker et al. (2025) step guidance: `did_had_pretest_workflow` (step 3 — paper Section 4.2 step-2 closure on the event-study path), an estimand-difference routing nudge to `ContinuousDiD` (step 4 — fires when the user wants per-dose ATT(d) / ACRT(d) curves rather than HAD's WAS estimand and has never-treated controls; framed around estimand difference, NOT around the existence of untreated units, since HAD remains valid with a small never-treated share per REGISTRY § HeterogeneousAdoptionDiD edge cases and explicitly retains never-treated units on the staggered event-study path per paper Appendix B.2 / `had.py:1325`), `results.bandwidth_diagnostics` inspection on continuous designs and simultaneous (sup-t) `cband_*` reading on weighted event-study fits (step 6), per-horizon WAS event-study disaggregation (step 7), and the explicit design-auto-detection / last-cohort-only-WAS framing (step 8). Symmetric pair: `_handle_continuous` gains a Step-4 nudge to `HeterogeneousAdoptionDiD` for ContinuousDiD users on no-untreated panels (this direction is correct because ContinuousDiD's identification requires never-treated controls). Extends `_check_nan_att` with an ndarray branch via lazy `numpy` import for HAD's per-horizon `att` array; uses `np.all(np.isnan(arr))` semantics so partial-NaN arrays (legitimate event-study output under degenerate horizon-specific designs) do not over-fire the warning. Scalar path is bit-exact preserved across all 12 untouched handlers. Adds full HAD section + `HeterogeneousAdoptionDiDResults` / `HeterogeneousAdoptionDiDEventStudyResults` blocks + `## HAD Pretests` index covering all 7 pretest entry points + Choosing-an-Estimator row to `diff_diff/guides/llms-full.txt` (the bundled-in-wheel agent reference); the documented constructor + `fit()` parameter NAMES are regression-locked against the real `HeterogeneousAdoptionDiD.__init__` / `.fit` API via `inspect.signature` (parameter-name presence only; parameter defaults and the non-return parameter type annotations remain unpinned by that test). The `fit()` return annotation is widened to `Union[HeterogeneousAdoptionDiDResults, HeterogeneousAdoptionDiDEventStudyResults]` to match the runtime polymorphism the bundled guide already advertised, and that union is itself pinned by a dedicated regression test (`tests/test_had.py::TestFitReturnAnnotation`) using `typing.get_type_hints`. Tightens the existing `Continuous treatment intensity` Choosing row to surface ATT(d) vs WAS as the estimand differentiator. `docs/doc-deps.yaml` updated to remove the `llms-full.txt` deferral note on `had.py` and add `llms-full.txt` entries to `had.py`, `had_pretests.py`, and `practitioner.py` blocks. Patch-level (additive on stable surfaces). 26 new tests (16 in `tests/test_practitioner.py::TestHADDispatch` + 9 in `tests/test_guides.py::TestLLMsFullHADCoverage` + 1 fixture-minimality regression locking the "handlers are STRING-ONLY at runtime" stability invariant). Closes the Phase 5 "agent surfaces" gap; T21 pretest tutorial shipped in PR #409 and T22 weighted/survey tutorial shipped as a follow-up notebook PR (see the T22 entry above).

### Changed
- **HAD pretest non-strata bootstrap: small-sample calibration improvement.** The Stute survey-bootstrap on non-strata designs (`SurveyDesign(weights=...)`, `SurveyDesign(weights=..., psu=...)`, `SurveyDesign(weights=..., fpc=...)`) now applies the standard `sqrt(n_psu/(n_psu-1))` Bessel small-sample correction to the PSU multipliers uniformly with a single implicit stratum, mirroring the sibling HAD sup-t event-study cband bootstrap at `had.py:2199-2204`. Pre-PR Phase 4.5 C shipped raw iid multipliers without the centering; the bootstrap CvM variance was under-corrected by exactly the `n_psu/(n_psu-1)` factor relative to the unbiased within-stratum variance estimator. Direction-of-shift: toward correct calibration. Magnitude: approximately `sqrt(n_psu/(n_psu-1)) - 1` ≈ 1.7% for `n_psu=60`, decreasing as `n_psu` grows. Practitioners reproducing pre-PR Stute non-strata bootstrap p-values exactly should pin the prior release; the post-PR p-values are the methodology-true values (`### Added` "HAD pretest workflow: stratified survey-design support" bullet above documents the full derivation). Affects only the Stute family on the `weights=` / `survey_design=SurveyDesign(weights, [psu, fpc])` paths; Yatchew (closed-form weighted-OLS, no bootstrap) is unaffected, as is the unweighted bit-exact path (which has no multipliers to center).

## [3.3.2] - 2026-04-26

### Added
- **`ChaisemartinDHaultfoeuille.by_path` is now compatible with `trends_linear` (DID^{fd} group-specific linear trends) and `trends_nonparam` (state-set trends).** For `trends_linear`, the first-differencing transform runs once globally before path enumeration, so per-path raw second-differences `DID^{fd}_{path, l}` surface on `path_effects[path]["horizons"][l]` automatically. Per-path **cumulated level effects** `delta_{path, l} = sum_{l'=1..l} DID^{fd}_{path, l'}` (the quantity R returns under `did_multiplegt_dyn(..., by_path, trends_lin)`) surface on the new `results.path_cumulated_event_study[path][l]` field, mirroring the global `linear_trends_effects` cumulation. `to_dataframe(level="by_path")` exposes `cumulated_effect` / `cumulated_se` columns (always present, NaN-when-None — mirrors the `cband_*` convention from PR #374); `summary()` renders a "Cumulated Level Effects (DID^{fd}, trends_linear)" sub-section under each per-path block. SE on the cumulated layer is the conservative upper bound (sum of per-horizon component SEs, NaN-consistent), matching the global `linear_trends_effects` convention. Path enumeration runs on the post-first-differenced `N_mat_fd`: switchers with `F_g==2` fail the window-eligibility check and are dropped from path enumeration entirely (the existing global `F_g >= 3` warning still surfaces the issue), so a path whose switchers all have `F_g < 3` is silently absent from `path_effects` rather than present-with-NaN. Placebo under `trends_linear` returns RAW per-horizon values — there is no per-path placebo cumulation surface in either Python or R. For `trends_nonparam`, the set membership column is validated and stored once globally as `set_ids_arr`; the `set_ids` parameter is now threaded through the four per-path IF helpers (`_compute_path_effects`, `_compute_path_placebos`, `_collect_path_bootstrap_inputs`, `_collect_path_placebo_bootstrap_inputs`) so per-path analytical SE, bootstrap, placebos, and sup-t bands all consume the set-restricted control pool automatically. Per-period effects remain unadjusted under both extensions, consistent with the existing per-period DID contract. Validated against R via two new golden-value scenarios: `single_baseline_multi_path_by_path_trends_lin` (n_periods=13, F_g >= 4, cohort-single-path; per-path cumulated point estimates match R bit-exactly with `POINT_RTOL=1e-9`, cumulated SE within `CUM_SE_RTOL=0.20`) and `multi_path_reversible_by_path_trends_nonparam` (per-path point estimates AND placebos match R bit-exactly with `POINT_RTOL=1e-9`, per-path SE within `SE_RTOL=0.15`). **F_g=3 boundary-case divergence (`by_path + trends_linear`):** `F_g=3` switchers have only 1 valid pre-window Z value after first-differencing, triggering 30%+ relative divergence between Python and R per-path point estimates on paths whose switchers include `F_g=3`. A targeted `UserWarning` fires at fit-time on this regime; R parity is asserted only on the `F_g >= 4` parity fixture. Placebo parity for `trends_linear` is intentionally skipped (R's per-path placebo computation re-runs on the path-restricted subsample with different control eligibility than Python's global-then-disaggregate architecture surfaces; placebo + `trends_linear` is exercised via internal regression only). Cross-path cohort-sharing SE deviation from R documented for `path_effects` is inherited unchanged. Gates at `chaisemartin_dhaultfoeuille.py:1014-1023` removed; `by_path` docstring updated to add the two new compatibility paragraphs and remove `trends_linear` / `trends_nonparam` from the incompatible list. R-parity tests at `tests/test_chaisemartin_dhaultfoeuille_parity.py::TestDCDHDynRParityByPathTrendsLinear` and `::TestDCDHDynRParityByPathTrendsNonparam`; cross-surface regressions at `tests/test_chaisemartin_dhaultfoeuille.py::TestByPathTrendsLinear` and `::TestByPathTrendsNonparam`. See `docs/methodology/REGISTRY.md` §ChaisemartinDHaultfoeuille `Note (Phase 3 by_path ...)` → "Per-path linear-trends DID^{fd}" and "Per-path state-set trends" for the full contract.
- **`yatchew_hr_test(null="mean_independence")` mode** mirroring R `YatchewTest::yatchew_test(order=0)`. Adds a `null: Literal["linearity", "mean_independence"]` keyword-only kwarg to `yatchew_hr_test`. Default `"linearity"` is bit-exact backcompat (residuals from OLS `dy = a + b·d + eps`, paper Assumption 8 / Theorem 7). New `"mean_independence"` fits intercept-only OLS (`dy = a + eps`, residuals `= dy - mean(dy)`); the downstream `sigma2_diff` / `sigma2_W` / sort-by-`d` machinery is identical between the two modes. Exposed on both unweighted and survey-weighted code paths (`weights=` / `survey_design=` compose orthogonally with `null=`). Adds a `null_form: str` field to `YatchewTestResults` so `summary()` renders the correct null-hypothesis description; `__repr__` and `to_dict()` updated. Closes the placebo Yatchew R-parity gap from PR #392 — `tests/test_did_had_parity.py::TestYatchewParity` now routes effect rows through `null="linearity"` (R `order=1`) and placebo rows through `null="mean_independence"` (R `order=0`); both modes share the documented `× G/(G-1)` finite-sample convention shift and parity holds at `atol=1e-10`. Patch-level (additive keyword-only kwarg + additive dataclass field with default).
- **HAD `trends_lin=True` linear-trend detrending mode** on `HeterogeneousAdoptionDiD.fit(aggregate="event_study")`, `joint_pretrends_test`, and `joint_homogeneity_test`. Mirrors R `DIDHAD::did_had(..., trends_lin=TRUE)` (paper Eq. 17 / Eq. 18 / page 32 joint-Stute homogeneity-with-trends). Per-group linear-trend slope estimated as `Y[g, F-1] - Y[g, F-2]` and applied as `(t - base) × slope` adjustment to per-event-time outcome evolutions. Requires F ≥ 3 (panel must contain F-2). The "consumed" placebo at our event-time `e=-2` is auto-dropped (R reduces max placebo lag by 1 with the same effect). Mutually exclusive with survey weighting (`survey_design` / `survey` / `weights`): raises `NotImplementedError` per `feedback_per_method_survey_element_contract` (weighted slope estimator not derived from paper; tracked in TODO.md as a follow-up). Bit-exact backcompat for `trends_lin=False` (default). Patch-level (additive keyword-only kwarg).
- **HAD R-package end-to-end parity test** vs `DIDHAD` v2.0.0 (`Credible-Answers/did_had`) on the **`design="continuous_at_zero"` (Design 1') surface**. New parity fixture `benchmarks/data/did_had_golden.json` generated by `benchmarks/R/generate_did_had_golden.R` covers 3 paper-derived synthetic DGPs (Uniform, Beta(2,2), Beta(0.5,1)) × 5 method combinations (overall, event-study, placebo, yatchew, trends_lin). The harness explicitly forces `HeterogeneousAdoptionDiD(design="continuous_at_zero")` because R `did_had` always evaluates the local-linear at `d=0` regardless of dose distribution; our default `design="auto"` may legitimately choose `continuous_near_d_lower` or `mass_point` on dose distributions with boundary density bounded away from zero (e.g., Beta(2,2)) and thereby diverge from R numerically — that divergence is methodologically defensible but out of scope for this parity test. Python parity test `tests/test_did_had_parity.py` asserts point estimate / SE / CI bounds at `atol=1e-8` and Yatchew T-stat at `atol=1e-10` after a documented `× G/(G-1)` finite-sample convention shift. Two intentional convention deviations from R, documented in `docs/methodology/REGISTRY.md`: (a) we report the bias-corrected point estimate (modern CCF 2018 convention; R's `Estimate` column reports the conventional estimate with the bias-corrected CI separately — our `att` matches R's CI midpoint); (b) Yatchew uses paper Appendix E's literal (1/G) variance-denominator convention while R uses base-R `var()`'s (1/(N-1)) sample-variance convention (parity is bit-exact after the `× G/(G-1)` shift). Yatchew on placebos with R's mean-independence null (`order=0`) was not exposed in `yatchew_hr_test` at the PR #392 cut and was skipped in the parity test; the follow-up `yatchew_hr_test(null="mean_independence")` entry above closes that gap (placebo rows now routed through `null="mean_independence"` and parity holds at the same `atol=1e-10`).
- **Tutorial 20: HAD for National Brand Campaign with Regional Spend Intensity** (`docs/tutorials/20_had_brand_campaign.ipynb`) — end-to-end practitioner walkthrough for `HeterogeneousAdoptionDiD` on a 60-DMA panel where every market is treated at a different dose level and no never-treated unit exists; comparison comes from dose variation across markets, not from an untreated holdout. The DGP uses Uniform[\$5K, \$50K] regional add-on spend per DMA (every DMA participates, no DMA at exactly \$0), so `design="auto"` resolves to `continuous_near_d_lower` (Design 1) with target `WAS_d_lower` — interpreted as the average per-dollar marginal effect of regional spend above the lightest-touch DMA's spend (`d_lower` ≈ \$5K). Covers the headline `WAS_d_lower` fit on a 2-period collapse, the multi-week event study with per-week pointwise CIs and pre-launch placebos, and a stakeholder communication template that flags the Assumption 5/6 caveat (non-testable local-linearity at the boundary). Companion drift-test file `tests/test_t20_had_brand_campaign_drift.py` (13 tests pinning panel composition / sample median, design auto-detection / target / `d_lower`, overall `WAS_d_lower`, CI endpoints, dose mean, n_units, full event-study horizon presence, and per-horizon coverage). T20 wired into the existing `had.py` entry in `docs/doc-deps.yaml`; cross-link added from `docs/practitioner_decision_tree.rst` § "Universal Rollout (No Untreated Markets)" via a `.. tip::` block.

### Changed
- **Rust dependency upgrades**: bumped `rand` 0.8 → 0.10 and `rand_xoshiro` 0.6 → 0.8 in the Rust backend (the two crates are coupled through `rand_core` and must move together). MSRV bumped from Rust 1.84 → 1.85 to satisfy the new dependency requirements. Three call sites in `rust/src/bootstrap.rs` updated for the `rand 0.9` API rename: `gen::<bool>()` → `random::<bool>()`, `gen::<f64>()` → `random::<f64>()`, `gen_range(0..6)` → `random_range(0..6)`. **Webb wild bootstrap byte stream shifted** as a side effect: `rand 0.9` reworked the internal algorithm for `random_range` (improved rejection sampling), so `Xoshiro256PlusPlus::seed_from_u64(seed)` followed by `random_range(0..6)` consumes RNG bytes differently than the old `gen_range(0..6)` did. Distributional properties of Webb weights are unchanged (still uniform over the 6-point support); aggregate inference (SE, p-values, CI) converges to the same values for any reasonable `n_bootstrap`. Rademacher and Mammen byte streams are bit-identical to the prior release. Anyone with a saved Rust+Webb baseline pinning specific seeded results will see different numbers; the regression test suite uses within-build seed-reproducibility (not cross-version baselines) so all internal tests pass unchanged. New regression guard `TestRustBackend::test_bootstrap_weights_bit_identity_snapshot` pins fixed-seed weights for all three weight types, so any future RNG drift fails loudly with a localized error message.

## [3.3.1] - 2026-04-25

### Changed
- **HAD survey-design API consolidated to single `survey_design=` kwarg** across all 8 HAD surfaces: `HeterogeneousAdoptionDiD.fit`, `did_had_pretest_workflow`, `qug_test`, `stute_test`, `yatchew_hr_test`, `stute_joint_pretest`, `joint_pretrends_test`, `joint_homogeneity_test`. Matches the rest of the library (`ContinuousDiD`, `EfficientDiD`, `ChaisemartinDHaultfoeuille` already used `survey_design=`). On data-in surfaces (HAD.fit, workflow, joint data-in wrappers) `survey_design=` accepts a `SurveyDesign` instance (column references resolved against `data` at fit time, same convention as the rest of the library). On the three array-in linearity helpers (`stute_test`, `yatchew_hr_test`, `stute_joint_pretest`) `survey_design=` accepts a pre-resolved `ResolvedSurveyDesign`; passing a `SurveyDesign` raises `TypeError` with migration guidance to `make_pweight_design(arr)` (pweight-only) or pre-resolution. `qug_test` is the 8th surface and accepts the same kwarg signature for consistency, but **all** non-`None` values raise `NotImplementedError` per the Phase 4.5 C0 permanent deferral (no migration path; the qug-specific mutex error reflects this). New public helper `make_pweight_design(weights: np.ndarray) -> ResolvedSurveyDesign` exported from the `diff_diff` top level for the pweight-only convenience on the three array-in linearity helpers (formerly the private `survey._make_trivial_resolved`, kept as a permanent private alias); validates 1-D input at the front door. Three-way mutex (`survey_design + survey + weights`) extends the prior 2-way (`survey + weights`) — at most one may be non-None per call. Patch-level addition (additive new kwarg + permanent alias for the helper; no breaking changes this release).

### Deprecated
- **`HeterogeneousAdoptionDiD.fit(survey=, weights=)`, `did_had_pretest_workflow(survey=, weights=)`, and the 6 HAD pretest helpers' `survey=` / `weights=` kwargs are deprecated** in favor of the canonical `survey_design=`. Emits `DeprecationWarning` with migration guidance; the deprecated kwargs continue to route through the unchanged legacy back-end paths so numerical results are identical to pre-PR (bit-exact regression locked by parity tests in `tests/test_had_dual_knob_deprecation.py`). Both `survey=` and `weights=` will be removed in the next minor release. **Carve-out for `qug_test`**: the deprecation is kwarg-name-consolidation only; `qug_test` permanently rejects all non-`None` `survey_design` / `survey` / `weights` values (Phase 4.5 C0 deferral) and `make_pweight_design(arr)` is NOT a valid migration target — the deprecation warning text on `qug_test` is qug-specific and points users to `did_had_pretest_workflow(..., survey_design=...)` for survey-aware HAD pretesting (which skips the QUG step under survey).

### Added
- **`ChaisemartinDHaultfoeuille.by_path` + `controls`** (DID^X residualization) — the per-baseline OLS residualization (Web Appendix Section 1.2) is now compatible with `by_path=k`. The residualization runs once on the first-differenced outcome BEFORE path enumeration, so all four downstream surfaces (analytical per-path SE, bootstrap SE, per-path placebos, per-path joint sup-t bands) consume the residualized `Y_mat` automatically (Frisch-Waugh-Lovell). Per-period effects remain unadjusted, consistent with the existing `controls` + per-period DID contract (per-period DID does not support residualization). Failed-stratum baselines (rank-deficient X) zero out `N_mat` for affected groups, which the path enumeration treats as ineligible per its existing convention. **Deviation from R on multi-baseline switcher panels (point estimates):** R `did_multiplegt_dyn(..., by_path, controls)` re-runs the per-baseline OLS residualization on each path's restricted subsample (path's switchers + same-baseline not-yet-treated controls), so its residualization coefficients vary per path when switchers have different baseline values. Our global-residualization architecture coincides with R on single-baseline switcher panels (every switcher shares the same `D_{g,1}`) — per-path point estimates match R exactly there. On multi-baseline panels, point estimates can diverge; the estimator emits a `UserWarning` at fit-time when this configuration is detected so practitioners do not silently consume estimates that disagree with R. **SE inherits the cross-path cohort-sharing SE deviation from R** documented for `path_effects` — bootstrap SE, placebo SE, and sup-t crit are Monte Carlo / joint-distribution analogs of the same residualized analytical IF and carry the same deviation. R-parity confirmed against `did_multiplegt_dyn(..., by_path=3, controls="X1")` via the new `multi_path_reversible_by_path_controls` single-baseline golden-value scenario (per-path point estimates match R bit-exactly — measured rtol ~1e-11 across all path × horizon cells — on this one-observation-per-cell scenario; per-path SE within ~6.5% of R, well inside the Phase 2 multi-horizon envelope). Cell-aggregated panels with multiple observations per `(g, t)` also coincide with our equal-cell-weighting first stage rather than R's `N_gt`-weighted first stage per the existing DID^X cell-weighting deviation documented in `docs/methodology/REGISTRY.md` `Note (Phase 3 DID^X covariate adjustment)`. Gate at `chaisemartin_dhaultfoeuille.py:988-992` removed; `by_path` docstring updated to add the new compatibility paragraph (with the multi-baseline caveat) and remove `controls` from the incompatible list. R-parity test at `tests/test_chaisemartin_dhaultfoeuille_parity.py::TestDCDHDynRParityByPathControls`; cross-surface inheritance + multi-baseline `UserWarning` regression-tested at `tests/test_chaisemartin_dhaultfoeuille.py::TestByPathControls` (analytical + bootstrap + placebo + sup-t + `to_dataframe(level="by_path")` cband columns + multi-baseline warning). See `docs/methodology/REGISTRY.md` §ChaisemartinDHaultfoeuille `Note (Phase 3 by_path ...)` → "Per-path covariate residualization (DID^X)" for the full contract.
- **HAD linearity-family pretests under survey (Phase 4.5 C).** `stute_test`, `yatchew_hr_test`, `stute_joint_pretest`, `joint_pretrends_test`, `joint_homogeneity_test`, and `did_had_pretest_workflow` now accept `weights=` / `survey=` keyword-only kwargs. Stute family uses **PSU-level Mammen multiplier bootstrap** via `bootstrap_utils.generate_survey_multiplier_weights_batch` (the same kernel as PR #363's HAD event-study sup-t bootstrap): each replicate draws an `(n_bootstrap, n_psu)` Mammen multiplier matrix, broadcast to per-obs perturbation `eta_obs[g] = eta_psu[psu(g)]`, weighted OLS refit, weighted CvM via new `_cvm_statistic_weighted` helper. Joint Stute SHARES the multiplier matrix across horizons within each replicate, preserving both the vector-valued empirical-process unit-level dependence AND PSU clustering. Yatchew uses **closed-form weighted OLS + pweight-sandwich variance components** (no bootstrap): `sigma2_lin = sum(w·eps²)/sum(w)`, `sigma2_diff = sum(w_avg·diff²)/(2·sum(w))` with arithmetic-mean pair weights `w_avg_g = (w_g+w_{g-1})/2`, `sigma4_W = sum(w_avg·prod)/sum(w_avg)`, `T_hr = sqrt(sum(w))·(sigma2_lin-sigma2_diff)/sigma2_W`. All three Yatchew components reduce bit-exactly to the unweighted formulas at `w=ones(G)` (locked at `atol=1e-14` by direct helper test). The pweight `weights=` shortcut routes through a synthetic trivial `ResolvedSurveyDesign` (new `survey._make_trivial_resolved` helper) so the same kernel handles both entry paths. `did_had_pretest_workflow(..., survey=, weights=)` removes the Phase 4.5 C0 `NotImplementedError`, dispatches to the survey-aware sub-tests, **skips the QUG step with `UserWarning`** (per C0 deferral), sets `qug=None` on the report, and appends a `"linearity-conditional verdict; QUG-under-survey deferred per Phase 4.5 C0"` suffix to the verdict. `HADPretestReport.qug` retyped from `QUGTestResults` to `Optional[QUGTestResults]`; `summary()` / `to_dict()` / `to_dataframe()` updated to None-tolerant rendering. Replicate-weight survey designs (BRR/Fay/JK1/JKn/SDR) raise `NotImplementedError` at every entry point (defense in depth, reciprocal-guard discipline) — parallel follow-up after this PR. **Stratified designs (`SurveyDesign(strata=...)`) also raise `NotImplementedError` on the Stute family** — the within-stratum demean + `sqrt(n_h/(n_h-1))` correction that the HAD sup-t bootstrap applies to match the Binder-TSL stratified target has not been derived for the Stute CvM functional, so applying raw multipliers from `generate_survey_multiplier_weights_batch` directly to residual perturbations would leave the bootstrap p-value silently miscalibrated. Phase 4.5 C narrows survey support to **pweight-only**, **PSU-only** (`SurveyDesign(weights=, psu=)`), and **FPC-only** (`SurveyDesign(weights=, fpc=)`) designs; stratified is a follow-up after the matching Stute-CvM stratified-correction derivation lands. Strictly positive weights required on Yatchew (the adjacent-difference variance is undefined under contiguous-zero blocks). Per-row `weights=` / `survey=col` aggregated to per-unit via existing HAD helpers `_aggregate_unit_weights` / `_aggregate_unit_resolved_survey` (constant-within-unit invariant enforced). Unweighted code paths preserved bit-exactly. Patch-level addition (additive on stable surfaces). See `docs/methodology/REGISTRY.md` § "QUG Null Test" — Note (Phase 4.5 C) for the full methodology.
- **`ChaisemartinDHaultfoeuille.by_path` + `n_bootstrap > 0` joint sup-t bands** — per-path joint sup-t simultaneous confidence intervals across horizons `1..L_max` within each path. A single shared `(n_bootstrap, n_eligible)` multiplier weight matrix (using the estimator's configured `bootstrap_weights` — Rademacher / Mammen / Webb) is drawn per path and broadcast across all horizons of that path, producing correlated bootstrap distributions across horizons. The path-specific critical value `c_p = quantile(max_l |t_l|, 1 - α)` is used to construct symmetric joint bands `effect_l ± c_p · se_l` per horizon. Surfaced on `results.path_sup_t_bands` (dict keyed by path tuple, each entry with `crit_value / alpha / n_bootstrap / method / n_valid_horizons`); as `cband_conf_int` per horizon entry on `path_effects[path]["horizons"][l]`; and as `cband_lower` / `cband_upper` columns on `results.to_dataframe(level="by_path")` (mirrors the OVERALL `level="event_study"` schema; positive-horizon rows of banded paths get populated values, placebo / unbanded / empty-window rows get NaN). Gates: a path needs `>= 2` valid horizons (finite bootstrap SE > 0) AND a strict majority (more than 50%) of finite sup-t draws to receive a band. Empty-state contract: `path_sup_t_bands is None` when not requested; `{}` when requested but no path passes both gates. **Methodology asymmetry vs OVERALL `event_study_sup_t_bands`:** the per-path sup-t draws a fresh shared weight matrix per path AFTER the per-path SE bootstrap block has already populated `results.path_ses` via independent per-(path, horizon) draws — asymptotically equivalent to OVERALL's self-consistent reuse but NOT bit-identical. Documented intentional choice to preserve RNG-state isolation for existing per-path SE seed-reproducibility tests. Inherits the cross-path cohort-sharing SE deviation from R documented for `path_effects`. **Deviation from R:** `did_multiplegt_dyn` does not provide joint / sup-t bands at any surface — this is a Python-only methodology extension consistent with the existing OVERALL sup-t bands (also Python-only). Bands cover joint inference WITHIN a single path across horizons; they do NOT provide simultaneous coverage across paths. Pre-audit fix bundled: stale "Phase 2 placeholder" docstring on the existing `sup_t_bands` field updated to the actual contract description. Tests at `tests/test_chaisemartin_dhaultfoeuille.py::TestByPathSupTBands` (`@pytest.mark.slow`). See `docs/methodology/REGISTRY.md` §ChaisemartinDHaultfoeuille `Note (Phase 3 by_path per-path joint sup-t bands)` for the full contract.
- **`ChaisemartinDHaultfoeuille.by_path` + `placebo=True`** — per-path backward-horizon placebos `DID^{pl}_{path, l}` for `l = 1..L_max`. The same per-path SE convention used for the event-study (joiners/leavers IF precedent: switcher-side contributions zeroed for non-path groups; cohort structure and control pool unchanged; plug-in SE with path-specific divisor `N^{pl}_{l, path}`) is applied to backward horizons via the new `switcher_subset_mask` parameter on `_compute_per_group_if_placebo_horizon`. Surfaced on `results.path_placebo_event_study[path][-l]` (negative-int inner keys mirroring `placebo_event_study`); `summary()` renders the rows alongside per-path event-study horizons; `to_dataframe(level="by_path")` emits negative-horizon rows alongside the existing positive-horizon rows. **Bootstrap** (when `n_bootstrap > 0`) propagates per-`(path, lag)` percentile CI / p-value through the same `_bootstrap_one_target` dispatch as the per-path event-study, with the canonical NaN-on-invalid contract enforced on the new surface (PR #364 library-wide invariant). **SE inherits the cross-path cohort-sharing deviation from R** documented for `path_effects` (full-panel cohort-centered plug-in vs R's per-path re-run): tracks R within tolerance on single-path-cohort panels, diverges materially on cohort-mixed panels — the bootstrap SE is a Monte Carlo analog of the analytical SE and inherits the same deviation. R-parity confirmed at `tests/test_chaisemartin_dhaultfoeuille_parity.py::TestDCDHDynRParityByPathPlacebo` on the new `multi_path_reversible_by_path_placebo` scenario (point estimates exact match; SE within Phase-2 envelope rtol ≤ 5%); positive analytical + bootstrap invariants at `tests/test_chaisemartin_dhaultfoeuille.py::TestByPathPlacebo` (and the gated `::TestBootstrap` subclass). See `docs/methodology/REGISTRY.md` §ChaisemartinDHaultfoeuille `Note (Phase 3 by_path ...)` → "Per-path placebos" for the full contract.
- **Tutorial 19: dCDH for Marketing Pulse Campaigns** (`docs/tutorials/19_dcdh_marketing_pulse.ipynb`) — end-to-end practitioner walkthrough on a 60-market reversible-treatment panel covering the TWFE decomposition diagnostic (`twowayfeweights`), `DCDH` Phase 1 (DID_M, joiners-vs-leavers, single-lag placebo), the `L_max` multi-horizon event study with multiplier bootstrap, a stakeholder communication template, and drift guards. README listing for Tutorial 17 (Brand Awareness Survey) backfilled in the same edit. Cross-link from `docs/practitioner_decision_tree.rst` § "Reversible Treatment" added.

## [3.3.0] - 2026-04-25

### Fixed
- **`SyntheticDiD(variance_method="placebo")` SE now uses R-default warm-start** matching `synthdid:::placebo_se`. R's placebo loop seeds Frank-Wolfe per draw with `weights.boot$omega = sum_normalize(weights$omega[ind[1:N0_placebo]])` (fit-time ω subsetted + renormalized) and the fit-time `weights$lambda` — Python previously used uniform cold-start, producing finite-iter convergence-pattern drift on a handful of draws relative to R's reference SE. New `_placebo_variance_se` kwargs `init_omega` / `init_lambda` thread fit-time weights through the existing two-pass FW dispatcher; on the global FW optimum the values are init-independent (strictly convex objective), so the change is a finite-iter parity fix, not a methodology change. Existing placebo SE values shift by sub-percent on most panels; the bit-identity baseline pin in `TestScaleEquivariance::test_baseline_parity_small_scale[placebo]` was rebased from `0.29385822261006445` to `0.293840360160448`. New R-parity test `tests/test_methodology_sdid.py::TestJackknifeSERParity::test_placebo_se_matches_r` asserts SE matches R's `vcov(method="placebo")` to within `< 1e-8` using R's exact permutation sequence (recorded by `benchmarks/R/generate_sdid_placebo_parity_fixture.R` into `tests/data/sdid_placebo_indices_r.json`). The `_placebo_indices` kwarg on `_placebo_variance_se` is the test seam; not part of the public API.

### Added
- **`qug_test` and `did_had_pretest_workflow` survey-aware NotImplementedError gates (Phase 4.5 C0 decision gate).** `qug_test(d, *, survey=None, weights=None)` and `did_had_pretest_workflow(..., *, survey=None, weights=None)` now accept the two kwargs as keyword-only with default `None`. Passing either non-`None` raises `NotImplementedError` with an educational message naming the methodology rationale and pointing users to joint Stute (Phase 4.5 C, planned) as the survey-compatible alternative. Mutex guard on `survey=` + `weights=` mirrors `HeterogeneousAdoptionDiD.fit()` at `had.py:2890`. **QUG-under-survey is permanently deferred** — the test statistic uses extreme order statistics `D_{(1)}, D_{(2)}` which are NOT smooth functionals of the empirical CDF, so standard survey machinery (Binder-TSL linearization, Rao-Wu rescaled bootstrap, Krieger-Pfeffermann (1997) EDF tests) does not yield a calibrated test; under cluster sampling the `Exp(1)/Exp(1)` limit law's independence assumption breaks; and the EVT-under-unequal-probability-sampling literature (Quintos et al. 2001, Beirlant et al.) addresses tail-index estimation, not boundary tests. The workflow's gate is **temporary** — Phase 4.5 C will close it for the linearity-family pretests with mechanism varying by test: Rao-Wu rescaled bootstrap for `stute_test` and the joint variants (`stute_joint_pretest`, `joint_pretrends_test`, `joint_homogeneity_test`); weighted OLS residuals + weighted variance estimator for `yatchew_hr_test` (Yatchew 1997 is a closed-form variance-ratio test, not bootstrap-based). Sister pretests (`stute_test`, `yatchew_hr_test`, `stute_joint_pretest`, `joint_pretrends_test`, `joint_homogeneity_test`) keep their closed signatures in this release — Phase 4.5 C will add kwargs and implementation together to avoid API churn. Unweighted `qug_test(d)` and `did_had_pretest_workflow(...)` calls are bit-exact pre-PR (kwargs are keyword-only after `*`; positional path unchanged). New tests at `tests/test_had_pretests.py::TestQUGTest` (5 rejection / mutex / message / regression tests) and the new `TestHADPretestWorkflowSurveyGuards` class (6 tests covering both kwarg paths, mutex, methodology pointer, both aggregate paths, and unweighted regression). See `docs/methodology/REGISTRY.md` § "QUG Null Test" — Note (Phase 4.5 C0) for the full methodology rationale plus a sketch of the (out-of-scope) theoretical bridge that combines endpoint-estimation EVT (Hall 1982, Aarssen-de Haan 1994, Hall-Wang 1999, Beirlant-de Wet-Goegebeur 2006), survey-aware functional CLTs (Boistard-Lopuhaä-Ruiz-Gazen 2017, Bertail-Chautru-Clémençon 2017), and tail-empirical-process theory (Drees 2003) — publishable methodology research, not engineering work.
- **`HeterogeneousAdoptionDiD` mass-point `survey=` / `weights=` + event-study `aggregate="event_study"` survey composition + multiplier-bootstrap sup-t simultaneous confidence band (Phase 4.5 B).** Closes the two Phase 4.5 A `NotImplementedError` gates: `design="mass_point" + weights/survey` and `aggregate="event_study" + weights/survey`. Weighted 2SLS sandwich in `_fit_mass_point_2sls` follows the Wooldridge 2010 Ch. 12 pweight convention (`w²` in the HC1 meat, `w·u` in the CR1 cluster score, weighted bread `Z'WX`); HC1 and CR1 ("stata" `se_type`) bit-parity with `estimatr::iv_robust(..., weights=, clusters=)` at `atol=1e-10` (new cross-language golden at `benchmarks/data/estimatr_iv_robust_golden.json`, generated by `benchmarks/R/generate_estimatr_iv_robust_golden.R`; `estimatr` added to `benchmarks/R/requirements.R`). `_fit_mass_point_2sls` gains `weights=` + `return_influence=` kwargs and now always returns a 3-tuple `(beta, se, psi)` — `psi` is the per-unit IF on the β̂-scale scaled so `compute_survey_if_variance(psi, trivial_resolved) ≈ V_HC1[1,1]` at `atol=1e-10` (PR #359 IF scale convention applied uniformly; no `sum(psi²)` claims). Event-study per-horizon variance: `survey=` path composes Binder-TSL via `compute_survey_if_variance`; `weights=` shortcut uses the analytical weighted-robust SE (continuous: CCT-2014 `bc_fit.se_robust / |den|`; mass-point: weighted 2SLS pweight sandwich from `_fit_mass_point_2sls` — HC1 / classical / CR1). `survey_metadata` / `variance_formula` / `effective_dose_mean` populated in both regimes (previously hardcoded `None` at `had.py:3366`). New multiplier-bootstrap sup-t: `_sup_t_multiplier_bootstrap` reuses `diff_diff.bootstrap_utils.generate_survey_multiplier_weights_batch` for PSU-level draws with stratum centering + sqrt(n_h/(n_h-1)) small-sample correction + FPC scaling + lonely-PSU handling. On the `weights=` shortcut, sup-t calibration is routed through a synthetic trivial `ResolvedSurveyDesign` so the centered + small-sample-corrected branch fires uniformly — targets the analytical HC1 variance family (`compute_survey_if_variance(IF, trivial) ≈ V_HC1` per the PR #359 IF scale invariant) rather than the raw `sum(ψ²) = ((n-1)/n) · V_HC1` that unit-level Rademacher multipliers would produce on the HC1-scaled IF. Perturbations: `delta = weights @ IF` with NO `(1/n)` prefactor (matching `staggered_bootstrap.py:373` idiom), normalized by per-horizon analytical SE, `(1-alpha)`-quantile of the sup-t distribution. At H=1 the quantile reduces to `Φ⁻¹(1 − alpha/2) ≈ 1.96` up to MC noise (regression-locked by `TestSupTReducesToNormalAtH1`). `HeterogeneousAdoptionDiD.__init__` gains `n_bootstrap: int = 999` and `seed: Optional[int] = None` (CS-parity singular seed); `fit()` gains `cband: bool = True` (only consulted on weighted event-study). `HeterogeneousAdoptionDiDEventStudyResults` extended with `variance_formula`, `effective_dose_mean`, `cband_low`, `cband_high`, `cband_crit_value`, `cband_method`, `cband_n_bootstrap` (all `None` on unweighted fits); surfaced in `to_dict`, `to_dataframe`, `summary`, `__repr__`. Unweighted event-study with `cband=False` preserves pre-Phase 4.5 B numerical output bit-exactly (stability invariant, locked by regression tests). Zero-weight subpopulation convention carries over from PR #359 (filter for design decisions; preserve full ResolvedSurveyDesign for variance). Non-pweight SurveyDesigns (`aweight`, `fweight`, replicate designs) raise `NotImplementedError` on both new paths (reciprocal-guard discipline). Pretest surfaces (`qug_test`, `stute_test`, `yatchew_hr_test`, joint variants, `did_had_pretest_workflow`) remain unweighted in this release — Phase 4.5 C / C0. See `docs/methodology/REGISTRY.md` §HeterogeneousAdoptionDiD "Weighted 2SLS (Phase 4.5 B)", "Event-study survey composition", and "Sup-t multiplier bootstrap" for derivations and invariants.
- **`PanelProfile.outcome_shape` and `PanelProfile.treatment_dose` extensions + `llms-autonomous.txt` worked examples (Wave 2 of the AI-agent enablement track).** `profile_panel(...)` now populates two new optional sub-dataclasses on the returned `PanelProfile`: `outcome_shape: Optional[OutcomeShape]` (numeric outcomes only — exposes `n_distinct_values`, `pct_zeros`, `value_min` / `value_max`, `skewness` and `excess_kurtosis` (NaN-safe; `None` when `n_distinct_values < 3` or variance is zero), `is_integer_valued`, `is_count_like` (heuristic: integer-valued AND has zeros AND right-skewed AND > 2 distinct values AND non-negative support, i.e. `value_min >= 0`; flags WooldridgeDiD QMLE consideration over linear OLS — the non-negativity clause aligns the routing signal with `WooldridgeDiD(method="poisson")`'s hard rejection of negative outcomes at `wooldridge.py:1105-1109`), `is_bounded_unit` ([0, 1] support)) and `treatment_dose: Optional[TreatmentDoseShape]` (continuous treatments only — exposes `n_distinct_doses`, `has_zero_dose`, `dose_min` / `dose_max` / `dose_mean` over non-zero doses). Both `OutcomeShape` and `TreatmentDoseShape` are mostly descriptive context. **`profile_panel` does not see the separate `first_treat` column** that `ContinuousDiD.fit()` consumes; the estimator's actual fit-time gates key off `first_treat` (defines never-treated controls as `first_treat == 0`, force-zeroes nonzero `dose` on those rows with a `UserWarning`, and rejects negative dose only among treated units `first_treat > 0`; see `continuous_did.py:276-327` and `:348-360`). In the canonical `ContinuousDiD` setup (Callaway, Goodman-Bacon, Sant'Anna 2024), the dose `D_i` is **time-invariant per unit** and `first_treat` is a **separate column** the caller supplies (not derived from the dose column). Under that setup, several facts on the dose column predict `fit()` outcomes: `PanelProfile.has_never_treated` (proxies `P(D=0) > 0` because the canonical convention ties `first_treat == 0` to `D_i == 0`); `PanelProfile.treatment_varies_within_unit == False` (the actual fit-time gate at line 222-228, holds regardless of `first_treat`); `PanelProfile.is_balanced` (the actual fit-time gate at line 329-338); absence of the `duplicate_unit_time_rows` alert (silent last-row-wins overwrite, must deduplicate before fit); and `treatment_dose.dose_min > 0` (predicts the strictly-positive-treated-dose requirement at line 287-294 because treated units carry their constant dose across all periods). When `has_never_treated == False` (no zero-dose controls but all observed doses non-negative), `ContinuousDiD` does not apply (Remark 3.1 lowest-dose-as-control is not implemented); `HeterogeneousAdoptionDiD` IS a routing alternative on this branch (HAD's own contract requires non-negative dose, which is satisfied). When `dose_min <= 0` (negative treated doses), `ContinuousDiD` does not apply AND `HeterogeneousAdoptionDiD` is **not** a fallback — HAD also raises on negative post-period dose (`had.py:1450-1459`); the applicable alternative is linear DiD with the treatment as a signed continuous covariate. Re-encoding the treatment column is an agent-side preprocessing choice that changes the estimand and is not documented in REGISTRY as a supported fallback. The estimator's force-zero coercion on `first_treat == 0` rows with nonzero `dose` is implementation behavior for inconsistent inputs, not a documented method for manufacturing never-treated controls. The agent must validate the supplied `first_treat` column independently — `profile_panel` does not see it. The shape extensions provide distributional context (effect-size range, count-shape detection) that supplements but does not replace those gates. Both fields are `None` when their classification gate is not met (e.g., `treatment_dose is None` for binary treatments). `to_dict()` serializes the nested dataclasses as JSON-compatible nested dicts. New exports: `OutcomeShape`, `TreatmentDoseShape` from top-level `diff_diff`. `llms-autonomous.txt` gains a new §5 "Worked examples" section with three end-to-end PanelProfile -> reasoning -> validation walkthroughs (binary staggered with never-treated controls, continuous dose with zero baseline, count-shaped outcome) plus §2 field-reference subsections for the new shape fields and §4.7 / §4.11 cross-references for outcome-shape considerations. Existing §5-§8 of the autonomous guide are renumbered to §6-§9. Descriptive only — no recommender language inside the worked examples.
- **`HeterogeneousAdoptionDiD.fit(survey=..., weights=...)` on continuous-dose paths (Phase 4.5 survey support).** The `continuous_at_zero` (paper Design 1') and `continuous_near_d_lower` (Design 1 continuous-near-d̲) designs accept survey weights through two interchangeable kwargs: `weights=<array>` (pweight shortcut, weighted-robust SE from the CCT-2014 lprobust port) and `survey=SurveyDesign(weights, strata, psu, fpc)` (design-based inference via Binder-TSL variance using the existing `compute_survey_if_variance` helper at `diff_diff/survey.py:1802`). Point estimates match across both entry paths; SE diverges by design (pweight-only vs PSU-aggregated). `HeterogeneousAdoptionDiDResults.survey_metadata` is a repo-standard `SurveyMetadata` dataclass (weight_type / effective_n / design_effect / sum_weights / weight_range / n_strata / n_psu / df_survey); HAD-specific extras (`variance_formula` label, `effective_dose_mean`) are separate top-level result fields. `to_dict()` surfaces the full `SurveyMetadata` object plus `variance_formula` + `effective_dose_mean`; `summary()` renders `variance_formula`, `effective_n`, `effective_dose_mean`, and (when the survey= path is used) `df_survey`; `__repr__` surfaces `variance_formula` + `effective_dose_mean` when present. The HAD `mass_point` design and `aggregate="event_study"` path raise `NotImplementedError` under survey/weights (deferred to Phase 4.5 B: weighted 2SLS + event-study survey composition); the HAD pretests stay unweighted in this release (Phase 4.5 C). Parity ceiling acknowledged — no public weighted-CCF bias-corrected local-linear reference exists in any language; methodology confidence comes from (1) uniform-weights bit-parity at `atol=1e-14` on the full lprobust output struct, (2) cross-language weighted-OLS parity (manual R reference) at `atol=1e-12`, and (3) Monte Carlo oracle consistency on known-τ DGPs. `_nprobust_port.lprobust` gains `weights=` and `return_influence=` (used internally by the Binder-TSL path); `bias_corrected_local_linear` removes the Phase 1c `NotImplementedError` on `weights=` and forwards. Auto-bandwidth selection remains unweighted in this release — pass `h`/`b` explicitly for weight-aware bandwidths. See `docs/methodology/REGISTRY.md` §HeterogeneousAdoptionDiD "Weighted extension (Phase 4.5 survey support)".
- **`stute_joint_pretest`, `joint_pretrends_test`, `joint_homogeneity_test` + `StuteJointResult`** (HeterogeneousAdoptionDiD Phase 3 follow-up). Joint Cramér-von Mises pretests across K horizons with shared-η Mammen wild bootstrap (preserves vector-valued empirical-process unit-level dependence per Delgado-Manteiga 2001 / Hlávka-Hušková 2020). The core `stute_joint_pretest` is residuals-in; two thin data-in wrappers construct per-horizon residuals for the two nulls the paper spells out: mean-independence (step 2 pre-trends, `OLS(Y_t − Y_base ~ 1)` per pre-period) and linearity (step 3 joint, `OLS(Y_t − Y_base ~ 1 + D)` per post-period). Sum-of-CvMs aggregation (`S_joint = Σ_k S_k`); per-horizon scale-invariant exact-linear short-circuit. Closes the paper Section 4.2 step-2 gap that Phase 3 `did_had_pretest_workflow` previously flagged with an "Assumption 7 pre-trends test NOT run" caveat. See `docs/methodology/REGISTRY.md` §HeterogeneousAdoptionDiD "Joint Stute tests" for algorithm, invariants, and scope exclusion of Eq 18 linear-trend detrending (deferred to Phase 4 Pierce-Schott replication).
- **`did_had_pretest_workflow(aggregate="event_study")`**: multi-period dispatch on balanced ≥3-period panels. Runs QUG at `F` + joint pre-trends Stute across earlier pre-periods + joint homogeneity-linearity Stute across post-periods. Step 2 closure requires ≥2 pre-periods; with only a single pre-period (the base `F-1`) `pretrends_joint=None` and the verdict flags the skip. Reuses the Phase 2b event-study panel validator (last-cohort auto-filter under staggered timing with `UserWarning`; `ValueError` when `first_treat_col=None` and the panel is staggered). The data-in wrappers `joint_pretrends_test` and `joint_homogeneity_test` also route through that same validator internally, so direct wrapper calls inherit the last-cohort filter and constant-post-dose invariant. `HADPretestReport` extended with `pretrends_joint`, `homogeneity_joint`, and `aggregate` fields; serialization methods (`summary`, `to_dict`, `to_dataframe`, `__repr__`) preserve the Phase 3 output bit-exactly on `aggregate="overall"` — no `aggregate` key, no header row, no schema drift — and only surface the new fields on `aggregate="event_study"`.
- **`ChaisemartinDHaultfoeuille.by_path`** — per-path event-study disaggregation, mirroring R `did_multiplegt_dyn(..., by_path=k)`. Passing `by_path=k` (positive int) to the estimator reports separate `DID_{path,l}` + SE + inference for the top-k most common observed treatment paths in the window `[F_g-1, F_g-1+L_max]`, answering the practitioner question "is a single pulse enough, or do you need sustained exposure?" across paths like `(0,1,0,0)` vs `(0,1,1,0)` vs `(0,1,1,1)`. The per-path SE follows the joiners-only / leavers-only IF precedent (switcher-side contribution zeroed for non-path groups; control pool and cohort structure unchanged; plug-in SE with path-specific divisor). Requires `drop_larger_lower=False` (multi-switch groups are the object of interest) and `L_max >= 1`. Binary treatment was the only supported case at the initial cut; subsequent entries in this `[Unreleased]` block lifted that and the original gates one by one. Currently still gated: `design2` and `honest_did` raise `NotImplementedError` (deferred to follow-up PRs). All other combinations — `n_bootstrap > 0`, `placebo=True`, joint sup-t bands, `controls`, `trends_linear`, `trends_nonparam`, `survey_design`, `heterogeneity`, non-binary integer treatment, and the `paths_of_interest` user-specified selector — are now supported, with the per-feature contracts in their dedicated entries elsewhere in `[Unreleased]`. Results expose `results.path_effects: Dict[Tuple[int, ...], Dict[str, Any]]` and `results.to_dataframe(level="by_path")`; the summary grows a "Treatment-Path Disaggregation" block. Ties in path frequency are broken lexicographically on the path tuple for deterministic ranking. Overflow (`by_path > n_observed_paths`) returns all observed paths with a `UserWarning`. See `docs/methodology/REGISTRY.md` §ChaisemartinDHaultfoeuille `Note (Phase 3 by_path per-path event-study disaggregation)` for the full contract.
- **`ChaisemartinDHaultfoeuille.by_path` + `n_bootstrap > 0`** — bootstrap SE for per-path event-study effects. The top-k paths are enumerated once on the observed data (R-faithful path-stability semantics: matches `did_multiplegt_dyn(..., by_path=k, bootstrap=B)`, confirmed empirically against `DIDmultiplegtDYN 2.3.3`), and the existing multiplier bootstrap (`bootstrap_weights ∈ {"rademacher", "mammen", "webb"}`) runs per `(path, horizon)` target via the shared `_bootstrap_one_target` / `compute_effect_bootstrap_stats` helpers. Point estimates are unchanged from the analytical path. Bootstrap SE replaces the analytical SE in `path_effects[path]["horizons"][l]["se"]`, and `p_value` / `conf_int` propagate the **bootstrap percentile** statistics (library Round-10 convention, same as `overall` / `joiners` / `leavers` / `multi_horizon`); `t_stat` is SE-derived via `safe_inference` per the anti-pattern rule. Interpretation is *conditional on the observed path set* — practitioners wanting unconditional inference capturing path-selection uncertainty need a pairs-bootstrap (no R precedent). **SE inherits the analytical cross-path cohort-sharing deviation:** bootstrap input is the same full-panel cohort-centered path IF as the analytical path, so the bootstrap SE is a Monte Carlo analog of the analytical SE and inherits the existing analytical-path divergence from R on mixed-path cohorts (see REGISTRY.md for the full mechanism). On single-path-cohort panels, bootstrap and analytical SE both track R up to the Phase 2 envelope. **Deviation from R (CI method):** R's per-path bootstrap CI is normal-theory around the bootstrap SE (half-width ≈ `1.96·se`); ours is the bootstrap percentile CI, intentionally diverging from R to keep the dCDH inference surface internally consistent across all bootstrap targets. Positive regressions at `tests/test_chaisemartin_dhaultfoeuille.py::TestByPathBootstrap` (`@pytest.mark.slow`): point-estimate invariance, finite SE on non-degenerate panels, bootstrap-vs-analytical SE within 30% rtol on cohort-clean panels, degenerate-cohort NaN propagation, Rademacher/Mammen/Webb parity, seed reproducibility, and percentile-vs-normal-theory CI pinning. See `docs/methodology/REGISTRY.md` §ChaisemartinDHaultfoeuille `Note (Phase 3 by_path ...)` → **Bootstrap SE** for the full write-up.
- **R-parity for `ChaisemartinDHaultfoeuille.by_path`** against `DIDmultiplegtDYN 2.3.3`. Two new scenarios in `benchmarks/data/dcdh_dynr_golden_values.json` generated from `did_multiplegt_dyn(..., by_path=k)`: `mixed_single_switch_by_path` (2 paths, `by_path=2`) and `multi_path_reversible_by_path` (4 observed paths, `by_path=3`, via a new deterministic multi-path DGP pattern in the R generator). Per-path point estimates and per-path switcher counts match R exactly; per-path SE matches within the Phase 2 multi-horizon SE envelope (observed rtol ≤ 10.2% on the 2-path scenario, ≤ 4.2% on the 4-path scenario). Parity tests live at `tests/test_chaisemartin_dhaultfoeuille_parity.py::TestDCDHDynRParityByPath`, matching paths by tuple label via set-equality (robust to R's undocumented frequency-tie tiebreak) and cross-checking per-path switcher counts before SE comparison. **Deviation documented:** cross-path cohort sharing — our full-panel cohort-centered plug-in vs R's per-path re-run diverges materially when a `(D_{g,1}, F_g, S_g)` cohort spans multiple observed paths; the two coincide when every cohort is single-path. The parity scenarios are constructed to keep cohorts single-path (scenario 13 by design, scenario 14 via path-assignment-deterministic-on-F_g). See `docs/methodology/REGISTRY.md` §ChaisemartinDHaultfoeuille `Note (Phase 3 by_path...)` for the full write-up.
- **`profile_panel()` utility + `llms-autonomous.txt` reference guide (agent-facing)** — new `diff_diff.profile_panel(df, *, unit, time, treatment, outcome)` returns a frozen `PanelProfile` dataclass of structural facts (panel balance, treatment-type classification — `"binary_absorbing"` / `"binary_non_absorbing"` / `"continuous"` / `"categorical"`, cohort structure, outcome characteristics, and a `tuple[Alert, ...]` of factual observations). `.to_dict()` returns a JSON-serializable view. Paired with a new bundled `"autonomous"` variant on `get_llm_guide()` — `get_llm_guide("autonomous")` returns a reference-shaped guide (distinct from the existing workflow-prose `"practitioner"` variant) with §1 audience disclaimer, §2 `PanelProfile` field reference, §3 embedded 17-estimator × 9-design-feature support matrix, §4 per-design-feature reasoning citing Baker et al. (2025) and Roth / Sant'Anna (2023), §5 post-fit validation index, §6 BR/DR schema reference, §7 citations, §8 intentional omissions. Both pieces are bundled inside the wheel (no GitHub / RTD dependency at runtime); `diff_diff/__init__.py` module docstring leads with an agent-entry block listing `profile_panel`, `get_llm_guide("autonomous")`, `get_llm_guide("practitioner")`, and `BusinessReport` so `help(diff_diff)` surfaces them. Descriptive, not opinionated — `profile_panel` alerts never recommend a specific estimator, and the guide enumerates trade-offs rather than dispatching. Exports: `profile_panel`, `PanelProfile`, `Alert` from top-level `diff_diff`.
- **`target_parameter` block in BR/DR schemas (experimental; schema version bumped to 2.0)** — `BUSINESS_REPORT_SCHEMA_VERSION` and `DIAGNOSTIC_REPORT_SCHEMA_VERSION` bumped from `"1.0"` to `"2.0"` because the new `"no_scalar_by_design"` value on the `headline.status` / `headline_metric.status` enum (dCDH `trends_linear=True, L_max>=2` configuration) is a breaking change per the REPORTING.md stability policy. BusinessReport and DiagnosticReport now emit a top-level `target_parameter` block naming what the headline scalar actually represents for each of the 16 result classes. Closes BR/DR foundation gap #6 (target-parameter clarity). Fields: `name`, `definition`, `aggregation` (machine-readable dispatch tag), `headline_attribute` (raw result attribute), `reference` (citation pointer). BR's summary emits the short `name` right after the headline; DR's overall-interpretation paragraph does the same; both full reports carry a "## Target Parameter" section with the full definition. Per-estimator dispatch is sourced from REGISTRY.md and lives in the new `diff_diff/_reporting_helpers.py::describe_target_parameter`. A few branches read fit-time config (`EfficientDiDResults.pt_assumption`, `StackedDiDResults.clean_control`, `ChaisemartinDHaultfoeuilleResults.L_max` / `covariate_residuals` / `linear_trends_effects`); others emit a fixed tag (the fit-time `aggregate` kwarg on CS / Imputation / TwoStage / Wooldridge does not change the `overall_att` scalar — disambiguating horizon / group tables is tracked under gap #9). See `docs/methodology/REPORTING.md` "Target parameter" section.
- SyntheticDiD coverage Monte Carlo calibration table added to `docs/methodology/REGISTRY.md` §SyntheticDiD — rejection rates at α ∈ {0.01, 0.05, 0.10} across `placebo` / `bootstrap` / `jackknife` on 3 representative DGPs (balanced / exchangeable, unbalanced, and Arkhangelsky et al. (2021) AER §6.3 non-exchangeable). Artifact at `benchmarks/data/sdid_coverage.json` (500 seeds × B=200), regenerable via `benchmarks/python/coverage_sdid.py`.

### Fixed
- **SyntheticDiD `variance_method="bootstrap"` now runs the paper-faithful refit bootstrap** with R-default warm-start. Re-estimates ω̂_b and λ̂_b via two-pass sparsified Frank-Wolfe on each pairs-bootstrap draw using the fit-time normalized-scale zeta — Arkhangelsky et al. (2021) Algorithm 2 step 2, matching the behavior of R's default `synthdid::vcov(method="bootstrap")` (which rebinds `attr(estimate, "opts")` so the renormalized ω serves as Frank-Wolfe initialization). The Python path threads that warm-start through `compute_sdid_unit_weights(..., init_weights=_sum_normalize(ω̂[boot_control_idx]))` and `compute_time_weights(..., init_weights=λ̂)` on each bootstrap draw. `compute_sdid_unit_weights` and `compute_time_weights` gain a new `init_weights` kwarg; when provided, the Rust top-level fast-path is skipped in favor of the Python two-pass dispatcher (whose inner FW calls still dispatch to Rust). Without this kwarg both helpers remain backward-compatible and keep the Rust fast-path. The previous fixed-weight bootstrap path is removed entirely — it was not paper-faithful and, despite prior documentation claiming otherwise, also did not match R's default bootstrap (the previous R-parity test fixture invoked `synthdid_estimate(weights=...)` without rebinding `opts`, which silently runs fixed-weight, so the 1e-10 parity was between two paths both wrong in the same direction). Coverage MC at the new artifact above quantifies the correctness fix on 3 representative null DGPs. **Users' existing `variance_method="bootstrap"` fits will return materially different SE / p-value / CI values on the next release** — same enum name, corrected semantics. Bootstrap is now ~5–30× slower per fit than the old fixed-weight shortcut (panel-size dependent; warm-start converges faster than cold-start so the slowdown is less than the 10–100× prior estimate). The PR #349 follow-on bullets below (analytical p-value dispatch, sqrt((r-1)/r) SE formula, retry-to-B contract) all carry over to the refit path unchanged.
- SyntheticDiD `variance_method="bootstrap"` now computes p-values from the analytical normal-theory formula using the bootstrap SE (matching R's `synthdid::vcov()` convention), rather than an empirical null-distribution formula that is not valid for bootstrap draws. `is_significant` and `significance_stars` are derived from `p_value` and will also change for bootstrap fits. Placebo and jackknife are unchanged. Point estimates are unaffected.
- SyntheticDiD bootstrap SE formula applies the `sqrt((r-1)/r)` correction matching R's synthdid and the placebo SE formula.
- SyntheticDiD bootstrap now retries degenerate resamples (all-control or all-treated, or non-finite `τ_b`) until exactly `n_bootstrap` valid replicates are accumulated, matching R's `synthdid::bootstrap_sample` and Arkhangelsky et al. (2021) Algorithm 2. Previously the Python path counted attempts (with degenerate draws silently dropped), producing fewer valid replicates than requested. A bounded-attempt guard (`20 × n_bootstrap`) prevents pathological-input hangs.
- **TROP global bootstrap SE backend parity under fixed seed** — Rust and Python backends now produce bit-identical bootstrap SE under the same `seed`. Previously Rust's `bootstrap_trop_variance_global` seeded `rand_xoshiro::Xoshiro256PlusPlus` per replicate while Python's fallback consumed `numpy.random.default_rng` (PCG64), producing ~28% SE divergence on tiny panels under `seed=42`. Fixed by extracting a shared `stratified_bootstrap_indices` helper in `diff_diff/bootstrap_utils.py` that pre-generates per-replicate stratified sample indices via numpy on the Python side; both backends consume the same integer arrays through the PyO3 surface. Sampling law (stratified: controls then treated, with replacement) is unchanged. Closes the bootstrap-RNG half of silent-failures audit finding #23 (grid-search half closed in PR #348; local-method methodology half closed by the two Fixed entries below). Local-method TROP also adopts the Python-canonical index contract for the RNG layer here.
- **TROP local-method Rust weight-matrix no longer normalized** — `rust/src/trop.rs::compute_weight_matrix` no longer divides time-weights or unit-weights by their respective sums before the outer product. The paper's Equation 2/3 (Athey, Imbens, Qu, Viviano 2025) and REGISTRY.md Requirements checklist (line 2037: `[x] Unit weights: exp(-λ_unit × distance) (unnormalized, matching Eq. 2)`) both specify raw-exponential weights; Python's `_compute_observation_weights` was already REGISTRY-compliant. **User-visible effect**: Rust local-method ATT values may shift for any fit with `lambda_nn < infinity` — normalizing the weight-matrix inflated the effective nuclear-norm penalty relative to the data-fit term, changing the regularization trade-off. For `lambda_nn = infinity` (factor model disabled) outputs are unchanged because uniform weight scaling leaves the minimum-norm WLS argmin invariant. Rust LOOCV-selected lambdas may also shift on this boundary; both backends now converge on the same REGISTRY-compliant selection.
- **TROP local-method Python `_compute_observation_weights` now uses the function-argument `Y, D` and treats all non-target units as donors** — two coupled changes that bring Python structurally in line with Rust and the paper's Eq. 2/3:
    1. Removed the `if self._precomputed is not None:` branch that silently substituted `self._precomputed["Y"]` / `["D"]` / `["time_dist_matrix"]` (original-panel cache populated during main fit) for the function-argument `Y, D`. Under bootstrap, `_fit_with_fixed_lambda` computes fresh `Y, D` from the resampled `boot_data` and passes them in; the helper was discarding those and recomputing unit distances from the original panel, so Python's local bootstrap resampled units but reused stale unit-distance weights. Rust's bootstrap was already correct (always consumed `y_boot, d_boot`).
    2. Removed the `valid_control_at_t = D[t, :] == 0` target-period donor gate that zeroed `ω_j` for any unit `j` treated at the target period (other than the target unit itself). Per REGISTRY Eq. 2/3 and Rust's `compute_weight_matrix`, `ω_j = exp(-λ_unit × dist(j, i))` for all `j ≠ i`; treated-cell exclusion happens via the `(1 − W_{js})` factor applied inside `_estimate_model`. Same-cohort donors now contribute via their pre-treatment rows. Empirically the main-fit ATT is unchanged on tested fixtures because same-cohort pre-treatment observations are exactly absorbed by their own unit fixed effect `alpha_j` without propagating into `mu`, `beta`, or other units' parameters — so this change is structural alignment rather than a numerical shift in output. Users on same-cohort panels with very few controls may still see tiny differences in edge cases; the new `test_local_method_same_cohort_donor_parity` regression guards the aligned behavior.
  Together with the normalization fix above, TROP local-method backend parity on the main-fit ATT is regime-dependent: `atol=rtol=1e-14` for `lambda_nn=inf` (no nuclear-norm regularization, uniform weight scaling leaves the WLS argmin invariant) and `atol=1e-10` for finite `lambda_nn` (FISTA inner loop + BLAS reduction ordering introduce sub-1e-10 roundoff across Rust `faer` vs numpy paths). Bootstrap SE parity is asserted at `atol=1e-5` to accommodate ~1e-7 roundoff between Rust's `estimate_model` matrix factorization and numpy's `lstsq` that accumulates across per-replicate fits; sub-1e-14 bootstrap parity is tracked as a follow-up in `TODO.md` under "unify Rust local-method solver path". Closes silent-failures audit finding #23 (local-method half; the RNG half closed in PR #354 and the grid-search half in PR #348).

### Changed
- **`did_had_pretest_workflow(aggregate="event_study")` verdict no longer emits the "paper step 2 deferred to Phase 3 follow-up" caveat** — the joint pre-trends Stute test closes that gap. The two-period `aggregate="overall"` path retains the existing caveat since the joint variant does not apply to single-pre-period panels. Downstream code that greps verdict strings for the Phase 3 caveat will see it suppressed on the event-study path.
- **SyntheticDiD bootstrap no longer supports survey designs** (capability regression in PR #351, **restored in PR #355** — see Added/Changed entries directly below). The removed fixed-weight bootstrap path was the only SDID variance method that supported strata/PSU/FPC (via Rao-Wu rescaled bootstrap); the PR #351 paper-faithful refit bootstrap initially rejected all survey designs (including pweight-only) with `NotImplementedError`. PR #355 restores the capability via a weighted-FW + Rao-Wu composition; the lock-out window applies only to the v3.2.x line that ships PR #351 alone (without PR #355). Composing Rao-Wu rescaled weights with Frank-Wolfe re-estimation: see `docs/methodology/REGISTRY.md` §SyntheticDiD `Note (survey + bootstrap composition)`.

### Added (PR #355)
- **SDID `variance_method="bootstrap"` survey support restored** via a hybrid pairs-bootstrap + Rao-Wu rescaling composed with a weighted Frank-Wolfe kernel. Each bootstrap draw first performs the unit-level pairs-bootstrap resampling specified by Arkhangelsky et al. (2021) Algorithm 2 (`boot_idx = rng.choice(n_total)`), and *then* applies Rao-Wu rescaled per-unit weights (Rao & Wu 1988) sliced over the resampled units — NOT a standalone Rao-Wu bootstrap. New Rust kernel `sc_weight_fw_weighted` (and `_with_convergence` sibling) accepts a per-coordinate `reg_weights` argument so the FW objective becomes `min ||A·ω - b||² + ζ²·Σ_j reg_w[j]·ω[j]²`. New Python helpers `compute_sdid_unit_weights_survey` and `compute_time_weights_survey` thread per-control survey weights through the two-pass sparsify-refit dispatcher (column-scaling Y by `rw` for the loss, `reg_weights=rw` for the penalty on the unit-weights side; weighted column-centering + row-scaling Y by `sqrt(rw)` for the loss with uniform reg on the time-weights side). `_bootstrap_se` survey branch composes the per-draw `rw` (Rao-Wu rescaling for full designs, constant `w_control` for pweight-only fits) with the weighted-FW helpers, then composes `ω_eff = rw·ω/Σ(rw·ω)` for the SDID estimator. Coverage MC artifact extended with a `stratified_survey` DGP (BRFSS-style: N=40, strata=2, PSU=2/stratum); the bootstrap row's near-nominal calibration is the validation gate (target rejection ∈ [0.02, 0.10] at α=0.05). New regression tests across `test_methodology_sdid.py::TestBootstrapSE` (single-PSU short-circuit, full-design and pweight-only succeeds-tests, zero-treated-mass retry, deterministic Rao-Wu × boot_idx slice) and `test_survey_phase5.py::TestSyntheticDiDSurvey` (full-design ↔ pweight-only SE differs assertion). See REGISTRY.md §SyntheticDiD ``Note (survey + bootstrap composition)`` for the full objective and the argmin-set caveat.

### Changed (PR #355)
- **SDID bootstrap SE values under survey fits now differ numerically from the v3.2.x line that shipped PR #351 alone**: the fit no longer raises `NotImplementedError`, and instead returns the weighted-FW + Rao-Wu SE. Non-survey fits are unaffected (the bootstrap dispatcher routes only the survey branch through the new `_survey` helpers; non-survey fits continue to call the existing `compute_sdid_unit_weights` / `compute_time_weights` and stay bit-identical at rel=1e-14 on the `_BASELINE["bootstrap"]` regression). SDID's `placebo` and `jackknife` paths still reject `strata/PSU/FPC` on the v3.2.x line; full-design support for those methods lands separately in the entries below.

### Added
- **SDID `variance_method="placebo"` and `"jackknife"` now support strata/PSU/FPC designs.** Closes the last SDID survey gap. All three variance methods (bootstrap from PR #355, plus placebo and jackknife here) now handle full survey designs. New private methods `SyntheticDiD._placebo_variance_se_survey` and `_jackknife_se_survey` route the full-design path through method-specific allocators:
  - **Placebo** — stratified permutation (Pesarin 2001). Each draw samples pseudo-treated indices uniformly without replacement from controls *within each stratum* containing actual treated units; non-treated strata contribute their controls unconditionally. The weighted Frank-Wolfe kernel from PR #355 (`compute_sdid_unit_weights_survey` / `compute_time_weights_survey`) re-estimates ω and λ per draw with per-control survey weights threaded into both loss and regularization; post-optimization composition `ω_eff = rw·ω/Σ(rw·ω)`. Arkhangelsky Algorithm 4 SE formula unchanged.
  - **Jackknife** — PSU-level leave-one-out with stratum aggregation (Rust & Rao 1996). `SE² = Σ_h (1-f_h)·(n_h-1)/n_h·Σ_{j∈h}(τ̂_{(h,j)} - τ̄_h)²` with `f_h = n_h_sampled / fpc[h]` (population-count FPC form). λ held fixed across LOOs; ω subsetted, composed with rw, renormalized. Strata with `n_h < 2` silently skipped (matches R `survey::svyjkn` with `lonely_psu="remove"` / `"certainty"`; `"adjust"` raises `NotImplementedError`). Full-census strata (`f_h ≥ 1`) short-circuit to zero contribution before any LOO feasibility check. `SE = 0` is returned for legitimate zero variance (e.g., every stratum full-census); `SE = NaN` with a targeted `UserWarning` is reserved for undefined cases — all strata skipped, or any delete-one replicate in a non-full-census contributing stratum is undefined (all-treated-in-one-PSU LOO, kept ω_eff / w_treated mass zero, estimator raises). Unstratified single-PSU short-circuits to NaN.
  - **Fit-time feasibility guards** (placebo): `ValueError` on stratum-level infeasibility with targeted messages distinguishing three cases — **Case B** (treated-containing stratum has zero controls), **Case C** (fewer controls than treated in a treated stratum), **Case D** (every treated stratum is exact-count `n_c_h == n_t_h` → permutation support is 1, null distribution collapses). Partial-permutation fallback rejected because it would silently change the null-distribution semantics.
  - **Gate relaxed**: the fit-time guard at `synthetic_did.py:352-369` that rejected placebo/jackknife + strata/PSU/FPC is removed. Replicate-weight designs remain rejected (separate methodology — replicate variance is closed-form and would double-count with Rao-Wu-like rescaling). Non-survey and pweight-only paths bit-identical by construction — the new code is gated on `resolved_survey_unit.(strata|psu|fpc) is not None`.
  - **Coverage MC**: `benchmarks/data/sdid_coverage.json` extended with jackknife on `stratified_survey`. Bootstrap validates near-nominal (α=0.05 rejection = 0.058, SE/trueSD = 1.13). Jackknife reported with an anti-conservatism caveat: with only 2 PSUs per stratum the stratified jackknife formula has 1 effective DoF per stratum, a well-documented limitation of Rust & Rao (1996) — `se_over_truesd ≈ 0.46` on this DGP. Users needing tight SE calibration with few PSUs should prefer `variance_method="bootstrap"`. Placebo is structurally infeasible on the existing `stratified_survey` DGP (its cohort packs into one stratum with 0 never-treated units — by design a bootstrap-suited DGP); the placebo survey path is exercised via unit tests on a feasible fixture.
  - **Regression tests** across `tests/test_survey_phase5.py`: two new classes `TestSDIDSurveyPlaceboFullDesign` and `TestSDIDSurveyJackknifeFullDesign`. Placebo: pseudo-treated-stratum contract, Case B / Case C front-door guards with targeted-message regression, SE-differs-from-pweight-only, deterministic dispatch. Jackknife: stratum-aggregation self-consistency, **FPC magnitude regression** (2-stratum handcrafted panel asserts `SE_fpc == SE_nofpc · sqrt(1-f)` at `rtol=1e-10`), single-PSU-stratum skip, unstratified short-circuit, all-strata-skipped warning + NaN, SE-differs-from-pweight-only, deterministic dispatch. Existing `test_full_design_placebo_raises` and `test_full_design_jackknife_raises` flipped to `_succeeds` assertions. All 19 existing pweight-only and non-survey placebo/jackknife tests pass unchanged (bit-identity preserved via the new-path gating).
  - **Allocator asymmetry** (documented in REGISTRY): placebo ignores the PSU axis (unit-level within-stratum permutation — the classical stratified permutation test; PSU-level permutation on few PSUs is near-degenerate); jackknife respects PSU (PSU-level LOO is the canonical survey jackknife). Both respect strata. See `docs/methodology/REGISTRY.md` §SyntheticDiD `Note (survey + placebo composition)` and `Note (survey + jackknife composition)`.

## [3.2.0] - 2026-04-19

### Added
- **`BusinessReport` and `DiagnosticReport` (experimental preview)** (PR #318) - practitioner-ready output layer. `BusinessReport(results, ...)` produces plain-English narrative summaries (`.summary()`, `.full_report()`, `.export_markdown()`, `.to_dict()`) from any of the 16 fitted result types. `DiagnosticReport(results, ...)` orchestrates the existing diagnostic battery (parallel trends, pre-trends power, HonestDiD sensitivity, Goodman-Bacon, heterogeneity, design-effect, EPV) plus estimator-native diagnostics for SyntheticDiD (`pre_treatment_fit`, weight concentration, in-time placebo, zeta sensitivity) and TROP (factor-model fit metrics). Both classes expose an AI-legible `to_dict()` schema (single source of truth; prose renders from the dict). BR auto-constructs DR by default so summaries mention pre-trends, robustness, and design-effect findings in one call. See `docs/methodology/REPORTING.md` for methodology deviations including the no-traffic-light-gates decision, pre-trends verdict thresholds (0.05 / 0.30), and power-aware phrasing driven by `compute_pretrends_power`. **Both schemas are marked experimental in this release** - wording, verdict thresholds, and schema shape will change; do not anchor downstream tooling on them yet.
- **Kernel / local-linear / nonparametric infrastructure** (PRs #327, #335) - bandwidth selector, local linear regression, HC2 / Bell-McCaffrey variance helpers, and a port of R `nprobust`'s point-estimate path. Foundation for the upcoming `HeterogeneousAdoptionDiD` estimator (de Chaisemartin, Ciccia, D'Haultfœuille & Knau 2024 — "DiD with no untreated group"). Released as internal modules with full test coverage (`tests/test_bandwidth_selector.py`, `tests/test_local_linear.py`, `tests/test_linalg_hc2_bm.py`, `tests/test_nprobust_port.py`); the user-facing estimator ships in a later phase.
- **Cell-period IF allocator for dCDH survey variance (Class A contract)** (PR #323) - replaces the group-level allocator `ψ_i = ψ_g * (w_i / W_g)` with a cell-period allocator `ψ_i = ψ_g * (w_i / W_{g, out_idx})` on the post-period cell for the DID_l replicate-weight ATT path. Is the allocator shape that the v3.2.0 heterogeneity and bootstrap extensions below build on. Documents the post-period attribution convention in REGISTRY.md with a hand-computed row-sum identity test.

### Performance
- **`aggregate_survey` stratum-PSU scaffolding precompute** — the per-cell Taylor-series variance inside `aggregate_survey` no longer rebuilds stratum-PSU scaffolding on every cell. A frozen `_PsuScaffolding` (strata codes, global PSU codes unique across strata, per-stratum counts and FPC ratios, singleton mask, static legitimate-zero counts and variance-computable flag) is precomputed once per design at the top of `aggregate_survey` and threaded through `_cell_mean_variance` to a new `_compute_if_variance_fast` path that replaces the per-stratum pandas groupby with two vectorized `np.bincount` passes. BRFSS-shaped 50-state × 10-year × 1M-row microdata → state-year panel drops from ~24s to sub-2s under both backends (the path is pure Python, so Python and Rust track each other). Numerical output is preserved to sub-ULP tolerance; seven-case equivalence tests (`TestAggregateSurveyScaffolding`) assert `assert_allclose(atol=1e-14, rtol=1e-14)` between fast and legacy paths across stratified+PSU+FPC, stratified no FPC, PSU-only, weights-only, and all three `lonely_psu` modes (remove / certainty / adjust). Replicate-weight designs continue to route through `compute_replicate_if_variance` unchanged. `_compute_stratified_psu_meat` is untouched — all other TSL callers (DiD / TWFE / CS / etc.) are unaffected.

### Changed
- Add Zenodo DOI badge to README; upgrade the BibTeX citation block with the concept DOI (`10.5281/zenodo.19646175`) and list author as Isaac Gerber (matching `CITATION.cff`). `CITATION.cff` carries the concept DOI as its top-level `doi:` field — Zenodo auto-mints a versioned DOI for every release, but the CFF file tracks the concept DOI only so it doesn't need a follow-up edit per release. DOI was minted by Zenodo when v3.1.3 was released.
- **`ChaisemartinDHaultfoeuille` heterogeneity + within-group-varying PSU/strata now supported under Binder TSL** - `fit(heterogeneity=..., survey_design=...)` no longer raises `NotImplementedError` when the resolved design's PSU or strata vary across the cells of a group. On the **Binder TSL** branch (`compute_survey_if_variance`), the heterogeneity WLS coefficient IF is expanded to observation level via the cell-period allocator `ψ_i = ψ_g * (w_i / W_{g, out_idx})` on the post-period cell — the DID_l post-period single-cell convention shipped in v3.1.x. Under PSU=group the PSU-level Binder TSL variance is byte-identical to the previous release (PSU-level aggregate telescopes to `ψ_g`); under within-group-varying PSU, mass lands in the post-period PSU of the transition. The **Rao-Wu replicate-weight** branch (`compute_replicate_if_variance`) retains the legacy group-level allocator `ψ_i = ψ_g * (w_i / W_g)`: replicate variance computes `θ_r = sum_i ratio_ir * ψ_i` at observation level and is therefore not PSU-telescoping, so the cell-period allocator would silently change the replicate SE whenever a replicate column's ratios vary within group (e.g., per-row replicate matrices). Replicate + heterogeneity fits therefore produce byte-identical SE to the previous release, and the newly-unblocked `heterogeneity=` + within-group-varying PSU combination is unreachable under replicate designs by construction (`SurveyDesign` rejects `replicate_weights` combined with explicit `strata/psu/fpc`).
- **`ChaisemartinDHaultfoeuille.fit(survey_design=..., n_bootstrap > 0)` now supports within-group-varying PSU** — the PSU-level Hall-Mammen wild multiplier bootstrap has been extended from a group-level PSU map (one multiplier per group) to a cell-level PSU map (one multiplier per `(g, t)` cell's PSU). A dispatcher in `_compute_dcdh_bootstrap` detects PSU-within-group-constant regimes (including PSU=group auto-inject and strictly-coarser PSU with within-group constancy) and routes them through the legacy group-level path so the bootstrap SE is bit-identical to the previous release (guarded by the new `test_bootstrap_se_matches_pre_pr4_baseline` and the pre-existing `test_auto_inject_bit_identical_to_group_level`). Under within-group-varying PSU, a group contributing cells to multiple PSUs receives independent multiplier draws per PSU — the correct Hall-Mammen wild PSU clustering at cell granularity. Multi-horizon bootstraps draw a single shared `(n_bootstrap, n_psu)` PSU-level weight matrix per block and broadcast per-horizon via each horizon's cell-to-PSU map, so the sup-t simultaneous confidence band remains a valid joint distribution. Closes the last `NotImplementedError` gate in the dCDH survey contract; replicate-weight variance and `n_bootstrap > 0` remain mutually exclusive by construction. **Scope note:** panels with *terminal missingness* where the terminally-missing group is in a cohort whose other groups still contribute at the missing period now raise a targeted `ValueError` on every survey variance path that uses the cell-period allocator: Binder TSL with within-group-varying PSU, Rao-Wu replicate-weight ATT (which always uses the cell allocator per the Class A contract shipped in PR #323), and the cell-level wild PSU bootstrap. Cohort-recentering leaks centered IF mass onto cells with no positive-weight observations, which the cell-period allocator cannot attach to any observation/PSU. This closes a silent mass-drop bug the cell-period allocator introduced across all three paths in v3.1.x; pre-process the panel to remove terminal missingness (drop late-exit groups or trim to a balanced sub-panel) as the documented workaround. For Binder TSL only, using an explicit `psu=<group_col>` routes through the legacy group-level allocator where the row-sum identity makes the two allocators statistically equivalent. Replicate-weight ATT and within-group-varying-PSU bootstrap have no such allocator fallback — the panel itself must be pre-processed. PSU-within-group-constant Binder TSL (including PSU=group auto-inject) is unaffected.
- **Performance review: practitioner-scale scenarios + benchmark harness extension** (PR #333) - new `docs/performance-scenarios.md` documents 5-7 realistic practitioner workflows (marketing lift, geo-experiment, BRFSS state-policy, dCDH reversible treatment) grounded in the practitioner docs and the paper literature, not cookie-cutter textbook data. `benchmarks/speed_review/` extended with practitioner-scale scripts and per-backend bit-identity baselines. Baselines refreshed against current main. Finding: the biggest leverage areas are bootstrap resampling loops and per-replicate survey-design rebuilds in the bootstrap path; documented in `docs/performance-plan.md` for follow-up optimization PRs.
- **Wall-clock timing tests excluded from default CI** (PRs #330, #336) - `TestCallawaySantAnnaSEAccuracy.test_timing_performance` and `TestPerformanceRegression` marked `@pytest.mark.slow`, removing false-positive CI failures from runner-noise variance (BLAS path variation, neighbor VM contention). Tests remain runnable via `pytest -m slow` for ad-hoc local benchmarking; the perf-review harness above is the principled replacement for CI-gated performance tracking.

### Fixed
- **Silent-failures audit: axis A** (PR #334) — minor solver paths numerical-precision / scale-fragility closeouts, completing the SDID extreme-Y-scale work started in v3.1.2.
- **Silent-failures audit: axis C & J** (PR #339) — B-spline derivative warning scope broadened; `SurveyPowerConfig` stale-cache wording narrowed.
- **Silent-failures audit: axis E** (PR #331) — row-drop counters surfaced across estimator paths so silent validator row-drops leave an explicit count on the result.
- **Silent-failures audit: axis G** (PR #337) — Rust vs Python backend edge-case parity tests added for rank-deficient, extreme-scale, and constant-column inputs.
- **SyntheticDiD diagnostic Y-normalization parity** (PR #328) — extends the PR #312 catastrophic-cancellation fix from the main fit path into `SyntheticDiDResults.in_time_placebo()` and `.sensitivity_to_zeta_omega()`. Diagnostics now apply the same `Y_shift / Y_scale` normalization the main fit uses, pass `zeta / Y_scale` and a normalized `min_decrease` into Frank-Wolfe, then rescale `att` / `pre_fit_rmse` back to original-Y units.
- **TROP bootstrap failure-rate guards** (PR #324) — alternating-minimization bootstrap loops now emit a `UserWarning` on silent high-failure-rate runs (LOOCV and bootstrap aggregation paths both covered); attempt-count-based warning replaces the previous observation-count denominator that could silently mask sparse runs.
- **`simulate_power()` failure-count surface + narrow except clause** (PR #326) — power-simulation replicate loop narrows the exception whitelist from `except Exception` to estimation/data-path failures (`TypeError` and friends now propagate, not silently absorb), and surfaces `n_simulation_failures` on `SimulationPowerResults`. Failure count included in `summary()` and `to_dict()`.

## [3.1.3] - 2026-04-18

### Added
- **Replicate-weight variance and PSU-level bootstrap for dCDH** (PR #311) - `ChaisemartinDHaultfoeuille` now accepts `variance_method="replicate"` for BRR / Fay / JK1 / JKn / SDR inference, and PSU-level multiplier bootstrap when `survey_design.psu` is set. Adds df-aware inference (reduced effective df under replicate variance; propagated through delta / HonestDiD surfaces) plus group-level PSU map construction. Validated via per-cohort aggregation, shared-draw multi-horizon bootstrap alignment, and cross-surface df consistency.
- **Zenodo DOI auto-minting configuration** (PR #321) - `.zenodo.json` at repo root defines release metadata so the next GitHub Release automatically mints a Zenodo DOI (concept DOI + versioned DOI). Also adds a top-level `LICENSE` file for Zenodo archival.

### Fixed
- **Silent sparse→dense lstsq fallback in `ImputationDiD` and `TwoStageDiD`** (PR #319) - when the sparse solver fails and the dense fallback runs, the estimator now emits a `UserWarning` instead of silently switching paths. Regression tests assert the dense fallback SEs remain usable.
- **Non-convergence signaling in TROP alternating-minimization solvers** (PR #317) - the global- and local-TROP solvers now emit a `UserWarning` when the alternating-minimization loop exits without meeting tolerance, including LOOCV and bootstrap aggregation paths. Warnings aggregate at top-level call sites to avoid log spam.

### Changed
- **`/bump-version` skill updates `CITATION.cff`** (PR #320) - internal release-management tooling now keeps `CITATION.cff` `version:` and `date-released:` in sync with the other version surfaces. Resolves a single `RELEASE_DATE` upfront (from the CHANGELOG header if pre-populated, else today's date) and threads it through all date-bearing files — fixes drift that caused v3.1.2 to ship with `CITATION.cff` still pinned at 3.1.1.

## [3.1.2] - 2026-04-18

### Fixed
- **SyntheticDiD catastrophic cancellation at extreme Y scale** (PR #312) - the Frank-Wolfe weight solver lost precision when outcome magnitudes were very large or very small; results are now numerically stable across scales.
- **Non-convergence signaling in FE imputation alternating-projection solvers** (PR #314) - `ImputationDiD`, `TwoStageDiD`, and shared `within_transform` now emit a `UserWarning` when the alternating-projection / weighted-demean loop exits without meeting the tolerance. `max_iter` and `tol` are documented on `within_transform`.
- **Non-convergence signaling in SyntheticDiD Frank-Wolfe solver** (PR #315) - the numpy-path Frank-Wolfe SC weight solver now emits a `UserWarning` when the loop exits without meeting `min_decrease`. Wrapper-level and `max_iter=0` regression tests added.

### Changed
- Refresh `ROADMAP.md` to drop top-level phase numbering and reflect shipped state through v3.1.1 (PR #313). Absorbs dCDH into the Current State estimator list; adds Recently Shipped summary; reorganizes open work as Shipping Next / Under Consideration / AI-Agent Track / Long-term. Updates `docs/business-strategy.md`, `docs/survey-roadmap.md`, `docs/practitioner_decision_tree.rst`, `docs/choosing_estimator.rst`, `docs/api/chaisemartin_dhaultfoeuille.rst`, `README.md`, and `diff_diff/guides/llms-full.txt` to remove stale phase-deferral language now that the deferred items have shipped.
- Bump the `SyntheticDiD(lambda_reg=...)` and `SyntheticDiD(zeta=...)` deprecation warnings' removal target from `v3.1` to `v4.0.0`. Removing public kwargs in a patch / minor release would violate Semantic Versioning; the deprecation stays warning-only throughout the `3.x` line and will be removed in the next major release. Use `zeta_omega` / `zeta_lambda` instead.

## [3.1.1] - 2026-04-16

### Added
- **Jackknife variance estimation for SyntheticDiD** - `variance_method='jackknife'` implements the delete-one-unit jackknife from Arkhangelsky et al. (2021) Section 5. Supports both standard and survey-weighted jackknife with automatic `pweight` propagation. Validated against R `synthdid` package.
- **LinkedIn carousel** for dCDH estimator announcement (`carousel/diff-diff-dcdh-carousel.pdf`)

## [3.1.0] - 2026-04-14

### Added
- **dCDH Phase 3: Complete feature set for `ChaisemartinDHaultfoeuille`** - three sub-releases completing the estimator:
  - **Phase 3a** (PR #300): Placebo SE via multiplier bootstrap (resolves Phase 1 deferral), non-binary treatment support with crossing-cell detection and automatic cell dropping, R parity SE assertions tightened
  - **Phase 3b** (PR #302): Covariate adjustment via `controls` parameter (OLS residualization, Design 2 per-period path for non-binary treatment), group-specific linear trends via `trends_linear=True` (absorbs group-specific slopes before DiD), R `DIDmultiplegtDYN` parity tests for covariates and trends
  - **Phase 3c** (PR #303): HonestDiD sensitivity analysis integration - `honest_did()` method on results with automatic event-study-to-sensitivity bridge, support trimming for non-consecutive horizons, `l_vec` target specification, Delta-RM and Delta-SD smoothness bounds

### Changed
- ROADMAP.md updated: dCDH Phase 3 items marked shipped

## [3.0.2] - 2026-04-12

### Added
- **`ChaisemartinDHaultfoeuille`** (alias `DCDH`) - de Chaisemartin & D'Haultfœuille estimator for **non-absorbing (reversible) treatments**. The only modern staggered DiD estimator that handles treatment switching on AND off. Implements `DID_M` from AER 2020, validated against R `DIDmultiplegtDYN` v2.3.3. Ships Phases 1 and 2:
  - Phase 1: headline `DID_M` with analytical SE, joiners/leavers decompositions, single-lag placebo, multiplier bootstrap, TWFE decomposition diagnostic
  - Phase 2: multi-horizon event study (`L_max`), dynamic placebos, normalized estimator, cost-benefit aggregate (Lemma 4), sup-t simultaneous confidence bands, `plot_event_study()` integration
- **`twowayfeweights()`** - standalone TWFE decomposition diagnostic (Theorem 1, AER 2020)
- **`generate_reversible_did_data()`** - reversible-treatment panel data generator with 7 switch patterns
- **Survey-aware power analysis** - analytical helpers (`compute_power()`, `compute_mde()`, `compute_sample_size()`) accept a `deff` parameter for design-effect adjustment. Simulation helpers (`simulate_power`, `simulate_mde`, `simulate_sample_size`) accept a `survey_config` (`SurveyPowerConfig`) that generates data with complex survey structure and injects a `SurveyDesign` into each simulated fit.
- **`aggregate_survey()` `second_stage_weights` parameter** - choose `"pweight"` (default, population weights) or `"aweight"` (precision weights). pweight output is compatible with all survey-capable estimators; aweight is opt-in for GLS efficiency with estimators marked Full in the survey support matrix.
- **`conditional_pt` parameter** on `generate_survey_did_data()` - simulates scenarios where unconditional parallel trends fail but conditional PT holds after covariate adjustment
- **Tutorial 18: Geo-Experiment Analysis** (`18_geo_experiments.ipynb`) - SyntheticDiD walkthrough for marketing analytics: simulated DMA panel, 5 treated markets, fit + diagnostics + stakeholder summary
- **Practitioner decision tree** (`docs/practitioner_decision_tree.rst`) - "which method fits my business problem?" guide
- **Practitioner getting started guide** (`docs/practitioner_getting_started.rst`) - end-to-end walkthrough with terminology bridge
- **JOSS paper** (`paper.md`, `paper.bib`) - software paper for Journal of Open Source Software submission
- **CONTRIBUTORS.md** - author and contributor credit
- **Standalone CI Gate workflow** (`.github/workflows/ci-gate.yml`) - doc-only PRs no longer block on path-filtered test workflows

### Changed
- `aggregate_survey()` default second-stage weights changed from `aweight` (precision) to `pweight` (population). Users who need the old precision-weighting behavior can pass `second_stage_weights="aweight"`.
- README "For Data Scientists" section with practitioner-facing links and `aggregate_survey()` documentation
- CITATION.cff updated with version and release date
- ROADMAP.md updated: B1a-d marked done, B2b marked done, B3d marked shipped, dCDH entry updated with correct citations

### Fixed
- Doc-only PRs no longer block indefinitely on CI Gate (standalone gate workflow runs on all PRs regardless of path filters)
- `aggregate_survey()` docs no longer overclaim universal estimator compatibility - explicitly document aweight/pweight restrictions per the survey support matrix

## [3.0.1] - 2026-04-07

### Added
- **`aggregate_survey()`** — new function in `diff_diff.prep` that bridges individual-level survey microdata to geographic-period panels for DiD estimation. Computes design-based cell means and precision weights using domain estimation (Lumley 2004), with SRS fallback for small cells. Returns a panel DataFrame and pre-configured `SurveyDesign` for second-stage estimation. Supports both TSL and replicate-weight variance.
- **Python 3.14 support** — upgraded PyO3 from 0.22 to 0.28, updated CI and publish workflow matrices, bumped Rust MSRV to 1.84 for faer 0.24 compatibility.

### Changed
- Updated README Python support matrix to include 3.14

### Fixed
- Fix domain estimation zero-padding for correct design-based cell variance
- Fix SRS fallback weight normalization for scale invariance across replicate designs
- Validate numeric dtype for outcomes/covariates before aggregation (nullable dtype support)
- Validate grouping columns for NaN values

## [3.0.0] - 2026-04-07

v3.0 completes the survey support roadmap: all 16 estimators (15 inference-level +
BaconDecomposition diagnostic) now accept `survey_design`. See v2.8.0–v2.9.1 entries
for the full feature history leading to this release.

### Breaking Changes
- **Remove `bootstrap_weight_type` parameter** from CallawaySantAnna — use `bootstrap_weights` instead (deprecated since v1.0.1)
- **Remove TROP `method="twostep"` alias** — use `method="local"` (deprecated since v2.7.2)
- **Remove TROP `method="joint"` alias** — use `method="global"` (deprecated since v2.7.2)

### Upgrading from v2.x
- `CallawaySantAnna(bootstrap_weight_type="mammen")` → `CallawaySantAnna(bootstrap_weights="mammen")`
- `TROP(method="twostep")` → `TROP(method="local")`
- `TROP(method="joint")` → `TROP(method="global")`

### Deprecated
- SyntheticDiD `lambda_reg` and `zeta` parameters formally scheduled for removal in v3.1 — use `zeta_omega`/`zeta_lambda` instead

### Changed
- Internal attribute `bootstrap_weight_type` renamed to `bootstrap_weights` in bootstrap mixin and StaggeredTripleDifference for consistency
- TROP `set_params()` now validates `method` against `("local", "global")` — previously only validated in `__init__`
- Documentation updated: all survey gap notes for WooldridgeDiD removed, ROADMAP Phase 10 items marked shipped

## [2.9.1] - 2026-04-06

### Added
- **Survey theory document** (`docs/methodology/survey-theory.md`) — formal justification for design-based variance estimation with modern DiD influence functions, citing Binder (1983), Rao & Wu (1988), Shao (1996)
- **Research-grade survey DGP** — 8 new parameters on `generate_survey_did_data()`: `icc`, `weight_cv`, `informative_sampling`, `heterogeneous_te_by_strata`, `te_covariate_interaction`, `covariate_effects`, `strata_sizes`, `return_true_population_att`. All backward-compatible.
- **R validation expansion** — 4 additional estimators cross-validated against R's `survey::svyglm()`: ImputationDiD, StackedDiD, SunAbraham, TripleDifference. Survey R validation coverage now 8 of 16 estimators.
- **LinkedIn carousel** for Wooldridge ETWFE estimator announcement

### Changed
- Survey tutorial rewritten: leads with "Why Survey Design Matters" section showing flat-weight vs design-based comparison with known ground truth, coverage simulation, and false pre-trend detection rates
- Documentation refresh: ROADMAP.md, llms.txt, llms-full.txt, llms-practitioner.txt, choosing_estimator.rst updated for v2.9.0 — added WooldridgeDiD and StaggeredTripleDifference, DDD flowchart branch, standardized estimator counts, qualified survey claims
- Survey roadmap updated: Phase 10a-10d marked shipped, conditional PT noted for 10e

### Fixed
- Fix stale "EfficientDiD covariates + survey not supported" note in choosing_estimator.rst
- Fix WooldridgeDiD described as "ASF-based" for OLS path (OLS uses direct coefficients; ASF only for logit/Poisson)
- Fix dead StaggeredTripleDifference API link in llms.txt
- Fix survey example attribute: `.design_effect` not `.deff` in llms-full.txt
- Fix `subpopulation()` example to show tuple unpacking in llms-full.txt
- Remove 8 resolved items from TODO.md

## [2.9.0] - 2026-04-04

### Added
- **WooldridgeDiD (ETWFE)** estimator — Extended Two-Way Fixed Effects from Wooldridge (2025, 2023). Supports OLS, logit, and Poisson QMLE paths with ASF-based ATT and delta-method SEs. Four aggregation types (simple, group, calendar, event) matching Stata `jwdid_estat`. Alias: `ETWFE`. (PR #216, thanks @wenddymacro)
- **EfficientDiD survey + covariates** — doubly robust covariate path now threads survey weights through all four nuisance estimation stages (outcome regression, propensity ratio sieve, inverse propensity sieve, kernel-smoothed conditional Omega*). Previously raised `NotImplementedError`.
- **Survey real-data validation** (Phase 9) — 15 cross-validation tests against R's `survey` package using three real federal survey datasets:
  - **API** (R `survey` package): TSL variance with strata, FPC, subpopulations, covariates, and Fay's BRR replicates
  - **NHANES** (CDC/NCHS): TSL variance with strata + PSU + nest=TRUE, validating the ACA young adult coverage provision DiD
  - **RECS 2020** (U.S. EIA): JK1 replicate weight variance with 60 pre-computed replicate columns
  - ATT, SE, df, and CI match R to machine precision (< 1e-10) where directly comparable; known deviations documented in REGISTRY.md (TWFE SE differs due to unit FE absorption; subpopulation df differs due to strata preservation)
- **Label-gated CI** — test workflows now require `ready-for-ci` label before running, reducing wasted CI during AI review rounds. AI review workflow always runs.
- **Documentation dependency map** (`docs/doc-deps.yaml`) — maps source files to impacted documentation. New `/docs-impact` skill flags which docs need updating when source files change.

### Changed
- WooldridgeDiD: full interacted covariate basis (D_g × X, f_t × X) for OLS path
- `/submit-pr`, `/push-pr-update`, `/pre-merge-check`, `/docs-check` skills updated for label-gated CI and doc-deps workflow

### Fixed
- Fix WooldridgeDiD OLS unbalanced demeaning and nonlinear never-treated identification
- Fix WooldridgeDiD Poisson dropped-cell bug and anticipation propagation
- Fix EfficientDiD IF-scale mismatch in survey aggregation and zero-weight never-treated guard
- Fix bootstrap clustering and delta-method reduced space in WooldridgeDiD

## [2.8.4] - 2026-04-04

### Added
- **SDR replicate method** (Phase 8a) — Successive Difference Replication for ACS PUMS users. `SurveyDesign(replicate_method="SDR")` with variance formula `V = 4/R * sum((theta_r - theta)^2)`.
- **FPC support for ImputationDiD and TwoStageDiD** (Phase 8b) — finite population correction now threaded through TSL variance for both estimators.
- **Lonely PSU "adjust" in bootstrap** (Phase 8d) — `lonely_psu="adjust"` now works with survey-aware bootstrap (previously raised `NotImplementedError`). Uses Rust & Rao (1996) grand-mean centering.
- **CV on estimates** (Phase 8e) — `coef_var` property on all results objects (SE/estimate). Handles edge cases (SE=0, estimate=0).
- **Weight trimming utility** (Phase 8e) — `trim_weights(data, weight_col, upper=None, lower=None, quantile=None)` in `prep.py` for capping extreme survey weights.
- **ImputationDiD pretrends + survey** (Phase 8e) — pre-trends F-test now survey-aware using subpopulation approach for correct variance under complex designs.
- Updated ImputationDiD tutorial to demonstrate `pretrends=True` event study
- Updated survey tutorial: narrative improvements, chart rendering fixes

### Fixed
- Fix survey pretrend F-test df calculation and rank-deficient survey VCV handling
- Fix `trim_weights` NaN poisoning when weight column contains missing values
- Fix single-singleton PSU warning for lonely_psu="adjust"

## [2.8.3] - 2026-04-02

### Added
- **Silent operation warnings** — 8 operations that previously altered analysis results without informing the user now emit `UserWarning`:
  - TROP lstsq → pseudo-inverse numerical fallback
  - TwoStageDiD NaN masking of unidentified fixed effects (zeroed out with treatment indicator)
  - TwoStageDiD always-treated unit removal (sample size change)
  - CallawaySantAnna silent (g,t) pair skipping (zero treated or control observations)
  - TROP missing treatment indicator fill with 0 (control)
  - Rust → Python backend fallback (previously debug log only)
  - Survey weight normalization (pweights/aweights rescaled to mean=1)
  - `np.inf` → 0 never-treated convention conversion
- **ImputationDiD pre-period event study coefficients** — pre-treatment "effects" (should be ~0 under parallel trends) for visual pre-trends assessment, following BJS (2024) Test 1
- **TwoStageDiD pre-period event study coefficients** — same pre-trends extension
- **Replicate weight expansion** to 7 additional estimators: DifferenceInDifferences, TwoWayFixedEffects, MultiPeriodDiD, SunAbraham, StackedDiD, ImputationDiD, TwoStageDiD (coverage: 4/13 → 11/13)

### Changed
- ImputationDiD pre-period coefficients use BJS Test 1 (impute Y(0) for treated units in pre-treatment periods)
- SunAbraham replicate weights use full interaction-weighted refit per replicate with cohort-level SEs

### Fixed
- Fix zero-weight demeaning safety in replicate weight paths
- Fix `df_survey` writeback for rank-deficient replicate designs (df=0)
- Fix ImputationDiD `balance_e` zero-qualifying-cohort fallback in pretrends path
- Fix survey zero-mass (g,t) skip warning gap
- Fix SunAbraham positional assignment in replicate loop

## [2.8.2] - 2026-04-02

### Added
- **EPV diagnostics for propensity score logit** — events-per-variable (EPV) checks with Peduzzi convention (predictors excluding intercept) for CallawaySantAnna IPW/DR, TripleDifference IPW/DR, and StaggeredTripleDifference
- `epv_summary()` / `epv_diagnostics` on post-fit results for CallawaySantAnna, TripleDifference, and StaggeredTripleDifference
- `diagnose_propensity()` pre-estimation helper on CallawaySantAnna
- EPV summary block in TripleDifference `summary()` output
- `epv_threshold` parameter for propensity score estimation — warns on low EPV (default) or escalates via `rank_deficient_action="error"`

### Changed
- Default propensity score fallback behavior: safer defaults with method-specific warning messages
- EPV denominator uses predictor count excluding intercept (Peduzzi et al. 1996 convention)

### Fixed
- Fix TripleDifference survey-weighted fallback propensity score
- Fix NaN cache poisoning in propensity score estimation
- Fix `epv_summary` column schema on empty results
- Fix SDDD EPV: use min-EPV across comparison cohorts with cache diagnostic propagation
- Fix `diagnose_propensity` `np.inf` handling

## [2.8.1] - 2026-04-01

### Added
- **Survey-aware DiD tutorial** (`docs/tutorials/16_survey_did.ipynb`) — Phase 7c complete. Full workflow with strata, PSU, FPC, replicate weights, subpopulation analysis, and DEFF diagnostics. Includes `generate_survey_did_data()` DGP function.
- **Survey R cross-validation** — benchmark scripts and tests comparing TSL variance against R's `survey::svyglm()` for basic DiD and TWFE with full survey designs (strata, PSU, FPC). Committed JSON fixtures for CI without R.
- **HonestDiD methodology review and validation** — 478 lines of methodology tests, paper review document, rewritten optimal FLCI with first-difference reparameterization.
- **StaggeredTripleDifference survey support** — full `SurveyDesign` integration with strata/PSU/FPC, replicate weights, and survey-aware bootstrap.

### Changed
- HonestDiD: rewrite optimal FLCI with proper first-difference reparameterization and centrosymmetric LP optimization
- HonestDiD: use `conf_int` from results instead of hardcoded `1.96*se` in event study plots
- Survey tutorial cross-referenced from choosing_estimator.rst and quickstart.rst

### Fixed
- Fix HonestDiD identified set computation and inference (F1-F6 from Rambachan & Roth 2023)
- Fix FLCI slope count (T not T-1) and constraint formula
- Fix NaN CI misclassification as significant (P0 finding)
- Fix M=0 linear extrapolation and survey df folded nct in REGISTRY.md
- Fix replicate-weight scale invariance and BRR test fixtures
- Fix JK1 populated-PSU guard and narrow warning filter

## [2.8.0] - 2026-03-31

### Added
- **Staggered Triple Difference estimator** (Ortiz-Villavicencio & Sant'Anna 2025)
  - `StaggeredTripleDifference` class with group-time ATT(g,t) for DDD designs with staggered adoption
  - Event study aggregation, pre-treatment placebo effects, multiplier bootstrap inference
  - R benchmark validation against `triplediff` package
  - DGP function `generate_staggered_ddd_data()` for simulation and testing
- **Survey Phase 7a: CS IPW/DR + covariates + survey**
  - DRDID panel nuisance-estimation IF corrections (PS + OR) under survey weights
  - Survey-weighted propensity score estimation and outcome regression
  - IFs account for nuisance parameter estimation uncertainty (Sant'Anna & Zhao 2020, Theorem 3.1)
- **Survey Phase 7b: Repeated cross-sections**
  - `CallawaySantAnna(panel=False)` for repeated cross-section surveys (BRFSS, ACS, CPS)
  - Cross-sectional DRDID: `reg` matches `DRDID::reg_did_rc`, `dr` matches `DRDID::drdid_rc`, `ipw` matches `DRDID::std_ipw_did_rc`
  - Survey weights, covariates, and all estimation methods supported
- **Survey Phase 7d: HonestDiD + survey variance**
  - Survey df and full event-study VCV from IF vectors propagated to sensitivity analysis
  - t-distribution critical values with survey degrees of freedom
  - Bootstrap/replicate designs fall back to diagonal VCV with warning
- **Plotly visualization styling**: thread `marker`, `markersize`, `linewidth`, `capsize`, `ci_linewidth` kwargs through plotly backends (previously silently ignored)
- AI agent discoverability for practitioner guide

### Changed
- HonestDiD now raises `ValueError` on non-consecutive event-time grid (was warning)
- HonestDiD validates full grid around reference period
- Panel IPW/DR PS correction scaling matches R's `H/n`, `asy_rep/n`, `colMeans` convention
- RC IF normalization follows R's `psi` convention with explicit `phi` conversion

### Fixed
- Fix HonestDiD reference-aware pre/post split for varying-base event studies
- Fix HonestDiD `_estimate_max_pre_violation` to use reference-aware pre_periods
- Fix panel M2 gradient scaling for IPW/DR nuisance IF corrections
- Fix VCV index alignment for repeated cross-section aggregation
- Fix replicate-weight df propagation: return per-statistic df instead of mutating shared state
- Fix WIF population consistency: zero df `first_treat` for ineligible units
- Fix bootstrap RCS cohort-mass weighting and stale event-study VCV reset

## [2.7.6] - 2026-03-28

### Added
- **AI practitioner guardrails** based on Baker et al. (2025) "Difference-in-Differences Designs: A Practitioner's Guide"
  - `practitioner.py` module with 8-step workflow enforcement for AI agents
  - Estimator-specific handlers ensuring correct diagnostic ordering (pre-trends before estimation, Bacon decomposition before estimator selection)
  - `docs/llms.txt`, `docs/llms-practitioner.txt`, `docs/llms-full.txt` for AI agent discoverability
  - Evaluation rubric (`docs/practitioner-guide-evaluation.md`) with correctness-aware scoring
- **Survey Phase 6: Advanced features**
  - Survey-aware bootstrap for all 8 bootstrap-using estimators (PSU-level multiplier for CS/Imputation/TwoStage/Continuous/Efficient; Rao-Wu rescaled for SA/SyntheticDiD/TROP)
  - Replicate weight variance estimation (BRR, Fay's BRR, JK1, JKn) for OLS-based and IF-based estimators
  - Per-coefficient DEFF diagnostics comparing survey vs SRS variance
  - Subpopulation analysis via `SurveyDesign.subpopulation()` preserving full design structure
  - CS analytical expansion: strata/PSU/FPC for aggregated SEs via `compute_survey_if_variance()`
  - TROP cross-classified pseudo-strata for survey-aware bootstrap

### Changed
- Estimator-specific guidance for parallel trends tests and placebo checks (no shared templates)
- SDiD and TROP split into separate decision tree branches in practitioner workflow

### Fixed
- Fix replicate weight df calculation using pivoted QR rank with R-compatible tolerance
- Fix replicate IF variance score scaling for EfficientDiD, TripleDiff, ContinuousDiD
- Fix panel-to-unit replicate weight propagation and normalization
- Fix CS zero-mass return type and vectorized guard for survey paths
- Fix `solve_logit` effective-sample validation for zero-weight designs
- Fix subpopulation mask validation and EfficientDiD bootstrap guard

## [2.7.5] - 2026-03-23

### Added
- **Phase 4 survey support** for ImputationDiD, TwoStageDiD, and CallawaySantAnna estimators
  - ImputationDiD/TwoStageDiD: analytical survey inference with weights, strata, and PSU (FPC not supported; bootstrap+survey deferred)
  - CallawaySantAnna: weights-only analytical IF/WIF inference matching R `did::wif()` (strata/PSU/FPC deferred)
  - Survey-aware aggregation for group-time, event-study, and overall ATT
- **EfficientDiD enhancements**: doubly robust covariates path, sieve inverse propensity (Eq 3.12), conditional Omega*
- **Cluster-robust SEs** for EfficientDiD with last-cohort control and Hausman pretest
- **Enhanced visualizations**: synth weights, staircase, dose-response, group-time heatmap, plotly backend
- **Local AI review skill** (`/ai-review-local`) with Responses API, delta-diff re-review, and cost visibility
- Add `plotly` optional dependency group (`pip install diff-diff[plotly]`)

### Changed
- Migrate AI local review from Chat Completions to Responses API
- Split TROP estimator into mixin modules (`trop_local.py`, `trop_global.py`) for maintainability
- Refactor `visualization.py` into `visualization/` subpackage
- Improve review script: full-file context, content-first parsing, tiered matching, fingerprint stability

### Fixed
- Fix CallawaySantAnna reg+cov control IF normalization and survey df calculation
- Fix TripleDifference TSL double-weighting and RA nuisance linearization with survey weights
- Fix ContinuousDiD bread normalization, fweight TSL scaling, and weighted-mass IF linearization
- Fix BaconDecomposition exact-weight survey unit_share and empty-cell guard
- Fix SunAbraham survey weight floor in overall ATT aggregation
- Fix plotly event study for non-numeric periods, heatmap masking, color parser

## [2.7.4] - 2026-03-21

### Added
- **Survey/sampling weights support** (`survey_design` parameter) for `DifferenceInDifferences` and `TwoWayFixedEffects`
  - Taylor-series linearization (TSL) variance estimation with stratified multi-stage designs
  - Probability weights (pweight), frequency weights (fweight), and analytic weights (aweight)
  - Finite population correction (FPC) support
  - PSU-based clustering with lonely PSU handling
  - New `diff_diff/survey.py` module with `SurveyDesign` and `compute_survey_vcov`
- **EfficientDiD validation tests** against Chen, Sant'Anna & Xie (2025) using HRS dataset
  - HRS validation fixture with provenance documentation
  - Shared DGP helper in `tests/helpers/edid_dgp.py`
- Simulation-based power analysis for all registry-backed estimators (MDE, sample size, power curves); unregistered estimators supported via custom `data_generator` and `result_extractor`

### Changed
- Extend power analysis to support all registry-backed estimators with `result_extractor` parameter
- Update power analysis tutorial with simulation-based features
- Reject `absorb + fixed_effects` combination (FWL violation) in both survey and non-survey paths

### Fixed
- TWFE cluster-as-PSU injection for no-PSU survey designs
- Non-unique PSU labels across strata with `nest=False`
- FPC validation moved to `compute_survey_vcov` for effective PSU structure
- Survey HC1 meat formula and weighted rank-deficiency handling
- Zero-SE inference, full-census FPC, fweight contract corrections
- Bootstrap+survey fallback in MultiPeriodDiD
- DDD `_snap_n` floor mismatch and `n_per_cell` suppression scope

## [2.7.3] - 2026-03-19

### Added
- Add aarch64 Linux wheel builds to publish workflow

### Changed
- Improve documentation information architecture
- Fix silent interpreter skip and consolidate Linux jobs in publish workflow

## [2.7.2] - 2026-03-18

### Added
- SEO infrastructure: meta tags, sitemap, llms.txt/llms-full.txt for AI discoverability

### Changed
- Rename TROP `method="twostep"` to `method="local"`; `"twostep"` deprecated, removal in v3.0
- Rename internal TROP `_joint_*` methods to `_global_*` for consistency

### Fixed
- Fix TROPResults schema: report unit counts not observation counts
- Fix llms-full.txt accuracy and dynamic canonical URLs

## [2.7.1] - 2026-03-15

### Changed
- Replace BFGS logit with IRLS for propensity score estimation in CallawaySantAnna
- Reject `pscore_trim=0.0` to prevent infinite IPW weights
- Honor `rank_deficient_action="error"` in propensity score paths
- Validate `pscore_trim` at `fit()` to guard against `set_params` bypass
- Mark slow tests (`@pytest.mark.slow`) and exclude by default for faster local iteration
- Use per-class slow markers in `test_trop.py` for faster pure Python CI

### Fixed
- Vectorize Sun-Abraham bootstrap resampling loop for improved performance

## [2.7.0] - 2026-03-15

### Added
- **EfficientDiD estimator** (`EfficientDiD`) implementing Chen, Sant'Anna & Xie (2025) efficient DiD
- CallawaySantAnna event study SEs (WIF-based) and simultaneous confidence bands (sup-t)
- R comparison tests for event-study SEs and cband critical values
- Non-finite outcome validation in `EfficientDiD.fit()`
- CallawaySantAnna speed benchmarks with baseline results
- Estimator alias documentation in README, quickstart, and API docs

### Changed
- **BREAKING: TROP nuclear norm solver step size fix** — The proximal gradient
  threshold for the L matrix (both `method="global"` and `method="twostep"` with
  finite `lambda_nn`) was over-shrinking singular values by a factor of 2. The
  soft-thresholding threshold was λ_nn/max(δ) when the correct value is
  λ_nn/(2·max(δ)), derived from the Lipschitz constant L_f=2·max(δ) of the
  quadratic gradient. This fix produces higher-rank L matrices and closer
  agreement with exact convex optimization solutions. Users with finite
  `lambda_nn` will observe different ATT estimates. Added FISTA/Nesterov
  acceleration to the twostep inner solver for faster L convergence.
- Add (1-W) weight masking to TROP global method, rename joint→global
- Optimize CallawaySantAnna covariate path with Cholesky and pscore caching
- Update Codex AI review model from gpt-5.2-codex to gpt-5.4

### Fixed
- Fix CallawaySantAnna event study SEs (missing WIF) and simultaneous confidence bands
- Fix analytical and bootstrap WIF pg scaling to use global N
- Fix TROP nuclear norm solver threshold scaling for non-uniform weights
- Fix stale coefficients in TROP global low-rank solver and NaN bootstrap poisoning
- Fix NaN-cell preservation in CallawaySantAnna balance_e aggregation
- Fix not-yet-treated cache keys and dropped-cell warning
- Fix rank-deficiency handling with Cholesky rank checks and reduced-column solve
- Fix Rust convergence criterion, n_valid_treated consistency, and NaN bootstrap SE

## [2.6.1] - 2026-03-08

### Added
- Short aliases for all estimators (e.g., `DiD`, `TWFE`, `EventStudy`, `CS`, `SDiD`)

### Changed
- Update roadmap for v2.6.0: reflect completed work and refresh priorities
- Add ContinuousDiD to ReadTheDocs API reference and choosing guide
- Add SPT identification caveat and data requirements per review
- Add time-invariant dose requirement to data requirements

### Fixed
- Fix alias docs wording: clarify TROP has no alias
- Fix ContinuousDiD SE method: influence function, not delta method
- Fix methodology doc: influence functions, not delta method for ContinuousDiD SEs
- Fix dollar sign escaping in continuous DiD tutorial
- Fix continuous DiD tutorial formatting: escape dollar signs and split chart cell
- Fix methodology claims and slide numbering per PR review

## [2.6.0] - 2026-02-22

### Added
- **Continuous DiD estimator** (`ContinuousDiD`) implementing Callaway, Goodman-Bacon & Sant'Anna (2024)
  for continuous treatment dose-response analysis
  - `ContinuousDiDResults` with dose-response curves and event-study effects
  - `DoseResponseCurve` with bootstrap p-values
  - Analytical and bootstrap event-study SEs
  - P(D=0) warning for low-probability control groups
- Stacked DiD tutorial (Tutorial 13) with Q-weight computation walkthrough

### Changed
- Clarify aggregate Q-weight computation for unbalanced panels in Stacked DiD tutorial
- Replace SunAbraham manual bootstrap stats with NaN-gated utility

### Fixed
- Fix not-yet-treated control mask to respect anticipation parameter in ContinuousDiD
- Guard non-finite `original_effect` in `compute_effect_bootstrap_stats`
- Fix bootstrap NaN propagation for rank-deficient cells
- Fix NaN propagation in rank-deficient spline predictions
- Guard bootstrap NaN propagation: SE/CI/p-value all NaN when SE invalid
- Fix bootstrap ACRT^{glob} centering bug
- Fix bootstrap percentile inference and analytical event-study SE scaling
- Fix control group bug and dose validation in ContinuousDiD

## [2.5.0] - 2026-02-19

### Added
- Stacked DiD estimator (`StackedDiD`) implementing Wing, Freedman & Hollingsworth (2024)
  with corrective Q-weights for compositional balance across event times
- Sub-experiment construction per adoption cohort with clean (never-yet-treated) controls
- IC1/IC2 trimming for compositional balance across event times
- Q-weights for aggregate, population, or sample share estimands (Table 1)
- WLS event study regression via sqrt(w) transformation
- `stacked_did()` convenience function
- R benchmark scripts for Stacked DiD validation (`benchmarks/R/benchmark_stacked_did.R`)
- Comprehensive test suite for Stacked DiD (`tests/test_stacked_did.py`)

### Fixed
- NaN inference handling in pure Python mode for edge cases

## [2.4.3] - 2026-02-19

### Changed
- Rewrite TripleDifference estimator to match R's `triplediff::ddd()` — all 3 estimation
  methods (DR, IPW, RA) now use three-DiD decomposition with influence function SE, achieving
  <0.001% relative difference from R across all 24 comparisons (4 DGPs × 3 methods × 2 covariate settings)
- Validate cluster column in TripleDifference for proper cluster-robust SEs
- Handle non-finite influence function propagation in TripleDifference edge cases
- Propensity score fallback uses Hessian-based SE when score optimization fails
- Improved R-squared consistency across estimation methods

### Fixed
- Fix low cell count warning and overlap detection in TripleDifference IPW
- Fix cluster SE computation to use functional (groupby) approach instead of loop
- Fix rank deficiency handling in TripleDifference regression adjustment

### Added
- 91 methodology verification tests for TripleDifference (`tests/test_methodology_triple_diff.py`)
- R benchmark scripts for triple difference validation (`benchmarks/R/benchmark_triplediff.R`)
- Update METHODOLOGY_REVIEW.md to reflect completed TripleDifference review

## [2.4.2] - 2026-02-18

### Added
- **Conditional BLAS linking for Rust backend** — Apple Accelerate on macOS, OpenBLAS on Linux.
  Pre-built wheels now use platform-optimized BLAS for matrix-vector and matrix-matrix
  operations across all Rust-accelerated code paths (weights, OLS, TROP). Windows continues
  using pure Rust (no external dependencies). Improves Rust backend performance at larger scales.
- `rust_backend_info()` diagnostic function in `diff_diff._backend` — reports compile-time
  BLAS feature status (blas, accelerate, openblas)

### Fixed
- **Rust SDID backend performance regression at scale** — Frank-Wolfe solver was 3-10x slower than pure Python at 1k+ scale
  - Gram-accelerated FW loop for time weights: precomputes A^T@A, reducing per-iteration cost from O(N×T0) to O(T0) (~100x speedup per iteration at 5k scale)
  - Allocation-free FW loop for unit weights: 1 GEMV per iteration (was 3), zero heap allocations (was ~8)
  - Dispatch based on problem dimensions: Gram path when T0 < N, standard path when T0 >= N
  - Rust backend now faster than pure Python at all scales

## [2.4.1] - 2026-02-17

### Added
- Tutorial notebook for Two-Stage DiD (Gardner 2022) (`docs/tutorials/12_two_stage_did.ipynb`)

### Changed
- Module splits for large files: ImputationDiD, TwoStageDiD, and TROP each split into separate results and bootstrap submodules
- Migrated remaining inline inference computations to `safe_inference()` utility
- Replaced `@` operator with `np.dot()` at observation-dimension sites to avoid Apple M4 BLAS warnings
- Updated TODO.md and ROADMAP.md for accuracy post-v2.4.0

### Fixed
- Matplotlib import guards added to tutorials 11 and 12
- Various bug fixes from code quality cleanup (diagnostics, estimators, linalg, staggered, sun_abraham, synthetic_did, triple_diff)

## [2.4.0] - 2026-02-16

### Added
- **Gardner (2022) Two-Stage DiD estimator** (`TwoStageDiD`)
  - Two-stage estimator: (1) estimate unit+time FE on untreated obs, (2) regress residualized outcomes on treatment indicators
  - `TwoStageDiDResults` with overall ATT, event study, group effects, per-observation treatment effects
  - `TwoStageBootstrapResults` for multiplier bootstrap inference on GMM influence function
  - `two_stage_did()` convenience function for quick estimation
  - Point estimates identical to ImputationDiD; different variance estimator (GMM sandwich vs. conservative)
  - No finite-sample adjustments (raw asymptotic sandwich, matching R `did2s`)
- Proposition 5 detection for unidentified long-run horizons without never-treated units

### Changed
- Workflow improvements to reduce PR review rounds

### Fixed
- Zero-observation horizons/cohorts producing se=0 instead of NaN in TwoStageDiD
- Edge case fixes for TwoStageDiD (PR review feedback)
- Grep PCRE patterns updated to use POSIX character classes

## [2.3.2] - 2026-02-16

### Added
- **Python 3.13 support** with upper version cap (`>=3.9,<3.14`)

### Changed
- **Sun-Abraham methodology review** (PR #153)
  - IW aggregation weights now use event-time observation counts (not group sizes)
  - Normalize `np.inf` never-treated encoding before treatment group detection
  - Add R benchmark scripts and methodology-aligned tests
- Use `rank_deficient_action` and `np.errstate` instead of broad `RuntimeWarning` filter in SDID tutorial

### Fixed
- Sun-Abraham bootstrap NaN propagation for non-finite ATT estimates
- Sun-Abraham df_adjustment off-by-one in analytical SE computation
- CI pandas compatibility for SunAbraham bootstrap inference
- SyntheticDiD tutorial: eliminate pre-treatment fit warnings

## [2.3.1] - 2026-02-15

### Fixed
- Fix docs/PyPI version mismatch (issue #146) — RTD now builds versioned docs from source
- Fix RTD docs build failure caused by Rust/maturin compilation timeout on ReadTheDocs

### Changed
- Remove Rust outer-loop variance estimation for SyntheticDiD (placebo and bootstrap)
  - Fixes SE mismatch between pure Python and Rust backends (different RNG sequences)
  - Fixes Rust performance regression at 1k+ scale (memory bandwidth saturation from rayon parallelism)
  - Inner Frank-Wolfe weight computation still uses Rust when available

### Documentation
- Re-run SyntheticDiD benchmarks against R after Frank-Wolfe methodology rewrite
- Updated `docs/benchmarks.rst` SDID validation results, performance tables, and known differences
- ATT now matches R to < 1e-10 (previously 0.3% diff) since both use Frank-Wolfe optimizer

## [2.3.0] - 2026-02-09

### Added
- **Borusyak-Jaravel-Spiess (2024) Imputation DiD estimator** (`ImputationDiD`)
  - Efficient imputation estimator for staggered DiD designs
  - OLS on untreated observations for unit+time FE, impute counterfactual Y(0), aggregate
  - Conservative variance (Theorem 3) with `aux_partition` parameter for SE tightness
  - Pre-trend test (Equation 9) via `results.pretrend_test()`
  - Percentile bootstrap inference
  - Influence-function bootstrap with sparse variance and weight/covariate fixes
  - Absorbing-treatment validation for non-constant `first_treat`
  - Empty event-study warning for unidentified long-run horizons
- **`/paper-review` skill** for academic paper methodology extraction
- **`/read-feedback-revise` skill** for addressing PR review comments
- **`--pr` flag for `/review-plan` skill** to review plans posted as PR comments
- **`--updated` flag for `/review-plan` skill** for re-reviewing revised plans
- **MultiPeriodDiD vs R (fixest) benchmark** for cross-language validation

### Changed
- Shortened test suite runtime with parallel execution and reduced iterations

### Fixed
- **TWFE within-transformation bug** identified during methodology review
- TWFE: added non-{0,1} binary time warning, ATT invariance tests, and R fixture caching
- TWFE: single-pass demeaning, HC1 test fix, fixest coeftable comparison
- MultiPeriodDiD: added unit FE and NaN guard for R comparison benchmark
- Removed tracked PDF from repo and gitignored papers directory

## [2.2.1] - 2026-02-07

### Changed
- **MultiPeriodDiD: Full event-study specification** (BREAKING)
  - Treatment × period interactions now created for ALL periods (pre and post),
    not just post-treatment
  - Pre-period coefficients available for parallel trends assessment
  - Default reference period changed from first to last pre-period (e=-1 convention)
    with FutureWarning for one release cycle
  - `period_effects` dict now contains both pre and post period effects
  - `to_dataframe()` includes `is_post` column
  - `summary()` output now shows pre-period effects section
  - t_stat uses `np.isfinite(se) and se > 0` guard (consistent with other estimators)

### Added
- Time-varying treatment warning when `unit` is provided and treatment varies
  within units (guides users toward ever-treated indicator D_i)
- `unit` parameter to `MultiPeriodDiD.fit()` for staggered adoption detection
- `reference_period` and `interaction_indices` attributes on `MultiPeriodDiDResults`
- `pre_period_effects` and `post_period_effects` convenience properties on results
- Pre-period section in `summary()` output with reference period indicator
- `ValueError` when `reference_period` is set to a post-treatment period
- Staggered adoption warning when treatment timing varies across units (with `unit` param)
- Informative KeyError when accessing reference period via `get_effect()`

### Removed
- **TROP `variance_method` parameter** — Jackknife variance estimation removed.
  Bootstrap (the only method specified in Athey et al. 2025) is now always used.
  The `variance_method` field has also been removed from `TROPResults`.
- **TROP `max_loocv_samples` parameter** — Control observation subsampling removed
  from LOOCV tuning parameter selection. Equation 5 of Athey et al. (2025) explicitly
  sums over ALL control observations where D=0; the previous subsampling (default 100)
  was not specified in the paper. LOOCV now uses all control observations, making
  tuning fully deterministic. Inner LOOCV loops in the Rust backend are parallelized
  to compensate for the increased observation count.

### Fixed
- HonestDiD: filter non-finite period effects from MultiPeriodDiD results
  (prevents NaN propagation into sensitivity bounds; raises ValueError
  when no finite pre- or post-period effects remain)
- HonestDiD VCV extraction: now uses interaction sub-VCV instead of full regression VCV
  (via `interaction_indices` period → column index mapping)
- MultiPeriodDiD: `avg_se` guard now checks `np.isfinite()` (matches per-period pattern;
  prevents `avg_t_stat=0` / `avg_p_value=1` when variance is infinite)
- HonestDiD: extraction now uses explicit pre-then-post ordering instead of sorted period
  labels (prevents misclassification when period labels don't sort chronologically)
- Backend-aware test parameter scaling for pure Python CI performance
- Lower TROP stratified bootstrap threshold floor from 11 to 5 for pure Python CI

## [2.2.0] - 2026-01-27

### Added
- **Windows wheel builds** using pure-Rust `faer` library for linear algebra (PR #115)
  - Eliminates external BLAS/LAPACK dependencies (no OpenBLAS or Intel MKL required)
  - Enables cross-platform wheel builds for Linux, macOS, and Windows
  - Simplifies installation on all platforms

### Changed
- **Rust backend migrated from nalgebra/ndarray to faer** (PR #115)
  - OLS solver now uses faer's SVD implementation
  - Robust variance estimation uses faer's matrix operations
  - TROP distance calculations use faer primitives
  - Maintains numerical parity with existing NumPy backend

### Fixed
- **Rust backend numerical stability improvements** (PR #115)
  - Improved singular matrix detection with condition number checks
  - NaN propagation in variance-covariance estimation
  - Fallback to Python backend on numerical instability with warning
  - Underdetermined SVD handling (n < k case)
- **macOS CI compatibility** for Python 3.14 with `PYO3_USE_ABI3_FORWARD_COMPATIBILITY`

## [2.1.9] - 2026-01-26

### Added
- **Unified LOOCV for TROP joint method** with Rust acceleration (PR #113)
  - Leave-one-out cross-validation for rank and regularization parameter selection
  - Rust backend provides significant speedup for LOOCV grid search

### Fixed
- **TROP joint method Rust/Python parity** (PR #113)
  - Fixed valid_count bug in LOOCV computation
  - Proper NaN exclusion for units with no valid pre-period data
  - Zero weight assignment for units missing pre-period data
  - Jackknife variance estimation fixes
  - Staggered adoption validation and simultaneous adoption enforcement
  - Treated-pre NaN handling improvements
  - LOOCV subsampling fix for Python-only path

## [2.1.8] - 2026-01-25

### Added
- **`/push-pr-update` skill** for committing and pushing PR revisions
  - Commits local changes to current branch and pushes to remote
  - Triggers AI code review automatically
  - Robust handling for fork repos, unpushed commits, and upstream tracking

### Fixed
- **TROP estimator methodology alignment** (PR #110)
  - Aligned with paper methodology (Equation 5, D matrix semantics)
  - NaN propagation and LOOCV warnings improvements
  - Rust backend test alignment with new loocv_grid_search return signature
  - LOOCV cycling, D matrix validation fixes
  - Final estimation infinity handling and edge case fixes
  - Absorbing-state gap detection and n_post_periods fix

### Changed
- **`/submit-pr` skill improvements** (PR #111)
  - Case-insensitive secret scanning with POSIX ERE regex
  - Verify origin ref exists before push
  - Dynamic default branch detection with fallback
  - Robust handling for unpushed commits, fork repos
  - Files count display in PR summary

## [2.1.7] - 2026-01-25

### Fixed
- **`plot_event_study` reference period normalization behavior**
  - Effects are now only normalized when `reference_period` is explicitly provided
  - Auto-inferred reference periods only apply hollow marker styling (no normalization)
  - Reference period SE is set to NaN during normalization (constraint, not estimate)
  - Updated docstring to clarify explicit vs auto-inferred behavior

### Changed
- Refactored visualization tests to reuse `cs_results` fixture for better performance

## [2.1.6] - 2026-01-24

### Added
- **Methodology verification tests** for DifferenceInDifferences estimator
  - Comprehensive test suite validating all REGISTRY.md requirements
  - Tests for formula interface, coefficient extraction, rank deficiency handling
  - Singleton cluster variance estimation behavioral tests

### Changed
- **REGISTRY.md documentation improvements**
  - Clarified singleton cluster formula notation (u_i² X_i X_i' instead of ambiguous residual² × X'X)
  - Verified DifferenceInDifferences behavior against documented requirements

## [2.1.5] - 2026-01-22

### Added
- **METHODOLOGY_REVIEW.md** tracking document for methodology review progress
  - Review status summary table for all 12 estimators
  - Detailed notes template for each estimator by category
  - Review process guidelines with checklist and priority ordering
- **`base_period` parameter** for CallawaySantAnna pre-treatment effect computation
  - "varying" (default): Pre-treatment uses t-1 as base (consecutive comparisons)
  - "universal": All comparisons use g-anticipation-1 as base
  - Matches R `did::att_gt()` base_period parameter
- **Pre-merge-check skill** (`/pre-merge-check`) for automated PR validation
  - Pattern checks for NaN handling consistency
  - Context-specific checklist generation

### Changed
- **Tutorial 02 improvements**: Added pre-trends section, clarified base_period interaction with anticipation

### Fixed
- Not-yet-treated control group now properly excludes cohort g when computing ATT(g,t)
- Aggregation t_stat uses NaN (not 0.0) when SE is non-finite or zero
- Bootstrap inference for pre-treatment effects with `base_period="varying"`
- NaN propagation for empty post-treatment effects in CallawaySantAnna
- Grep word boundary pattern in pre-merge-check skill

## [2.1.4] - 2026-01-20

### Added
- **Development checklists and workflow improvements** in `CLAUDE.md`
  - Estimator inheritance map showing class hierarchy for `get_params`/`set_params`
  - Test writing guidelines for fallback paths, parameters, and warnings
  - Checklists for adding parameters and warning/error handling
- **R-style rank deficiency handling** across all estimators
  - `rank_deficient_action` parameter: "warn" (default), "error", or "silent"
  - Dropped columns have NaN coefficients (like R's `lm()`)
  - VCoV matrix has NaN for rows/cols of dropped coefficients
  - Propagated to all estimators: DifferenceInDifferences, MultiPeriodDiD, TwoWayFixedEffects, CallawaySantAnna, SunAbraham, TripleDifference, TROP, SyntheticDiD

### Fixed
- `get_params()` now includes `rank_deficient_action` parameter (fixes sklearn cloning)
- NaN vcov fallback in Rust backend for rank-deficient matrices
- MultiPeriodDiD vcov/df computation for rank-deficient designs
- Average ATT inference for rank-deficient designs

### Changed
- Rank tolerance aligned with R's `lm()` default for consistent behavior

## [2.1.3] - 2026-01-19

### Fixed
- TROP estimator paper conformance issues (Athey et al. 2025)
  - Control set now includes pre-treatment observations of eventually-treated units (Issue A)
  - Unit distance computation excludes target period per Equation 3 (Issue B)
  - Nuclear norm update uses weighted proximal gradient instead of unweighted soft-thresholding (Issue C)
  - Bootstrap sampling now stratifies by treatment status per Algorithm 3 (Issue D)
- TROP Rust backend alignment with paper specification
  - Weight normalization to sum to 1 (probability weights)
  - Weighted proximal gradient for L update with step size η ≤ 1/max(W)

### Changed
- Cleaned up unused parameters from TROP Rust API
  - Removed `control_unit_idx` and `unit_dist_matrix` from public functions
  - Per-observation distances now computed dynamically (more accurate, slightly slower)

## [2.1.2] - 2026-01-19

### Added
- **Consolidated DGP functions** in `prep.py` for all supported DiD designs
  - `generate_did_data()` - Basic 2x2 DiD data generation
  - `generate_staggered_data()` - Staggered adoption data for Callaway-Sant'Anna/Sun-Abraham
  - `generate_factor_data()` - Factor model data for TROP/SyntheticDiD
  - `generate_ddd_data()` - Triple Difference (DDD) design data
  - `generate_panel_data()` - Panel data with optional parallel trends violations
  - `generate_event_study_data()` - Event study data with simultaneous treatment

### Changed
- **Clean up development tracking files** for v2.1.1 release
  - Removed completed items from TODO.md (now tracked in CHANGELOG)
  - Updated ROADMAP.md version numbers and removed shipped TROP section
  - Updated `prep.py` line count in Large Module Files table (1338 → 1993)

## [2.1.1] - 2026-01-19

### Added
- **Rust backend acceleration for TROP estimator** delivering 5-20x overall speedup
  - `compute_unit_distance_matrix` - Parallel pairwise RMSE computation for donor matching
  - `loocv_grid_search` - Parallel leave-one-out cross-validation across 180 parameter combinations
  - `bootstrap_trop_variance` - Parallel bootstrap variance estimation
  - Automatic fallback to Python when Rust backend unavailable
  - Logging for Rust fallback events to aid debugging
- **`/bump-version` skill** for release management
  - Updates version in `__init__.py`, `pyproject.toml`, and `rust/Cargo.toml`
  - Generates CHANGELOG entries from git commits
  - Adds comparison links automatically
- **`/review-pr` skill** for code review workflow

### Changed
- **TROP estimator performance optimizations** (Python backend)
  - Vectorized distance matrix computation using NumPy broadcasting
  - Extracted tuning constants to module-level for clarity
  - Added `TROPTuningParams` TypedDict for parameter documentation

### Fixed
- Tutorial notebook validation errors in `10_trop.ipynb`
- Pre-existing RuntimeWarnings in CallawaySantAnna bootstrap (documented)
- TROP `pre_periods` parameter handling for edge cases

## [2.1.0] - 2026-01-17

### Added
- **Triply Robust Panel (TROP) estimator** implementing Athey, Imbens, Qu & Viviano (2025)
  - `TROP` class combining three robustness components:
    - Factor model adjustment via SVD (removes unobserved confounders with factor structure)
    - Synthetic control style unit weights
    - SDID style time weights
  - `TROPResults` dataclass with ATT, factors, loadings, unit/time weights
  - `trop()` convenience function for quick estimation
  - Automatic rank selection methods: cross-validation (`'cv'`), information criterion (`'ic'`), elbow detection (`'elbow'`)
  - Bootstrap and placebo-based variance estimation
  - Full integration with existing infrastructure (exports in `__init__.py`, sklearn-compatible API)
  - Tutorial notebook: `docs/tutorials/10_trop.ipynb`
  - Comprehensive test suite: `tests/test_trop.py`

**Reference**: Athey, S., Imbens, G. W., Qu, Z., & Viviano, D. (2025). "Triply Robust Panel Estimators." *Working Paper*. [arXiv:2508.21536](https://arxiv.org/abs/2508.21536)

## [2.0.3] - 2026-01-17

### Changed
- **Rust backend performance optimizations** delivering up to 32x speedup for bootstrap operations
  - Bootstrap weight generation now 16x faster on average (up to 32x for Webb distribution)
  - Direct `Array2` allocation eliminates intermediate `Vec<Vec<f64>>` (~50% memory reduction)
  - Rayon chunk size tuning (`min_len=64`) reduces parallel scheduling overhead
  - Webb distribution uses lookup table instead of 6-way if-else chain

### Added
- **LinearRegression helper class** in `linalg.py` for code deduplication
  - High-level OLS wrapper with unified coefficient extraction and inference
  - Used by DifferenceInDifferences, TwoWayFixedEffects, SunAbraham, TripleDifference
  - Provides `InferenceResult` dataclass for coefficient-level statistics
- **Cholesky factorization** for symmetric positive-definite matrix inversion in Rust backend
  - ~2x faster than LU decomposition for well-conditioned matrices
  - Automatic fallback to LU for near-singular or indefinite matrices
- **Vectorized variance computation** in Rust backend
  - HC1 meat computation: `X' @ (X * e²)` via BLAS instead of O(n×k²) loop
  - Score computation: broadcast multiplication instead of O(n×k) loop
- **Static BLAS linking options** in `rust/Cargo.toml`
  - `openblas-static` and `intel-mkl-static` features for standalone distribution
  - Eliminates runtime BLAS dependency at cost of larger binary size

## [2.0.2] - 2026-01-15

### Fixed
- **CallawaySantAnna SE computation** now exactly matches R's `did` package
  - Fixed weight influence function (wif) formula for "simple" aggregation
  - Corrected `pg` computation: uses `n_g / n_all` (matching R) instead of `n_g / total_treated`
  - Fixed wif iteration: iterates over keepers (post-treatment pairs) with individual ATT(g,t) values
  - SE difference reduced from ~2.5% to <0.01% vs R's `did` package (essentially exact match)
  - Point estimates unchanged; all existing tests pass

## [2.0.1] - 2026-01-13

### Added
- **Shared within-transformation utilities** in `utils.py`
  - `demean_by_group()` - One-way fixed effects demeaning
  - `within_transform()` - Two-way (unit + time) FE transformation
  - Reduces code duplication across `estimators.py`, `twfe.py`, `sun_abraham.py`, `bacon.py`

### Fixed
- **DataFrame fragmentation warning** - Build columns in batch instead of iteratively

### Changed
- Reverted untested Rust backend optimizations (Cholesky factorization, reduced allocations) - these will be re-added when proper testing infrastructure is available

## [2.0.0] - 2026-01-12

### Added
- **Optional Rust backend** for accelerated computation
  - 4-8x speedup for SyntheticDiD and bootstrap operations
  - Parallel bootstrap weight generation (Rademacher, Mammen, Webb)
  - Accelerated OLS solver using OpenBLAS/MKL
  - Cluster-robust variance estimation
  - Synthetic control weight optimization with simplex projection
  - Pre-built wheels for Linux x86_64 and macOS ARM64
  - Pure Python fallback for all other platforms
- **`diff_diff/_backend.py`** - Backend detection and configuration module
  - `HAS_RUST_BACKEND` flag exported in main package
  - `DIFF_DIFF_BACKEND` environment variable for backend control:
    - `'auto'` (default) - Use Rust if available, fall back to Python
    - `'python'` - Force pure Python mode
    - `'rust'` - Force Rust mode (fails if unavailable)
- **Rust source code** in `rust/` directory
  - `rust/src/lib.rs` - PyO3 module definition
  - `rust/src/bootstrap.rs` - Parallel bootstrap weight generation
  - `rust/src/linalg.rs` - OLS solver and robust variance estimation
  - `rust/src/weights.rs` - Synthetic control weights and simplex projection
- **Rust backend test suite** - `tests/test_rust_backend.py` for equivalence testing

### Changed
- Package version bumped from 1.4.0 to 2.0.0 (major version for new backend)
- CI/CD updated to build Rust extensions with maturin
- ReadTheDocs now installs from PyPI (pre-built wheels with Rust backend)

## [1.4.0] - 2026-01-11

### Added
- **Unified linear algebra backend** (`diff_diff/linalg.py`)
  - `solve_ols()` - Optimized OLS solver using scipy's gelsy LAPACK driver
  - `compute_robust_vcov()` - Vectorized (clustered) robust variance-covariance
  - Single optimization point for all estimators; prepares for future Rust backend
  - New `tests/test_linalg.py` with comprehensive tests

### Changed
- **Major performance improvements** - All estimators now significantly faster
  - BasicDiD/TWFE @ 10K: 0.835s → 0.011s (76x faster, now 4.2x faster than R)
  - CallawaySantAnna @ 10K: 2.234s → 0.109s (20x faster, now 7.2x faster than R)
  - All results numerically identical to previous versions
- **CallawaySantAnna optimizations** (`staggered.py`)
  - Pre-computed wide-format outcome matrix and cohort masks
  - Vectorized ATT(g,t) computation using numpy operations (23x faster)
  - Batch bootstrap weight generation
  - Vectorized multiplier bootstrap using matrix operations (26x faster)
- **TWFE optimization** (`twfe.py`)
  - Cached groupby indexes for within-transformation
- **All estimators migrated** to unified `linalg.py` backend
  - `estimators.py`, `twfe.py`, `staggered.py`, `triple_diff.py`,
    `synthetic_did.py`, `sun_abraham.py`, `utils.py`

### Behavioral Changes
- **Rank-deficient design matrices**: The new `gelsy` LAPACK driver handles
  rank-deficient matrices gracefully (returning a least-norm solution) rather
  than raising an explicit error. Previously, `DifferenceInDifferences` would
  raise `ValueError("Design matrix is rank-deficient")`. Users relying on this
  error for collinearity detection should validate their design matrices
  separately. Results remain numerically correct for well-specified models.

## [1.3.1] - 2026-01-10

### Added
- **SyntheticDiD placebo-based variance estimation** matching R's `synthdid` package methodology
  - New `variance_method` parameter with options `"bootstrap"` (default) and `"placebo"`
  - Placebo method implements Algorithm 4 from Arkhangelsky et al. (2021):
    1. Randomly permutes control unit indices
    2. Designates N₁ controls as pseudo-treated (matching actual treated count)
    3. Renormalizes original unit weights for remaining pseudo-controls
    4. Computes SDID estimate with renormalized weights
    5. Repeats for `n_bootstrap` replications
    6. SE = sqrt((r-1)/r) × sd(estimates)
  - Provides methodological parity with R's `synthdid::vcov(method = "placebo")`
  - `n_bootstrap` parameter now used for both bootstrap and placebo replications
  - `SyntheticDiDResults` now tracks `variance_method` and `n_bootstrap` attributes
  - Results summary displays variance method and replications count

**Reference**: Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S. (2021). Synthetic Difference-in-Differences. *American Economic Review*, 111(12), 4088-4118.

## [1.3.0] - 2026-01-09

### Added
- **Triple Difference (DDD) estimator** implementing Ortiz-Villavicencio & Sant'Anna (2025)
  - `TripleDifference` class for DDD designs where treatment requires two criteria (group AND partition)
  - `TripleDifferenceResults` dataclass with ATT, SEs, cell means, and diagnostics
  - `triple_difference()` convenience function for quick estimation
  - Three estimation methods: regression adjustment (`reg`), inverse probability weighting (`ipw`), and doubly robust (`dr`)
  - Proper covariate handling (unlike naive DDD implementations that difference two DiDs)
  - Propensity score trimming for IPW/DR methods
  - Cluster-robust standard errors support
  - Tutorial notebook: `docs/tutorials/08_triple_diff.ipynb`

**Reference**: Ortiz-Villavicencio, M., & Sant'Anna, P. H. C. (2025). "Better Understanding Triple Differences Estimators." *Working Paper*. [arXiv:2505.09942](https://arxiv.org/abs/2505.09942)

## [1.2.1] - 2026-01-08

### Added
- **Expanded test coverage** for edge cases:
  - Wild bootstrap with very few clusters (< 5), including 2-3 cluster scenarios
  - Unbalanced panels with missing periods across units
  - Single treated unit scenarios for DiD and Synthetic DiD
  - Perfect collinearity detection (validates clear error messages)
  - CallawaySantAnna with single treatment cohort
  - SyntheticDiD with insufficient pre-treatment periods

### Changed
- **Refactored CallawaySantAnna bootstrap**: Extracted `_compute_effect_bootstrap_stats()` helper method for cleaner code and reduced duplication in bootstrap statistics computation.

## [1.2.0] - 2026-01-07

### Added
- **Pre-Trends Power Analysis** (Roth 2022) for assessing informativeness of pre-trends tests
  - `PreTrendsPower` class for computing power and minimum detectable violation (MDV)
  - `PreTrendsPowerResults` dataclass with power, MDV, and test statistics
  - `PreTrendsPowerCurve` for power curves across violation magnitudes
  - `compute_pretrends_power()` and `compute_mdv()` convenience functions
  - Multiple violation types: `linear`, `constant`, `last_period`, `custom`
  - Integration with Honest DiD via `sensitivity_to_honest_did()` method
  - `plot_pretrends_power()` visualization for power curves
  - Tutorial notebook: `docs/tutorials/07_pretrends_power.ipynb`
  - Full API documentation: `docs/api/pretrends.rst`

**Reference**: Roth, J. (2022). "Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends." *American Economic Review: Insights*, 4(3), 305-322.

### Fixed
- **Reference period handling in pre-trends analysis**: Fixed bug where reference period was incorrectly assigned `avg_se` instead of being excluded from power calculations. Now properly excludes the omitted reference period from the joint Wald test.

## [1.1.1] - 2026-01-06

### Fixed
- **SyntheticDiD bootstrap error handling**: Bootstrap now raises clear `ValueError` when all iterations fail, instead of silently returning SE=0.0. Added warnings for edge cases (single successful iteration, high failure rate).

- **Diagnostics module error handling**: Improved error messages in `permutation_test()` and `leave_one_out_test()` with actionable guidance. Added warnings when significant iterations fail. Enhanced `run_all_placebo_tests()` to return structured error info including error type.

### Changed
- **Code deduplication**: Extracted wild bootstrap inference logic to shared `_run_wild_bootstrap_inference()` method in `DifferenceInDifferences` base class, used by both `DifferenceInDifferences` and `TwoWayFixedEffects`.

- **Type hints**: Added missing type hints to nested functions:
  - `compute_trend()` in `utils.py`
  - `neg_log_likelihood()` and `gradient()` in `staggered.py`
  - `format_label()` in `prep.py`

## [1.1.0] - 2026-01-05

### Added
- **Sun-Abraham (2021) interaction-weighted estimator** for staggered DiD
  - `SunAbraham` class implementing saturated regression approach
  - `SunAbrahamResults` with event study effects, cohort weights, and overall ATT
  - `SABootstrapResults` for bootstrap inference (SEs, CIs, p-values)
  - Support for `never_treated` and `not_yet_treated` control groups
  - Analytical and cluster-robust standard errors
  - Multiplier bootstrap with Rademacher, Mammen, or Webb weights
  - Integration with `plot_event_study()` visualization
  - Useful robustness check alongside Callaway-Sant'Anna

**Reference**: Sun, L., & Abraham, S. (2021). "Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects." *Journal of Econometrics*, 225(2), 175-199.

## [1.0.2] - 2026-01-04

### Changed
- Refactored `estimators.py` to reduce module size
  - Moved `TwoWayFixedEffects` to `diff_diff/twfe.py`
  - Moved `SyntheticDiD` to `diff_diff/synthetic_did.py`
  - Backward compatible re-exports maintained in `estimators.py`

### Fixed
- Fixed ReadTheDocs version display by importing from package `__version__`

## [1.0.1] - 2026-01-04

### Fixed
- Tech debt cleanup (Tier 1 + Tier 2)
  - Improved code organization and documentation
  - Fixed minor issues identified in tech debt review

## [1.0.0] - 2026-01-04

### Added
- **Goodman-Bacon decomposition** for TWFE diagnostics
  - `BaconDecomposition` class for decomposing TWFE into weighted 2x2 comparisons
  - `Comparison2x2` dataclass for individual comparisons (treated_vs_never, earlier_vs_later, later_vs_earlier)
  - `BaconDecompositionResults` with weights and estimates by comparison type
  - `bacon_decompose()` convenience function
  - `plot_bacon()` visualization for decomposition results
  - Integration via `TwoWayFixedEffects.decompose()` method
- **Power analysis** for study design
  - `PowerAnalysis` class for analytical power calculations
  - `PowerResults` and `SimulationPowerResults` dataclasses
  - `compute_mde()`, `compute_power()`, `compute_sample_size()` convenience functions
  - `simulate_power()` for Monte Carlo simulation-based power analysis
  - `plot_power_curve()` visualization for power analysis
  - Tutorial notebook: `docs/tutorials/06_power_analysis.ipynb`
- **Callaway-Sant'Anna multiplier bootstrap** for inference
  - `CSBootstrapResults` with standard errors, confidence intervals, p-values
  - Rademacher, Mammen, and Webb weight distributions
  - Bootstrap inference for all aggregation methods
- **Troubleshooting guide** in documentation
- **Standard error computation guide** explaining SE differences across estimators

### Changed
- Updated package status to Production/Stable (was Alpha)
- SyntheticDiD bootstrap now warns when >5% of iterations fail

### Fixed
- Silent bootstrap failures in SyntheticDiD now produce warnings

## [0.6.0]

### Added
- **CallawaySantAnna covariate adjustment** for conditional parallel trends
  - Outcome regression (`estimation_method='reg'`)
  - Inverse probability weighting (`estimation_method='ipw'`)
  - Doubly robust estimation (`estimation_method='dr'`)
  - Pass covariates via `covariates` parameter in `fit()`
- **Honest DiD sensitivity analysis** (Rambachan & Roth 2023)
  - `HonestDiD` class for computing bounds under parallel trends violations
  - Relative magnitudes restriction (`DeltaRM`) - bounds post-treatment violations by pre-treatment
  - Smoothness restriction (`DeltaSD`) - bounds second differences of trend violations
  - Combined restrictions (`DeltaSDRM`)
  - FLCI and C-LF confidence interval methods
  - Breakdown value computation via `breakdown_value()`
  - Sensitivity analysis over M grid via `sensitivity_analysis()`
  - `HonestDiDResults` and `SensitivityResults` dataclasses
  - `compute_honest_did()` convenience function
  - `plot_sensitivity()` for sensitivity analysis visualization
  - `plot_honest_event_study()` for event study with honest CIs
  - Tutorial notebook: `docs/tutorials/05_honest_did.ipynb`
- **API documentation site** with Sphinx
  - Full API reference auto-generated from docstrings
  - "Which estimator should I use?" decision guide
  - Comparison with R packages (did, HonestDiD)
  - Getting started / quickstart guide

### Changed
- Updated mypy configuration for better numpy type compatibility
- Modernized ruff configuration to use `[tool.ruff.lint]` section

### Fixed
- Fixed 21 ruff linting issues (import ordering, unused variables, ambiguous names)
- Fixed 94 mypy type checking issues (Optional types, numpy type casts, assertions)
- Added missing return statement in `run_placebo_test()`

## [0.5.0]

### Added
- **Wild cluster bootstrap** for valid inference with few clusters
  - Rademacher weights (default, good for most cases)
  - Webb's 6-point distribution (recommended for <10 clusters)
  - Mammen's two-point distribution
  - `WildBootstrapResults` dataclass
  - `wild_bootstrap_se()` utility function
  - Integration with `DifferenceInDifferences` and `TwoWayFixedEffects` via `inference='wild_bootstrap'`
- **Placebo tests module** (`diff_diff.diagnostics`)
  - `placebo_timing_test()` - fake treatment timing test
  - `placebo_group_test()` - fake treatment group test
  - `permutation_test()` - permutation-based inference
  - `leave_one_out_test()` - sensitivity to individual treated units
  - `run_placebo_test()` - unified dispatcher for all test types
  - `run_all_placebo_tests()` - comprehensive diagnostic suite
  - `PlaceboTestResults` dataclass
- **Tutorial notebooks** in `docs/tutorials/`
  - `01_basic_did.ipynb` - Basic 2x2 DiD, formula interface, covariates, fixed effects, wild bootstrap
  - `02_staggered_did.ipynb` - Staggered adoption with Callaway-Sant'Anna
  - `03_synthetic_did.ipynb` - Synthetic DiD with unit/time weights
  - `04_parallel_trends.ipynb` - Parallel trends testing and diagnostics
- Comprehensive test coverage (380+ tests)

## [0.4.0]

### Added
- **Callaway-Sant'Anna estimator** for staggered difference-in-differences
  - `CallawaySantAnna` class with group-time ATT(g,t) estimation
  - Support for `never_treated` and `not_yet_treated` control groups
  - Aggregation methods: `simple`, `group`, `calendar`, `event_study`
  - `CallawaySantAnnaResults` with group-time effects and aggregations
  - `GroupTimeEffect` dataclass for individual effects
- **Event study visualization** via `plot_event_study()`
  - Works with `MultiPeriodDiDResults`, `CallawaySantAnnaResults`, or DataFrames
  - Publication-ready formatting with customization options
- **Group effects visualization** via `plot_group_effects()`
- **Parallel trends testing utilities**
  - `check_parallel_trends()` - simple slope-based test
  - `check_parallel_trends_robust()` - Wasserstein distance test
  - `equivalence_test_trends()` - TOST equivalence test

## [0.3.0]

### Added
- **Synthetic Difference-in-Differences** (`SyntheticDiD`)
  - Unit weight optimization for synthetic control
  - Time weight computation for pre-treatment periods
  - Placebo-based and bootstrap inference
  - `SyntheticDiDResults` with weight accessors
- **Multi-period DiD** (`MultiPeriodDiD`)
  - Event-study style estimation with period-specific effects
  - `MultiPeriodDiDResults` with `period_effects` dictionary
  - `PeriodEffect` dataclass for individual period results
- **Data preparation utilities** (`diff_diff.prep`)
  - `generate_did_data()` - synthetic data generation
  - `make_treatment_indicator()` - create treatment from categorical/numeric
  - `make_post_indicator()` - create post-treatment indicator
  - `wide_to_long()` - reshape wide to long format
  - `balance_panel()` - ensure balanced panel data
  - `validate_did_data()` - data validation
  - `summarize_did_data()` - summary statistics by group
  - `create_event_time()` - event time for staggered designs
  - `aggregate_to_cohorts()` - aggregate to cohort means
  - `rank_control_units()` - rank controls by similarity

## [0.2.0]

### Added
- **Two-Way Fixed Effects** (`TwoWayFixedEffects`)
  - Within-transformation for unit and time fixed effects
  - Efficient handling of high-dimensional fixed effects via `absorb`
- **Fixed effects support** in base `DifferenceInDifferences`
  - `fixed_effects` parameter for dummy variable approach
  - `absorb` parameter for within-transformation approach
- **Cluster-robust standard errors**
  - `cluster` parameter for cluster-robust inference
- **Formula interface**
  - R-style formulas like `"outcome ~ treated * post"`
  - Support for covariates in formulas

## [0.1.0]

### Added
- Initial release
- **Basic Difference-in-Differences** (`DifferenceInDifferences`)
  - sklearn-like API with `fit()` method
  - Column name interface for outcome, treatment, time
  - Heteroskedasticity-robust (HC1) standard errors
  - `DiDResults` dataclass with ATT, SE, p-value, confidence intervals
  - `summary()` and `print_summary()` methods
  - `to_dict()` and `to_dataframe()` export methods
  - `is_significant` and `significance_stars` properties

[3.0.1]: https://github.com/igerber/diff-diff/compare/v3.0.0...v3.0.1
[3.0.0]: https://github.com/igerber/diff-diff/compare/v2.9.1...v3.0.0
[2.9.1]: https://github.com/igerber/diff-diff/compare/v2.9.0...v2.9.1
[2.9.0]: https://github.com/igerber/diff-diff/compare/v2.8.4...v2.9.0
[2.8.4]: https://github.com/igerber/diff-diff/compare/v2.8.3...v2.8.4
[2.8.3]: https://github.com/igerber/diff-diff/compare/v2.8.2...v2.8.3
[2.8.2]: https://github.com/igerber/diff-diff/compare/v2.8.1...v2.8.2
[2.8.1]: https://github.com/igerber/diff-diff/compare/v2.8.0...v2.8.1
[2.8.0]: https://github.com/igerber/diff-diff/compare/v2.7.6...v2.8.0
[2.7.6]: https://github.com/igerber/diff-diff/compare/v2.7.5...v2.7.6
[2.7.5]: https://github.com/igerber/diff-diff/compare/v2.7.4...v2.7.5
[2.7.4]: https://github.com/igerber/diff-diff/compare/v2.7.3...v2.7.4
[2.7.3]: https://github.com/igerber/diff-diff/compare/v2.7.2...v2.7.3
[2.7.2]: https://github.com/igerber/diff-diff/compare/v2.7.1...v2.7.2
[2.7.1]: https://github.com/igerber/diff-diff/compare/v2.7.0...v2.7.1
[2.7.0]: https://github.com/igerber/diff-diff/compare/v2.6.1...v2.7.0
[2.6.1]: https://github.com/igerber/diff-diff/compare/v2.6.0...v2.6.1
[2.6.0]: https://github.com/igerber/diff-diff/compare/v2.5.0...v2.6.0
[2.5.0]: https://github.com/igerber/diff-diff/compare/v2.4.3...v2.5.0
[2.4.3]: https://github.com/igerber/diff-diff/compare/v2.4.2...v2.4.3
[2.4.2]: https://github.com/igerber/diff-diff/compare/v2.4.1...v2.4.2
[2.4.1]: https://github.com/igerber/diff-diff/compare/v2.4.0...v2.4.1
[2.4.0]: https://github.com/igerber/diff-diff/compare/v2.3.2...v2.4.0
[2.3.2]: https://github.com/igerber/diff-diff/compare/v2.3.1...v2.3.2
[2.3.1]: https://github.com/igerber/diff-diff/compare/v2.3.0...v2.3.1
[2.3.0]: https://github.com/igerber/diff-diff/compare/v2.2.1...v2.3.0
[2.2.1]: https://github.com/igerber/diff-diff/compare/v2.2.0...v2.2.1
[2.2.0]: https://github.com/igerber/diff-diff/compare/v2.1.9...v2.2.0
[2.1.9]: https://github.com/igerber/diff-diff/compare/v2.1.8...v2.1.9
[2.1.8]: https://github.com/igerber/diff-diff/compare/v2.1.7...v2.1.8
[2.1.7]: https://github.com/igerber/diff-diff/compare/v2.1.6...v2.1.7
[2.1.6]: https://github.com/igerber/diff-diff/compare/v2.1.5...v2.1.6
[2.1.5]: https://github.com/igerber/diff-diff/compare/v2.1.4...v2.1.5
[2.1.4]: https://github.com/igerber/diff-diff/compare/v2.1.3...v2.1.4
[2.1.3]: https://github.com/igerber/diff-diff/compare/v2.1.2...v2.1.3
[2.1.2]: https://github.com/igerber/diff-diff/compare/v2.1.1...v2.1.2
[2.1.1]: https://github.com/igerber/diff-diff/compare/v2.1.0...v2.1.1
[2.1.0]: https://github.com/igerber/diff-diff/compare/v2.0.3...v2.1.0
[3.4.0]: https://github.com/igerber/diff-diff/compare/v3.3.3...v3.4.0
[3.3.3]: https://github.com/igerber/diff-diff/compare/v3.3.2...v3.3.3
[3.3.2]: https://github.com/igerber/diff-diff/compare/v3.3.1...v3.3.2
[3.3.1]: https://github.com/igerber/diff-diff/compare/v3.3.0...v3.3.1
[3.3.0]: https://github.com/igerber/diff-diff/compare/v3.2.0...v3.3.0
[3.2.0]: https://github.com/igerber/diff-diff/compare/v3.1.3...v3.2.0
[3.1.3]: https://github.com/igerber/diff-diff/compare/v3.1.2...v3.1.3
[3.1.2]: https://github.com/igerber/diff-diff/compare/v3.1.1...v3.1.2
[3.1.1]: https://github.com/igerber/diff-diff/compare/v3.1.0...v3.1.1
[3.1.0]: https://github.com/igerber/diff-diff/compare/v3.0.2...v3.1.0
[3.0.2]: https://github.com/igerber/diff-diff/compare/v3.0.1...v3.0.2
[2.0.3]: https://github.com/igerber/diff-diff/compare/v2.0.2...v2.0.3
[2.0.2]: https://github.com/igerber/diff-diff/compare/v2.0.1...v2.0.2
[2.0.1]: https://github.com/igerber/diff-diff/compare/v2.0.0...v2.0.1
[2.0.0]: https://github.com/igerber/diff-diff/compare/v1.4.0...v2.0.0
[1.4.0]: https://github.com/igerber/diff-diff/compare/v1.3.1...v1.4.0
[1.3.1]: https://github.com/igerber/diff-diff/compare/v1.3.0...v1.3.1
[1.3.0]: https://github.com/igerber/diff-diff/compare/v1.2.1...v1.3.0
[1.2.1]: https://github.com/igerber/diff-diff/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/igerber/diff-diff/compare/v1.1.1...v1.2.0
[1.1.1]: https://github.com/igerber/diff-diff/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/igerber/diff-diff/compare/v1.0.2...v1.1.0
[1.0.2]: https://github.com/igerber/diff-diff/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/igerber/diff-diff/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/igerber/diff-diff/compare/v0.6.0...v1.0.0
[0.6.0]: https://github.com/igerber/diff-diff/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/igerber/diff-diff/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/igerber/diff-diff/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/igerber/diff-diff/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/igerber/diff-diff/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/igerber/diff-diff/releases/tag/v0.1.0
