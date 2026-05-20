# Development TODO

Internal tracking for technical debt, known limitations, and maintenance tasks.

For the public feature roadmap, see [ROADMAP.md](ROADMAP.md).

---

## Known Limitations

Current limitations that may affect users:

| Issue | Location | Priority | Notes |
|-------|----------|----------|-------|
| MultiPeriodDiD wild bootstrap not supported (falls back to analytical) | `estimators.py:1647` | Low | Edge case |
| `predict()` raises NotImplementedError | `estimators.py:890-911` | Low | Rarely needed |

For survey-specific limitations (NotImplementedError paths), see the
[Current Limitations](docs/survey-roadmap.md#current-limitations) section
of survey-roadmap.md.

## Code Quality

### Large Module Files

Target: ideally < 1000 lines per module; modules ≥3000 lines are candidates for splitting, 2000-3000 are monitored, 1000-2000 are accepted as a cohesion / scope trade-off. Updated 2026-05-15.

| File | Lines | Action |
|------|-------|--------|
| `chaisemartin_dhaultfoeuille.py` | 8636 | Consider splitting (per-path / placebos / survey IF / aggregation) |
| `had_pretests.py` | 4951 | Consider splitting (Stute / Yatchew / QUG / joint pretests) |
| `had.py` | 4593 | Consider splitting (continuous / mass-point / event-study / survey paths) |
| `staggered.py` | 3963 | Consider splitting — grew through survey + aggregation features |
| `linalg.py` | 3601 | Consider splitting (vcov surfaces) only if cohesion can be preserved — unified backend; vcov / solver paths are tightly coupled |
| `diagnostic_report.py` | 3380 | Consider splitting (per-method renderers + provenance) |
| `power.py` | 3196 | Consider splitting (power analysis + MDE + sample size) |
| `synthetic_did.py` | 2819 | Monitor — variance methods + survey paths |
| `honest_did.py` | 2785 | Monitor |
| `business_report.py` | 2653 | Monitor — per-method narrative renderers |
| `imputation.py` | 2475 | Monitor |
| `survey.py` | 2466 | Monitor — grew with Phase 6 features |
| `utils.py` | 2396 | Monitor |
| `prep_dgp.py` | 2057 | Monitor |
| `triple_diff.py` | 2053 | Monitor |
| `estimators.py` | 1991 | Acceptable |
| `two_stage.py` | 1985 | Acceptable |
| `chaisemartin_dhaultfoeuille_results.py` | 1981 | Acceptable |
| `prep.py` | 1876 | Acceptable |
| `efficient_did.py` | 1793 | Acceptable |
| `sun_abraham.py` | 1713 | Acceptable |
| `continuous_did.py` | 1682 | Acceptable |
| `results.py` | 1676 | Acceptable |
| `staggered_triple_diff.py` | 1619 | Acceptable |
| `_nprobust_port.py` | 1412 | Acceptable |
| `practitioner.py` | 1402 | Acceptable |
| `trop_global.py` | 1350 | Acceptable |
| `trop_local.py` | 1339 | Acceptable |
| `local_linear.py` | 1332 | Acceptable |
| `wooldridge.py` | 1305 | Acceptable |
| `chaisemartin_dhaultfoeuille_bootstrap.py` | 1175 | Acceptable |
| `bacon.py` | 1144 | Acceptable |
| `pretrends.py` | 1133 | Acceptable |
| `stacked_did.py` | 1050 | Acceptable |
| `conley.py` | 1006 | Acceptable |
| `visualization/` | 4316 | Subpackage (split across 7 files) — OK |

---

### Tech Debt from Code Reviews

Deferred items from PR reviews that were not addressed before merge.

#### Methodology/Correctness

| Issue | Location | PR | Priority |
|-------|----------|----|----------|
| dCDH: Phase 1 per-period placebo DID_M^pl has NaN SE (no IF derivation for the per-period aggregation path). Multi-horizon placebos (L_max >= 1) have valid SE. | `chaisemartin_dhaultfoeuille.py` | #294 | Low |
| dCDH: Survey cell-period allocator's post-period attribution is a library convention, not derived from the observation-level survey linearization. MC coverage is empirically close to nominal on the test DGP; a formal derivation (or a covariance-aware two-cell alternative) is deferred. Documented in REGISTRY.md survey IF expansion Note. | `chaisemartin_dhaultfoeuille.py`, `docs/methodology/REGISTRY.md` | #408 | Medium |
| dCDH: Parity test SE/CI assertions only cover pure-direction scenarios; mixed-direction SE comparison is structurally apples-to-oranges (cell-count vs obs-count weighting). | `test_chaisemartin_dhaultfoeuille_parity.py` | #294 | Low |
| dCDH by_path: survey-aware backward-horizon (`placebo + predict_het + survey_design`) raises `NotImplementedError` because the Binder TSL cell-period allocator's REGISTRY justification is tied to post-period attribution. Backward horizons would put ψ_g mass on a pre-period cell. Deriving the pre-period cell allocator (or adding a covariance-aware two-cell alternative) is deferred to a follow-up methodology PR. | `diff_diff/chaisemartin_dhaultfoeuille.py`, `docs/methodology/REGISTRY.md` | follow-up | Medium |
| CallawaySantAnna: consider materializing NaN entries for non-estimable (g,t) cells in group_time_effects dict (currently omitted with consolidated warning); would require updating downstream consumers (event study, balance_e, aggregation) | `staggered.py` | #256 | Low |
| ImputationDiD dense `(A0'A0).toarray()` scales O((U+T+K)^2), OOM risk on large panels | `imputation.py` | #141 | Medium (deferred — only triggers when sparse solver fails) |
| Multi-absorb weighted demeaning needs iterative alternating projections for N > 1 absorbed FE with survey weights; unweighted multi-absorb also uses single-pass (pre-existing, exact only for balanced panels) | `estimators.py` | #218 | Medium |
| Survey design resolution/collapse patterns are inconsistent across panel estimators — ContinuousDiD rebuilds unit-level design in SE code, EfficientDiD builds once in fit(), StackedDiD re-resolves on stacked data; extract shared helpers for panel-to-unit collapse, post-filter re-resolution, and metadata recomputation | `continuous_did.py`, `efficient_did.py`, `stacked_did.py` | #226 | Low |
| Survey-weighted Silverman bandwidth in EfficientDiD conditional Omega* — `_silverman_bandwidth()` uses unweighted mean/std for bandwidth selection; survey-weighted statistics would better reflect the population distribution but is a second-order refinement | `efficient_did_covariates.py` | — | Low |
| TROP: extend Wave 4's `_setup_trop_data` helper to also cover the duplicated bootstrap resampling loop in `_bootstrap_variance` / `_bootstrap_variance_global` (~40 LoC dedup; mirrors the data-setup helper pattern with a `fit_callable` parameter for the per-draw refit step). | `trop_local.py`, `trop_global.py` | follow-up | Low |
| TripleDifference power auto-routing: `power.simulate_power` ignores `n_periods` for DDD because `_ddd_dgp_kwargs` is hard-coded to the cross-sectional `generate_ddd_data`. Now that `generate_ddd_panel_data` exists (Wave 4), add a new `_EstimatorProfile` registry entry (or extend the existing one) to route to the panel DGP when `n_periods > 2`. | `power.py`, `prep_dgp.py` | follow-up | Low |
| StaggeredTripleDifference R cross-validation: CSV fixtures not committed (gitignored); tests skip without local R + triplediff. Commit fixtures or generate deterministically. | `tests/test_methodology_staggered_triple_diff.py` | #245 | Medium |
| StaggeredTripleDifference R parity: benchmark only tests no-covariate path (xformla=~1). Add covariate-adjusted scenarios and aggregation SE parity assertions. | `benchmarks/R/benchmark_staggered_triplediff.R` | #245 | Medium |
| StaggeredTripleDifference: per-cohort group-effect SEs include WIF (conservative vs R's wif=NULL). Documented in REGISTRY. Could override mixin for exact R match. | `staggered_triple_diff.py` | #245 | Low |
| HonestDiD Delta^RM: uses naive FLCI instead of paper's ARP conditional/hybrid confidence sets (Sections 3.2.1-3.2.2). ARP infrastructure exists but moment inequality transformation needs calibration. CIs are conservative (wider, valid coverage). | `honest_did.py` | #248 | Medium |
| Replicate weight tests use Fay-like BRR perturbations (0.5/1.5), not true half-sample BRR. Add true BRR regressions per estimator family. Existing `test_survey_phase6.py` covers true BRR at the helper level. | `tests/test_replicate_weight_expansion.py` | #253 | Low |
| WooldridgeDiD: QMLE sandwich uses `aweight` cluster-robust adjustment `(G/(G-1))*(n-1)/(n-k)` vs Stata's `G/(G-1)` only. Conservative (inflates SEs). Add `qmle` weight type if Stata golden values confirm material difference. | `wooldridge.py`, `linalg.py` | #216 | Medium |
| WooldridgeDiD: aggregation weights use cell-level n_{g,t} counts. Paper (W2025 Eqs. 7.2-7.4) defines cohort-share weights. Add optional `weights="cohort_share"` parameter to `aggregate()`. | `wooldridge_results.py` | #216 | Medium |
| WooldridgeDiD: optional *efficiency hint* (NOT a canonical-link violation per W2023 Prop 3.1) when method/outcome pairing is sub-optimal — e.g., `method="ols"` on binary data is consistent under QMLE, but `method="logit"` is typically more efficient. The original framing in this row as a "canonical link requirement" tied to Prop 3.1 was incorrect: Wooldridge (2023) Table 1 lists Gaussian/OLS for "any response" and logistic-Bernoulli for "binary OR fractional". A useful hint exists (efficiency), but should not be framed as a methodology violation. See PR #453 R1 review for the corrected reading. | `wooldridge.py` | #216 | Low |
| WooldridgeDiD: Stata `jwdid` golden value tests — add R/Stata reference script and `TestReferenceValues` class. | `tests/test_wooldridge.py` | #216 | Medium |
<!-- The PreTrendsPower R parity row (PR-C, 2026-05-19) and the four PR-A-tagged PreTrendsPower rows (CS/SA Σ_22 fidelity, helper `violation_weights`, custom-weight persistence, linear γ-unit MDV; resolved in PR-B 2026-05-18) are all closed — see CHANGELOG.md [Unreleased] Added/Changed/Fixed entries for the new behavior. -->
| PreTrendsPower: CS/SA `anticipation=1` R-parity fixture. The PR-C R-parity goldens cover NIS power + γ_p MDV at `atol=1e-4` on four shifted-grid / regular / irregular / K=1 fixtures, but R `pretrends` has no anticipation parameter so the Python-side `_extract_pre_period_params` anticipation filter (`if t < _pre_cutoff` in `pretrends.py` lines 1138-1150 for CS; mirror in SA branch) is not R-parity-locked. Build a synthetic `CallawaySantAnnaResults` (or `SunAbrahamResults`) with `anticipation=1` and a t=-1 event-study entry that should be filtered before reaching `_compute_power_nis`, then assert the resulting γ_p matches R's `slope_for_power()` on the K=4 shifted-grid fixture. Existing PR-B MC-based tests (`TestPretrendsPropositions`) and full-VCV tests (`TestPretrendsCovarianceSource`) already cover the filter mechanically; this would close the loop against R. | `tests/test_methodology_pretrends.py::TestPretrendsParityR`, `benchmarks/R/generate_pretrends_golden.R` | PR-C follow-up | Low |


| Thread `vcov_type` (classical / hc1 / hc2 / hc2_bm) through the 7 standalone estimators that expose `cluster=` but not yet `vcov_type=`: `CallawaySantAnna`, `ImputationDiD`, `TwoStageDiD`, `TripleDifference`, `StackedDiD`, `WooldridgeDiD`, `EfficientDiD`. Phase 1a added the chain to DiD/MPD/TWFE; Phase 1b PR 1/8 added `SunAbraham` (this row tracks the remaining 7). | multiple | Phase 1b | Medium |
| Extend `SunAbraham` with `vcov_type="conley"` (Conley spatial-HAC) as a first-class feature: thread `conley_coords` / `conley_cutoff_km` / `conley_metric` / `conley_kernel` / `conley_time` / `conley_unit` / `conley_lag_cutoff` through `_fit_saturated_regression`. Phase 1b PR 1/8 deferred this; SA currently rejects `vcov_type="conley"` at `__init__` with a deferral message. | `diff_diff/sun_abraham.py` | follow-up | Medium |
| Harmonize SunAbraham's HC1 within-transform finite-sample correction with `fixest::sunab()`. SA's `solve_ols` applies `n / (n - k_dm)` (within-transform columns only); fixest applies `n / (n - k_total)` (counts absorbed FE). SE values differ by ~1-2% on typical panel sizes (documented in REGISTRY.md "Deviation from R"; pinned at `atol=5e-3` in `tests/test_methodology_sun_abraham.py`). Either thread `df_adjustment` into the vcov scaling or document as an intentional difference. | `diff_diff/sun_abraham.py`, `diff_diff/linalg.py::compute_robust_vcov` | follow-up | Low |
<!-- Rows 104-105 LIFTED 2026-05-20 via the clubSandwich WLS-CR2 port. The diff-diff
     form matches clubSandwich's specific algebra (W not sqrt(W), W^2 in bias term,
     unweighted residuals) rather than the textbook PT2018 transform-once derivation
     which historically diverged by 0.5-30%. Pinned at atol=1e-10 on 6 weighted
     scenarios in benchmarks/data/clubsandwich_cr2_golden.json. -->
| ~~Weighted one-way Bell-McCaffrey (`vcov_type="hc2_bm"` + `weights`, no cluster)~~ — LIFTED via clubSandwich WLS-CR2 port. Routes through `_compute_cr2_bm_contrast_dof` with singleton-cluster reduction (each obs is its own cluster). See REGISTRY.md Phase 1a row for the algebra. | `linalg.py` | LIFTED | — |
| ~~Weighted CR2 Bell-McCaffrey cluster-robust (`vcov_type="hc2_bm"` + `cluster_ids` + `weights`)~~ — LIFTED via clubSandwich WLS-CR2 port. `_compute_cr2_bm` now accepts `weights=` and threads through clubSandwich's specific WLS-CR2 algebra. | `linalg.py::_compute_cr2_bm` | LIFTED | — |
| `TwoWayFixedEffects(vcov_type in {"hc2","hc2_bm"})` with replicate-weight survey designs raises `NotImplementedError` (`twfe.py:~233`). The replicate path re-demeans per replicate (re-demeaning depends on the per-replicate weight vector), which doesn't compose with the full-dummy HC2/HC2-BM build — a correct implementation would need per-replicate full-dummy refit. Workaround: use `vcov_type="hc1"` for replicate-weight CR1. | `twfe.py::fit` | follow-up | Low |
| TWFE's HC2/HC2-BM inline full-dummy build (`twfe.py:280-315`) duplicates the dummy-construction logic in `DifferenceInDifferences(fixed_effects=...)` (`estimators.py:478-486`). Extract a shared helper (or delegate TWFE's HC2/HC2-BM path to DiD's `fixed_effects=` branch, with TWFE-specific cluster default threading) to reduce drift risk on FE naming, survey behavior, and result-surface conventions. Substantive refactor — touches both estimators. | `twfe.py::fit`, `estimators.py::DifferenceInDifferences.fit` | follow-up | Low |
| Unify Rust local-method `estimate_model` solver path to `solve_wls_svd` (the same SVD helper used by the global-method since PR #348) for sub-1e-14 bootstrap SE parity. Current local-method bootstrap parity test (`tests/test_rust_backend.py::TestTROPRustEdgeCaseParity::test_bootstrap_seed_reproducibility_local`) passes at `atol=1e-5` — the residual ~1e-7 gap is roundoff between Rust's `estimate_model` matrix factorization and numpy's `lstsq`, which accumulates differently across per-replicate bootstrap fits. Main-fit ATT parity is regime-dependent (`atol=1e-14` for `lambda_nn=inf`, `atol=1e-10` for finite `lambda_nn` — see `test_local_method_main_fit_parity`); the bootstrap gap is a same-solver-path roundoff concern and not a user-visible correctness bug. | `rust/src/trop.rs::estimate_model`, `rust/src/linalg.rs::solve_wls_svd` | follow-up | Low |
| Rust multiplier-bootstrap weight RNG (`generate_bootstrap_weights_batch` in `rust/src/bootstrap.rs:9-10, 57-75`) uses `Xoshiro256PlusPlus::seed_from_u64(seed + i)` per row for Rademacher/Mammen/Webb generation. If any Python caller (SDID / efficient-DiD multiplier bootstrap) has a numpy-canonical equivalent, the two backends likely diverge under the same seed. Audit Python callers (`diff_diff/sdid.py`, `diff_diff/efficient_did_bootstrap.py`, `diff_diff/bootstrap_utils.py::generate_bootstrap_weights_batch_numpy`) for parity-test gaps. Same fix shape as TROP RNG parity (PR #354): pre-generate weights in Python via numpy and pass them to Rust through PyO3. | `rust/src/bootstrap.rs`, `diff_diff/bootstrap_utils.py` | follow-up | Medium |
| `bias_corrected_local_linear`: extend golden parity to `kernel="triangular"` and `kernel="uniform"` (currently epa-only; all three kernels share `kernel_W` and the `lprobust` math, so parity is expected but not separately asserted). | `benchmarks/R/generate_nprobust_lprobust_golden.R`, `tests/test_bias_corrected_lprobust.py` | Phase 1c | Low |
| `bias_corrected_local_linear`: expose `vce in {"hc0", "hc1", "hc2", "hc3"}` on the public wrapper once R parity goldens exist (currently raises `NotImplementedError`). The port-level `lprobust` and `lprobust_res` already support all four; expanding the public surface requires a golden generator for each hc mode and a decision on hc2/hc3 q-fit leverage (R reuses p-fit `hii` for q-fit residuals; whether to match that or stage-match deserves a derivation before the wrapper advertises CCT-2014 conformance). | `diff_diff/local_linear.py::bias_corrected_local_linear`, `benchmarks/R/generate_nprobust_lprobust_golden.R`, `tests/test_bias_corrected_lprobust.py` | Phase 1c | Medium |
| `bias_corrected_local_linear`: support `weights=` once survey-design adaptation lands. nprobust's `lprobust` has no weight argument so there is no parity anchor; derivation needed. | `diff_diff/local_linear.py`, `diff_diff/_nprobust_port.py::lprobust` | Phase 1c | Medium |
| `bias_corrected_local_linear`: support multi-eval grid (`neval > 1`) with cross-covariance (`covgrid=TRUE` branch of `lprobust.R:253-378`). Not needed for HAD but useful for multi-dose diagnostics. | `diff_diff/_nprobust_port.py::lprobust` | Phase 1c | Low |
| Clustered-DGP parity: Phase 1c's DGP 4 uses manual `h=b=0.3` to sidestep an nprobust-internal singleton-cluster bug in `lpbwselect.mse.dpi`'s pilot fits. Once nprobust ships a fix (or we derive one independently), add a clustered-auto-bandwidth parity test. | `benchmarks/R/generate_nprobust_lprobust_golden.R` | Phase 1c | Low |
| `HeterogeneousAdoptionDiD` joint cross-horizon covariance on event study: per-horizon SEs use INDEPENDENT sandwiches in Phase 2b (paper-faithful pointwise CIs per Pierce-Schott Figure 2). A follow-up could derive an IF-based stacking of per-horizon scores for joint cross-horizon inference (needed for joint hypothesis tests across event-time horizons). Block-bootstrap is a reasonable alternative. | `diff_diff/had.py::_fit_event_study` | Phase 2b | Low |
| `HeterogeneousAdoptionDiD` event-study staggered-timing beyond last cohort: Phase 2b auto-filters staggered panels to the last cohort per paper Appendix B.2. Earlier-cohort treatment effects are not identified by HAD; redirecting to `ChaisemartinDHaultfoeuille` / `did_multiplegt_dyn` is the paper's prescription. A full staggered HAD would require a different identification path (out of paper scope). | `diff_diff/had.py::_validate_had_panel_event_study` | Phase 2b | Low |
| `HeterogeneousAdoptionDiD` joint cross-horizon analytical covariance on the weighted event-study path: Phase 4.5 B ships multiplier-bootstrap sup-t simultaneous CIs on the weighted event-study path but pointwise analytical variance is still independent across horizons. A follow-up could derive the full H × H analytical covariance from the per-horizon IF matrix (`Psi.T @ Psi` under survey weighting) for an analytical alternative to the bootstrap. Would also let the unweighted event-study path ship a sup-t band. | `diff_diff/had.py::_fit_event_study` | follow-up | Low |
| `HeterogeneousAdoptionDiD` unweighted event-study sup-t band: Phase 4.5 B ships sup-t only on the WEIGHTED event-study path (to preserve pre-PR bit-exact output on unweighted). Extending sup-t to unweighted event-study (either via the multiplier bootstrap with unit-level iid multipliers or via analytical joint cross-horizon covariance) is a symmetric follow-up. | `diff_diff/had.py::_fit_event_study` | follow-up | Low |
| `HeterogeneousAdoptionDiD` survey-aware support-endpoint test (research, not engineering): if the academic literature ever publishes a calibrated support-infimum test under complex sampling — combining endpoint-estimation EVT (Hall 1982, Aarssen-de Haan 1994, Hall-Wang 1999) with survey-aware functional CLTs for the empirical process (Boistard-Lopuhaä-Ruiz-Gazen 2017, Bertail-Chautru-Clémençon 2017) and tail-empirical-process theory (Drees 2003) — Phase 4.5 C0's permanent NotImplementedError on `qug_test(..., survey=...)` / `weights=` can be revisited and the bridge implemented against the published recipe. See `docs/methodology/REGISTRY.md` § "QUG Null Test" — Note (Phase 4.5 C0) for the decision rationale and the research-direction sketch. | `diff_diff/had_pretests.py::qug_test` | Phase 4.5 C0 (2026-04, decision shipped) | Low |
| `HeterogeneousAdoptionDiD` survey-aware pretests Phase 4.5 C still-open follow-ups: (a) **replicate-weight designs** (BRR/Fay/JK1/JKn/SDR) — the per-replicate weight-ratio rescaling for the OLS-on-residuals refit step is not covered by the multiplier-bootstrap composition; each linearity-family helper raises `NotImplementedError` on `survey.replicate_weights is not None`. (b) **`lonely_psu='adjust'` + singleton-strata** on the Stute family — the pseudo-stratum centering transform has not been derived for the Stute CvM functional (same gap as the HAD sup-t deviation at REGISTRY:2382). Stratified-design support on the Stute family **SHIPPED** in the Phase 4.5 C strata extension PR (within-stratum demean + sqrt(n_h/(n_h-1)) Bessel rescale on PSU multipliers via `bootstrap_utils.apply_stratum_centering`; see REGISTRY § "Note (Stute stratified survey-bootstrap calibration)"). Phase 4.5 C now ships **pweight + PSU + FPC + strata** support via PSU-level Mammen multiplier bootstrap (Stute family) + closed-form weighted variance components (Yatchew). Replicate-weight pretests = bootstrap-composition work; lonely_psu='adjust'+singleton on Stute = pseudo-stratum centering derivation. | `diff_diff/had_pretests.py` | Phase 4.5 C follow-up | Low |
| `HeterogeneousAdoptionDiD` Phase 4.5: weight-aware auto-bandwidth MSE-DPI selector. Phase 4.5 A ships weighted `lprobust` with an unweighted DPI selector; users who want a weight-aware bandwidth must pass `h`/`b` explicitly. Extending `lpbwselect_mse_dpi` to propagate weights through density, second-derivative, and variance stages is ~300 LoC of methodology and was out of scope. | `diff_diff/_nprobust_port.py::lpbwselect_mse_dpi` | Phase 4.5 | Low |
| `HeterogeneousAdoptionDiD` Phase 4.5 C: replicate-weight SurveyDesigns (BRR / Fay / JK1 / JKn / SDR) on the continuous-dose paths. Phase 4.5 A raises `NotImplementedError` on replicate designs in `_aggregate_unit_resolved_survey`. Rao-Wu-style replicate bootstrap for HAD paths requires deriving the per-replicate weight-ratio rescaling for the local-linear intercept IF. | `diff_diff/had.py::_aggregate_unit_resolved_survey` | Phase 4.5 C | Low |
| `HeterogeneousAdoptionDiD` mass-point: `vcov_type in {"hc2", "hc2_bm"}` raises `NotImplementedError` pending a 2SLS-specific leverage derivation. The OLS leverage `x_i' (X'X)^{-1} x_i` is wrong for 2SLS; the correct finite-sample correction uses `x_i' (Z'X)^{-1} (...) (X'Z)^{-1} x_i`. Needs derivation plus an R / Stata (`ivreg2 small robust`) parity anchor. | `diff_diff/had.py::_fit_mass_point_2sls` | Phase 2a | Medium |
| `HeterogeneousAdoptionDiD` survey-design API consolidation, **next minor bump**: drop the deprecated `survey=` and `weights=` kwargs on all 8 HAD surfaces (`HeterogeneousAdoptionDiD.fit`, `did_had_pretest_workflow`, `qug_test`, `stute_test`, `yatchew_hr_test`, `stute_joint_pretest`, `joint_pretrends_test`, `joint_homogeneity_test`); only `survey_design=` remains. Also fold the legacy back-end `weights=` paths (e.g. `_aggregate_unit_weights` ad-hoc routing) into the unified `_resolve_survey_for_fit`-driven path. The `_make_trivial_resolved` underscore alias on `survey.py` stays (one-line, harmless). DeprecationWarning ships in this PR; the removal PR is ~50 LoC of cleanup. | `diff_diff/had.py`, `diff_diff/had_pretests.py` | next minor bump | Medium |
| `HeterogeneousAdoptionDiD` continuous paths: thread `cluster=` through `bias_corrected_local_linear` (Phase 1c's wrapper already supports cluster; Phase 2a ignores it with a `UserWarning` on the continuous path to keep scope tight). | `diff_diff/had.py`, `diff_diff/local_linear.py` | Phase 2a | Low |
| `HeterogeneousAdoptionDiD` `trends_lin × survey_design` follow-up: per-group linear-trend slope under survey weighting (weighted slope estimator? per-PSU slope?) is not derived from the paper. PR #389 raises `NotImplementedError` on the combination across all 3 trends_lin surfaces. If user demand emerges, derive the weighted variant and lift the gate. | `diff_diff/had.py::HeterogeneousAdoptionDiD.fit`, `diff_diff/had_pretests.py::joint_pretrends_test`, `diff_diff/had_pretests.py::joint_homogeneity_test` | follow-up | Low |
| `HeterogeneousAdoptionDiD` Stute family Stata-bridge parity: PR #389 R-parity covers the full HAD fit + Yatchew surfaces but skips Stute family (`stute_test`, `stute_joint_pretest`, `joint_pretrends_test`, `joint_homogeneity_test`) because no R `Stutetest` package exists publicly (chaisemartinPackages publishes only the Stata `stute_test` module; the paper cites a 2024c R Stutetest module that is not on GitHub or CRAN). Stata-bridge parity would add `benchmarks/stata/generate_stute_golden.do` + a Stata installation requirement. Low priority unless user demand emerges. | `benchmarks/stata/`, `tests/test_stute_test_parity.py` | follow-up | Low |
| `HeterogeneousAdoptionDiD` Phase 3 Stute performance: Appendix D vectorized matrix form replaces the per-iteration OLS refit with a single precomputed `M = I - X(X'X)^{-1}X'` applied to `eps * eta`. Functionally identical, ~2x faster. Shipped literal-refit form in Phase 3 to match paper text and keep reviewer surface small. | `diff_diff/had_pretests.py::stute_test` | Phase 3 | Low |
| `HeterogeneousAdoptionDiD` Phase 3 R-parity: Phase 3 ships coverage-rate validation on synthetic DGPs (not tight point parity against `chaisemartin::stute_test` / `yatchew_test`). Tight numerical parity requires aligning bootstrap seed semantics and `B` across numpy/R and is deferred. | `tests/test_had_pretests.py` | Phase 3 | Low |
| `HeterogeneousAdoptionDiD` Phase 3 nprobust bandwidth for Stute: some Stute variants on continuous regressors use nprobust-style optimal bandwidth selection. Phase 3 uses OLS residuals from a 2-parameter linear fit (no bandwidth selection). nprobust integration is a future enhancement; not in paper scope. | `diff_diff/had_pretests.py::stute_test` | Phase 3 | Low |
| `HeterogeneousAdoptionDiD` Phase 4: Pierce-Schott (2016) replication harness; reproduce paper Figure 2 values and Table 1 coverage rates. **Waived in tracker-promotion PR (2026-05-20):** R parity at `atol=1e-8` on the same 3 DGPs (`tests/test_did_had_parity.py`) is a strictly stronger correctness anchor than reproducing Figure 2's pointwise CIs on the LBD-restricted PNTR panel; paper Section 5.2 self-acknowledges NP estimators too noisy to be informative there. Table 1 coverage-rate MC would re-verify the CCF asymptotic coverage already pinned by R parity (Python ≡ R ≡ paper). See REGISTRY HAD Deviations Notes #3 / #4 for full scope-caveat statements. Re-open if user demand emerges for an empirical-application replication harness. | `benchmarks/`, `tests/` | Phase 2a | Low |
| `HeterogeneousAdoptionDiD` `covariates=` kwarg with Theorem 6 multivariate-covariate extension: current behavior is a Python `TypeError` (the `covariates=` kwarg is absent from `HAD.fit()` signature) — fail-closed, but doesn't surface the Theorem 6 future-work pointer to the user. Add an explicit `**kwargs`-trap with `NotImplementedError` and a Theorem 6 / `nprobust` multivariate-NP-regression pointer. ~10 LoC follow-up. | `diff_diff/had.py::HeterogeneousAdoptionDiD.fit` | follow-up | Low |
| `HeterogeneousAdoptionDiD` extensive-margin / positive-mass-of-untreated warning on the main `fit()` path. Paper recommends warning users with positive zero-dose mass that standard DiD may be more appropriate. Currently surfaced via the `qug_test()` zero-dose `UserWarning` (which only fires when the user runs pre-tests). Add a fit-time `UserWarning` when the panel's post-period dose contains a non-trivial fraction at exactly zero, with a "consider running standard DiD" pointer. Paper-review checklist L191 in `dechaisemartin-2026-review.md` left unchecked pending this addition. | `diff_diff/had.py::HeterogeneousAdoptionDiD.fit` | follow-up | Low |
| `HeterogeneousAdoptionDiD` time-varying dose on event study: Phase 2b REJECTS panels where `D_{g,t}` varies within a unit for `t >= F` (the aggregation uses `D_{g, F}` as the single regressor for all horizons, paper Appendix B.2 constant-dose convention). A follow-up PR could add a time-varying-dose estimator for these panels; current behavior is front-door rejection with a redirect to `ChaisemartinDHaultfoeuille`. | `diff_diff/had.py::_validate_had_panel_event_study` | Phase 2b | Low |
| `HeterogeneousAdoptionDiD` repeated-cross-section support: paper Section 2 defines HAD on panel OR repeated cross-section, but Phase 2a is panel-only. RCS inputs (disjoint unit IDs between periods) are rejected by the balanced-panel validator with the generic "unit(s) do not appear in both periods" error. A follow-up PR will add an RCS identification path based on pre/post cell means (rather than unit-level first differences), with its own validator and a distinct `data_mode` / API surface. | `diff_diff/had.py::_validate_had_panel`, `diff_diff/had.py::_aggregate_first_difference` | Phase 2a | Medium |
| SyntheticDiD: bootstrap cross-language parity anchor against R's default `synthdid::vcov(method="bootstrap")` (refit; rebinds `opts` per draw) or Julia `Synthdid.jl::src/vcov.jl::bootstrap_se` (refit by construction). Same-library validation (placebo-SE tracking, AER §6.3 MC truth) is in place; a cross-language anchor is desirable to bolster the methodology contract. Julia is the cleanest target — minimal wrapping work and refit-native vcov. Tolerance target: 1e-6 on Monte Carlo samples (different BLAS + RNG paths preclude 1e-10). The R-parity fixture from the previous release was deleted because it pinned the now-removed fixed-weight path. | `benchmarks/R/`, `benchmarks/julia/`, `tests/` | follow-up | Low |
| Conley + survey weights / `survey_design`. Score-reweighted meat `s_i = w_i · X_i · ε_i` is mechanical, but PSU clustering interaction with the spatial kernel and replicate-weights variance under spatial correlation are non-trivial (Bertanha-Imbens 2014 covers cluster-sample but not the explicit Conley case). Phase 5 of the spillover-conley initiative; paper review prerequisite. Currently raises `NotImplementedError` at the linalg validator. | `linalg.py::_validate_vcov_args` | Phase 5 (spillover-conley) | Medium |
| `SyntheticDiD(vcov_type="conley")` support. Currently raises `TypeError` at `__init__` because SyntheticDiD uses `variance_method ∈ {bootstrap, jackknife, placebo}` rather than the analytical sandwich that Conley plugs into. Wiring would require either reimplementing an analytical sandwich path for SyntheticDiD or designing a spatial-block bootstrap (new methodology, Politis-Romano 1994 territory). | `synthetic_did.py::SyntheticDiD` | follow-up (spillover-conley) | Low |
| `SpilloverDiD(survey_design=...)` replicate-weight variance (BRR / Fay / JK1 / JKn / SDR). Wave E.1 ships Taylor-linearization only. Per Gerber (2026) Appendix A, the IF-reweighting shortcut does NOT apply to TwoStageDiD-class estimators because `gamma_hat` is weight-sensitive; correct support requires per-replicate full re-fit of stage 1 and stage 2 (200+ LoC of test surface beyond E.1). | `spillover.py::SpilloverDiD.fit`, `survey.py::compute_replicate_refit_variance` | follow-up | Low |
| `SpilloverDiD(survey_design=...)` subpopulation preservation (Wave E.3). Wave E.1's `finite_mask` block physically removes zero-weight rows that lose stage-1 FE support, so `SurveyDesign.subpopulation()`-derived designs see `n_psu` / `df_survey` / Binder centering recomputed on the reduced fit sample rather than the full domain design. Standard domain-estimation practice (R `survey::svyrecvar` on a `subset()` design) preserves the original PSU/strata counts and treats out-of-domain rows as zero-score padding. Fix requires separating fit-sample alignment (Psi array) from design-level bookkeeping: preserve a full-design `resolved_survey` for inference metadata + zero-pad dropped zero-weight rows' IF contribution. Add `SurveyDesign.subpopulation()` regression test to lock the contract. | `spillover.py::SpilloverDiD.fit`, `two_stage.py::_compute_binder_tsl_meat` | follow-up (Wave E.3) | Medium |
| `SpilloverDiD(vcov_type="conley", conley_lag_cutoff > 0, survey_design=...)` serial Bartlett HAC composition. Wave E.2 ships the panel-aware `conley_lag_cutoff = 0` case ("within-period spatial only" — `sum_t sum_h M_h_t` per `tests/test_spillover.py::TestSpilloverDiDWaveE2ConleySurveyDesign::test_b_panel_aware_per_period_sum_invariant`) and raises `NotImplementedError` upfront at `spillover.py:fit` on `conley_lag_cutoff > 0`. The serial Bartlett component (within-unit / within-PSU temporal HAC at lag ≤ L) needs to compose with the panel-aware stratified-Conley spatial sandwich — the natural addition is `meat_serial = sum_g sum_{|t-s|<=L, t!=s} (1 - |t-s|/(L+1)) * (S_psu_t[g] - S_bar_h_t)(S_psu_s[g] - S_bar_h_s)'` per PSU, summed across all PSUs in each stratum, with appropriate Binder FPC scaling — plus a methodology call on whether to include cross-period spatial pairs in the serial term. Regression goldens vs the cross-sectional limit (lag=0, which is now the shipped path). | `spillover.py::SpilloverDiD.fit`, `two_stage.py::_compute_stratified_conley_meat` | follow-up (Wave E.2 follow-up) | Medium |
| `SpilloverDiD(ring_method="count")` extension. Currently only the nearest-treated-ring specification is exposed. Count-of-treated-in-ring (paper Section 3.2 end) is methodologically supported by Butts but re-introduces functional-form dependence; expose with an explicit kwarg gate and documentation warning. | `spillover.py::SpilloverDiD.fit` | follow-up | Low |
| `SpilloverDiD` data-driven `d_bar` selection (Butts 2021b / Butts 2023 JUE Insight cross-validation). | `spillover.py::SpilloverDiD` | follow-up | Low |
| `SpilloverDiD` T22 TVA tutorial (`docs/tutorials/22_spillover_did.ipynb`): synthetic TVA-style DGP reproducing Butts (2021) Section 4 Table 1 Panel A bias-correction direction (~40% understatement). Split from the methodology PR per user-confirmed scope split (2026-05-15). | `docs/tutorials/`, `tests/test_t22_*_drift.py` | follow-up (Wave B) | Medium |
| Extend `TwoStageDiD` with Conley vcov as a first-class feature (mirrors Wave A's TWFE/MPD/DiD extension). Currently `TwoStageDiD.__init__` lacks `vcov_type` / `conley_*` kwargs; `SpilloverDiD` works around this by threading Conley directly via `solve_ols` at stage 2. Promoting Conley to TwoStageDiD's API removes the workaround and lets non-spillover users access Conley + Gardner two-stage. | `diff_diff/two_stage.py` | follow-up | Medium |
| `SpilloverDiD` sparse cKDTree path for the staggered nearest-treated-distance helper (mirrors the static helper's sparse branch). Currently `_compute_nearest_treated_distance_staggered` always builds dense `(n_units, n_treated_by_onset)` pairwise distance matrices per cohort; on large staggered panels with many cohorts this is avoidable memory/runtime. Add a sparse k-d-tree branch analogous to `_compute_nearest_treated_distance_sparse`, gated on `n > _CONLEY_SPARSE_N_THRESHOLD`. | `spillover.py::_compute_nearest_treated_distance_staggered` | follow-up (Wave B) | Low |
| `SpilloverDiDResults` in `DiagnosticReport` dispatch tables. Wave C event-study emits a TwoStageDiD-compatible `event_study_effects: Dict[int, Dict]` alias that `plot_event_study` consumes via the new `reference_period` attribute fallback in `_extract_plot_data`, but `SpilloverDiDResults` is NOT registered in `DiagnosticReport`'s `_APPLICABILITY` / `_PT_METHOD` tables — so `DiagnosticReport(spillover_result)` doesn't currently route to event-study diagnostics. Registering requires (a) deciding which diagnostics apply (parallel trends, pre-trends power, heterogeneity, design-effect) AND (b) adding an end-to-end test. | `diff_diff/diagnostic_report.py::_APPLICABILITY`, `_PT_METHOD` | follow-up (Wave C) | Low |

#### Performance

| Issue | Location | PR | Priority |
|-------|----------|----|----------|
| ImputationDiD event-study SEs recompute full conservative variance per horizon (should cache A0/A1 factorization) | `imputation.py` | #141 | Low |
| Rust faer SVD ndarray-to-faer conversion overhead (minimal vs SVD cost) | `rust/src/linalg.rs:67` | #115 | Low |
| Unrelated label events (e.g., adding `bug` label) re-trigger CI workflows when `ready-for-ci` is already present; filter `labeled`/`unlabeled` events to only `ready-for-ci` transitions | `.github/workflows/rust-test.yml`, `notebooks.yml`, `docs-tests.yml` | #269 | Low |
| `bread_inv` as a performance kwarg on `compute_robust_vcov` to avoid re-inverting `(X'WX)` when the caller already has it. Deferred from Phase 1a for scope. HC2 and HC2+BM both need the bread inverse, so a shared hint would save one `np.linalg.solve` per sandwich. | `linalg.py::compute_robust_vcov` | Phase 1a | Low |
| MPD cluster+hc2_bm path computes CR2 precomputes twice — once via `solve_ols` → `_compute_cr2_bm` for vcov + per-coefficient DOF, then again via `_compute_cr2_bm_contrast_dof` from `MultiPeriodDiD.fit()` for the post-period-average contrast DOF. Both rebuild `H = X bread_inv X'`, the residual-maker `M`, and the per-cluster `A_g = (I - H_gg)^{-1/2}` matrices. O(n²k) redundant work; acceptable for typical cluster-robust DiD panel sizes (n ≤ a few thousand). Fix would plumb the contrast DOF through the existing CR2 vcov path (intrusive API change) or share the precomputes via a cached helper. | `linalg.py::_compute_cr2_bm_contrast_dof`, `estimators.py::MultiPeriodDiD.fit` | follow-up | Low |
| Rust-backend HC2 implementation. Current Rust path only supports HC1; HC2 and CR2 Bell-McCaffrey fall through to the NumPy backend. For large-n fits this is noticeable. | `rust/src/linalg.rs` | Phase 1a | Low |
| CR2 Bell-McCaffrey DOF uses a naive `O(n² k)` per-coefficient loop over cluster pairs. Pustejovsky-Tipton (2018) Appendix B has a scores-based formulation that avoids the full `n × n` `M` matrix. Switch when a user hits a large-`n` cluster-robust design. | `linalg.py::_compute_cr2_bm` | Phase 1a | Low |

#### Testing/Docs

| Issue | Location | PR | Priority |
|-------|----------|----|----------|
| R comparison tests spawn separate `Rscript` per test (slow CI) | `tests/test_methodology_twfe.py:294` | #139 | Low |
| CS R helpers hard-code `xformla = ~ 1`; no covariate-adjusted R benchmark for IRLS path | `tests/test_methodology_callaway.py` | #202 | Low |
| Doc-snippet smoke tests only cover `.rst` files; `.txt` AI guides outside CI validation | `tests/test_doc_snippets.py` | #239 | Low |
| Add CI validation for `docs/doc-deps.yaml` integrity (stale paths, unmapped source files) | `docs/doc-deps.yaml` | #269 | Low |
| SyntheticDiD: rename internal `placebo_effects` variable to `variance_effects` (or `resampled_effects`). Misleading name across the placebo/bootstrap/jackknife dispatch paths — holds three different contents depending on variance method. Low-risk refactor; user-facing field rename should preserve `placebo_effects` as a deprecated alias for one release. | `synthetic_did.py`, `results.py` | follow-up | Medium |
| AI review CI: pin workflow contract via test (uses `openai/codex-action@v1`, passes `prompt-file`, reads `steps.run_codex.outputs.final-message`, preserves diff-exclude paths and comment markers). Currently only the wrapper-tag and closing-tag-escape strings are asserted. | `tests/test_openai_review.py`, `.github/workflows/ai_pr_review.yml` | #416 | Low |
| `TestWorkflowDoesNotExecutePRHeadCode` (CodeQL #14 dismissal guard) does not model: `bash <script>` / `sh <script>` / `./<script>` / `source <script>` / `. <script>` direct shell-script execution; multi-line `python3 -c` bodies (line-by-line shlex can't reassemble across newlines — the workflow's 5 sanitizer bodies are exempt by invisibility); shell-variable-expansion indirection (`SCRIPT="$X"; python3 "$SCRIPT"`); `eval`; `find -exec`; `xargs -I {}`. Each represents a path by which PR-head bytes COULD execute without the test failing. The guard catches accidental regressions of common forms (16 tests covering pip/npm/cargo/maturin/etc. installs, python file exec, bash -c indirection with compound flags, env-var prefixes, line continuations, subshells/brace groups, single-line python -c, write-overwrites of allowlisted /tmp paths). Closing the residuals would require multi-line shell parsing with command-substitution awareness + script-execution allowlists — significant work for diminishing return given the dismissal's primary defense is the documented threat model on the alert and in `.github/workflows/ai_pr_review.yml` comment block. | `tests/test_openai_review.py`, `.github/workflows/ai_pr_review.yml` | #436 | Low |
| Render `docs/methodology/REPORTING.md` and `docs/methodology/REGISTRY.md` as in-site Sphinx pages so cross-references can use `:doc:` instead of off-site GitHub `blob/main` URLs. Current state (#410 fix-audit-r2) restores navigable links via `blob/main`, but stable-docs readers can land on a different revision than the package version they are reading. Two viable paths: (a) add `myst-parser` to `docs/conf.py` extensions + docs extras and link with `:doc:`, or (b) convert both files to `.rst`. | `docs/conf.py`, `docs/api/business_report.rst`, `docs/api/diagnostic_report.rst`, `docs/tutorials/18_geo_experiments.ipynb`, `docs/tutorials/19_dcdh_marketing_pulse.ipynb` | follow-up | Low |

---

### Prioritized Tech-Debt Backlog

Ordered paydown view across the tables above. Tier A → D is by effort × risk, not severity — every item here already carries its own `Low / Medium` priority in the source-of-truth tables. The intent is to give a flat ordering to draw from wave-by-wave without re-litigating priority each time. Anchors point to the location reference of the originating row.

#### Tier A — Quick wins (≤1 day, ≤3 CI rounds expected)

- WooldridgeDiD: optional efficiency hint when method/outcome pairing is sub-optimal (NOT a canonical-link violation per W2023 Prop 3.1 — see Methodology/Correctness row for the corrected framing)

(SyntheticDiD `placebo_effects` → `variance_effects` rename moved to Tier B — the user-facing field rename + one-release deprecation alias is too large for ≤1 day / ≤3 CI rounds.)

#### Tier B — Mid-size methodology (5-10 CI rounds expected, per memory cascade priors)

- Thread `vcov_type` through 8 standalone estimators: `CallawaySantAnna`, `SunAbraham`, `ImputationDiD`, `TwoStageDiD`, `TripleDifference`, `StackedDiD`, `WooldridgeDiD`, `EfficientDiD` (none currently expose `self.vcov_type`)
- SyntheticDiD: rename internal `placebo_effects` → `variance_effects` AND public `placebo_effects` field with deprecation alias retained for one release (`synthetic_did.py`, `results.py`)
- StaggeredTripleDifference R parity: commit CSV fixtures + add covariate-adjusted scenarios + aggregation-SE assertions (`tests/test_methodology_staggered_triple_diff.py`, `benchmarks/R/benchmark_staggered_triplediff.R`)
- StaggeredTripleDifference: per-cohort group-effect SE WIF override for exact R `triplediff` match (`staggered_triple_diff.py`)
- WooldridgeDiD: QMLE Stata-parity `qmle` weight type + Stata golden values (`wooldridge.py`, `linalg.py`, `tests/test_wooldridge.py`)
- WooldridgeDiD: optional `weights="cohort_share"` on `aggregate()` (`wooldridge_results.py`)
- HAD survey-design API consolidation: drop deprecated `survey=`/`weights=` kwargs (`had.py`, `had_pretests.py`; gated on next minor bump)
- Survey-design resolution / collapse helper extraction across `continuous_did.py`, `efficient_did.py`, `stacked_did.py`
- dCDH survey + backward-horizon `predict_het` allocator derivation: lift the warn-and-skip fallback at `_compute_heterogeneity_test` once the pre-period Binder TSL cell-period allocator is derived (currently the gate emits a `UserWarning` and falls back to forward-horizon-only heterogeneity under `survey_design + placebo + heterogeneity`) (`chaisemartin_dhaultfoeuille.py`, `docs/methodology/REGISTRY.md`)
- Rust local-method solver path unification to `solve_wls_svd` + bootstrap-weight RNG parity audit (`rust/src/trop.rs`, `rust/src/bootstrap.rs`)
- AI review CI workflow-contract pin test expansion (`tests/test_openai_review.py`)
- In-site Sphinx render of `REPORTING.md` and `REGISTRY.md` (`docs/conf.py` + `:doc:` link migration)

#### Tier C — Heavy / derivation required

- HonestDiD Δ^RM ARP conditional/hybrid confidence sets (`honest_did.py`)
- Weighted one-way Bell-McCaffrey + weighted CR2 Bell-McCaffrey (linalg derivations + R parity harness) (`linalg.py::_compute_bm_dof_from_contrasts`, `linalg.py::_compute_cr2_bm`)
- Multi-absorb weighted demeaning: alternating-projection iteration for N>1 absorb + weights (`estimators.py`)
- ImputationDiD dense `(A0'A0).toarray()` OOM: alternative dense fallback or richer sparse strategy (`imputation.py:1531`)
- HAD mass-point `vcov_type ∈ {hc2, hc2_bm}`: 2SLS-specific leverage derivation (`had.py::_fit_mass_point_2sls`)
- HAD repeated-cross-section identification path (`had.py::_validate_had_panel`)
- HAD time-varying-dose event study estimator (`had.py::_validate_had_panel_event_study`)
- Conley + `survey_design` (`linalg.py::_validate_vcov_args`, `conley.py`)
- SyntheticDiD `vcov_type="conley"` (`synthetic_did.py::SyntheticDiD` — new analytical sandwich path OR spatial-block bootstrap)

#### Tier D — Deferred / research (no active action planned)

- HAD survey-aware support-endpoint test (`had_pretests.py::qug_test`; waits on literature — endpoint EVT × survey-aware functional CLT)
- HAD joint cross-horizon analytical covariance / unweighted event-study sup-t band (low user demand)
- HAD Phase 4.5 replicate-weight pretests (BRR/Fay/JK1/JKn/SDR composition derivation)
- HAD Stute family Stata-bridge parity (no R `Stutetest` package exists publicly)
- HAD `trends_lin × survey_design` weighted-slope derivation
- Phase 1c lprobust follow-ups (`vce` modes, weights, multi-eval grid, clustered-DGP auto-bandwidth) — deferred to Phase 2+ of `bias_corrected_local_linear`
- TestWorkflowDoesNotExecutePRHeadCode (CodeQL #14) residual bypass paths — diminishing return given documented threat model
- All remaining `Low`-priority Performance and Testing/Docs rows (R-script-per-test, CS R covariate-adjusted IRLS benchmark, doc-deps integrity CI, Rust faer SVD overhead, etc.)

---

### Standard Error Consistency

`vcov_type` has subsumed the previously-proposed `se_type` knob. `DifferenceInDifferences` and `TwoWayFixedEffects` accept `vcov_type ∈ {"classical", "hc1", "hc2", "hc2_bm", "conley"}` (the validated set in `linalg.py::_VALID_VCOV_TYPES`); cluster-robust variance is obtained by passing `cluster=` alongside the heteroscedasticity kind (`hc1 + cluster` ⇒ CR1 Liang-Zeger; `hc2_bm + cluster` ⇒ CR2 Bell-McCaffrey, gated by the open weighted-CR2 / absorbed-FE rows in the table above); wild cluster bootstrap is a separate `inference="wild_bootstrap"` path on the same estimator. Threading `vcov_type` through the 8 standalone estimators (`CallawaySantAnna`, `SunAbraham`, `ImputationDiD`, `TwoStageDiD`, `TripleDifference`, `StackedDiD`, `WooldridgeDiD`, `EfficientDiD`) remains open and is tracked as a single methodology row in the table above (Phase 1a row).

### Type Annotations

Mypy reports 0 errors. All mixin `attr-defined` errors resolved via
`TYPE_CHECKING`-guarded method stubs in bootstrap mixin classes.

## Deprecated Code

Deprecated parameters still present for backward compatibility:

- `lambda_reg` and `zeta` in `SyntheticDiD` (`synthetic_did.py`)
  - Deprecated in favor of `zeta_omega`/`zeta_lambda` parameters
  - Remove in v4.0.0 (SemVer-safe: public kwarg removal requires a major bump)

---

## Test Coverage

Visualization tests skip when matplotlib / plotly are not installed (see `pytest.importorskip` markers in `tests/test_visualization*.py`).

---

## Honest DiD Improvements

Enhancements for `honest_did.py`:

- [ ] Improved C-LF implementation with direct optimization instead of grid search
  (current implementation uses simplified FLCI approach with estimation uncertainty
  adjustment; see `honest_did.py:947`)
- [x] Support for CallawaySantAnnaResults (implemented in `honest_did.py:612-653`;
  requires `aggregate='event_study'` when calling `CallawaySantAnna.fit()`)
- [ ] Event-study-specific bounds for each post-period
- [ ] Hybrid inference methods
- [ ] Simulation-based power analysis for honest bounds

---

## CallawaySantAnna Bootstrap Improvements

- [ ] Consider aligning p-value computation with R `did` package (symmetric percentile method)

---

## RuntimeWarnings in Linear Algebra Operations

### Apple Silicon M4 BLAS Bug (numpy < 2.3)

Spurious RuntimeWarnings ("divide by zero", "overflow", "invalid value") are emitted by `np.matmul`/`@` on Apple Silicon M4 + macOS Sequoia with numpy < 2.3. The warnings appear for matrices with ≥260 rows but **do not affect result correctness** — coefficients and fitted values are valid (no NaN/Inf), and the design matrices are full rank.

**Root cause**: Apple's BLAS SME (Scalable Matrix Extension) kernels corrupt the floating-point status register, causing spurious FPE signals. Tracked in [numpy#28687](https://github.com/numpy/numpy/issues/28687) and [numpy#29820](https://github.com/numpy/numpy/issues/29820). Fixed in numpy ≥ 2.3 via [PR #29223](https://github.com/numpy/numpy/pull/29223).

**Not reproducible** on M3, Intel, or Linux.

- [ ] `linalg.py:162` - Warnings in fitted value computation (`X @ coefficients`)
  - Caused by M4 BLAS bug, not extreme coefficient values
  - Seen in test_prep.py during treatment effect recovery tests (n > 260)
- [ ] `triple_diff.py:307,323` - Warnings in propensity score computation
  - Occurs in IPW and DR estimation methods with covariates
  - Related to logistic regression overflow in edge cases (separate from BLAS bug)

- **Long-term:** Revert to `@` operator when numpy ≥ 2.3 becomes the minimum supported version.

---

## Feature Gaps (from R `did` package comparison)

Features in R's `did` package that block porting additional tests:

| Feature | R tests blocked | Priority | Status |
|---------|----------------|----------|--------|
| Calendar time aggregation | 1 test in test-att_gt.R | Low | |

---

## Performance Optimizations

Potential future optimizations:

- [ ] JIT compilation for bootstrap loops (numba)
- [ ] Sparse matrix handling for large fixed effects

### QR+SVD Redundancy in Rank Detection

**Background**: The current `solve_ols()` implementation performs both QR (for rank detection) and SVD (for solving) decompositions on rank-deficient matrices. This is technically redundant since SVD can determine rank directly.

**Current approach** (R-style, chosen for robustness):
1. QR with pivoting for rank detection (`_detect_rank_deficiency()`)
2. scipy's `lstsq` with 'gelsd' driver (SVD-based) for solving

**Why we use QR for rank detection**:
- QR with pivoting provides the canonical ordering of linearly dependent columns
- R's `lm()` uses this approach for consistent dropped-column reporting
- Ensures consistent column dropping across runs (SVD column selection can vary)

**Potential optimization** (future work):
- Skip QR when `rank_deficient_action="silent"` since we don't need column names
- Use SVD rank directly in the Rust backend (already implemented)
- Add `skip_rank_check` parameter for hot paths where matrix is known to be full-rank (implemented in v2.2.0)

**Priority**: Low - the QR overhead is minimal compared to SVD solve, and correctness is more important than micro-optimization.

### Incomplete `check_finite` Bypass

**Background**: The `solve_ols()` function accepts a `check_finite=False` parameter intended to skip NaN/Inf validation for performance in hot paths where data is known to be clean.

**Current limitation**: When `check_finite=False`, our explicit validation is skipped, but scipy's internal QR decomposition in `_detect_rank_deficiency()` still validates finite values. This means callers cannot fully bypass all finite checks.

**Impact**: Minimal - the scipy check is fast and only affects edge cases where users explicitly pass `check_finite=False` with non-finite data (which would be a bug in their code anyway).

**Potential fix** (future work):
- Pass `check_finite=False` through to scipy's QR call (requires scipy >= 1.9.0)
- Or skip `_detect_rank_deficiency()` entirely when `check_finite=False` and `_skip_rank_check=True`

**Priority**: Low - this is an edge case optimization that doesn't affect correctness.

