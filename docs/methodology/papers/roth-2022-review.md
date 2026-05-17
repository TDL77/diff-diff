# Paper Review: Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends

**Authors:** Jonathan Roth
**Citation:** Roth, J. (2022). Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends. *American Economic Review: Insights*, 4(3), 305-322.
**PDF reviewed:** papers/roth-2022.pdf (18 pages, content pages 1-15)
**Review date:** 2026-05-16
**DOI:** https://doi.org/10.1257/aeri.20210236

---

## Methodology Registry Entry

*Formatted to match docs/methodology/REGISTRY.md structure. Heading levels and labels align with existing entries — copy the `## PreTrendsPower` section into the registry (replacing the existing `## PreTrendsPower` stub).*

## PreTrendsPower

**Primary source:** [Roth, J. (2022). Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends. *American Economic Review: Insights*, 4(3), 305-322.](https://doi.org/10.1257/aeri.20210236)

**Key implementation requirements:**

*Assumption checks / warnings:*
- Input: event-study coefficient vector beta_hat = (beta_hat_pre, beta_hat_post)' that is asymptotically normal under the underlying estimator (Equation 1; Remark 1 lists TWFE, GMM, Freyaldenhoven-Hansen-Shapiro, regression-adjustment/IPW/DR DiD per Sant'Anna-Zhao, Callaway-Sant'Anna, Sun-Abraham)
- Input: estimated variance-covariance matrix Sigma_hat in R^{(K+M) x (K+M)} where K = # pre-period coefficients, M = # post-period coefficients
- **Block decomposition convention (per Roth, Section II.A-B)**: throughout this entry, the variance partition uses Roth's *post-first* ordering for the proofs, i.e., Var[(beta_hat_post, beta_hat_pre)'] = [[Sigma_11, Sigma_12], [Sigma_21, Sigma_22]] where Sigma_11 = Var[beta_hat_post] in R^{M x M} (the post-treatment block), Sigma_22 = Var[beta_hat_pre] in R^{K x K} (the pre-treatment block), Sigma_12 = Cov(beta_hat_post, beta_hat_pre) in R^{M x K}, Sigma_21 = Sigma_12'. The stacked input vector beta_hat is (pre, post)' as stated above; the (post, pre) block ordering is internal to the propositions and matches Roth's paper notation. Implementations must use the post-treatment block Sigma_11 (not the full Sigma_hat) wherever they need Var[beta_hat_post].
- Pre-trend zero-anticipation assumption: tau_pre = 0 (Equation 2) — same identifying convention as Rambachan-Roth (2023)
- Warn if pretest has low power: e.g., if the slope at 80% power (gamma_{0.8}) produces a |bias| > |estimated treatment effect|, the pretest is uninformative for the magnitudes that matter
- Warn that pretest-conditioning distortions are NOT removed by larger samples — they persist as long as the pretest can fail with non-vanishing probability (footnote 12)

*Causal decomposition (Equation 2):*

    beta = (delta_pre, delta_post)' + (0, tau_post)'
              \--------------/         \---------/
                  delta = bias               tau = causal effect
                  from trends

where tau_pre = 0 by the no-anticipation assumption and delta is the bias from a difference in trends. The pretest acts on beta_hat_pre, which equals delta_pre under no anticipation.

*Acceptance region of the standard "no individually significant" (NIS) pretest:*

    B_NIS(Sigma) = { b in R^K : |b_t| <= 1.96 * sigma_t, for all t in {-K, ..., -1} }

This corresponds to checking individual 95% CIs of each pre-period coefficient (the dominant convention in applied work: 11 of 12 surveyed papers, per Section I.B).

Alternative acceptance regions:
- **Joint Wald (chi-squared)**: B_W(Sigma) = { b in R^K : b' Sigma_22^{-1} b <= chi^2_{1-alpha, K} }. **Note:** mentioned in the paper as a less common applied convention (1 of 12 surveyed papers, Section I.B). Propositions 1, 3, 4 apply to this B since it is convex; Roth does NOT separately tabulate power/bias/coverage for the Wald form.
- **Slope-of-best-fit-line t-test**: the paper's Table 1 reports the t-statistic for the slope as an observed property of surveyed papers, but **Note (deviation from paper):** Roth does NOT analyze a slope-t-statistic acceptance region as a pretest framework. Library support for this acceptance form is an extension beyond Roth (2022).
- **Custom user-supplied B(Sigma)**: any (measurable) acceptance set; Propositions 1, 3, 4 apply for any B (paper-supported framework). Proposition 2 (sign of bias under monotone trend) requires the specific NIS form plus Assumption 1.

*Conditional bias after pretesting (Proposition 1):*

    E[beta_hat_post | beta_hat_pre in B(Sigma)]
        = tau_post + delta_post + Sigma_{12} Sigma_{22}^{-1} ( E[beta_hat_pre | beta_hat_pre in B(Sigma)] - beta_pre )

The third (pretest bias) term depends on:
- Sigma_{12} Sigma_{22}^{-1}: the regression coefficient of beta_hat_post on beta_hat_pre (akin to "leakage" from pre to post via the covariance)
- The distortion E[beta_hat_pre | beta_hat_pre in B(Sigma)] - beta_pre: how much pretest conditioning skews the pre-period means

*Sign-of-bias result under monotone trend (Proposition 2 + Assumption 1):*

    Assumption 1: Sigma has a common term sigma^2 on the diagonal and a common term rho > 0 off the diagonal, with sigma^2 > rho.

    If delta_pre < 0 elementwise and delta_post > 0 (upward pretrend), then:
        E[beta_hat_post | beta_hat_pre in B_NIS(Sigma)] > beta_post > tau_post

(Bias is worse after pretesting under monotone violations; symmetric statement for downward pretrend.)

*Variance after pretesting (Proposition 3):*

    Var[beta_hat_post | beta_hat_pre in B(Sigma)]
        = Var[beta_hat_post]
          + (Sigma_{12} Sigma_{22}^{-1}) (Var[beta_hat_pre | beta_hat_pre in B(Sigma)] - Var[beta_hat_pre]) (Sigma_{12} Sigma_{22}^{-1})'

*Convexity gives variance reduction (Proposition 4):*

    If B(Sigma) is a convex set, then Var[beta_hat_post | beta_hat_pre in B(Sigma)] <= Var[beta_hat_post].

Implication: under parallel trends (delta = 0), conventional 95% CIs OVER-cover conditional on passing the pretest (CIs are based on the unconditional variance, which is too large). When parallel trends is violated, conventional 95% CIs UNDER-cover, because the bias dominates the variance reduction.

*Target parameter (Section I.C):*

    tau_* = l' tau_post, for some user-specified l in R^M

Defaults Roth uses:
- **Average post-treatment effect**: tau_bar = (1/M)(tau_1 + ... + tau_M), i.e., l = (1/M, ..., 1/M)' (main text emphasis)
- **First-period-after-treatment effect**: tau_1, i.e., l = (1, 0, ..., 0)' (online Appendix)
- **Custom**: any user-specified contrast l

*Plug-in estimator and CI (Section I.C):*

    tau_hat = l' beta_hat_post
    CI_{tau_*} = tau_hat +/- 1.96 * sigma_{tau_hat}, where sigma^2_{tau_hat} = l' Sigma_11 l

(Note: Sigma_11 is the post-treatment covariance block per the convention above, not the full Sigma_hat.)

*Power calculation against a linear violation (Section I.C "Power Calculations"):*

For a linear violation with slope gamma (so delta_t = gamma * t with relative time t),
the pretest "passes" probability is

    P( beta_hat_pre in B_NIS(Sigma) ) = P( |beta_hat_pre,t| <= 1.96 * sigma_t, for all t )

where beta_hat_pre ~ N(delta_pre, Sigma_22) with delta_pre,t = gamma * t. The library should
solve for gamma at target power 1 - p in {0.5, 0.8}:

    gamma_{1 - p} = inf{ gamma : P( beta_hat_pre NOT in B_NIS(Sigma) | delta = gamma * t ) >= 1 - p }

These are Roth's gamma_{0.5} and gamma_{0.8} ("the slopes against which pretests have 50%
or 80% power"). Roth uses 80% as a benchmark following Cohen (1988); 50% is supplementary.

*Bias and size calculations against a given gamma (Section I.C):*

- **Unconditional bias**: E[tau_hat - tau_*] = l' delta_post (with delta_t = gamma * t for relevant t)
- **Conditional bias**: E[tau_hat - tau_* | beta_hat_pre in B_NIS(Sigma)] (computed via Proposition 1)
- **Unconditional null rejection**: P(tau_* in CI_{tau_*}^c) under linear trend
- **Conditional null rejection**: P(tau_* in CI_{tau_*}^c | beta_hat_pre in B_NIS(Sigma))

*Computational shortcut (footnote 8):*

Under joint normality, these probabilities and conditional moments can be calculated
ANALYTICALLY using results from Cartinhour (1990) and Manjunath & Wilhelm (2012) — Roth
implements via the R package `tmvtnorm`. Roth verifies simulations yield similar results.
The library should support both an analytical truncated-multivariate-normal path AND a
simulation fallback.

*Standard errors (Section II.C; footnote 7 equivariance):*
- Power calculations are EXACT (no sampling variability — gamma is computed against a hypothesized population trend, not estimated)
- Uncertainty comes entirely from the user-supplied Sigma
- Roth's bias and coverage results have NO dependence on the value of tau_post (footnote 7: the distribution of beta_hat_post conditional on beta_hat_pre passing the pretest is equivariant w.r.t. tau_post)

*Edge cases (paper-stated):*
- **Linear vs nonlinear violations**: paper formally analyzes linear trends; Caveats (Section I.D) note results extend to monotone nonlinear violations under homoskedasticity (Proposition 2); arbitrarily nonlinear violations addressed heuristically — bias is worse for exponentially-growing trends, better for log/shallow trends as pre-periods grow
- **Adding more pretreatment periods**: helps power for linear/log trends, does NOT help (and can hurt) for trends concentrated near treatment (e.g., COVID-19-like shocks)
- **K = 1 (single pre-period)**: explicit closed-form intuition via univariate truncated normal in proof of Proposition 2: E[beta_hat_pre | beta_hat_pre in B_NIS] - beta_pre proportional to phi(-1.96 - beta_pre/sigma) - phi(1.96 - beta_pre/sigma)
- **Symmetric two-sided pretests under parallel trends**: beta_hat_post remains UNBIASED for tau_post (E[beta_hat_pre | beta_hat_pre in B] = 0 if B is symmetric and beta_pre = 0)
- **Heteroskedastic Sigma (off-diagonal not constant)**: Proposition 2 requires Assumption 1; under arbitrary Sigma, sign of pretest-bias term is ambiguous (worked out in Proposition 1's general form)
- **Publication-bias trade-off (Equation 4, Section II.D)**: pretest-as-screen can REDUCE or INCREASE published bias depending on Bayes-factor of design type vs the bias-given-publication ratio; uninformative pretests are unambiguously harmful

*Algorithm (no numbered algorithm in paper; implementation distilled from Section I.C):*

1. Take user-supplied (beta_hat, Sigma, K, M) and a target estimand l in R^M (default: l = uniform 1/M)
2. Compute B_NIS(Sigma) acceptance region using diagonal sigma_t = sqrt(Sigma_22[t, t]) for t in pre periods (Sigma_22 = Var[beta_hat_pre] per the block convention above)
3. **Power**: solve gamma_{1-p} = root of P(reject pretest | gamma) = 1 - p
   - For each candidate gamma, compute P(beta_hat_pre in B_NIS) under beta_hat_pre ~ N(gamma * t_pre, Sigma_22) using `tmvtnorm`-style multivariate normal CDF; or via simulation
4. **Bias**: for gamma in {0, gamma_{0.5}, gamma_{0.8}, user-custom}:
   - Compute unconditional bias = l' delta_post where delta_post,m = gamma * m
   - Compute conditional bias via Proposition 1: requires E[beta_hat_pre | beta_hat_pre in B_NIS] from truncated MVN
5. **Coverage**: for the same gamma values, compute unconditional and conditional null rejection probabilities P(tau_* not in CI):
   - Unconditional: P(|tau_hat - tau_*|/sigma_{tau_hat} > 1.96) under beta_hat ~ N(beta, Sigma)
   - Conditional: P(|tau_hat - tau_*|/sigma_{tau_hat} > 1.96 | beta_hat_pre in B_NIS) — joint truncated MVN
6. Return a structured summary (Roth's Table 2/Table 3 layout)

**Reference implementation(s):**
- R: [`pretrends`](https://github.com/jonathandroth/pretrends) (Jonathan Roth's own package) and the accompanying Shiny app
- R dependency: [`tmvtnorm`](https://cran.r-project.org/package=tmvtnorm) (Manjunath & Wilhelm 2012) for truncated multivariate normal moments and CDF

**Requirements checklist:**
- [ ] Acceptance regions: NIS (individual t, paper-analyzed); joint Wald and custom B (paper-supported via Propositions 1, 3, 4, not separately tabulated by Roth); **Note (deviation from paper):** slope-of-best-fit-line is an extension beyond Roth (2022) — paper tabulates the slope t-stat but does not analyze a slope-t pretest framework
- [ ] Power calculation against linear violation with slope gamma — solve for gamma_{0.5} and gamma_{0.8}
- [ ] Analytical truncated multivariate normal path (tmvtnorm-equivalent) + simulation fallback
- [ ] Unconditional and conditional bias for arbitrary linear contrast l in R^M (using Sigma_11 for the post-treatment variance)
- [ ] Unconditional and conditional null rejection / coverage for the same linear contrast
- [ ] **Note (deviation from paper):** non-linear trend hypotheses — Roth (2022) formally analyzes only LINEAR violations; "constant level shift", "last-period jump", and "custom delta vector" patterns are extensions from Roth's R `pretrends` package, applied via the same Proposition 1/3/4 framework
- [ ] Plot of bias against pretest power for visual reporting (Roth's Figure 1 style)
- [ ] Composes with HonestDiD result objects (shared beta_hat, Sigma_hat input contract)

---

## Implementation Notes

### Data Structure Requirements
- **Input**: beta_hat in R^{K+M} (concatenated pre + post event-study coefficients), Sigma_hat in R^{(K+M) x (K+M)} (variance-covariance matrix), integer K (# pre-period coefficients), integer M (# post-period coefficients)
- **Optional input**: linear contrast l in R^M (defaults to uniform 1/M for average post-treatment effect, or e_1 for first-period-only)
- **Optional input**: significance level alpha (default 0.05 → critical value 1.96)
- **Optional input**: target power levels (default {0.5, 0.8} per Roth)
- The pre-period coefficients are typically indexed by relative time t in {-K, -K+1, ..., -1}, with t = 0 omitted as the reference period
- Compatible with the result classes of: MultiPeriodDiD (event study), CallawaySantAnna (staggered), SunAbraham (interaction-weighted), Freyaldenhoven-Hansen-Shapiro (covariate-based)

### Computational Considerations
- **Truncated MVN moments and probabilities**: scipy.stats has only the univariate case; library options for K > 1 are (a) port `tmvtnorm` (Manjunath-Wilhelm closed-form for orthant moments + Cartinhour 1990 for the rectangular box), (b) Monte Carlo simulation with rejection sampling. Recommend implementing both paths and validating equivalence at alpha-tol = 1e-3 for small K.
- **Cost**: dominated by the multivariate normal box probability evaluations. For K <= 5, analytical methods are fast. For K > 10, simulation is preferable.
- **Root-finding for gamma_p**: monotone function of gamma; use bisection over [0, gamma_max] with gamma_max derived from a univariate upper bound (largest |gamma| at which power = 1).
- **Memoization**: power and bias share intermediate quantities (truncated MVN moments); cache by gamma.

### Tuning Parameters

| Parameter | Type | Default | Selection Method |
|-----------|------|---------|-----------------|
| `alpha` | float in (0, 1) | 0.05 | Standard significance level for pretest and reporting CI |
| `target_power` | list[float] in (0, 1) | [0.5, 0.8] | Roth's reported benchmarks (Cohen 1988 conventional 0.8; 0.5 for "even-odds detection") |
| `l` (contrast) | array in R^M | uniform 1/M | User-specified linear functional of tau_post |
| `pretest_form` | enum | "individual" (NIS) | "individual" (paper-analyzed); "joint_wald" / "custom" (paper-supported via Propositions 1/3/4); "slope" — **deviation from paper**, R-package extension |
| `acceptance_region` | callable or set | B_NIS | Custom B(Sigma) for "custom" pretest_form (paper-supported: Propositions 1, 3, 4 apply to any B) |
| `method` | enum | "analytical" | "analytical" (tmvtnorm-equivalent) or "simulation" |
| `n_sim` | int | 10000 | Monte Carlo iterations when method="simulation" |

### Relation to Existing diff-diff Estimators
- **Pre-existing `diff_diff/pretrends.py`** (1133 lines) — implements a Roth-2022 framework; this paper review's main use is to audit the existing surface against the paper's exact equations
- **Composes with**: `MultiPeriodDiD`, `CallawaySantAnna`, `SunAbraham`, `TwoWayFixedEffects` — any estimator producing an event-study coefficient vector and a consistent variance estimator
- **Complement to `HonestDiD` (Rambachan-Roth 2023)**: Roth 2022 asks "what bias survives a pretest under linear violations?"; Rambachan-Roth 2023 asks "what is the identified set of tau_post under bounded violations?" Both use the same (beta_hat, Sigma_hat) input contract — the library should expose a unified entry-point that can produce both Roth-2022 and HonestDiD reports from one event-study result object.
- **Shares zero-anticipation convention with HonestDiD**: tau_pre = 0, so beta_pre = delta_pre. Cross-reference the existing `diff_diff/honest_did.py` for the contract.

---

## Key Theorems / Propositions

| # | Statement | Implementation use |
|---|-----------|---------------------|
| **Proposition 1** | For any B(Sigma): E[beta_hat_post | beta_hat_pre in B] = tau_post + delta_post + Sigma_{12} Sigma_{22}^{-1} (E[beta_hat_pre | beta_hat_pre in B] - beta_pre) | The main bias decomposition formula. Drives the conditional-bias computation in step 4 of the algorithm. |
| **Proposition 2** | Under Assumption 1 (homoskedastic-equicorrelated Sigma) and monotone trend (delta_pre < 0, delta_post > 0): E[beta_hat_post | beta_hat_pre in B_NIS] > beta_post > tau_post | Justifies WARN that conditional bias is worse than unconditional bias under monotone trends — applicable in many but not all empirical settings. Library should detect when Assumption 1 holds (e.g., balanced panel + cluster-robust at unit level + equicorrelated errors) and surface this warning more strongly. |
| **Proposition 3** | Var[beta_hat_post | beta_hat_pre in B] = Var[beta_hat_post] + (Sigma_{12} Sigma_{22}^{-1}) (Var[beta_hat_pre | beta_hat_pre in B] - Var[beta_hat_pre]) (Sigma_{12} Sigma_{22}^{-1})' | The conditional-variance formula; drives the over/under-coverage analysis. |
| **Proposition 4** | If B(Sigma) is convex: Var[beta_hat_post | beta_hat_pre in B] <= Var[beta_hat_post]. CIs based on unconditional Sigma OVER-cover under parallel trends, UNDER-cover under violations. | Justifies the "do not interpret a wide CI as ample power" warning. |

No formal theorems are stated for the publication-rules analysis (Section II.D); Equation 4 is the operational result.

---

## Calibrated DGP for Simulations (Section I.C "Calibrating the Model")

For each paper in Roth's empirical survey:

1. Calibrate finite-sample normal model (Equation 1): beta_hat ~ N(beta, Sigma) with K pre-periods + M post-periods matching the original paper
2. Set Sigma = estimated variance-covariance matrix from the original paper (using whatever clustering method the authors specified)
3. Set tau_post = original paper's beta_hat_post (footnote 7: has no impact on bias/coverage results by equivariance)
4. Calibrate delta to a linear trend with slope gamma_{0.5} or gamma_{0.8}
5. Re-compute power, bias, and coverage analytically (or by simulation)

**Test fixture suggestion for the library**: a Roth-2022 parity test against one of the 12 papers in Table 1 (e.g., Bailey & Goodman-Bacon 2015 has 5 pre-periods + a clean calibrated VCV available in his replication data — `https://doi.org/10.3886/E151982V1`).

---

## Empirical Findings (Section I.C "Results"; Tables 2-3)

Quoting Roth's key empirical results (for cross-validation):

- **Power**: in the most extreme paper (Deryugina 2017), an unconditional bias of magnitude comparable to the estimated effect is detected only 50% of the time
- **Coverage**: under gamma_{0.8} (80%-power slope), unconditional null rejection rates of 95% CIs range from 53% to 98% across the 12 papers
- **Pretest bias**: percent additional bias from pretest conditioning (Table 3, gamma_{0.8}, tau_1): from -34% (Bosch-Campos-Vazquez 2014, beneficial — rare) to +120% (Deryugina 2017, harmful — common); paper-aggregate finding is that conditional bias EXCEEDS unconditional bias in 9 of 12 papers for tau_1 and in 10 of 12 for tau_bar
- **Equation 4 sign**: the relative-fraction term is < 1 (pretest helps screen out biased designs); the conditional-bias term is typically > 1 (pretest amplifies bias when a biased design is published); net sign depends on which dominates — the paper does not provide closed-form criteria

---

## Gaps and Uncertainties

- **Joint Wald acceptance region**: paper mentions joint tests only briefly (Section I.B notes 1 of 12 papers uses one). Power, bias, and coverage formulas all apply by replacing B_NIS with the joint Wald acceptance region B_W, but Roth does not work out a separate table. Library should implement both but test against R `pretrends` for the joint-Wald case (Roth's package supports it).
- **"Slope-of-best-fit-line t-test" acceptance region**: Table 1 column shows the t-stat for the slope of the linear pre-trend. Paper does not analyze pretests based on this t-stat as a separate acceptance region; library should NOT extrapolate without further reading the `pretrends` package source.
- **Nonlinear violations**: Section I.D acknowledges results extend to monotone violations under homoskedasticity (Proposition 2), but the linear-violation framework is the operational benchmark. Library's `violation_type in {"linear", "constant", "last_period", "custom"}` (per the existing REGISTRY entry) appears to predate the paper — the paper itself only formally analyzes linear violations. "Constant" and "last_period" are likely Roth-package extensions for practical reasoning; library should document this as an extension beyond Roth's published analysis.
- **Custom delta**: paper does not propose a "custom delta vector" interface; this is an extension by Roth's R package. The library should preserve the convention.
- **Choice of contrast l**: paper highlights l = uniform 1/M (average post-treatment) and l = e_1 (first period after treatment). No guidance on other contrasts (e.g., long-run effect l = e_M, dynamic-weighted contrast) — library should document defaults and warn that bias and coverage depend on l.
- **K = 0 (no pre-periods)**: trivially no pretest possible; library should error.
- **Heteroskedastic Sigma**: Proposition 2 requires Assumption 1. Library implements computations under arbitrary Sigma via Proposition 1; the sign of the bias-amplification effect is then NOT guaranteed. Library should NOT print "pretest amplifies bias under monotone trends" unless Assumption 1 is approximately satisfied (or just always issue the conditional warning).
- **Equation 4 publication-rules analysis**: not standardly implemented in PreTrendsPower-style tools. Roth notes it as part of the discussion (Section II.D) but does not provide a numerical workflow for users. Library should NOT attempt to implement Equation 4 unless requested.
- **Connection to `compute_pretrends_power` library helper** (referenced in feedback memory `feedback_verdict_powered_by_tools.md`): the paper review confirms that "minimum slope detectable at 80% power" is exactly Roth's gamma_{0.8}, and the library helper should compute and surface this. Need to verify the existing helper's calling convention against the paper's framework when auditing `diff_diff/pretrends.py`.
- **R `pretrends` package version**: paper cites the package at https://github.com/jonathandroth/pretrends; no specific version cited. R-parity work should pin to a specific commit and document.
- **Compatibility with multi-cohort estimators**: Remark 1 lists Callaway-Sant'Anna, Sun-Abraham, etc. as compatible. The paper does not detail how to construct (beta_hat, Sigma_hat) from those estimators when the event-study output is multi-cohort (e.g., cohort × event-time matrix). Library should document the aggregation convention (per Sun-Abraham overall ATT or per Callaway-Sant'Anna `aggregate=event`).
