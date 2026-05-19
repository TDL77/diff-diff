#!/usr/bin/env Rscript
# Generate R `pretrends` parity goldens for diff-diff PreTrendsPower (PR-C).
#
# This script is committed in PR-B (PreTrendsPower implementation audit,
# Roth 2022); the JSON goldens at ../data/r_pretrends_golden.json are
# DEFERRED to PR-C. Running this script writes the JSON to that path; PR-C
# pins the R `pretrends` package commit / release, runs this script, and
# commits the resulting JSON to land the parity tests.
#
# Requires:
#   - R 4.4+ (tested on 4.5.2)
#   - install.packages("remotes")
#   - remotes::install_github("jonathandroth/pretrends", ref = "<PR-C-PIN>")
#   - install.packages("jsonlite")
#
# **R `pretrends` commit pin (TODO — PR-C):** the audited revision MUST be
# recorded here before parity assertions are committed. As of 2026-05-18
# (PR-B implementation date) the script targets the default `main` branch
# at https://github.com/jonathandroth/pretrends with no pin. PR-C will
# replace `<PR-C-PIN>` with the exact commit hash AND verify the surface
# claims documented in REGISTRY.md `## PreTrendsPower` and the paper
# review's "R `pretrends` package version pin (provisional)" Gaps bullet.
#
# Output: ../data/r_pretrends_golden.json
#
# diff-diff PreTrendsPower with `pretest_form='nis'` (the new default per
# PR-B Step 2) is expected to match the values in this JSON at atol=1e-6
# along a three-tier contract:
#   (1) NIS box probability `P(β̂_pre ∈ B_NIS(Σ))` at fixed M values on
#       all 3 fixtures;
#   (2) MDV / gamma_p (slope at target power 0.5 and 0.8) on regular and
#       irregular pre-period grids;
#   (3) γ-unit MDV invariance: PR-B's "skip L2 norm for linear with
#       relative_times" path produces MDV in Roth's γ units exactly,
#       matching R's `slope_for_power()` which also reports γ.
#
# Three fixtures (matched to test_methodology_pretrends.py expectations):
#   1. uniform_3_pre_periods_no_anticipation — K=3 regular grid (t ∈ {-3, -2, -1}),
#      never-treated control. Default-case parity baseline.
#   2. irregular_pre_periods — K=3 with relative_times = [-5, -3, -1].
#      Exercises the PR-B γ-unit linear-pattern fix.
#   3. anticipation_shifted — K=4 with anticipation=1 (pre-cutoff at t<-1,
#      so pre-periods are {-5, -4, -3, -2}). Verifies the pre-period filter
#      logic in `_extract_pre_period_params`.
#
# Run:
#   cd benchmarks/R && Rscript generate_pretrends_golden.R

suppressPackageStartupMessages({
  library(pretrends)
  library(jsonlite)
})

stopifnot(packageVersion("pretrends") >= "0.1.0")

# ---------------------------------------------------------------------------
# DGP helper: build a synthetic event-study coefficient vector + VCV under a
# stylized null DGP (β = 0, Σ_22 ~ correlated). Mirrors the simulation
# fixtures in test_methodology_pretrends.py.
# ---------------------------------------------------------------------------

build_event_study_fixture <- function(
  pre_periods,
  post_periods,
  sigma2 = 0.04,
  rho = 0.3,
  seed = 42L
) {
  # Generate a correlated equicorrelation Σ across all (pre + post) periods.
  # Realized β̂ drawn from N(0, Σ) — null DGP, no real treatment effect.
  set.seed(seed)
  all_periods <- c(pre_periods, post_periods)
  K_total <- length(all_periods)
  Sigma <- sigma2 * (rho * matrix(1, K_total, K_total) + (1 - rho) * diag(K_total))
  beta_hat <- MASS::mvrnorm(1, mu = rep(0, K_total), Sigma = Sigma)

  list(
    beta_hat = beta_hat,
    Sigma = Sigma,
    all_periods = all_periods,
    pre_periods = pre_periods,
    post_periods = post_periods
  )
}

# ---------------------------------------------------------------------------
# Extract R pretrends() output into a fixture-shaped list.
# ---------------------------------------------------------------------------

extract_pretrends <- function(fixture_data, fixture_name) {
  beta_hat <- fixture_data$beta_hat
  Sigma <- fixture_data$Sigma
  pre_periods <- fixture_data$pre_periods
  post_periods <- fixture_data$post_periods
  all_periods <- fixture_data$all_periods

  # R `pretrends` expects: betahat (coefficient vector), sigma (VCV matrix),
  # tVec (relative-time labels including the reference period 0, omitted
  # from betahat / sigma per convention), referencePeriod = 0, alpha = 0.05.

  # The `slopes_for_power` helper returns gamma values at target power.
  # For the three-tier parity contract, we capture both NIS power at a fixed
  # slope and the inverse (γ_p MDV) at target power 0.5 and 0.8.

  # NIS power at fixed gamma values (for tier-1 parity):
  gamma_test_values <- c(0.0, 0.2, 0.5, 1.0)
  power_values <- sapply(gamma_test_values, function(g) {
    # Build δ = γ * |t| for pre-periods (Roth's δ_t = γ·t convention,
    # using |t| since pre-period t < 0).
    delta_pre <- g * abs(pre_periods)
    # `pretrends` package: pretrends() with explicit delta vector.
    # The exact R API: pretrends(betahat, sigma, tVec, referencePeriod,
    #                            deltahypothesis, ...).
    # PR-C: replace this stub with the actual R pretrends() call and
    # extract the rejection probability.
    NA_real_  # PR-C will populate
  })

  # γ_p MDV: solve for γ such that NIS rejection probability = target power.
  # R `slope_for_power(betahat, sigma, tVec, referencePeriod, power)`.
  gamma_p_values <- sapply(c(0.5, 0.8), function(p) {
    # PR-C: replace with actual R slope_for_power() call.
    NA_real_
  })

  list(
    panel = list(
      pre_periods = as.integer(pre_periods),
      post_periods = as.integer(post_periods),
      all_periods = as.integer(all_periods),
      beta_hat = as.numeric(beta_hat),
      Sigma = Sigma
    ),
    r_power_at_gamma = list(
      gamma_test_values = as.numeric(gamma_test_values),
      power_values = as.numeric(power_values)
    ),
    r_gamma_p = list(
      target_power = c(0.5, 0.8),
      gamma_p_values = as.numeric(gamma_p_values)
    ),
    fixture_name = fixture_name
  )
}

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

cat("Building fixture 1: uniform_3_pre_periods_no_anticipation...\n")
f1 <- build_event_study_fixture(
  pre_periods = c(-3L, -2L, -1L),
  post_periods = c(1L, 2L, 3L),
  seed = 101L
)
fixture_1 <- extract_pretrends(f1, "uniform_3_pre_periods_no_anticipation")

cat("Building fixture 2: irregular_pre_periods...\n")
# K=3 with t ∈ {-5, -3, -1}. Tests PR-B's γ-unit linear-pattern fix:
# pre-PR-B Python with normalized count-based weights would silently report
# MDV in [0.45, 0.30, 0.15] / sqrt(0.3) units, not γ. R `slope_for_power()`
# always reports γ; Python's PR-B Step 4 makes the two match at atol=1e-6.
f2 <- build_event_study_fixture(
  pre_periods = c(-5L, -3L, -1L),
  post_periods = c(1L, 2L, 3L),
  seed = 202L
)
fixture_2 <- extract_pretrends(f2, "irregular_pre_periods")

cat("Building fixture 3: anticipation_shifted...\n")
# K=4 pre-periods with anticipation=1. Real pre-treatment cutoff is t < -1,
# so the {-5, -4, -3, -2} cells are the genuine pre-periods; t=-1 is the
# anticipation window. Tests the pre-period filtering logic.
f3 <- build_event_study_fixture(
  pre_periods = c(-5L, -4L, -3L, -2L),  # genuine pre-periods (cutoff = -1)
  post_periods = c(1L, 2L, 3L),
  seed = 303L
)
fixture_3 <- extract_pretrends(f3, "anticipation_shifted")

# ---------------------------------------------------------------------------
# Write JSON
# ---------------------------------------------------------------------------

out <- list(
  meta = list(
    generated_at = format(Sys.Date()),
    pretrends_version = as.character(packageVersion("pretrends")),
    pretrends_commit = "<PR-C-PIN>",  # TODO PR-C: replace with actual git SHA
    r_version = R.version.string,
    description = paste(
      "Roth (2022) PreTrendsPower parity goldens for diff-diff",
      "compute_pretrends_power / PreTrendsPower (PR-C parity target).",
      "Parity at atol=1e-6 along a three-tier contract:",
      "(1) NIS box probability at fixed γ values on all 3 fixtures;",
      "(2) γ_p MDV (slope at target power 0.5 and 0.8) on regular and",
      "irregular grids;",
      "(3) γ-unit MDV invariance: PR-B's skip-L2-norm path produces MDV",
      "in Roth's γ units exactly, matching R's slope_for_power().",
      "See diff-diff/docs/methodology/papers/roth-2022-review.md for",
      "the full derivation."
    )
  ),
  uniform_3_pre_periods_no_anticipation = fixture_1,
  irregular_pre_periods = fixture_2,
  anticipation_shifted = fixture_3
)

out_path <- "../data/r_pretrends_golden.json"
write_json(out, out_path, pretty = TRUE, digits = NA, auto_unbox = TRUE)
cat(sprintf("Wrote %s\n", out_path))
cat("\n")
cat("PR-C TODO checklist:\n")
cat("  [ ] Replace <PR-C-PIN> commit-hash placeholder above with actual\n")
cat("      git SHA from https://github.com/jonathandroth/pretrends.\n")
cat("  [ ] Replace the NA_real_ stubs in extract_pretrends() with the\n")
cat("      actual pretrends::pretrends() / slope_for_power() calls.\n")
cat("  [ ] Verify the surface claims in REGISTRY.md PreTrendsPower\n")
cat("      Reference implementations section against the pinned revision.\n")
cat("  [ ] Activate tests/test_methodology_pretrends.py::TestPretrendsParityR\n")
cat("      (currently skips via @pytest.mark.skipif when the JSON is missing).\n")
cat("  [ ] Flip METHODOLOGY_REVIEW.md PreTrendsPower row from\n")
cat("      **Complete** (R parity pending) → **Complete**.\n")
