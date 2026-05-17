# Generate CR2 Bell-McCaffrey golden values via R clubSandwich.
#
# This script is the parity source for CR2 Bell-McCaffrey cluster-robust
# inference implemented in diff_diff/linalg.py::_compute_cr2_bm.
#
# Usage:
#   Rscript benchmarks/R/generate_clubsandwich_golden.R
#
# Requirements:
#   clubSandwich (CRAN), jsonlite
#
# Output:
#   benchmarks/data/clubsandwich_cr2_golden.json
#
# Phase 1a of the HeterogeneousAdoptionDiD implementation (de Chaisemartin,
# Ciccia, D'Haultfoeuille & Knau 2026, arXiv:2405.04465v6). The parity
# dataset below consists of three small deterministic designs; the Python
# test at tests/test_linalg_hc2_bm.py::TestCR2BMParityClubSandwich loads
# this JSON and checks agreement to 6 digits.

suppressPackageStartupMessages({
  library(clubSandwich)
  library(jsonlite)
})

set.seed(20260420)

# --- Three deterministic datasets ---------------------------------------------

make_dataset <- function(name, n_clusters, cluster_sizes, seed) {
  set.seed(seed)
  cluster_ids <- rep(seq_len(n_clusters), times = cluster_sizes)
  n <- length(cluster_ids)
  x <- runif(n, 0, 1)
  # Cluster-level shock to induce within-cluster correlation, plus idiosyncratic noise.
  shock <- rnorm(n_clusters, sd = 0.5)
  y <- 1 + 0.5 * x + shock[cluster_ids] + rnorm(n, sd = 0.2)
  data.frame(name = name, cluster = cluster_ids, x = x, y = y)
}

datasets <- list(
  balanced_small = make_dataset("balanced_small", 5, rep(6, 5), 101),
  unbalanced_medium = make_dataset("unbalanced_medium", 8, c(3, 4, 5, 6, 7, 8, 9, 10), 202),
  singletons_present = make_dataset("singletons_present", 10, c(1, 1, 2, 3, 4, 5, 6, 7, 8, 9), 303)
)

output <- list()

for (nm in names(datasets)) {
  d <- datasets[[nm]]
  fit <- lm(y ~ x, data = d)
  vcov_cr2 <- vcovCR(fit, cluster = d$cluster, type = "CR2")
  # Per-coefficient Bell-McCaffrey Satterthwaite DOF via coef_test()$df_Satt.
  # (clubSandwich 0.7+ removed `Wald_test(..., test="Satterthwaite")`; the
  # `df_Satt` column from coef_test() is the idiomatic per-coefficient form
  # and is numerically identical to the old per-unit-contrast path.)
  ct <- coef_test(fit, vcov = vcov_cr2)
  coef_names <- names(coef(fit))
  output[[nm]] <- list(
    x = d$x,
    y = d$y,
    cluster = d$cluster,
    coef = as.numeric(coef(fit)),
    coef_names = coef_names,
    vcov_cr2 = as.numeric(vcov_cr2),
    vcov_shape = dim(vcov_cr2),
    dof_bm = as.numeric(ct$df_Satt),
    cluster_sizes = as.numeric(table(d$cluster))
  )
}

# --- Absorbed-FE DiD scenario (PR for absorb + hc2/hc2_bm gate lift) ---------
# Canonical DiD-with-FE: y ~ treat_post + factor(unit) + factor(period). The
# Python side uses `DiD(vcov_type="hc2_bm").fit(absorb=["unit","period"])`
# which auto-routes to fixed_effects= internally and builds the same full-
# dummy design as R's `lm()`. R parity targets are computed via the
# singleton-cluster CR2 trick (for HC2-BM one-way) and cluster=unit (for CR2).

make_did_panel <- function(n_units, n_periods, treatment_period, seed) {
  set.seed(seed)
  unit <- rep(seq_len(n_units), each = n_periods)
  period <- rep(seq_len(n_periods), times = n_units)
  treated <- rep(rep(c(0L, 1L), each = n_units / 2L), each = n_periods)
  post <- as.integer(period >= treatment_period)
  treat_post <- treated * post
  unit_fe <- rnorm(n_units, sd = 1.5)
  time_fe <- rnorm(n_periods, sd = 0.5)
  eps <- rnorm(length(unit), sd = 0.3)
  y <- 2.0 * treat_post + unit_fe[unit] + time_fe[period] + eps
  data.frame(unit = factor(unit), period = factor(period),
             treated = treated, post = post, treat_post = treat_post,
             y = y, unit_int = unit, period_int = period)
}

d_did <- make_did_panel(n_units = 8, n_periods = 4, treatment_period = 2, seed = 404)
fit_did <- lm(y ~ treat_post + unit + period, data = d_did)
# HC2 via sandwich::vcovHC(type = "HC2"). Pins the in-tree HC2-parity claim
# the changelog/registry make for the absorb auto-route on the hc2 lane.
vcov_did_hc2 <- sandwich::vcovHC(fit_did, type = "HC2")
# HC2-BM unclustered via singleton-cluster CR2 (PT2018-blessed workaround,
# since clubSandwich::vcovCR requires a cluster arg).
vcov_did_hc2_bm <- vcovCR(fit_did, cluster = seq_len(nrow(d_did)), type = "CR2")
ct_did_hc2_bm <- coef_test(fit_did, vcov = vcov_did_hc2_bm)
# CR2-BM clustered by unit.
vcov_did_cr2 <- vcovCR(fit_did, cluster = d_did$unit, type = "CR2")
ct_did_cr2 <- coef_test(fit_did, vcov = vcov_did_cr2)
output$absorbed_fe_did <- list(
  unit = d_did$unit_int,
  period = d_did$period_int,
  treated = d_did$treated,
  post = d_did$post,
  y = d_did$y,
  coef = as.numeric(coef(fit_did)),
  coef_names = names(coef(fit_did)),
  vcov_hc2 = as.numeric(vcov_did_hc2),
  vcov_hc2_shape = dim(vcov_did_hc2),
  vcov_hc2_bm = as.numeric(vcov_did_hc2_bm),
  vcov_hc2_bm_shape = dim(vcov_did_hc2_bm),
  dof_hc2_bm = as.numeric(ct_did_hc2_bm$df_Satt),
  vcov_cr2 = as.numeric(vcov_did_cr2),
  vcov_cr2_shape = dim(vcov_did_cr2),
  dof_cr2 = as.numeric(ct_did_cr2$df_Satt)
)

# --- Absorbed-FE MultiPeriodDiD event-study scenario (gate lift PR) ----------
# Mirrors MPD(fixed_effects=["unit"]) destination of the absorb auto-route on
# MultiPeriodDiD. MPD parameterization: const + treated + period_f (non-ref)
# + treated:period_X (non-ref) + factor(unit). Build the interaction columns
# explicitly so the R fit's coefficient names match MPD's `treated:period_X`.

make_mpd_panel <- function(n_total, units_per_cohort, n_periods, seed) {
  set.seed(seed)
  d <- expand.grid(unit = seq_len(n_total), period = seq_len(n_periods))
  d$cohort <- ((d$unit - 1L) %/% units_per_cohort) + 1L
  n_cohorts <- n_total %/% units_per_cohort
  # Last cohort is never-treated control; preceding cohorts ever-treated.
  d$treated <- as.integer(d$cohort < n_cohorts)
  d$y <- 1 + 0.5 * d$treated * (d$period >= 3) +
         rnorm(nrow(d), sd = 0.5) +
         0.1 * d$unit + 0.2 * d$period
  d
}

d_mpd <- make_mpd_panel(n_total = 25, units_per_cohort = 5, n_periods = 5,
                        seed = 12345)
d_mpd$period_f <- relevel(factor(d_mpd$period), ref = "1")
# Explicit interaction columns to match MPD's parameterization exactly.
for (p in 2:5) {
  d_mpd[[paste0("treated_period_", p)]] <- d_mpd$treated * (d_mpd$period == p)
}
fit_mpd <- lm(y ~ treated + period_f +
                  treated_period_2 + treated_period_3 +
                  treated_period_4 + treated_period_5 +
                  factor(unit),
              data = d_mpd)
vcov_mpd_hc2 <- sandwich::vcovHC(fit_mpd, type = "HC2")
vcov_mpd_hc2_bm <- vcovCR(fit_mpd, cluster = seq_len(nrow(d_mpd)), type = "CR2")
ct_mpd_hc2_bm <- coef_test(fit_mpd, vcov = vcov_mpd_hc2_bm)
output$mpd_absorbed_fe_did <- list(
  unit = d_mpd$unit,
  period = d_mpd$period,
  treated = d_mpd$treated,
  y = d_mpd$y,
  coef = as.numeric(coef(fit_mpd)),
  coef_names = names(coef(fit_mpd)),
  vcov_hc2 = as.numeric(vcov_mpd_hc2),
  vcov_hc2_shape = dim(vcov_mpd_hc2),
  vcov_hc2_bm = as.numeric(vcov_mpd_hc2_bm),
  vcov_hc2_bm_shape = dim(vcov_mpd_hc2_bm),
  dof_hc2_bm = as.numeric(ct_mpd_hc2_bm$df_Satt),
  reference_period = 1L,
  target_period = 4L
)

output$meta <- list(
  source = "clubSandwich",
  clubSandwich_version = as.character(packageVersion("clubSandwich")),
  R_version = R.version.string,
  generated_at = format(Sys.time(), tz = "UTC", usetz = TRUE),
  note = "CR2 Bell-McCaffrey cluster-robust parity target for diff_diff._compute_cr2_bm"
)

out_path <- file.path("benchmarks", "data", "clubsandwich_cr2_golden.json")
writeLines(toJSON(output, pretty = TRUE, digits = 15, auto_unbox = TRUE), out_path)
cat("Wrote", out_path, "\n")
