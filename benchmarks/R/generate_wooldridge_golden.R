# Generate R-parity goldens for WooldridgeDiD OLS path vcov_type variants.
#
# Phase 1b PR 3/8: pins Python `WooldridgeDiD(method='ols', vcov_type=...)` SE
# output against `lm()` + clubSandwich / sandwich on the fixed-seed staggered
# panel from `benchmarks/data/wooldridge_test_panel.csv`.
#
# Variants generated:
#   - hc1 (CR1 Liang-Zeger cluster-robust at unit; matches `type="CR1S"` —
#     Stata-style G/(G-1) * (n-1)/(n-p) correction)
#   - hc2_bm (CR2 Bell-McCaffrey at unit; per-coef DOF via coef_test()$df_Satt;
#     overall ATT BM contrast DOF via Wald_test(test="HTZ")$df_denom)
#   - classical (lm() summary's heteroskedasticity-only SE)
#   - hc2 (sandwich::vcovHC type="HC2"; no clustering)
#
# clubSandwich >= 0.7.0 required (matches PR #475 / PR #479 pin).

suppressPackageStartupMessages({
  library(clubSandwich)
  library(sandwich)
  library(jsonlite)
})

stopifnot(packageVersion("clubSandwich") >= "0.7.0")
stopifnot(packageVersion("sandwich") >= "3.0.0")

panel_path <- file.path("benchmarks", "data", "wooldridge_test_panel.csv")
out_path <- file.path("benchmarks", "data", "wooldridge_golden.json")

df <- read.csv(panel_path)
stopifnot(all(c("unit", "time", "cohort", "y") %in% names(df)))
# Force integer types on unit/time/cohort so the cluster formula resolves
# cleanly (clubSandwich's `cluster = df$unit` calls `unique(model$unit)` which
# fails on factor-coerced columns from intermediate model frames).
df$unit <- as.integer(df$unit)
df$time <- as.integer(df$time)
df$cohort <- as.integer(df$cohort)

# Build treated (g, t) interaction dummies, matching the Python OLS path's
# `_build_interaction_matrix` (control_group="not_yet_treated", anticipation=0):
# one indicator per treated (g, t) cell with g > 0 and t >= g.
treated_cohorts <- sort(unique(df$cohort[df$cohort > 0]))
times <- sort(unique(df$time))
gt_pairs <- list()
for (g in treated_cohorts) {
  for (t in times) {
    if (t >= g) {
      gt_pairs[[length(gt_pairs) + 1L]] <- c(g, t)
    }
  }
}
gt_names <- vapply(gt_pairs, function(p) sprintf("D_%d_%d", p[1], p[2]), character(1))
for (i in seq_along(gt_pairs)) {
  g <- gt_pairs[[i]][1]
  t <- gt_pairs[[i]][2]
  df[[gt_names[i]]] <- as.integer((df$cohort == g) & (df$time == t))
}
n_int <- length(gt_names)

# Fit lm(y ~ <interactions> + as.factor(unit) + as.factor(time)). The
# `as.factor(...)` form drops the first level of each FE block, matching the
# Python full-dummy build (`drop_first=True` on `pd.get_dummies(unit)` and
# `pd.get_dummies(time)`), and adds a single intercept — matching Python's
# `[intercept, X_design, unit_dummies, time_dummies]`.
formula_str <- paste0(
  "y ~ ", paste(gt_names, collapse = " + "),
  " + as.factor(unit) + as.factor(time)"
)
fit <- lm(as.formula(formula_str), data = df)

# Extract the (interaction) coefficient indices in fit$coefficients. R places
# them right after the intercept (positions 2..(1+n_int) in 1-indexed R).
coef_names <- names(coef(fit))
int_idx <- match(gt_names, coef_names)
stopifnot(!any(is.na(int_idx)))

# Cell weights n_{g,t} for the overall ATT contrast (matches Python's
# `_compute_weighted_agg` with default `weights=n_{g,t}`).
n_gt <- vapply(seq_along(gt_pairs), function(i) {
  g <- gt_pairs[[i]][1]
  t <- gt_pairs[[i]][2]
  sum(df$cohort == g & df$time == t)
}, integer(1))
n_post_total <- sum(n_gt)
contrast_weights <- n_gt / n_post_total  # length n_int

# Build the overall ATT contrast in full-coef space (intercept = 0, then n_int
# weights, then 0 for FE dummies).
n_total_coef <- length(coef_names)
overall_contrast <- numeric(n_total_coef)
overall_contrast[int_idx] <- contrast_weights

# 1. hc1 + CR1S (Stata-style cluster-robust; matches diff-diff's hc1+cluster)
vcov_cr1s <- vcovCR(fit, cluster = df$unit, type = "CR1S")
se_hc1 <- sqrt(diag(vcov_cr1s)[int_idx])
overall_se_hc1 <- sqrt(
  t(overall_contrast) %*% vcov_cr1s %*% overall_contrast
)[1, 1]

# 2. hc2_bm + CR2 + BM Satterthwaite DOF
vcov_cr2 <- vcovCR(fit, cluster = df$unit, type = "CR2")
se_hc2_bm <- sqrt(diag(vcov_cr2)[int_idx])
coef_test_out <- coef_test(fit, vcov = vcov_cr2, test = "Satterthwaite")
df_satt_hc2_bm <- coef_test_out$df[int_idx]

# Overall ATT BM contrast DOF via Wald_test (HTZ reduces to Satterthwaite on
# 1-row constraint matrices; df_denom is the BM contrast DOF).
constraint_matrix <- matrix(overall_contrast, nrow = 1)
overall_dof_hc2_bm <- tryCatch(
  {
    wt <- Wald_test(
      fit,
      constraints = constrain_equal(int_idx, reg_ex = FALSE),
      vcov = vcov_cr2,
      test = "HTZ"
    )
    # HTZ test on multi-row constraints reports a single F + df_num/df_denom
    # row; df_denom is the Bell-McCaffrey-style aggregated DOF.
    wt$df_denom
  },
  error = function(e) NA_real_
)

# For the OVERALL ATT scalar contrast (1-row weights vector), build directly:
# Wald_test with `constraints` requiring a list of `constrain_*` calls
# (clubSandwich >= 0.5.0); for an arbitrary linear contrast pass the matrix
# directly via `constraints = matrix(...)`. The `df_denom` is the BM
# Satterthwaite DOF for the scalar contrast.
overall_wt <- tryCatch(
  Wald_test(
    fit,
    constraints = constraint_matrix,
    vcov = vcov_cr2,
    test = "HTZ"
  ),
  error = function(e) NULL
)
overall_att_contrast_dof <- if (!is.null(overall_wt)) overall_wt$df_denom else NA_real_

overall_se_hc2_bm <- sqrt(
  t(overall_contrast) %*% vcov_cr2 %*% overall_contrast
)[1, 1]

# 3. classical (lm summary SE; OLS sigma^2 * (X'X)^-1)
vcov_classical <- vcov(fit)
se_classical <- sqrt(diag(vcov_classical)[int_idx])
overall_se_classical <- sqrt(
  t(overall_contrast) %*% vcov_classical %*% overall_contrast
)[1, 1]

# 4. hc2 (sandwich::vcovHC type="HC2"; no clustering)
vcov_hc2 <- vcovHC(fit, type = "HC2")
se_hc2 <- sqrt(diag(vcov_hc2)[int_idx])
overall_se_hc2 <- sqrt(
  t(overall_contrast) %*% vcov_hc2 %*% overall_contrast
)[1, 1]

# 5. Aggregate hc2_bm BM contrast DOFs for group / calendar / event
# aggregations. These mirror WooldridgeDiDResults.aggregate(...) at fit time:
# each aggregation key gets a 1-row constraint matrix in full-coef space whose
# entries are the per-cell `n_{g,t} / w_total` weights at the (g, t) coefficient
# columns. Compute the BM Satterthwaite DOF via Wald_test(test="HTZ"). diff-diff
# uses lazy contrast-DOF computation in aggregate() with the same algebra;
# pinning here proves R-parity across all three non-simple aggregation surfaces.
build_contrast_for_cells <- function(cells, weights_by_pair) {
  col <- numeric(n_total_coef)
  if (length(cells) == 0L) return(NULL)
  w_total <- sum(vapply(cells, function(p) weights_by_pair[[paste(p, collapse = "_")]], numeric(1)))
  if (w_total == 0) return(NULL)
  for (p in cells) {
    key <- paste(p, collapse = "_")
    cell_w <- weights_by_pair[[key]]
    # find the lm coef index for D_{g}_{t}
    nm <- sprintf("D_%d_%d", p[1], p[2])
    pos <- match(nm, names(coef(fit)))
    if (!is.na(pos)) {
      col[pos] <- cell_w / w_total
    }
  }
  col
}
weights_by_pair <- setNames(as.list(n_gt), vapply(gt_pairs, function(p) paste(p, collapse = "_"), character(1)))

compute_bm_dof_for_contrast <- function(col) {
  if (is.null(col)) return(NA_real_)
  cm <- matrix(col, nrow = 1)
  wt <- tryCatch(
    Wald_test(fit, constraints = cm, vcov = vcov_cr2, test = "HTZ"),
    error = function(e) NULL
  )
  if (is.null(wt)) NA_real_ else wt$df_denom
}

# group: one contrast per treated cohort g, cells = (g, t) for t >= g
agg_group_dofs <- list()
agg_group_keys <- treated_cohorts
for (g in treated_cohorts) {
  cells <- lapply(gt_pairs, function(p) if (p[1] == g && p[2] >= g) p else NULL)
  cells <- Filter(Negate(is.null), cells)
  col <- build_contrast_for_cells(cells, weights_by_pair)
  agg_group_dofs[[as.character(g)]] <- compute_bm_dof_for_contrast(col)
}

# calendar: one contrast per time period t, cells = (g, t) for g > 0 and t >= g
agg_calendar_dofs <- list()
agg_calendar_keys <- times
for (t in times) {
  cells <- lapply(gt_pairs, function(p) if (p[2] == t && p[1] <= t) p else NULL)
  cells <- Filter(Negate(is.null), cells)
  col <- build_contrast_for_cells(cells, weights_by_pair)
  agg_calendar_dofs[[as.character(t)]] <- compute_bm_dof_for_contrast(col)
}

# event: one contrast per relative period k = t - g
all_k <- sort(unique(vapply(gt_pairs, function(p) p[2] - p[1], numeric(1))))
agg_event_dofs <- list()
for (k in all_k) {
  cells <- lapply(gt_pairs, function(p) if ((p[2] - p[1]) == k) p else NULL)
  cells <- Filter(Negate(is.null), cells)
  col <- build_contrast_for_cells(cells, weights_by_pair)
  agg_event_dofs[[as.character(k)]] <- compute_bm_dof_for_contrast(col)
}

# Coefficient point estimates (for cross-check; identical across all 4 variants
# since they share the lm fit).
beta_int <- coef(fit)[int_idx]

golden <- list(
  meta = list(
    panel_csv = panel_path,
    n_obs = nrow(df),
    n_units = length(unique(df$unit)),
    n_periods = length(times),
    cohorts = sort(unique(df$cohort)),
    gt_pairs = lapply(gt_pairs, function(p) list(g = p[1], t = p[2])),
    n_int = n_int,
    n_post_total = n_post_total,
    contrast_weights = contrast_weights,
    clubsandwich_version = as.character(packageVersion("clubSandwich")),
    sandwich_version = as.character(packageVersion("sandwich"))
  ),
  point_estimates = list(
    interaction_coefs = unname(beta_int),
    gt_keys = lapply(gt_pairs, function(p) list(g = p[1], t = p[2]))
  ),
  hc1 = list(
    per_coef_se = unname(se_hc1),
    overall_att_se = overall_se_hc1
  ),
  hc2_bm = list(
    per_coef_se = unname(se_hc2_bm),
    per_coef_df_satt = unname(df_satt_hc2_bm),
    overall_att_se = overall_se_hc2_bm,
    overall_att_contrast_dof = overall_att_contrast_dof,
    aggregate_group_dof = agg_group_dofs,
    aggregate_calendar_dof = agg_calendar_dofs,
    aggregate_event_dof = agg_event_dofs,
    aggregate_event_keys = all_k
  ),
  classical = list(
    per_coef_se = unname(se_classical),
    overall_att_se = overall_se_classical
  ),
  hc2 = list(
    per_coef_se = unname(se_hc2),
    overall_att_se = overall_se_hc2
  )
)

write_json(golden, out_path, auto_unbox = TRUE, pretty = TRUE, digits = 18)
cat(sprintf("Wrote %s\n", out_path))
cat(sprintf("  n_obs=%d, n_int=%d, n_units=%d\n",
            nrow(df), n_int, length(unique(df$unit))))
cat(sprintf("  hc1 overall_se=%.10f\n", overall_se_hc1))
cat(sprintf("  hc2_bm overall_se=%.10f, overall_dof=%.4f\n",
            overall_se_hc2_bm, overall_att_contrast_dof))
cat(sprintf("  classical overall_se=%.10f\n", overall_se_classical))
cat(sprintf("  hc2 overall_se=%.10f\n", overall_se_hc2))
