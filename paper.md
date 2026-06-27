---
title: "diff-diff: Comprehensive Difference-in-Differences Causal Inference for Python"
tags:
  - difference-in-differences
  - causal-inference
  - econometrics
  - Python
  - treatment-effects
  - survey-data
authors:
  - name: Isaac Gerber
    orcid: 0009-0009-3275-5591
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 3 July 2026
bibliography: paper.bib
---

# Summary

`diff-diff` is a Python library for Difference-in-Differences (DiD) causal inference
analysis. It provides 19 estimators covering the full modern DiD toolkit - from classic
two-group/two-period designs through heterogeneity-robust staggered adoption methods,
synthetic control hybrids, and sensitivity analysis - under a consistent scikit-learn-style
API. Most estimators accept an optional `SurveyDesign` object for design-based variance
estimation with complex survey data, a capability absent from existing DiD software in any
language; the underlying design-based variance methodology is derived in the companion
preprint [@Gerber2026]. Point estimates are validated against established R packages to
machine precision, with standard errors matching exactly or to sub-percent relative
differences.

# Statement of Need

Difference-in-differences is the most widely used quasi-experimental research design in
applied economics and the social sciences. Since 2018, a wave of methodological advances
has addressed fundamental limitations of the conventional two-way fixed effects (TWFE)
estimator under staggered treatment adoption and heterogeneous effects [@Roth2023]. These
modern methods - including Callaway and Sant'Anna [-@Callaway2021], Sun and Abraham
[-@Sun2021], Borusyak, Jaravel, and Spiess [-@Borusyak2024], and others - are now standard
practice in applied work.

The R ecosystem provides mature implementations across several packages: `did`
[@Callaway2021], `fixest` [@Berge2018], `synthdid` [@Arkhangelsky2021], and `HonestDiD`
[@Rambachan2023]. Stata offers `csdid` and `didregress`. Python, however, lacks a unified
DiD library. Practitioners working in Python-based data science workflows - increasingly
common in industry settings for marketing measurement, product experimentation, and policy
evaluation - must either context-switch to R, reimplement methods from scratch, or rely on
partial implementations scattered across unrelated packages.

`diff-diff` fills this gap by providing a single-import library that covers 19 estimators
with a consistent API, survey-weighted inference, and numerical validation against R. It
is also the companion software for the design-based variance framework of @Gerber2026,
which establishes design-consistent standard errors for modern DiD estimators under
complex survey designs. It targets both applied researchers who need rigorous econometric
methods and data science practitioners who need accessible causal inference tools
integrated into Python workflows.

# Key Features

**Breadth of methods.** `diff-diff` implements 19 estimators organized across the modern
DiD taxonomy. Classic designs include two-group/two-period DiD, two-way fixed effects, and
event-study estimation with period-specific effects. Heterogeneity-robust staggered-adoption
estimators include Callaway-Sant'Anna [@Callaway2021], Sun-Abraham [@Sun2021], imputation
[@Borusyak2024], two-stage [@Gardner2022], stacked [@Wing2024], and efficient [@Chen2025]
approaches, together with reversible-treatment DiD for non-absorbing interventions
[@deChaisemartin2020] and a ring-indicator estimator for spatial spillovers [@Butts2021].
Synthetic-control hybrids include synthetic DiD [@Arkhangelsky2021] and the classic
synthetic control method [@Abadie2010]. Extended designs include triple-difference and
staggered triple-difference estimators [@OrtizVillavicencio2025], continuous-treatment DiD
with dose-response curves [@Callaway2024], heterogeneous-adoption designs where no unit
remains untreated [@deChaisemartin2026], nonlinear ETWFE [@Wooldridge2025; @Wooldridge2023],
and triply robust panel estimation [@Athey2025]. Separate diagnostic and sensitivity tools -
outside the 19 estimators - include Goodman-Bacon decomposition [@GoodmanBacon2021], Honest
DiD sensitivity analysis [@Rambachan2023], placebo tests, and pre-trends power analysis
[@Roth2022]. All estimators share a consistent `fit()` interface with
`get_params()`/`set_params()` for configuration, R-style formula support, and rich results
objects with `summary()` output. An optional Rust backend via PyO3 accelerates
compute-intensive operations.

**Survey-weighted inference.** A `SurveyDesign` class supports stratification, primary
sampling units, finite population corrections, and probability weights. Variance estimation
includes Taylor series linearization, five replicate weight methods (BRR, Fay's BRR, JK1,
JKn, SDR), and survey-aware bootstrap. Survey variance is validated against R's `survey`
package [@Lumley2004] on three real complex-survey datasets - NHANES, RECS 2020, and the
California API school dataset - to a tight tolerance (test gaps < 1e-8, typically below
1e-10). The design-based variance result - that the influence functions of modern DiD
estimators satisfy Binder's (1983) smoothness conditions, so stratified-cluster
linearization yields design-consistent standard errors - is derived in @Gerber2026. No
other DiD package in any language provides integrated survey support.

**Validation against R.** Point estimates match the R `did`, `synthdid`, and `fixest`
packages to machine precision (differences < 1e-10). Standard errors match exactly for
core estimators including Callaway-Sant'Anna and basic DiD. Validation includes the
canonical MPDTA minimum-wage dataset from Callaway and Sant'Anna [-@Callaway2021].

**Practitioner tooling.** Beyond estimation, `diff-diff` includes a practitioner decision
tree for estimator selection, an 8-step diagnostic workflow based on Baker et al.
[-@Baker2025], AI agent integration with structured next-steps guidance, and microdata
aggregation utilities for converting individual-level survey responses into
geographic-period panels suitable for DiD analysis.

# AI Usage Disclosure

Generative AI tools were used in developing this software and manuscript. Anthropic's
Claude models (the Opus and Sonnet families, via the Claude Code CLI) assisted with code
generation and refactoring, test scaffolding, documentation, and drafting and editing of
this manuscript. The author reviewed, modified, and validated all AI-generated code and
text and made all primary architectural and methodological decisions. Numerical results
were independently verified against established R reference packages (`did`, `synthdid`,
`fixest`, `survey`) for every estimator with an R equivalent, and against the author's
reference derivations or simulation otherwise. The author takes full responsibility for the
accuracy and integrity of the software and this paper.

# References
