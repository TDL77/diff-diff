# R `conleyreg` parity benchmark for Conley spatial HAC SE

`benchmarks/R/generate_conley_golden.R` produces the golden JSON used by
`tests/test_conley_vcov.py::TestConleyParityR` to verify that diff-diff's
`vcov_type="conley"` matches R `conleyreg` (Düsterhöft 2021, CRAN v0.1.9)
to ≤ 1e-6 on three benchmark fixtures.

## Why R `conleyreg`

`conleyreg` is the canonical open-source Conley (1999) implementation in R
(Christian Düsterhöft, https://github.com/cdueben/conleyreg). It uses
RcppArmadillo for the inner loops and is widely cited in applied work.
Stata `acreg` (Colella et al. 2019) is the parallel canonical
implementation in the Stata ecosystem; we cite both in REGISTRY but only
parity-test against `conleyreg` because it is free and open source.

## Earth radius constant

`conleyreg::haversine_dist` uses **6371.01 km** (mean Earth radius) — see
[`src/distance_functions.cpp`](https://github.com/cdueben/conleyreg/blob/master/src/distance_functions.cpp).
diff-diff's `_CONLEY_EARTH_RADIUS_KM` is set to `6371.01` to match. WGS-84
equatorial radius is 6378.137 km; the 0.01 km vs 6371.0 delta is
methodologically negligible (Earth mean radius is approximate at many
more digits) but matters for the 1e-6 cross-language parity bound.

## Regenerating the fixtures

Requires:
- R installed (`/opt/homebrew/bin/Rscript` on Apple Silicon Mac)
- System libraries: `brew install gdal proj geos pkg-config udunits`
  (needed by sf, lwgeom — transitive deps of conleyreg)
- R packages: `Rscript -e 'install.packages(c("conleyreg","sf","lwgeom","jsonlite"))'`

```bash
cd benchmarks/R
Rscript generate_conley_golden.R
# Produces benchmarks/data/r_conleyreg_conley_golden.json
```

The output JSON is **committed to the repo** so CI doesn't need R. Only
re-run when:
- conleyreg is updated (verify version in `meta.tool` field)
- The set of benchmark fixtures changes

## Skip behavior

`tests/test_conley_vcov.py::TestConleyParityR` calls
`pytest.skip("Golden JSON not present...")` when the JSON is absent, so
CI passes without R. The 64 internal tests (`TestConleyKernels`,
`TestConleyDistanceMetrics`, `TestConleyReductions`,
`TestConleyDirectHelper`, `TestConleyValidatorHelpers`,
`TestConleyValidationDispatch`, `TestConleyEstimatorIntegration`,
`TestConleyTWFE`, `TestConleyEstimatorValidation`,
`TestConleySetParamsAtomicity`, `TestConleyLinearRegression`,
`TestConleyReductionsAddendum`) verify the implementation independently.

## Fixtures

Six fixtures total: three cross-sectional (Phase 1) and three panel
fixtures with `lag_cutoff > 0` (Phase 2, block-decomposed Conley).

**Cross-sectional** (`build_fixture`, `lag_cutoff=0`):

| Fixture | n | k | Cutoff | Stress test |
|---|---|---|---|---|
| `small_haversine` | 50 | 2 | 500 km | Small-n, simple regressor |
| `dense_haversine` | 200 | 3 | 1000 km | Dense, 2 covariates, large cutoff |
| `lat_lon_realistic` | 300 | 3 | 200 km | Continental US lat/lon range |

**Panel block-decomposed** (`build_panel_fixture`, `lag_cutoff > 0`):

| Fixture | n_units × T | k | Cutoff | Lag | Stress test |
|---|---|---|---|---|---|
| `panel_haversine_lag1` | 60 × 3 | 2 | 500 km | 1 | Short panel, 1-period serial |
| `panel_haversine_lag2` | 80 × 5 | 3 | 1000 km | 2 | Longer panel, 2-period serial |
| `panel_lat_lon_realistic_lag1` | 100 × 4 | 3 | 200 km | 1 | Continental US, 1-period serial |

Each unit's `(lat, lon)` is time-invariant within the panel fixtures; the
block-decomposed sandwich (within-period spatial + within-unit Bartlett
serial) is independent of `lag_cutoff` for within-period contributions
and matches R `conleyreg::time_dist.cpp` for the serial component.

The euclidean code path (`conley_metric="euclidean"`) is verified
internally against `scipy.spatial.distance.cdist` in
`tests/test_conley_vcov.py::TestConleyDistanceMetrics::test_pairwise_distance_euclidean_matches_pdist`.
conleyreg's planar code path requires an `sf` CRS specification, which
adds noise without methodological value for parity testing.

## JSON schema

```json
{
  "meta": {
    "generated_at": "2026-05-10",
    "earth_radius_km": 6371.01,
    "tool": "R conleyreg 0.1.9 (Düsterhöft 2021)"
  },
  "small_haversine": {
    "x": [<n*k floats, row-major>],
    "x_shape": [n, k],
    "y": [<n floats>],
    "coords": [<n*2 floats: lat, lon, row-major>],
    "coords_shape": [n, 2],
    "metric": "haversine",
    "cutoff_km": 500.0,
    "kernel": "bartlett",
    "vcov": [<k*k floats, row-major>],
    "vcov_shape": [k, k],
    "n": <int>,
    "k": <int>
  },
  "dense_haversine": { ... },
  "lat_lon_realistic": { ... }
}
```

The R script transposes matrices before `as.vector` flatten so that
NumPy's `np.asarray(...).reshape(shape)` (row-major / C-order) decodes
the same orientation R wrote. Without the transpose, R's column-major
flatten misaligns when reshaped row-major.

## Known constraints

- `conleyreg` requires `unit` and `time` columns even with `lag_cutoff=0`
  (cross-sectional). The script fakes them with `unit = 1:n, time = 1L`;
  conleyreg emits a `Number of time periods: 1. Treating data as
  cross-sectional` warning which is informational.
- `conleyreg` uses OpenMP for parallelism; on macOS Apple Silicon with
  R's default toolchain, the `OpenMP not detected` warning is normal —
  the package falls back to single-threaded mode without affecting
  numerical output.
