"""Tests for memory-bounded multiplier-bootstrap weight chunking.

The chunking in :mod:`diff_diff.bootstrap_chunking` tiles the bootstrap *draw*
dimension to cap peak memory at ``O(block x n_units)`` instead of
``O(n_bootstrap x n_units)``. Its load-bearing guarantee is that tiling
reproduces the un-chunked weight *stream* exactly (bit-identical), on whichever
backend is active (Rust absolute per-row seeding; NumPy in-order stream). These
tests lock the weight-stream bit-identity at the helper level and end-to-end
chunk-invariance (to floating-point reassociation) through CallawaySantAnna,
under whatever ``DIFF_DIFF_BACKEND`` the CI matrix selects.
"""

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from diff_diff import CallawaySantAnna, EfficientDiD, HeterogeneousAdoptionDiD
from diff_diff.bootstrap_chunking import (
    compute_block_size,
    iter_survey_multiplier_weight_blocks,
    iter_weight_blocks,
)
from diff_diff.bootstrap_utils import (
    generate_bootstrap_weights_batch,
    generate_survey_multiplier_weights_batch,
)

WEIGHT_TYPES = ["rademacher", "mammen", "webb"]


def _stack(n_bootstrap, n_gen, weight_type, seed, block_size, expand_index=None):
    """Concatenate all weight blocks from a fresh, identically-seeded rng."""
    rng = np.random.default_rng(seed)
    blocks = list(
        iter_weight_blocks(
            n_bootstrap,
            n_gen,
            weight_type,
            rng,
            expand_index=expand_index,
            block_size=block_size,
        )
    )
    starts = [cs for cs, _ in blocks]
    mat = np.vstack([b for _, b in blocks])
    return starts, mat


class TestComputeBlockSize:
    def test_always_in_bounds(self):
        assert compute_block_size(1000, 200) <= 200
        assert compute_block_size(1000, 200) >= 1

    def test_huge_n_units_floors_to_one_row(self):
        assert compute_block_size(10**9, 200) == 1

    def test_tiny_n_units_fits_all_in_one_block(self):
        assert compute_block_size(1, 200) == 200

    def test_respects_target_bytes(self):
        # 100 units x 8 bytes = 800 B/row; an 8000 B budget -> 10 rows/block
        assert compute_block_size(100, 500, target_bytes=8000) == 10


class TestWeightStreamBitIdentity:
    """Tiling the draw dimension reproduces the single-block stream exactly."""

    @pytest.mark.parametrize("weight_type", WEIGHT_TYPES)
    @pytest.mark.parametrize("block_size", [1, 7, 33, 198])
    def test_chunked_equals_single_block(self, weight_type, block_size):
        n_bootstrap, n_gen, seed = 199, 53, 12345
        _, single = _stack(n_bootstrap, n_gen, weight_type, seed, block_size=n_bootstrap)
        starts, chunked = _stack(n_bootstrap, n_gen, weight_type, seed, block_size=block_size)
        assert single.shape == (n_bootstrap, n_gen)
        assert chunked.shape == (n_bootstrap, n_gen)
        # exact: the chunking promise is bit-identity, not approximate equality
        np.testing.assert_array_equal(chunked, single)
        # blocks cover every draw exactly once, in order
        assert starts == list(range(0, n_bootstrap, block_size))

    @pytest.mark.parametrize("weight_type", WEIGHT_TYPES)
    def test_expand_index_is_chunk_invariant(self, weight_type):
        # cluster/PSU fan-out: generate at n_gen, expand to unit width per block
        n_bootstrap, n_clusters, n_units, seed = 100, 9, 40, 7
        expand = np.array([i % n_clusters for i in range(n_units)])
        _, single = _stack(
            n_bootstrap,
            n_clusters,
            weight_type,
            seed,
            block_size=n_bootstrap,
            expand_index=expand,
        )
        _, chunked = _stack(
            n_bootstrap,
            n_clusters,
            weight_type,
            seed,
            block_size=11,
            expand_index=expand,
        )
        assert single.shape == (n_bootstrap, n_units)
        np.testing.assert_array_equal(chunked, single)

    @pytest.mark.parametrize("weight_type", WEIGHT_TYPES)
    def test_single_block_matches_legacy_generator(self, weight_type):
        # iter_weight_blocks in single-block mode must reproduce the legacy
        # generate_bootstrap_weights_batch wrapper exactly (matched seeds), so the
        # chunked path is anchored to the pre-existing generator, not just to its
        # own single-block mode.
        n_bootstrap, n_gen, seed = 199, 53, 999
        legacy = generate_bootstrap_weights_batch(
            n_bootstrap, n_gen, weight_type, np.random.default_rng(seed)
        )
        _, chunked = _stack(n_bootstrap, n_gen, weight_type, seed, block_size=n_bootstrap)
        assert chunked.shape == legacy.shape
        np.testing.assert_array_equal(chunked, legacy)


class TestCSBootstrapChunkInvariance:
    """CallawaySantAnna bootstrap output is invariant to the chunk size.

    The generated weight stream is bit-identical across chunk sizes (locked by
    ``TestWeightStreamBitIdentity``). The downstream ``weights @ influence``
    matmuls go through BLAS, whose reduction order depends on the operand
    row-count, so the resulting statistics match to within floating-point
    reassociation (~1 ULP) rather than bit-for-bit -- far below bootstrap
    Monte-Carlo error. This mirrors the repo's assert_allclose convention for
    float linalg.
    """

    @staticmethod
    def _panel():
        rng = np.random.default_rng(0)
        nu, nt = 120, 8
        units = np.repeat(np.arange(nu), nt)
        periods = np.tile(np.arange(nt), nu)
        n = nu * nt
        cohort = rng.integers(0, 3, nu)
        ft_unit = np.where(cohort == 0, 0, np.where(cohort == 1, 3, 5))
        ft = np.repeat(ft_unit, nt)
        post = (periods >= ft) & (ft > 0)
        y = rng.standard_normal(n) + 0.1 * periods + 2.0 * post + 0.5 * np.repeat(cohort, nt)
        return pd.DataFrame({"unit": units, "period": periods, "y": y, "first_treat": ft})

    def _fit(self):
        return CallawaySantAnna(
            control_group="never_treated",
            estimation_method="dr",
            cluster="unit",
            n_bootstrap=200,
            seed=42,
        ).fit(
            self._panel(),
            outcome="y",
            unit="unit",
            time="period",
            first_treat="first_treat",
            aggregate="all",
        )

    def test_tiny_chunks_match_single_chunk(self, monkeypatch):
        # Default path: this small panel fits in a single block.
        base = self._fit()
        # Force many tiny blocks on every weight path (unit + survey).
        monkeypatch.setattr("diff_diff.bootstrap_chunking.compute_block_size", lambda *a, **k: 7)
        monkeypatch.setattr("diff_diff.staggered_bootstrap.compute_block_size", lambda *a, **k: 7)
        tiny = self._fit()

        # Continuous bootstrap statistics match to within BLAS reassociation.
        assert tiny.overall_se == pytest.approx(base.overall_se, rel=1e-10, abs=1e-12)
        assert tiny.cband_crit_value == pytest.approx(base.cband_crit_value, rel=1e-10, abs=1e-12)
        # p-values are discrete proportions over draws; a borderline draw can
        # flip under reassociation, shifting a p-value by O(1/n_bootstrap).
        assert tiny.overall_p_value == pytest.approx(base.overall_p_value, abs=0.02)

        b = base.to_dataframe().sort_values(["group", "time"]).reset_index(drop=True)
        t = tiny.to_dataframe().sort_values(["group", "time"]).reset_index(drop=True)
        for col in ["se", "conf_int_lower", "conf_int_upper"]:
            np.testing.assert_allclose(t[col].to_numpy(), b[col].to_numpy(), rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(t["p_value"].to_numpy(), b["p_value"].to_numpy(), atol=0.02)

        # Event-study and group aggregate effects/SEs/CIs also match under chunking.
        for level in ("event_study", "group"):
            bl = base.to_dataframe(level=level).reset_index(drop=True)
            tl = tiny.to_dataframe(level=level).reset_index(drop=True)
            num_cols = [c for c in bl.columns if bl[c].dtype.kind in "fi" and c != "p_value"]
            assert num_cols, f"no numeric columns to compare for level={level}"
            for col in num_cols:
                np.testing.assert_allclose(
                    tl[col].to_numpy(), bl[col].to_numpy(), rtol=1e-9, atol=1e-10
                )

    def test_cluster_none_default_chunks_match_single(self, monkeypatch):
        # cluster=None is the public default (auto-clusters at unit); confirm the
        # default path is chunk-invariant end-to-end.
        def fit():
            return CallawaySantAnna(
                control_group="never_treated",
                estimation_method="dr",
                cluster=None,
                n_bootstrap=200,
                seed=42,
            ).fit(
                self._panel(),
                outcome="y",
                unit="unit",
                time="period",
                first_treat="first_treat",
                aggregate="simple",
            )

        base = fit()
        monkeypatch.setattr("diff_diff.bootstrap_chunking.compute_block_size", lambda *a, **k: 7)
        monkeypatch.setattr("diff_diff.staggered_bootstrap.compute_block_size", lambda *a, **k: 7)
        tiny = fit()
        assert tiny.overall_se == pytest.approx(base.overall_se, rel=1e-10, abs=1e-12)

    @staticmethod
    def _clustered_panel():
        # Units grouped into states (5 units/state) -> the unit->PSU map is a
        # genuine many-units-to-one-PSU fan-out (non-identity expansion), unlike
        # cluster="unit" above.
        rng = np.random.default_rng(1)
        n_states, units_per_state, nt = 10, 5, 8
        nu = n_states * units_per_state
        units = np.repeat(np.arange(nu), nt)
        periods = np.tile(np.arange(nt), nu)
        n = nu * nt
        cohort = rng.integers(0, 3, nu)
        ft_unit = np.where(cohort == 0, 0, np.where(cohort == 1, 3, 5))
        ft = np.repeat(ft_unit, nt)
        post = (periods >= ft) & (ft > 0)
        y = rng.standard_normal(n) + 0.1 * periods + 2.0 * post + 0.5 * np.repeat(cohort, nt)
        state = np.repeat(np.repeat(np.arange(n_states), units_per_state), nt)
        return pd.DataFrame(
            {"unit": units, "period": periods, "y": y, "first_treat": ft, "state": state}
        )

    def _fit_clustered(self):
        return CallawaySantAnna(
            control_group="never_treated",
            estimation_method="dr",
            cluster="state",
            n_bootstrap=200,
            seed=42,
        ).fit(
            self._clustered_panel(),
            outcome="y",
            unit="unit",
            time="period",
            first_treat="first_treat",
            aggregate="all",
        )

    def test_nonidentity_cluster_chunks_match_single(self, monkeypatch):
        # Exercises the non-identity PSU fan-out expansion under tiny chunks.
        base = self._fit_clustered()
        monkeypatch.setattr("diff_diff.bootstrap_chunking.compute_block_size", lambda *a, **k: 7)
        monkeypatch.setattr("diff_diff.staggered_bootstrap.compute_block_size", lambda *a, **k: 7)
        tiny = self._fit_clustered()

        assert tiny.overall_se == pytest.approx(base.overall_se, rel=1e-10, abs=1e-12)
        b = base.to_dataframe().sort_values(["group", "time"]).reset_index(drop=True)
        t = tiny.to_dataframe().sort_values(["group", "time"]).reset_index(drop=True)
        for col in ["se", "conf_int_lower", "conf_int_upper"]:
            np.testing.assert_allclose(t[col].to_numpy(), b[col].to_numpy(), rtol=1e-10, atol=1e-12)


def _design(psu=None, strata=None, fpc=None, weights=None, lonely_psu="adjust"):
    """Minimal duck-typed ResolvedSurveyDesign for the survey weight generators."""
    return SimpleNamespace(psu=psu, strata=strata, fpc=fpc, weights=weights, lonely_psu=lonely_psu)


class TestSurveyWeightBlocks:
    """`iter_survey_multiplier_weight_blocks` reproduces the full survey generator.

    The chunked survey path must yield the exact same PSU-weight matrix and
    psu_ids as `generate_survey_multiplier_weights_batch`, so the
    CallawaySantAnna survey/cluster bootstrap is bit-identical regardless of
    chunking. Covers unstratified generation (tiled), `psu=None`, the FPC
    scalar, and the stratified fallback (sliced).
    """

    def _assert_matches_full(self, design, weight_type, seed, block_size):
        full_w, full_ids = generate_survey_multiplier_weights_batch(
            199, design, weight_type, np.random.default_rng(seed)
        )
        ids, blocks = iter_survey_multiplier_weight_blocks(
            199, design, weight_type, np.random.default_rng(seed), block_size=block_size
        )
        chunked = np.vstack([b for _, b in blocks])
        np.testing.assert_array_equal(ids, full_ids)
        assert chunked.shape == full_w.shape
        np.testing.assert_array_equal(chunked, full_w)

    @pytest.mark.parametrize("weight_type", WEIGHT_TYPES)
    def test_unstratified_psu_tiled_matches_full(self, weight_type):
        # 8 PSUs, 3 units each; tiled generation (block_size << n_bootstrap)
        psu = np.repeat(np.arange(8), 3)
        design = _design(psu=psu, weights=np.ones(len(psu)))
        self._assert_matches_full(design, weight_type, seed=5, block_size=7)

    @pytest.mark.parametrize("weight_type", WEIGHT_TYPES)
    def test_psu_none_matches_full(self, weight_type):
        # psu=None -> each observation is its own PSU
        design = _design(psu=None, weights=np.ones(20))
        self._assert_matches_full(design, weight_type, seed=9, block_size=11)

    def test_unstratified_fpc_scaling_matches_full(self):
        # f = n_psu / fpc = 10/100 = 0.1 -> sqrt(0.9) scaling on every weight
        psu = np.arange(10)
        design = _design(psu=psu, fpc=np.full(10, 100.0), weights=np.ones(10))
        self._assert_matches_full(design, "rademacher", seed=3, block_size=13)

    def test_stratified_fallback_matches_full(self):
        # 2 strata x 3 PSUs -> falls back to full generation + sliced blocks
        psu = np.arange(6)
        strata = np.array([0, 0, 0, 1, 1, 1])
        design = _design(psu=psu, strata=strata, weights=np.ones(6))
        self._assert_matches_full(design, "rademacher", seed=7, block_size=9)


class TestEfficientDiDBootstrapChunkInvariance:
    """EfficientDiD multiplier bootstrap is invariant to the chunk size.

    Mirrors ``TestCSBootstrapChunkInvariance``: the weight stream is bit-identical
    across chunk sizes; the ``weights @ eif`` matmuls reassociate under BLAS, so
    SEs match to ~1 ULP (assert_allclose, not bit-for-bit). Covers all four
    bootstrap paths: unit (cluster=None), cluster (genuine many-units-to-cluster
    fan-out), survey-PSU, and weights-only ``SurveyDesign`` -- the last exercises
    the unit_level_weights / weight-path decoupling (unit weight generation but
    eif_scaled perturbation), which a "survey vs non-survey" mis-keying would
    silently break.
    """

    @staticmethod
    def _panel():
        rng = np.random.default_rng(2)
        n_states, units_per_state, nt = 10, 6, 6
        nu = n_states * units_per_state
        units = np.repeat(np.arange(nu), nt)
        periods = np.tile(np.arange(nt), nu)
        n = nu * nt
        cohort = rng.integers(0, 3, nu)
        ft_unit = np.where(cohort == 0, 0, np.where(cohort == 1, 3, 4))
        ft = np.repeat(ft_unit, nt)
        post = (periods >= ft) & (ft > 0)
        y = rng.standard_normal(n) + 0.1 * periods + 2.0 * post + 0.5 * np.repeat(cohort, nt)
        state = np.repeat(np.repeat(np.arange(n_states), units_per_state), nt)
        w = np.repeat(1.0 + 0.3 * np.abs(rng.standard_normal(nu)), nt)
        return pd.DataFrame(
            {"unit": units, "period": periods, "y": y, "first_treat": ft, "state": state, "w": w}
        )

    def _fit(self, cluster=None, survey_design=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return EfficientDiD(n_bootstrap=200, seed=42, cluster=cluster).fit(
                self._panel(),
                "y",
                "unit",
                "period",
                "first_treat",
                aggregate="all",
                survey_design=survey_design,
            )

    @staticmethod
    def _ses(r):
        # Flatten every bootstrap SE (overall + group_time + event_study + group)
        # into one vector, ordered by sorted keys, for an nan-safe comparison.
        gt = [r.group_time_effects[k]["se"] for k in sorted(r.group_time_effects)]
        es = (
            [r.event_study_effects[k]["se"] for k in sorted(r.event_study_effects)]
            if r.event_study_effects
            else []
        )
        gp = [r.group_effects[k]["se"] for k in sorted(r.group_effects)] if r.group_effects else []
        return np.array([r.overall_se, *gt, *es, *gp], dtype=float)

    def _run(self, monkeypatch, **fit_kwargs):
        base = self._fit(**fit_kwargs)
        base_ses = self._ses(base)
        # Guard the equal_nan comparison below: require the path actually produced
        # finite bootstrap inference (overall SE + at least one cell SE), so a
        # regression that NaN-outs both base and tiny chunk paths cannot pass.
        assert np.isfinite(base_ses[0]) and np.isfinite(base_ses[1:]).any()
        # Force many tiny blocks on every weight path: bootstrap_chunking covers
        # iter_weight_blocks' internal sizing (unit/cluster); the module-level
        # efficient_did_bootstrap target covers the survey-path block_size call.
        monkeypatch.setattr("diff_diff.bootstrap_chunking.compute_block_size", lambda *a, **k: 7)
        monkeypatch.setattr(
            "diff_diff.efficient_did_bootstrap.compute_block_size", lambda *a, **k: 7
        )
        tiny = self._fit(**fit_kwargs)
        np.testing.assert_allclose(self._ses(tiny), base_ses, rtol=1e-9, atol=1e-12, equal_nan=True)

    def test_unit_path(self, monkeypatch):
        self._run(monkeypatch)

    def test_cluster_path(self, monkeypatch):
        self._run(monkeypatch, cluster="state")

    def test_survey_psu_path(self, monkeypatch):
        from diff_diff.survey import SurveyDesign

        self._run(monkeypatch, survey_design=SurveyDesign(psu="state", weights="w"))

    def test_weights_only_survey_path(self, monkeypatch):
        # weights-only SurveyDesign: _use_survey_bootstrap is False (unit weight
        # generation) but unit_level_weights is set (eif_scaled perturbation).
        from diff_diff.survey import SurveyDesign

        self._run(monkeypatch, survey_design=SurveyDesign(weights="w"))


class TestHADBootstrapChunkInvariance:
    """HAD event-study sup-t bootstrap is invariant to the chunk size.

    The ``weights @ influence`` perturbations are tiled over draws into the small
    ``(B, n_horizons)`` matrix; the sup-t reduction (nanmax over horizons, then
    quantile) runs post-loop. The weight stream is bit-identical across chunk
    sizes; the simultaneous-band critical value matches to ~1 ULP. Covers the
    non-survey (iter_weight_blocks) and survey (iter_survey_multiplier_weight_blocks)
    paths.
    """

    @staticmethod
    def _panel():
        rng = np.random.default_rng(73)
        G, T = 150, 4
        d_post = rng.uniform(0.0, 1.0, G)
        rows = []
        for t in range(T):
            for g in range(G):
                dose = d_post[g] if t == T - 1 else 0.0
                y = 0.2 * t + (2.0 * dose if t == T - 1 else 0.0) + 0.5 * rng.standard_normal()
                rows.append((g, t, dose, y))
        panel = pd.DataFrame(rows, columns=["unit", "period", "dose", "outcome"])
        # HAD's continuous path requires unit-CONSTANT sampling weights.
        w_unit = 1.0 + 0.3 * np.abs(rng.standard_normal(G))
        panel["w"] = panel["unit"].map(lambda g: w_unit[g])
        return panel

    def _fit(self):
        from diff_diff.survey import SurveyDesign

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return HeterogeneousAdoptionDiD(
                design="continuous_at_zero", seed=42, n_bootstrap=400
            ).fit(
                self._panel(),
                "outcome",
                "dose",
                "period",
                "unit",
                aggregate="event_study",
                survey_design=SurveyDesign(weights="w"),
            )

    def test_survey_path(self, monkeypatch):
        # Public event-study cband always routes through the survey-aware branch
        # (iter_survey_multiplier_weight_blocks); a weights-only design makes
        # n_psu == n_units, the large-allocation case the chunking targets.
        base = self._fit()
        assert np.isfinite(base.cband_crit_value)
        monkeypatch.setattr("diff_diff.bootstrap_chunking.compute_block_size", lambda *a, **k: 9)
        monkeypatch.setattr("diff_diff.had.compute_block_size", lambda *a, **k: 9)
        tiny = self._fit()
        assert tiny.cband_crit_value == pytest.approx(base.cband_crit_value, rel=1e-8, abs=1e-10)

    def test_nonsurvey_branch_chunk_invariant(self, monkeypatch):
        # The iid (resolved_survey=None) else-branch is unreachable end-to-end --
        # the cband path always builds a (possibly synthetic) survey design, even
        # for the weights= shortcut -- so the refactored iter_weight_blocks path is
        # exercised by a direct call.
        from diff_diff.had import _sup_t_multiplier_bootstrap

        rng = np.random.default_rng(5)
        n_units, n_h = 80, 4
        infl = rng.standard_normal((n_units, n_h))
        att = rng.standard_normal(n_h) * 0.1
        se = np.abs(rng.standard_normal(n_h)) + 0.5

        def _crit():
            return _sup_t_multiplier_bootstrap(
                infl,
                att,
                se,
                None,
                n_bootstrap=400,
                alpha=0.05,
                seed=42,
                bootstrap_weights="rademacher",
            )[0]

        base = _crit()
        assert np.isfinite(base)
        monkeypatch.setattr("diff_diff.bootstrap_chunking.compute_block_size", lambda *a, **k: 9)
        tiny = _crit()
        assert tiny == pytest.approx(base, rel=1e-8, abs=1e-10)
