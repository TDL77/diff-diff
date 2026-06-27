"""Memory-bounded chunking for multiplier-bootstrap weight matrices.

The multiplier bootstrap perturbs cached influence functions with a dense
``(n_bootstrap, n_units)`` weight matrix. At large ``n_units`` that matrix
dominates peak memory (e.g. ``999 x 5_000_000 x 8`` bytes is ~40 GB). Every
consumer is a left-multiply ``weights @ influence_vector`` whose result is small
(``(n_bootstrap,)`` or ``(n_bootstrap, n_gt)``), so the bootstrap can be tiled
over the *draw* dimension: generate and consume the weights in row-blocks of
``B``, capping the live intermediate at ``(B, n_units)``. FLOPs are identical to
the un-chunked path -- only the draw axis is tiled. The generated weight stream
is *bit-identical* to the un-chunked matrix (see below); the downstream
``weights @ influence`` matmuls go through BLAS, whose reduction order depends on
the operand row-count, so the resulting statistics match the un-chunked path to
within floating-point reassociation (typically <~1 ULP), far below bootstrap
Monte-Carlo error -- not bit-for-bit.

Bit-identity of the weight *generation* is preserved on **both** backends:

- **Rust** seeds each row absolutely as ``base_seed + row_index``
  (``rust/src/bootstrap.rs``), so calling the generator per block with base seed
  ``base_seed + chunk_start`` reproduces the exact un-chunked rows. Exactly one
  ``rng.integers`` draw is consumed, matching the un-chunked wrapper.
- The **NumPy** fallback draws the matrix row-major from the ``Generator``
  stream, so consuming it in contiguous, in-order blocks from the same generator
  reproduces the identical sequence.
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np

from diff_diff._backend import HAS_RUST_BACKEND, _rust_bootstrap_weights
from diff_diff.bootstrap_utils import generate_bootstrap_weights_batch_numpy

# Byte ceiling for a single ``(B, n_units)`` float64 weight block. 256 MB keeps
# the live intermediate small at millions of units while staying large enough
# that the per-block matmuls remain BLAS-efficient and chunk overhead (a handful
# of extra Python iterations / FFI calls) is negligible.
_TARGET_BLOCK_BYTES = 256 * 1024 * 1024


def compute_block_size(
    n_units: int, n_bootstrap: int, target_bytes: int = _TARGET_BLOCK_BYTES
) -> int:
    """Number of bootstrap rows per block so a ``(B, n_units)`` float64 block
    stays under ``target_bytes``. Always in ``[1, n_bootstrap]``."""
    if n_units <= 0:
        return max(1, n_bootstrap)
    b = target_bytes // (n_units * 8)
    return int(max(1, min(max(1, n_bootstrap), b)))


def iter_weight_blocks(
    n_bootstrap: int,
    n_gen: int,
    weight_type: str,
    rng: np.random.Generator,
    *,
    expand_index: Optional[np.ndarray] = None,
    block_size: Optional[int] = None,
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield ``(chunk_start, block)`` pairs covering all ``n_bootstrap`` draws.

    ``block`` has shape ``(B, width)`` where ``width = len(expand_index)`` when
    ``expand_index`` is given, else ``n_gen``. Weights are generated at width
    ``n_gen`` (unit / cluster / PSU level) and, when ``expand_index`` is given,
    expanded to unit level via ``block[:, expand_index]`` (cluster->unit or
    PSU->unit fan-out). The concatenation of all yielded blocks is bit-identical
    to a single ``generate_bootstrap_weights_batch(n_bootstrap, n_gen, ...)``
    followed by the same expansion.

    Generation is in-order and stateful on ``rng`` (NumPy fallback) -- the caller
    must consume the iterator sequentially, which the chunk loop does.
    """
    width = n_gen if expand_index is None else int(len(expand_index))
    if block_size is None:
        block_size = compute_block_size(width, n_bootstrap)
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")

    rust_gen = (
        _rust_bootstrap_weights
        if (HAS_RUST_BACKEND and _rust_bootstrap_weights is not None)
        else None
    )
    # Draw exactly one base seed (matching the un-chunked Rust wrapper); the
    # NumPy fallback consumes the rng stream directly per block instead.
    base_seed = int(rng.integers(0, 2**63 - 1)) if rust_gen is not None else 0

    for chunk_start in range(0, n_bootstrap, block_size):
        rows = min(block_size, n_bootstrap - chunk_start)
        if rust_gen is not None:
            block = rust_gen(rows, n_gen, weight_type, base_seed + chunk_start)
        else:
            block = generate_bootstrap_weights_batch_numpy(rows, n_gen, weight_type, rng)
        if expand_index is not None:
            block = block[:, expand_index]
        yield chunk_start, block


def iter_survey_multiplier_weight_blocks(
    n_bootstrap: int,
    resolved_survey: object,
    weight_type: str,
    rng: np.random.Generator,
    *,
    block_size: int,
) -> Tuple[np.ndarray, Iterator[Tuple[int, np.ndarray]]]:
    """Chunked PSU-level multiplier weights for the survey-aware bootstrap.

    Returns ``(psu_ids, blocks)`` where ``blocks`` yields
    ``(chunk_start, (B, n_psu))`` PSU-weight blocks covering all draws.

    For UNSTRATIFIED designs (``strata is None``, ``n_psu >= 2``) the
    ``(n_bootstrap, n_psu)`` matrix is generated one draw-block at a time via
    :func:`iter_weight_blocks` plus the unstratified FPC scalar -- bit-identical
    to the unstratified branch of
    :func:`diff_diff.bootstrap_utils.generate_survey_multiplier_weights_batch`,
    but the full matrix is never materialized. This is the path taken by
    ``cluster="unit"`` (each unit its own PSU, ``n_psu == n_units``), the case
    that otherwise dominates bootstrap memory at large n_units.

    Stratified designs (and the ``n_psu < 2`` degenerate case) fall back to full
    generation + sliced blocks: per-stratum / lonely-PSU generation is not tiled
    here, but stratified designs have few PSUs so the full matrix is small.
    """
    from diff_diff.bootstrap_utils import generate_survey_multiplier_weights_batch

    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")

    psu = getattr(resolved_survey, "psu", None)
    strata = getattr(resolved_survey, "strata", None)
    if psu is None:
        n_psu = len(resolved_survey.weights)  # type: ignore[attr-defined]
        psu_ids = np.arange(n_psu)
    else:
        psu_ids = np.unique(psu)
        n_psu = len(psu_ids)

    if strata is not None or n_psu < 2:
        # Stratified or degenerate single-PSU: full generation (small here).
        weights, psu_ids = generate_survey_multiplier_weights_batch(
            n_bootstrap, resolved_survey, weight_type, rng
        )

        def _sliced() -> Iterator[Tuple[int, np.ndarray]]:
            for chunk_start in range(0, n_bootstrap, block_size):
                yield chunk_start, weights[chunk_start : chunk_start + block_size]

        return psu_ids, _sliced()

    # Unstratified, n_psu >= 2: tile the generation over draws. Mirror the
    # unstratified FPC scaling from generate_survey_multiplier_weights_batch.
    fpc = getattr(resolved_survey, "fpc", None)
    fpc_scale = 1.0
    fpc_zero = False
    if fpc is not None:
        # psu=None already sets n_psu = len(weights), so n_units_for_fpc == n_psu
        # on both branches of the original generator.
        n_units_for_fpc = n_psu
        if fpc[0] < n_units_for_fpc:
            raise ValueError(
                f"FPC ({fpc[0]}) is less than the number of PSUs "
                f"({n_units_for_fpc}). FPC must be >= number of PSUs."
            )
        f = n_units_for_fpc / fpc[0]
        if f < 1.0:
            fpc_scale = float(np.sqrt(1.0 - f))
        else:
            fpc_zero = True

    def _generated() -> Iterator[Tuple[int, np.ndarray]]:
        for chunk_start, block in iter_weight_blocks(
            n_bootstrap, n_psu, weight_type, rng, block_size=block_size
        ):
            if fpc_zero:
                block = np.zeros_like(block)
            elif fpc_scale != 1.0:
                block = block * fpc_scale
            yield chunk_start, block

    return psu_ids, _generated()
