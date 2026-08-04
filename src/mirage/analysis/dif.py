"""Differential item functioning: the detectable half of Theorem 5.

Theorem 5 splits translation drift into a component that is identified and one
that is not.  This module tests the identified half.

What is tested is deliberately narrow.  Under the declared ``sum_i delta = 0``
normalisation, the null for item ``i`` in language ``l`` is
``delta_tilde[i, l] = 0``, i.e. *this item drifted no more than the average item
in this language*.  It is **not** a test of ``delta[i, l] = 0``: the common
offset is exactly the unidentified direction, so no test can address it. A
rejection therefore means "this item drifted unusually", never "this item
drifted".
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = ["benjamini_hochberg", "dif_test", "drift_bootstrap_se"]


def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Boolean rejection mask controlling FDR at ``alpha`` (BH, independent/PRDS)."""
    p = np.asarray(pvals, dtype=float).ravel()
    if p.size == 0:
        return np.zeros(0, dtype=bool)
    if np.any((p < 0) | (p > 1) | ~np.isfinite(p)):
        raise ValueError("p-values must be finite and in [0, 1]")

    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * np.arange(1, n + 1) / n
    passing = np.nonzero(ranked <= thresh)[0]
    out = np.zeros(n, dtype=bool)
    if passing.size:
        out[order[: passing[-1] + 1]] = True
    return out


def dif_test(
    delta_tilde: np.ndarray,
    delta_se: np.ndarray,
    *,
    alpha: float = 0.05,
    source_index: int = 0,
) -> dict[str, np.ndarray]:
    """Per-item Wald test of ``delta_tilde[i, l] = 0`` with BH-FDR control.

    Args:
        delta_tilde: (I, L) fitted drift, sum-zero per language.
        delta_se: (I, L) standard errors, from :meth:`mirage.irt.IRTFit.drift_se`.
        alpha: target false discovery rate across all non-source cells jointly.
        source_index: the source language column, excluded (its drift is 0 by
            definition, not by estimation).

    Returns:
        ``z``, ``p``, ``rejected`` (all (I, L), source column zeroed/False), and
        ``n_rejected`` / ``frac_rejected`` scalars over the tested cells.

    FDR is controlled over the whole ``I x (L-1)`` grid at once rather than per
    language. Per-language correction would let the number of discoveries grow
    with the number of languages purely by chance, which is the multiplicity
    error the paper criticises elsewhere.
    """
    d = np.asarray(delta_tilde, dtype=float)
    se = np.asarray(delta_se, dtype=float)
    if d.shape != se.shape:
        raise ValueError(f"shape mismatch: delta {d.shape} vs se {se.shape}")
    if d.ndim != 2:
        raise ValueError(f"expected (I, L), got {d.shape}")

    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, d / se, 0.0)
    p = 2.0 * stats.norm.sf(np.abs(z))
    p = np.clip(np.nan_to_num(p, nan=1.0), 0.0, 1.0)

    cols = [c for c in range(d.shape[1]) if c != source_index]
    rejected = np.zeros_like(p, dtype=bool)
    if cols:
        flat = p[:, cols].ravel()
        rejected[:, cols] = benjamini_hochberg(flat, alpha).reshape(len(d), len(cols))

    z[:, source_index] = 0.0
    p[:, source_index] = 1.0
    n_tested = len(d) * len(cols)
    return {
        "z": z,
        "p": p,
        "rejected": rejected,
        "n_rejected": np.array(int(rejected.sum())),
        "n_tested": np.array(n_tested),
        "frac_rejected": np.array(rejected.sum() / n_tested if n_tested else 0.0),
    }


def drift_bootstrap_se(
    responses: np.ndarray,
    fit_fn,
    *,
    n_boot: int = 20,
    seed: int = 0,
    source_index: int = 0,
) -> dict[str, np.ndarray]:
    """Bootstrap over **models** for the drift and gap estimates.

    Resampling models rather than items is the right unit here: ``delta[i, l]``
    is estimated from the ``M`` responses in its cell, so model-level
    resampling is what propagates the dominant source of noise. It also matches
    the inferential question -- would this conclusion hold with a different
    panel of models?

    Args:
        responses: (M, I, L) binary tensor.
        fit_fn: callable taking a (M, I, L) tensor and returning an ``IRTFit``.
        n_boot: bootstrap replicates. Twenty is enough for a standard error;
            the identified-set width, not this, dominates the reported interval.

    Returns:
        ``delta_se`` (I, L), ``gap_se`` (L,), and ``n_boot``.
    """
    y = np.asarray(responses)
    if y.ndim != 3:
        raise ValueError(f"expected (M, I, L), got {y.shape}")
    M = y.shape[0]
    rng = np.random.default_rng(seed)

    deltas, gaps = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, M, size=M)
        fit = fit_fn(y[idx])
        deltas.append(fit.delta)
        gaps.append(fit.gap.mean(axis=0))

    D = np.stack(deltas)
    G = np.stack(gaps)
    return {
        "delta_se": D.std(axis=0, ddof=1),
        "gap_se": G.std(axis=0, ddof=1),
        "n_boot": np.array(n_boot),
    }
