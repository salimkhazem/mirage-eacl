"""Identification: what can and cannot be recovered from the response tensor.

Implements Theorem 3(a) (sharp partial identification via the sliding-window
set), Theorem 4 (median anchoring under majority invariance), and the
translation of both into bounds on the cross-lingual gap ``G``.

Recall the reduction from ``docs/THEORY.md`` Section 0: after fitting under the
declared ``sum_i delta = 0`` normalisation we observe ``delta_tilde = delta - c``
and ``G_tilde = G + c``, so the estimand is ``G = G_tilde - c``, where the
per-language scalar ``c_l`` is the entire content of the identification problem.
Every function here answers a question about ``c``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "Interval",
    "IdentifiedSet",
    "sliding_window_set",
    "median_anchor",
    "gap_bounds",
    "breakeven_drift",
    "minimax_floor",
]


@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float

    def __contains__(self, x: float) -> bool:
        return self.lo - 1e-12 <= x <= self.hi + 1e-12

    @property
    def width(self) -> float:
        return self.hi - self.lo


@dataclass(frozen=True)
class IdentifiedSet:
    """A union of closed intervals: the sharp identified set for a scalar.

    ``parts`` is disjoint and sorted.  ``hull`` is the convex hull, which is what
    a table normally reports; ``width`` is the hull width, i.e. the honest
    uncertainty that no amount of extra data can shrink (Theorem 3b).
    """

    parts: tuple[Interval, ...]

    def __post_init__(self) -> None:
        if not self.parts:
            raise ValueError("identified set is empty: assumptions are refuted by the data")

    def __contains__(self, x: float) -> bool:
        return any(x in p for p in self.parts)

    @property
    def hull(self) -> Interval:
        return Interval(self.parts[0].lo, self.parts[-1].hi)

    @property
    def width(self) -> float:
        return self.hull.width

    @property
    def measure(self) -> float:
        """Total length of the union, which can be much less than the hull."""
        return float(sum(p.width for p in self.parts))

    def shift(self, offset: float) -> IdentifiedSet:
        return IdentifiedSet(tuple(Interval(p.lo + offset, p.hi + offset) for p in self.parts))

    def negate(self) -> IdentifiedSet:
        return IdentifiedSet(tuple(Interval(-p.hi, -p.lo) for p in reversed(self.parts)))


def sliding_window_set(
    delta_tilde: np.ndarray, pi: float, tau: float, *, atol: float = 1e-9
) -> IdentifiedSet:
    """Sharp identified set for the shift ``c`` under ``(pi, tau)``-invariance.

    Assumption: at least a fraction ``pi`` of items are ``tau``-invariant,
    ``|delta[i]| <= tau``, but *which* items is unknown.  Since
    ``delta[i] = delta_tilde[i] + c``, the assumption holds at a candidate ``c``
    exactly when at least ``ceil(pi * I)`` of the values ``delta_tilde[i] + c``
    lie in ``[-tau, tau]``, i.e. when ``u = -c`` stabs at least that many of the
    intervals ``[delta_tilde[i] - tau, delta_tilde[i] + tau]``.

    The stabbing-count function is piecewise constant with breakpoints at
    ``delta_tilde[i] +/- tau``, so the super-level set is a union of closed
    intervals, found here by an ``O(I log I)`` event sweep.

    Args:
        delta_tilde: (I,) fitted drift for one language, sum-zero normalised.
        pi: assumed minimum fraction of invariant items, in (0, 1].
        tau: invariance tolerance in logits; ``tau = 0`` means exact invariance.

    Returns:
        The sharp identified set for ``c``.

    Raises:
        ValueError: if the assumptions are refuted, i.e. no ``c`` achieves the
            required count.  With ``tau = 0`` this happens whenever fewer than
            ``ceil(pi * I)`` of the fitted drifts share a common value, which is
            generic for continuous estimates -- use ``tau > 0`` in practice.
    """
    d = np.asarray(delta_tilde, dtype=float).ravel()
    if d.size == 0:
        raise ValueError("delta_tilde is empty")
    if not 0.0 < pi <= 1.0:
        raise ValueError(f"pi must be in (0, 1], got {pi}")
    if tau < 0:
        raise ValueError(f"tau must be non-negative, got {tau}")
    if not np.all(np.isfinite(d)):
        raise ValueError("delta_tilde contains non-finite values")

    n = d.size
    k = math.ceil(pi * n - atol)

    # Sweep over interval endpoints. Opens sort before closes at equal position
    # so that closed intervals touching at a point still register the overlap.
    events = [(v - tau, 1) for v in d] + [(v + tau, -1) for v in d]
    events.sort(key=lambda e: (e[0], -e[1]))

    parts: list[Interval] = []
    count = 0
    start: float | None = None
    for pos, kind in events:
        if kind == 1:
            count += 1
            if count == k and start is None:
                start = pos
        else:
            if count == k and start is not None:
                parts.append(Interval(start, pos))
                start = None
            count -= 1

    if not parts:
        raise ValueError(
            f"identified set for c is empty: no shift makes >= {k} of {n} items "
            f"tau-invariant at tau={tau:g}. The (pi, tau) assumption is refuted "
            f"by the data -- lower pi or raise tau."
        )

    # Merge touching parts, then map u -> c = -u.
    merged = [parts[0]]
    for p in parts[1:]:
        if p.lo <= merged[-1].hi + atol:
            merged[-1] = Interval(merged[-1].lo, max(merged[-1].hi, p.hi))
        else:
            merged.append(p)
    return IdentifiedSet(tuple(merged)).negate()


def median_anchor(delta_tilde: np.ndarray) -> float:
    """Theorem 4 estimator of the shift: ``c_hat = -median_i delta_tilde[i]``.

    Consistent when a strict majority of items are invariant, with breakdown
    point 1/2 against arbitrary corruption of the non-anchor items.  Returns the
    shift ``c`` such that ``delta = delta_tilde + c`` has zero median.
    """
    d = np.asarray(delta_tilde, dtype=float).ravel()
    if d.size == 0:
        raise ValueError("delta_tilde is empty")
    return float(-np.median(d))


def gap_bounds(
    gap_tilde: np.ndarray,
    delta_tilde: np.ndarray,
    pi: float,
    tau: float,
    *,
    inflate: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, list[IdentifiedSet]]:
    """Sharp bounds on the cross-lingual gap ``G`` for every (model, language).

    ``G[m, l] = G_tilde[m, l] - c_l`` (see ``docs/THEORY.md`` Section 0), with
    ``c_l`` ranging over its sharp identified set. The bounds are therefore a
    *reflected* translation of that set -- note the minus sign, which is the
    difference between correcting a gap and doubling its error. The source
    language has ``c = 0`` exactly and receives a degenerate set.

    Args:
        gap_tilde: (M, L) from :attr:`mirage.irt.IRTFit.gap`.
        delta_tilde: (I, L) from :attr:`mirage.irt.IRTFit.delta`.
        pi, tau: the ``(pi, tau)``-invariance assumption.
        inflate: half-width added to each side for **finite-sample estimation
            error** in ``gap_tilde`` and ``delta_tilde``. Theorem 3 describes a
            population object; with finite ``M`` the plug-in interval
            under-covers, badly when the truth sits near the edge of the set.
            Pass the bootstrap standard error here (see
            :func:`mirage.analysis.bootstrap.gap_se`). Reported intervals in the
            paper are always inflated; the raw set is diagnostic only.

    Returns:
        ``(lo, hi, sets)`` where ``lo``/``hi`` are (M, L) hull bounds on ``G`` and
        ``sets`` is the length-L list of identified sets for ``c_l`` itself.
    """
    g = np.asarray(gap_tilde, dtype=float)
    d = np.asarray(delta_tilde, dtype=float)
    if g.ndim != 2 or d.ndim != 2 or g.shape[1] != d.shape[1]:
        raise ValueError(f"shape mismatch: gap_tilde {g.shape}, delta_tilde {d.shape}")

    if inflate < 0:
        raise ValueError(f"inflate must be non-negative, got {inflate}")

    lo = np.empty_like(g)
    hi = np.empty_like(g)
    sets: list[IdentifiedSet] = []
    for l in range(g.shape[1]):
        col = d[:, l]
        if np.allclose(col, 0.0):  # source language: no drift by definition
            s = IdentifiedSet((Interval(0.0, 0.0),))
        else:
            s = sliding_window_set(col, pi, tau)
        sets.append(s)
        # G = G_tilde - c, so the bounds reflect the set for c.
        lo[:, l] = g[:, l] - s.hull.hi - inflate
        hi[:, l] = g[:, l] - s.hull.lo + inflate
    return lo, hi, sets


def breakeven_drift(
    gap_tilde: np.ndarray,
    delta_tilde: np.ndarray,
    delta_se: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Sensitivity analysis: how much invisible drift would erase a reported gap?

    Since ``G = G_tilde - c``, the gap is exactly zero at ``c = G_tilde``. So the
    reported gap *is* its own breakeven point: a uniform translation drift of
    ``G_tilde`` logits explains the entire finding, and by Theorem 5 no test can
    rule that value out.

    Reporting the breakeven value is far more informative than a wide interval.
    An interval that contains zero says only "we cannot tell"; the breakeven says
    "this conclusion survives only if mean translation drift is below X", and X
    can be judged against the drift dispersion we *can* see.

    The yardstick is the spread of item-level drift after removing estimation
    noise: ``sd_true^2 = var(delta_tilde) - mean(se^2)``, clipped at zero. Since
    ``delta_tilde`` is mean-zero by construction, this is the dispersion of the
    *non-uniform* component -- the detectable half of Theorem 5. A breakeven that
    is small relative to that spread is a gap with no protection: drift of the
    size we already observe, merely distributed uniformly instead of unevenly,
    would account for it entirely.

    Args:
        gap_tilde: (M, L) fitted gaps under the declared normalisation.
        delta_tilde: (I, L) fitted drift.
        delta_se: (I, L) standard errors of the drift.
        weights: optional (M,) non-negative model weights for the gap average.
            Used to down-weight or drop models whose responses in a language
            collapsed onto one option: such a cell reports that the model fails
            the task, but carries no item-difficulty information, so it inflates
            the gap while contributing nothing to the ``sigma`` yardstick.

    Returns:
        ``breakeven`` (L,) mean gap per language, i.e. the drift that zeroes it;
        ``drift_sd`` (L,) noise-corrected sd of observed item drift;
        ``ratio`` (L,) breakeven expressed in units of that sd.
    """
    g = np.asarray(gap_tilde, dtype=float)
    d = np.asarray(delta_tilde, dtype=float)
    se = np.asarray(delta_se, dtype=float)
    if d.shape != se.shape:
        raise ValueError(f"shape mismatch: delta {d.shape} vs se {se.shape}")

    breakeven = g.mean(axis=0) if weights is None else np.average(g, axis=0, weights=weights)
    var_obs = d.var(axis=0)
    var_noise = (se**2).mean(axis=0)
    drift_sd = np.sqrt(np.maximum(var_obs - var_noise, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(drift_sd > 0, np.abs(breakeven) / drift_sd, np.inf)
    return {"breakeven": breakeven, "drift_sd": drift_sd, "ratio": ratio}


def minimax_floor(tau: float) -> float:
    """Theorem 3(b): the minimax risk floor for estimating ``G``.

    Under the mean constraint ``|c_l| <= tau`` no estimator attains expected
    absolute error below ``tau``, at *any* sample size, because the two-point
    construction is observationally identical (total variation zero).  Provided
    as a function to make the point explicit in tables: the floor does not
    depend on M, I, or L.
    """
    if tau < 0:
        raise ValueError(f"tau must be non-negative, got {tau}")
    return float(tau)
