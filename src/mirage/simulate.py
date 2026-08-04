"""Synthetic response tensors with known ground truth.

Used for the toy proof-of-concept and for every theorem test.  The generator
deliberately exposes ``uniform_drift`` separately from the item-level drift,
because Theorem 5 is precisely the claim that the first is invisible and the
second is not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Truth", "simulate"]


@dataclass
class Truth:
    """Ground-truth parameters plus the sampled responses."""

    responses: np.ndarray  # (M, I, L) binary
    theta: np.ndarray  # (M, L)
    b: np.ndarray  # (I,)
    delta: np.ndarray  # (I, L) total drift, including the uniform part
    a: np.ndarray | None  # (I, L) or None
    uniform_drift: np.ndarray  # (L,) the c_l that Theorem 1 says is invisible
    invariant_mask: np.ndarray  # (I, L) True where the item is exactly invariant
    source_index: int

    @property
    def gap(self) -> np.ndarray:
        """True cross-lingual gap ``G[m, l]``, shape (M, L)."""
        return self.theta[:, [self.source_index]] - self.theta

    def probs(self) -> np.ndarray:
        eta = self.theta[:, None, :] - self.b[None, :, None] - self.delta[None, :, :]
        if self.a is not None:
            eta = self.a[None, :, :] * eta
        return 1.0 / (1.0 + np.exp(-eta))


def simulate(
    *,
    n_models: int = 20,
    n_items: int = 300,
    n_langs: int = 8,
    invariant_frac: float = 0.7,
    drift_scale: float = 0.8,
    drift_mean: float = 0.0,
    uniform_drift: float | np.ndarray = 0.0,
    ability_gap_scale: float = 0.6,
    model: str = "rasch",
    discrimination_spread: float = 0.25,
    source_index: int = 0,
    seed: int = 0,
) -> Truth:
    """Draw a fully crossed response tensor from M1 or M2.

    Args:
        invariant_frac: fraction of items with exactly zero drift in each
            non-source language. Values above 0.5 satisfy the majority-invariance
            assumption (A3) of Theorem 4; values below it do not, which is the
            regime where only Theorem 3 bounds apply.
        drift_scale: sd of the Gaussian drift on the non-invariant items.
        drift_mean: mean drift on the non-invariant items. Non-zero values make
            the drift distribution asymmetric, which is the realistic case --
            translation tends to make items *harder*, not to perturb them
            symmetrically. It is also what makes mean- and median-based anchors
            genuinely disagree; with symmetric drift they coincide and the
            choice of anchor looks (misleadingly) inconsequential.
        uniform_drift: scalar or (L,) array added to *every* item's drift in a
            language. This is the unidentified direction; tests assert that
            changing it leaves the fitted ``delta_tilde`` and the response law
            unchanged in distribution.
        ability_gap_scale: sd of the true per-language ability decrement.
        discrimination_spread: sd of log-discrimination under ``model="2pl"``.

    Returns:
        A :class:`Truth` bundle.
    """
    if model not in {"rasch", "2pl"}:
        raise ValueError(f"model must be 'rasch' or '2pl', got {model!r}")
    if not 0.0 <= invariant_frac <= 1.0:
        raise ValueError(f"invariant_frac must be in [0, 1], got {invariant_frac}")
    rng = np.random.default_rng(seed)
    M, I, L = n_models, n_items, n_langs

    # Ability: a shared per-model level plus a per-language decrement that is
    # strictly non-positive relative to the source, matching the empirical fact
    # that models are best in their source language.
    base = rng.normal(0.0, 1.0, size=M)
    decrement = np.abs(rng.normal(0.0, ability_gap_scale, size=(M, L)))
    decrement[:, source_index] = 0.0
    theta = base[:, None] - decrement

    b = rng.normal(0.0, 1.0, size=I)

    # Item-level drift: an exactly-invariant subset plus Gaussian drift elsewhere.
    invariant_mask = np.zeros((I, L), dtype=bool)
    delta = np.zeros((I, L))
    n_inv = int(round(invariant_frac * I))
    for l in range(L):
        if l == source_index:
            invariant_mask[:, l] = True
            continue
        idx = rng.permutation(I)
        inv, var = idx[:n_inv], idx[n_inv:]
        invariant_mask[inv, l] = True
        delta[var, l] = rng.normal(drift_mean, drift_scale, size=var.size)

    c = np.zeros(L) if np.isscalar(uniform_drift) and uniform_drift == 0 else None
    if c is None:
        c = np.broadcast_to(np.asarray(uniform_drift, dtype=float), (L,)).copy()
    c[source_index] = 0.0
    delta = delta + c[None, :]
    delta[:, source_index] = 0.0

    a = None
    if model == "2pl":
        a = np.exp(rng.normal(0.0, discrimination_spread, size=(I, L)))

    eta = theta[:, None, :] - b[None, :, None] - delta[None, :, :]
    if a is not None:
        eta = a[None, :, :] * eta
    p = 1.0 / (1.0 + np.exp(-eta))
    responses = (rng.random(p.shape) < p).astype(np.int8)

    return Truth(
        responses=responses,
        theta=theta,
        b=b,
        delta=delta,
        a=a,
        uniform_drift=c,
        invariant_mask=invariant_mask,
        source_index=source_index,
    )
