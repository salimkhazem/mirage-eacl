"""Multilingual IRT with translation-induced item drift.

Implements models M1 (Rasch) and M2 (2PL with language-varying discrimination)
from ``docs/THEORY.md``.

The single most important implementation decision in this file is the **declared
normalisation**: we always fit subject to ``sum_i delta[i, l] == 0`` for every
language ``l``.  That constraint is arbitrary -- Theorem 1 says *some* arbitrary
choice is unavoidable -- but making it explicit is what lets every downstream
identification statement be phrased as a question about a single unknown scalar
``c_l`` per language.  A fitting routine that instead regularises ``delta``
toward zero would silently pick a different anchoring and quietly bias every
reported gap; see ``docs/THEORY.md`` Section 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

__all__ = ["IRTFit", "fit_irt"]


@dataclass
class IRTFit:
    """Fitted parameters under the declared ``sum_i delta = 0`` normalisation.

    Attributes carry the tilde semantics of ``docs/THEORY.md``: ``theta`` is
    ``theta_tilde = theta + c``, ``delta`` is ``delta_tilde = delta - c``, and
    therefore ``gap`` is ``G_tilde = G - c``.  Recovering ``G`` requires an
    identifying assumption; see :mod:`mirage.identify`.
    """

    theta: np.ndarray  # (M, L) ability, tilde-scale
    b: np.ndarray  # (I,) source difficulty
    delta: np.ndarray  # (I, L) translation drift, sum-zero per language
    a: np.ndarray | None  # (I, L) discrimination, None under Rasch
    model: str
    source_index: int
    n_iter: int
    final_loss: float
    converged: bool

    @property
    def gap(self) -> np.ndarray:
        """``G_tilde[m, l] = theta[m, source] - theta[m, l]``, shape (M, L).

        NOT the cross-lingual gap.  It differs from the estimand ``G`` by the
        unidentified per-language shift ``c_l`` (Theorem 1).
        """
        return self.theta[:, [self.source_index]] - self.theta

    def drift_se(self) -> np.ndarray:
        """(I, L) standard error of each fitted ``delta[i, l]``.

        Each cell is informed by only ``M`` Bernoulli draws, so
        ``se ~= (sum_m p(1-p))^(-1/2)`` -- about 0.45 logits at ``M = 20``. This is
        why the plug-in identified set of Theorem 3 is *not* a valid confidence
        set at finite ``M``: the tolerance ``tau`` must absorb this noise before
        the set covers. Pass ``tau + z * median(drift_se())`` rather than ``tau``.
        """
        p = 1.0 / (1.0 + np.exp(-self.cell_logits()))
        info = (p * (1.0 - p)).sum(axis=0)  # (I, L)
        if self.a is not None:
            info = info * self.a**2
        return 1.0 / np.sqrt(np.maximum(info, 1e-12))

    def cell_logits(self) -> np.ndarray:
        """(M, I, L) linear predictors, for likelihood checks and simulation."""
        eta = self.theta[:, None, :] - self.b[None, :, None] - self.delta[None, :, :]
        if self.a is not None:
            eta = self.a[None, :, :] * eta
        return eta


def _normalise(
    theta: Tensor,
    delta: Tensor,
    source_index: int,
    model: str,
    a: Tensor | None,
    *,
    rescale: bool = True,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Apply the declared normalisation inside the computation graph.

    Constraints, in order:
      1. ``delta[:, source] = 0`` -- the source language has no translation drift
         by definition (it was not translated).
      2. ``mean_i delta[:, l] = 0`` for every non-source ``l`` -- the declared
         anchoring that pins the otherwise-flat direction of Theorem 1.
      3. ``mean_m theta[:, source] = 0``, the ordinary Rasch location convention.
      4. under 2PL only, and only when ``rescale`` is set, unit sd of source
         ability with the compensating factor moved into ``a``.

    Args:
        rescale: apply constraint 4. Must be ``False`` during optimisation.
            Dividing ``theta`` by its own standard deviation while multiplying
            ``a`` by the same factor is exact algebraically but degenerate as a
            gradient step: shrinking ability shrinks the divisor, which shrinks
            ``a``, which flattens every logit. The optimiser rides that spiral to
            ``eta = 0``, where the loss is exactly ``ln 2 = 0.693`` -- the score
            of predicting one half everywhere. This happened on four of five
            benchmarks before the flag existed. The 2PL scale is unidentified
            anyway (Theorem 2), so it is fixed once at extraction rather than
            chased at every step.
    """
    delta = delta - delta.mean(dim=0, keepdim=True)
    src_mask = torch.zeros_like(delta)
    src_mask[:, source_index] = 1.0
    delta = delta * (1.0 - src_mask)

    theta = theta - theta[:, source_index].mean()
    if model == "2pl" and rescale:
        scale = theta[:, source_index].std().clamp_min(1e-3)
        theta = theta / scale
        if a is not None:
            a = a * scale
    return theta, delta, a


def fit_irt(
    responses: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    model: str = "rasch",
    source_index: int = 0,
    n_iter: int = 4000,
    lr: float = 0.05,
    ridge_theta: float = 1e-4,
    ridge_delta: float = 1e-3,
    tol: float = 1e-8,
    seed: int = 0,
    device: str = "cpu",
    verbose: bool = False,
) -> IRTFit:
    """Fit M1 or M2 by penalised joint maximum likelihood.

    Args:
        responses: (M, I, L) binary array. Entries where ``mask`` is False are
            ignored and may be arbitrary.
        mask: (M, I, L) boolean observation mask; ``None`` means fully crossed.
        model: ``"rasch"`` (M1) or ``"2pl"`` (M2).
        source_index: index of the source language along the L axis.
        ridge_theta, ridge_delta: small quadratic penalties for numerical
            stability only. ``ridge_delta`` acts *within* the sum-zero subspace,
            so it shrinks the identified contrasts ``delta_i - delta_j`` slightly
            but cannot move the unidentified mean; it therefore does not smuggle
            in an anchoring. Keep it small and report it.

    Returns:
        :class:`IRTFit` under the declared normalisation.

    Note:
        Joint MLE with fixed ``I`` is inconsistent as ``M -> inf`` (Neyman-Scott;
        Andersen 1970) -- see Proposition 6. That affects the *level* of ``b`` and
        the scale of ``a``; the identification results of Theorems 1-5 concern the
        flat direction and are unaffected. Use ``mirage.conditional`` for the
        consistent conditional-MLE variant on an anchor set.
    """
    if model not in {"rasch", "2pl"}:
        raise ValueError(f"model must be 'rasch' or '2pl', got {model!r}")
    y = np.asarray(responses)
    if y.ndim != 3:
        raise ValueError(f"responses must be (M, I, L), got shape {y.shape}")
    M, I, L = y.shape
    if not 0 <= source_index < L:
        raise ValueError(f"source_index {source_index} out of range for L={L}")

    torch.manual_seed(seed)
    dev = torch.device(device)
    Y = torch.as_tensor(y, dtype=torch.float32, device=dev)
    W = (
        torch.ones_like(Y)
        if mask is None
        else torch.as_tensor(np.asarray(mask), dtype=torch.float32, device=dev)
    )
    if W.shape != Y.shape:
        raise ValueError(f"mask shape {tuple(W.shape)} != responses {tuple(Y.shape)}")
    n_obs = W.sum().clamp_min(1.0)

    theta = torch.zeros(M, L, device=dev, requires_grad=True)
    b = torch.zeros(I, device=dev, requires_grad=True)
    delta = torch.zeros(I, L, device=dev, requires_grad=True)
    params = [theta, b, delta]
    a_raw = None
    if model == "2pl":
        # softplus(0.5413) == 1.0, i.e. start at unit discrimination.
        a_raw = torch.full((I, L), 0.5413, device=dev, requires_grad=True)
        params.append(a_raw)

    opt = torch.optim.Adam(params, lr=lr)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    prev = float("inf")
    converged = False
    step = 0

    for step in range(1, n_iter + 1):
        opt.zero_grad(set_to_none=True)
        a = torch.nn.functional.softplus(a_raw) if a_raw is not None else None
        th, dl, a = _normalise(theta, delta, source_index, model, a, rescale=False)
        eta = th[:, None, :] - b[None, :, None] - dl[None, :, :]
        if a is not None:
            eta = a[None, :, :] * eta
        loss = (bce(eta, Y, reduction="none") * W).sum() / n_obs
        loss = loss + ridge_theta * th.pow(2).mean() + ridge_delta * dl.pow(2).mean()
        loss.backward()
        opt.step()

        cur = float(loss.detach())
        if abs(prev - cur) < tol:
            converged = True
            break
        prev = cur
        if verbose and step % 500 == 0:
            print(f"[mirage.irt] step {step:5d}  loss {cur:.6f}")

    with torch.no_grad():
        a = torch.nn.functional.softplus(a_raw) if a_raw is not None else None
        th, dl, a = _normalise(theta, delta, source_index, model, a, rescale=True)
        return IRTFit(
            theta=th.cpu().numpy(),
            b=b.detach().cpu().numpy(),
            delta=dl.cpu().numpy(),
            a=None if a is None else a.cpu().numpy(),
            model=model,
            source_index=source_index,
            n_iter=step,
            final_loss=prev,
            converged=converged,
        )
