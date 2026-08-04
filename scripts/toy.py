"""Toy validation: machine-check every claim the paper makes, on synthetic data.

Unlike ``tests/``, which asserts pass/fail, this script *measures* each claim and
writes the measured value into a ledger, so that ``docs/CLAIMS.md`` can never
drift from what the code actually produces.  Every row of the ledger is either
VERIFIED (with a number) or is not claimed.

    python -m scripts.toy --out results/runs/toy --seeds 0,1,2

Runs on CPU in a couple of minutes. No downloads, no GPU.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

from mirage.identify import gap_bounds, median_anchor, sliding_window_set
from mirage.irt import fit_irt
from mirage.simulate import simulate

# --------------------------------------------------------------------------
# Ledger
# --------------------------------------------------------------------------


@dataclass
class Claim:
    """One falsifiable claim, its predicted value, and what we measured."""

    claim_id: str
    statement: str
    predicted: str
    measured: float
    passed: bool
    detail: str = ""


class Ledger:
    def __init__(self) -> None:
        self.claims: list[Claim] = []

    def record(
        self,
        claim_id: str,
        statement: str,
        predicted: str,
        measured: float,
        passed: bool,
        detail: str = "",
    ) -> None:
        self.claims.append(
            Claim(claim_id, statement, predicted, float(measured), bool(passed), detail)
        )
        flag = "VERIFIED" if passed else "**FAILED**"
        print(f"  [{flag:>10}] {claim_id:<8} {statement[:62]:<62} measured={measured: .4f}")

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.claims)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------
# Claim checks
# --------------------------------------------------------------------------


def check_thm1_invariance_is_exact(led: Ledger, rng: np.random.Generator) -> None:
    """C1.1 -- the shift changes no cell probability, to machine precision."""
    M, I, L = 10, 60, 5
    theta, b = rng.normal(size=(M, L)), rng.normal(size=I)
    delta = rng.normal(size=(I, L))
    delta[:, 0] = 0.0
    c = rng.normal(size=L)
    c[0] = 0.0
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))  # noqa: E731
    p0 = sig(theta[:, None, :] - b[None, :, None] - delta[None])
    p1 = sig((theta + c)[:, None, :] - b[None, :, None] - (delta + c[None, :])[None])
    err = float(np.max(np.abs(p0 - p1)))
    led.record(
        "C1.1",
        "shift leaves every cell probability exactly invariant",
        "max |dp| < 1e-12",
        err,
        err < 1e-12,
    )


def check_thm1_contamination_slope(led: Ledger, seeds: list[int]) -> dict:
    """C1.2 -- the headline: G_tilde = G + c, so contamination is one-for-one."""
    drifts = np.linspace(-0.8, 0.8, 7)
    curves = []
    for s in seeds:
        est = []
        for j, c in enumerate(drifts):
            t = simulate(
                n_models=40,
                n_items=250,
                n_langs=3,
                invariant_frac=0.6,
                drift_scale=0.7,
                uniform_drift=float(c),
                seed=1000 * s + j,
            )
            f = fit_irt(t.responses, n_iter=3000, seed=s)
            # True gap is held fixed across the sweep by construction.
            est.append(float(np.mean(f.gap[:, 1:] - t.gap[:, 1:])))
        curves.append(est)
    curves = np.array(curves)
    slopes = [float(np.polyfit(drifts, c, 1)[0]) for c in curves]
    mean_slope = float(np.mean(slopes))
    led.record(
        "C1.2",
        "fitted gap absorbs uniform drift one-for-one (slope = 1)",
        "slope in [0.85, 1.15]",
        mean_slope,
        0.85 <= mean_slope <= 1.15,
        detail=f"per-seed slopes {[round(s, 3) for s in slopes]}",
    )
    return {"drifts": drifts.tolist(), "curves": curves.tolist(), "slopes": slopes}


def check_thm1_identified_survive(led: Ledger, seeds: list[int]) -> None:
    """C1.3/C1.4 -- within-language contrasts and DiD are recovered."""
    within, did = [], []
    for s in seeds:
        t = simulate(n_models=40, n_items=400, n_langs=5, drift_scale=0.7, seed=20 + s)
        f = fit_irt(t.responses, n_iter=4000, seed=s)
        for l in range(5):
            within.append(
                np.corrcoef(
                    f.theta[:, l] - f.theta[:, l].mean(),
                    t.theta[:, l] - t.theta[:, l].mean(),
                )[0, 1]
            )
        de = f.gap[:, 1:] - f.gap[:, 1:].mean(axis=0, keepdims=True)
        dt = t.gap[:, 1:] - t.gap[:, 1:].mean(axis=0, keepdims=True)
        did.append(np.corrcoef(de.ravel(), dt.ravel())[0, 1])
    w, d = float(np.mean(within)), float(np.mean(did))
    led.record("C1.3", "within-language model contrasts are recovered", "r > 0.90", w, w > 0.90)
    led.record("C1.4", "difference-in-differences is recovered", "r > 0.85", d, d > 0.85)


def check_thm1_uniform_invisible(led: Ledger, seeds: list[int]) -> None:
    """C1.5 -- fitted drift carries no information about the uniform component."""
    rs = []
    for s in seeds:
        kw = dict(n_models=40, n_items=250, n_langs=3, invariant_frac=0.6, drift_scale=0.7)
        a = fit_irt(
            simulate(uniform_drift=0.0, seed=300 + s, **kw).responses, n_iter=3000, seed=s
        )
        b = fit_irt(
            simulate(uniform_drift=1.0, seed=300 + s, **kw).responses, n_iter=3000, seed=s
        )
        rs += [np.corrcoef(a.delta[:, l], b.delta[:, l])[0, 1] for l in (1, 2)]
    r = float(np.mean(rs))
    led.record(
        "C1.5",
        "fitted drift is unchanged by the uniform component",
        "r > 0.80",
        r,
        r > 0.80,
    )


def check_thm2_affine(led: Ledger, rng: np.random.Generator) -> None:
    """C2.1 -- affine group is exact. C2.2 -- DiD is destroyed, not merely rescaled."""
    M, I, L = 8, 50, 4
    theta, b = rng.normal(size=(M, L)), rng.normal(size=I)
    delta = rng.normal(size=(I, L))
    a = np.exp(rng.normal(0, 0.3, size=(I, L)))
    s = np.exp(rng.normal(0, 0.4, size=L))
    c = rng.normal(size=L)
    eta0 = a[None] * (theta[:, None, :] - b[None, :, None] - delta[None])
    eta1 = (a / s[None, :])[None] * (
        (s[None, :] * theta + c[None, :])[:, None, :]
        - (s[None, :] * (b[:, None] + delta) + c[None, :])[None]
    )
    err = float(np.max(np.abs(eta0 - eta1)))
    led.record(
        "C2.1",
        "2PL affine reparameterisation is exact",
        "max |d.eta| < 1e-10",
        err,
        err < 1e-10,
    )

    th = rng.normal(size=(20, 3))
    sc = np.array([1.0, 2.3, 0.4])
    ths = sc[None, :] * th
    gap, gaps = th[:, [0]] - th, ths[:, [0]] - ths
    dd = gap - gap.mean(axis=0, keepdims=True)
    dds = gaps - gaps.mean(axis=0, keepdims=True)
    spread = float(np.mean([np.std(dds[:, l] / dd[:, l]) for l in (1, 2)]))
    led.record(
        "C2.2",
        "DiD under 2PL is not a rescaling of the truth (it mixes scales)",
        "sd of implied ratio > 0.05",
        spread,
        spread > 0.05,
    )


def check_thm3_sharpness(led: Ledger, rng: np.random.Generator) -> dict:
    """C3.1 sweep matches brute force. C3.2 every point attainable. C3.3 monotone."""
    d = rng.normal(0, 0.5, size=60)
    pi, tau = 0.4, 0.3
    s = sliding_window_set(d, pi, tau)
    k = int(np.ceil(pi * d.size))
    grid = np.linspace(-3, 3, 24001)
    counts = np.array([np.sum(np.abs(d + c) <= tau) for c in grid])
    ok = grid[counts >= k]
    err = float(max(abs(s.hull.lo - ok.min()), abs(s.hull.hi - ok.max())))
    led.record(
        "C3.1",
        "O(I log I) sweep matches exhaustive grid search",
        "max endpoint err < 2e-3",
        err,
        err < 2e-3,
    )

    worst = min(
        int(np.sum(np.abs(d + c) <= tau + 1e-9)) - k
        for part in s.parts
        for c in np.linspace(part.lo, part.hi, 21)
    )
    led.record(
        "C3.2",
        "every point of the set is attainable (sharpness)",
        "min surplus >= 0",
        worst,
        worst >= 0,
    )

    d2 = rng.normal(0, 0.5, size=200)
    w_weak = sliding_window_set(d2, 0.2, 0.4).width
    w_strong = sliding_window_set(d2, 0.5, 0.4).width
    led.record(
        "C3.3",
        "set narrows as the invariance assumption strengthens",
        "width(pi=.5) < width(pi=.2)",
        w_strong - w_weak,
        w_strong < w_weak,
        detail=f"width {w_weak:.3f} -> {w_strong:.3f}",
    )

    # A refuted (pi, tau) pair is a *finding*, not a crash: it says the data are
    # inconsistent with that much assumed invariance. Record it as NaN so the
    # figure shows the feasible frontier.
    def width_or_nan(p: float, t: float) -> float:
        try:
            return sliding_window_set(d2, p, t).width
        except ValueError:
            return float("nan")

    taus = [0.2, 0.3, 0.4, 0.5]
    widths = {f"pi={p}": [width_or_nan(p, t) for t in taus] for p in (0.2, 0.3, 0.4, 0.5)}
    n_refuted = sum(np.isnan(v) for vs in widths.values() for v in vs)
    led.record(
        "C3.7",
        "over-strong (pi, tau) assumptions are refuted, not silently accepted",
        "at least one refutation in the sweep",
        n_refuted,
        n_refuted >= 1,
        detail=f"{n_refuted} of {len(taus) * 4} (pi, tau) pairs refuted by the data",
    )
    return {"widths": widths, "taus": taus}


def check_thm3_coverage(led: Ledger, seeds: list[int]) -> dict:
    """C3.4 set covers true c. C3.5 raw bounds under-cover. C3.6 inflated bounds cover."""
    in_plug, in_adj, cov_raw, cov_inf, taus = [], [], [], [], []
    for s in seeds:
        t = simulate(
            n_models=80,
            n_items=300,
            n_langs=4,
            invariant_frac=0.75,
            drift_scale=0.9,
            uniform_drift=0.4,
            seed=40 + s,
        )
        f = fit_irt(t.responses, n_iter=4000, seed=s)

        # (a) Plug-in set at the nominal tolerance: the population object.
        _, _, sets_plug = gap_bounds(f.gap, f.delta, pi=0.5, tau=0.45)
        in_plug += [t.uniform_drift[l] in sets_plug[l] for l in range(1, 4)]

        # (b) Noise-adjusted set: tau must absorb the finite-M error in delta-hat
        # before the set is a valid confidence set.
        se_d = float(np.median(f.drift_se()[:, 1:]))
        tau_adj = 0.45 + 2.0 * se_d
        taus.append(tau_adj)
        lo, hi, sets_adj = gap_bounds(f.gap, f.delta, pi=0.5, tau=tau_adj)
        in_adj += [t.uniform_drift[l] in sets_adj[l] for l in range(1, 4)]
        cov_raw.append(((lo <= t.gap + 1e-9) & (t.gap <= hi + 1e-9))[:, 1:].mean())

        # (c) ...plus inflation for the estimation error in G-tilde itself.
        resid = f.gap[:, 1:] - (t.gap[:, 1:] + t.uniform_drift[None, 1:])
        se_g = float(resid.std())
        lo2, hi2, _ = gap_bounds(f.gap, f.delta, pi=0.5, tau=tau_adj, inflate=2 * se_g)
        cov_inf.append(((lo2 <= t.gap + 1e-9) & (t.gap <= hi2 + 1e-9))[:, 1:].mean())

    plug, adj = float(np.mean(in_plug)), float(np.mean(in_adj))
    raw, inf = float(np.mean(cov_raw)), float(np.mean(cov_inf))
    led.record(
        "C3.4",
        "plug-in set at nominal tau UNDER-covers c at finite M",
        "coverage < 0.90 (a finding, not a bug)",
        plug,
        plug < 0.90,
    )
    led.record(
        "C3.5",
        "set with tau widened by 2*se(delta-hat) covers c",
        "coverage = 1.0",
        adj,
        adj >= 0.99,
        detail=f"mean adjusted tau = {np.mean(taus):.3f}",
    )
    led.record(
        "C3.6",
        "bounds on G cover once inflated for estimation error in G-tilde too",
        "coverage > 0.90",
        inf,
        inf > 0.90,
        detail=f"without the G-tilde inflation: {raw:.3f}",
    )
    return {
        "cov_c_plugin": plug,
        "cov_c_adjusted": adj,
        "cov_G_raw": raw,
        "cov_G_inflated": inf,
    }


def check_thm4_anchor(led: Ledger, rng: np.random.Generator) -> None:
    """C4.1 exact under (A3). C4.2 misses by exactly kappa when (A3) fails. C4.3 breakdown."""
    I = 400
    delta = np.zeros(I)
    delta[: int(0.3 * I)] = rng.normal(0, 1.0, size=int(0.3 * I))
    c = delta.mean()
    err = abs(median_anchor(delta - c) - c)
    led.record(
        "C4.1", "median anchor is exact when (A3) holds", "|error| < 1e-9", err, err < 1e-9
    )

    kappa = 0.8
    d2 = np.full(I, kappa)
    d2[: int(0.3 * I)] += rng.normal(0, 1.0, size=int(0.3 * I))
    c2 = d2.mean()
    miss = abs((c2 - median_anchor(d2 - c2)) - kappa)
    led.record(
        "C4.2",
        "when (A3) fails the anchor misses by exactly the uniform drift",
        "|miss - kappa| < 0.05",
        miss,
        miss < 0.05,
    )

    d3 = rng.normal(0, 0.3, size=401)
    clean = median_anchor(d3)

    def corrupt(frac: float) -> float:
        x = d3.copy()
        x[: int(frac * 401)] = 1e6
        return abs(median_anchor(x) - clean)

    lo_c, hi_c = corrupt(0.45), corrupt(0.55)
    led.record(
        "C4.3",
        "breakdown point is 1/2 (45% bounded, 55% unbounded)",
        "shift(45%) < 1 and shift(55%) > 1e5",
        hi_c / max(lo_c, 1e-12),
        lo_c < 1.0 and hi_c > 1e5,
        detail=f"45% -> {lo_c:.4f}, 55% -> {hi_c:.3e}",
    )


def check_thm5_dichotomy(led: Ledger, seeds: list[int]) -> None:
    """C5.1 non-uniform drift is detectable."""
    rs = []
    for s in seeds:
        t = simulate(
            n_models=80,
            n_items=300,
            n_langs=4,
            invariant_frac=0.5,
            drift_scale=1.0,
            uniform_drift=0.4,
            seed=60 + s,
        )
        f = fit_irt(t.responses, n_iter=4000, seed=s)
        rs += [
            np.corrcoef(f.delta[:, l], t.delta[:, l] - t.delta[:, l].mean())[0, 1]
            for l in range(1, 4)
        ]
    r = float(np.mean(rs))
    led.record(
        "C5.1", "non-uniform drift IS identified and detectable", "r > 0.85", r, r > 0.85
    )


def check_reordering(led: Ledger, seeds: list[int]) -> dict:
    """C6.1 -- the killer figure: raw language rankings move under admissible
    normalisations, while difference-in-differences rankings do not."""
    tau_cross, tau_did, tau_raw_true = [], [], []
    payload = None
    L = 8
    for s in seeds:
        rng = np.random.default_rng(500 + s)
        # Languages differ in how much of the test drifts, and drift is
        # asymmetric (translation makes items harder). Both are needed for the
        # three anchors to be genuinely different rather than coincidentally equal.
        t = simulate(
            n_models=30,
            n_items=400,
            n_langs=L,
            invariant_frac=0.55,
            drift_scale=0.7,
            drift_mean=0.9,
            uniform_drift=rng.normal(0, 0.4, size=L),
            seed=80 + s,
        )
        f = fit_irt(t.responses, n_iter=4000, seed=s)
        base = np.mean(f.gap, axis=0)

        # A: c = 0. Assumes scalar invariance. This is what the literature reports.
        rank_raw = base
        # B: median anchor. Assumes a majority of items are invariant (Thm 4).
        rank_med = base - np.array([median_anchor(f.delta[:, l]) for l in range(L)])
        # C: external anchor set -- items certified invariant from outside the
        # response data. This is the Global-MMLU move (culturally-agnostic +
        # human-translated items) and is the only anchor using outside information.
        c_ext = np.zeros(L)
        for l in range(1, L):
            anchor = t.invariant_mask[:, l]
            c_ext[l] = -float(np.mean(f.delta[anchor, l])) if anchor.any() else 0.0
        rank_ext = base - c_ext

        for other in (rank_med, rank_ext):
            tau_cross.append(kendalltau(rank_raw[1:], other[1:]).statistic)
        tau_raw_true.append(kendalltau(rank_raw[1:], np.mean(t.gap, axis=0)[1:]).statistic)

        did = f.gap - f.gap.mean(axis=0, keepdims=True)
        did_t = t.gap - t.gap.mean(axis=0, keepdims=True)
        tau_did.append(kendalltau(did[:, 1:].ravel(), did_t[:, 1:].ravel()).statistic)

        if payload is None:
            payload = {
                "raw": rank_raw.tolist(),
                "median": rank_med.tolist(),
                "external": rank_ext.tolist(),
                "true": np.mean(t.gap, axis=0).tolist(),
            }

    tc, td, tt = (
        float(np.mean(tau_cross)),
        float(np.mean(tau_did)),
        float(np.mean(tau_raw_true)),
    )
    led.record(
        "C6.1",
        "language ranking by raw gap moves across defensible anchors",
        "Kendall tau < 0.95",
        tc,
        tc < 0.95,
    )
    led.record(
        "C6.2",
        "difference-in-differences tracks the truth (the Thm 1 survivor)",
        "Kendall tau > 0.60",
        td,
        td > 0.60,
    )
    led.record(
        "C6.3",
        "raw gap ranking is a poor guide to the TRUE ability ranking",
        "Kendall tau vs truth < 0.80",
        tt,
        tt < 0.80,
    )

    return payload or {}


def check_consistency_sweep(led: Ledger, seeds: list[int]) -> dict:
    """The decisive experiment: identified quantities converge, the gap does not.

    Earlier versions of this check varied only ``M`` and found almost no effect.
    That was a design error, not a property of the estimand: difference-in-
    differences error is dominated by per-model ability estimation, which is
    driven by the number of *items*. Scaling ``M`` and ``I`` together separates
    the two behaviours cleanly, and lets us compare the raw gap's error against
    the floor that Theorem 3(b) predicts, ``sqrt(E[c^2])``.
    """
    sizes = [(10, 100), (20, 200), (40, 400), (80, 800)]
    err_did, err_gap, floors = [], [], []
    for M, I in sizes:
        ed, eg, fl = [], [], []
        for s in seeds:
            rng = np.random.default_rng(700 + s)
            t = simulate(
                n_models=M,
                n_items=I,
                n_langs=6,
                invariant_frac=0.55,
                drift_scale=0.7,
                drift_mean=0.9,
                uniform_drift=rng.normal(0, 0.4, size=6),
                seed=90 + s,
            )
            f = fit_irt(t.responses, n_iter=4000, seed=s)
            de = f.gap[:, 1:] - f.gap[:, 1:].mean(axis=0, keepdims=True)
            dt = t.gap[:, 1:] - t.gap[:, 1:].mean(axis=0, keepdims=True)
            ed.append(float(np.sqrt(np.mean((de - dt) ** 2))))
            eg.append(float(np.sqrt(np.mean((f.gap[:, 1:] - t.gap[:, 1:]) ** 2))))
            # Theoretical floor: the invisible drift c_l = mean_i delta[i, l].
            c = t.delta[:, 1:].mean(axis=0)
            fl.append(float(np.sqrt(np.mean(c**2))))
        err_did.append(float(np.mean(ed)))
        err_gap.append(float(np.mean(eg)))
        floors.append(float(np.mean(fl)))

    did_ratio = err_did[-1] / max(err_did[0], 1e-9)
    led.record(
        "C6.4",
        "DiD error converges toward zero as (M, I) grow (identified)",
        "RMSE(largest) < 0.60 * RMSE(smallest)",
        did_ratio,
        did_ratio < 0.60,
        detail=f"RMSE {' -> '.join(f'{e:.3f}' for e in err_did)}",
    )
    rel = abs(err_gap[-1] - floors[-1]) / max(floors[-1], 1e-9)
    led.record(
        "C6.5",
        "raw gap error plateaus at the theory-predicted floor sqrt(E[c^2])",
        "within 20% of the floor at the largest size",
        rel,
        rel < 0.20,
        detail=(f"RMSE {' -> '.join(f'{e:.3f}' for e in err_gap)} | floor {floors[-1]:.3f}"),
    )
    # A third claim comparing the two ratios was dropped: it passed at 2.02
    # against a >2 threshold, which is threshold-shopping, and it adds nothing
    # that C6.4 and C6.5 do not already establish more directly.
    return {
        "sizes": sizes,
        "err_did": err_did,
        "err_gap": err_gap,
        "floors": floors,
    }


def check_prop7(led: Ledger, rng: np.random.Generator) -> None:
    """C7.1 -- the accuracy-gap decomposition holds to first order."""
    M, I = 6, 4000
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))  # noqa: E731
    th_en = rng.normal(0, 1, size=M)
    gap = np.abs(rng.normal(0, 0.4, size=M))
    b = rng.normal(0, 1, size=I)
    delta = rng.normal(0, 0.05, size=I)
    observed = sig(th_en[:, None] - b[None, :]).mean(1) - sig(
        (th_en - gap)[:, None] - b[None, :] - delta[None, :]
    ).mean(1)
    p = sig(th_en[:, None] - b[None, :])
    w = p * (1 - p)
    predicted = w.mean(1) * gap + (w * delta[None, :]).mean(1)
    err = float(np.max(np.abs(observed - predicted)))
    led.record(
        "C7.1",
        "accuracy gap = ability term + drift term (first order)",
        "max err < 0.02",
        err,
        err < 0.02,
    )


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def make_figures(art: dict, outdir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    # Fig 1 -- one-for-one contamination
    s = art["contamination"]
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    d = np.array(s["drifts"])
    for curve in s["curves"]:
        ax.plot(d, curve, "o-", alpha=0.55, ms=4, lw=1.2)
    ax.plot(d, d, "k--", lw=1.4, label="slope 1 (theory)")
    ax.set_xlabel(r"invisible uniform drift $\bar\delta_\ell$")
    ax.set_ylabel(r"error in reported gap $\tilde G - G$")
    ax.set_title("Reported gap absorbs invisible drift\none-for-one (Thm 1)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figdir / "fig1_contamination.pdf")
    plt.close(fig)

    # Fig 2 -- identified-set width
    w = art["widths"]
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    for k, v in w["widths"].items():
        ax.plot(w["taus"], v, "o-", ms=4, label=k)
    ax.set_xlabel(r"tolerance $\tau$")
    ax.set_ylabel("identified-set width")
    ax.set_title("Width of the sharp identified set (Thm 3)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figdir / "fig2_width.pdf")
    plt.close(fig)

    # Fig 3 -- the killer figure: reordering under admissible normalisations
    r = art["reordering"]
    if r:
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        L = len(r["raw"])
        x = np.arange(1, L)
        for key, mark in (("raw", "o"), ("median", "s"), ("external", "^"), ("true", "d")):
            ax.plot(x, np.array(r[key])[1:], mark + "-", ms=5, lw=1.2, label=f"anchor: {key}")
        ax.set_xlabel("language index")
        ax.set_ylabel("mean reported gap (logits)")
        ax.set_title(
            "Same data, three defensible anchors,\ndifferent language ordering", fontsize=10
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(figdir / "fig3_reordering.pdf")
        plt.close(fig)

    # Fig 4 -- the decisive plot: convergence vs. an irreducible floor
    cs = art.get("consistency")
    if cs:
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        n = [m * i for m, i in cs["sizes"]]
        ax.plot(n, cs["err_did"], "o-", ms=5, label="difference-in-differences (identified)")
        ax.plot(n, cs["err_gap"], "s-", ms=5, label="raw cross-lingual gap")
        ax.axhline(
            cs["floors"][-1],
            color="k",
            ls="--",
            lw=1.2,
            label=r"floor $\sqrt{\mathbb{E}[c^2]}$ (Thm 3b)",
        )
        ax.set_xscale("log")
        ax.set_xlabel(r"data size $M \times I$")
        ax.set_ylabel("RMSE (logits)")
        ax.set_ylim(bottom=0)
        ax.set_title("More data fixes the identified quantity,\nnot the gap", fontsize=10)
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(figdir / "fig4_consistency.pdf")
        plt.close(fig)

    print(f"  figures -> {figdir}")


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/runs/toy"))
    ap.add_argument("--seeds", type=str, default="0,1,2")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    led = Ledger()
    rng = np.random.default_rng(12345)
    art: dict = {}

    print("\nTheorem 1 -- invariance and non-identifiability")
    check_thm1_invariance_is_exact(led, rng)
    art["contamination"] = check_thm1_contamination_slope(led, seeds)
    check_thm1_identified_survive(led, seeds)
    check_thm1_uniform_invisible(led, seeds)

    print("\nTheorem 2 -- 2PL affine escalation")
    check_thm2_affine(led, rng)

    print("\nTheorem 3 -- sharp partial identification")
    art["widths"] = check_thm3_sharpness(led, rng)
    art["coverage"] = check_thm3_coverage(led, seeds)

    print("\nTheorem 4 -- median anchoring")
    check_thm4_anchor(led, rng)

    print("\nTheorem 5 -- detectability dichotomy")
    check_thm5_dichotomy(led, seeds)

    print("\nEmpirical consequence -- normalisation sensitivity")
    art["reordering"] = check_reordering(led, seeds)

    print("\nConsistency sweep -- identified converges, the gap does not")
    art["consistency"] = check_consistency_sweep(led, seeds)

    print("\nProposition 7 -- accuracy-gap decomposition")
    check_prop7(led, rng)

    make_figures(art, args.out)

    n_pass = sum(c.passed for c in led.claims)
    summary = {
        "n_claims": len(led.claims),
        "n_passed": n_pass,
        "all_passed": led.all_passed,
        "seeds": seeds,
        "elapsed_sec": round(time.time() - t0, 1),
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "claims": [asdict(c) for c in led.claims],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    rows = ["| id | claim | predicted | measured | status |", "|---|---|---|---|---|"]
    rows += [
        f"| {c.claim_id} | {c.statement} | {c.predicted} | {c.measured:.4f} | "
        f"{'VERIFIED' if c.passed else '**FAILED**'} |"
        for c in led.claims
    ]
    (args.out / "claims.md").write_text("\n".join(rows) + "\n")

    print(f"\n{n_pass}/{len(led.claims)} claims verified in {summary['elapsed_sec']}s")
    print(f"  ledger  -> {args.out / 'claims.md'}")
    print(f"  summary -> {args.out / 'summary.json'}")
    return 0 if led.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
