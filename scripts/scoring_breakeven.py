"""Letter vs log-likelihood scoring, computed through the *main* analysis path.

    python -m scripts.scoring_breakeven

An earlier version of this comparison lived in a scratch script that used the
analytic ``IRTFit.drift_se`` and whatever model panel happened to be scored under
both modes. Table 1 meanwhile uses the bootstrap standard error over all 19
models. The two therefore disagreed -- Table 1 reported 5/10 gaps below one sigma
on XStoryCloze while the Limitations paragraph reported 0/10 -- for reasons that
had nothing to do with scoring and everything to do with which estimator was
used. A reader has no way to tell those apart, so both numbers were untrustworthy
as presented.

This script removes the discrepancy by construction: both scoring modes go
through exactly the machinery of ``scripts.analyse`` (bootstrap SE over models,
the same noise-inflated tolerance), restricted to the panel scored under both.
Any difference that survives is attributable to the scorer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from mirage.analysis.dif import drift_bootstrap_se
from mirage.data.loaders import load_specs
from mirage.identify import breakeven_drift
from mirage.irt import fit_irt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyse import assemble  # noqa: E402


def _panel(a: dict, b: dict) -> tuple[list[str], list[str]]:
    """Models and languages present under both scoring modes."""
    return (
        sorted(set(a["models"]) & set(b["models"])),
        sorted(set(a["languages"]) & set(b["languages"])),
    )


def _slice(blob: dict, models: list[str], langs: list[str]) -> np.ndarray:
    mi = [blob["models"].index(m) for m in models]
    li = [blob["languages"].index(x) for x in langs]
    n_items = blob["responses"].shape[1]
    return blob["responses"][np.ix_(mi, range(n_items), li)]


def evaluate(Y: np.ndarray, si: int, n_boot: int, n_iter: int, tau0: float) -> dict:
    """Run the analyse.py estimator on one response tensor."""
    fit = fit_irt(Y, source_index=si, n_iter=n_iter, seed=0)
    boot = drift_bootstrap_se(
        Y,
        lambda t: fit_irt(t, source_index=si, n_iter=n_iter, seed=0),
        n_boot=n_boot,
        seed=0,
        source_index=si,
    )
    delta_se = np.maximum(boot["delta_se"], 1e-6)
    be = breakeven_drift(fit.gap, fit.delta, delta_se)
    keep = [x for x in range(Y.shape[2]) if x != si]
    ratio = np.asarray(be["ratio"])[keep]
    finite = ratio[np.isfinite(ratio)]
    other = [x for x in range(Y.shape[2]) if x != si]
    return {
        "n_languages": len(keep),
        "median_ratio": float(np.median(finite)) if finite.size else float("nan"),
        "n_below_1sd": int((ratio < 1.0).sum()),
        "drift_sd": float(np.mean(np.asarray(be["drift_sd"])[keep])),
        "mean_gap": float(np.mean(fit.gap[:, keep])),
        "median_delta_se": float(np.median(delta_se[:, other])),
        "tau_effective": float(tau0 + 2.0 * np.median(delta_se[:, other])),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tensors", type=Path, default=ROOT / "results" / "tensors")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "runs" / "ablation" / "scoring_breakeven.json")
    ap.add_argument("--benchmarks", default="xstorycloze,xcopa")
    ap.add_argument("--n-boot", type=int, default=20)
    ap.add_argument("--n-iter", type=int, default=4000)
    ap.add_argument("--tau", type=float, default=0.30)
    args = ap.parse_args()

    specs = load_specs(ROOT / "configs" / "benchmarks.yaml")
    out = []
    for bench in args.benchmarks.split(","):
        a = assemble(args.tensors, bench, "letter")
        b = assemble(args.tensors, bench, "loglik")
        if a is None or b is None:
            print(f"[{bench}] missing one scoring mode, skipping")
            continue
        models, langs = _panel(a, b)
        src = specs[bench].source_language
        if src not in langs:
            print(f"[{bench}] source language {src!r} not in the shared panel")
            continue
        si = langs.index(src)

        rec = {"benchmark": bench, "n_models": len(models), "n_languages": len(langs)}
        for mode, blob in (("letter", a), ("loglik", b)):
            rec[mode] = evaluate(_slice(blob, models, langs), si,
                                 args.n_boot, args.n_iter, args.tau)
        out.append(rec)
        L, G = rec["letter"], rec["loglik"]
        print(
            f"[{bench}] panel {len(models)} models x {len(langs)} langs\n"
            f"    letter: median {L['median_ratio']:.2f} sd | "
            f"<1sd {L['n_below_1sd']}/{L['n_languages']} | drift sd {L['drift_sd']:.2f}\n"
            f"    loglik: median {G['median_ratio']:.2f} sd | "
            f"<1sd {G['n_below_1sd']}/{G['n_languages']} | drift sd {G['drift_sd']:.2f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\n-> {args.out}")
    return 0 if out else 1


if __name__ == "__main__":
    sys.exit(main())
