"""Validity ablations: is the measured drift real, or an artefact of scoring?

    python -m scripts.ablate --check scoring   --benchmarks belebele
    python -m scripts.ablate --check translation

``scoring``
    Compare drift estimated under letter scoring against loglik scoring on the
    same items and models. This is the paper's main threat to validity: letter
    scoring depends on the model following an answer format, format-following
    degrades in lower-resource languages, and that degradation would enter the
    response tensor as language-specific item difficulty -- exactly the signal we
    attribute to translation. If the two scoring modes yield correlated drift and
    similar breakeven values, the finding is about translation. If they diverge,
    it is about formatting, and the paper must say so.

``translation``
    Compare drift dispersion on machine-translated Global-MMLU against fully
    human-translated Global-MMLU-Lite over their shared languages. This is the
    closest the data come to a direct read on the size of translation-induced
    difficulty change, and it speaks to the plausibility of the uniform component
    that Theorem 5 says is invisible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from mirage.data.loaders import load_specs
from mirage.identify import breakeven_drift
from mirage.irt import fit_irt

ROOT = Path(__file__).resolve().parents[1]
BENCH_CFG = ROOT / "configs" / "benchmarks.yaml"

sys.path.insert(0, str(ROOT / "scripts"))
from analyse import assemble  # noqa: E402


def _fit(blob: dict, spec, n_iter: int) -> tuple:
    langs = blob["languages"]
    src = spec.source_language
    si = langs.index(src)
    fit = fit_irt(blob["responses"], source_index=si, n_iter=n_iter, seed=0)
    return fit, si


def check_scoring(args) -> dict:
    """Letter vs loglik drift on the same (model, language) cells."""
    specs = load_specs(BENCH_CFG)
    out = []
    for bench in args.benchmarks.split(","):
        a = assemble(args.tensors, bench, "letter")
        b = assemble(args.tensors, bench, "loglik")
        if a is None or b is None:
            print(f"[{bench}] missing one scoring mode, skipping")
            continue

        # Restrict to the cells scored under both modes, so the comparison is
        # paired rather than an average over different panels.
        models = sorted(set(a["models"]) & set(b["models"]))
        langs = sorted(set(a["languages"]) & set(b["languages"]))
        if len(models) < 4 or len(langs) < 3:
            print(f"[{bench}] overlap too small ({len(models)}m x {len(langs)}l)")
            continue
        ai = [a["models"].index(m) for m in models]
        al = [a["languages"].index(x) for x in langs]
        bi = [b["models"].index(m) for m in models]
        bl = [b["languages"].index(x) for x in langs]
        Ya = a["responses"][np.ix_(ai, range(a["responses"].shape[1]), al)]
        Yb = b["responses"][np.ix_(bi, range(b["responses"].shape[1]), bl)]

        spec = specs[bench]
        si = langs.index(spec.source_language)
        fa = fit_irt(Ya, source_index=si, n_iter=args.n_iter, seed=0)
        fb = fit_irt(Yb, source_index=si, n_iter=args.n_iter, seed=0)

        rs, gs = [], []
        for l in range(len(langs)):
            if l == si:
                continue
            r = float(np.corrcoef(fa.delta[:, l], fb.delta[:, l])[0, 1])
            rs.append(r)
            gs.append((float(fa.gap[:, l].mean()), float(fb.gap[:, l].mean())))
        gap_a = np.array([g[0] for g in gs])
        gap_b = np.array([g[1] for g in gs])
        rec = {
            "benchmark": bench,
            "n_models": len(models),
            "n_languages": len(langs),
            "drift_corr_mean": float(np.mean(rs)),
            "drift_corr_min": float(np.min(rs)),
            "gap_corr": float(np.corrcoef(gap_a, gap_b)[0, 1]),
            "gap_mean_abs_diff": float(np.mean(np.abs(gap_a - gap_b))),
            "gap_letter_mean": float(gap_a.mean()),
            "gap_loglik_mean": float(gap_b.mean()),
        }
        out.append(rec)
        print(
            f"[{bench}] drift r={rec['drift_corr_mean']:.3f} "
            f"(min {rec['drift_corr_min']:.3f}) | gap r={rec['gap_corr']:.3f} | "
            f"mean |Delta gap|={rec['gap_mean_abs_diff']:.3f}"
        )
    return {"check": "scoring", "results": out}


def check_translation(args) -> dict:
    """Machine-translated vs human-translated drift dispersion."""
    specs = load_specs(BENCH_CFG)
    mt = assemble(args.tensors, "global_mmlu", "letter")
    ht = assemble(args.tensors, "global_mmlu_lite", "letter")
    if mt is None or ht is None:
        print("need both global_mmlu and global_mmlu_lite; skipping")
        return {"check": "translation", "results": []}

    shared_l = sorted(set(mt["languages"]) & set(ht["languages"]))
    shared_m = sorted(set(mt["models"]) & set(ht["models"]))
    if len(shared_l) < 3 or len(shared_m) < 4:
        print(f"overlap too small: {len(shared_m)} models x {len(shared_l)} langs")
        return {"check": "translation", "results": []}

    rows = []
    for name, blob in (("machine", mt), ("human", ht)):
        spec = specs["global_mmlu" if name == "machine" else "global_mmlu_lite"]
        mi = [blob["models"].index(m) for m in shared_m]
        li = [blob["languages"].index(x) for x in shared_l]
        Y = blob["responses"][np.ix_(mi, range(blob["responses"].shape[1]), li)]
        si = shared_l.index(spec.source_language)
        fit = fit_irt(Y, source_index=si, n_iter=args.n_iter, seed=0)
        se = fit.drift_se()
        be = breakeven_drift(fit.gap, fit.delta, se)
        keep = [l for l in range(len(shared_l)) if l != si]
        rows.append({
            "arm": name,
            "n_items": Y.shape[1],
            "drift_sd_mean": float(np.mean(be["drift_sd"][keep])),
            "gap_mean": float(np.mean(fit.gap[:, keep])),
            "breakeven_in_sd_median": float(np.median(be["ratio"][keep])),
        })
        print(
            f"[{name:>7}] I={Y.shape[1]:<5} drift sd={rows[-1]['drift_sd_mean']:.3f} "
            f"gap={rows[-1]['gap_mean']:+.3f} "
            f"breakeven={rows[-1]['breakeven_in_sd_median']:.2f} sd"
        )

    if len(rows) == 2:
        ratio = rows[0]["drift_sd_mean"] / max(rows[1]["drift_sd_mean"], 1e-9)
        print(
            f"\nmachine/human drift-dispersion ratio: {ratio:.2f}\n"
            "  >1 means machine translation perturbs item difficulty more, which\n"
            "  makes a non-zero *mean* drift more plausible on the MT arm."
        )
    return {
        "check": "translation",
        "shared_languages": shared_l,
        "shared_models": shared_m,
        "results": rows,
    }


def check_position(args) -> dict:
    """Position bias: is the modal answer option constant across languages?

    A constant preference for option A is absorbed into item difficulty ``b_i``
    and does not confound the gap. A preference that *varies by language* does
    confound it. This reports both.
    """
    out = []
    for bench in args.benchmarks.split(","):
        blob = assemble(args.tensors, bench, "letter")
        if blob is None:
            continue
        ms = blob["mode_share"]
        langs = blob["languages"]
        spread = float(np.std(ms.mean(axis=0)))
        rec = {
            "benchmark": bench,
            "mode_share_mean": float(ms.mean()),
            "mode_share_by_language_sd": spread,
            "frac_degenerate": float((ms > 0.90).mean()),
            "worst_languages": [
                langs[i] for i in np.argsort(-ms.mean(axis=0))[:3]
            ],
        }
        out.append(rec)
        print(
            f"[{bench}] mode share {rec['mode_share_mean']:.3f} "
            f"(sd across languages {spread:.3f}) | "
            f"degenerate {100 * rec['frac_degenerate']:.1f}% | "
            f"worst: {', '.join(rec['worst_languages'])}"
        )
    return {"check": "position", "results": out}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", required=True,
                    choices=["scoring", "translation", "position"])
    ap.add_argument("--tensors", type=Path, default=ROOT / "results" / "tensors")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "runs" / "ablation")
    ap.add_argument("--benchmarks", default="belebele")
    ap.add_argument("--n-iter", type=int, default=4000)
    args = ap.parse_args()

    fn = {"scoring": check_scoring, "translation": check_translation,
          "position": check_position}[args.check]
    rec = fn(args)
    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / f"{args.check}.json"
    dest.write_text(json.dumps(rec, indent=2))
    print(f"\n-> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
