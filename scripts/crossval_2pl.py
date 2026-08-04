"""Does language-varying discrimination generalise, or only fit?

    python -m scripts.crossval_2pl

The paper claimed that our data "reject Rasch" on the strength of 2PL improving
log-loss. That claim did not follow: 2PL adds ``I x L`` discrimination parameters,
so it fits the training responses better by construction, and a better in-sample
number is evidence of nothing at all.

This script settles it out of sample. A random 10% of ``(model, item, language)``
cells is masked, both models are fitted on the remainder, and log-loss is scored
**only on the masked cells**. ``fit_irt`` already accepts an observation mask, so
no new estimation machinery is involved.

The design is fixed in advance -- 10% of cells, three seeds, every benchmark --
and the result is reported whichever way it falls. If 2PL does not win here, the
paper's wording moves from "our data reject Rasch" to "the 2PL specification
improves in-sample fit, suggesting language-varying discrimination may be
empirically relevant", and the claim that difference-in-differences is unavailable
in practice weakens to theoretically fragile.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from mirage.data.loaders import load_specs
from mirage.irt import fit_irt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyse import assemble  # noqa: E402


def held_out_logloss(fit, Y: np.ndarray, mask_out: np.ndarray) -> float:
    """Mean binary log-loss on the cells the fit never saw.

    Computed as ``softplus(eta) - y*eta`` rather than through an explicit
    sigmoid. The 2PL fit drives some real-data logits past 150 in magnitude,
    where ``exp(-eta)`` overflows and the naive expression returns NaN -- which
    an earlier version of this script reported as "Rasch wins" on all eight
    benchmarks. ``logaddexp(0, eta)`` is exact over the whole range.
    """
    eta = np.asarray(fit.cell_logits(), dtype=np.float64)[mask_out]
    y = np.asarray(Y, dtype=np.float64)[mask_out]
    if not np.all(np.isfinite(eta)):
        n_bad = int((~np.isfinite(eta)).sum())
        raise ValueError(
            f"{n_bad} non-finite logits in the fitted model; the fit did not "
            f"converge and its held-out loss would be meaningless"
        )
    return float(np.mean(np.logaddexp(0.0, eta) - y * eta))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tensors", type=Path, default=ROOT / "results" / "tensors")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "runs" / "ablation" / "crossval.json")
    ap.add_argument("--holdout", type=float, default=0.10)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n-iter", type=int, default=4000)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    specs = load_specs(ROOT / "configs" / "benchmarks.yaml")
    out = []

    for bench, spec in specs.items():
        mode = "generative" if spec.task == "generative" else "letter"
        blob = assemble(args.tensors, bench, mode)
        if blob is None or blob["missing"]:
            continue
        Y = blob["responses"]
        src = spec.source_language
        if src not in blob["languages"]:
            continue
        si = blob["languages"].index(src)

        rasch, twopl = [], []
        for s in seeds:
            rng = np.random.default_rng(1000 + s)
            keep = rng.random(Y.shape) >= args.holdout
            held = ~keep
            if held.sum() == 0:
                continue
            f1 = fit_irt(Y, mask=keep, model="rasch", source_index=si,
                         n_iter=args.n_iter, seed=s)
            f2 = fit_irt(Y, mask=keep, model="2pl", source_index=si,
                         n_iter=args.n_iter, seed=s)
            rasch.append(held_out_logloss(f1, Y, held))
            twopl.append(held_out_logloss(f2, Y, held))

        if not rasch:
            continue
        r, t = float(np.mean(rasch)), float(np.mean(twopl))
        rec = {
            "benchmark": bench,
            "n_models": len(blob["models"]),
            "n_items": Y.shape[1],
            "n_languages": len(blob["languages"]),
            "holdout": args.holdout,
            "seeds": seeds,
            "rasch_heldout_logloss": r,
            "twopl_heldout_logloss": t,
            "gain": r - t,          # positive means 2PL generalises better
            "twopl_wins": bool(t < r),
            "per_seed_rasch": rasch,
            "per_seed_2pl": twopl,
        }
        out.append(rec)
        verdict = "2PL wins" if t < r else "Rasch wins"
        print(f"{bench:<18} rasch {r:.4f}  2pl {t:.4f}  gain {r - t:+.4f}  -> {verdict}")

    if out:
        wins = sum(r["twopl_wins"] for r in out)
        print(f"\n2PL generalises better on {wins}/{len(out)} benchmarks")
        print(f"mean held-out gain: {np.mean([r['gain'] for r in out]):+.4f} nats")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2))
        print(f"-> {args.out}")
    return 0 if out else 1


if __name__ == "__main__":
    sys.exit(main())
