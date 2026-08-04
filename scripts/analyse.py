"""Assemble response tensors and produce every number the paper reports.

    python -m scripts.analyse --benchmarks belebele,xnli --alpha 0.05

For each benchmark this
  1. assembles the (M, I, L) tensor from the shards written by ``scripts.score``,
  2. fits the multilingual IRT model under the declared sum-zero normalisation,
  3. bootstraps over models for standard errors,
  4. tests the identified half of the drift with BH-FDR control (Theorem 5),
  5. reports the raw accuracy gap, the naive IRT gap, the median-anchored
     estimate, and the sharp identified interval side by side,
and writes a JSON record plus a LaTeX table per benchmark.

Only benchmarks whose tensor is complete are analysed. A partially scored
benchmark is reported as such and skipped -- an unbalanced tensor would break
the crossed design that every estimand assumes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from mirage.analysis.dif import dif_test, drift_bootstrap_se
from mirage.data.loaders import load_specs
from mirage.identify import breakeven_drift, gap_bounds, median_anchor
from mirage.irt import fit_irt

ROOT = Path(__file__).resolve().parents[1]
BENCH_CFG = ROOT / "configs" / "benchmarks.yaml"

# Marks a language whose identified interval for the gap contains zero.
DAGGER = "$^\\dagger$"


def assemble(tensor_dir: Path, bench: str, mode: str = "letter") -> dict | None:
    """Build the (M, I, L) tensor for one benchmark from its shards."""
    bdir = tensor_dir / bench
    if not bdir.is_dir():
        return None
    # Shard names are ``model__language`` for letter scoring and
    # ``model__language__<mode>`` otherwise. Selecting letter mode must exclude
    # *every* suffixed variant, not just loglik: MGSM's generative shards were
    # otherwise read as letter shards whose language was "en__generative".
    other_modes = ("loglik", "generative")
    suffix = "" if mode == "letter" else f"__{mode}"
    shards: dict[tuple[str, str], dict] = {}
    for f in sorted(bdir.glob("*.npz")):
        stem = f.stem
        if suffix:
            if not stem.endswith(suffix):
                continue
        elif any(stem.endswith(f"__{m}") for m in other_modes):
            continue
        core = stem[: -len(suffix)] if suffix else stem
        model, _, lang = core.partition("__")
        if not lang:
            continue
        z = np.load(f, allow_pickle=False)
        shards[(model, lang)] = {
            "item_ids": [str(x) for x in z["item_ids"]],
            "correct": z["correct"],
            "predicted": z["predicted"],
            "meta": json.loads(str(z["meta"])),
        }
    if not shards:
        return None

    models = sorted({m for m, _ in shards})
    langs = sorted({l for _, l in shards})
    missing = [(m, l) for m in models for l in langs if (m, l) not in shards]

    # Align every shard to one canonical item order.
    #
    # Shards may legitimately list the same items in different orders, because a
    # benchmark's language versions can order their rows differently (Belebele
    # does). Each shard records its own item_ids alongside its responses, so the
    # responses are permuted into the canonical order here rather than assumed to
    # match by position -- assuming position would silently pair English item k
    # with a different question in another language.
    ref_ids = sorted(shards[(models[0], langs[0])]["item_ids"])
    pos = {iid: k for k, iid in enumerate(ref_ids)}
    n_items = len(ref_ids)
    Y = np.zeros((len(models), n_items, len(langs)), dtype=np.int8)
    for (m, l), blob in shards.items():
        ids = blob["item_ids"]
        if set(ids) != set(ref_ids):
            extra = len(set(ids) - set(ref_ids))
            miss = len(set(ref_ids) - set(ids))
            raise RuntimeError(
                f"{bench}: shard {m}/{l} covers a different item *set* than "
                f"{models[0]}/{langs[0]} ({extra} extra, {miss} missing). Shards "
                f"were produced under different subsample seeds or configs; "
                f"delete results/tensors/{bench} and rescore."
            )
        order = np.array([pos[i] for i in ids])
        Y[models.index(m), order, langs.index(l)] = blob["correct"]

    # Degeneracy per (model, language): a shard whose predictions collapse onto
    # one option carries no information about item difficulty, only about the
    # model failing the task in that language. It is tracked rather than dropped,
    # because degeneracy is language-correlated and excluding it would select the
    # sample on the outcome -- the same trap as dropping high-fertility scripts.
    mode_share = np.ones((len(models), len(langs)))
    for (m, l), blob in shards.items():
        ms = blob["meta"].get("mode_share")
        if ms is None:
            p = blob["predicted"]
            valid = p[p >= 0]
            ms = float(np.bincount(valid).max() / valid.size) if valid.size else 1.0
        mode_share[models.index(m), langs.index(l)] = ms

    return {
        "responses": Y,
        "models": models,
        "languages": langs,
        "item_ids": ref_ids,
        "missing": missing,
        "mode_share": mode_share,
        "n_shards": len(shards),
        "n_expected": len(models) * len(langs),
    }


def analyse_one(bench: str, blob: dict, spec, args) -> dict:
    Y = blob["responses"]
    models, langs = blob["models"], blob["languages"]
    src = spec.source_language
    if src not in langs:
        raise RuntimeError(
            f"{bench}: source language {src!r} absent from scored languages "
            f"{langs}; delta = 0 would have no anchor"
        )
    si = langs.index(src)

    fit = fit_irt(Y, model=args.irt_model, source_index=si, n_iter=args.n_iter, seed=0)
    delta_se_analytic = fit.drift_se()

    boot = drift_bootstrap_se(
        Y,
        lambda t: fit_irt(t, model=args.irt_model, source_index=si, n_iter=args.n_iter, seed=0),
        n_boot=args.n_boot,
        seed=0,
        source_index=si,
    )
    delta_se = np.maximum(boot["delta_se"], 1e-6)

    dif = dif_test(fit.delta, delta_se, alpha=args.alpha, source_index=si)

    # tau must absorb estimation noise before the identified set is reportable.
    tau = args.tau + 2.0 * float(
        np.median(delta_se[:, [i for i in range(len(langs)) if i != si]])
    )
    inflate = 2.0 * float(np.median(boot["gap_se"]))

    ms = blob["mode_share"]
    acc = Y.mean(axis=1)  # (M, L)
    raw_acc_gap = acc[:, si][:, None] - acc

    be = breakeven_drift(fit.gap, fit.delta, delta_se)

    # Robustness: recompute the breakeven using only models that did *not*
    # degenerate in each language. The languages with the largest gaps are also
    # the ones where models most often collapse onto a single option, so the
    # headline claim must be shown to survive their removal. When every model
    # degenerates in a language the fallback is the unweighted value, flagged by
    # `n_nondegenerate = 0`.
    be_robust_gap = np.zeros(len(langs))
    n_nondeg = np.zeros(len(langs), dtype=int)
    for l in range(len(langs)):
        # NaN (generative shards) is never treated as degenerate.
        ok = ~(ms[:, l] > 0.90)
        n_nondeg[l] = int(ok.sum())
        be_robust_gap[l] = float(fit.gap[ok, l].mean()) if ok.any() else float(
            fit.gap[:, l].mean()
        )
    lo, hi, sets = gap_bounds(fit.gap, fit.delta, pi=args.pi, tau=tau, inflate=inflate)
    c_med = np.array(
        [0.0 if l == si else median_anchor(fit.delta[:, l]) for l in range(len(langs))]
    )
    gap_med = fit.gap - c_med[None, :]

    rows = []
    for l, lang in enumerate(langs):
        if l == si:
            continue
        contains_zero = bool(lo[:, l].min() <= 0 <= hi[:, l].max())
        rows.append(
            {
                "language": lang,
                "acc_source": float(acc[:, si].mean()),
                "acc_target": float(acc[:, l].mean()),
                "raw_acc_gap": float(raw_acc_gap[:, l].mean()),
                "irt_gap_naive": float(fit.gap[:, l].mean()),
                "gap_median_anchored": float(gap_med[:, l].mean()),
                "bound_lo": float(lo[:, l].mean()),
                "bound_hi": float(hi[:, l].mean()),
                "interval_width": float((hi[:, l] - lo[:, l]).mean()),
                "c_set_lo": sets[l].hull.lo,
                "c_set_hi": sets[l].hull.hi,
                "contains_zero": contains_zero,
                "dif_rejected": int(dif["rejected"][:, l].sum()),
                "dif_frac": float(dif["rejected"][:, l].mean()),
                # Sensitivity: the uniform drift that would erase this gap, and
                # its size relative to the item drift we can actually see.
                "breakeven_drift": float(be["breakeven"][l]),
                "observed_drift_sd": float(be["drift_sd"][l]),
                "breakeven_in_sd": float(be["ratio"][l]),
                "gap_se": float(boot["gap_se"][l]),
                "mode_share_mean": float(ms[:, l].mean()),
                "n_degenerate": int((ms[:, l] > 0.90).sum()),
                "n_nondegenerate": int(n_nondeg[l]),
                "breakeven_robust": float(be_robust_gap[l]),
                "breakeven_robust_in_sd": (
                    float(abs(be_robust_gap[l]) / be["drift_sd"][l])
                    if be["drift_sd"][l] > 0
                    else float("inf")
                ),
            }
        )

    n_zero = sum(r["contains_zero"] for r in rows)
    return {
        "benchmark": bench,
        "n_models": len(models),
        "n_items": Y.shape[1],
        "n_languages": len(langs),
        "models": models,
        "languages": langs,
        "source_language": src,
        "irt_model": args.irt_model,
        "pi": args.pi,
        "tau_nominal": args.tau,
        "tau_effective": tau,
        "inflate": inflate,
        "n_boot": args.n_boot,
        "alpha": args.alpha,
        "median_delta_se": float(np.median(delta_se)),
        "median_delta_se_analytic": float(np.median(delta_se_analytic)),
        "dif_frac_rejected": float(dif["frac_rejected"]),
        "dif_n_rejected": int(dif["n_rejected"]),
        "dif_n_tested": int(dif["n_tested"]),
        "n_languages_gap_contains_zero": n_zero,
        "frac_languages_gap_contains_zero": n_zero / max(len(rows), 1),
        "mean_interval_width": float(np.mean([r["interval_width"] for r in rows])),
        "frac_shards_degenerate": float(np.nan_to_num(ms > 0.90).mean()),
        "median_breakeven_in_sd": float(np.median([r["breakeven_in_sd"] for r in rows])),
        "n_gaps_breakeven_below_1sd": int(
            sum(r["breakeven_in_sd"] < 1.0 for r in rows)
        ),
        "n_gaps_breakeven_robust_below_1sd": int(
            sum(r["breakeven_robust_in_sd"] < 1.0 for r in rows)
        ),
        "median_breakeven_robust_in_sd": float(
            np.median([r["breakeven_robust_in_sd"] for r in rows])
        ),
        "rows": rows,
    }


def latex_table(rec: dict) -> str:
    head = (
        "\\begin{tabular}{lrrrrc}\n\\toprule\n"
        "Language & Acc. gap & IRT gap & Anchored & Identified interval & DIF \\\\\n"
        "\\midrule\n"
    )
    body = "".join(
        f"{r['language']} & {r['raw_acc_gap']:.3f} & {r['irt_gap_naive']:.3f} & "
        f"{r['gap_median_anchored']:.3f} & "
        f"[{r['bound_lo']:.2f}, {r['bound_hi']:.2f}]"
        f"{DAGGER if r['contains_zero'] else ''} & "
        f"{r['dif_frac']:.2f} \\\\\n"
        for r in rec["rows"]
    )
    return head + body + "\\bottomrule\n\\end{tabular}\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tensors", type=Path, default=ROOT / "results" / "tensors")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "runs" / "analysis")
    ap.add_argument("--benchmarks", default=None)
    ap.add_argument("--mode", default="letter")
    ap.add_argument("--irt-model", default="rasch", choices=["rasch", "2pl"])
    ap.add_argument("--pi", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=0.30)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n-boot", type=int, default=20)
    ap.add_argument("--n-iter", type=int, default=4000)
    ap.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args()

    specs = load_specs(BENCH_CFG)
    names = args.benchmarks.split(",") if args.benchmarks else list(specs)
    args.out.mkdir(parents=True, exist_ok=True)

    records = []
    for bench in names:
        # A generative benchmark has no letter-mode shards; analyse it in the
        # mode it was scored in rather than reporting it as missing.
        bench_mode = "generative" if specs[bench].task == "generative" else args.mode
        blob = assemble(args.tensors, bench, bench_mode)
        if blob is None:
            print(f"[{bench}] no shards, skipping")
            continue
        if blob["missing"] and not args.allow_incomplete:
            print(
                f"[{bench}] INCOMPLETE: {blob['n_shards']}/{blob['n_expected']} shards "
                f"({len(blob['missing'])} missing). Skipping -- an unbalanced tensor "
                f"breaks the crossed design. Use --allow-incomplete to override."
            )
            continue
        print(
            f"[{bench}] {blob['responses'].shape} "
            f"(M={len(blob['models'])}, I={blob['responses'].shape[1]}, "
            f"L={len(blob['languages'])})",
            flush=True,
        )
        try:
            rec = analyse_one(bench, blob, specs[bench], args)
        except Exception as exc:  # noqa: BLE001
            print(f"[{bench}] FAILED: {type(exc).__name__}: {exc}")
            continue

        records.append(rec)
        (args.out / f"{bench}.json").write_text(json.dumps(rec, indent=2))
        (ROOT / "paper" / "tables").mkdir(parents=True, exist_ok=True)
        (ROOT / "paper" / "tables" / f"{bench}.tex").write_text(latex_table(rec))
        print(
            f"    DIF rejected {rec['dif_n_rejected']}/{rec['dif_n_tested']} "
            f"({rec['dif_frac_rejected']:.1%}) | "
            f"gaps containing zero: {rec['n_languages_gap_contains_zero']}"
            f"/{len(rec['rows'])} | mean width {rec['mean_interval_width']:.2f}",
            flush=True,
        )

    if records:
        (args.out / "summary.json").write_text(json.dumps(records, indent=2))
        print(f"\n{len(records)} benchmarks analysed -> {args.out}")
    return 0 if records else 1


if __name__ == "__main__":
    sys.exit(main())
