"""Verify benchmark coordinates and prefetch them into the HF cache.

Run ``--verify-only`` first. Any benchmark that fails here must not be cited:
the config marks it ``verified: false`` and the loader refuses to use it.

    python -m scripts.download --verify-only
    python -m scripts.download                # verify, then materialise
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mirage.data.loaders import _item_ids, load_specs, select_item_ids

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "benchmarks.yaml"


def probe(spec, n_probe: int, cache_dir: str) -> dict:
    """Try to load a couple of languages and report what actually exists."""
    from datasets import get_dataset_config_names, load_dataset

    out: dict = {"benchmark": spec.name, "hf_id": spec.hf_id, "ok": False}
    if not spec.usable:
        out["error"] = f"unresolved coordinates (hf_id={spec.hf_id!r})"
        return out

    try:
        configs = get_dataset_config_names(spec.hf_id, revision=spec.revision)
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"cannot list configs: {type(exc).__name__}: {exc}"
        return out

    out["n_configs_available"] = len(configs)
    missing = [lg for lg in spec.languages if lg not in configs]
    out["languages_missing"] = missing
    present = [lg for lg in spec.languages if lg in configs]
    out["n_languages_present"] = len(present)
    if not present:
        out["error"] = f"none of the configured languages exist; first available: {configs[:8]}"
        return out

    ids: dict[str, list[str]] = {}
    for lg in present[:n_probe]:
        try:
            ds = load_dataset(
                spec.hf_id,
                lg,
                split=spec.split,
                revision=spec.revision,
                cache_dir=cache_dir,
            )
        except Exception as exc:  # noqa: BLE001
            out["error"] = f"load_dataset({lg}) failed: {type(exc).__name__}: {exc}"
            return out
        out.setdefault("n_rows", {})[lg] = len(ds)
        out.setdefault("columns", sorted(ds.column_names))
        fld = spec.item_id_field
        needed_id = fld if isinstance(fld, list) else ([] if fld == "__row_index__" else [fld])
        absent = [c for c in needed_id if c not in ds.column_names]
        if absent:
            out["error"] = (
                f"item_id_field {fld!r} references missing columns {absent}; "
                f"available: {sorted(ds.column_names)}"
            )
            return out
        ids[lg] = _item_ids(ds, fld)
        n_uniq = len(set(ids[lg]))
        out.setdefault("n_unique_ids", {})[lg] = n_uniq
        # The identifier must be one-per-row. Comparing uniques to rows catches a
        # non-identifier (Belebele's question_number had 2 values for 900 rows)
        # without tripping on a legitimate max_items cap applied later.
        if n_uniq < 0.9 * len(ds):
            out["error"] = (
                f"item_id_field {fld!r} is not unique per row in {lg}: "
                f"{n_uniq} distinct values for {len(ds)} rows. Use a composite "
                f"key or __row_index__."
            )
            return out

    # Field mapping must resolve against the real columns.
    cols = set(out.get("columns", []))
    needed: list[str] = []
    for key, val in (spec.fields or {}).items():
        if key == "answer_base":
            continue
        needed += [val] if isinstance(val, str) else list(val)
    unknown = [c for c in needed if c not in cols]
    if unknown:
        out["error"] = f"field mapping references missing columns: {unknown}"
        return out

    if len(ids) >= 2:
        try:
            sel, coverage = select_item_ids(ids, spec.max_items, spec.subsample_seed)
            out["n_selected"] = len(sel)
            out["max_unshared_frac"] = max((d / t if t else 0.0) for t, d in coverage.values())
        except ValueError as exc:
            out["error"] = f"not parallel: {exc}"
            return out

    out["ok"] = True
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--probe-languages", type=int, default=3)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--only", default=None, help="comma-separated benchmark names")
    ap.add_argument("--out", type=Path, default=Path("results/runs/download_report.json"))
    args = ap.parse_args()

    specs = load_specs(args.config)
    if args.only:
        want = set(args.only.split(","))
        specs = {k: v for k, v in specs.items() if k in want}

    reports = []
    for name, spec in specs.items():
        print(f"\n=== {name} ({spec.hf_id}) ===", flush=True)
        rep = probe(spec, args.probe_languages, args.cache_dir)
        reports.append(rep)
        if rep["ok"]:
            rows = rep.get("n_rows", {})
            print(
                f"  OK  langs {rep['n_languages_present']}/{len(spec.languages)}"
                f"  rows {rows}  selected {rep.get('n_selected', '-')}"
            )
            if rep.get("languages_missing"):
                print(f"  WARN missing configs: {rep['languages_missing']}")
        else:
            print(f"  FAIL {rep['error']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(reports, indent=2))
    ok = [r["benchmark"] for r in reports if r["ok"]]
    bad = [r["benchmark"] for r in reports if not r["ok"]]
    print(f"\nusable: {ok}")
    print(f"unusable: {bad}")
    print(f"report -> {args.out}")

    if not args.verify_only:
        from datasets import load_dataset

        for rep in reports:
            if not rep["ok"]:
                continue
            spec = specs[rep["benchmark"]]
            for lg in spec.languages:
                if lg in (rep.get("languages_missing") or []):
                    continue
                load_dataset(
                    spec.hf_id,
                    lg,
                    split=spec.split,
                    revision=spec.revision,
                    cache_dir=args.cache_dir,
                )
            print(f"materialised {rep['benchmark']}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
