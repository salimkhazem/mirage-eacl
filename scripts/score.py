"""Build the response tensor: score every (model, benchmark, language) shard.

One process per GPU. Each process takes a slice of the model grid, loads each
model once with vLLM, and sweeps its benchmarks and languages. Every shard is
written as soon as it finishes, and existing shards are skipped, so a crash
fifteen hours in costs one shard rather than the run.

    python -m scripts.score --gpu 0 --shard 0/3
    python -m scripts.score --gpu 0 --shard 0/3 --models qwen2.5-0.5b --benchmarks belebele

Shards land in ``results/tensors/<benchmark>/<model>__<language>.npz`` and are
assembled by ``scripts.analyse``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCH_CFG = ROOT / "configs" / "benchmarks.yaml"
MODEL_CFG = ROOT / "configs" / "models.yaml"


def shard_path(out: Path, bench: str, model: str, lang: str, mode: str) -> Path:
    suffix = "" if mode == "letter" else f"__{mode}"
    return out / bench / f"{model}__{lang}{suffix}.npz"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="i/n over the model grid")
    ap.add_argument("--tiers", default="1,2")
    ap.add_argument("--models", default=None, help="comma-separated names")
    ap.add_argument("--benchmarks", default=None, help="comma-separated names")
    ap.add_argument("--languages", default=None, help="comma-separated, for smoke tests")
    ap.add_argument("--mode", default="letter",
                    choices=["letter", "loglik", "generative"])
    ap.add_argument("--max-items", type=int, default=None, help="override, for smoke tests")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "tensors")
    ap.add_argument("--gpu-mem-frac", type=float, default=0.85)
    ap.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help=(
            "override the per-model context length. Needed for the highest-"
            "fertility scripts: a Telugu MMLU-ProX item (10 options, ~6x English "
            "token cost) exceeds 8192. Raising the budget only for those shards "
            "is preferable to dropping them, which would select the sample on the "
            "outcome."
        ),
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Must precede any torch/vllm import. The machine mixes a 6000 Ada with two
    # 5000 Adas, so PCI_BUS_ID ordering is required for --gpu to mean what the
    # launcher thinks it means.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # FlashInfer JIT-compiles its sampling kernels at engine start and the ninja
    # build fails on this toolchain. We only ever sample greedily with one token,
    # so the native sampler is equivalent and costs nothing here.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    from mirage.data.loaders import load_benchmark, load_specs
    from mirage.models.generative import score_generative
    from mirage.models.scorer import load_model_grid, score_items

    i, n = (int(x) for x in args.shard.split("/"))
    tiers = tuple(int(t) for t in args.tiers.split(","))

    grid = load_model_grid(MODEL_CFG, tiers=tiers)
    if args.models:
        want = set(args.models.split(","))
        grid = [m for m in grid if m.name in want]
    mine = [m for j, m in enumerate(grid) if j % n == i]

    specs = load_specs(BENCH_CFG)
    if args.benchmarks:
        want = set(args.benchmarks.split(","))
        specs = {k: v for k, v in specs.items() if k in want}
    specs = {k: v for k, v in specs.items() if v.usable}
    # Task and scorer must agree: multiple choice needs options, generative needs
    # a match rule. Mixing them silently produced accuracy 1.000 before the
    # scorer learned to refuse.
    want = "generative" if args.mode == "generative" else "multiple_choice"
    dropped = [k for k, v in specs.items() if v.task != want]
    if dropped:
        print(f"[skip] not scorable in mode={args.mode}: {dropped}")
    specs = {k: v for k, v in specs.items() if v.task == want}

    print(f"[gpu {args.gpu}] shard {i}/{n}: {len(mine)} models x {len(specs)} benchmarks")
    for m in mine:
        print(f"    {m.name:<20} {m.hf_id}")
    if args.dry_run:
        return 0

    args.out.mkdir(parents=True, exist_ok=True)

    # Load benchmarks once; they are shared across every model on this GPU.
    loaded: dict[str, dict] = {}
    for name, spec in specs.items():
        if args.max_items is not None:
            spec.max_items = args.max_items
        langs = args.languages.split(",") if args.languages else None
        try:
            items_by_lang, manifest = load_benchmark(spec, languages=langs)
            loaded[name] = {"items": items_by_lang, "spec": spec, "manifest": manifest}
            n_items = manifest["n_items"]
            print(f"[data] {name}: {len(items_by_lang)} langs x {n_items} items")
        except Exception as exc:  # noqa: BLE001
            print(f"[data] {name}: SKIPPED -- {type(exc).__name__}: {exc}")

    if not loaded:
        print("no usable benchmarks; nothing to do")
        return 1

    from vllm import LLM

    for spec_m in mine:
        todo = [
            (b, lg)
            for b, blob in loaded.items()
            for lg in blob["items"]
            if not shard_path(args.out, b, spec_m.name, lg, args.mode).exists()
        ]
        if not todo:
            print(f"[{spec_m.name}] all shards present, skipping model load")
            continue

        print(f"\n[{spec_m.name}] loading ({len(todo)} shards outstanding)", flush=True)
        t0 = time.time()
        # Other users share these cards, so a fixed fraction of *total* memory
        # intermittently exceeds what is actually free and the engine refuses to
        # start. Size the request against free memory instead, leaving headroom.
        frac = args.gpu_mem_frac
        try:
            import torch

            free_b, total_b = torch.cuda.mem_get_info()
            frac = min(frac, 0.92 * free_b / total_b)
        except Exception:  # noqa: BLE001
            pass
        print(f"[{spec_m.name}] gpu_memory_utilization={frac:.2f}", flush=True)
        try:
            llm = LLM(
                model=spec_m.hf_id,
                revision=spec_m.revision,
                dtype=spec_m.dtype,
                max_model_len=args.max_model_len or spec_m.max_model_len,
                gpu_memory_utilization=frac,
                trust_remote_code=spec_m.trust_remote_code,
                quantization=spec_m.quantization,
                enforce_eager=False,
                disable_log_stats=True,
                max_logprobs=64,
            )
            tok = llm.get_tokenizer()
        except Exception as exc:  # noqa: BLE001
            print(f"[{spec_m.name}] LOAD FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            continue
        print(f"[{spec_m.name}] loaded in {time.time() - t0:.0f}s", flush=True)

        for bench, lang in todo:
            blob = loaded[bench]
            items = blob["items"][lang]
            dest = shard_path(args.out, bench, spec_m.name, lang, args.mode)
            dest.parent.mkdir(parents=True, exist_ok=True)
            t1 = time.time()
            try:
                if args.mode == "generative":
                    res = score_generative(llm, items)
                else:
                    res = score_items(
                        llm, tok, items,
                        mode=args.mode,
                        n_options=blob["spec"].n_options,
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"  {bench}/{lang}: FAILED {type(exc).__name__}: {exc}")
                continue

            np.savez_compressed(
                dest,
                item_ids=np.array(res.item_ids),
                correct=res.correct,
                predicted=res.predicted,
                meta=json.dumps({
                    "model": spec_m.name,
                    "hf_id": spec_m.hf_id,
                    "revision": spec_m.revision,
                    "benchmark": bench,
                    "language": lang,
                    "mode": args.mode,
                    "n_items": len(items),
                    "accuracy": res.accuracy,
                    "unparseable": res.unparseable,
                    "mode_share": res.mode_share,
                    "seconds": round(time.time() - t1, 1),
                }),
            )
            flags = ""
            if res.unparseable > 0.15:
                flags += "  <-- HIGH UNPARSEABLE"
            if res.mode_share == res.mode_share and res.mode_share > 0.90:
                flags += "  <-- DEGENERATE (one option dominates)"
            print(
                f"  {bench}/{lang}: acc={res.accuracy:.3f} "
                f"unparsed={res.unparseable:.3f} mode={res.mode_share:.2f} "
                f"({time.time() - t1:.0f}s){flags}",
                flush=True,
            )

        del llm
        try:
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001, S110
            pass

    print("\nshard complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
