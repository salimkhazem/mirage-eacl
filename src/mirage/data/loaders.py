"""Load parallel multilingual benchmarks into a common item representation.

The one invariant this module exists to protect: **every language must receive
exactly the same set of source items**.  MIRAGE's entire argument rests on the
design being fully crossed -- Theorem 1's invariance group, the cell-wise
estimation of ``delta[i, l]``, and the interpretation of ``G`` all assume item
``i`` is the same question in every language.  A subsample drawn independently
per language would silently destroy that and produce numbers that look fine.

So :func:`select_item_ids` draws the item set **once**, from the sorted
*intersection* of identifiers, using a fixed seed, and every language is then
filtered to that set.  Drawing from the intersection makes it impossible for a
language to lack a selected item; what remains detectable is a benchmark whose
language versions are not the same test at all, which surfaces as a large
unshared fraction and is refused by :func:`load_benchmark`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = ["BenchmarkSpec", "Item", "load_specs", "select_item_ids", "load_benchmark"]

# A parallel benchmark loses essentially nothing to the intersection of item
# sets. Anything above this is evidence the language versions are not the same
# test, which invalidates the crossed design the estimands assume.
_MAX_UNSHARED_FRACTION = 0.01

# Isolated bad rows happen in real corpora and are dropped from every language to
# keep the design crossed. A large fraction instead means the field mapping is
# wrong, which must fail loudly rather than silently shrink the benchmark.
_MAX_MALFORMED_FRACTION = 0.02

# XCOPA stores only "cause"/"effect" in its question column.
_COPA_QUESTION = {
    "cause": "What was the cause of this?",
    "effect": "What happened as a result?",
}


@dataclass(frozen=True)
class Item:
    """One scored unit: a question in one language, with its options."""

    item_id: str
    language: str
    prompt: str
    options: tuple[str, ...]
    answer_index: int
    strata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 <= self.answer_index < len(self.options):
            raise ValueError(
                f"item {self.item_id}/{self.language}: answer_index "
                f"{self.answer_index} out of range for {len(self.options)} options"
            )


@dataclass
class BenchmarkSpec:
    """One benchmark's pinned coordinates and field mapping."""

    name: str
    hf_id: str | None
    revision: str | None
    task: str
    languages: list[str]
    fields: dict[str, Any]
    split: str = "test"
    source_language: str | None = "en"
    n_options: int | None = None
    max_items: int | None = 2000
    subsample_seed: int = 20260803
    item_id_field: str = "__row_index__"
    config_is_language: bool = True
    strata: dict[str, str] = field(default_factory=dict)
    verified: bool = False
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return bool(self.hf_id) and bool(self.languages)


def load_specs(path: str | Path) -> dict[str, BenchmarkSpec]:
    """Parse ``configs/benchmarks.yaml`` into specs, applying the defaults block."""
    raw = yaml.safe_load(Path(path).read_text())
    defaults = raw.get("defaults", {}) or {}
    known = {
        "hf_id",
        "revision",
        "task",
        "languages",
        "fields",
        "split",
        "source_language",
        "n_options",
        "max_items",
        "subsample_seed",
        "item_id_field",
        "config_is_language",
        "strata",
        "verified",
        "notes",
    }
    specs: dict[str, BenchmarkSpec] = {}
    for name, cfg in (raw.get("benchmarks") or {}).items():
        merged = {**defaults, **(cfg or {})}
        kwargs = {k: v for k, v in merged.items() if k in known}
        kwargs.setdefault("fields", {})
        kwargs.setdefault("languages", [])
        kwargs.setdefault("task", "multiple_choice")
        specs[name] = BenchmarkSpec(
            name=name,
            extra={k: v for k, v in merged.items() if k not in known},
            **kwargs,
        )
    return specs


def select_item_ids(
    ids_per_language: dict[str, list[str]],
    max_items: int | None,
    seed: int,
) -> tuple[list[str], dict[str, tuple[int, int]]]:
    """Choose one item set shared by every language.

    Args:
        ids_per_language: identifiers present in each language.
        max_items: cap, or ``None`` to keep every shared item.
        seed: fixed seed, recorded in the manifest.

    Returns:
        ``(selected, coverage)``. ``selected`` is the sorted chosen set, drawn
        from the intersection so it is by construction present in every language.
        ``coverage[lang]`` reports ``(n_total, n_dropped)`` -- how many items that
        language carries and how many are *not* shared with the others.

        Because selection is from the intersection, a language can never be
        missing a selected item; the failure mode that remains is a benchmark
        that is not really parallel, which shows up as a large ``n_dropped``.
        Callers must inspect ``coverage`` -- see :func:`load_benchmark`, which
        refuses to proceed past a small tolerance.

    The draw is over the **intersection** of identifiers, so the result is
    invariant to which languages happen to carry extra items, and the shuffle is
    seeded by a hash of ``(seed, sorted intersection)`` so it depends only on the
    data and the declared seed -- never on dict ordering or load order.
    """
    if not ids_per_language:
        raise ValueError("no languages supplied")

    sets = [set(v) for v in ids_per_language.values()]
    shared = sorted(set.intersection(*sets)) if sets else []
    if not shared:
        raise ValueError(
            "the languages share no common item identifiers; the benchmark is "
            "not parallel under the configured item_id_field"
        )

    if max_items is None or max_items >= len(shared):
        selected = shared
    else:
        # Deterministic order from a content hash, so the draw is reproducible
        # across machines, Python versions, and dataset row ordering.
        def key(item_id: str) -> str:
            h = hashlib.sha256(f"{seed}:{item_id}".encode()).hexdigest()
            return h

        selected = sorted(sorted(shared, key=key)[:max_items])

    shared_set = set(shared)
    coverage = {
        lang: (len(ids), len(set(ids) - shared_set)) for lang, ids in ids_per_language.items()
    }
    return selected, coverage


def _item_ids(ds: Any, field: str | list[str]) -> list[str]:
    """Stable per-row identifiers, from a column, a composite key, or row order.

    A composite key (``["link", "question_number"]``) is strongly preferred where
    one exists. Belebele is the cautionary case: its ``question_number`` column
    takes only two distinct values -- it indexes the question within its passage,
    not the item -- so using it alone silently reduces the benchmark from 900
    items to 2, while every downstream stage keeps working.

    ``__row_index__`` remains available for benchmarks with no identifier at all
    (XNLI), but it *assumes* rather than verifies that every language version
    lists items in the same order. Prefer a real key wherever the data has one.
    """
    if isinstance(field, list):
        cols = [[str(v) for v in ds[c]] for c in field]
        return ["␟".join(parts) for parts in zip(*cols, strict=True)]
    if field == "__row_index__":
        return [str(i) for i in range(len(ds))]
    return [str(v) for v in ds[field]]


def _mcq_prompt(question: str, options: list[str], context: str | None = None) -> str:
    """Letter-choice prompt: one forward pass scores all options.

    Scoring the letter rather than the full option text is what keeps the grid
    affordable (one forward pass per item instead of one per option). The
    alternative -- per-option log-likelihood -- is implemented in
    ``mirage.models.scorer`` and compared in the scoring-method ablation, since
    the two disagree in ways that could otherwise be mistaken for DIF.
    """
    letters = "ABCDEFGHIJ"[: len(options)]
    body = "\n".join(f"{ltr}. {opt}" for ltr, opt in zip(letters, options, strict=True))
    head = f"{context.strip()}\n\n" if context else ""
    return (
        f"{head}{question.strip()}\n{body}\n"
        f"Answer with a single letter from {letters[0]} to {letters[-1]}.\nAnswer:"
    )


def load_benchmark(
    spec: BenchmarkSpec,
    languages: list[str] | None = None,
    cache_dir: str | Path = "data/cache",
) -> tuple[dict[str, list[Item]], dict[str, Any]]:
    """Load one benchmark across languages, subsampled to a shared item set.

    Returns ``(items_by_language, manifest)``. The manifest records the resolved
    revision, the selected item identifiers, the seed, and the per-language
    coverage, and is written next to the response tensors so a run can be
    audited without re-downloading anything.

    Raises:
        RuntimeError: if the benchmark is unusable (unresolved coordinates) or
            any language's unshared fraction exceeds ``_MAX_UNSHARED_FRACTION``.
    """
    if not spec.usable:
        raise RuntimeError(
            f"benchmark {spec.name!r} has unresolved coordinates "
            f"(hf_id={spec.hf_id!r}); run scripts.download --verify-only"
        )
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional GPU-stage dep
        raise RuntimeError(
            "the `datasets` package is required for benchmark loading; "
            "install the GPU extra: uv pip install -e '.[gpu]'"
        ) from exc

    langs = languages or spec.languages
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    raw_by_lang: dict[str, Any] = {}
    ids_by_lang: dict[str, list[str]] = {}
    for lang in langs:
        ds = load_dataset(
            spec.hf_id,
            lang if spec.config_is_language else None,
            split=spec.split,
            revision=spec.revision,
            cache_dir=str(cache),
        )
        raw_by_lang[lang] = ds
        ids_by_lang[lang] = _item_ids(ds, spec.item_id_field)

    # Drop items malformed in *any* language, from *every* language.
    #
    # Global-MMLU has 2 Spanish rows (of 14042) whose first option is empty. The
    # crossed design requires item i to be the same question everywhere, so an
    # item we cannot score in one language must not be scored in the others
    # either. Excluding the two items costs 0.014% of the benchmark; excluding
    # Spanish would cost a whole language for no reason.
    malformed: set[str] = set()
    if spec.task == "multiple_choice":
        for lang, ds in raw_by_lang.items():
            for idx, row in enumerate(ds):
                try:
                    _extract_options(spec.fields, row)
                except ValueError:
                    malformed.add(ids_by_lang[lang][idx])
    if malformed:
        n_rows = max(len(v) for v in ids_by_lang.values())
        frac = len(malformed) / max(n_rows, 1)
        if frac > _MAX_MALFORMED_FRACTION:
            raise RuntimeError(
                f"benchmark {spec.name!r}: {len(malformed)} items ({frac:.2%}) are "
                f"malformed in at least one language, above the "
                f"{_MAX_MALFORMED_FRACTION:.1%} tolerance. That looks like a field "
                f"mapping error, not isolated bad rows."
            )
    # Filter for *selection* only. ``ids_by_lang`` stays row-aligned with the
    # datasets, because the row index is how items are fetched below.
    ids_for_selection = (
        {lang: [i for i in ids if i not in malformed] for lang, ids in ids_by_lang.items()}
        if malformed
        else ids_by_lang
    )

    selected, coverage = select_item_ids(
        ids_for_selection, spec.max_items, spec.subsample_seed
    )
    # A genuinely parallel benchmark loses almost nothing to the intersection.
    # Large losses mean the language versions do not share an item set, so the
    # design is not crossed and every estimand in the paper is ill-defined.
    offenders = {
        lang: (total, dropped)
        for lang, (total, dropped) in coverage.items()
        if total and dropped / total > _MAX_UNSHARED_FRACTION
    }
    if offenders:
        raise RuntimeError(
            f"benchmark {spec.name!r} is not parallel: "
            + ", ".join(
                f"{lang} drops {d}/{t} ({d / t:.1%}) unshared items"
                for lang, (t, d) in sorted(offenders.items())
            )
            + f". Tolerance is {_MAX_UNSHARED_FRACTION:.1%}. Refusing to proceed -- "
            "MIRAGE assumes every model answers the same item in every language."
        )

    # Emit items in the canonical ``selected`` order, identical in every language.
    #
    # Iterating each language in its own dataset row order looks equivalent and is
    # not: Belebele lists the same (link, question_number) items in different row
    # orders across variants, so position k would be a different question per
    # language. The response tensor would then align English item k with Bengali
    # item k' -- destroying the crossed design while still producing plausible
    # accuracies and entirely fictitious drift estimates.
    items_by_lang: dict[str, list[Item]] = {}
    for lang, ds in raw_by_lang.items():
        row_of = {iid: idx for idx, iid in enumerate(ids_by_lang[lang])}
        keep = [row_of[iid] for iid in selected]
        items_by_lang[lang] = _build_items(spec, ds.select(keep), selected, lang)
        got = [it.item_id for it in items_by_lang[lang]]
        if got != selected:
            first = next(
                i
                for i, (a, b) in enumerate(zip(got, selected, strict=True))
                if a != b
            )
            raise RuntimeError(
                f"{spec.name}/{lang}: item order does not match the canonical "
                f"selection after reindexing (first mismatch at index {first})"
            )

    manifest = {
        "benchmark": spec.name,
        "hf_id": spec.hf_id,
        "revision": spec.revision,
        "split": spec.split,
        "task": spec.task,
        "languages": langs,
        "source_language": spec.source_language,
        "n_items": len(selected),
        "n_malformed_excluded": len(malformed),
        "malformed_item_ids": sorted(malformed),
        "max_items": spec.max_items,
        "subsample_seed": spec.subsample_seed,
        "item_ids": selected,
        "coverage": {k: {"n_total": t, "n_unshared": d} for k, (t, d) in coverage.items()},
    }
    (cache / f"manifest_{spec.name}.json").write_text(json.dumps(manifest, indent=2))
    return items_by_lang, manifest


def _build_items(spec: BenchmarkSpec, rows: Any, ids: list[str], lang: str) -> list[Item]:
    """Map raw dataset rows onto :class:`Item` using the spec's field mapping."""
    f = spec.fields
    out: list[Item] = []
    for idx, row in enumerate(rows):
        if spec.task == "generative":
            out.append(
                Item(
                    item_id=ids[idx],
                    language=lang,
                    prompt=str(row[f["question"]]).strip(),
                    options=(str(row[f["answer"]]),),
                    answer_index=0,
                )
            )
            continue

        options = _extract_options(f, row)
        question, context = _extract_question(f, row)
        answer = _normalise_answer(f, row, len(options))
        strata = {k: str(row.get(v, "")) for k, v in (spec.strata or {}).items()}
        out.append(
            Item(
                item_id=ids[idx],
                language=lang,
                prompt=_mcq_prompt(question, options, context),
                options=tuple(options),
                answer_index=answer,
                strata=strata,
            )
        )
    return out


def _strip_trailing_nulls(opts: list[Any]) -> list[str]:
    """Drop unused trailing option slots, refusing to drop interior ones.

    MMLU-ProX stores ten fixed columns ``option_0..option_9`` and leaves the
    unused ones ``None``, so items genuinely vary in option count. Trailing
    nulls are safe to remove because they do not shift any index. An *interior*
    null would shift every later option down by one while ``answer_index`` kept
    pointing at the original slot -- silently relabelling the correct answer --
    so that case raises instead.
    """
    n = len(opts)
    while n > 0 and (opts[n - 1] is None or str(opts[n - 1]).strip() == ""):
        n -= 1
    head = opts[:n]
    holes = [i for i, o in enumerate(head) if o is None or str(o).strip() == ""]
    if holes:
        raise ValueError(
            f"option slots {holes} are empty but later slots are filled; "
            f"dropping them would shift answer_index and mislabel the answer"
        )
    return [str(o) for o in head]


def _extract_options(f: dict[str, Any], row: Any) -> list[str]:
    spec_opts = f.get("options")
    if isinstance(spec_opts, str):  # a single list-valued column
        return _strip_trailing_nulls(list(row[spec_opts]))
    if isinstance(spec_opts, list):
        return _strip_trailing_nulls([row[c] for c in spec_opts])
    if "hypothesis" in f:
        # "True / Neither / False" rather than the label names. The label-name
        # form, paired with a yes/no question stem, collapsed Llama-3.2-1B onto
        # option A for 1992 of 2000 items -- accuracy pinned at the base rate of
        # the entailment class in every language. A benchmark on which every
        # model is degenerate carries no information about item difficulty, so it
        # would contribute pure noise to the response tensor.
        return ["True", "Neither", "False"]
    raise ValueError(f"cannot determine options from field mapping {f!r}")


def _extract_question(f: dict[str, Any], row: Any) -> tuple[str, str | None]:
    if "hypothesis" in f:
        # The lm-eval-harness XNLI/MNLI zero-shot stem. Keeping to the community
        # format also keeps our per-language accuracies comparable to published
        # numbers, which the paper re-analyses.
        return (
            f"{row[f['premise']]}\nQuestion: {row[f['hypothesis']]} "
            f"True, False, or Neither?",
            None,
        )
    ctx = f.get("context")
    if isinstance(ctx, list):
        context = " ".join(str(row[c]) for c in ctx)
    elif isinstance(ctx, str):
        context = str(row[ctx])
    else:
        context = None

    question = str(row[f["question"]]) if "question" in f else ""

    if "premise" in f:
        # XCOPA-style: the premise is the context and the "question" column holds
        # only the word "cause" or "effect". Without this branch the premise was
        # dropped entirely and the prompt read just "cause\nA. ...\nB. ...", which
        # is unanswerable -- the whole 18-model panel scored 54% on a two-choice
        # task, i.e. chance, and the resulting IRT fit was pure noise.
        context = str(row[f["premise"]])
        question = _COPA_QUESTION.get(question.strip().lower(), question)

    return question, context


def _normalise_answer(f: dict[str, Any], row: Any, n_options: int) -> int:
    """Coerce a benchmark's answer encoding to a 0-based index.

    The base is read from the config (``answer_base``), never inferred. Belebele
    is 1-indexed and XNLI is 0-indexed, and both store small integers, so any
    value-based guess is wrong for one of them -- silently, and in a way that
    would corrupt every response in the tensor while still producing plausible
    accuracies. Letter encodings (Global-MMLU) are detected explicitly because
    they are unambiguous.
    """
    raw = row[f["answer"]]
    base = f.get("answer_base")

    if isinstance(raw, str):
        s = raw.strip()
        if len(s) == 1 and s.upper() in "ABCDEFGHIJ":
            idx = "ABCDEFGHIJ".index(s.upper())
            if not 0 <= idx < n_options:
                raise ValueError(f"letter answer {s!r} out of range for {n_options} options")
            return idx
        raw = int(s)

    if base is None:
        raise ValueError(
            f"field mapping {f!r} stores a numeric answer but declares no "
            f"`answer_base`; set it to 0 or 1 in configs/benchmarks.yaml rather "
            f"than letting the loader guess"
        )
    if base not in (0, 1):
        raise ValueError(f"answer_base must be 0 or 1, got {base!r}")

    idx = int(raw) - base
    if not 0 <= idx < n_options:
        raise ValueError(
            f"answer {raw!r} with answer_base={base} gives index {idx}, "
            f"out of range for {n_options} options"
        )
    return idx
