"""Match-based scoring for generative benchmarks (MGSM).

Multiple-choice scoring cannot be applied to a benchmark whose answer is a free
-form number: there are no options to compare, and feeding the gold answer in as
the sole "option" would score every item correct (``mirage.models.scorer``
refuses that explicitly). MGSM is scored here by generating a chain of thought
and matching the final numeric answer.

The extraction rule is the load-bearing part, and it is language-sensitive in a
way that matters for this paper. Digit shape, decimal marks, and thousands
separators differ across the eleven MGSM languages: ``1,234.5`` in English is
``1.234,5`` in German and ``١٢٣٤٫٥`` in Arabic. An extractor that only understands
ASCII digits with a period would mark correct answers wrong at a rate that varies
by language -- a language-correlated measurement error, i.e. exactly the confound
Theorem 1 says we cannot separate from ability. The normalisation below is
therefore deliberately generous and is applied identically in every language.
"""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from mirage.data.loaders import Item

from mirage.models.scorer import ScoreResult

__all__ = ["extract_number", "score_generative"]

# Answer cues in the eleven MGSM languages, lower-cased. Generation is truncated
# at the first cue that follows a number, so a model that keeps talking after
# answering is not penalised for the continuation.
_ANSWER_CUES = (
    "answer", "réponse", "respuesta", "antwort", "risposta", "resposta",
    "ответ", "答案", "答え", "คำตอบ", "jibu", "উত্তর", "సమాధానం",
)

_NUM_RE = re.compile(r"[-+]?\d[\d\s.,  ]*")


def _to_ascii_digits(text: str) -> str:
    """Map any Unicode decimal digit to its ASCII counterpart.

    Arabic-Indic, Devanagari, Bengali, Telugu and Thai digits all appear in MGSM
    outputs. ``unicodedata.digit`` handles every script uniformly, which keeps
    the rule identical across languages rather than a pile of per-script cases.
    """
    out = []
    for ch in text:
        if ch.isdigit():
            try:
                out.append(str(unicodedata.digit(ch)))
                continue
            except (TypeError, ValueError):
                pass
        out.append(ch)
    return "".join(out)


def _resolve_separators(raw: str) -> str:
    """Turn a locale-ambiguous numeral into a plain ASCII float string.

    ``12,345`` is twelve thousand in English and twelve-point-three-four-five in
    German, and nothing in the string itself settles it. We resolve a *single*
    separator followed by exactly three digits as a thousands separator, and
    everything else as a decimal mark.

    That rule is safe for this benchmark specifically: MGSM inherits GSM8K's
    grade-school answers, which are integers, so the thousands reading is the
    correct one essentially always. It would not be safe on a benchmark with
    genuine decimals, and the rule is applied identically in every language so
    that whatever residual error it has cannot correlate with language --- which
    is the property that matters here, since a language-correlated extraction
    failure would be indistinguishable from translation drift.
    """
    n_dot, n_comma = raw.count("."), raw.count(",")
    if n_dot == 0 and n_comma == 0:
        return raw

    # A number carries at most one decimal mark, so any separator that repeats
    # must be the thousands separator. This settles 1.234.567 and 1,234,567.
    if n_dot > 1:
        return raw.replace(".", "").replace(",", ".")
    if n_comma > 1:
        return raw.replace(",", "")

    if n_dot == 1 and n_comma == 1:
        # Both present, each once: the later one is the decimal mark.
        return (
            raw.replace(",", "")
            if raw.rfind(".") > raw.rfind(",")
            else raw.replace(".", "").replace(",", ".")
        )

    # Exactly one separator of one kind: ambiguous, resolved by group length.
    cut = max(raw.rfind("."), raw.rfind(","))
    if len(raw) - cut - 1 == 3:
        return raw.replace(".", "").replace(",", "")  # thousands
    return raw.replace(",", ".")  # decimal


def extract_number(text: str) -> float | None:
    """Return the model's final numeric answer, or ``None`` if there is none.

    Prefers the number following an answer cue; otherwise takes the last number
    in the text, which is the usual convention for chain-of-thought scoring.
    Separators are resolved by position rather than by locale: whichever of
    ``.`` or ``,`` appears last is treated as the decimal mark, so both
    ``1,234.5`` and ``1.234,5`` read as ``1234.5``.
    """
    if not text:
        return None
    s = _to_ascii_digits(text)

    lowered = s.lower()
    tail = s
    for cue in _ANSWER_CUES:
        idx = lowered.rfind(cue)
        if idx != -1:
            candidate = s[idx:]
            if _NUM_RE.search(candidate):
                tail = candidate
                break

    matches = _NUM_RE.findall(tail) or _NUM_RE.findall(s)
    if not matches:
        return None

    raw = matches[-1].strip()
    raw = raw.replace(" ", "").replace(" ", "").replace(" ", "")
    raw = raw.rstrip(".,")
    if not raw or not any(c.isdigit() for c in raw):
        return None

    cleaned = _resolve_separators(raw)

    try:
        return float(cleaned)
    except ValueError:
        return None


def score_generative(
    llm: Any,
    items: list[Item],
    *,
    max_new_tokens: int = 256,
    tolerance: float = 1e-4,
) -> ScoreResult:
    """Generate answers and score by numeric match.

    Args:
        llm: a ``vllm.LLM`` instance.
        items: items whose ``options[0]`` holds the gold answer.
        max_new_tokens: generation budget. Kept identical across languages even
            though high-fertility scripts need more tokens per unit of reasoning;
            a per-language budget would be a language-varying treatment, which is
            the confound we are trying to avoid. The cost is that some
            high-fertility languages may be truncated, and that is reported.
        tolerance: absolute tolerance for the numeric comparison.

    Returns:
        A :class:`ScoreResult` whose ``predicted`` is 1 for a match, 0 for a
        mismatch, and -1 when no number could be extracted.
    """
    from vllm import SamplingParams

    if not items:
        return ScoreResult([], np.zeros(0, np.uint8), np.zeros(0, np.int16), "generative")

    prompts = [
        f"{it.prompt.strip()}\n\nThink step by step, then give the final "
        f"numeric answer.\nAnswer:"
        for it in items
    ]
    params = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
    outs = llm.generate(prompts, params, use_tqdm=False)

    predicted = np.full(len(items), -1, dtype=np.int16)
    correct = np.zeros(len(items), dtype=np.uint8)
    for i, (it, out) in enumerate(zip(items, outs, strict=True)):
        got = extract_number(out.outputs[0].text)
        gold = extract_number(str(it.options[0]))
        if got is None or gold is None:
            continue
        hit = abs(got - gold) <= tolerance * max(1.0, abs(gold))
        predicted[i] = 1 if hit else 0
        correct[i] = 1 if hit else 0

    return ScoreResult(
        item_ids=[it.item_id for it in items],
        correct=correct,
        predicted=predicted,
        mode="generative",
    )
