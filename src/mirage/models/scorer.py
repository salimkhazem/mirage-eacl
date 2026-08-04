"""Score benchmark items with a language model to build the response tensor.

Two scoring modes, both single-pass:

``letter``
    Present the options as ``A.``/``B.``/... and compare the next-token logits
    of the letter tokens. One forward pass per item; this is what makes the
    nine-benchmark grid affordable.

``loglik``
    Score each option's continuation log-likelihood, length-normalised. Costs
    one pass per option, so it is used only for the scoring-method ablation.

The ablation matters more than it sounds. Letter scoring is sensitive to how
well a model follows the answer format, and format-following degrades in
lower-resource languages. That degradation would appear in the response tensor
as **language-specific item difficulty** -- i.e. it would masquerade as exactly
the translation drift MIRAGE sets out to measure. Running both modes on a
subset and showing the drift estimates agree is what separates a real finding
from a scoring artefact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

if TYPE_CHECKING:
    from mirage.data.loaders import Item

__all__ = ["ScoreResult", "ModelSpec", "load_model_grid", "score_items"]

_LETTERS = "ABCDEFGHIJ"


@dataclass
class ModelSpec:
    name: str
    hf_id: str
    revision: str = "main"
    tier: int = 2
    params_b: float = 0.0
    dtype: str = "bfloat16"
    batch_size: int = 32
    max_model_len: int = 2048
    quantization: str | None = None
    trust_remote_code: bool = False
    gated: bool = False
    cached: bool = False


@dataclass
class ScoreResult:
    """Per-item outcomes for one (model, benchmark, language) shard."""

    item_ids: list[str]
    correct: np.ndarray  # (n_items,) uint8
    predicted: np.ndarray  # (n_items,) int16, -1 if unparseable
    mode: str

    @property
    def accuracy(self) -> float:
        return float(self.correct.mean()) if self.correct.size else float("nan")

    @property
    def unparseable(self) -> float:
        return float((self.predicted < 0).mean()) if self.predicted.size else 0.0

    @property
    def mode_share(self) -> float:
        """Fraction of items assigned the single most frequent option.

        Defined only for multiple-choice scoring. Under generative scoring
        ``predicted`` holds correctness (1/0) rather than an option index, so the
        statistic would degenerate into ``max(accuracy, 1 - accuracy)``: an MGSM
        shard scoring 0.02 would report 0.98 and be flagged as collapsed when it
        is merely difficult. Since the degeneracy filter drops shards above 0.90,
        that false positive would remove exactly the lowest-accuracy languages --
        selection on the outcome. Returns NaN for generative shards instead.

        A degeneracy detector. When a model answers "A" to nearly everything, its
        accuracy sits at the base rate of the first class and looks like a
        plausible weak score, while the shard actually carries no information
        about item difficulty. Such a shard contributes noise to the response
        tensor rather than signal, so it must be visible, not averaged in.
        """
        if self.mode == "generative":
            return float("nan")
        valid = self.predicted[self.predicted >= 0]
        if valid.size == 0:
            return 1.0
        return float(np.bincount(valid).max() / valid.size)


def load_model_grid(path: str | Path, tiers: tuple[int, ...] = (1, 2)) -> list[ModelSpec]:
    """Parse ``configs/models.yaml``, keeping the requested tiers in order."""
    raw = yaml.safe_load(Path(path).read_text())
    defaults = raw.get("defaults", {}) or {}
    known = {
        "name", "hf_id", "revision", "tier", "params_b", "dtype", "batch_size",
        "max_model_len", "quantization", "trust_remote_code", "gated", "cached",
    }
    out = []
    for entry in raw.get("models", []):
        merged = {**defaults, **entry}
        spec = ModelSpec(**{k: v for k, v in merged.items() if k in known})
        if spec.tier in tiers:
            out.append(spec)
    # Smallest first: a format or prompt bug surfaces in seconds, not hours.
    return sorted(out, key=lambda s: (s.tier, s.params_b))


def _letter_token_ids(tokenizer: Any, n_options: int) -> list[list[int]]:
    """Candidate token ids for each answer letter.

    Tokenizers differ on whether "A" after a newline is ``"A"`` or ``"▁A"``, so
    every plausible surface form is collected and the max logit over the group
    is used. Getting this wrong silently biases towards whichever option the
    tokenizer happens to encode canonically -- and would do so *differently per
    language*, which is precisely the confound the paper is about.
    """
    groups = []
    for i in range(n_options):
        ltr = _LETTERS[i]
        ids = set()
        for form in (ltr, f" {ltr}", f"\n{ltr}", f"▁{ltr}"):
            try:
                toks = tokenizer.encode(form, add_special_tokens=False)
            except Exception:  # noqa: BLE001 - tokenizer quirks vary wildly
                continue
            if toks:
                ids.add(toks[-1] if form.startswith(("\n", " ", "▁")) else toks[0])
        if not ids:
            raise RuntimeError(f"tokenizer produced no token for letter {ltr!r}")
        groups.append(sorted(ids))
    return groups


def score_items(
    llm: Any,
    tokenizer: Any,
    items: list[Item],
    *,
    mode: str = "letter",
    n_options: int | None = None,
) -> ScoreResult:
    """Score one shard of items with an already-loaded vLLM engine.

    Args:
        llm: a ``vllm.LLM`` instance.
        tokenizer: the matching tokenizer.
        items: items for a single (benchmark, language) shard.
        mode: ``"letter"`` or ``"loglik"``.
        n_options: option count; inferred from the items when omitted.

    Returns:
        A :class:`ScoreResult` aligned with ``items``.
    """
    if not items:
        return ScoreResult([], np.zeros(0, np.uint8), np.zeros(0, np.int16), mode)
    if mode not in {"letter", "loglik"}:
        raise ValueError(f"mode must be 'letter' or 'loglik', got {mode!r}")

    k = n_options or max(len(it.options) for it in items)
    if k < 2:
        # A generative benchmark reaching the multiple-choice path would carry a
        # single "option" (the gold answer), so argmax would select it every time
        # and the shard would record a perfect score. Fail instead of writing
        # accuracy 1.0 that looks like a result.
        raise ValueError(
            f"multiple-choice scoring needs at least 2 options, got {k}. This "
            f"benchmark is probably generative and needs a match-based scorer, "
            f"not mode={mode!r}."
        )
    if mode == "letter":
        predicted = _score_letter(llm, tokenizer, items, k)
    else:
        predicted = _score_loglik(llm, tokenizer, items)

    gold = np.array([it.answer_index for it in items], dtype=np.int16)
    correct = (predicted == gold).astype(np.uint8)
    return ScoreResult(
        item_ids=[it.item_id for it in items],
        correct=correct,
        predicted=predicted,
        mode=mode,
    )


def _score_letter(llm: Any, tokenizer: Any, items: list[Item], k: int) -> np.ndarray:
    from vllm import SamplingParams

    groups = _letter_token_ids(tokenizer, k)
    flat = sorted({t for g in groups for t in g})
    # Restrict the sampler to the answer letters rather than reading a top-k list.
    #
    # This is a correctness requirement, not an optimisation. With plain top-k,
    # an item counts as unparseable whenever no letter token makes the model's
    # top-k -- which happens more often for weaker models and for lower-resource
    # languages. That missingness is language-correlated, so it would enter the
    # response tensor as language-specific item difficulty and be indistinguishable
    # from the translation drift this paper measures. Constraining the support
    # guarantees every item yields a comparison among the same K options in every
    # language.
    params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=len(flat),
        allowed_token_ids=flat,
    )
    outs = llm.generate([it.prompt for it in items], params, use_tqdm=False)

    pred = np.full(len(items), -1, dtype=np.int16)
    for i, out in enumerate(outs):
        lp = out.outputs[0].logprobs
        if not lp:
            continue
        table = lp[0]
        scores = []
        for g in groups:
            vals = [table[t].logprob for t in g if t in table]
            scores.append(max(vals) if vals else -np.inf)
        n_valid = len(items[i].options)
        scores = scores[:n_valid]
        if all(np.isneginf(s) for s in scores):
            continue  # unparseable: recorded as -1, never guessed
        pred[i] = int(np.argmax(scores))
    return pred


def _score_loglik(llm: Any, tokenizer: Any, items: list[Item]) -> np.ndarray:
    """Length-normalised log-likelihood of the *option continuation* only.

    The scored span matters more than it looks. Averaging log-probabilities over
    the whole prompt puts a ~100-token shared stem in the average alongside a
    handful of option tokens, so the option contributes about one part in a
    hundred and the comparison is swamped: an earlier version of this function
    did exactly that and scored 0.648 on English XStoryCloze against 0.858 for
    letter scoring, i.e. barely above the 0.5 chance level on a two-choice task.
    Only the continuation tokens are averaged here.
    """
    from vllm import SamplingParams

    prompts, owner, n_stem = [], [], []
    for i, it in enumerate(items):
        stem = it.prompt.rsplit("\n", 1)[0] + "\nAnswer:"
        stem_len = len(tokenizer.encode(stem, add_special_tokens=True))
        for opt in it.options:
            prompts.append(f"{stem} {opt}")
            owner.append(i)
            n_stem.append(stem_len)

    params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    outs = llm.generate(prompts, params, use_tqdm=False)

    per_item: dict[int, list[float]] = {}
    for idx, out in enumerate(outs):
        plp = out.prompt_logprobs or []
        vals = [
            next(iter(d.values())).logprob
            for d in plp[n_stem[idx] :]
            if isinstance(d, dict) and d
        ]
        # Normalise by continuation length so long options are not penalised,
        # which would otherwise correlate with language (translations differ in
        # length systematically).
        score = float(np.mean(vals)) if vals else -np.inf
        per_item.setdefault(owner[idx], []).append(score)

    pred = np.full(len(items), -1, dtype=np.int16)
    for i, scores in per_item.items():
        if scores and not all(np.isneginf(s) for s in scores):
            pred[i] = int(np.argmax(scores))
    return pred
