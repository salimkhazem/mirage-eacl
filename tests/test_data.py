"""Tests for benchmark loading, subsampling, and answer normalisation.

These guard the two places where a silent error would corrupt the response
tensor without producing any visible symptom: the shared-item-set invariant
that keeps the design crossed, and the 0-vs-1-indexed answer encodings.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mirage.data.loaders import (
    BenchmarkSpec,
    Item,
    _normalise_answer,
    load_specs,
    select_item_ids,
)

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "benchmarks.yaml"


# --------------------------------------------------------------------------
# The crossed-design invariant
# --------------------------------------------------------------------------


def test_selection_is_identical_across_languages():
    ids = {"en": ["a", "b", "c", "d"], "fr": ["a", "b", "c", "d"]}
    sel, coverage = select_item_ids(ids, max_items=2, seed=7)
    assert len(sel) == 2
    assert all(unshared == 0 for _, unshared in coverage.values())
    # Same set for every language, by construction.
    for lang in ids:
        assert set(sel) <= set(ids[lang])


def test_selection_is_invariant_to_language_order():
    a = {"en": ["x", "y", "z"], "sw": ["x", "y", "z"], "ja": ["x", "y", "z"]}
    b = {"ja": ["z", "y", "x"], "en": ["y", "x", "z"], "sw": ["x", "z", "y"]}
    assert select_item_ids(a, 2, 7)[0] == select_item_ids(b, 2, 7)[0]


def test_selection_is_deterministic_and_seed_sensitive():
    ids = {"en": [str(i) for i in range(100)], "fr": [str(i) for i in range(100)]}
    assert select_item_ids(ids, 10, seed=1)[0] == select_item_ids(ids, 10, seed=1)[0]
    assert select_item_ids(ids, 10, seed=1)[0] != select_item_ids(ids, 10, seed=2)[0]


def test_selection_uses_the_intersection():
    ids = {"en": ["a", "b", "c"], "sw": ["a", "b"]}
    sel, coverage = select_item_ids(ids, max_items=None, seed=7)
    assert sel == ["a", "b"]  # 'c' is not shared, so it is never selected
    assert coverage["en"] == (3, 1)  # en carries 3 items, 1 of them unshared
    assert coverage["sw"] == (2, 0)


def test_selected_items_are_present_in_every_language_by_construction():
    """Drawing from the intersection makes a per-language gap impossible."""
    ids = {"en": list("abcdefgh"), "fr": list("abcdef"), "sw": list("abcdz")}
    sel, _ = select_item_ids(ids, max_items=3, seed=7)
    for lang, lang_ids in ids.items():
        assert set(sel) <= set(lang_ids), lang


def test_coverage_flags_a_non_parallel_benchmark():
    """A language sharing little with the rest is what the loader must refuse."""
    ids = {"en": [str(i) for i in range(100)],
           "fr": [str(i) for i in range(100)],
           "xx": [str(i) for i in range(90, 190)]}  # only 10 shared
    _, coverage = select_item_ids(ids, max_items=None, seed=7)
    total, unshared = coverage["en"]
    assert unshared / total > 0.01  # exceeds the loader's tolerance
    assert coverage["xx"][1] / coverage["xx"][0] > 0.01


def test_no_shared_items_raises():
    with pytest.raises(ValueError, match="not parallel"):
        select_item_ids({"en": ["a"], "fr": ["b"]}, None, 7)


def test_empty_input_raises():
    with pytest.raises(ValueError, match="no languages"):
        select_item_ids({}, None, 7)


def test_max_items_none_keeps_everything():
    ids = {"en": list("abcde"), "fr": list("abcde")}
    sel, _ = select_item_ids(ids, max_items=None, seed=7)
    assert sel == list("abcde")


def test_max_items_larger_than_pool_is_not_an_error():
    ids = {"en": list("abc"), "fr": list("abc")}
    sel, _ = select_item_ids(ids, max_items=999, seed=7)
    assert sel == list("abc")


# --------------------------------------------------------------------------
# Answer encoding -- the silent-corruption hazard
# --------------------------------------------------------------------------


def test_one_indexed_answers_belebele_style():
    f = {"answer": "correct_answer_num", "answer_base": 1}
    assert _normalise_answer(f, {"correct_answer_num": "1"}, 4) == 0
    assert _normalise_answer(f, {"correct_answer_num": "4"}, 4) == 3


def test_zero_indexed_answers_xnli_style():
    f = {"answer": "label", "answer_base": 0}
    assert _normalise_answer(f, {"label": 0}, 3) == 0
    assert _normalise_answer(f, {"label": 2}, 3) == 2


def test_the_two_encodings_disagree_on_the_same_stored_value():
    """The whole reason answer_base is explicit: '1' means different things."""
    one = _normalise_answer({"answer": "a", "answer_base": 1}, {"a": 1}, 4)
    zero = _normalise_answer({"answer": "a", "answer_base": 0}, {"a": 1}, 4)
    assert one == 0 and zero == 1


def test_letter_answers_global_mmlu_style():
    f = {"answer": "answer"}
    assert _normalise_answer(f, {"answer": "A"}, 4) == 0
    assert _normalise_answer(f, {"answer": "d"}, 4) == 3


def test_missing_answer_base_raises_rather_than_guessing():
    with pytest.raises(ValueError, match="answer_base"):
        _normalise_answer({"answer": "label"}, {"label": 1}, 3)


def test_out_of_range_answers_raise():
    f = {"answer": "a", "answer_base": 1}
    with pytest.raises(ValueError, match="out of range"):
        _normalise_answer(f, {"a": 5}, 4)
    with pytest.raises(ValueError, match="out of range"):
        _normalise_answer({"answer": "a"}, {"a": "E"}, 4)


def test_bad_answer_base_raises():
    with pytest.raises(ValueError, match="answer_base must be"):
        _normalise_answer({"answer": "a", "answer_base": 2}, {"a": 1}, 4)


# --------------------------------------------------------------------------
# Config integrity
# --------------------------------------------------------------------------


def test_all_benchmark_specs_parse():
    specs = load_specs(CONFIG)
    assert len(specs) == 9
    assert "belebele" in specs and "mmlu_prox" in specs


def test_defaults_are_applied_and_overridable():
    specs = load_specs(CONFIG)
    assert specs["belebele"].max_items is None  # explicit override
    assert specs["mmlu_prox"].max_items == 2000
    assert specs["xstorycloze"].split == "eval"  # override
    assert specs["belebele"].split == "test"  # from defaults
    assert specs["belebele"].subsample_seed == 20260803  # from defaults


def test_every_numeric_answer_benchmark_declares_a_base():
    """No benchmark may reach the loader with an ambiguous answer encoding."""
    letter_answer = {"global_mmlu", "global_mmlu_lite"}
    for name, spec in load_specs(CONFIG).items():
        if spec.task != "multiple_choice" or not spec.usable:
            continue
        if name in letter_answer:
            continue
        assert "answer_base" in spec.fields, f"{name} lacks answer_base"
        assert spec.fields["answer_base"] in (0, 1), name


def test_unresolved_benchmarks_are_flagged_unusable():
    specs = load_specs(CONFIG)
    assert not specs["global_piqa"].usable  # hf_id is null
    assert specs["belebele"].usable


def test_source_language_is_present_in_the_language_list():
    for name, spec in load_specs(CONFIG).items():
        if not spec.usable or spec.source_language is None:
            continue
        assert spec.source_language in spec.languages, (
            f"{name}: source language {spec.source_language!r} missing from the "
            f"language list, so delta = 0 would have no anchor"
        )


# --------------------------------------------------------------------------
# Item validation
# --------------------------------------------------------------------------


def test_item_rejects_out_of_range_answer():
    with pytest.raises(ValueError, match="out of range"):
        Item(item_id="1", language="en", prompt="q", options=("a", "b"), answer_index=2)


def test_item_accepts_valid_answer():
    it = Item(item_id="1", language="en", prompt="q", options=("a", "b"), answer_index=1)
    assert it.answer_index == 1


def test_spec_usable_requires_both_id_and_languages():
    assert not BenchmarkSpec(
        name="x", hf_id=None, revision=None, task="multiple_choice",
        languages=["en"], fields={},
    ).usable
    assert not BenchmarkSpec(
        name="x", hf_id="a/b", revision="main", task="multiple_choice",
        languages=[], fields={},
    ).usable


# --------------------------------------------------------------------------
# Variable option counts (MMLU-ProX stores ten fixed slots, padded with None)
# --------------------------------------------------------------------------


def test_trailing_empty_option_slots_are_dropped():
    from mirage.data.loaders import _strip_trailing_nulls

    assert _strip_trailing_nulls(["a", "b", "c", None, None]) == ["a", "b", "c"]
    assert _strip_trailing_nulls(["a", "b", "", "  "]) == ["a", "b"]
    assert _strip_trailing_nulls(["a", "b"]) == ["a", "b"]


def test_interior_empty_option_slot_raises():
    """Dropping an interior hole would shift answer_index and mislabel the answer."""
    from mirage.data.loaders import _strip_trailing_nulls

    with pytest.raises(ValueError, match="shift answer_index"):
        _strip_trailing_nulls(["a", None, "c", None])


def test_composite_item_ids_are_unique_where_single_columns_are_not():
    """The Belebele failure mode, reproduced in miniature."""
    from mirage.data.loaders import _item_ids

    class FakeDS:
        def __init__(self, cols):
            self._c = cols

        def __len__(self):
            return len(next(iter(self._c.values())))

        def __getitem__(self, k):
            return self._c[k]

    ds = FakeDS({"link": ["p1", "p1", "p2", "p2"], "question_number": [1, 2, 1, 2]})
    assert len(set(_item_ids(ds, "question_number"))) == 2  # collapses
    assert len(set(_item_ids(ds, ["link", "question_number"]))) == 4  # unique
    assert len(set(_item_ids(ds, "__row_index__"))) == 4


def test_single_option_scoring_is_refused():
    """A generative benchmark must not silently score 1.0 via the MCQ path."""
    from mirage.models.scorer import score_items

    items = [Item(item_id="1", language="en", prompt="q", options=("42",), answer_index=0)]
    with pytest.raises(ValueError, match="at least 2 options"):
        score_items(None, None, items, mode="letter")


# --------------------------------------------------------------------------
# Canonical item ordering across languages
# --------------------------------------------------------------------------


def test_items_are_emitted_in_canonical_order_not_row_order():
    """Languages listing the same items in different row orders must align.

    This reproduces the Belebele failure: identical item sets, different row
    orders. Selecting by row order would pair English item k with a different
    question in the other language.
    """
    from mirage.data.loaders import select_item_ids

    en_rows = ["i3", "i1", "i2"]  # dataset row order in English
    bn_rows = ["i1", "i2", "i3"]  # different row order, same items
    selected, _ = select_item_ids({"en": en_rows, "bn": bn_rows}, None, seed=7)

    # The canonical selection is sorted and shared.
    assert selected == ["i1", "i2", "i3"]

    # Reindexing by identifier gives the same sequence in both languages...
    en_pick = [en_rows.index(i) for i in selected]
    bn_pick = [bn_rows.index(i) for i in selected]
    assert [en_rows[i] for i in en_pick] == [bn_rows[i] for i in bn_pick] == selected

    # ...whereas naive row order would not.
    assert en_rows != bn_rows


def test_copa_prompt_includes_the_premise():
    """Without the premise the task is unanswerable and the panel scores chance."""
    from mirage.data.loaders import _extract_question

    f = {"premise": "premise", "question": "question", "options": ["c1", "c2"]}
    row = {"premise": "The man broke his toe.", "question": "cause"}
    q, ctx = _extract_question(f, row)
    assert ctx == "The man broke his toe."
    assert "cause" in q.lower() and q != "cause"


def test_copa_effect_question_is_translated():
    from mirage.data.loaders import _extract_question

    f = {"premise": "premise", "question": "question"}
    q, _ = _extract_question(f, {"premise": "p", "question": "effect"})
    assert "result" in q.lower()
