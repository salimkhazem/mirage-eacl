"""Tests for the generative (MGSM) scorer.

The extraction rule is language-sensitive by necessity: digits, decimal marks and
separators differ across MGSM's eleven languages. An extractor that silently
failed more often in some scripts would inject language-correlated measurement
error -- the exact confound the paper is about -- so every convention is tested.
"""

from __future__ import annotations

import pytest

from mirage.models.generative import extract_number

# --------------------------------------------------------------------------
# Digit scripts
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("The answer is 42", 42.0),
        ("答案是 42", 42.0),
        ("Jibu ni 42", 42.0),
        ("الجواب ١٢٣", 123.0),          # Arabic-Indic digits
        ("उत्तर ४२ है", 42.0),            # Devanagari digits
        ("উত্তর ৪২", 42.0),              # Bengali digits
        ("คำตอบคือ ๔๒", 42.0),           # Thai digits
    ],
)
def test_digit_scripts(text, want):
    assert extract_number(text) == want


# --------------------------------------------------------------------------
# Separator conventions -- resolved by position, not by locale
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("1,234.5", 1234.5),      # English
        ("1.234,5", 1234.5),      # German / Spanish
        ("1234.5", 1234.5),
        ("1 234,5", 1234.5),      # French, thin space
        ("12,345", 12345.0),      # single sep + 3 digits -> thousands
        ("1.234.567", 1234567.0),  # repeated sep -> thousands
        ("1,234,567", 1234567.0),  # repeated sep -> thousands
        ("1.234.567,89", 1234567.89),  # repeated + decimal
        ("3,14", 3.14),            # single sep + 2 digits -> decimal
        ("-17", -17.0),
        ("+8", 8.0),
    ],
)
def test_separators(text, want):
    assert extract_number(text) == pytest.approx(want)


# --------------------------------------------------------------------------
# Answer cues and chain-of-thought tails
# --------------------------------------------------------------------------


def test_prefers_number_after_an_answer_cue():
    t = "First 5 apples, then 3 more, so 5+3. The answer is 8."
    assert extract_number(t) == 8.0


def test_cue_in_other_languages():
    assert extract_number("Er hat 5 und 3. Antwort: 8") == 8.0
    assert extract_number("Il a 5 et 3. Réponse : 8") == 8.0
    assert extract_number("Ответ: 8") == 8.0


def test_falls_back_to_last_number_without_a_cue():
    assert extract_number("5 plus 3 equals 8") == 8.0


def test_trailing_punctuation_is_stripped():
    assert extract_number("The answer is 8.") == 8.0
    assert extract_number("The answer is 8,") == 8.0


# --------------------------------------------------------------------------
# Failure is explicit, never a silent zero
# --------------------------------------------------------------------------


def test_no_number_returns_none():
    assert extract_number("I don't know") is None
    assert extract_number("") is None
    assert extract_number("...") is None


def test_none_is_distinct_from_zero():
    """A missing answer must never be scored as the number zero."""
    assert extract_number("no idea") is None
    assert extract_number("the answer is 0") == 0.0


def test_mode_share_is_nan_for_generative_shards():
    """A hard generative shard must not be mistaken for a collapsed one.

    Under generative scoring `predicted` holds correctness, so a naive mode share
    equals max(acc, 1-acc): a 2%-accuracy MGSM shard would report 0.98 and be
    filtered out as degenerate, removing precisely the lowest-resource languages.
    """
    import numpy as np

    from mirage.models.scorer import ScoreResult

    r = ScoreResult(
        item_ids=[str(i) for i in range(100)],
        correct=np.array([1] * 2 + [0] * 98, dtype=np.uint8),
        predicted=np.array([1] * 2 + [0] * 98, dtype=np.int16),
        mode="generative",
    )
    assert r.accuracy == pytest.approx(0.02)
    assert r.mode_share != r.mode_share, "must be NaN, not 0.98"

    mc = ScoreResult(
        item_ids=[str(i) for i in range(100)],
        correct=np.zeros(100, dtype=np.uint8),
        predicted=np.zeros(100, dtype=np.int16),
        mode="letter",
    )
    assert mc.mode_share == 1.0, "genuine collapse must still be detected"
