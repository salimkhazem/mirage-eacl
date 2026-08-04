"""MIRAGE -- measurement invariance and the cross-lingual gap.

The public surface is deliberately small:

    from mirage import fit_irt, gap_bounds, median_anchor, simulate

``fit_irt`` estimates the multilingual IRT model under a *declared* sum-zero
normalisation; ``gap_bounds`` converts that fit into sharp bounds on the
cross-lingual gap; ``median_anchor`` gives the point estimate that is valid only
under majority invariance.  See ``docs/THEORY.md`` for the statements these
implement.
"""

from mirage.identify import (
    IdentifiedSet,
    Interval,
    breakeven_drift,
    gap_bounds,
    median_anchor,
    minimax_floor,
    sliding_window_set,
)
from mirage.irt import IRTFit, fit_irt
from mirage.simulate import Truth, simulate

__version__ = "0.1.0"

__all__ = [
    "IRTFit",
    "IdentifiedSet",
    "Interval",
    "Truth",
    "fit_irt",
    "breakeven_drift",
    "gap_bounds",
    "median_anchor",
    "minimax_floor",
    "simulate",
    "sliding_window_set",
]
