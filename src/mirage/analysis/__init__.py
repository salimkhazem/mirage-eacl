"""Post-hoc analysis: DIF testing, bootstrap, and tensor assembly."""

from mirage.analysis.dif import benjamini_hochberg, dif_test, drift_bootstrap_se

__all__ = ["benjamini_hochberg", "dif_test", "drift_bootstrap_se"]
