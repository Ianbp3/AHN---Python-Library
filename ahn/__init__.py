# ahn/__init__.py
# ──────────────────────────────────────────────────────────────────────────────
# Artificial Hydrocarbon Networks (AHN) — public API
#
# Quick start
# ───────────
#   from ahn import AHNMixture
#   from ahn.metrics import evaluate, cross_validate, compare
#   from ahn.visualization import plot_all
#   from ahn.experiments import data_scarcity, feature_noise, label_noise
#
# Advanced
#   from ahn import AHNMolecule, AHNCompound  # internal building blocks
# ──────────────────────────────────────────────────────────────────────────────

from ahn._version import (
    __version__,
    __author__,
    __license__,
    __doc__,
)

# Main estimator (always available)
from ahn.mixture import AHNMixture

# Internal building blocks (semi-public for power users)
from ahn._core import AHNMolecule, AHNCompound

# Convenience re-exports so users can do `ahn.evaluate(...)` etc.
from ahn.metrics import evaluate, cross_validate, compare
from ahn.visualization import (
    plot_roc_curves,
    plot_confusion_matrices,
    plot_metrics_bar,
    plot_radar,
    plot_convergence,
    plot_robustness,
    plot_all,
)
from ahn.experiments import (
    data_scarcity,
    feature_noise,
    label_noise,
    robustness_summary,
)

__all__ = [
    # Core
    "AHNMixture",
    "AHNCompound",
    "AHNMolecule",
    # Metrics
    "evaluate",
    "cross_validate",
    "compare",
    # Visualization
    "plot_roc_curves",
    "plot_confusion_matrices",
    "plot_metrics_bar",
    "plot_radar",
    "plot_convergence",
    "plot_robustness",
    "plot_all",
    # Experiments
    "data_scarcity",
    "feature_noise",
    "label_noise",
    "robustness_summary",
    # Metadata
    "__version__",
    "__author__",
    "__license__",
]
