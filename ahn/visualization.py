# ahn/visualization.py
# ──────────────────────────────────────────────────────────────────────────────
# Plotting utilities for AHN experiments.
#
# All functions accept an optional ``save_path`` argument; when provided the
# figure is saved at 300 dpi and closed.  When omitted the figure is returned
# for interactive use.
#
# Public API
# ──────────
#   plot_roc_curves(df, y_test, ...)
#   plot_confusion_matrices(df, y_test, ...)
#   plot_metrics_bar(df, ...)
#   plot_radar(df, ...)
#   plot_convergence(model, ...)
#   plot_robustness(rob_df, x_col, ...)
#   plot_all(df, y_test, model, output_dir, ...)
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import warnings
from math import pi
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve

__all__ = [
    "plot_roc_curves",
    "plot_confusion_matrices",
    "plot_metrics_bar",
    "plot_radar",
    "plot_convergence",
    "plot_robustness",
    "plot_all",
]

# ── Palette (consistent with experiment scripts) ──────────────────────────────
_PALETTE = {
    "AHN":           "#E74C3C",
    "SVM":           "#3498DB",
    "Random Forest": "#2ECC71",
    "MLP":           "#9B59B6",
}
_MARKERS = {"AHN": "o", "SVM": "s", "Random Forest": "^", "MLP": "D"}
_LW      = {"AHN": 2.5, "SVM": 1.8, "Random Forest": 1.8, "MLP": 1.8}
_HUSL    = sns.color_palette("husl", 8)

_METRICS_INFO = [
    ("test_acc",  "Test Accuracy"),
    ("precision", "Precision"),
    ("recall",    "Recall"),
    ("f1",        "F1-Score"),
    ("roc_auc",   "ROC-AUC"),
]


def _model_color(name: str) -> str:
    return _PALETTE.get(name, _HUSL[hash(name) % len(_HUSL)])


def _save_or_return(fig, save_path):
    if save_path is not None:
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  plot_roc_curves
# ══════════════════════════════════════════════════════════════════════════════

def plot_roc_curves(
    comparison_df: pd.DataFrame,
    y_test: np.ndarray,
    *,
    title: str = "ROC Curves — AHN vs Baseline Models",
    ax=None,
    save_path: Optional[Union[str, Path]] = None,
):
    """Plot superimposed ROC curves for all models in *comparison_df*.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output of :func:`~ahn.metrics.compare`; must contain ``_y_proba``
        and ``Model`` columns.
    y_test : array-like
    title : str
    ax : matplotlib Axes, optional
    save_path : str or Path, optional

    Returns
    -------
    matplotlib Figure (or ``None`` if saved).
    """
    y_test = np.asarray(y_test)
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 7))
    else:
        fig = ax.get_figure()

    plt.style.use("seaborn-v0_8-darkgrid")

    for _, row in comparison_df.iterrows():
        name   = row["Model"]
        y_prob = row["_y_proba"]
        auc    = row["ROC-AUC"]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        lw    = 3.0 if name == "AHN" else 1.8
        alpha = 1.0 if name == "AHN" else 0.8
        ax.plot(fpr, tpr, lw=lw, alpha=alpha,
                color=_model_color(name),
                label=f"{name} (AUC={auc:.4f})")
        if name == "AHN":
            ax.fill_between(fpr, tpr, alpha=0.07, color=_model_color(name))

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Random (0.5000)")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ══════════════════════════════════════════════════════════════════════════════
#  plot_confusion_matrices
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrices(
    comparison_df: pd.DataFrame,
    *,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrices",
    save_path: Optional[Union[str, Path]] = None,
):
    """Plot a row of confusion matrix heat-maps, one per model.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Must contain ``_cm``, ``Model``, ``Test Acc``, ``F1-Score``.
    class_names : list of str, optional
        Default ``["Class 0", "Class 1"]``.
    title : str
    save_path : str or Path, optional
    """
    n     = len(comparison_df)
    names = class_names or ["Class 0", "Class 1"]
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for ax, (_, row) in zip(axes, comparison_df.iterrows()):
        name = row["Model"]
        cm   = row["_cm"]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            cbar=False,
            xticklabels=names, yticklabels=names,
        )
        marker = "* " if name == "AHN" else ""
        ax.set_title(
            f"{marker}{name}\n"
            f"Acc={row['Test Acc']:.4f}  F1={row['F1-Score']:.4f}",
            fontsize=10, fontweight="bold", color=_model_color(name),
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ══════════════════════════════════════════════════════════════════════════════
#  plot_metrics_bar
# ══════════════════════════════════════════════════════════════════════════════

def plot_metrics_bar(
    comparison_df: pd.DataFrame,
    *,
    title: str = "Quantitative Comparison",
    save_path: Optional[Union[str, Path]] = None,
):
    """Four-panel bar chart: Accuracy, F1, ROC-AUC, Precision vs Recall."""
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(title, fontsize=15, fontweight="bold")

    models  = comparison_df["Model"].tolist()
    colors  = [_model_color(m) for m in models]
    x_pos   = np.arange(len(models))
    width   = 0.35

    def _add_labels(ax, bars):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.003,
                f"{h:.4f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold",
            )

    # Accuracy
    ax = axes[0, 0]
    vals = comparison_df["Test Acc"].tolist()
    _add_labels(ax, ax.bar(x_pos, vals, color=colors, alpha=0.85,
                           edgecolor="black", lw=1.2))
    ax.set_title("Test Accuracy", fontsize=13, fontweight="bold")
    ax.set_xticks(x_pos); ax.set_xticklabels(models, rotation=35, ha="right")
    ax.set_ylim([max(0, min(vals) - 0.1), 1.0])
    ax.set_ylabel("Accuracy"); ax.grid(axis="y", alpha=0.3)

    # F1
    ax = axes[0, 1]
    vals = comparison_df["F1-Score"].tolist()
    _add_labels(ax, ax.bar(x_pos, vals, color=colors, alpha=0.85,
                           edgecolor="black", lw=1.2))
    ax.set_title("F1-Score", fontsize=13, fontweight="bold")
    ax.set_xticks(x_pos); ax.set_xticklabels(models, rotation=35, ha="right")
    ax.set_ylim([max(0, min(vals) - 0.1), 1.0])
    ax.set_ylabel("F1"); ax.grid(axis="y", alpha=0.3)

    # ROC-AUC
    ax = axes[1, 0]
    vals = comparison_df["ROC-AUC"].tolist()
    _add_labels(ax, ax.bar(x_pos, vals, color=colors, alpha=0.85,
                           edgecolor="black", lw=1.2))
    ax.set_title("ROC-AUC", fontsize=13, fontweight="bold")
    ax.set_xticks(x_pos); ax.set_xticklabels(models, rotation=35, ha="right")
    ax.set_ylim([max(0, min(vals) - 0.1), 1.0])
    ax.set_ylabel("AUC"); ax.grid(axis="y", alpha=0.3)

    # Precision vs Recall
    ax = axes[1, 1]
    pres = comparison_df["Precision"].tolist()
    recs = comparison_df["Recall"].tolist()
    b1 = ax.bar(x_pos - width / 2, pres, width, label="Precision",
                alpha=0.85, edgecolor="black", lw=1.2, color=colors)
    b2 = ax.bar(x_pos + width / 2, recs, width, label="Recall",
                alpha=0.85, edgecolor="black", lw=1.2, color=colors)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Precision vs Recall", fontsize=13, fontweight="bold")
    ax.set_xticks(x_pos); ax.set_xticklabels(models, rotation=35, ha="right")
    ax.set_ylim([0, 1.05]); ax.legend(); ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ══════════════════════════════════════════════════════════════════════════════
#  plot_radar
# ══════════════════════════════════════════════════════════════════════════════

def plot_radar(
    comparison_df: pd.DataFrame,
    *,
    metrics: Optional[List[Tuple[str, str]]] = None,
    title:   str = "Metric Radar Chart",
    save_path: Optional[Union[str, Path]] = None,
):
    """Polar radar chart comparing all models across multiple metrics.

    Parameters
    ----------
    comparison_df : pd.DataFrame
    metrics : list of (col, label) pairs, optional
        Defaults to Accuracy / Precision / Recall / F1 / AUC.
    """
    plt.style.use("seaborn-v0_8-darkgrid")
    if metrics is None:
        metrics = [
            ("Test Acc",  "Accuracy"),
            ("Precision", "Precision"),
            ("Recall",    "Recall"),
            ("F1-Score",  "F1-Score"),
            ("ROC-AUC",   "ROC-AUC"),
        ]

    categories = [m[1] for m in metrics]
    N          = len(categories)
    angles     = [n / N * 2 * pi for n in range(N)] + [0]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection="polar"))

    for _, row in comparison_df.iterrows():
        name  = row["Model"]
        vals  = [float(row[c]) for c, _ in metrics] + [float(row[metrics[0][0]])]
        lw    = 3.0 if name == "AHN" else 2.0
        alpha = 0.7
        ax.plot(angles, vals, "o-", lw=lw, label=name,
                color=_model_color(name), alpha=alpha)
        ax.fill(angles, vals, alpha=0.1 if name != "AHN" else 0.2,
                color=_model_color(name))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=10)
    ax.set_title(title, size=14, fontweight="bold", pad=20)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ══════════════════════════════════════════════════════════════════════════════
#  plot_convergence
# ══════════════════════════════════════════════════════════════════════════════

def plot_convergence(
    model,
    X_train: Optional[np.ndarray] = None,
    *,
    title:     str = "AHN Internal Structure",
    tolerance: Optional[float] = None,
    save_path: Optional[Union[str, Path]] = None,
):
    """Plot partition sizes and E_global convergence for a fitted AHNMixture.

    Parameters
    ----------
    model : AHNMixture
    X_train : ndarray, optional
        If provided, partition bar chart is shown.
    tolerance : float, optional
        Horizontal reference line on the convergence plot.
    """
    from ahn.mixture import AHNMixture

    if not model.is_fitted_:
        raise RuntimeError("Model must be fitted before plotting convergence.")

    comp   = model.compounds[0]
    n_axes = 2 if X_train is not None else 1
    fig, axes = plt.subplots(1, n_axes, figsize=(7 * n_axes, 5))
    if n_axes == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Partition bar chart
    if X_train is not None:
        ax          = axes[0]
        assignments = comp._partition(np.asarray(X_train, dtype=float))
        counts      = [(assignments == j).sum() for j in range(comp.m)]
        mol_lbl = [
            "CH3\n(k=3)" if j == 0 or j == comp.m - 1 else "CH2\n(k=2)"
            for j in range(comp.m)
        ]
        bar_colors = [_model_color("AHN"), "#3498DB", "#2ECC71"][:comp.m]
        bars = ax.bar(
            [f"Mol {j + 1}\n{mol_lbl[j]}" for j in range(comp.m)],
            counts, color=bar_colors, alpha=0.85, edgecolor="white", lw=1.5,
        )
        for bar, n in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1, str(n),
                ha="center", fontweight="bold",
            )
        ax.set_ylabel("Samples assigned (train)")
        ax.set_title(f"Partition Σ_j  (m={comp.m} molecules)")

    # Convergence curve
    ax   = axes[-1]
    hist = comp.history
    ax.plot(
        range(1, len(hist) + 1), hist, "-o",
        color=_model_color("AHN"), lw=2, markersize=5,
        label="E_global = Σ E_j",
    )
    tol = tolerance or model.tolerance
    ax.axhline(
        tol, color="gray", ls="--", lw=1.5,
        label=f"ε = {tol}",
    )
    ax.set_xlabel("Iteration"); ax.set_ylabel("Global error")
    ax.set_title("Convergence — Algorithm 1")
    ax.legend()
    if hist and min(hist) > 0:
        ax.set_yscale("log")

    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ══════════════════════════════════════════════════════════════════════════════
#  plot_robustness
# ══════════════════════════════════════════════════════════════════════════════

def plot_robustness(
    rob_df: pd.DataFrame,
    x_col: str,
    *,
    x_labels: Optional[List[str]] = None,
    xlabel:   str = "",
    title:    str = "Robustness Experiment",
    highlight_ref: Optional[int] = None,
    save_path: Optional[Union[str, Path]] = None,
):
    """Five-panel robustness plot (one per metric) for a sweep DataFrame.

    Parameters
    ----------
    rob_df : pd.DataFrame
        Must contain columns: ``model``, ``<x_col>``, ``acc``, ``precision``,
        ``recall``, ``f1``, ``auc``.
    x_col : str
        Name of the sweep column (e.g. ``'fraction'``, ``'sigma'``,
        ``'flip_rate'``).
    x_labels : list of str, optional
        Tick labels for the x axis.
    xlabel : str
    title : str
    highlight_ref : int, optional
        Index of the reference (clean) condition — drawn as a dashed line.
    save_path : str or Path, optional
    """
    plt.style.use("seaborn-v0_8-darkgrid")
    x_vals  = sorted(rob_df[x_col].unique())
    models  = rob_df["model"].unique().tolist()
    if x_labels is None:
        x_labels = [str(v) for v in x_vals]

    metric_cols = ["acc", "precision", "recall", "f1", "auc"]
    metric_lbls = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)

    for ax, met, mlbl in zip(axes, metric_cols, metric_lbls):
        for m in models:
            sub  = rob_df[rob_df["model"] == m]
            vals = [
                sub[sub[x_col] == v][met].values[0]
                for v in x_vals
            ]
            mk = _MARKERS.get(m, "o")
            lw = _LW.get(m, 1.8)
            ax.plot(
                range(len(x_vals)), vals,
                marker=mk, color=_model_color(m),
                lw=lw, markersize=7, label=m, alpha=0.9,
            )

        if highlight_ref is not None:
            ax.axvline(
                highlight_ref, color="gray", lw=1.2,
                ls="--", alpha=0.6, label="Clean ref.",
            )
        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels(x_labels, fontsize=8, rotation=30, ha="right")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(mlbl, fontsize=9)
        ax.set_title(mlbl, fontsize=10, fontweight="bold")
        ax.set_ylim([0.0, 1.05])
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ══════════════════════════════════════════════════════════════════════════════
#  plot_all  — convenience wrapper
# ══════════════════════════════════════════════════════════════════════════════

def plot_all(
    comparison_df: pd.DataFrame,
    y_test: np.ndarray,
    *,
    model=None,
    X_train: Optional[np.ndarray] = None,
    output_dir: Optional[Union[str, Path]] = None,
    prefix: str = "",
    class_names: Optional[List[str]] = None,
    title_prefix: str = "",
) -> None:
    """Generate and save all standard comparison plots.

    Saves the following files in *output_dir*:

    * ``radar_comparison.png``
    * ``roc_curves.png``
    * ``metrics_bar.png``
    * ``confusion_matrices.png``
    * ``convergence.png``  (if *model* is provided)

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output of :func:`~ahn.metrics.compare`.
    y_test : array-like
    model : AHNMixture, optional
        If provided, convergence plot is generated.
    X_train : ndarray, optional
        If provided, partition plot is shown in the convergence figure.
    output_dir : str or Path, optional
        Directory to save figures.  Defaults to current directory.
    prefix : str
        Filename prefix (e.g. ``'cr_'``).
    class_names : list of str, optional
    title_prefix : str
        Prepended to all figure titles.
    """
    outdir = Path(output_dir) if output_dir else Path(".")
    outdir.mkdir(parents=True, exist_ok=True)

    tp = f"{title_prefix} " if title_prefix else ""

    plot_radar(
        comparison_df,
        title=f"{tp}Radar Chart — All Metrics",
        save_path=outdir / f"{prefix}radar_comparison.png",
    )
    print(f"  Saved: {prefix}radar_comparison.png")

    plot_roc_curves(
        comparison_df, y_test,
        title=f"{tp}ROC Curves",
        save_path=outdir / f"{prefix}roc_curves.png",
    )
    print(f"  Saved: {prefix}roc_curves.png")

    plot_metrics_bar(
        comparison_df,
        title=f"{tp}Quantitative Comparison",
        save_path=outdir / f"{prefix}metrics_bar.png",
    )
    print(f"  Saved: {prefix}metrics_bar.png")

    plot_confusion_matrices(
        comparison_df,
        class_names=class_names,
        title=f"{tp}Confusion Matrices",
        save_path=outdir / f"{prefix}confusion_matrices.png",
    )
    print(f"  Saved: {prefix}confusion_matrices.png")

    if model is not None:
        plot_convergence(
            model, X_train,
            title=f"{tp}AHN Internal Structure",
            save_path=outdir / f"{prefix}convergence.png",
        )
        print(f"  Saved: {prefix}convergence.png")
