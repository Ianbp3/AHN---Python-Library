# ahn/metrics.py
# ──────────────────────────────────────────────────────────────────────────────
# Evaluation utilities for AHN and baseline models.
#
# Public API
# ──────────
#   evaluate(model, X, y, ...)           → metrics dict
#   cross_validate(config, X, y, ...)    → CV summary dict
#   compare(models, X_train, y_train, X_test, y_test, ...) → pd.DataFrame
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

__all__ = ["evaluate", "cross_validate", "compare"]


# ══════════════════════════════════════════════════════════════════════════════
#  evaluate
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    X_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    X_val:   Optional[np.ndarray] = None,
    y_val:   Optional[np.ndarray] = None,
    verbose: bool = False,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute a full set of classification metrics for *model* on test data.

    Parameters
    ----------
    model
        Any fitted estimator exposing ``predict(X)`` and ``predict_proba(X)``.
    X_test, y_test
        Held-out evaluation set.
    X_train, y_train : optional
        If provided, train accuracy is also computed.
    X_val, y_val : optional
        If provided, validation accuracy is also computed.
    verbose : bool
        Print the classification report.
    class_names : list of str, optional
        Labels for the two classes (used by the verbose report).

    Returns
    -------
    dict with keys:
        ``test_acc``, ``precision``, ``recall``, ``f1``, ``roc_auc``,
        ``y_pred``, ``y_proba``, ``confusion_matrix``
        and optionally ``train_acc``, ``val_acc``.
    """
    X_test  = np.asarray(X_test,  dtype=float)
    y_test  = np.asarray(y_test)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results: Dict[str, Any] = {
        "test_acc":        accuracy_score(y_test,  y_pred),
        "precision":       precision_score(y_test, y_pred, zero_division=0),
        "recall":          recall_score(y_test,    y_pred, zero_division=0),
        "f1":              f1_score(y_test,         y_pred, zero_division=0),
        "roc_auc":         roc_auc_score(y_test,   y_proba),
        "y_pred":          y_pred,
        "y_proba":         y_proba,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    if X_train is not None and y_train is not None:
        results["train_acc"] = accuracy_score(
            np.asarray(y_train), model.predict(np.asarray(X_train, dtype=float))
        )
    if X_val is not None and y_val is not None:
        results["val_acc"] = accuracy_score(
            np.asarray(y_val), model.predict(np.asarray(X_val, dtype=float))
        )

    if verbose:
        names = class_names or ["Class 0", "Class 1"]
        print(classification_report(y_test, y_pred, target_names=names, digits=4))

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  cross_validate
# ══════════════════════════════════════════════════════════════════════════════

def cross_validate(
    config: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits:      int  = 5,
    random_state:  int  = 42,
    platt_fraction: float = 0.20,
    verbose:       bool = True,
) -> Dict[str, Any]:
    """Stratified K-Fold cross-validation for an :class:`~ahn.AHNMixture`.

    In each fold:

    1. The last ``platt_fraction`` of the train fold is reserved for Platt
       Scaling (strict anti-leakage — never from the held-out fold).
    2. ``AHNMixture(**config).fit(X_tr_pure, y_tr_pure)``
    3. ``model.calibrate(X_platt, y_platt)``
    4. Metrics evaluated on the held-out fold.

    Parameters
    ----------
    config : dict
        Constructor kwargs for :class:`~ahn.AHNMixture`.
    X, y : array-like
        Full (train + val) dataset — test set must remain separated.
    n_splits : int, default ``5``
    random_state : int
    platt_fraction : float
        Fraction of the train fold to reserve for Platt calibration.
    verbose : bool

    Returns
    -------
    dict with keys ``aucs``, ``accs``, ``f1s`` (raw lists) and
    ``mean_auc``, ``std_auc``, ``mean_acc``, ``std_acc``,
    ``mean_f1``,  ``std_f1``.
    """
    from ahn.mixture import AHNMixture   # local import avoids circular

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    aucs, accs, f1s = [], [], []

    if verbose:
        w = 70
        print("─" * w)
        print(
            f"K-FOLD CROSS VALIDATION  "
            f"(K={n_splits}, StratifiedKFold, "
            f"platt_frac={platt_fraction:.0%})"
        )
        print("─" * w)
        print(f"  {'Fold':>5}  {'AUC':>7}  {'ACC':>7}  {'F1':>7}  Platt (a, b)")

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y), 1):
        X_tr, X_ho = X[tr_idx], X[val_idx]
        y_tr, y_ho = y[tr_idx], y[val_idx]

        n_platt   = max(20, int(platt_fraction * len(X_tr)))
        X_platt   = X_tr[-n_platt:]
        y_platt   = y_tr[-n_platt:]
        X_tr_pure = X_tr[:-n_platt]
        y_tr_pure = y_tr[:-n_platt]

        fold_config = dict(config)
        fold_config["random_state"] = config.get("random_state", 42) + fold

        m = AHNMixture(**fold_config)
        m.fit(X_tr_pure, y_tr_pure, verbose=False)
        m.calibrate(X_platt, y_platt)

        yp   = m.predict(X_ho)
        ypr  = m.predict_proba(X_ho)[:, 1]
        auc  = roc_auc_score(y_ho, ypr)
        acc  = accuracy_score(y_ho, yp)
        f1   = f1_score(y_ho, yp, zero_division=0)

        aucs.append(auc)
        accs.append(acc)
        f1s.append(f1)

        if verbose:
            print(
                f"  {fold:>5}  {auc:.4f}   {acc:.4f}   {f1:.4f}   "
                f"a={m.platt_a:.3f}  b={m.platt_b:.3f}"
            )

    if verbose:
        w = 70
        print("─" * w)
        print(
            f"  {'Media':>5}  {np.mean(aucs):.4f}   "
            f"{np.mean(accs):.4f}   {np.mean(f1s):.4f}"
        )
        print(
            f"  {'±std':>5}  {np.std(aucs):.4f}   "
            f"{np.std(accs):.4f}   {np.std(f1s):.4f}"
        )

    return {
        "aucs":     aucs,
        "accs":     accs,
        "f1s":      f1s,
        "mean_auc": float(np.mean(aucs)),
        "std_auc":  float(np.std(aucs)),
        "mean_acc": float(np.mean(accs)),
        "std_acc":  float(np.std(accs)),
        "mean_f1":  float(np.mean(f1s)),
        "std_f1":   float(np.std(f1s)),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  compare
# ══════════════════════════════════════════════════════════════════════════════

def compare(
    models: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    *,
    X_val:        Optional[np.ndarray] = None,
    y_val:        Optional[np.ndarray] = None,
    ahn_key:      str  = "AHN",
    sort_by:      str  = "ROC-AUC",
    overall_weights: tuple = (0.3, 0.3, 0.4),
    verbose: bool = True,
) -> pd.DataFrame:
    """Train and evaluate multiple models, returning a ranked comparison table.

    AHN models (instances of :class:`~ahn.AHNMixture`) are assumed to be
    **already fitted**.  All other models (e.g. sklearn estimators) are fitted
    inside this function with ``fit(X_train, y_train)``.

    Parameters
    ----------
    models : dict
        Mapping ``name → estimator``.  AHN models must already be fitted;
        sklearn estimators will be fitted here.
    X_train, y_train
        Training data (used to fit baseline estimators).
    X_test, y_test
        Held-out evaluation data.
    X_val, y_val : optional
        If provided, val accuracy is also reported.
    ahn_key : str
        Dictionary key that identifies AHN models (prefix match).
    sort_by : str
        Column to sort the output table by (descending).
    overall_weights : tuple of (w_acc, w_f1, w_auc)
        Weights for the overall score column.
    verbose : bool
        Print the comparison table.

    Returns
    -------
    pd.DataFrame
        One row per model with columns: Model, Type, Train Acc, Val Acc,
        Test Acc, Precision, Recall, F1-Score, ROC-AUC, Overall Score.
    """
    from ahn.mixture import AHNMixture   # local import

    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    X_test  = np.asarray(X_test,  dtype=float)
    y_test  = np.asarray(y_test)

    rows = []
    w_acc, w_f1, w_auc = overall_weights

    for name, model in models.items():
        is_ahn = isinstance(model, AHNMixture) or name.startswith(ahn_key)

        if not is_ahn:
            model.fit(X_train, y_train)

        res = evaluate(
            model, X_test, y_test,
            X_train=X_train, y_train=y_train,
            X_val=X_val,     y_val=y_val,
        )

        overall = (
            w_acc * res["test_acc"]
            + w_f1 * res["f1"]
            + w_auc * res["roc_auc"]
        )

        rows.append({
            "Model":         name,
            "Type":          "AHN" if is_ahn else "Baseline",
            "Train Acc":     res.get("train_acc", float("nan")),
            "Val Acc":       res.get("val_acc",   float("nan")),
            "Test Acc":      res["test_acc"],
            "Precision":     res["precision"],
            "Recall":        res["recall"],
            "F1-Score":      res["f1"],
            "ROC-AUC":       res["roc_auc"],
            "Overall Score": overall,
            # store raw predictions for plotting
            "_y_pred":       res["y_pred"],
            "_y_proba":      res["y_proba"],
            "_cm":           res["confusion_matrix"],
        })

    df = (
        pd.DataFrame(rows)
        .sort_values(sort_by, ascending=False)
        .reset_index(drop=True)
    )

    if verbose:
        display_cols = [
            "Model", "Train Acc", "Val Acc", "Test Acc",
            "Precision", "Recall", "F1-Score", "ROC-AUC", "Overall Score",
        ]
        print("\n" + "=" * 70)
        print("COMPARISON TABLE  "
              f"(overall = {w_acc}·Acc + {w_f1}·F1 + {w_auc}·AUC)")
        print("=" * 70)
        print(
            df[display_cols]
            .to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )

        best = df.iloc[0]
        print(f"\n  ★ Best model  : {best['Model']}  "
              f"(overall={best['Overall Score']:.4f})")

        ahn_rows = df[df["Type"] == "AHN"]
        bl_rows  = df[df["Type"] == "Baseline"]
        if not ahn_rows.empty and not bl_rows.empty:
            a = ahn_rows.iloc[0]
            direction = lambda x: "▲" if x >= 0 else "▼"
            delta_acc = a["Test Acc"] - bl_rows["Test Acc"].mean()
            delta_f1  = a["F1-Score"] - bl_rows["F1-Score"].mean()
            delta_auc = a["ROC-AUC"]  - bl_rows["ROC-AUC"].mean()
            print(
                f"\n  AHN vs Baseline avg | "
                f"Acc {direction(delta_acc)}{abs(delta_acc):.4f}  "
                f"F1 {direction(delta_f1)}{abs(delta_f1):.4f}  "
                f"AUC {direction(delta_auc)}{abs(delta_auc):.4f}"
            )

    return df
