# ahn/experiments.py
# ──────────────────────────────────────────────────────────────────────────────
# Standardised robustness experiment runners (Spec §10).
#
# Public API
# ──────────
#   data_scarcity(...)   → pd.DataFrame  (EXP 1)
#   feature_noise(...)   → pd.DataFrame  (EXP 2)
#   label_noise(...)     → pd.DataFrame  (EXP 3)
#   robustness_summary(sc_df, fn_df, ln_df)
#
# Each function returns a tidy DataFrame with columns:
#   model, <x_col>, acc, precision, recall, f1, auc
#
# This DataFrame can be passed directly to
#   ahn.visualization.plot_robustness()
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

__all__ = ["data_scarcity", "feature_noise", "label_noise", "robustness_summary"]

# ── Default sweep values (mirror experiment scripts) ──────────────────────────
_DEFAULT_FRACTIONS  = [0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00]
_DEFAULT_SIGMAS     = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
_DEFAULT_FLIP_RATES = [0.00, 0.05, 0.10, 0.15, 0.20]


def _eval_model(model, X_te, y_te) -> Dict[str, float]:
    yp  = model.predict(X_te)
    ypr = model.predict_proba(X_te)[:, 1]
    return {
        "acc":       float(accuracy_score(y_te, yp)),
        "precision": float(precision_score(y_te, yp, zero_division=0)),
        "recall":    float(recall_score(y_te,    yp, zero_division=0)),
        "f1":        float(f1_score(y_te,         yp, zero_division=0)),
        "auc":       float(roc_auc_score(y_te,    ypr)),
    }


def _sweep_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Return a faster AHN config for sweep iterations (Spec §10)."""
    cfg = dict(base_config)
    cfg.setdefault("n_restarts",     1)
    cfg.setdefault("max_iterations", 40)
    cfg.setdefault("patience",       10)
    # Override to lightweight values for sweeps
    cfg["n_restarts"]     = 1
    cfg["max_iterations"] = 40
    cfg["patience"]       = 10
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
#  EXP 1 — Data Scarcity  (Spec §10.1)
# ══════════════════════════════════════════════════════════════════════════════

def data_scarcity(
    ahn_config: Dict[str, Any],
    X_train:    np.ndarray,
    y_train:    np.ndarray,
    X_val:      np.ndarray,
    y_val:      np.ndarray,
    X_test:     np.ndarray,
    y_test:     np.ndarray,
    baselines_factory: Callable[[], Dict[str, Any]],
    *,
    fractions: Optional[Sequence[float]] = None,
    verbose:   bool = True,
) -> pd.DataFrame:
    """EXP 1 — Measure performance degradation under training-set scarcity.

    For each fraction *f*, a stratified subsample of size
    ``max(int(N_train · f), 10)`` is drawn from *X_train*.
    ``X_val``, ``X_test`` are kept fixed (Spec §10.1).

    Parameters
    ----------
    ahn_config : dict
        Constructor kwargs for :class:`~ahn.AHNMixture`.  A lightweight
        sweep variant (1 restart, 40 iterations) is used automatically.
    X_train, y_train : array-like
        Full training set.
    X_val, y_val : array-like
        Validation set for Platt calibration (fixed across all fractions).
    X_test, y_test : array-like
        Held-out evaluation set (fixed across all fractions).
    baselines_factory : callable
        Zero-argument callable returning a fresh ``dict[name → estimator]``.
    fractions : sequence of float, optional
        Train-set fractions to test.  Defaults to
        ``[0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00]``.
    verbose : bool

    Returns
    -------
    pd.DataFrame  with columns ``model``, ``fraction``, ``n_train``,
                  ``acc``, ``precision``, ``recall``, ``f1``, ``auc``.
    """
    from ahn.mixture import AHNMixture

    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    X_val   = np.asarray(X_val,   dtype=float)
    y_val   = np.asarray(y_val)
    X_test  = np.asarray(X_test,  dtype=float)
    y_test  = np.asarray(y_test)

    fractions = list(fractions or _DEFAULT_FRACTIONS)
    cfg       = _sweep_config(ahn_config)
    rows: List[Dict] = []

    if verbose:
        print("\n" + "=" * 70)
        print("EXP 1 — DATA SCARCITY")
        print("=" * 70)
        model_names = ["AHN"] + list(baselines_factory().keys())
        print(f"  Fractions : {[f'{f:.0%}' for f in fractions]}")
        print(f"  Models    : {model_names}\n")

    for frac in fractions:
        n = max(int(len(X_train) * frac), 10)
        if frac < 1.0:
            idx, _ = train_test_split(
                np.arange(len(X_train)),
                train_size=n, stratify=y_train, random_state=42,
            )
        else:
            idx = np.arange(len(X_train))

        X_sub, y_sub = X_train[idx], y_train[idx]

        # AHN
        ahn = AHNMixture(**{**cfg, "random_state": cfg.get("random_state", 42)})
        ahn.fit(X_sub, y_sub, verbose=False)
        ahn.calibrate(X_val, y_val)
        metrics = _eval_model(ahn, X_test, y_test)
        rows.append({"model": "AHN", "fraction": frac, "n_train": n, **metrics})

        # Baselines
        for name, bl in baselines_factory().items():
            bl.fit(X_sub, y_sub)
            m = _eval_model(bl, X_test, y_test)
            rows.append({"model": name, "fraction": frac, "n_train": n, **m})

        if verbose:
            line = f"  frac={frac:.0%}  n={n:5d}  |  "
            all_m = ["AHN"] + list(baselines_factory().keys())
            mline = "  ".join(
                f"{mn} AUC={next(r['auc'] for r in rows if r['model']==mn and r['fraction']==frac):.3f}"
                for mn in all_m
            )
            print(line + mline)

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  EXP 2 — Feature Noise  (Spec §10.2)
# ══════════════════════════════════════════════════════════════════════════════

def feature_noise(
    ahn_config: Dict[str, Any],
    X_train:    np.ndarray,
    y_train:    np.ndarray,
    X_val:      np.ndarray,
    y_val:      np.ndarray,
    X_test:     np.ndarray,
    y_test:     np.ndarray,
    baselines_factory: Callable[[], Dict[str, Any]],
    *,
    sigmas: Optional[Sequence[float]] = None,
    seed:   int  = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """EXP 2 — Measure performance under Gaussian feature noise at inference.

    Models are trained **once** on clean *X_train*.  Noise N(0, σ²·I) is
    added to *X_test* for each σ value.  Labels y_test are never corrupted
    (Spec §10.2).

    Parameters
    ----------
    sigmas : sequence of float, optional
        Noise standard deviations (relative to the scaled ``[-1, 1]`` range).
        Defaults to ``[0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]``.
    seed : int
        RNG seed for noise generation (fixed per σ for reproducibility).

    Returns
    -------
    pd.DataFrame  with columns ``model``, ``sigma``,
                  ``acc``, ``precision``, ``recall``, ``f1``, ``auc``.
    """
    from ahn.mixture import AHNMixture

    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    X_val   = np.asarray(X_val,   dtype=float)
    y_val   = np.asarray(y_val)
    X_test  = np.asarray(X_test,  dtype=float)
    y_test  = np.asarray(y_test)

    sigmas = list(sigmas or _DEFAULT_SIGMAS)
    cfg    = _sweep_config(ahn_config)
    rng    = np.random.default_rng(seed)

    if verbose:
        print("\n" + "=" * 70)
        print("EXP 2 — FEATURE NOISE")
        print("=" * 70)
        print("  Train on clean X_train  |  noise added to X_test at inference")
        print(f"  σ values : {sigmas}\n")

    # Train once on clean data
    ahn = AHNMixture(**{**cfg, "random_state": cfg.get("random_state", 42)})
    ahn.fit(X_train, y_train, verbose=False)
    ahn.calibrate(X_val, y_val)

    baselines = baselines_factory()
    for bl in baselines.values():
        bl.fit(X_train, y_train)

    rows: List[Dict] = []
    for sigma in sigmas:
        noise = rng.normal(0, sigma, X_test.shape) if sigma > 0 else 0
        X_te_n = X_test + noise

        metrics = _eval_model(ahn, X_te_n, y_test)
        rows.append({"model": "AHN", "sigma": sigma, **metrics})

        for name, bl in baselines.items():
            m = _eval_model(bl, X_te_n, y_test)
            rows.append({"model": name, "sigma": sigma, **m})

        if verbose:
            all_m = ["AHN"] + list(baselines.keys())
            auc_str = "  ".join(
                f"{mn} AUC={next(r['auc'] for r in rows if r['model']==mn and r['sigma']==sigma):.3f}"
                for mn in all_m
            )
            print(f"  σ={sigma:.2f}  |  {auc_str}")

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  EXP 3 — Label Noise  (Spec §10.3)
# ══════════════════════════════════════════════════════════════════════════════

def label_noise(
    ahn_config: Dict[str, Any],
    X_train:    np.ndarray,
    y_train:    np.ndarray,
    X_val:      np.ndarray,
    y_val:      np.ndarray,
    X_test:     np.ndarray,
    y_test:     np.ndarray,
    baselines_factory: Callable[[], Dict[str, Any]],
    *,
    flip_rates: Optional[Sequence[float]] = None,
    seed:       int  = 1,
    verbose:    bool = True,
) -> pd.DataFrame:
    """EXP 3 — Measure performance under label corruption in training data.

    A random fraction *p* of *y_train* labels are flipped (symmetric flip,
    both classes equally affected).  *y_test* is never corrupted.  Platt
    calibration uses the clean *y_val* (Spec §10.3).

    Parameters
    ----------
    flip_rates : sequence of float, optional
        Label-flip probabilities.
        Defaults to ``[0.00, 0.05, 0.10, 0.15, 0.20]``.
    seed : int
        RNG seed for flip mask generation.

    Returns
    -------
    pd.DataFrame  with columns ``model``, ``flip_rate``, ``n_flipped``,
                  ``acc``, ``precision``, ``recall``, ``f1``, ``auc``.
    """
    from ahn.mixture import AHNMixture

    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    X_val   = np.asarray(X_val,   dtype=float)
    y_val   = np.asarray(y_val)
    X_test  = np.asarray(X_test,  dtype=float)
    y_test  = np.asarray(y_test)

    flip_rates = list(flip_rates or _DEFAULT_FLIP_RATES)
    cfg        = _sweep_config(ahn_config)
    rng        = np.random.default_rng(seed)

    if verbose:
        print("\n" + "=" * 70)
        print("EXP 3 — LABEL NOISE")
        print("=" * 70)
        print("  Flip random p% of y_train  |  y_test always clean")
        print(f"  Flip rates : {[f'{p:.0%}' for p in flip_rates]}\n")

    rows: List[Dict] = []
    for p in flip_rates:
        y_noisy = y_train.copy()
        if p > 0:
            flip_mask = rng.random(len(y_noisy)) < p
            y_noisy[flip_mask] = 1 - y_noisy[flip_mask]
        n_flipped = int((y_noisy != y_train).sum())

        # AHN
        ahn = AHNMixture(**{**cfg, "random_state": cfg.get("random_state", 42)})
        ahn.fit(X_train, y_noisy, verbose=False)
        ahn.calibrate(X_val, y_val)
        m = _eval_model(ahn, X_test, y_test)
        rows.append({"model": "AHN", "flip_rate": p, "n_flipped": n_flipped, **m})

        # Baselines
        for name, bl in baselines_factory().items():
            bl.fit(X_train, y_noisy)
            bm = _eval_model(bl, X_test, y_test)
            rows.append({
                "model": name, "flip_rate": p, "n_flipped": n_flipped, **bm
            })

        if verbose:
            all_m = ["AHN"] + list(baselines_factory().keys())
            auc_str = "  ".join(
                f"{mn} AUC={next(r['auc'] for r in rows if r['model']==mn and r['flip_rate']==p):.3f}"
                for mn in all_m
            )
            print(f"  flip={p:.0%}  ({n_flipped:4d} labels)  |  {auc_str}")

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  robustness_summary
# ══════════════════════════════════════════════════════════════════════════════

def robustness_summary(
    sc_df: pd.DataFrame,
    fn_df: pd.DataFrame,
    ln_df: pd.DataFrame,
) -> None:
    """Print a consolidated ΔAUC summary across all three robustness experiments.

    Parameters
    ----------
    sc_df : DataFrame from :func:`data_scarcity`
    fn_df : DataFrame from :func:`feature_noise`
    ln_df : DataFrame from :func:`label_noise`
    """

    def _get(df, model, x_col, x_val, metric="auc"):
        sub = df[(df["model"] == model) & (df[x_col] == x_val)]
        return float(sub[metric].values[0]) if len(sub) else float("nan")

    models = sc_df["model"].unique().tolist()

    print("\n" + "=" * 70)
    print("ROBUSTNESS SUMMARY — ΔAUC  (clean → worst condition)")
    print("=" * 70)
    print(
        f"\n  {'Experiment':<24}  {'Model':<16}  "
        f"{'AUC clean':>10}  {'AUC worst':>10}  {'ΔAUC':>7}"
    )
    print("  " + "-" * 72)

    # Scarcity: clean = 1.00, worst = 0.05
    sc_fracs = sorted(sc_df["fraction"].unique())
    for m in models:
        a_c = _get(sc_df, m, "fraction", sc_fracs[-1])
        a_e = _get(sc_df, m, "fraction", sc_fracs[0])
        print(
            f"  {'Scarcity (5%→100%)':<24}  {m:<16}  "
            f"{a_c:>10.4f}  {a_e:>10.4f}  {a_e - a_c:>+7.4f}"
        )

    print()
    # Feature noise: clean = σ=0, worst = max σ
    fn_sigmas = sorted(fn_df["sigma"].unique())
    for m in models:
        a_c = _get(fn_df, m, "sigma", fn_sigmas[0])
        a_e = _get(fn_df, m, "sigma", fn_sigmas[-1])
        print(
            f"  {f'Feature Noise (σ={fn_sigmas[-1]})':<24}  {m:<16}  "
            f"{a_c:>10.4f}  {a_e:>10.4f}  {a_e - a_c:>+7.4f}"
        )

    print()
    # Label noise: clean = 0%, worst = max flip
    ln_flips = sorted(ln_df["flip_rate"].unique())
    for m in models:
        a_c = _get(ln_df, m, "flip_rate", ln_flips[0])
        a_e = _get(ln_df, m, "flip_rate", ln_flips[-1])
        print(
            f"  {f'Label Noise ({ln_flips[-1]:.0%} flip)':<24}  {m:<16}  "
            f"{a_c:>10.4f}  {a_e:>10.4f}  {a_e - a_c:>+7.4f}"
        )

    print("\n" + "=" * 70)
