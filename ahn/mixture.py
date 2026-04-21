# ahn/mixture.py
# ──────────────────────────────────────────────────────────────────────────────
# AHNMixture — top-level estimator for Artificial Hydrocarbon Networks.
#
# Wraps one or more AHNCompound objects and adds:
#   • Multi-compound weighted combination (Spec §2.8, Eq. 4)
#   • Multi-restart training  (Spec §5.3 / §12.1)
#   • Platt Scaling probability calibration (Spec §6)
#   • sklearn-compatible API (fit / predict / predict_proba / get_params)
#   • Persistence helpers (save / load)
#   • Rich summary display
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from scipy.linalg import lstsq

from ahn._core import AHNCompound
from ahn._version import __version__

__all__ = ["AHNMixture"]


class AHNMixture:
    """Mixture of *c* AHN compounds with optional Platt calibration.

    This is the main estimator of the AHN library.  It follows the
    scikit-learn estimator interface and can therefore be used inside
    pipelines and grid-search utilities.

    Mixture output (Spec §2.8, Eq. 4):

    .. math::

        S(x) = \\sum_{i=1}^{c} \\alpha_i \\, \\psi_i(x)

    For ``c=1`` (default) the weight vector is fixed to ``α = [1.0]``, so
    ``S(x) = ψ₁(x)`` exactly.

    Calibrated probability (Spec §6):

    .. math::

        P(y=1 \\mid x) = \\sigma(a \\cdot S(x) + b)

    where *a, b* are estimated by :meth:`calibrate` on a held-out validation
    set.

    Parameters
    ----------
    n_compounds : int, default ``1``
        Number of AHN compounds *c* in the mixture.
    n_molecules : int, default ``2``
        Number of molecules *m* per compound (chain length).
    learning_rate : float, default ``0.1``
        Partition-boundary update step size η (Spec §3.2, Eq. 3).
    tolerance : float, default ``0.01``
        Convergence threshold ε — training stops when E_global ≤ ε.
    max_iterations : int, default ``100``
        Maximum number of Algorithm-1 iterations per compound.
    random_state : int, default ``42``
        Base seed.  Each compound/restart receives ``seed + i + restart*c``.
    use_bias : bool, default ``True``
        Enable trainable bias term *b* in each molecule (for k ≥ 2).
    use_bce : bool, default ``False``
        When ``c > 1``: use logistic regression to estimate α (BCE loss)
        instead of ordinary least squares.
    threshold : float, default ``0.5``
        Decision boundary τ for :meth:`predict` (Spec §2.9, Eq. A5).
        Note: :meth:`predict` always uses the raw score, never Platt.
    patience : int, default ``20``
        Stagnation patience — number of no-improvement iterations before
        partition boundaries are re-initialised (Spec §5.1).
    n_restarts : int, default ``1``
        Full training repetitions.  The restart with the lowest ``best_E_``
        is retained.

    Attributes
    ----------
    compounds : list of AHNCompound
        Fitted compound objects (populated after :meth:`fit`).
    alphas : ndarray of shape (n_compounds,)
        Mixture weights α (set by :meth:`fit`).
    platt_a : float or None
        Platt slope *a* (set by :meth:`calibrate`).
    platt_b : float or None
        Platt intercept *b* (set by :meth:`calibrate`).
    is_fitted_ : bool
        Whether :meth:`fit` has been called.
    is_calibrated_ : bool
        Whether :meth:`calibrate` has been called.

    Examples
    --------
    >>> from ahn import AHNMixture
    >>> model = AHNMixture(n_molecules=2, learning_rate=0.3,
    ...                    tolerance=0.1, max_iterations=500,
    ...                    use_bias=True, n_restarts=3)
    >>> model.fit(X_train, y_train)
    >>> model.calibrate(X_val, y_val)
    >>> model.summary()
    >>> y_pred  = model.predict(X_test)
    >>> y_proba = model.predict_proba(X_test)[:, 1]
    """

    # ── Construction ─────────────────────────────────────────────────────────

    def __init__(
        self,
        n_compounds:    int   = 1,
        n_molecules:    int   = 2,
        learning_rate:  float = 0.1,
        tolerance:      float = 0.01,
        max_iterations: int   = 100,
        random_state:   int   = 42,
        use_bias:       bool  = True,
        use_bce:        bool  = False,
        threshold:      float = 0.5,
        patience:       int   = 20,
        n_restarts:     int   = 1,
    ) -> None:
        self.n_compounds    = n_compounds
        self.n_molecules    = n_molecules
        self.learning_rate  = learning_rate
        self.tolerance      = tolerance
        self.max_iterations = max_iterations
        self.random_state   = random_state
        self.use_bias       = use_bias
        self.use_bce        = use_bce
        self.threshold      = threshold
        self.patience       = patience
        self.n_restarts     = n_restarts

        # Populated by fit / calibrate
        self.compounds: List[AHNCompound] = []
        self.alphas:    Optional[np.ndarray] = None
        self.platt_a:   Optional[float] = None
        self.platt_b:   Optional[float] = None

        self._n_features: Optional[int] = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_compound(self, seed: int) -> AHNCompound:
        return AHNCompound(
            n_molecules    = self.n_molecules,
            n_features     = None,
            learning_rate  = self.learning_rate,
            tolerance      = self.tolerance,
            max_iterations = self.max_iterations,
            random_state   = seed,
            use_bias       = self.use_bias,
            threshold      = self.threshold,
            patience       = self.patience,
        )

    # ══════════════════════════════════════════════════════════════════════════
    #  Training
    # ══════════════════════════════════════════════════════════════════════════

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = True,
    ) -> "AHNMixture":
        """Fit the mixture of AHN compounds to training data.

        Parameters
        ----------
        X : array-like of shape (N, n_features)
            Training features — should be scaled to ``[-1, 1]``.
        y : array-like of shape (N,)
            Binary labels {0, 1}.
        verbose : bool, default ``True``
            Print per-iteration and per-compound diagnostics.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._n_features = X.shape[1]

        best_compounds: Optional[List[AHNCompound]] = None
        best_E_total = np.inf

        for restart in range(self.n_restarts):
            if verbose and self.n_restarts > 1:
                print(f"\n  [Restart {restart + 1}/{self.n_restarts}]")

            compounds_try: List[AHNCompound] = []
            E_total = 0.0

            for i in range(self.n_compounds):
                if verbose:
                    mid = "CH2-" * (self.n_molecules - 2)
                    print(
                        f"\n  Compuesto {i + 1}/{self.n_compounds}  "
                        f"(CH3-{mid}CH3):"
                    )
                seed = self.random_state + i + restart * self.n_compounds
                comp = self._make_compound(seed)
                comp.fit(X, y, verbose=verbose)
                E_total += getattr(comp, "best_E_", np.inf)
                compounds_try.append(comp)

            if E_total < best_E_total:
                best_E_total   = E_total
                best_compounds = compounds_try
                if verbose and self.n_restarts > 1:
                    print(
                        f"  ★ Nuevo mejor  E={E_total:.6f}  "
                        f"(restart {restart + 1})"
                    )

        self.compounds = best_compounds or []

        # Mixture weights (Spec §2.8)
        if self.n_compounds > 1:
            Psi = np.column_stack(
                [c.predict_raw(X) for c in self.compounds]
            )
            if self.use_bce:
                from sklearn.linear_model import LogisticRegression
                lr = LogisticRegression(
                    C=1e4, solver="lbfgs", max_iter=1000, fit_intercept=False
                )
                lr.fit(Psi, y.astype(int))
                self.alphas = lr.coef_[0]
            else:
                self.alphas, *_ = lstsq(Psi, y)
        else:
            self.alphas = np.array([1.0])

        return self

    # ══════════════════════════════════════════════════════════════════════════
    #  Probability calibration — Platt Scaling  (Spec §6)
    # ══════════════════════════════════════════════════════════════════════════

    def calibrate(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "AHNMixture":
        """Calibrate probabilities with Platt Scaling on a held-out set.

        Fits a logistic regression on the raw mixture scores to obtain
        calibrated probability estimates (Spec §6.2):

        .. math::

            P(y=1 \\mid x) = \\sigma(a \\cdot S(x) + b)

        .. warning::

            ``X_val`` must be **disjoint** from both the training set used in
            :meth:`fit` and the test set used for evaluation (Spec §6.3).

        Parameters
        ----------
        X_val : array-like of shape (N_val, n_features)
        y_val : array-like of shape (N_val,)

        Returns
        -------
        self
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "Call fit() before calibrate()."
            )
        from sklearn.linear_model import LogisticRegression

        X_val = np.asarray(X_val, dtype=float)
        y_val = np.asarray(y_val)
        scores = self.predict_raw(X_val).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        lr.fit(scores, y_val)
        self.platt_a = float(lr.coef_[0, 0])
        self.platt_b = float(lr.intercept_[0])
        return self

    # backward-compatible alias
    def fit_platt(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "AHNMixture":
        """Alias for :meth:`calibrate` (backward compatibility)."""
        return self.calibrate(X_val, y_val)

    # ══════════════════════════════════════════════════════════════════════════
    #  Inference
    # ══════════════════════════════════════════════════════════════════════════

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Return raw mixture score S(x) = Σ α_i ψ_i(x) (Spec §2.8, Eq. 4).

        This is the uncalibrated continuous output before thresholding or
        Platt transformation.
        """
        self._check_fitted()
        X   = np.asarray(X, dtype=float)
        Psi = np.array([c.predict_raw(X) for c in self.compounds])
        return self.alphas @ Psi

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels using threshold τ (Spec §2.9, Eq. A5).

        .. note::

            This method always uses the **raw score** (never Platt-transformed)
            to ensure deterministic, calibration-independent decisions.

        Parameters
        ----------
        X : array-like of shape (N, n_features)

        Returns
        -------
        ndarray of int, shape (N,)  — values in {0, 1}
        """
        return (self.predict_raw(X) >= self.threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated class probabilities.

        Uses Platt Scaling when :meth:`calibrate` has been called, otherwise
        applies a plain sigmoid to the raw score.

        Parameters
        ----------
        X : array-like of shape (N, n_features)

        Returns
        -------
        ndarray of shape (N, 2)
            Column 0 → P(y=0 | x),  Column 1 → P(y=1 | x).
        """
        self._check_fitted()
        X   = np.asarray(X, dtype=float)
        raw = self.predict_raw(X)
        logit = (self.platt_a * raw + self.platt_b) if self.is_calibrated_ else raw
        prob  = 1.0 / (1.0 + np.exp(-logit))
        return np.column_stack([1 - prob, prob])

    # ══════════════════════════════════════════════════════════════════════════
    #  sklearn compatibility
    # ══════════════════════════════════════════════════════════════════════════

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return constructor parameters (sklearn API)."""
        return {
            "n_compounds":    self.n_compounds,
            "n_molecules":    self.n_molecules,
            "learning_rate":  self.learning_rate,
            "tolerance":      self.tolerance,
            "max_iterations": self.max_iterations,
            "random_state":   self.random_state,
            "use_bias":       self.use_bias,
            "use_bce":        self.use_bce,
            "threshold":      self.threshold,
            "patience":       self.patience,
            "n_restarts":     self.n_restarts,
        }

    def set_params(self, **params: Any) -> "AHNMixture":
        """Set parameters (sklearn API)."""
        for key, val in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid parameter '{key}'.")
            setattr(self, key, val)
        return self

    # ══════════════════════════════════════════════════════════════════════════
    #  Persistence
    # ══════════════════════════════════════════════════════════════════════════

    def save(self, path: Union[str, Path]) -> None:
        """Serialise the fitted model to a pickle file.

        Parameters
        ----------
        path : str or Path
            Destination file path (e.g. ``'model.ahn'`` or ``'model.pkl'``).

        Examples
        --------
        >>> model.save("credit_risk_model.ahn")
        >>> loaded = AHNMixture.load("credit_risk_model.ahn")
        """
        path = Path(path)
        payload = {
            "_ahn_version": __version__,
            "params":       self.get_params(),
            "compounds":    self.compounds,
            "alphas":       self.alphas,
            "platt_a":      self.platt_a,
            "platt_b":      self.platt_b,
            "_n_features":  self._n_features,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AHNMixture":
        """Load a model saved with :meth:`save`.

        Parameters
        ----------
        path : str or Path

        Returns
        -------
        AHNMixture
        """
        path = Path(path)
        with open(path, "rb") as fh:
            payload = pickle.load(fh)

        saved_ver = payload.get("_ahn_version", "?")
        if saved_ver != __version__:
            warnings.warn(
                f"Model was saved with ahn {saved_ver}; "
                f"current version is {__version__}.",
                UserWarning,
                stacklevel=2,
            )

        model             = cls(**payload["params"])
        model.compounds   = payload["compounds"]
        model.alphas      = payload["alphas"]
        model.platt_a     = payload["platt_a"]
        model.platt_b     = payload["platt_b"]
        model._n_features = payload["_n_features"]
        return model

    # ══════════════════════════════════════════════════════════════════════════
    #  Properties
    # ══════════════════════════════════════════════════════════════════════════

    @property
    def is_fitted_(self) -> bool:
        """``True`` if :meth:`fit` has been called."""
        return len(self.compounds) > 0

    @property
    def is_calibrated_(self) -> bool:
        """``True`` if :meth:`calibrate` has been called."""
        return self.platt_a is not None

    @property
    def n_features_(self) -> Optional[int]:
        """Input dimensionality (set during :meth:`fit`)."""
        return self._n_features

    @property
    def n_parameters_(self) -> int:
        """Total number of trainable parameters across all compounds."""
        return sum(c.n_parameters for c in self.compounds)

    @property
    def formula(self) -> str:
        """Chemical formula of one compound, e.g. ``'CH3-CH3'``."""
        m = self.n_molecules
        if m == 1:
            return "CH3"
        mid = "CH2-" * (m - 2)
        return f"CH3-{mid}CH3"

    # ══════════════════════════════════════════════════════════════════════════
    #  Display
    # ══════════════════════════════════════════════════════════════════════════

    def summary(self) -> None:
        """Print a human-readable model summary (inspired by Keras / TF)."""
        w = 62
        line = "═" * w

        print(line)
        print(f"  AHNMixture  —  ahn v{__version__}")
        print(line)
        print(f"  Structure      : {self.formula}  (m={self.n_molecules} molecules/compound)")
        print(f"  Compounds      : {self.n_compounds}")
        if self.is_fitted_:
            print(f"  Input features : {self.n_features_}")
            print(f"  Total params   : {self.n_parameters_:,}")
        print(f"  Learning rate  : {self.learning_rate}")
        print(f"  Tolerance (ε)  : {self.tolerance}")
        print(f"  Max iterations : {self.max_iterations}")
        print(f"  Patience       : {self.patience}")
        print(f"  Bias           : {self.use_bias}")
        print(f"  Restarts       : {self.n_restarts}")
        print(f"  Threshold (τ)  : {self.threshold}")
        print("─" * w)

        if self.is_fitted_:
            for ci, comp in enumerate(self.compounds):
                print(f"  Compound {ci + 1}:")
                for j, mol in enumerate(comp.molecules):
                    print(
                        f"    Molecule {j + 1}  {mol.formula:<4}  "
                        f"(k={mol.k})  params={mol.n_parameters}"
                    )
                if hasattr(comp, "best_E_"):
                    print(f"    Best E_global : {comp.best_E_:.6f}")
            if self.n_compounds > 1:
                alpha_str = ", ".join(f"{a:.4f}" for a in self.alphas)
                print(f"  Mixture α      : [{alpha_str}]")
        else:
            print("  ⚠  Not fitted — call fit(X_train, y_train) first.")

        print("─" * w)
        if self.is_calibrated_:
            print(
                f"  Calibration    : Platt Scaling  "
                f"(a={self.platt_a:.4f}, b={self.platt_b:.4f})"
            )
        else:
            print("  Calibration    : None  (call calibrate(X_val, y_val))")
        print(line)

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"AHNMixture(n_compounds={self.n_compounds}, "
            f"n_molecules={self.n_molecules}, "
            f"learning_rate={self.learning_rate}, "
            f"tolerance={self.tolerance}, "
            f"max_iterations={self.max_iterations}, "
            f"use_bias={self.use_bias}, "
            f"n_restarts={self.n_restarts}, "
            f"fitted={self.is_fitted_}, "
            f"calibrated={self.is_calibrated_})"
        )

    def __str__(self) -> str:  # noqa: D105
        return repr(self)

    # ── Private ──────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(
                "AHNMixture is not fitted yet.  Call fit(X_train, y_train) first."
            )
