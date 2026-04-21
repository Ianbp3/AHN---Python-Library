# ahn/_core.py
# ──────────────────────────────────────────────────────────────────────────────
# Internal building blocks of Artificial Hydrocarbon Networks.
#
# Hierarchy
# ─────────
#   AHNMolecule   φ_k(x)  — single CH_k unit, order-k polynomial per feature
#   AHNCompound   ψ(x)    — linear chain CH₃-(CH₂)_{m-2}-CH₃ of m molecules
#
# These classes are semi-private (prefixed _core) but are exported for advanced
# users who want direct access to the compound / molecule level.
#
# Mathematical reference: AHN Study Specification v1.0, §2–§5
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import copy
import warnings
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize

__all__ = ["AHNMolecule", "AHNCompound"]


# ══════════════════════════════════════════════════════════════════════════════
#  AHNMolecule — CH_k
# ══════════════════════════════════════════════════════════════════════════════

class AHNMolecule:
    """Single AHN molecule of order *k* (CH_k).

    Implements the non-linear basis function (Spec §2.2, Eq. 1):

    .. math::

        \\varphi_k(x) = \\sum_{r=1}^{n} \\sigma_r
                        \\prod_{i=1}^{k} (x_r - H_{i,r}) + b

    Parameters
    ----------
    k : int
        Polynomial order (number of hydrogen atoms).
        ``k=3`` → CH₃ (terminal, cubic); ``k=2`` → CH₂ (interior, quadratic).
    n_features : int
        Dimensionality of the input space.
    rng : numpy.random.Generator
        Random number generator (shared with parent compound).
    use_bias : bool, default ``False``
        If ``True`` **and** ``k ≥ 2``, a trainable scalar bias *b* is added.
        For ``k=1`` the bias is structurally redundant and is left as 0.

    Attributes
    ----------
    sigma : ndarray of shape (n_features,)
        Carbon values σ_r per feature (trainable).
    H : ndarray of shape (k, n_features)
        Hydrogen positions H_{i,r} (trainable).
    bias : float
        Scalar bias term *b* (trainable only when ``use_bias=True`` and ``k≥2``).
    """

    def __init__(
        self,
        k: int,
        n_features: int,
        rng: np.random.Generator,
        use_bias: bool = False,
    ) -> None:
        self.k          = k
        self.n_features = n_features
        self.use_bias   = use_bias and (k >= 2)   # bias is redundant for k=1
        self.sigma: np.ndarray = rng.standard_normal(n_features) * 0.01
        self.H:     np.ndarray = rng.standard_normal((k, n_features)) * 0.1
        self.bias:  float      = 0.0

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate φ_k(x) for a batch of inputs.

        Parameters
        ----------
        X : ndarray of shape (N, n_features)

        Returns
        -------
        ndarray of shape (N,)
        """
        result = np.zeros(len(X))
        for r in range(self.n_features):
            prod = np.ones(len(X))
            for i in range(self.k):
                prod *= X[:, r] - self.H[i, r]
            result += self.sigma[r] * prod
        return result + self.bias

    # ── Parameter serialisation ───────────────────────────────────────────────

    def get_params(self) -> np.ndarray:
        """Flatten all trainable parameters into a 1-D vector.

        Layout: ``[σ₁…σₙ | H₁₁…H_{k,n} | (b)]``
        """
        base = np.concatenate([self.sigma, self.H.ravel()])
        return np.append(base, self.bias) if self.use_bias else base

    def set_params(self, params: np.ndarray) -> None:
        """Restore parameters from a flat vector produced by :meth:`get_params`."""
        n = self.n_features
        self.sigma = params[:n].copy()
        self.H     = params[n : n + self.k * n].reshape(self.k, n).copy()
        if self.use_bias:
            self.bias = float(params[-1])

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def n_parameters(self) -> int:
        """Total number of trainable parameters: ``n·(k+1) + [1 if bias]``."""
        return self.n_features * (self.k + 1) + int(self.use_bias)

    @property
    def formula(self) -> str:
        """Chemical formula string, e.g. ``'CH3'`` or ``'CH2'``."""
        return f"CH{self.k}"

    # ── Dunder ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"AHNMolecule(k={self.k}, n_features={self.n_features}, "
            f"use_bias={self.use_bias}, n_parameters={self.n_parameters})"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  AHNCompound — CH₃-(CH₂)_{m-2}-CH₃
# ══════════════════════════════════════════════════════════════════════════════

class AHNCompound:
    """AHN compound: a linear chain of *m* molecules (Spec §2.4, Algorithm 1).

    The compound partitions the feature space into *m* regions along the first
    principal component axis of the training data (Spec §4, correction C2).
    Each region is modelled by a dedicated :class:`AHNMolecule`.

    Molecule orders follow the saturated-chain convention (Spec §2.4):

    .. code-block:: text

        m=1  →  CH₃              k_orders = [3]
        m=2  →  CH₃–CH₃          k_orders = [3, 3]
        m=3  →  CH₃–CH₂–CH₃      k_orders = [3, 2, 3]
        m=4  →  CH₃–CH₂–CH₂–CH₃  k_orders = [3, 2, 2, 3]

    Parameters
    ----------
    n_molecules : int, default ``3``
        Number of molecules *m*.
    n_features : int or None
        Input dimensionality (inferred from training data when ``None``).
    learning_rate : float, default ``0.1``
        Step size η for the partition boundary update (Spec §3.2, Eq. 3).
    tolerance : float, default ``0.01``
        Convergence threshold ε: training stops when E_global ≤ ε.
    max_iterations : int, default ``80``
        Maximum number of training iterations (Algorithm 1).
    random_state : int, default ``42``
        Seed for the internal :class:`numpy.random.Generator`.
    use_bias : bool, default ``False``
        Pass-through to each :class:`AHNMolecule`.
    threshold : float, default ``0.5``
        Decision threshold τ for :meth:`predict` (Spec §2.9, Eq. A5).
    patience : int, default ``20``
        Number of iterations without E_global improvement before the
        partition boundaries are re-initialised (Spec §5.1).

    Attributes
    ----------
    molecules : list of AHNMolecule
        Trained molecule objects.
    history : list of float
        E_global value recorded at each training iteration.
    best_E_ : float
        Best (minimum) E_global achieved during training.
    """

    def __init__(
        self,
        n_molecules:    int   = 3,
        n_features:     Optional[int] = None,
        learning_rate:  float = 0.1,
        tolerance:      float = 0.01,
        max_iterations: int   = 80,
        random_state:   int   = 42,
        use_bias:       bool  = False,
        threshold:      float = 0.5,
        patience:       int   = 20,
    ) -> None:
        self.m          = n_molecules
        self.n_feat     = n_features
        self.eta        = learning_rate
        self.epsilon    = tolerance
        self.max_iter   = max_iterations
        self.use_bias   = use_bias
        self.threshold  = threshold
        self.patience   = patience
        self.rng        = np.random.default_rng(random_state)

        # Saturated-chain k-order assignment (Spec §2.4)
        if   n_molecules == 1: self.k_orders = [3]
        elif n_molecules == 2: self.k_orders = [3, 3]
        else:                  self.k_orders = [3] + [2] * (n_molecules - 2) + [3]

        self.molecules: list = []
        self.L       = self.r = self.L_min = self.L_max = self.centers = None
        self.history: list = []
        self.best_E_: float = np.inf

        # Private PCA state (set by _init_bounds)
        self._chain_axis   = None
        self._proj_bounds  = None
        self._proj_centers = None
        self._p_min        = None
        self._p_max        = None

    # ══════════════════════════════════════════════════════════════════════════
    #  Partition geometry  (Spec §2.5, §2.6, Corrections C1–C3)
    # ══════════════════════════════════════════════════════════════════════════

    def _init_bounds(self, X: np.ndarray) -> None:
        """Initialise partition boundaries using quantile-based 1-D projection.

        Implements corrections C1 (quantile init with noise) and C3 (anchor
        extreme projection points) from Spec §4.
        """
        self.L_min = X.min(axis=0)
        self.L_max = X.max(axis=0)

        if self.m > 1:
            X_c = X - X.mean(axis=0)
            try:
                _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
                self._chain_axis = Vt[0]
            except np.linalg.LinAlgError:
                self._chain_axis = np.zeros(self.n_feat)
                self._chain_axis[np.argmax(X.var(axis=0))] = 1.0

            X_proj = X @ self._chain_axis
            p_min, p_max = X_proj.min(), X_proj.max()
            self._p_min, self._p_max = p_min, p_max

            # C1 — quantile init + small perturbation (Spec §4)
            q_steps  = np.linspace(0.0, 1.0, self.m + 1)
            p_bounds = np.quantile(X_proj, q_steps)
            step     = (p_max - p_min) / self.m
            noise    = self.rng.uniform(-0.10, 0.10, self.m + 1) * step
            noise[[0, -1]] = 0
            p_bounds = np.sort(p_bounds + noise)
            p_bounds[0], p_bounds[-1] = p_min, p_max  # C3 — anchor extremes
            self._proj_bounds  = p_bounds
            self._proj_centers = (p_bounds[:-1] + p_bounds[1:]) / 2

            X_mean  = X.mean(axis=0)
            self.L  = np.zeros((self.m + 1, self.n_feat))
            self.L[0]       = self.L_min
            self.L[-1]      = self.L_max
            for j in range(1, self.m):
                self.L[j] = np.clip(
                    X_mean + p_bounds[j] * self._chain_axis,
                    self.L_min + 1e-9, self.L_max - 1e-9,
                )

            self.r = np.diff(self.L, axis=0)[:-1]
            self._clip_r()
        else:
            self._chain_axis   = None
            self._proj_bounds  = None
            self._proj_centers = None
            self.r = np.zeros((0, self.n_feat))

        self._compute_bounds()

    def _clip_r(self) -> None:
        """Enforce minimum & maximum range constraints on r_j (Spec §5.3)."""
        ranges  = self.L_max - self.L_min
        min_val = np.maximum(ranges * 0.02, 1e-8)
        for j in range(len(self.r)):
            self.r[j] = np.maximum(self.r[j], min_val)
        for f in range(self.n_feat):
            total = self.r[:, f].sum()
            avail = ranges[f] * 0.98
            if total > avail and avail > 0:
                self.r[:, f] *= avail / total

    def _compute_bounds(self) -> None:
        """Reconstruct L and centres from r_j (Spec §2.5, Eq. 2)."""
        self.L      = np.zeros((self.m + 1, self.n_feat))
        self.L[0]   = self.L_min
        for j in range(1, self.m):
            self.L[j] = np.minimum(self.L[j - 1] + self.r[j - 1], self.L_max - 1e-9)
        self.L[self.m] = self.L_max
        self.centers   = np.array(
            [(self.L[j] + self.L[j + 1]) / 2 for j in range(self.m)]
        )

        if self._chain_axis is not None:
            p = (
                [self._p_min]
                + [float(np.dot(self.L[j], self._chain_axis)) for j in range(1, self.m)]
                + [self._p_max]
            )
            self._proj_centers = np.array(
                [(p[j] + p[j + 1]) / 2 for j in range(self.m)]
            )

    def _partition(self, X: np.ndarray) -> np.ndarray:
        """Assign each sample to its nearest molecule region (Spec §2.6, Eq. A1).

        Uses 1-D projection onto the chain axis (correction C2, Spec §4).
        Falls back to L2 distance for ``m=1``.

        Returns
        -------
        assignments : ndarray of int, shape (N,)
        """
        if self._chain_axis is not None:
            X_proj = X @ self._chain_axis
            dists  = np.abs(X_proj[:, None] - self._proj_centers[None, :])
        else:
            dists = np.stack(
                [np.linalg.norm(X - self.centers[j], axis=1) for j in range(self.m)],
                axis=1,
            )
        return np.argmin(dists, axis=1)

    # ══════════════════════════════════════════════════════════════════════════
    #  Molecule optimisation  (Spec §3.1)
    # ══════════════════════════════════════════════════════════════════════════

    def _fit_molecule(
        self,
        mol: AHNMolecule,
        X_p: np.ndarray,
        y_p: np.ndarray,
    ) -> float:
        """Optimise a single molecule with L-BFGS-B and analytic gradients.

        Returns E_j — the classification error on partition Σ_j (Spec §2.9, Eq. A6).
        """
        if len(X_p) == 0:
            return 0.0

        n, k = self.n_feat, mol.k

        def objective_and_grad(
            params: np.ndarray,
        ) -> Tuple[float, np.ndarray]:
            sigma  = params[:n]
            H      = params[n : n + k * n].reshape(k, n)
            bias   = float(params[-1]) if mol.use_bias else 0.0

            # Forward pass
            terms  = np.ones((len(X_p), n))
            factor = np.ones((k, len(X_p), n))
            for i in range(k):
                factor[i] = X_p - H[i]
                terms     *= factor[i]

            phi      = (sigma * terms).sum(axis=1) + bias
            residual = y_p - phi
            loss     = 0.5 * np.mean(residual ** 2)

            # Gradients (Spec §3.1)
            g_sigma = -np.mean(residual[:, None] * terms, axis=0)
            g_H = np.zeros((k, n))
            for i in range(k):
                with np.errstate(divide="ignore", invalid="ignore"):
                    cofactor = np.where(factor[i] != 0, terms / factor[i], 0.0)
                g_H[i] = np.mean(residual[:, None] * sigma * cofactor, axis=0)

            grad = np.concatenate([g_sigma, g_H.ravel()])
            if mol.use_bias:
                grad = np.append(grad, -np.mean(residual))
            return loss, grad

        res = minimize(
            objective_and_grad,
            mol.get_params(),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 150, "ftol": 1e-10, "gtol": 1e-7},
        )
        mol.set_params(res.x)

        # Classification error E_j (Spec §2.9, Eq. A6)
        preds = mol.evaluate_batch(X_p)
        E_j   = 0.5 * np.mean(
            (y_p - np.round(np.clip(preds, -2.0, 3.0))) ** 2
        )
        return float(E_j)

    # ══════════════════════════════════════════════════════════════════════════
    #  Best-state tracking  (Spec §5.2)
    # ══════════════════════════════════════════════════════════════════════════

    def _snapshot(self) -> Tuple:
        return (copy.deepcopy(self.molecules), self.r.copy())

    def _restore(self, snapshot: Tuple) -> None:
        self.molecules, self.r = snapshot
        self._compute_bounds()

    # ══════════════════════════════════════════════════════════════════════════
    #  Training — Algorithm 1  (Spec §3)
    # ══════════════════════════════════════════════════════════════════════════

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = True,
    ) -> "AHNCompound":
        """Train the compound on labelled data.

        Implements Algorithm 1 from the AHN paper with all v2.1 corrections
        and robustness improvements (Spec §3, §4, §5).

        Parameters
        ----------
        X : ndarray of shape (N, n_features)
        y : ndarray of shape (N,)
            Binary labels {0, 1}.
        verbose : bool
            Print per-iteration diagnostics.

        Returns
        -------
        self
        """
        self.n_feat    = X.shape[1]
        self.molecules = [
            AHNMolecule(k, self.n_feat, self.rng, use_bias=self.use_bias)
            for k in self.k_orders
        ]
        self._init_bounds(X)
        self.history = []

        best_E     = np.inf
        best_snap  = self._snapshot()
        no_improve = 0

        for it in range(self.max_iter):
            assignments = self._partition(X)
            errors      = []
            for j in range(self.m):
                mask = assignments == j
                errors.append(
                    self._fit_molecule(self.molecules[j], X[mask], y[mask])
                    if mask.sum() > 0 else 0.0
                )

            E_global = sum(errors)
            self.history.append(E_global)

            # Best-state tracking (Spec §5.2)
            if E_global < best_E:
                best_E     = E_global
                best_snap  = self._snapshot()
                no_improve = 0
            else:
                no_improve += 1

            if verbose and (it % 10 == 0 or it < 3):
                sizes = [(assignments == j).sum() for j in range(self.m)]
                star  = "*" if E_global == best_E else " "
                print(
                    f"  Iter {it + 1:3d}{star}| E_global={E_global:.6f} "
                    f"| particiones={sizes}"
                )

            # Convergence check
            if E_global <= self.epsilon:
                if verbose:
                    print(
                        f"  Convergido en iter {it + 1}  "
                        f"(E={E_global:.6f} <= {self.epsilon})"
                    )
                break

            # Partition boundary update (Spec §3.2, Eq. 3)
            if self.m > 1:
                E_ext = [0.0] + errors
                for j in range(self.m - 1):
                    self.r[j] += -self.eta * (E_ext[j] - E_ext[j + 1])
                self._clip_r()
                self._compute_bounds()

                # Stagnation reinit (Spec §5.1)
                if no_improve >= self.patience:
                    self._init_bounds(X)
                    no_improve = 0
                    if verbose:
                        print(
                            f"  ↺ Reinit bounds en iter {it + 1}  "
                            f"(sin mejora por {self.patience} iters)"
                        )

        # Restore global best (Spec §5.2)
        self._restore(best_snap)
        self.best_E_ = best_E
        if verbose:
            print(f"  ✓ Best-state restaurado  (E_best={best_E:.6f})")
        return self

    # ══════════════════════════════════════════════════════════════════════════
    #  Prediction
    # ══════════════════════════════════════════════════════════════════════════

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Return raw compound score ψ(x) (Spec §2.7, Eq. A2)."""
        assignments = self._partition(X)
        result      = np.zeros(len(X))
        for j in range(self.m):
            mask = assignments == j
            if mask.sum() > 0:
                result[mask] = self.molecules[j].evaluate_batch(X[mask])
        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels using threshold τ (Spec §2.9, Eq. A5)."""
        return (self.predict_raw(X) >= self.threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return uncalibrated sigmoid probabilities (shape N×2)."""
        raw  = self.predict_raw(X)
        prob = 1.0 / (1.0 + np.exp(-raw))
        return np.column_stack([1 - prob, prob])

    # ══════════════════════════════════════════════════════════════════════════
    #  Introspection
    # ══════════════════════════════════════════════════════════════════════════

    @property
    def formula(self) -> str:
        """Chemical formula string, e.g. ``'CH3-CH3'``."""
        if self.m == 1:
            return "CH3"
        mid   = "CH2-" * (self.m - 2)
        return f"CH3-{mid}CH3"

    @property
    def n_parameters(self) -> int:
        """Total trainable parameter count across all molecules."""
        return sum(mol.n_parameters for mol in self.molecules)

    @property
    def partition_sizes(self) -> Optional[list]:
        """Return molecule assignments on the last seen X (None if not fitted)."""
        return None  # populated externally when needed

    def __repr__(self) -> str:  # noqa: D105
        fitted = len(self.molecules) > 0
        s = (
            f"AHNCompound(m={self.m}, formula='{self.formula}', "
            f"eta={self.eta}, epsilon={self.epsilon}, "
            f"max_iter={self.max_iter}, fitted={fitted}"
        )
        if fitted:
            s += f", best_E={self.best_E_:.6f}"
        return s + ")"
