# AHN — Artificial Hydrocarbon Networks

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-orange.svg)]()

A bio-inspired machine learning library for binary classification based on
**Artificial Hydrocarbon Networks (AHN)** — a structural analogy to organic
chemistry where molecules, compounds, and mixtures map directly to
hierarchical learning components.

---

## Hierarchy

```
AHNMolecule  →  AHNCompound  →  AHNMixture
   φ_k(x)          ψ(x)            S(x)
```

Each **molecule** (CH₃ / CH₂) fits a polynomial basis function.  
A **compound** chains *m* molecules over a partition of the feature space.  
A **mixture** combines *c* compounds with learned weights and optional Platt calibration.

---

## Quick Start

```python
from sklearn.preprocessing import MinMaxScaler
from ahn import AHNMixture

# 1 — Scale features to [-1, 1]  (required)
scaler  = MinMaxScaler(feature_range=(-1, 1))
X_train = scaler.fit_transform(X_train_raw)
X_val   = scaler.transform(X_val_raw)
X_test  = scaler.transform(X_test_raw)

# 2 — Create and train the model
model = AHNMixture(
    n_molecules    = 2,       # CH3-CH3 chain
    learning_rate  = 0.3,     # η for partition boundary update
    tolerance      = 0.1,     # ε convergence threshold
    max_iterations = 500,
    use_bias       = True,
    n_restarts     = 3,       # keep best of 3 random restarts
)
model.fit(X_train, y_train)

# 3 — Calibrate probabilities (Platt Scaling on a held-out val set)
model.calibrate(X_val, y_val)

# 4 — Inspect
model.summary()

# 5 — Predict
y_pred  = model.predict(X_test)          # binary labels {0, 1}
y_proba = model.predict_proba(X_test)    # calibrated probabilities, shape (N, 2)
```

---

## Full Comparison Workflow

```python
from ahn import AHNMixture
from ahn.metrics import compare, cross_validate
from ahn.visualization import plot_all
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

AHN_CONFIG = dict(n_molecules=2, learning_rate=0.3, tolerance=0.1,
                  max_iterations=500, use_bias=True, n_restarts=3)

model = AHNMixture(**AHN_CONFIG)
model.fit(X_train, y_train)
model.calibrate(X_val, y_val)

# K-Fold CV (anti-leakage: Platt fitted inside each fold)
cv = cross_validate(AHN_CONFIG, X_cv, y_cv, n_splits=5)

# Train baselines and compare all models
df = compare(
    {"AHN": model,
     "SVM": SVC(kernel="rbf", probability=True),
     "RF":  RandomForestClassifier(n_estimators=100),
     "MLP": MLPClassifier(hidden_layer_sizes=(64, 32))},
    X_train, y_train, X_test, y_test,
    X_val=X_val, y_val=y_val,
)

# Generate all plots (radar, ROC, bars, confusion matrices, convergence)
plot_all(df, y_test, model=model, X_train=X_train, output_dir="outputs/")
```

---

## Robustness Experiments

```python
from ahn.experiments import data_scarcity, feature_noise, label_noise, robustness_summary
from ahn.visualization import plot_robustness

def make_baselines():
    return {"SVM": SVC(...), "RF": RandomForestClassifier(...)}

# EXP 1 — Data Scarcity
sc_df = data_scarcity(AHN_CONFIG, X_train, y_train, X_val, y_val, X_test, y_test,
                       make_baselines, fractions=[0.05, 0.1, 0.2, 0.5, 1.0])

# EXP 2 — Feature Noise at Inference
fn_df = feature_noise(AHN_CONFIG, X_train, y_train, X_val, y_val, X_test, y_test,
                       make_baselines, sigmas=[0.0, 0.1, 0.3, 0.5, 1.0])

# EXP 3 — Label Noise in Training
ln_df = label_noise(AHN_CONFIG, X_train, y_train, X_val, y_val, X_test, y_test,
                     make_baselines, flip_rates=[0.0, 0.05, 0.10, 0.20])

# ΔAUC summary table
robustness_summary(sc_df, fn_df, ln_df)

# Plots
plot_robustness(sc_df, x_col="fraction", xlabel="Train fraction",
                title="Data Scarcity", save_path="scarcity.png")
```

---

## Model Persistence

```python
model.save("my_model.ahn")
loaded = AHNMixture.load("my_model.ahn")
```

---

## Installation

```bash
# From source (development)
pip install -e .

# Dependencies (installed automatically)
# numpy, scipy, scikit-learn, pandas, matplotlib, seaborn
```

---

## Module Reference

| Module | Contents |
|--------|----------|
| `ahn` | `AHNMixture`, `AHNCompound`, `AHNMolecule` + all re-exports |
| `ahn.metrics` | `evaluate`, `cross_validate`, `compare` |
| `ahn.visualization` | `plot_roc_curves`, `plot_confusion_matrices`, `plot_metrics_bar`, `plot_radar`, `plot_convergence`, `plot_robustness`, `plot_all` |
| `ahn.experiments` | `data_scarcity`, `feature_noise`, `label_noise`, `robustness_summary` |

---

## Key Design Choices

| Choice | Detail |
|--------|--------|
| **Partition axis** | PCA first component (PC1) — prevents L2 distance collapse in high-dim space |
| **Quantile init** | Boundaries initialised from data quantiles + ±10% noise |
| **Best-state tracking** | Snapshot at every improvement; restored unconditionally at end |
| **Stagnation reinit** | Full boundary re-init after `patience` non-improving iterations |
| **Platt separation** | `predict()` always uses raw score; `predict_proba()` uses Platt |
| **Multi-restart** | `n_restarts` full training runs; best `E_global` wins |

---

## Mathematical Reference

| Symbol | Description |
|--------|-------------|
| φ_k(x) | Molecule output: Σ_r σ_r · Π_i (x_r − H_{i,r}) + b |
| ψ(x) | Compound output: φ_{j*(x)}(x) |
| S(x) | Mixture output: Σ_i α_i · ψ_i(x) |
| j*(x) | argmin_j \| x·v₁ − c_j \| (1-D projection, PC1 axis) |
| E_j | ½·mean((y − round(clip(φ_j(x), −2, 3)))²) |

---

