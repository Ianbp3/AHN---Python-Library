import os
import sys

# ── Allow running without installation: add library root to path ──────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

# ── AHN library ───────────────────────────────────────────────────────────────
import ahn
from ahn import AHNMixture
from ahn.metrics import compare, cross_validate, evaluate
from ahn.visualization import plot_all, plot_robustness
from ahn.experiments import (
    data_scarcity,
    feature_noise,
    label_noise,
    robustness_summary,
)

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_cr_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  BLOCK 0 — Banner
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print(f"VERIFY CRv2.4 — using ahn v{ahn.__version__}")
print("German Credit Risk  |  AHN vs SVM / Random Forest / MLP")
print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
#  BLOCK 1 — Load and prepare data  (mirrors CRv2.4.py Bloque 2)
# ══════════════════════════════════════════════════════════════════════════════

print("\nLoading dataset...")

try:
    df = pd.read_csv(
        "hf://datasets/inGeniia/german-credit-risk_credit-scoring_mlp/"
        "german_credit_risk.csv"
    )
    print("  Dataset loaded from HuggingFace")
    y       = df["Risk"].map({"good": 0, "bad": 1}).values
    X_raw   = df.drop(columns=["Risk"])
    X_array = pd.get_dummies(X_raw, drop_first=True).values.astype(float)
    DATA_SOURCE = "German Credit Risk (HuggingFace)"
except Exception:
    try:
        df = pd.read_csv("german_credit_risk.csv")
        print("  Dataset loaded from local file")
        y       = df["Risk"].map({"good": 0, "bad": 1}).values
        X_raw   = df.drop(columns=["Risk"])
        X_array = pd.get_dummies(X_raw, drop_first=True).values.astype(float)
        DATA_SOURCE = "German Credit Risk (local)"
    except Exception:
        print("  ⚠  Real dataset unavailable — using synthetic replica")
        from sklearn.datasets import make_classification
        X_array, y = make_classification(
            n_samples=1000, n_features=48, n_informative=12, n_redundant=10,
            n_clusters_per_class=3, weights=[0.7, 0.3],
            flip_y=0.05, random_state=42,
        )
        DATA_SOURCE = "Synthetic (GCR replica)"

# Stratified 60 / 20 / 20 split (Spec §7.3)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_array, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)

# MinMax scaling to [-1, 1] (Spec §7.3)
scaler  = MinMaxScaler(feature_range=(-1, 1))
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

print(f"  Source    : {DATA_SOURCE}")
print(f"  Features  : {X_train.shape[1]}")
print(f"  Train {X_train.shape}  Val {X_val.shape}  Test {X_test.shape}")
print(f"  Balance   : Good={int((y_test==0).sum())}  Bad={int((y_test==1).sum())}")


# ══════════════════════════════════════════════════════════════════════════════
#  BLOCK 2 — AHN configuration  (mirrors CRv2.4.py Bloque 3)
# ══════════════════════════════════════════════════════════════════════════════

AHN_CONFIG = dict(
    n_compounds    = 1,
    n_molecules    = 2,
    learning_rate  = 0.3,
    tolerance      = 0.1,
    max_iterations = 500,
    random_state   = 42,
    use_bias       = True,
    use_bce        = False,
    threshold      = 0.5,
    patience       = 20,
    n_restarts     = 3,
)

print("\n" + "=" * 70)
print("Training AHN (Artificial Hydrocarbon Networks)...")
print("=" * 70)

model = AHNMixture(**AHN_CONFIG)
model.fit(X_train, y_train, verbose=True)
model.calibrate(X_val, y_val)

# Rich model summary (new in library)
print()
model.summary()


# ══════════════════════════════════════════════════════════════════════════════
#  BLOCK 3 — K-Fold Cross Validation  (mirrors CRv2.4.py Bloque 3b)
# ══════════════════════════════════════════════════════════════════════════════

X_cv = np.vstack([X_train, X_val])
y_cv = np.concatenate([y_train, y_val])

cv_results = cross_validate(
    AHN_CONFIG, X_cv, y_cv,
    n_splits=5,
    random_state=42,
    platt_fraction=0.20,
    verbose=True,
)
print(
    f"  [Test AHN final:  "
    f"AUC={evaluate(model, X_test, y_test)['roc_auc']:.4f}  "
    f"ACC={evaluate(model, X_test, y_test)['test_acc']:.4f}  "
    f"F1={evaluate(model, X_test, y_test)['f1']:.4f}]"
)


# ══════════════════════════════════════════════════════════════════════════════
#  BLOCK 4 — Baseline factory  (mirrors CRv2.4.py Bloque 4)
# ══════════════════════════════════════════════════════════════════════════════

def make_baselines():
    """Return fresh (unfitted) baseline estimators."""
    return {
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=5, random_state=42
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
            alpha=0.001, learning_rate_init=0.001, max_iter=300,
            early_stopping=True, validation_fraction=0.15, random_state=42,
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  BLOCK 5 — Full comparison  (mirrors CRv2.4.py Bloques 4–5)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Training baselines & building comparison table...")
print("=" * 70)

# compare() trains baselines, evaluates all models, and prints the table
comparison_df = compare(
    {"AHN": model, **make_baselines()},
    X_train, y_train,
    X_test,  y_test,
    X_val=X_val, y_val=y_val,
    ahn_key="AHN",
    verbose=True,
)

comparison_df.drop(columns=["_y_pred", "_y_proba", "_cm"]).to_csv(
    os.path.join(OUTPUT_DIR, "final_comparison_table.csv"), index=False
)
print("\n  Saved: final_comparison_table.csv")


# ══════════════════════════════════════════════════════════════════════════════
#  BLOCK 6 — Visualizations  (mirrors CRv2.4.py Bloque 6)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Generating visualizations...")
print("=" * 70)

plot_all(
    comparison_df,
    y_test,
    model=model,
    X_train=X_train,
    output_dir=OUTPUT_DIR,
    prefix="cr_",
    class_names=["Good", "Bad"],
    title_prefix="Credit Risk",
)

# Save model
model.save(os.path.join(OUTPUT_DIR, "ahn_credit_risk.ahn"))
print("  Saved: ahn_credit_risk.ahn")

# Verify round-trip load
loaded = AHNMixture.load(os.path.join(OUTPUT_DIR, "ahn_credit_risk.ahn"))
assert np.allclose(
    model.predict_proba(X_test),
    loaded.predict_proba(X_test),
    atol=1e-9,
), "❌  Save/load round-trip failed!"
print("  ✓  Save/load round-trip verified")


# ══════════════════════════════════════════════════════════════════════════════
#  BLOCK 7 — Robustness Experiments  (mirrors CRv2.4.py Bloques 8–12)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Running robustness experiments...")
print("=" * 70)

# ── EXP 1: Data Scarcity ──────────────────────────────────────────────────────
sc_df = data_scarcity(
    AHN_CONFIG,
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    make_baselines,
    fractions=[0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00],
    verbose=True,
)
sc_df.to_csv(os.path.join(OUTPUT_DIR, "robustness_scarcity.csv"), index=False)

plot_robustness(
    sc_df, x_col="fraction",
    x_labels=[f"{f:.0%}\n(n={int(sc_df[sc_df.fraction==f]['n_train'].iloc[0])})"
               for f in sorted(sc_df["fraction"].unique())],
    xlabel="Train-set fraction",
    title="EXP 1 — Data Scarcity [Credit Risk]",
    highlight_ref=len(sc_df["fraction"].unique()) - 1,
    save_path=os.path.join(OUTPUT_DIR, "robustness_scarcity.png"),
)
print("  Saved: robustness_scarcity.png / .csv")

# ── EXP 2: Feature Noise ─────────────────────────────────────────────────────
fn_df = feature_noise(
    AHN_CONFIG,
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    make_baselines,
    sigmas=[0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0],
    seed=0,
    verbose=True,
)
fn_df.to_csv(os.path.join(OUTPUT_DIR, "robustness_feature_noise.csv"), index=False)

plot_robustness(
    fn_df, x_col="sigma",
    x_labels=[f"σ={s}" for s in sorted(fn_df["sigma"].unique())],
    xlabel="Gaussian noise σ  (scale [-1,1])",
    title="EXP 2 — Feature Noise [Credit Risk]",
    highlight_ref=0,
    save_path=os.path.join(OUTPUT_DIR, "robustness_feature_noise.png"),
)
print("  Saved: robustness_feature_noise.png / .csv")

# ── EXP 3: Label Noise ───────────────────────────────────────────────────────
ln_df = label_noise(
    AHN_CONFIG,
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    make_baselines,
    flip_rates=[0.00, 0.05, 0.10, 0.15, 0.20],
    seed=1,
    verbose=True,
)
ln_df.to_csv(os.path.join(OUTPUT_DIR, "robustness_label_noise.csv"), index=False)

plot_robustness(
    ln_df, x_col="flip_rate",
    x_labels=[f"{p:.0%}" for p in sorted(ln_df["flip_rate"].unique())],
    xlabel="Label flip rate (train)",
    title="EXP 3 — Label Noise [Credit Risk]",
    highlight_ref=0,
    save_path=os.path.join(OUTPUT_DIR, "robustness_label_noise.png"),
)
print("  Saved: robustness_label_noise.png / .csv")

# ── Consolidated summary ──────────────────────────────────────────────────────
robustness_summary(sc_df, fn_df, ln_df)


# ══════════════════════════════════════════════════════════════════════════════
#  BLOCK 8 — Final report
# ══════════════════════════════════════════════════════════════════════════════

ahn_row = comparison_df[comparison_df["Model"] == "AHN"].iloc[0]
bl_rows = comparison_df[comparison_df["Type"] == "Baseline"]

print("\n" + "=" * 70)
print("FINAL REPORT")
print("=" * 70)
print(f"\n  Dataset        : {DATA_SOURCE}")
print(f"  AHN structure  : {model.formula}  |  n_params={model.n_parameters_:,}")
print(
    f"\n  Test metrics   : "
    f"ACC={ahn_row['Test Acc']:.4f}  "
    f"F1={ahn_row['F1-Score']:.4f}  "
    f"AUC={ahn_row['ROC-AUC']:.4f}"
)
print(
    f"  CV results     : "
    f"AUC={cv_results['mean_auc']:.4f}±{cv_results['std_auc']:.4f}  "
    f"ACC={cv_results['mean_acc']:.4f}±{cv_results['std_acc']:.4f}"
)
print(
    f"\n  AHN vs Baseline avg | "
    f"ACC Δ={ahn_row['Test Acc'] - bl_rows['Test Acc'].mean():+.4f}  "
    f"F1 Δ={ahn_row['F1-Score'] - bl_rows['F1-Score'].mean():+.4f}  "
    f"AUC Δ={ahn_row['ROC-AUC'] - bl_rows['ROC-AUC'].mean():+.4f}"
)

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
print(f"\nOutput files in: {OUTPUT_DIR}")
files = [
    "1. final_comparison_table.csv",
    "2. cr_radar_comparison.png",
    "3. cr_roc_curves.png",
    "4. cr_metrics_bar.png",
    "5. cr_confusion_matrices.png",
    "6. cr_convergence.png",
    "7. ahn_credit_risk.ahn",
    "8. robustness_scarcity.png / .csv",
    "9. robustness_feature_noise.png / .csv",
    "10. robustness_label_noise.png / .csv",
]
for f in files:
    print(f"  {f}")
print("=" * 70)
