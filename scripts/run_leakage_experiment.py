"""Leakage experiment: objective-only vs full features on the PoQeMoN QoE dataset.

This script is the single source of truth for every modeling number reported in
README.md and docs/DETAILS.md. It is self-contained: it loads the committed raw
CSV (data/raw/pokemon.csv, 1,543 sessions), applies the same preprocessing and
feature engineering as notebooks/03_data_preprocessing.ipynb, then trains four
classifiers on two feature variants:

  * Objective — the deployable feature set: drops the four subjective QoF_*
    feedback ratings (leakage) plus two near-constant columns, QoU_Ustedy
    (92.4% identical values) and QoA_VLCresolution (95% of sessions at 360p).
  * Full — all 25 engineered features including QoF_* (the leakage benchmark).

Class imbalance handling: class_weight='balanced' for Logistic Regression,
Decision Tree and Random Forest. GradientBoostingClassifier has no class_weight
parameter, so it is fitted with sample_weight=compute_sample_weight('balanced').
All other hyperparameters are scikit-learn library defaults (LogisticRegression
max_iter raised to 1000 for convergence). Split: stratified 80/20, seed 42.

Outputs:
  results/metrics/model_comparison.csv
  results/figures/04_modeling_and_evaluation/8_model_comparison.png
  results/figures/04_modeling_and_evaluation/4_confusion_matrix_logistic_regression.png
  results/figures/04_modeling_and_evaluation/5_confusion_matrix_decision_tree.png
  results/figures/04_modeling_and_evaluation/6_confusion_matrix_random_forest.png
  results/figures/04_modeling_and_evaluation/7_confusion_matrix_gradient_boosting.png
  results/figures/04_modeling_and_evaluation/6_feature_importance_rf.png

Usage:
    python scripts/run_leakage_experiment.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = REPO_ROOT / "data" / "raw" / "pokemon.csv"
FEATURE_NAMES_TXT = REPO_ROOT / "data" / "processed" / "feature_names.txt"
METRICS_CSV = REPO_ROOT / "results" / "metrics" / "model_comparison.csv"
FIGURES_DIR = REPO_ROOT / "results" / "figures" / "04_modeling_and_evaluation"

SEED = 42
MOS_LABELS = ["Bad", "Poor", "Fair", "Good", "Excellent"]

# Columns excluded from the Objective (deployable) variant:
# QoF_* are subjective in-session user ratings (data leakage); QoU_Ustedy and
# QoA_VLCresolution are near-constant columns dropped from the deployable set.
NON_OBJECTIVE_COLS = [
    "QoF_begin",
    "QoF_shift",
    "QoF_audio",
    "QoF_video",
    "QoU_Ustedy",
    "QoA_VLCresolution",
]

# Chart chrome (light mode, validated reference palette)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"  # Objective variant / single-series bars
RED = "#e34948"  # Full (leakage) variant
SEQ_RAMP = ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]


def load_and_engineer() -> pd.DataFrame:
    """Mirror notebooks/03_data_preprocessing.ipynb exactly."""
    df = pd.read_csv(RAW_CSV)
    if len(df) != 1543:
        raise ValueError(f"Expected 1,543 sessions, got {len(df)}")

    # Drop identifiers and high-cardinality device columns
    df = df.drop(columns=["id", "user_id", "QoD_model", "QoD_os-version"])

    # One-hot encode operator with readable names (spelling kept from notebook 03)
    operator_mapping = {1: "SFR", 2: "BOUYEGUES", 3: "ORANGE", 4: "FREE"}
    operator_encoded = pd.get_dummies(df["QoS_operator"], prefix="Operator").astype(int)
    operator_encoded.columns = [
        f"Operator_{operator_mapping[int(col.split('_')[1])]}"
        for col in operator_encoded.columns
    ]
    df = pd.concat([df.drop(columns=["QoS_operator"]), operator_encoded], axis=1)

    # Engineered features (same formulas as notebook 03)
    df["Buffering_Severity"] = (
        df["QoA_BUFFERINGcount"] * df["QoA_BUFFERINGtime"] / 1000
    ).fillna(0)
    network_gen_mapping = {1: 2, 2: 3, 3: 3, 4: 3, 5: 4}  # EDGE=2G .. LTE=4G
    df["Network_Generation"] = df["QoS_type"].map(network_gen_mapping)
    df["Video_Quality_Index"] = (
        (df["QoA_VLCbitrate"] / df["QoA_VLCbitrate"].max()) * 0.4
        + (df["QoA_VLCframerate"] / df["QoA_VLCframerate"].max()) * 0.3
        + (df["QoA_VLCresolution"] / df["QoA_VLCresolution"].max()) * 0.3
    )
    df["Audio_Quality"] = df["QoA_VLCaudiorate"] * (1 - df["QoA_VLCaudioloss"] / 100)

    # Reorder so MOS comes last, and verify against the committed feature list
    feature_cols = [c for c in df.columns if c != "MOS"]
    committed = FEATURE_NAMES_TXT.read_text().split()
    if feature_cols != committed:
        raise ValueError(
            "Engineered feature list does not match data/processed/feature_names.txt:\n"
            f"  script:    {feature_cols}\n  committed: {committed}"
        )
    return df[feature_cols + ["MOS"]]


def build_models(y_train: np.ndarray) -> list[dict]:
    """The four classifiers. Library defaults except where noted in the docstring."""
    return [
        {
            "name": "Logistic Regression",
            "model": LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=SEED
            ),
            "scaled": True,
            "sample_weight": None,
        },
        {
            "name": "Decision Tree",
            "model": DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
            "scaled": False,
            "sample_weight": None,
        },
        {
            "name": "Random Forest",
            "model": RandomForestClassifier(class_weight="balanced", random_state=SEED),
            "scaled": False,
            "sample_weight": None,
        },
        {
            "name": "Gradient Boosting",
            # No class_weight parameter in the API -> balanced sample weights
            "model": GradientBoostingClassifier(random_state=SEED),
            "scaled": False,
            "sample_weight": compute_sample_weight("balanced", y_train),
        },
    ]


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=10)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_model_comparison(results: pd.DataFrame, baseline_acc: float, path: Path):
    """Grouped bars, objective vs full, for test accuracy and macro F1."""
    models = ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=SURFACE)
    metrics = [("Test Acc", "Test accuracy"), ("Macro F1", "Macro F1")]
    x = np.arange(len(models))
    width = 0.38
    for ax, (col, title) in zip(axes, metrics):
        style_axes(ax)
        obj = [
            results.loc[(results["Model"] == m) & (results["Variant"] == "Objective"), col].item()
            for m in models
        ]
        full = [
            results.loc[(results["Model"] == m) & (results["Variant"] == "Full"), col].item()
            for m in models
        ]
        ax.bar(x - width / 2, obj, width * 0.94, color=BLUE, label="Objective (deployable)")
        ax.bar(x + width / 2, full, width * 0.94, color=RED, label="Full (leakage benchmark)")
        for xi, v in zip(x - width / 2, obj):
            ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=9, color=INK)
        for xi, v in zip(x + width / 2, full):
            ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=9, color=INK)
        if col == "Test Acc":
            ax.axhline(baseline_acc, color=MUTED, linestyle="--", linewidth=1.2)
            ax.text(
                len(models) - 0.45,
                baseline_acc + 0.015,
                f"majority baseline {baseline_acc:.3f}",
                ha="right",
                fontsize=9,
                color=INK_2,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" ", "\n") for m in models], color=INK)
        ax.set_ylim(0, 1.0)
        ax.set_title(title, fontsize=12, color=INK, fontweight="bold")
    axes[0].legend(loc="upper left", frameon=False, fontsize=10, labelcolor=INK)
    fig.suptitle(
        "Objective-only vs full features: the data-leakage gap", fontsize=14, color=INK, y=1.0
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, model_name: str, path: Path):
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_RAMP)
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=SURFACE)
    im = ax.imshow(cm, cmap=cmap)
    ax.set_xticks(range(5), MOS_LABELS, color=INK_2)
    ax.set_yticks(range(5), MOS_LABELS, color=INK_2)
    threshold = cm.max() / 2
    for i in range(5):
        for j in range(5):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=11,
                color="white" if cm[i, j] > threshold else INK,
            )
    ax.set_xlabel("Predicted MOS", fontsize=11, color=INK)
    ax.set_ylabel("True MOS", fontsize=11, color=INK)
    ax.set_title(f"Confusion Matrix — {model_name} (objective features)",
                 fontsize=12, color=INK, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_feature_importance(names: list[str], importances: np.ndarray, path: Path):
    order = np.argsort(importances)[::-1][:15]
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    y = np.arange(len(order))
    ax.barh(y, importances[order], color=BLUE)
    ax.set_yticks(y, [names[i] for i in order], color=INK)
    ax.invert_yaxis()
    for yi, v in zip(y, importances[order]):
        ax.text(v + 0.002, yi, f"{v:.3f}", va="center", fontsize=9, color=INK_2)
    ax.tick_params(colors=INK_2, labelsize=10)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Feature importance", fontsize=11, color=INK)
    ax.set_title("Top 15 features — Random Forest (objective variant)",
                 fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def per_class_recall(cm: np.ndarray) -> list[str]:
    lines = []
    for i, label in enumerate(MOS_LABELS):
        total = cm[i].sum()
        lines.append(f"  MOS={i + 1} ({label}): {cm[i, i]}/{total} = {cm[i, i] / total:.1%}")
    return lines


def main():
    df = load_and_engineer()
    X_full = df.drop(columns=["MOS"])
    y = df["MOS"].to_numpy()

    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X_full, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"Sessions: {len(df)}  |  train: {len(y_train)}  |  test: {len(y_test)}")
    majority_share = float(np.mean(y == pd.Series(y).mode()[0]))
    print(f"Majority class share (whole dataset): {majority_share:.1%}\n")

    # Majority-class baseline (identical for both variants)
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train_full, y_train)
    rows = [
        {
            "Model": "Baseline (majority class)",
            "Variant": "-",
            "Train Acc": accuracy_score(y_train, baseline.predict(X_train_full)),
            "Test Acc": accuracy_score(y_test, baseline.predict(X_test_full)),
            "Macro F1": f1_score(y_test, baseline.predict(X_test_full), average="macro"),
            "Kappa": cohen_kappa_score(y_test, baseline.predict(X_test_full)),
        }
    ]
    baseline_acc = rows[0]["Test Acc"]

    variants = {
        "Objective": (
            X_train_full.drop(columns=NON_OBJECTIVE_COLS),
            X_test_full.drop(columns=NON_OBJECTIVE_COLS),
        ),
        "Full": (X_train_full, X_test_full),
    }

    confusions = {}  # (variant, model name) -> confusion matrix
    rf_importances = {}

    for variant, (X_tr, X_te) in variants.items():
        scaler = StandardScaler().fit(X_tr)
        X_tr_scaled = scaler.transform(X_tr)
        X_te_scaled = scaler.transform(X_te)
        print(f"=== Variant: {variant} ({X_tr.shape[1]} features) ===")
        for spec in build_models(y_train):
            tr = X_tr_scaled if spec["scaled"] else X_tr
            te = X_te_scaled if spec["scaled"] else X_te
            fit_kwargs = {}
            if spec["sample_weight"] is not None:
                fit_kwargs["sample_weight"] = spec["sample_weight"]
            model = spec["model"].fit(tr, y_train, **fit_kwargs)
            y_tr_pred = model.predict(tr)
            y_te_pred = model.predict(te)
            row = {
                "Model": spec["name"],
                "Variant": variant,
                "Train Acc": accuracy_score(y_train, y_tr_pred),
                "Test Acc": accuracy_score(y_test, y_te_pred),
                "Macro F1": f1_score(y_test, y_te_pred, average="macro"),
                "Kappa": cohen_kappa_score(y_test, y_te_pred),
            }
            rows.append(row)
            confusions[(variant, spec["name"])] = confusion_matrix(
                y_test, y_te_pred, labels=[1, 2, 3, 4, 5]
            )
            if spec["name"] == "Random Forest":
                rf_importances[variant] = (list(X_tr.columns), model.feature_importances_)
            print(
                f"  {spec['name']:22s} train {row['Train Acc']:.3f}  test {row['Test Acc']:.3f}"
                f"  macroF1 {row['Macro F1']:.3f}  kappa {row['Kappa']:.3f}"
            )
        print()

    results = pd.DataFrame(rows)
    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.round(4).to_csv(METRICS_CSV, index=False)
    print(f"Wrote {METRICS_CSV.relative_to(REPO_ROOT)}")

    # --- Figures ---
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_model_comparison(results, baseline_acc, FIGURES_DIR / "8_model_comparison.png")
    slug = {
        "Logistic Regression": "4_confusion_matrix_logistic_regression.png",
        "Decision Tree": "5_confusion_matrix_decision_tree.png",
        "Random Forest": "6_confusion_matrix_random_forest.png",
        "Gradient Boosting": "7_confusion_matrix_gradient_boosting.png",
    }
    for name, filename in slug.items():
        plot_confusion_matrix(confusions[("Objective", name)], name, FIGURES_DIR / filename)
    names, importances = rf_importances["Objective"]
    plot_feature_importance(names, importances, FIGURES_DIR / "6_feature_importance_rf.png")
    print(f"Wrote figures to {FIGURES_DIR.relative_to(REPO_ROOT)}\n")

    # --- Report the numbers the docs cite ---
    obj = results[results["Variant"] == "Objective"]
    full = results[results["Variant"] == "Full"]
    best_obj = obj.loc[obj["Test Acc"].idxmax()]
    best_full = full.loc[full["Test Acc"].idxmax()]
    print("=== Headline numbers ===")
    print(f"Best objective: {best_obj['Model']} {best_obj['Test Acc']:.1%}"
          f" (macro F1 {best_obj['Macro F1']:.3f}, kappa {best_obj['Kappa']:.3f})")
    print(f"Best full:      {best_full['Model']} {best_full['Test Acc']:.1%}"
          f" (macro F1 {best_full['Macro F1']:.3f}, kappa {best_full['Kappa']:.3f})")
    print(f"Leakage gap (best vs best): "
          f"{(best_full['Test Acc'] - best_obj['Test Acc']) * 100:+.1f} points")
    print(f"Best objective vs baseline: "
          f"{(best_obj['Test Acc'] - baseline_acc) * 100:+.1f} points\n")
    for m in ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]:
        o = obj.loc[obj["Model"] == m, "Test Acc"].item()
        f = full.loc[full["Model"] == m, "Test Acc"].item()
        print(f"  {m:22s} objective {o:.1%} -> full {f:.1%}  ({(f - o) * 100:+.1f} pts)")

    for variant, name in [("Objective", best_obj["Model"]), ("Full", best_full["Model"])]:
        cm = confusions[(variant, name)]
        print(f"\nConfusion matrix — {name} ({variant}):")
        print(pd.DataFrame(cm, index=MOS_LABELS, columns=MOS_LABELS).to_string())
        print("Per-class recall:")
        print("\n".join(per_class_recall(cm)))

    print("\nTop 10 feature importances — Random Forest (Objective):")
    order = np.argsort(importances)[::-1][:10]
    for rank, i in enumerate(order, 1):
        print(f"  {rank:2d}. {names[i]:25s} {importances[i]:.1%}")
    buffering = [n for n in names if "BUFFERING" in n or n == "Buffering_Severity"]
    share = sum(importances[names.index(n)] for n in buffering)
    print(f"Combined buffering importance ({', '.join(buffering)}): {share:.1%}")


if __name__ == "__main__":
    main()
