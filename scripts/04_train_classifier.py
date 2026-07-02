#!/usr/bin/env python3
"""
Train a pan-cancer molecular classifier.

Predicts cancer type (BRCA / LUAD / COAD) from:
  • Gene mutation binary matrix (top 150 cancer genes)
  • TMB (total mutation burden)
  • SBS-6 mutation type fractions (mutational process signature)
  • Mutation class fractions
  • PCA(20) of DNABERT driver-mutation embeddings  [added if embeddings exist]

Model: XGBoost (gradient boosting — handles mixed features well, robust on this scale)
Evaluation: 5-fold StratifiedKFold — accuracy, macro F1, confusion matrix
Explainability: SHAP TreeExplainer — feature importance + per-class beeswarm plots

Outputs:
  results/classification_report.txt
  results/figures/confusion_matrix.png
  results/figures/shap_beeswarm.png
  results/figures/feature_importance.png
  results/figures/umap_patients.png
  data/processed/predictions.parquet
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score,
)
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings("ignore")

PROC    = Path(__file__).parent.parent / "data" / "processed"
RESULTS = Path(__file__).parent.parent / "results"
FIGS    = RESULTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

CANCER_COLORS = {"BRCA": "#E87D72", "LUAD": "#56B4E9", "COAD": "#009E73"}


def load_data():
    features = pd.read_parquet(PROC / "features.parquet")
    labels   = pd.read_parquet(PROC / "labels.parquet")["cancer_type"]

    # Align
    common = features.index.intersection(labels.index)
    X = features.loc[common].astype(float)
    y = labels.loc[common]

    # Optionally add DNABERT embeddings — keep ALL patients, fill missing with 0
    emb_file = PROC / "patient_embeddings.npy"
    id_file  = PROC / "embedding_patient_ids.txt"
    if emb_file.exists() and id_file.exists():
        print("Loading DNABERT embeddings...")
        emb_matrix = np.load(emb_file)
        emb_pids   = id_file.read_text().strip().split("\n")
        emb_df     = pd.DataFrame(emb_matrix, index=emb_pids)

        # PCA(20) on 768-dim embeddings (fit on patients who have embeddings)
        pca = PCA(n_components=20, random_state=42)
        pca_vals = pca.fit_transform(emb_df.values)
        var_ret  = pca.explained_variance_ratio_.sum() * 100
        print(f"  PCA(20): {var_ret:.1f}% variance retained")
        pca_df = pd.DataFrame(
            pca_vals,
            index=emb_pids,
            columns=[f"dna_pc{i+1}" for i in range(20)],
        )
        # Reindex to ALL patients in X; patients without embeddings get zeros
        pca_aligned = pca_df.reindex(X.index, fill_value=0.0)
        n_with_emb = (pca_aligned.abs().sum(axis=1) > 0).sum()
        print(f"  Patients with DNABERT embeddings: {n_with_emb}/{len(X)}")
        X = X.join(pca_aligned)
    else:
        print("No DNABERT embeddings found — running tabular features only.")

    return X, y


def plot_confusion_matrix(y_true, y_pred, classes, ax):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(
        cm_norm, annot=cm, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        linewidths=0.5, ax=ax, cbar_kws={"label": "Row-normalised fraction"},
        vmin=0, vmax=1,
    )
    ax.set_xlabel("Predicted cancer type", fontsize=11)
    ax.set_ylabel("True cancer type", fontsize=11)
    ax.set_title("Confusion matrix (5-fold CV)", fontsize=12)


def plot_feature_importance(model, feature_names, top_n=25):
    imp = model.feature_importances_
    idx = np.argsort(imp)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#D55E00" if "gene_" in feature_names[i]
              else "#0072B2" if "dna_pc" in feature_names[i]
              else "#CC79A7" if "sbs6_" in feature_names[i]
              else "#999999"
              for i in idx]
    ax.barh(range(len(idx)), imp[idx], color=colors)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i].replace("gene_", "").replace("sbs6_", "SBS6 ").replace("dna_pc", "DNABERT PC") for i in idx], fontsize=9)
    ax.set_xlabel("XGBoost feature importance (gain)", fontsize=11)
    ax.set_title(f"Top {top_n} features — cancer type classifier", fontsize=12)
    patches = [
        mpatches.Patch(color="#D55E00", label="Gene mutation"),
        mpatches.Patch(color="#CC79A7", label="SBS-6 signature"),
        mpatches.Patch(color="#0072B2", label="DNABERT PC"),
        mpatches.Patch(color="#999999", label="TMB / other"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=8)
    plt.tight_layout()
    return fig


def plot_umap(X, y):
    try:
        import umap
        print("\nComputing UMAP...")
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=20)
        scaler  = StandardScaler()
        Xs = scaler.fit_transform(X.values)
        emb = reducer.fit_transform(Xs)

        fig, ax = plt.subplots(figsize=(7, 6))
        for cancer in y.unique():
            mask = (y == cancer).values
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=CANCER_COLORS.get(cancer, "gray"),
                       label=cancer, s=15, alpha=0.7, edgecolors="none")
        ax.set_xlabel("UMAP 1", fontsize=11)
        ax.set_ylabel("UMAP 2", fontsize=11)
        ax.set_title("Patient mutation profiles — UMAP projection", fontsize=12)
        ax.legend(title="Cancer type", fontsize=9)
        plt.tight_layout()
        return fig
    except ImportError:
        print("umap-learn not available; skipping UMAP plot")
        return None


def main():
    print("Loading features...")
    X, y = load_data()
    feature_names = list(X.columns)
    print(f"Feature matrix: {X.shape}  |  Classes: {y.value_counts().to_dict()}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = le.classes_

    # ── XGBoost classifier ────────────────────────────────────────────────
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    # ── 5-fold stratified CV ──────────────────────────────────────────────
    print("\nRunning 5-fold stratified CV...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X.values, y_enc, cv=cv, n_jobs=-1)
    y_pred_labels = le.inverse_transform(y_pred)

    acc = accuracy_score(y_enc, y_pred)
    f1  = f1_score(y_enc, y_pred, average="macro")
    report = classification_report(y, y_pred_labels, target_names=classes)

    print(f"\nAccuracy:  {acc:.3f}")
    print(f"Macro F1:  {f1:.3f}")
    print(f"\nClassification Report:\n{report}")

    result_text = (
        f"Pan-Cancer Molecular Classifier — Results\n"
        f"{'='*50}\n"
        f"Cancer types: {list(classes)}\n"
        f"Patients: {len(y)}\n"
        f"Features: {len(feature_names)}\n"
        f"Model: XGBoost (5-fold StratifiedKFold CV)\n\n"
        f"Accuracy:  {acc:.3f}\n"
        f"Macro F1:  {f1:.3f}\n\n"
        f"{report}"
    )
    (RESULTS / "classification_report.txt").write_text(result_text)

    # ── Fit final model on full data for SHAP ────────────────────────────
    print("\nFitting final model for SHAP...")
    model.fit(X.values, y_enc)

    # ── Confusion matrix ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_confusion_matrix(y, y_pred_labels, list(classes), ax)
    plt.tight_layout()
    fig.savefig(FIGS / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix → {FIGS}/confusion_matrix.png")

    # ── Feature importance ────────────────────────────────────────────────
    fig = plot_feature_importance(model, feature_names)
    fig.savefig(FIGS / "feature_importance.png", dpi=150)
    plt.close(fig)
    print(f"Saved feature importance → {FIGS}/feature_importance.png")

    # ── SHAP ──────────────────────────────────────────────────────────────
    print("\nComputing SHAP values...")
    explainer = shap.TreeExplainer(model)
    # Use a sample for speed (SHAP on full dataset can be slow)
    sample_idx = np.random.RandomState(42).choice(len(X), min(500, len(X)), replace=False)
    X_sample = X.values[sample_idx]
    shap_values = explainer.shap_values(X_sample)

    # shap_values: list of (n_samples, n_features) arrays for multi-class XGBoost
    # Detect whether output is list (multiclass) or single array (binary)
    if isinstance(shap_values, list):
        shap_per_class = shap_values  # one 2D array per class
    else:
        # Binary case: single 2D array → wrap in list
        shap_per_class = [shap_values]
        classes = [classes[-1]]  # positive class only

    fig, axes = plt.subplots(1, len(classes), figsize=(6 * len(classes), 6))
    if len(classes) == 1:
        axes = [axes]
    for i, (cls, ax) in enumerate(zip(classes, axes)):
        plt.sca(ax)
        shap.summary_plot(
            shap_per_class[i], X_sample,
            feature_names=feature_names,
            max_display=15,
            show=False,
            plot_type="dot",
        )
        ax.set_title(f"SHAP — {cls}", fontsize=11)
    plt.suptitle("SHAP feature importance per cancer type", y=1.01, fontsize=13)
    plt.tight_layout()
    fig.savefig(FIGS / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved SHAP → {FIGS}/shap_beeswarm.png")

    # ── UMAP ──────────────────────────────────────────────────────────────
    fig = plot_umap(X, y)
    if fig:
        fig.savefig(FIGS / "umap_patients.png", dpi=150)
        plt.close(fig)
        print(f"Saved UMAP → {FIGS}/umap_patients.png")

    # ── Save predictions ──────────────────────────────────────────────────
    pred_df = pd.DataFrame({
        "patient_id": X.index,
        "true_label": y.values,
        "pred_label": y_pred_labels,
        "correct":    y.values == y_pred_labels,
    })
    pred_df.to_parquet(PROC / "predictions.parquet", index=False)

    print(f"\nAll results saved to {RESULTS}/")
    print("Done.")


if __name__ == "__main__":
    main()
