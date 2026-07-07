#!/usr/bin/env python3
"""
08_tumor_gps.py — the "Tumor GPS" CUP hero card.

Simulates the Cancer-of-Unknown-Primary use case: a tumour arrives with its
primary site UNKNOWN. We feed only its somatic mutation profile to MutaTrace
and read back a probability over tissues of origin — then hand the call to the
GPT-4o layer for a clinical rationale. Three cases: two confident calls and one
hard case where the model hedges and the LLM supplies the missing knowledge.

Output:
  results/figures/tumor_gps_cards.png
"""

import json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import xgboost as xgb
import textwrap

PROC    = Path(__file__).parent.parent / "data" / "processed"
RESULTS = Path(__file__).parent.parent / "results"
FIGS    = RESULTS / "figures"

CANCER_COLORS = {"BRCA": "#E87D72", "LUAD": "#56B4E9", "COAD": "#009E73"}
TISSUE_NAME   = {"BRCA": "Breast", "LUAD": "Lung", "COAD": "Colorectal"}

# Three demonstrative patients (must exist in clinical_interpretations.json)
CASES = ["TCGA-05-4244", "TCGA-3L-AA1B", "TCGA-A1-A0SH"]


def load_Xy():
    features = pd.read_parquet(PROC / "features.parquet")
    labels   = pd.read_parquet(PROC / "labels.parquet")["cancer_type"]
    common = features.index.intersection(labels.index)
    X = features.loc[common].astype(float)
    y = labels.loc[common]

    emb_file = PROC / "patient_embeddings.npy"
    id_file  = PROC / "embedding_patient_ids.txt"
    if emb_file.exists():
        emb = np.load(emb_file)
        pids = id_file.read_text().strip().split("\n")
        emb_df = pd.DataFrame(emb, index=pids)
        pca = PCA(n_components=20, random_state=42)
        vals = pca.fit_transform(emb_df.values)
        pca_df = pd.DataFrame(vals, index=pids,
                              columns=[f"dna_pc{i+1}" for i in range(20)])
        X = X.join(pca_df.reindex(X.index, fill_value=0.0))
    return X, y


def draw_card(ax, case, X, classes, proba_row, interp, profile):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    pred_idx = int(np.argmax(proba_row))
    pred = classes[pred_idx]
    pred_color = CANCER_COLORS[pred]
    conf = proba_row[pred_idx]

    # Card background
    ax.add_patch(FancyBboxPatch((0.15, 0.15), 9.7, 9.7,
                 boxstyle="round,pad=0.02,rounding_size=0.35",
                 fc="white", ec="#E2E2E2", lw=1.4, zorder=1))
    # Accent header strip in predicted-tissue colour
    ax.add_patch(FancyBboxPatch((0.15, 8.55), 9.7, 1.3,
                 boxstyle="round,pad=0.02,rounding_size=0.35",
                 fc=pred_color, ec="none", zorder=2))
    ax.add_patch(plt.Rectangle((0.17, 8.55), 9.66, 0.55, fc=pred_color, ec="none", zorder=2))

    ax.text(0.55, 9.35, "PRIMARY SITE:  UNKNOWN", fontsize=11.5, fontweight="bold",
            color="white", va="center", zorder=3)
    ax.text(0.55, 8.85, f"{case}", fontsize=9.5, color="white", alpha=0.92,
            va="center", zorder=3, family="monospace")

    # Detected drivers
    ax.text(0.55, 8.0, "Detected driver mutations", fontsize=9, fontweight="bold",
            color="#444", va="center")
    drivers = profile.get("driver_mutations", [])[:4]
    if not drivers:
        drivers = ["(no canonical driver SNV)"]
    y0 = 7.5
    for d in drivers:
        short = d.replace("_Mutation", "").replace("Frame_Shift", "FS")
        short = textwrap.shorten(short, width=42, placeholder="…")
        ax.add_patch(FancyBboxPatch((0.55, y0-0.24), 8.9, 0.44,
                     boxstyle="round,pad=0.02,rounding_size=0.12",
                     fc="#F1F5F9", ec="#DBE4EC", lw=0.8))
        ax.text(0.75, y0-0.02, short, fontsize=8.4, color="#1F2D3A", va="center",
                family="monospace")
        y0 -= 0.56
    ax.text(0.55, y0+0.05, f"Tumour mutational burden:  {profile.get('tmb','?')} mutations",
            fontsize=8.3, color="#666", va="center")

    # Probability bars
    bar_top = y0 - 0.55
    ax.text(0.55, bar_top, "MutaTrace — tissue-of-origin probability",
            fontsize=9, fontweight="bold", color="#444", va="center")
    order = np.argsort(proba_row)[::-1]
    by = bar_top - 0.55
    for idx in order:
        c = classes[idx]; p = proba_row[idx]
        col = CANCER_COLORS[c]
        is_pred = (idx == pred_idx)
        ax.text(0.55, by, TISSUE_NAME[c], fontsize=8.8,
                fontweight="bold" if is_pred else "normal",
                color="#222" if is_pred else "#888", va="center")
        # track
        ax.add_patch(FancyBboxPatch((2.7, by-0.17), 6.0, 0.34,
                     boxstyle="round,pad=0,rounding_size=0.14",
                     fc="#EEF1F4", ec="none"))
        # fill
        w = max(0.001, 6.0 * p)
        ax.add_patch(FancyBboxPatch((2.7, by-0.17), w, 0.34,
                     boxstyle="round,pad=0,rounding_size=0.14",
                     fc=col, ec="none", alpha=1.0 if is_pred else 0.5))
        ax.text(8.85, by, f"{p*100:.0f}%", fontsize=8.8,
                fontweight="bold" if is_pred else "normal",
                color=col if is_pred else "#999", va="center", ha="left")
        by -= 0.62

    # Verdict line
    verdict_y = by - 0.05
    ax.text(0.55, verdict_y, "MutaTrace infers:", fontsize=8.6, color="#666", va="center")
    ax.text(3.0, verdict_y, f"{TISSUE_NAME[pred]}  ({conf*100:.0f}%)",
            fontsize=11, fontweight="bold", color=pred_color, va="center")

    # GPT-4o rationale
    rat = interp.get("confidence_statement", "")
    ax.add_patch(FancyBboxPatch((0.55, 0.5), 8.9, verdict_y-0.9,
                 boxstyle="round,pad=0.02,rounding_size=0.12",
                 fc="#FAFBF4", ec="#E7EBD3", lw=0.9))
    ax.text(0.75, verdict_y-0.85, "◆ GPT-4o clinical read", fontsize=7.8,
            fontweight="bold", color="#7A8B2F", va="top")
    wrapped = textwrap.fill(rat, width=64)
    ax.text(0.75, verdict_y-1.25, wrapped, fontsize=7.9, color="#3A3F2C",
            va="top", linespacing=1.35)


def main():
    X, y = load_Xy()
    le = LabelEncoder(); y_enc = le.fit_transform(y)
    classes = list(le.classes_)

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
        random_state=42, n_jobs=-1)

    # Leak-free: out-of-fold probabilities (each patient scored by a model that
    # never saw them in training). This is the honest held-out prediction —
    # NOT a full-data refit, which would memorise the demo patients.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_proba = cross_val_predict(model, X.values, y_enc, cv=cv,
                                  method="predict_proba", n_jobs=-1)
    proba_by_pid = {pid: oof_proba[i] for i, pid in enumerate(X.index)}

    interp_data = {r["patient_id"]: r for r in
                   json.loads((RESULTS / "clinical_interpretations.json").read_text())}

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 7.2))
    for ax, case in zip(axes, CASES):
        proba = proba_by_pid[case]
        rec = interp_data.get(case, {})
        draw_card(ax, case, X, classes, proba,
                  rec.get("interpretation", {}), rec.get("profile", {}))

    fig.suptitle("Tumor GPS — inferring a cancer's tissue of origin from its DNA alone",
                 fontsize=16, fontweight="bold", y=1.0)
    fig.text(0.5, 0.02,
             "The Cancer-of-Unknown-Primary problem, simulated: primary site withheld, "
             "only somatic mutations given. Right-hand card is the hard case — the model "
             "hedges, and the GPT-4o layer flags the BRCA1 truncation the tabular model under-weighted.",
             ha="center", fontsize=9, color="#666")
    plt.tight_layout(rect=[0.01, 0.04, 0.99, 0.95])
    out = FIGS / "tumor_gps_cards.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {out}")


if __name__ == "__main__":
    main()
