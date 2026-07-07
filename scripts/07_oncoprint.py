#!/usr/bin/env python3
"""
07_oncoprint.py — the classic gene x patient mutation "waterfall".

An oncoprint is the single most recognisable cohort-genomics visual: driver
genes as rows, patients as columns, cells coloured by mutation class, sorted so
the mutations cascade. It shows at a glance that each cancer type has its own
driver architecture — APC in colorectal, PIK3CA/GATA3 in breast, EGFR/STK11 in
lung — which is exactly the signal MutaTrace's classifier exploits.

Output:
  results/figures/oncoprint.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb

RAW  = Path(__file__).parent.parent / "data" / "raw"
FIGS = Path(__file__).parent.parent / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

CANCERS = ["BRCA", "LUAD", "COAD"]
CANCER_COLORS = {"BRCA": "#E87D72", "LUAD": "#56B4E9", "COAD": "#009E73"}
CANCER_FULL   = {"BRCA": "Breast", "LUAD": "Lung adeno", "COAD": "Colorectal"}

# Driver genes to display (rows), curated for cross-cancer contrast.
# Deliberately excludes TTN / MUC16 — long-gene artefacts that score as
# false-positive "drivers" by mutation count; a curated panel signals real genes.
DISPLAY_GENES = [
    "TP53", "PIK3CA", "APC", "KRAS", "GATA3", "CDH1", "ERBB2", "BRCA2",
    "PTEN", "SMAD4", "FBXW7", "BRAF", "EGFR", "STK11", "KEAP1", "NF1",
    "ARID1A", "MAP3K1", "RB1", "CTNNB1",
]

# Mutation-class → colour (truncating = the damaging class, shown darkest)
CLASS_COLOR = {
    "Missense":    "#2C8C5A",  # green
    "Truncating":  "#141414",  # near-black (nonsense/frameshift/splice)
    "Inframe":     "#7B4FA3",  # purple
    "Other":       "#C4C4C4",  # light grey
}
def classify(mt):
    mt = str(mt)
    if mt == "Missense_Mutation":
        return "Missense"
    if mt in ("Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
              "Splice_Site", "Nonstop_Mutation", "Translation_Start_Site"):
        return "Truncating"
    if mt in ("In_Frame_Del", "In_Frame_Ins"):
        return "Inframe"
    return "Other"


def main():
    frames = []
    for c in CANCERS:
        df = pd.read_csv(RAW / f"{c}_mutations.tsv", sep="\t", low_memory=False)
        df["cancer_type"] = c
        frames.append(df)
    muts = pd.concat(frames, ignore_index=True)
    muts = muts[muts["gene"].isin(DISPLAY_GENES)].copy()
    muts["mclass"] = muts["mutation_type"].map(classify)

    # Patient → cancer type
    pat_cancer = (pd.concat(frames)[["patient_id", "cancer_type"]]
                  .drop_duplicates("patient_id").set_index("patient_id")["cancer_type"])
    all_patients = pat_cancer.index.tolist()

    # Per (patient, gene) dominant class: truncating > missense > inframe > other
    prio = {"Truncating": 3, "Missense": 2, "Inframe": 1, "Other": 0}
    muts["prio"] = muts["mclass"].map(prio)
    dom = (muts.sort_values("prio", ascending=False)
                .drop_duplicates(["patient_id", "gene"])
                .set_index(["patient_id", "gene"])["mclass"])

    # Order genes by overall mutation frequency (rows top→bottom)
    gene_freq = muts.groupby("gene")["patient_id"].nunique()
    genes = [g for g in DISPLAY_GENES if g in gene_freq.index]
    genes = sorted(genes, key=lambda g: gene_freq.get(g, 0), reverse=True)

    # Build binary matrix for the memo-sort (genes x patients)
    mutated = {(p, g): (p, g) in dom.index for p in all_patients for g in genes}

    # Sort patients: group by cancer type (BRCA, LUAD, COAD), then memo-sort within
    def memo_key(p):
        # big-endian across genes (top gene = highest weight) → waterfall cascade
        bits = 0
        for g in genes:
            bits = (bits << 1) | (1 if mutated[(p, g)] else 0)
        return bits
    ordered_patients = []
    group_bounds = []
    start = 0
    for c in CANCERS:
        grp = [p for p in all_patients if pat_cancer[p] == c]
        grp.sort(key=memo_key, reverse=True)
        ordered_patients.extend(grp)
        group_bounds.append((c, start, start + len(grp)))
        start += len(grp)

    n_p = len(ordered_patients)
    n_g = len(genes)
    p_index = {p: i for i, p in enumerate(ordered_patients)}

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(15, 7.5))
    bg = "#F4F4F4"
    ax.add_patch(plt.Rectangle((0, 0), n_p, n_g, color=bg, zorder=0))

    # draw mutated cells as thin vertical ticks
    for (p, g), mclass in dom.items():
        if g not in genes or p not in p_index:
            continue
        xi = p_index[p]
        yi = n_g - 1 - genes.index(g)
        ax.add_patch(plt.Rectangle((xi, yi + 0.08), 1.0, 0.84,
                                   color=CLASS_COLOR[mclass], linewidth=0, zorder=2))

    ax.set_xlim(0, n_p)
    ax.set_ylim(0, n_g)

    # Gene labels + % mutated on the right
    for i, g in enumerate(genes):
        yi = n_g - 1 - i
        pct = 100 * gene_freq.get(g, 0) / n_p
        ax.text(-6, yi + 0.5, g, ha="right", va="center", fontsize=10, fontweight="bold")
        ax.text(n_p + 6, yi + 0.5, f"{pct:.0f}%", ha="left", va="center", fontsize=9, color="#555")

    # Cancer-type annotation bar above the grid
    bar_h = n_g * 0.05
    for c, s, e in group_bounds:
        ax.add_patch(plt.Rectangle((s, n_g + 0.15), e - s, bar_h,
                                   color=CANCER_COLORS[c], clip_on=False, zorder=3))
        ax.text((s + e) / 2, n_g + 0.15 + bar_h + 0.25,
                f"{CANCER_FULL[c]}  (n={e-s})", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=CANCER_COLORS[c], clip_on=False)
        # faint divider
        if s > 0:
            ax.axvline(s, color="white", lw=1.5, zorder=4)

    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Legend
    legend_el = [Patch(facecolor=CLASS_COLOR[k], label=k) for k in
                 ["Truncating", "Missense", "Inframe", "Other"]]
    ax.legend(handles=legend_el, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=4, frameon=False, fontsize=10, title="Mutation class",
              title_fontsize=10)

    ax.set_title("Driver-gene architecture across 2,098 tumours — each cancer type has its own signature",
                 fontsize=14, fontweight="bold", pad=42)
    fig.text(0.5, 0.02,
             "Columns = patients (sorted into a waterfall within each cancer type). "
             "Rows = driver genes, right-labelled with cohort mutation frequency.",
             ha="center", fontsize=9, color="#555")

    plt.tight_layout(rect=[0.02, 0.04, 0.98, 1])
    out = FIGS / "oncoprint.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {out}")


if __name__ == "__main__":
    main()
