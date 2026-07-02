#!/usr/bin/env python3
"""
Build per-patient feature matrix from mutation data.

Features:
  1. Gene mutation binary matrix  — top 150 most-mutated cancer genes (0/1 per patient)
  2. TMB                          — total somatic mutation burden per patient
  3. SBS-6 mutation type fractions — C>A, C>G, C>T, T>A, T>C, T>G proportions
  4. Mutation class fractions      — missense, nonsense, frameshift, splice, etc.

Output:
  data/processed/features.parquet
  data/processed/labels.parquet
  data/processed/top_genes.txt
"""

from pathlib import Path
import pandas as pd
import numpy as np

RAW  = Path(__file__).parent.parent / "data" / "raw"
PROC = Path(__file__).parent.parent / "data" / "processed"
PROC.mkdir(exist_ok=True)

# Mutation type → SBS-6 category
SBS6_MAP = {
    "C>A": ["C>A", "G>T"],
    "C>G": ["C>G", "G>C"],
    "C>T": ["C>T", "G>A"],
    "T>A": ["T>A", "A>T"],
    "T>C": ["T>C", "A>G"],
    "T>G": ["T>G", "A>C"],
}
def to_sbs6(ref, alt):
    key = f"{ref}>{alt}".upper()
    for cat, variants in SBS6_MAP.items():
        if key in variants:
            return cat
    return "other"

MUTATION_CLASSES = [
    "Missense_Mutation", "Nonsense_Mutation",
    "Frame_Shift_Del", "Frame_Shift_Ins",
    "Splice_Site", "In_Frame_Del", "In_Frame_Ins",
    "Translation_Start_Site", "Nonstop_Mutation",
]


def load_mutations():
    import json
    # Load entrez → gene symbol lookup built from cBioPortal genes API
    lookup_path = RAW / "entrez_to_gene.json"
    gene_lookup = {}
    if lookup_path.exists():
        raw_map = json.loads(lookup_path.read_text())
        gene_lookup = {int(k): v for k, v in raw_map.items()}

    dfs = []
    for cancer in ["BRCA", "LUAD", "COAD"]:
        f = RAW / f"{cancer}_mutations.tsv"
        if not f.exists():
            raise FileNotFoundError(f"Run 01_download_data.py first: {f}")
        df = pd.read_csv(f, sep="\t", low_memory=False)
        # Populate gene column from entrez_id lookup where gene is missing
        if gene_lookup:
            missing = df["gene"].isna() | (df["gene"] == "")
            df.loc[missing, "gene"] = df.loc[missing, "entrez_id"].map(
                lambda x: gene_lookup.get(int(x)) if pd.notna(x) and x > 0 else np.nan
            )
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def build_features(df):
    print(f"Total mutations: {len(df):,}")
    print(f"Unique patients: {df['patient_id'].nunique()}")

    # ── 1. Gene mutation binary matrix ────────────────────────────────────
    # Select top N genes by mutation frequency (most patients affected)
    gene_counts = df.groupby("gene")["patient_id"].nunique().sort_values(ascending=False)
    top_genes = gene_counts.head(150).index.tolist()
    print(f"Top genes selected: {len(top_genes)}")

    # One row per patient; 1 if gene mutated in that patient
    snv_df = df[df["gene"].isin(top_genes)].copy()
    gene_matrix = (
        snv_df.groupby(["patient_id", "gene"])
        .size()
        .unstack(fill_value=0)
        .clip(upper=1)   # binarize
        .reindex(columns=top_genes, fill_value=0)
    )
    gene_matrix.columns = [f"gene_{g}" for g in gene_matrix.columns]

    # ── 2. TMB ────────────────────────────────────────────────────────────
    tmb = df.groupby("patient_id").size().rename("tmb")

    # ── 3. SBS-6 mutation type fractions ──────────────────────────────────
    # Only for SNPs (single base substitutions)
    snp_df = df[df["variant_type"].str.upper().isin(["SNP", "SNV"])].copy()
    snp_df["sbs6"] = snp_df.apply(
        lambda r: to_sbs6(str(r["ref_allele"]), str(r["alt_allele"])), axis=1
    )
    sbs6_counts = (
        snp_df.groupby(["patient_id", "sbs6"])
        .size()
        .unstack(fill_value=0)
    )
    # Normalize to fractions
    sbs6_fracs = sbs6_counts.div(sbs6_counts.sum(axis=1), axis=0).fillna(0)
    sbs6_fracs.columns = [f"sbs6_{c}" for c in sbs6_fracs.columns]

    # ── 4. Mutation class fractions ───────────────────────────────────────
    mc = df.copy()
    mc["mut_class"] = mc["mutation_type"].where(
        mc["mutation_type"].isin(MUTATION_CLASSES), "Other"
    )
    mc_counts = (
        mc.groupby(["patient_id", "mut_class"])
        .size()
        .unstack(fill_value=0)
    )
    mc_fracs = mc_counts.div(mc_counts.sum(axis=1), axis=0).fillna(0)
    mc_fracs.columns = [f"mutcls_{c}" for c in mc_fracs.columns]

    # ── Cancer type label ─────────────────────────────────────────────────
    labels = df.groupby("patient_id")["cancer_type"].first()

    # ── Combine all features ──────────────────────────────────────────────
    features = (
        gene_matrix
        .join(tmb, how="outer")
        .join(sbs6_fracs, how="outer")
        .join(mc_fracs, how="outer")
        .fillna(0)
    )

    # Align to patients with labels
    common = features.index.intersection(labels.index)
    features = features.loc[common]
    labels = labels.loc[common]

    print(f"\nFeature matrix: {features.shape}  (patients × features)")
    print(f"Label distribution:\n{labels.value_counts()}")

    return features, labels, top_genes


def main():
    df = load_mutations()
    features, labels, top_genes = build_features(df)

    features.to_parquet(PROC / "features.parquet")
    labels.to_frame("cancer_type").to_parquet(PROC / "labels.parquet")
    (PROC / "top_genes.txt").write_text("\n".join(top_genes))

    print(f"\nSaved → {PROC}/features.parquet")
    print(f"Saved → {PROC}/labels.parquet")
    print("Done.")


if __name__ == "__main__":
    main()
