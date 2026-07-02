#!/usr/bin/env python3
"""
GPT-4o clinical interpretation of per-patient cancer predictions.

For each patient:
  - Reports the model's predicted cancer type and confidence
  - Summarises which driver gene mutations and SBS signatures were present
  - GPT-4o generates a natural language clinical interpretation:
      • Why the mutation profile is consistent with that cancer type
      • Key driver alterations and their clinical significance
      • Relevant targeted therapy implications (where applicable)

Output:
  results/clinical_interpretations.md
  results/clinical_interpretations.json
"""

import json, os
from pathlib import Path
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict

load_dotenv()

PROC    = Path(__file__).parent.parent / "data" / "processed"
RAW     = Path(__file__).parent.parent / "data" / "raw"
RESULTS = Path(__file__).parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

GPT_MODEL = "gpt-4o-mini"

CANCER_CONTEXT = {
    "BRCA": "Breast Invasive Carcinoma. Key drivers: PIK3CA, TP53, CDH1, GATA3, MAP3K1. "
            "Subtypes: LumA (PIK3CA+), LumB (PIK3CA+TP53+), HER2+ (ERBB2), TNBC/Basal (TP53, BRCA1/2). "
            "Relevant therapies: PARP inhibitors (BRCA1/2), CDK4/6 inhibitors, HER2-targeted agents.",
    "LUAD": "Lung Adenocarcinoma. Key drivers: EGFR, KRAS, STK11, KEAP1, TP53, NF1, BRAF, MET, RET, ALK. "
            "APOBEC (C>G/C>T) and tobacco (C>A) signatures common. "
            "Relevant therapies: EGFR/ALK/RET/MET inhibitors, immunotherapy (high TMB, STK11/KEAP1 impact IO response).",
    "COAD": "Colorectal Adenocarcinoma. Key drivers: APC, TP53, KRAS, BRAF, PIK3CA, SMAD4. "
            "CpG island methylator phenotype (CIMP), microsatellite instability (MSI-H → hypermutation). "
            "Relevant therapies: EGFR antibodies (RAS/RAF WT), immunotherapy (MSI-H), MEK inhibitors (BRAF V600E).",
}

SYSTEM_PROMPT = """You are a clinical oncology AI assistant with deep expertise in cancer genomics.
You will receive a patient's somatic mutation profile and the molecular classifier's prediction.
Generate a concise, scientifically precise clinical interpretation.

Respond with a JSON object with these exact keys:
{
  "predicted_cancer_type": str,         // the predicted label
  "confidence_statement": str,          // 1 sentence on why the profile fits
  "key_drivers": [str],                 // top 3-5 driver mutations present, with clinical note
  "mutational_process": str,            // what the SBS signature pattern implies
  "therapy_implications": str,          // relevant targeted therapies or clinical flags
  "clinical_priority": "HIGH" | "MEDIUM" | "LOW"  // urgency of actionable findings
}
"""


class PatientState(TypedDict):
    patient_id: str
    profile: dict
    interpretation: dict
    error: str


def build_profile_summary(pid, mutations_df, predictions_df, top_genes):
    pred_row = predictions_df[predictions_df["patient_id"] == pid]
    if pred_row.empty:
        return None

    true_label = pred_row["true_label"].iloc[0]
    pred_label = pred_row["pred_label"].iloc[0]
    correct    = pred_row["correct"].iloc[0]

    # Patient mutations
    pmuts = mutations_df[mutations_df["patient_id"] == pid]
    tmb   = len(pmuts)
    driver_muts = pmuts[pmuts["gene"].isin([
        "TP53", "KRAS", "PIK3CA", "PTEN", "APC", "BRAF", "EGFR", "CDH1",
        "BRCA1", "BRCA2", "STK11", "KEAP1", "NF1", "SMAD4", "FBXW7",
        "ARID1A", "GATA3", "MAP3K1", "ERBB2", "MET", "ALK", "RET",
    ])]

    driver_list = []
    for _, row in driver_muts.iterrows():
        entry = f"{row['gene']} ({row['mutation_type']}"
        if row.get("protein_change") and str(row["protein_change"]) != "nan":
            entry += f", {row['protein_change']}"
        entry += ")"
        driver_list.append(entry)

    # SBS summary
    snps = pmuts[pmuts["variant_type"].str.upper().isin(["SNP", "SNV"])].copy()
    sbs_counts = {}
    sbs_map = {"C>A": ["C>A","G>T"], "C>G": ["C>G","G>C"], "C>T": ["C>T","G>A"],
               "T>A": ["T>A","A>T"], "T>C": ["T>C","A>G"], "T>G": ["T>G","A>C"]}
    for _, row in snps.iterrows():
        key = f"{row['ref_allele']}>{row['alt_allele']}".upper()
        for cat, variants in sbs_map.items():
            if key in variants:
                sbs_counts[cat] = sbs_counts.get(cat, 0) + 1
    total_snps = sum(sbs_counts.values()) or 1
    sbs_fracs  = {k: f"{v/total_snps:.1%}" for k, v in sorted(sbs_counts.items(), key=lambda x: -x[1])}

    return {
        "patient_id":   pid,
        "true_label":   true_label,
        "pred_label":   pred_label,
        "correct":      bool(correct),
        "tmb":          tmb,
        "driver_mutations": driver_list[:10],
        "sbs6_fractions":   sbs_fracs,
        "cancer_context":   CANCER_CONTEXT.get(pred_label, ""),
    }


def interpret_node(state: PatientState) -> PatientState:
    llm = ChatOpenAI(model=GPT_MODEL, temperature=0, response_format={"type": "json_object"})

    profile = state["profile"]
    user_msg = f"""
Patient: {profile['patient_id']}
True label: {profile['true_label']}
Model prediction: {profile['pred_label']}  (correct: {profile['correct']})
TMB: {profile['tmb']} somatic mutations

Driver gene mutations detected:
{chr(10).join('  - ' + m for m in profile['driver_mutations']) if profile['driver_mutations'] else '  (none in driver gene list)'}

SBS-6 signature fractions:
{chr(10).join(f'  {k}: {v}' for k, v in profile['sbs6_fractions'].items())}

Cancer type context:
{profile['cancer_context']}

Please interpret this mutation profile.
"""
    try:
        response = llm.invoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ])
        interpretation = json.loads(response.content)
        return {**state, "interpretation": interpretation}
    except Exception as e:
        return {**state, "error": str(e), "interpretation": {}}


def build_graph():
    g = StateGraph(PatientState)
    g.add_node("interpret", interpret_node)
    g.set_entry_point("interpret")
    g.add_edge("interpret", END)
    return g.compile()


def main():
    # Load data
    mutations_df   = pd.concat(
        [pd.read_csv(RAW / f"{c}_mutations.tsv", sep="\t", low_memory=False)
         for c in ["BRCA", "LUAD", "COAD"]], ignore_index=True
    )
    predictions_df = pd.read_parquet(PROC / "predictions.parquet")
    top_genes_path = PROC / "top_genes.txt"
    top_genes      = top_genes_path.read_text().strip().split("\n") if top_genes_path.exists() else []

    # Sample patients: 2 per cancer type, mix of correct + incorrect
    sample_pids = []
    for cancer in ["BRCA", "LUAD", "COAD"]:
        subset = predictions_df[predictions_df["true_label"] == cancer]
        correct   = subset[subset["correct"]].head(2)["patient_id"].tolist()
        incorrect = subset[~subset["correct"]].head(1)["patient_id"].tolist()
        sample_pids.extend(correct + incorrect)
    sample_pids = list(dict.fromkeys(sample_pids))[:12]  # max 12

    graph = build_graph()

    results    = []
    md_sections = ["# Pan-Cancer Classifier — Clinical Interpretations\n",
                   f"Model: XGBoost + DNABERT, GPT-{GPT_MODEL} clinical triage\n\n---\n"]

    for pid in sample_pids:
        profile = build_profile_summary(pid, mutations_df, predictions_df, top_genes)
        if not profile:
            continue

        print(f"Interpreting {pid} (true={profile['true_label']}, pred={profile['pred_label']}) ...")
        state = graph.invoke({
            "patient_id": pid, "profile": profile,
            "interpretation": {}, "error": "",
        })
        interp = state.get("interpretation", {})
        results.append({"patient_id": pid, "profile": profile, "interpretation": interp})

        correct_tag = "✓ CORRECT" if profile["correct"] else "✗ MISCLASSIFIED"
        md_sections.append(
            f"## Patient {pid}  ({correct_tag})\n\n"
            f"**True label:** {profile['true_label']}  |  "
            f"**Predicted:** {profile['pred_label']}  |  "
            f"**TMB:** {profile['tmb']}\n\n"
        )
        if interp:
            md_sections.append(f"**Priority:** {interp.get('clinical_priority','')}\n\n")
            md_sections.append(f"**Summary:** {interp.get('confidence_statement','')}\n\n")

            if interp.get("key_drivers"):
                md_sections.append("**Key driver alterations:**\n")
                for d in interp["key_drivers"]:
                    md_sections.append(f"- {d}\n")
                md_sections.append("\n")

            md_sections.append(f"**Mutational process:** {interp.get('mutational_process','')}\n\n")
            md_sections.append(f"**Therapy implications:** {interp.get('therapy_implications','')}\n\n")
        else:
            md_sections.append(f"*Error: {state.get('error','')}*\n\n")

        md_sections.append("---\n\n")

    # Save outputs
    (RESULTS / "clinical_interpretations.md").write_text("".join(md_sections))
    (RESULTS / "clinical_interpretations.json").write_text(
        json.dumps(results, indent=2)
    )
    print(f"\nSaved → {RESULTS}/clinical_interpretations.md")
    print("Done.")


if __name__ == "__main__":
    main()
