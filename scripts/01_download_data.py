#!/usr/bin/env python3
"""
Download mutation + clinical data for 3 TCGA cancer types from cBioPortal REST API.

Studies:
  - brca_tcga_pan_can_atlas_2018  (Breast Invasive Carcinoma, ~1000 patients)
  - luad_tcga_pan_can_atlas_2018  (Lung Adenocarcinoma, ~500 patients)
  - coadread_tcga_pan_can_atlas_2018 (Colorectal Adenocarcinoma, ~400 patients)

Outputs:
  data/raw/{study}_mutations.tsv   — one row per mutation
  data/raw/{study}_clinical.tsv    — one row per patient (survival, subtype)
"""

import requests, json, time
from pathlib import Path
import pandas as pd

BASE = "https://www.cbioportal.org/api"
OUT  = Path(__file__).parent.parent / "data" / "raw"
OUT.mkdir(parents=True, exist_ok=True)

STUDIES = {
    "BRCA": "brca_tcga_pan_can_atlas_2018",
    "LUAD": "luad_tcga_pan_can_atlas_2018",
    "COAD": "coadread_tcga_pan_can_atlas_2018",
}

HEADERS = {"Accept": "application/json"}


def get_json(url, params=None, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=120)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"  Retry {attempt+1}: {e}")
            time.sleep(5)
    raise RuntimeError(f"Failed after {retries} retries: {url}")


def post_json(url, payload, retries=3):
    for attempt in range(retries):
        try:
            r = requests.post(url, json=payload, headers=HEADERS, timeout=180)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"  Retry {attempt+1}: {e}")
            time.sleep(5)
    raise RuntimeError(f"POST failed after {retries} retries: {url}")


def fetch_mutations(study_id, cancer_label):
    print(f"\n{'='*60}")
    print(f"Fetching mutations for {cancer_label} ({study_id})")

    profile_id = f"{study_id}_mutations"

    # Get all sample IDs in the study
    samples = get_json(f"{BASE}/studies/{study_id}/samples", params={"projection": "ID"})
    sample_ids = [s["sampleId"] for s in samples]
    print(f"  {len(sample_ids)} samples")

    # Fetch mutations in batches of 500 samples
    all_mutations = []
    batch_size = 500
    for i in range(0, len(sample_ids), batch_size):
        batch = sample_ids[i:i+batch_size]
        payload = {
            "sampleMolecularIdentifiers": [
                {"molecularProfileId": profile_id, "sampleId": sid}
                for sid in batch
            ]
        }
        muts = post_json(f"{BASE}/mutations/fetch?projection=DETAILED", payload)
        all_mutations.extend(muts)
        print(f"  Batch {i//batch_size + 1}: {len(muts)} mutations (total so far: {len(all_mutations)})")
        time.sleep(0.5)

    if not all_mutations:
        print(f"  WARNING: No mutations returned for {cancer_label}")
        return pd.DataFrame()

    rows = []
    for m in all_mutations:
        gene = m.get("gene", {})
        rows.append({
            "sample_id":    m.get("sampleId", ""),
            "patient_id":   m.get("patientId", ""),
            "gene":         gene.get("hugoGeneSymbol", ""),
            "entrez_id":    gene.get("entrezGeneId", ""),
            "chromosome":   m.get("chr", ""),
            "start_pos":    m.get("startPosition", ""),
            "end_pos":      m.get("endPosition", ""),
            "ref_allele":   m.get("referenceAllele", ""),
            "alt_allele":   m.get("variantAllele", ""),
            "variant_type": m.get("variantType", ""),
            "mutation_type": m.get("mutationType", ""),
            "protein_change": m.get("proteinChange", ""),
            "vaf":          m.get("tumorAltCount", 0) / max(m.get("tumorRefCount", 1) + m.get("tumorAltCount", 1), 1),
            "cancer_type":  cancer_label,
        })

    df = pd.DataFrame(rows)
    outfile = OUT / f"{cancer_label}_mutations.tsv"
    df.to_csv(outfile, sep="\t", index=False)
    print(f"  Saved {len(df)} mutations → {outfile}")
    return df


def fetch_clinical(study_id, cancer_label):
    print(f"\nFetching clinical data for {cancer_label}...")
    patients = get_json(f"{BASE}/studies/{study_id}/patients", params={"projection": "DETAILED"})

    rows = []
    for p in patients:
        clin = {item["clinicalAttributeId"]: item["value"]
                for item in p.get("clinicalData", [])}
        rows.append({
            "patient_id":       p.get("patientId", ""),
            "cancer_type":      cancer_label,
            "os_months":        clin.get("OS_MONTHS", ""),
            "os_status":        clin.get("OS_STATUS", ""),
            "dfs_months":       clin.get("DFS_MONTHS", ""),
            "dfs_status":       clin.get("DFS_STATUS", ""),
            "stage":            clin.get("AJCC_PATHOLOGIC_TUMOR_STAGE", clin.get("TUMOR_STAGE", "")),
            "subtype":          clin.get("SUBTYPE", clin.get("CANCER_TYPE_DETAILED", "")),
        })

    df = pd.DataFrame(rows)
    outfile = OUT / f"{cancer_label}_clinical.tsv"
    df.to_csv(outfile, sep="\t", index=False)
    print(f"  Saved {len(df)} patients → {outfile}")
    return df


def main():
    for cancer_label, study_id in STUDIES.items():
        fetch_mutations(study_id, cancer_label)
        fetch_clinical(study_id, cancer_label)
    print("\nAll downloads complete.")


if __name__ == "__main__":
    main()
