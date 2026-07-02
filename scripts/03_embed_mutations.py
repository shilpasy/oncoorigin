#!/usr/bin/env python3
"""
DNABERT embeddings of somatic mutations in cancer driver genes.

For each patient:
  1. Select their mutations in top DRIVER genes (TP53, KRAS, PIK3CA, etc.)
  2. Fetch 129 bp of reference sequence around each mutation from Ensembl GRCh37 REST API
  3. Embed each sequence with DNABERT-1 (zhihan1996/DNA_bert_6, 6-mer tokenisation)
  4. Mean-pool across all driver mutations → single 768-dim patient embedding
  5. PCA(20) across all patients

The embedding captures WHERE in the genome the mutations occur and the local sequence
grammar (GC content, repeat context, splice proximity) — information not present in the
gene-mutation binary matrix or SBS fractions.

Output:
  data/processed/patient_embeddings.npy   — (n_patients, 768)
  data/processed/embedding_patient_ids.txt
"""

import warnings, time
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import requests
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

RAW  = Path(__file__).parent.parent / "data" / "raw"
PROC = Path(__file__).parent.parent / "data" / "processed"
PROC.mkdir(exist_ok=True)

ENSEMBL_API = "https://grch37.rest.ensembl.org"  # GRCh37 / hg19 (TCGA coordinates)
MODEL_NAME  = "zhihan1996/DNA_bert_6"
WINDOW      = 64   # bp each side → 129 bp total
KMER        = 6
BATCH_SIZE  = 32
MAX_MUTS_PER_PATIENT = 30  # cap to keep API calls manageable

# Top cancer driver genes — most clinically interpreted, most discriminating
DRIVER_GENES = [
    "TP53", "KRAS", "PIK3CA", "PTEN", "APC", "BRAF", "EGFR", "CDH1",
    "CDKN2A", "RB1", "BRCA1", "BRCA2", "STK11", "KEAP1", "NF1",
    "SMAD4", "FBXW7", "ARID1A", "CTNNB1", "IDH1", "IDH2", "ERBB2",
    "MET", "RET", "ALK", "NRAS", "HRAS", "MAP2K1", "MAP2K2",
    "GATA3", "MAP3K1", "RUNX1", "PTPN11",
]


# ── Ensembl sequence fetching ─────────────────────────────────────────────────

def fetch_sequences_batch(positions, window=WINDOW, batch_size=50):
    """
    Fetch multiple genomic sequences in one POST request to Ensembl batch API.
    positions: list of (chrom, pos) tuples (1-based coordinates, GRCh37).
    Returns dict: (chrom, pos) → sequence string.
    """
    results = {}
    for i in range(0, len(positions), batch_size):
        batch = positions[i:i+batch_size]
        regions = []
        for chrom, pos in batch:
            chrom_str = str(chrom).replace("chr", "").replace("23", "X").replace("24", "Y")
            start = max(1, int(pos) - window)
            end   = int(pos) + window
            regions.append(f"{chrom_str}:{start}..{end}:1")

        for attempt in range(3):
            try:
                r = requests.post(
                    f"{ENSEMBL_API}/sequence/region/human",
                    json={"regions": regions},
                    headers={"Content-Type": "application/json", "Accept": "application/json"},
                    timeout=60,
                )
                if r.status_code == 200:
                    seqs_json = r.json()
                    for item, (chrom, pos) in zip(seqs_json, batch):
                        seq = item.get("seq", "").upper()
                        seq = "".join(c if c in "ACGT" else "N" for c in seq)
                        if len(seq) >= 12:
                            results[(chrom, pos)] = seq
                    break
                time.sleep(1)
            except Exception as e:
                time.sleep(2)
        time.sleep(0.1)  # gentle rate limiting between batches
    return results


def seq_to_kmers(seq, k=KMER):
    return " ".join(seq[i:i+k] for i in range(len(seq) - k + 1))


# ── DNABERT embedding ─────────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def embed_sequences(seqs, tokenizer, model, device):
    kmer_seqs = [seq_to_kmers(s) for s in seqs]
    all_emb = []
    for i in range(0, len(kmer_seqs), BATCH_SIZE):
        batch = kmer_seqs[i:i+BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        cls = out.last_hidden_state[:, 0, :].cpu().numpy()
        all_emb.append(cls)
    return np.vstack(all_emb) if all_emb else np.zeros((0, 768))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load mutations
    dfs = []
    for cancer in ["BRCA", "LUAD", "COAD"]:
        f = RAW / f"{cancer}_mutations.tsv"
        df = pd.read_csv(f, sep="\t", low_memory=False)
        dfs.append(df)
    mutations = pd.concat(dfs, ignore_index=True)

    # Filter to driver gene mutations only
    driver_muts = mutations[mutations["gene"].isin(DRIVER_GENES)].copy()
    print(f"Driver gene mutations: {len(driver_muts):,} across {driver_muts['patient_id'].nunique()} patients")

    # Only keep SNPs (chromosomal coords reliable for sequence fetch)
    driver_muts = driver_muts[driver_muts["variant_type"].str.upper().isin(["SNP", "SNV"])]
    driver_muts = driver_muts.dropna(subset=["chromosome", "start_pos"])
    driver_muts["chromosome"] = driver_muts["chromosome"].astype(str)
    driver_muts["start_pos"] = driver_muts["start_pos"].astype(int)

    # Fetch unique positions from Ensembl in batches of 50 (single POST per batch)
    unique_positions = driver_muts[["chromosome", "start_pos"]].drop_duplicates()
    pos_list = list(zip(unique_positions["chromosome"], unique_positions["start_pos"]))
    print(f"\nFetching sequences for {len(pos_list)} unique positions from Ensembl GRCh37 (batch mode)...")

    pos_to_seq = fetch_sequences_batch(pos_list)
    print(f"Sequences retrieved: {len(pos_to_seq)} / {len(pos_list)}")

    # Load DNABERT
    print(f"\nLoading DNABERT ({MODEL_NAME})...")
    device = get_device()
    print(f"Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # Per-patient embedding: mean-pool over their driver mutation embeddings
    patients = driver_muts["patient_id"].unique()
    patient_embeddings = {}

    print(f"\nEmbedding {len(patients)} patients...")
    for pid in tqdm(patients):
        pmuts = driver_muts[driver_muts["patient_id"] == pid]
        # Cap per patient to keep tractable
        if len(pmuts) > MAX_MUTS_PER_PATIENT:
            pmuts = pmuts.head(MAX_MUTS_PER_PATIENT)

        seqs = []
        for _, row in pmuts.iterrows():
            key = (row["chromosome"], row["start_pos"])
            if key in pos_to_seq:
                seqs.append(pos_to_seq[key])

        if not seqs:
            continue

        embs = embed_sequences(seqs, tokenizer, model, device)
        patient_embeddings[pid] = embs.mean(axis=0)  # mean pool

    # Build output arrays
    pids = list(patient_embeddings.keys())
    emb_matrix = np.stack([patient_embeddings[p] for p in pids])

    print(f"\nEmbedding matrix: {emb_matrix.shape}")
    np.save(PROC / "patient_embeddings.npy", emb_matrix)
    (PROC / "embedding_patient_ids.txt").write_text("\n".join(pids))

    print(f"Saved → {PROC}/patient_embeddings.npy")
    print("Done.")


if __name__ == "__main__":
    main()
