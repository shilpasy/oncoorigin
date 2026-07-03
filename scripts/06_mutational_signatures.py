#!/usr/bin/env python3
"""
06_mutational_signatures.py — the iconic SBS-96 mutational signature "barcode".

Every tumor records what caused it in the trinucleotide context of its point
mutations. Tobacco smoke leaves a C>A-heavy signature (SBS4); APOBEC enzymes
leave C>T/C>G at TpC (SBS2/13); an ageing clock leaves C>T at CpG (SBS1).
This is the forensic fingerprint at the heart of OncoOrigin.

For each cancer type:
  1. Sample up to N clean single-base SNVs (seeded, reproducible).
  2. Fetch the +/-1 bp trinucleotide context from Ensembl GRCh37 (batch REST).
  3. Collapse to the pyrimidine strand (C/T reference) — the SBS-96 convention.
  4. Tally the 96 channels (6 substitutions x 4 5'-flanks x 4 3'-flanks).
  5. Render the standard 96-bar spectrum with the community-standard 6 colours.

Output:
  results/figures/mutational_signatures.png
  data/processed/sbs96_matrix.tsv   (96 channels x cancer types, fractions)
"""

import warnings, time
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

RAW  = Path(__file__).parent.parent / "data" / "raw"
PROC = Path(__file__).parent.parent / "data" / "processed"
FIGS = Path(__file__).parent.parent / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

ENSEMBL_API = "https://grch37.rest.ensembl.org"
CANCERS     = ["BRCA", "LUAD", "COAD"]
CANCER_FULL = {"BRCA": "Breast (BRCA)", "LUAD": "Lung adeno (LUAD)", "COAD": "Colorectal (COAD)"}
SAMPLE_N    = 8000      # SNVs sampled per cancer type for the spectrum
SEED        = 42

# Community-standard SBS substitution colours (COSMIC / SigProfiler convention)
SBS_COLORS = {
    "C>A": "#1EBFF0",  # sky blue
    "C>G": "#050708",  # black
    "C>T": "#E62725",  # red
    "T>A": "#CBCACB",  # grey
    "T>C": "#A1CF64",  # green
    "T>G": "#EDC8C5",  # pink
}
SUBS  = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
BASES = ["A", "C", "G", "T"]
COMP  = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}


def revcomp(seq):
    return "".join(COMP[b] for b in reversed(seq))


def channel_order():
    """The canonical 96 channels: for each substitution, 4x4 flanks in ACGT order."""
    order = []
    for sub in SUBS:
        ref = sub[0]
        for five in BASES:
            for three in BASES:
                order.append(f"{five}[{sub}]{three}")
    return order


def fetch_context_batch(positions, batch_size=50):
    """positions: list of (chrom, pos). Returns dict (chrom,pos) -> 3bp string (5',ref,3')."""
    results = {}
    for i in range(0, len(positions), batch_size):
        batch = positions[i:i + batch_size]
        regions = []
        for chrom, pos in batch:
            c = str(chrom).replace("chr", "").replace("23", "X").replace("24", "Y")
            regions.append(f"{c}:{int(pos)-1}..{int(pos)+1}:1")
        for attempt in range(3):
            try:
                r = requests.post(
                    f"{ENSEMBL_API}/sequence/region/human",
                    json={"regions": regions},
                    headers={"Content-Type": "application/json", "Accept": "application/json"},
                    timeout=60,
                )
                if r.status_code == 200:
                    for item, (chrom, pos) in zip(r.json(), batch):
                        seq = item.get("seq", "").upper()
                        if len(seq) == 3 and all(b in "ACGT" for b in seq):
                            results[(chrom, pos)] = seq
                    break
                time.sleep(1)
            except Exception:
                time.sleep(2)
        time.sleep(0.05)
        if (i // batch_size) % 20 == 0:
            print(f"    fetched {min(i+batch_size, len(positions))}/{len(positions)} contexts")
    return results


def build_spectrum(cancer, rng):
    df = pd.read_csv(RAW / f"{cancer}_mutations.tsv", sep="\t", low_memory=False)
    snv = df[df["variant_type"].str.upper().isin(["SNP", "SNV"])].copy()
    snv = snv[(snv["ref_allele"].str.len() == 1) & (snv["alt_allele"].str.len() == 1)]
    snv = snv[snv["ref_allele"].isin(BASES) & snv["alt_allele"].isin(BASES)]
    snv = snv.dropna(subset=["chromosome", "start_pos"])
    snv["start_pos"] = snv["start_pos"].astype(int)

    if len(snv) > SAMPLE_N:
        snv = snv.sample(SAMPLE_N, random_state=int(rng.integers(0, 1e6)))
    print(f"  {cancer}: sampling {len(snv):,} SNVs")

    positions = list(dict.fromkeys(zip(snv["chromosome"].astype(str), snv["start_pos"])))
    print(f"  {cancer}: fetching {len(positions):,} unique trinucleotide contexts...")
    ctx = fetch_context_batch(positions)
    print(f"  {cancer}: retrieved {len(ctx):,} contexts")

    channels = {c: 0 for c in channel_order()}
    used = 0
    for _, row in snv.iterrows():
        key = (str(row["chromosome"]), int(row["start_pos"]))
        tri = ctx.get(key)
        if not tri:
            continue
        ref, alt = row["ref_allele"], row["alt_allele"]
        five, mid, three = tri[0], tri[1], tri[2]
        # sanity: middle base should equal ref (Ensembl + strand)
        if mid != ref:
            # try reverse-complement interpretation
            if COMP.get(mid) == ref:
                five, mid, three = COMP[three], COMP[mid], COMP[five]
                ref = mid
                alt = COMP[alt]
            else:
                continue
        # collapse to pyrimidine strand
        if ref in ("A", "G"):
            five, three = revcomp(three), revcomp(five)
            ref, alt = COMP[ref], COMP[alt]
        sub = f"{ref}>{alt}"
        if sub not in SBS_COLORS:
            continue
        chan = f"{five}[{sub}]{three}"
        if chan in channels:
            channels[chan] += 1
            used += 1

    order = channel_order()
    counts = np.array([channels[c] for c in order], dtype=float)
    fracs = counts / counts.sum() if counts.sum() else counts
    print(f"  {cancer}: {used:,} mutations placed into 96 channels\n")
    return fracs


def main():
    rng = np.random.default_rng(SEED)
    order = channel_order()
    spectra = {}
    for cancer in CANCERS:
        spectra[cancer] = build_spectrum(cancer, rng)

    # Save matrix
    mat = pd.DataFrame(spectra, index=order)
    mat.to_csv(PROC / "sbs96_matrix.tsv", sep="\t")
    print(f"Saved SBS-96 matrix -> {PROC}/sbs96_matrix.tsv")

    # ── Plot: 3 stacked spectra, classic SBS-96 barcode ──────────────────
    fig, axes = plt.subplots(len(CANCERS), 1, figsize=(15, 8.5), sharex=True)
    bar_colors = []
    for sub in SUBS:
        bar_colors += [SBS_COLORS[sub]] * 16

    for ax, cancer in zip(axes, CANCERS):
        vals = spectra[cancer]
        x = np.arange(96)
        ax.bar(x, vals, color=bar_colors, width=0.82, edgecolor="none")
        ax.set_xlim(-0.7, 95.7)
        ymax = max(vals.max() * 1.25, 0.02)
        ax.set_ylim(0, ymax)
        ax.set_ylabel("Fraction\nof SNVs", fontsize=10)
        # cancer label box
        ax.text(0.5, ymax * 0.86, CANCER_FULL[cancer], fontsize=13, fontweight="bold",
                va="center", ha="left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_yticks(np.linspace(0, ymax, 4))
        ax.set_yticklabels([f"{v:.2f}" for v in np.linspace(0, ymax, 4)])
        ax.set_xticks([])

    # Coloured substitution-type header bar along the top axis
    top = axes[0]
    header_y = top.get_ylim()[1]
    for i, sub in enumerate(SUBS):
        start = i * 16
        top.add_patch(Rectangle((start - 0.5, header_y * 1.02), 16, header_y * 0.10,
                                color=SBS_COLORS[sub], clip_on=False, zorder=5))
        top.text(start + 8 - 0.5, header_y * 1.13, sub, ha="center", va="bottom",
                 fontsize=12, fontweight="bold",
                 color="white" if sub in ("C>G",) else "#222", clip_on=False,
                 bbox=dict(boxstyle="round,pad=0.15",
                           fc=SBS_COLORS[sub], ec="none", alpha=0.0))

    axes[-1].set_xlabel("96 trinucleotide contexts  (5′–[substitution]–3′,  pyrimidine strand)",
                        fontsize=11)
    fig.suptitle("Mutational signature fingerprints — each cancer type carries the trace of its cause",
                 fontsize=15, fontweight="bold", y=0.99)
    fig.text(0.5, 0.005,
             "SBS-96 spectra from seeded samples of TCGA SNVs (n≈8,000 per type). "
             "C>A enrichment = tobacco (SBS4); C>T at CpG = ageing clock (SBS1); "
             "TpC C>T/C>G = APOBEC (SBS2/13).",
             ha="center", fontsize=9, color="#555")
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    out = FIGS / "mutational_signatures.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {out}")


if __name__ == "__main__":
    main()
