# Pan-Cancer Molecular Classifier

**Predicting tumor type from somatic mutation profiles using XGBoost + DNABERT + GPT-4o**

---

## Clinical Question

> Given a patient's somatic mutation profile from tumor sequencing, can we predict whether this is breast cancer, colorectal cancer, or lung adenocarcinoma?

This is a **3-class molecular classification** problem. No clinical metadata, imaging, or histology is used — only the mutations found in the tumor's DNA.

---

## Results

| | BRCA (Breast) | COAD (Colorectal) | LUAD (Lung Adeno) | Overall |
|---|---|---|---|---|
| Precision | 0.87 | 0.94 | 0.93 | — |
| Recall | 0.95 | 0.92 | 0.79 | — |
| F1 | 0.91 | 0.93 | 0.85 | — |
| **Accuracy** | | | | **90.0%** |
| **Macro F1** | | | | **89.7%** |

Evaluated with 5-fold stratified cross-validation on **2,098 real TCGA patients**.  
Random chance for 3 classes = 33%.

---

## Data

All data are from **The Cancer Genome Atlas (TCGA) PanCancer Atlas 2018**, accessed via the [cBioPortal](https://www.cbioportal.org/) public REST API. No registration is required. All patient data are de-identified and publicly available.

- 450,107 somatic mutations across 2,098 patients
- 3 cancer types: BRCA (n=1,009), COAD (n=528), LUAD (n=561)
- Mutations called from whole-exome sequencing (WES)

---

## Pipeline Architecture

```
cBioPortal REST API  (TCGA somatic mutations, 3 cancer types)
        │
        ▼
  Feature Engineering
  ├── Gene mutation binary matrix  (top 150 cancer genes, 0/1 per patient)
  ├── TMB                          (total somatic mutation burden)
  ├── SBS-6 fractions              (C>A, C>G, C>T, T>A, T>C, T>G — mutational process)
  └── Mutation class fractions     (missense, nonsense, frameshift, splice…)
        │
        ▼
  DNABERT-1  [Foundation Model]
  ├── Driver gene mutations selected (TP53, KRAS, PIK3CA, APC, EGFR, BRCA1/2…)
  ├── 129 bp reference context fetched from Ensembl GRCh37 REST API (batch)
  ├── 6-mer tokenisation → BERT encoder (zhihan1996/DNA_bert_6)
  ├── 768-dim CLS embeddings mean-pooled per patient
  └── PCA(20) → 20 additional features (60.8% variance retained)
        │
        ▼
  XGBoost Classifier  [ML Model]
  ├── 187 total features
  ├── 5-fold StratifiedKFold cross-validation
  └── SHAP TreeExplainer → per-feature, per-class importance
        │
        ▼
  GPT-4o LangGraph Agent  [LLM Layer]
  ├── Takes SHAP top features + driver mutations for each patient
  ├── Generates structured clinical interpretation (JSON)
  └── Outputs: predicted type, key drivers, mutational process, therapy implications
```

---

## Key Biological Findings

**What discriminates the three cancer types (from SHAP):**

| Cancer | Top discriminating features |
|---|---|
| BRCA | PIK3CA, CDH1, GATA3 mutations; APOBEC SBS signature |
| COAD | APC mutation (73% of patients), SMAD4, very high TMB (MSI-H) |
| LUAD | EGFR, STK11 mutations; high C>A (tobacco) SBS fraction |

**The LLM layer catches what the classifier misses:**  
Patient TCGA-A1-A0SH (true: BRCA) was misclassified as LUAD by XGBoost. GPT-4o's interpretation flagged: *"The presence of a BRCA1 nonsense mutation (Q934\*) strongly suggests a BRCA-related cancer rather than lung adenocarcinoma."* The LLM applied external biological knowledge not encoded in the tabular features.

---

## Project Structure

```
pan-cancer-classifier/
├── scripts/
│   ├── 01_download_data.py      # cBioPortal API download (TCGA mutations + clinical)
│   ├── 02_build_features.py     # Feature engineering (gene matrix, TMB, SBS-6)
│   ├── 03_embed_mutations.py    # DNABERT driver mutation embeddings via Ensembl API
│   ├── 04_train_classifier.py   # XGBoost + SHAP + UMAP
│   └── 05_interpret.py          # GPT-4o LangGraph clinical interpretation agent
├── results/
│   ├── figures/
│   │   ├── confusion_matrix.png
│   │   ├── feature_importance.png
│   │   ├── shap_beeswarm.png
│   │   └── umap_patients.png
│   ├── classification_report.txt
│   └── clinical_interpretations.md
├── RESULTS_AND_DISCUSSION.md
├── requirements.txt
└── .env.example
```

---

## How to Run

### 1. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# macOS (Apple Silicon): XGBoost requires OpenMP
brew install libomp
```

### 2. API Key

```bash
cp .env.example .env
# Add your OpenAI API key to .env
```

### 3. Run the pipeline

```bash
# Download data from cBioPortal (no account needed, ~10 min)
python scripts/01_download_data.py

# Feature engineering
python scripts/02_build_features.py

# DNABERT embeddings (~5 min, downloads model on first run)
python scripts/03_embed_mutations.py

# Train classifier + SHAP + UMAP
python scripts/04_train_classifier.py

# GPT-4o clinical interpretation
python scripts/05_interpret.py
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Data source | cBioPortal REST API (TCGA PanCancer Atlas 2018) |
| Sequence fetch | Ensembl GRCh37 REST API (batch, no download) |
| Foundation model | DNABERT-1 (`zhihan1996/DNA_bert_6`) via HuggingFace |
| ML classifier | XGBoost with SHAP interpretability |
| Dimensionality reduction | PCA + UMAP |
| LLM agent | GPT-4o-mini via LangGraph (structured JSON output) |
| Language | Python 3.9 |

---

## Limitations

- 3 cancer types only (TCGA has 33). Extension to pan-cancer 33-class classification is straightforward.
- Somatic mutations only — no copy number variation (CNV), RNA expression, or methylation. ALK/RET fusions in LUAD are invisible to this model.
- Cross-validation within TCGA; independent validation on ICGC or MSK-IMPACT data is needed.
- DNABERT was not fine-tuned on somatic mutation data.

See [RESULTS_AND_DISCUSSION.md](RESULTS_AND_DISCUSSION.md) for full discussion.

---

## Why this matters

Tumor molecular profiling is increasingly used clinically to guide treatment decisions (targeted therapy eligibility, immunotherapy selection). This project demonstrates that the *pattern of somatic mutations alone* — without histology, imaging, or gene expression — carries enough signal to identify cancer type with 90% accuracy. The pipeline is fully reproducible from public data and can be extended to all 33 TCGA cancer types.

---

*Data: TCGA PanCancer Atlas 2018 via cBioPortal. All data are publicly available and de-identified.*  
*Built as a portfolio project demonstrating ML/AI applied to clinical genomics.*
