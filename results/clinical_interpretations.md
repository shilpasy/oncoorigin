# Pan-Cancer Classifier — Clinical Interpretations
Model: XGBoost + DNABERT, GPT-gpt-4o-mini clinical triage

---
## Patient TCGA-3C-AAAU  (✓ CORRECT)

**True label:** BRCA  |  **Predicted:** BRCA  |  **TMB:** 24

**Priority:** HIGH

**Summary:** The mutation profile aligns with BRCA-related breast cancer due to the presence of a GATA3 driver mutation and a high tumor mutational burden (TMB).

**Key driver alterations:**
- GATA3 (Frame_Shift_Ins, P408Afs*99) - Implicated in breast cancer progression and subtype differentiation.
- PIK3CA - Commonly mutated in breast cancer, associated with the Luminal A subtype.
- TP53 - Frequently altered in breast cancer, indicating poor prognosis.

**Mutational process:** The SBS-6 signature suggests a mutational process associated with exposure to tobacco smoke or other environmental factors, which may influence tumor behavior.

**Therapy implications:** Consideration for PARP inhibitors due to BRCA-related mutations, along with potential use of CDK4/6 inhibitors for hormone receptor-positive subtypes.

---

## Patient TCGA-3C-AALI  (✓ CORRECT)

**True label:** BRCA  |  **Predicted:** BRCA  |  **TMB:** 459

**Priority:** HIGH

**Summary:** The mutation profile aligns with BRCA-related breast cancer due to the presence of TP53 and FBXW7 mutations, which are commonly associated with this cancer type.

**Key driver alterations:**
- TP53 (Nonsense_Mutation, S183*): A critical tumor suppressor gene frequently mutated in breast cancer, indicating poor prognosis.
- FBXW7 (Missense_Mutation, E177Q): Involved in cell cycle regulation and tumor suppression, its mutation can contribute to oncogenesis.

**Mutational process:** The SBS-6 signature indicates a high level of mutational burden, often associated with homologous recombination deficiency and exposure to certain carcinogens.

**Therapy implications:** Consideration for PARP inhibitors due to BRCA-related pathways, along with potential use of CDK4/6 inhibitors and HER2-targeted therapies if applicable.

---

## Patient TCGA-A1-A0SH  (✗ MISCLASSIFIED)

**True label:** BRCA  |  **Predicted:** LUAD  |  **TMB:** 94

**Priority:** HIGH

**Summary:** The presence of a BRCA1 nonsense mutation strongly suggests a BRCA-related cancer rather than lung adenocarcinoma.

**Key driver alterations:**
- BRCA1 (Nonsense_Mutation, Q934*) - Implicated in homologous recombination repair deficiency, leading to increased sensitivity to PARP inhibitors.

**Mutational process:** The SBS-6 signature indicates a mutational process associated with APOBEC activity, which is often linked to DNA damage and repair mechanisms.

**Therapy implications:** Consideration for PARP inhibitors due to BRCA1 mutation; further evaluation for potential immunotherapy options given high TMB.

---

## Patient TCGA-05-4244  (✓ CORRECT)

**True label:** LUAD  |  **Predicted:** LUAD  |  **TMB:** 192

**Priority:** HIGH

**Summary:** The mutation profile aligns with lung adenocarcinoma due to the presence of a KRAS G12C mutation and a high tumor mutational burden (TMB).

**Key driver alterations:**
- KRAS (G12C) - a well-established driver mutation in lung adenocarcinoma associated with poor prognosis and resistance to certain therapies.

**Mutational process:** The SBS-6 signature indicates a significant contribution from tobacco exposure, which is consistent with lung cancer etiology.

**Therapy implications:** Targeted therapies such as KRAS inhibitors may be considered, along with immunotherapy options due to the high TMB.

---

## Patient TCGA-05-4249  (✓ CORRECT)

**True label:** LUAD  |  **Predicted:** LUAD  |  **TMB:** 300

**Priority:** HIGH

**Summary:** The mutation profile aligns with lung adenocarcinoma due to the presence of key driver mutations and a high tumor mutational burden.

**Key driver alterations:**
- BRAF (A762E): Potential target for BRAF inhibitors.
- KRAS (G12C): Targetable with specific KRAS G12C inhibitors.
- PIK3CA (E545K): May respond to PI3K pathway inhibitors.

**Mutational process:** The SBS-6 signature indicates a significant contribution from tobacco exposure, consistent with lung cancer etiology.

**Therapy implications:** Consider targeted therapies for BRAF, KRAS, and PIK3CA mutations, along with immunotherapy options due to high TMB.

---

## Patient TCGA-05-4402  (✗ MISCLASSIFIED)

**True label:** LUAD  |  **Predicted:** BRCA  |  **TMB:** 122

**Priority:** HIGH

**Summary:** The mutation profile shows significant alterations in key oncogenes and tumor suppressor genes commonly associated with LUAD, despite the model's incorrect prediction of BRCA-related breast cancer.

**Key driver alterations:**
- TP53 (Frame_Shift_Del, V173Afs*69) - a critical tumor suppressor mutation linked to poor prognosis in lung cancer.
- EGFR (Missense_Mutation, I759N) - an oncogenic mutation that may indicate sensitivity to EGFR inhibitors.
- EGFR (In_Frame_Del, T751_I759delinsN) - another alteration in EGFR, reinforcing the potential for targeted therapy.

**Mutational process:** The SBS-6 signature indicates a high level of C>T transitions, which is often associated with tobacco exposure and may suggest a mutagenic process related to environmental factors.

**Therapy implications:** Targeted therapies such as EGFR inhibitors (e.g., osimertinib) may be effective due to the presence of EGFR mutations, and consideration for immunotherapy based on TMB is warranted.

---

## Patient TCGA-3L-AA1B  (✓ CORRECT)

**True label:** COAD  |  **Predicted:** COAD  |  **TMB:** 120

**Priority:** HIGH

**Summary:** The mutation profile aligns with colorectal adenocarcinoma, characterized by key driver mutations and a high tumor mutational burden.

**Key driver alterations:**
- APC (Nonsense_Mutation, R1450*): Critical for tumor suppression in colorectal cancer.
- BRAF (Missense_Mutation, L597V): Associated with poor prognosis and potential for targeted therapy.
- PIK3CA (Missense_Mutation, E545K): Implicated in oncogenic signaling pathways, offering therapeutic targets.

**Mutational process:** The SBS-6 signature indicates a high prevalence of C>T transitions, often associated with age-related mutational processes and environmental factors.

**Therapy implications:** Consideration for immunotherapy due to MSI-H status, and potential use of BRAF inhibitors if further testing confirms BRAF V600E mutation.

---

## Patient TCGA-4N-A93T  (✓ CORRECT)

**True label:** COAD  |  **Predicted:** COAD  |  **TMB:** 85

**Priority:** HIGH

**Summary:** The mutation profile aligns with colorectal adenocarcinoma, characterized by the presence of key driver mutations and a high tumor mutational burden.

**Key driver alterations:**
- KRAS (G12D) - Commonly associated with poor prognosis and resistance to EGFR inhibitors.
- TP53 (R273H) - Implicated in tumor progression and associated with genomic instability.

**Mutational process:** The SBS-6 signature indicates a high level of cytosine deamination, often linked to age-related mutational processes and environmental factors.

**Therapy implications:** Consideration for immunotherapy due to high TMB and potential MSI-H status; KRAS mutation suggests resistance to EGFR-targeted therapies.

---

## Patient TCGA-A6-2678  (✗ MISCLASSIFIED)

**True label:** COAD  |  **Predicted:** BRCA  |  **TMB:** 26

**Priority:** MEDIUM

**Summary:** The mutation profile shows a high TMB and a significant SBS-6 signature, which is often associated with DNA damage repair deficiencies, typically seen in BRCA-related cancers.

**Mutational process:** The predominance of C>T transitions suggests a mutational process consistent with exposure to alkylating agents or a defect in DNA repair mechanisms.

**Therapy implications:** Consideration for PARP inhibitors may be warranted due to the potential for homologous recombination deficiency, despite the absence of classic BRCA mutations.

---

