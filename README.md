# Amazon ML Challenge 2026: Multi-Source Business Entity Resolution

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Model: XGBoost GPU](https://img.shields.io/badge/model-XGBoost%20GPU-brightgreen.svg)](https://xgboost.readthedocs.io/)
[![Validation F0.5: 0.9408](https://img.shields.io/badge/Validation%20Macro%20F0.5-0.9408-success.svg)](output/metrics_summary.txt)
[![Validation Precision: 99.14%](https://img.shields.io/badge/Validation%20Precision-99.14%25-blueviolet.svg)](output/metrics_summary.txt)
[![Submission Status: PASS](https://img.shields.io/badge/Submission%20Validator-PASS-success.svg)](output/final_report.txt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A multi-stage business entity-resolution pipeline for the **Amazon ML Challenge 2026**, evaluated with the precision-weighted macro $F_{0.5}$ metric.

---

## 📌 Table of Contents
1. [Executive Summary](#-executive-summary)
2. [Problem Statement & Analysis](#-problem-statement--analysis)
3. [Architecture & Pipeline](#-architecture--pipeline)
4. [Candidate Generation (Blocking)](#-candidate-generation-blocking)
5. [Feature Engineering (36 Dimensions)](#-feature-engineering-36-dimensions)
6. [Modeling & Global Consistency Deduplication](#-modeling--global-consistency-deduplication)
7. [Benchmark Results & Ablation Studies](#-benchmark-results--ablation-studies)
8. [Directory Structure](#-directory-structure)
9. [Installation & Setup](#-installation--setup)
10. [End-to-End Reproduction Guide](#-end-to-end-reproduction-guide)
11. [Submission Compliance](#-submission-compliance)

---

## 🚀 Executive Summary

In commercial data ecosystems, business identity data arrives asynchronously from disparate vendor sources with partial, noisy, and conflicting information (names, abbreviations, transliterations, municipal numbering, landmarks).

Our solution establishes an **end-to-end entity resolution pipeline**:
- **Validation:** Saved held-out macro $F_{0.5}$ **0.9407647** (precision **99.14%**, recall **86.77%**).
- **Candidate recall:** **95.28%** on the recorded validation run.
- **Leaderboard:** Team-reported public score approximately **0.827 (82.7%)**; exact portal value is not saved locally and must be verified.
- **Test inference:** 1,732,544 Source 1 entities, 34,461,015 candidate pairs, 5,543,353 predicted links; recorded runtime **78.27 minutes**.
- **Model:** XGBoost GPU, 36 features, thresholds 0.70 (Source 2) and 0.75 (Source 3).
- **No external lookups:** The pipeline uses only supplied records and local artifacts.

---

## 🔍 Problem Statement & Analysis

### Data Sources
- **Source 1 ($S_1$):** Deduplicated reference entity source (100% complete names and addresses).
- **Source 2 ($S_2$) & Source 3 ($S_3$):** Independent commercial vendor records with missing fields, transliterated Indian scripts, legal suffix mismatches, and transposed address tokens.
- **Ground Truth Target:** For each $S_1$ record, output a comma-separated list of matching $S_2$ and $S_3$ IDs (or empty if singleton).

### Critical Data Insights
1. **Variable fields:** Names and addresses can differ through spelling, punctuation, legal suffixes, transliteration, missing components, and token order.
2. **Open-set countries:** Test data contains a country label not present in training. Country partitioning uses observed string values rather than a fixed list.
3. **Many-to-many reference mapping:** A Source 1 entity may have zero, one, or many matches across Sources 2 and 3; the model must also identify singletons.
4. **Macro $F_{0.5}$ objective:** $F_{0.5} = \frac{1.25 \times P \times R}{0.25 \times P + R}$. Precision is weighted more heavily than recall.

---

## 🏗️ Architecture & Pipeline

```text
               +-------------------------------------------------------------+
               |  Raw Multi-Vendor TSV Streams (S1 Reference, S2, S3 Vendors)|
               +-------------------------------------------------------------+
                                              |
                                              v
               +-------------------------------------------------------------+
               | Stage 1: Canonical Normalization & Transliteration          |
               | - Unicode canonicalization & unidecode (Indian scripts)     |
               | - Legal suffix stripping (Pvt Ltd, LLC, SARL, GmbH)         |
               | - Address token contraction & landmark cleaning             |
               +-------------------------------------------------------------+
                                              |
                                              v
               +-------------------------------------------------------------+
               | Stage 2: Country-Partitioned Inverted Index Blocking        |
               | - Multi-key union (compact name, core tokens, street/city)  |
              | - Frequency pruning (250 name-like / 100 other targets)     |
              | - Validation candidate recall: 95.28%; top 20 per entity   |
               +-------------------------------------------------------------+
                                              |
                                              v
               +-------------------------------------------------------------+
              | Stage 3: Pairwise 36-D Feature Extraction                   |
               | - 14 Name similarity signals (RapidFuzz C++ kernels)         |
              | - Address similarity and numeric consistency signals       |
               | - 8 Cross-field interaction & source indicator terms        |
               +-------------------------------------------------------------+
                                              |
                                              v
               +-------------------------------------------------------------+
              | Stage 4: XGBoost GPU Inference & Confidence Scoring        |
              | - 500 estimators, max depth 7, learning rate 0.035          |
               | - Trained on hard negatives mined directly from candidates  |
               +-------------------------------------------------------------+
                                              |
                                              v
               +-------------------------------------------------------------+
               | Stage 5: Calibration & Bipartite 1-to-1 Target Exclusivity  |
              | - Source-specific thresholds (τ_S2 = 0.70, τ_S3 = 0.75)    |
               | - Greedy bipartite conflict resolution (highest prob wins)  |
               +-------------------------------------------------------------+
                                              |
                                              v
               +-------------------------------------------------------------+
               | Verified Submission Files (matching_results.tsv, validator) |
               +-------------------------------------------------------------+
```

---

## ⚡ Candidate Generation (Blocking)

To avoid evaluating quadratic $O(N \times M)$ pairwise combinations across 12M+ records, we employ an inverted-index blocking system with the following multi-attribute keys:

| Blocking Key | Description | Target Coverage |
| :--- | :--- | :--- |
| `compact_n` | Space-stripped alphanumeric name & sorted token name | Catches domain name concatenation (`moonboards.com` $\leftrightarrow$ `Moon Boards`) |
| `core_n` / `n2` | Legal suffix-stripped name and first 2 core name tokens | High-precision exact business root matches |
| `n_tok` | Significant individual core name tokens ($\ge 4$ characters) | Partial name variations and brand transpositions |
| `num_street` | Primary street number paired with rare street words | Same address location verification |
| `num_city` | Street number paired with city/state tokens | Addresses with alternate street naming formats |
| `name_num` | First name token paired with primary address number | Strong cross-field anchor |
| `name_city` | First name token paired with city/state tokens | Distinguishes identical franchises across cities |

Keys with extreme frequency (>150–300 occurrences) are pruned to eliminate generic noise while retaining composite keys.

---

## 🧪 Feature Engineering (36 Dimensions)

Each candidate pair is represented by 36 features computed with RapidFuzz and custom token/numeric comparisons:

- **Name signals (14):** exact normalized/core equality, Levenshtein and Jaro-Winkler similarity, token-sort/token-set similarity, token Jaccard and overlap, length difference/ratio, prefix match, character 3-gram Jaccard, and first-token match/conflict.
- **Address and number signals (14):** target address presence, exact address match, Levenshtein/Jaro-Winkler and token similarities, common numeric-token count, number-match indicators, primary-number match/mismatch/difference, and missing-address/name evidence.
- **Cross-field and source signals (8):** name/address interactions, name-high/number-conflict indicator, Source 2/Source 3 indicators, and shared blocking-key count.

---

## 🎯 Benchmark Results & Ablation Studies

The local saved report evaluates 10,000 held-out Source 1 entities. It records **macro $F_{0.5}=0.9407647**, **99.14% precision**, **86.77% recall**, **95.28% candidate recall**, **97.97% singleton accuracy**, 262 false positives, and 4,586 false negatives.

The team-reported public leaderboard score is approximately **0.827 (82.7%)**. The exact score is not stored in the repository and should be checked against the competition portal. It is not the same as the local held-out validation metric.

---

## 📂 Directory Structure

```text
amazon-ml-challenge-2026/
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── __init__.py          # Package initialization
│       │   ├── normalization.py     # Transliteration, abbreviation & suffix stripping
│       │   ├── blocking.py          # Multi-attribute inverted index candidate generator
│       │   ├── features.py          # 36-D pairwise feature extraction
│       │   ├── model.py             # XGBoost and LightGBM model wrapper
│   └── val_s1_ids.txt               # Held-out Source 1 validation entity IDs
│   ├── final_entity_matcher.joblib  # Serialized production XGBoost model
│   ├── matching_results.tsv         # Final submission matches
│   ├── candidate_pairs.tsv          # Blocking candidate set
│   ├── final_report.txt             # Test inference and validation report
│   └── metrics_summary.txt          # Saved metrics summary
│       │   ├── thresholding.py      # Macro F0.5 grid search & 1-to-1 bipartite deduplication
│       │   ├── evaluation.py        # Competition macro F0.5 scoring metric
│       │   ├── data_loader.py       # Streaming TSV reader & country partitioner
│       │   ├── output.py            # Formats submission TSVs
│       │   ├── inference.py         # End-to-end country streaming test inference engine
│       │   └── main.py              # CLI entry point with automated submission validation
│       ├── requirements.txt         # Pinned Python package dependencies
│       └── README.md                # Standalone reproduction guide
├── experiments/
│   └── val_s1_ids.txt               # Held-out Source 1 validation entity IDs (5,000 records)
├── models/
│   ├── final_entity_matcher.joblib  # Serialized production XGBoost model
│   └── model_metadata.json          # Optimal hyperparameters, thresholds & feature list
├── output/
│   └── .gitkeep                     # Output placeholder & reproduction note
├── utils/
│   └── validate_submission.py       # Official submission validator
├── train.py                         # End-to-end training and validation script
├── Documentation_template.md        # Official filled methodology submission document
├── PS.txt                           # Official competition problem statement & rules
└── README.md                        # Master repository documentation
```

---

## 💻 Installation & Setup

1. **Open the repository:** Run the following setup commands from the root of the submission repository.

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r code/business_entity_resolution/requirements.txt
   pip install xgboost
   ```

---

## 🔁 End-to-End Reproduction Guide

### Option 1: Run Full Test Inference (Using Pre-trained Model)
The selected model and metadata are in the repository-root `models/` folder. From the repository root, run inference against the root-level test TSV files:
   --test-dir .
   --train-dir . --model-type xgboost_gpu
   --test-dir .

```bash
python code/business_entity_resolution/src/main.py \
   --test-dir . \
    --output-dir output
```

This generates:
- `output/matching_results.tsv` (Leaderboard submission file)
- `output/candidate_pairs.tsv` (Blocking candidate set file)

### Option 2: Retrain the Model from Scratch
To reproduce the training and validation pipeline on raw training data:

```bash
python train.py --train-dir . --model-type xgboost_gpu
```

### Option 3: Verify Submission Compliance
To run the competition validation suite locally:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
   --test-dir .
```
Expected output: **`PASS`**

---

## 📜 Submission Compliance

- **No external lookups:** The system uses supplied data and local model artifacts.
- **Output format:** The required matching and candidate TSV files are present under `output/`.
- **Model:** The saved model is XGBoost GPU; a compatible XGBoost installation and CUDA setup are needed for GPU execution.
- **Packaging limitation:** The package folder alone is not self-contained. Data, model and metadata, root-level training script, validator, and XGBoost dependency are outside the prescribed package folder or missing from its requirements file. The current Markdown documents these gaps; a documentation-only change cannot make the archive independently reproducible.

---

## 👥 Authors
- **Saksham Chaudhari** ([@saksham1326](https://github.com/saksham1326))
- **Akash Bhuyan** ([@AB-1817](https://github.com/AB-1817))
- **Rahul Atkare** ([@mystio1](https://github.com/mystio1))
- **Rushi Kedar** ([@Rushi9234](https://github.com/Rushi9234))
