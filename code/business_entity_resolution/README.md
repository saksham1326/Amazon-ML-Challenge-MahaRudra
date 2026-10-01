# Machine Learning Business Entity Resolution Pipeline

High-performance, scalable entity resolution pipeline built for the Amazon ML Challenge 2026. Resolves multi-source business entity fragments across three heterogeneous vendor sources under precision-heavy macro $F_{0.5}$ evaluation.

---

## 1. System Overview

- **Task:** For each Source 1 reference entity, identify all corresponding records from Source 2 and Source 3.
- **Evaluation Metric:** Macro-averaged $F_{0.5}$ over all Source 1 entities (including singletons).
- **Core Technology:** Country-partitioned inverted-index blocking (95.28% validation candidate recall), transliteration and text normalization, 36 pairwise features, an XGBoost GPU classifier, validation-selected thresholds ($\tau_{S2}=0.70$, $\tau_{S3}=0.75$), and greedy target-side exclusivity.

---

## 2. Directory Structure

```text
code/business_entity_resolution/
├── src/
│   ├── normalization.py      # Unicode cleaning, legal suffix removal, address expansion
│   ├── blocking.py           # Multi-attribute inverted index candidate generator
│   ├── features.py           # 36 name, address, numeric, and interaction features
│   ├── model.py              # XGBoost and LightGBM model wrapper
│   ├── thresholding.py       # Macro F0.5 threshold optimization and deduplication
│   ├── evaluation.py         # Official competition macro F0.5 scoring metric
│   ├── data_loader.py        # Streaming TSV reader and country partitioner
│   ├── output.py             # Submission TSV generation and report writing
│   ├── inference.py          # End-to-end country-streaming test inference engine
│   └── main.py               # CLI entry point with automated submission validation
├── requirements.txt          # Pinned Python dependencies
└── README.md                 # Complete documentation and reproduction guide
```

---

## 3. Installation & Requirements

Install the pinned packages from this directory:

```bash
pip install -r requirements.txt
```

### Dependencies:
The selected saved model uses XGBoost, which is **not included** in the current requirements file and must be installed separately. GPU inference also requires a compatible CUDA-enabled NVIDIA environment. The requirements file currently lists:
- `numpy==2.4.2`
- `scipy==1.17.1`
- `pandas==3.0.6`
- `scikit-learn==1.8.0`
- `lightgbm==4.7.0`
- `rapidfuzz==3.14.6`
- `text-unidecode==1.3`
- `joblib==1.5.3`

---

## 4. End-to-End Reproduction Guide

### Step 1: Model Training & Validation (Optional to retrain from scratch)
To run the full training pipeline on training data and evaluate against the held-out validation set:

```bash
python train.py --train-dir . --model-type xgboost_gpu
```
This fits the model, performs threshold calibration, logs validation metrics, and serializes the model to `models/final_entity_matcher.joblib`.

### Step 2: Test Inference & Submission Generation
To run candidate generation, feature extraction, model scoring, and generate submission files for all test records:

```bash
python code/business_entity_resolution/src/main.py \
    --test-dir . \
    --output-dir output
```

This generates:
- `output/matching_results.tsv` (Leaderboard submission file)
- `output/candidate_pairs.tsv` (Auditing candidate set file)

### Step 3: Submission Verification
The pipeline automatically runs the competition validator:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir .
```
Output: **PASS**

---

## 5. Validation Results & Benchmarks

| Metric | Validation Score |
| :--- | :--- |
| **Macro $F_{0.5}$** | **0.9407647** |
| **Precision** | **99.14%** |
| **Recall** | **86.77%** |
| **Candidate Recall** | **95.28%** |
| **Singleton Accuracy** | **97.97%** |
| **False Positives** | **262** |
| **False Negatives** | **4,586** |

The held-out validation set contains 10,000 Source 1 entities. The team-reported public leaderboard score is approximately **0.827 (82.7%)**; verify the exact portal score before final submission. Do not present the validation result as the leaderboard result.

---

## 6. Compliance & Fair Play

- **No external lookups:** The pipeline uses the supplied records and does not call external APIs or business registries.
- **Submission format:** Inference writes tab-separated outputs and emits rows for Source 1 entities, including empty predictions.
- **Target conflict handling:** Greedy target-side exclusivity retains the highest-scoring assignment for a target ID.

### Packaging Limitation

This folder is not currently self-contained as required by the submission specification. The data and model/metadata are outside this folder, training is launched by the repository-root `train.py`, the validator is under the root `utils/`, and XGBoost is absent from the pinned requirements. The commands above work from the full repository root when those assets and dependencies are available; copying only this folder will not reproduce the output files.
