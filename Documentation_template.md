# Amazon ML Challenge 2026: Business Entity Resolution

**Team Name:** MahaRudra  
**Team Members:** Saksham Chaudhari, Akash Bhuyan, Rahul Atkare, Rushi Kedar  
**Submission Date:** 02 October 2026  

---

## 1. Executive Summary
We built a multi-stage entity-resolution system to link each Source 1 business record to all corresponding records in Sources 2 and 3. The pipeline uses country-partitioned inverted-index blocking, a 36-feature gradient-boosted model, source-specific thresholds, and target-side conflict resolution. It is designed around the challenge's precision-weighted macro $F_{0.5}$ objective.

The saved held-out validation report records macro $F_{0.5}$ of **0.9408**, precision of **99.14%**, recall of **86.77%**, candidate recall of **95.28%**, and singleton accuracy of **97.97%**. The team-reported public leaderboard result was approximately **0.827 (82.7%)**. The exact portal score is not stored in the local artifacts, so it should be replaced with the exact displayed value before final packaging. Validation and leaderboard results are separate evaluations and are reported separately here.

The test run generated the required `matching_results.tsv` and `candidate_pairs.tsv`. Its saved report records 1,732,544 Source 1 entities, 34,461,015 candidate pairs, and 5,543,353 predicted links. No external business registries, geocoding services, or other external lookups are used.

---

## 2. Methodology

### 2.1 Problem Analysis
Each Source 1 entity can match zero, one, or many Source 2 and Source 3 records. Names and addresses vary in spelling, punctuation, legal suffixes, transliteration, completeness, and ordering. The test set includes a country not present in training, so the pipeline treats country as a data value instead of hard-coding a fixed set of countries. Correctly identifying entities with no matches is important because singletons are included in the macro metric.

### 2.2 Solution Strategy
The pipeline separates candidate generation from pair scoring. It normalizes names and addresses, uses a country-partitioned inverted index to keep pair comparisons tractable, ranks candidates by shared blocking keys, and scores up to 20 candidates per Source 1 entity. Validation-selected per-source thresholds and target-side conflict resolution produce the final links. The saved metadata identifies the selected model as XGBoost GPU and records 36 input features.

---

## 3. Candidate Generation (Blocking)
To limit pairwise comparisons, the pipeline creates a country-partitioned inverted index over target records and unions matches from multiple name/address keys. Keys include compact and core names, name tokens and token combinations, numeric/address combinations, name/number and name/city combinations, and pairs of significant address words. High-frequency keys are pruned to limit generic matches. Candidate records are ranked by shared-key count and capped at 20 per Source 1 entity.

The saved validation metadata reports **95.28% candidate recall**. This is the proportion of true links present in the candidate set; true links missed by blocking cannot be recovered at later stages. The test inference report records **34,461,015 candidate pairs**.

---

## 4. Matching Model
The final scorer uses **36 pairwise features**: 14 name similarity and token signals, address presence/similarity and numeric consistency signals, cross-field interactions, source indicators, and shared blocking-key count. Features include exact normalized and core-name equality, Levenshtein and Jaro-Winkler similarity, token overlap, character 3-gram similarity, address token similarity, and primary address-number agreement or conflict.

The selected model recorded in `models/model_metadata.json` is **XGBoost GPU** (`xgboost_gpu`). The training script's default configuration uses 500 estimators, learning rate 0.035, maximum depth 7, and row and feature subsampling of 0.85. Training pairs include ground-truth positive links and hard negatives drawn from plausible blocking candidates.

Thresholds selected on validation are $\tau_{S2}=0.70$ and $\tau_{S3}=0.75$. Inference applies an additional primary-number conflict penalty, preserves exact name-and-address matches with a confidence floor, and greedily enforces target-side exclusivity.

---

## 5. Results & Error Analysis
### Validation Performance

The figures below are from the saved validation report and model metadata. The validation set contains 10,000 Source 1 entities.

| Metric | Result |
| :--- | ---: |
| Macro $F_{0.5}$ | **0.9407647** |
| Precision | 0.9913620 (99.14%) |
| Recall | 0.8676670 (86.77%) |
| Candidate recall | 0.9527918 (95.28%) |
| Singleton accuracy | 0.9796673 (97.97%) |
| False positives | 262 |
| False negatives | 4,586 |

The team-reported public leaderboard score was approximately **0.827 (82.7%)**. The exact portal value is not present in the local artifacts and should be verified before final packaging. It is distinct from the held-out validation result.

### Error Considerations

False positives can arise when distinct businesses share generic names or nearby addresses; numeric address mismatch features and target-side conflict resolution are intended to reduce these errors. False negatives can result when both name and address evidence are incomplete or heavily altered. The observed 95.28% candidate recall also means some true links may be excluded before scoring.

---

## 6. Conclusion
The pipeline combines deterministic text normalization, country-partitioned blocking, pairwise feature scoring, validation-selected thresholds, and target-side conflict handling. On the saved held-out validation run it reached macro $F_{0.5}=0.9408$ with 99.14% precision; the team-reported leaderboard score is approximately 0.827. The local test report records the generated outputs and inference totals. These results should be interpreted with the validation/leaderboard distinction and the packaging limitation described below.

---

## Appendix

### A. Submission Structure

The requested archive layout is:

```text
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

The two required TSVs, the source directory, its README, and its requirements file are present in the current workspace. The pipeline is not yet reproducible from only `code/business_entity_resolution/`: test and training data, the saved model and metadata, `train.py`, and the validator are stored elsewhere in the repository. In addition, the pinned requirements file does not include XGBoost, which is required by the selected model. This documentation-only change does not alter code, dependencies, data, or model artifacts.

### B. Current Repository Commands

From the repository root, inference with the existing model and root-level test files is invoked as:

```bash
python code/business_entity_resolution/src/main.py --test-dir . --model-path models/final_entity_matcher.joblib --meta-path models/model_metadata.json --output-dir output
```

Training is launched by the root-level script, not by a script inside the package:

```bash
python train.py --train-dir . --model-type xgboost_gpu
```

The saved model uses XGBoost GPU; the selected execution mode requires a compatible XGBoost installation and CUDA-enabled NVIDIA environment. Replace the approximate public leaderboard score in this document with the exact value shown in the competition portal before final submission.
