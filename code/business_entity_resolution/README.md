# Entity Resolution Pipeline

This guide explains how to run the business-matching pipeline from the repository root. Given a Source 1 business, the pipeline finds matching records in Sources 2 and 3. It also handles businesses with no matches.

The pipeline has four stages: normalize text, retrieve likely candidates, score each pair with a 36-feature XGBoost model, and write the final matches. Validation-selected thresholds are 0.70 for Source 2 and 0.75 for Source 3.

---

## Files

- `src/normalization.py` standardizes business names and addresses.
- `src/blocking.py` retrieves likely matching records.
- `src/features.py` calculates pairwise comparison features.
- `src/model.py` loads and runs the classifier.
- `src/thresholding.py` applies confidence thresholds and resolves target conflicts.
- `src/inference.py` and `src/main.py` run inference and write output files.
- `requirements.txt` lists the pinned Python dependencies.

---

## Install

Run from the repository root:

```bash
pip install -r code/business_entity_resolution/requirements.txt
pip install xgboost
```

The pinned requirements file does not currently include XGBoost. The saved model uses GPU-backed XGBoost, so inference also requires a compatible NVIDIA/CUDA setup.

---

## Run the Pipeline

Run these commands from the repository root. The training and test TSV files should be in that root, and the model files should be in `models/`.

Generate the two submission files with the saved model:

```bash
python code/business_entity_resolution/src/main.py --test-dir . --output-dir output
```

To train a new model first:

```bash
python train.py --train-dir . --model-type xgboost_gpu
```

Check the generated files:

```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir .
```

The validator checks required rows, columns, and ID-list structure. It does not calculate the leaderboard score.

---

## Validation Snapshot

The saved report covers 10,000 held-out Source 1 records: macro $F_{0.5}$ **0.9407647**, precision **99.14%**, recall **86.77%**, and candidate recall **95.28%**. The team-reported public leaderboard score is approximately **0.827**; check the portal for the exact score. These are separate evaluations.

---

## Current Packaging Limitation

This source folder depends on files outside itself: the datasets and model are in the repository root, while `train.py` and the validator are in the root and `utils/`. XGBoost is also missing from the pinned requirements. The commands above work from the full repository with those files installed, but this folder alone is not a self-contained reproduction package. No external data lookup is used.
