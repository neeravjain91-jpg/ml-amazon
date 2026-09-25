# Business Entity Resolution Pipeline

**Competition**: Amazon ML Challenge 2026 — Business Entity Resolution  
**Team**: DeepResolve ER  
**Final Validation Macro F0.5**: **0.81172**  

---

## 1. System Overview

This repository contains the complete, self-contained, reproducible pipeline for the Business Entity Resolution task across three independent sources.

### Architecture Highlights
1. **Multi-Channel Inverted Index Blocking**: Employs exact normalized name, legal-suffix stripping, token-sorting, compact alphanumeric domain stems, and clean numeric address co-occurrence blocking ($85.62\%$ Candidate Recall, $99.991\%$ reduction ratio).
2. **Dense 30-Dimensional Pairwise Feature Extractor**: Jointly extracts sub-token character n-grams (3-gram Dice), token Jaccard, address length ratios, numeric token set agreement, and multiplicative cross-field interaction terms.
3. **Mined Hard-Negative Training**: Fits an MIT-licensed LightGBM GBDT classifier using naturally occurring candidate false positives from the blocking pool.
4. **Adaptive Score-Margin Decision Engine**: Calibrated decision threshold $\tau = 0.94$ with a relative score margin $\Delta = 0.05$ and maximum cardinality cap $K \le 5$, protecting singletons ($89.93\%$ accuracy) while capturing complex multi-matches ($1 \le k \le 11$).

---

## 2. Directory Layout

```
code/business_entity_resolution/
├── src/
│   ├── main.py                     # Unified entry point
│   ├── features.py                 # Dense 30-dim pairwise feature extractor
│   ├── evaluation.py               # Macro-averaged F0.5 metric engine
│   ├── train_model.py              # Hard-negative mining and LightGBM trainer
│   └── inference_partitioned.py    # Memory-bounded full test inference pipeline
├── README.md                       # Reproduction guide and system documentation
└── requirements.txt                # Pinned dependencies
```

---

## 3. Environment & Setup

Ensure Python 3.10+ is installed. Install the dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies
- `numpy >= 1.26.0`
- `pandas >= 2.0.0`
- `scipy >= 1.11.0`
- `scikit-learn >= 1.4.0`
- `lightgbm >= 4.0.0`

---

## 4. End-to-End Reproduction Instructions

From the root project directory:

### Step 1: Retrain the Model (Optional — Pre-trained Model Included)
```bash
python code/business_entity_resolution/src/main.py --mode train
```
This samples 40,000 training S1 queries, extracts true matches, mines candidate hard negatives, fits the LightGBM classifier, and saves `output/lgb_matching_model.pkl`.

### Step 2: Generate Full Test Submission Files
```bash
python code/business_entity_resolution/src/main.py --mode infer
```
This generates the two official competition outputs:
- `output/matching_results.tsv` (Leaderboard submission file)
- `output/candidate_pairs.tsv` (Blocking candidate verification file)

### Step 3: Run the Official Submission Validator
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
Verification passes with exit code 0 (`PASS`).

---

## 5. Verification & Benchmark Summary

| System | Macro F0.5 | Precision | Recall | Singleton Accuracy |
|---|:---:|:---:|:---:|:---:|
| Baseline 1 (Raw Exact) | 0.05587 | 0.00005 | 0.00001 | 1.00000 |
| Baseline 2A (Normalized Exact) | 0.08370 | 0.04560 | 0.01347 | 1.00000 |
| Baseline 2B (Norm Name Only) | 0.32161 | 0.40117 | 0.21958 | 0.64190 |
| Baseline 2C (Legal-Stripped Name) | 0.40452 | 0.46832 | 0.42761 | 0.38675 |
| Baseline 2D (Token-Sorted Name) | 0.41724 | 0.48015 | 0.44928 | 0.37422 |
| **Final Solution (DeepResolve ER)** | **0.81172** | **0.88801** | **0.66169** | **0.89933** |
