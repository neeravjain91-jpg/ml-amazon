# Workspace Inventory Report (Updated)

**Date**: September 2026  
**Auditor**: Senior ML Engineering & Research Team  
**Workspace Path**: `C:\Users\ASUS\Downloads\ml amazon`  

---

## 1. Directory Structure & Verified Files

```
c:\Users\ASUS\Downloads\ml amazon/
├── dataset/ -> student_resource/dataset [Directory Junction]
│   ├── train/
│   │   ├── train_source1.tsv       (210,069,713 bytes, ~210 MB)
│   │   ├── train_source2.tsv       (489,301,488 bytes, ~489 MB)
│   │   ├── train_source3.tsv       (503,705,637 bytes, ~504 MB)
│   │   └── train_ground_truth.tsv  (127,015,583 bytes, ~127 MB)
│   └── test/
│       ├── test_source1.tsv        (175,022,086 bytes, ~175 MB)
│       ├── test_source2.tsv        (509,456,422 bytes, ~509 MB)
│       └── test_source3.tsv        (506,002,772 bytes, ~506 MB)
├── docs/
│   └── problem_statement.pdf       (173,220 bytes)
├── output/                         [Target for matching_results.tsv and candidate_pairs.tsv]
├── reports/
│   ├── challenge_spec.md
│   ├── workspace_inventory.md
│   └── ... (Audit and analysis reports)
├── src/                            [Pipeline source modules]
├── student_resource/               [Extracted official competition package]
├── utils/
│   └── validate_submission.py      (Official submission validator)
├── Documentation_template.md       (Official methodology document)
└── official_README.md              (Official instructions & rules)
```

---

## 2. Complete Inventory Status Table

| File Path | Description | Verified File Size | Status |
|---|---|---|---|
| `docs/problem_statement.pdf` | Official problem statement specification | 173,220 B | **Verified** |
| `dataset/train/train_source1.tsv` | S1 deduplicated reference records | 210,069,713 B | **Verified** |
| `dataset/train/train_source2.tsv` | S2 candidate business records | 489,301,488 B | **Verified** |
| `dataset/train/train_source3.tsv` | S3 candidate business records | 503,705,637 B | **Verified** |
| `dataset/train/train_ground_truth.tsv`| S1 ground truth matches | 127,015,583 B | **Verified** |
| `dataset/test/test_source1.tsv` | S1 evaluation query records | 175,022,086 B | **Verified** |
| `dataset/test/test_source2.tsv` | S2 test candidate records | 509,456,422 B | **Verified** |
| `dataset/test/test_source3.tsv` | S3 test candidate records | 506,002,772 B | **Verified** |
| `utils/validate_submission.py` | Official submission verification script | 13,687 B | **Verified** |
| `Documentation_template.md` | Official methodology template | 2,175 B | **Verified** |

---

## 3. Environment & Runtime Audit

* **Python Runtime**: Python 3.11.9
* **Pre-installed ML / Data Engineering Libraries**:
  * `numpy`: 2.4.6 (BSD-3 License)
  * `pandas`: 2.3.3 (BSD-3 License)
  * `scipy`: 1.17.1 (BSD-3 License)
  * `scikit-learn`: 1.8.0 (BSD-3 License)
  * `lightgbm`: 4.7.0 (MIT License)
  * `xgboost`: 3.2.0 (Apache-2.0 License)
  * `catboost`: 1.2.10 (Apache-2.0 License)
  * `torch`: 2.13.0+cpu (BSD-3 License)
  * `tqdm`: 4.70.0 (MPL-2.0 / MIT)
* **Licensing Compliance Verification**: All candidate ML frameworks strictly comply with MIT / Apache 2.0 / permissive BSD requirements. Parameter counts for tree-based ensemble models are well under the 8B parameter ceiling.

Workspace is fully populated, verified, and ready for Phase 2.
