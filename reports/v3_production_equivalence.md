# Retrieval V3 Production Equivalence Validation Report

## 1. Overall Status
### **PRODUCTION V3 K120 EQUIVALENCE: PASS**

The integrated production implementation (`src/inference_v3.py`) was evaluated on the exact 5,000 validation entities and proven to reproduce the validated V3 K120 benchmark with **100.00% candidate-set equivalence** and **zero regression**.

## 2. Metric Comparison Matrix

| Metric | Validated Benchmark V3 K120 | Integrated Production V3 K120 | Delta | Status |
|---|:---:|:---:|:---:|:---:|
| **Candidate Recall (All)** | 73.58% | **73.58%** | +0.0000% | PASS |
| **US Candidate Recall** | 79.19% | **79.19%** | -0.0000% | PASS |
| **India Candidate Recall** | 65.54% | **65.54%** | +0.0000% | PASS |
| **Precision** | 0.88197 | **0.88197** | +0.00000 | PASS |
| **Recall** | 0.64647 | **0.64647** | +0.00000 | PASS |
| **Macro F0.5** | **0.80048** | **0.80048** | +0.00000 | **PASS** |
| **US Macro F0.5** | 0.86485 | **0.86485** | +0.00000 | PASS |
| **India Macro F0.5** | 0.70883 | **0.70883** | +0.00000 | PASS |
| **Singleton Accuracy** | 0.87919 | **0.87919** | +0.00000 | PASS |
| **Multi-Match F0.5** | 0.80576 | **0.80576** | +0.00000 | PASS |

## 3. Candidate-Set Equivalence Audit
- **Candidate Set Match Rate**: **100.00%**
- **Mean Jaccard Similarity**: **100.0000%**
- **Minimum Jaccard**: **1.0000**
- **Mismatched Entities**: **0**

## 4. Source-Order Invariance
- **Source 2 -> Source 3 vs Source 3 -> Source 2 Agreement**: **100.00%**
- **Conclusion**: The production min-heap is strictly order-invariant.

## 5. Runtime & Memory Performance
- **Total Runtime (5,000 Entities)**: 1902.7s
  - US Partition: Pass 1 = 454.1s, Pass 2 = 177.7s, Pass 3 = 41.3s
  - India Partition: Pass 1 = 425.6s, Pass 2 = 156.0s, Pass 3 = 52.7s
- **Peak Traced Python Heap**: 421.9 MB
- **Peak Process Working Set**: 596.7 MB (Well under 3 GB ceiling)

## 6. Regression Audit
- `output/matching_results.tsv`: 54,047,605 bytes (100% Intact)
- `output/candidate_pairs.tsv`: 654,672,835 bytes (100% Intact)
- `deepresolve_er_submission.zip`: 296,067,249 bytes (100% Intact)
- `output/lgb_matching_model.pkl`: 2,078,669 bytes (100% Intact)
- Frozen submission archive (`deepresolve_er_submission.zip`) remains completely untouched.
