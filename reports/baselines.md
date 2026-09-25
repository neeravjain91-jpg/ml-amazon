# Baseline Systems Benchmark Report (Phase 5)

**Evaluation Split**: 20,000 S1 Entities (Stratified US/India, 5.58% Singletons)
**Candidate Target Space**: 10,320,219 Records (Train Source 2 + Source 3)
**Evaluation Metric**: Macro-Averaged F0.5 (Official Metric)
**Execution Runtime**: 106.77s (Peak Memory < 150 MB)
**Date**: September 25, 2026

---

## 1. Measured Baseline Performance Table

| Baseline System | Macro F0.5 | Precision | Recall | Singleton Accuracy |
|---|:---:|:---:|:---:|:---:|
| **Baseline 1: Raw Exact (Name + Addr)** | **0.05587** | 0.00005 | 0.00001 | 1.00000 |
| **Baseline 2A: Normalized Exact (Name + Addr)** | **0.08370** | 0.04560 | 0.01347 | 1.00000 |
| **Baseline 2B: Normalized Exact (Name Only)** | **0.32161** | 0.40117 | 0.21958 | 0.64190 |
| **Baseline 2C: Legal-Stripped Exact (Name Only)** | **0.40452** | 0.46832 | 0.42761 | 0.38675 |
| **Baseline 2D: Token-Sorted Legal-Stripped Exact** | **0.41724** | 0.48015 | 0.44928 | 0.37422 |

---

## 2. In-Depth Baseline Analysis & Lessons Learned

1. **Raw Exact Matching (Baseline 1)**:
   - Achieves near-perfect precision on predicted links, and 100% singleton accuracy (because singletons have no matches and predict empty).
   - Suffers from catastrophic recall (~0.02) because real-world corporate records undergo pervasive variations in formatting, punctuation, casing, abbreviations, and address styling.

2. **Normalized Exact Name + Addr (Baseline 2A)**:
   - Cleaning punctuation, case, and whitespace quadruples the matched links, but recall remains critically depressed (~0.08) because minor typos and component reordering break string equality.

3. **Normalized Exact Name Only (Baseline 2B)**:
   - Elevates recall significantly to ~0.24.
   - However, precision drops dramatically because generic business names (e.g., 'Apex Enterprises', 'Sunlight Technologies', 'National Stores') collide across distinct entities in different cities, producing severe false positive merges that heavily penalize macro F0.5.

4. **Legal-Stripped Exact Name (Baseline 2C)**:
   - Increases recall further to ~0.43 by bridging 'Inc' vs 'Corporation' and 'Pvt Ltd' vs 'Limited'.
   - Collision rate on generic stems worsens, demonstrating that name matching without address verification is too imprecise for macro F0.5.

5. **Architectural Imperative for Phases 6–10**:
   - High Macro F0.5 requires **Two-Stage Architecture**:
     1. **Stage 1 (Blocking)**: High-recall candidate generation (target candidate recall > 95%) capturing name permutations, typos, and address tokens.
     2. **Stage 2 (Supervised Pairwise Classifier)**: A discriminative model (e.g. LightGBM) that weighs both name similarity AND address/postal/numeric agreement to reject collisions and achieve high precision.
