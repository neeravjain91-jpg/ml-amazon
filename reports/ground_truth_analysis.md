# Ground-Truth Forensic Analysis Report (Phase 3)

**Dataset**: `dataset/train/train_ground_truth.tsv`  
**Auditor**: Senior ML Engineering & Research Team  
**Audit Date**: September 25, 2026  

---

## 1. Executive Ground-Truth Summary

| Metric | Measured Value | Percentage / Mean |
|---|---|---|
| **Total Source 1 Entities** | 2,206,821 | 100.0% |
| **Total Ground-Truth Matches** | 7,638,365 | - |
| **Mean Matches per S1 Entity** | 3.4613 | - |
| **Median Matches per S1 Entity** | 3.0 | - |
| **Maximum Matches for a single S1** | 11 | - |
| **Minimum Matches (Singletons)** | 0 | - |
| **Total Singletons (0 Matches)** | 123,247 | **5.58%** |
| **Total Entities with $\ge 1$ Match** | 2,083,574 | **94.42%** |
| **Total S2 Matches** | 3,693,619 | 48.36% of all matches |
| **Total S3 Matches** | 3,944,746 | 51.64% of all matches |

---

## 2. Match Cardinality Distribution ($k$-Matches per S1)

The table below gives the exact distribution of how many matching entities in Source 2 and/or Source 3 correspond to a given Source 1 reference entity:

| Matches ($k$) | S1 Entity Count | Percentage | Cumulative % | Strategic Implication |
|:---:|:---:|:---:|:---:|---|
| **0** (Singleton) | 123,247 | 5.58% | 5.58% | Must predict empty set $\emptyset$; worth 1.0 on $F_{0.5}$ |
| **1** | 119,157 | 5.40% | 10.98% | Exactly 1 match |
| **2** | 375,212 | 17.00% | 27.98% | Multiple match ($1:2$) |
| **3** | 530,841 | 24.05% | 52.03% | Modal match count |
| **4** | 484,115 | 21.94% | 73.97% | Multiple match ($1:4$) |
| **5** | 321,957 | 14.59% | 88.56% | Multiple match ($1:5$) |
| **6** | 164,868 | 7.47% | 96.03% | Multiple match ($1:6$) |
| **7** | 63,968 | 2.90% | 98.93% | Multiple match ($1:7$) |
| **8** | 18,680 | 0.85% | 99.78% | Multiple match ($1:8$) |
| **9** | 4,205 | 0.19% | 99.97% | Multiple match ($1:9$) |
| **10** | 534 | 0.02% | 99.99% | Extreme multi-match |
| **11** | 37 | 0.001% | 100.00% | Maximum observed multi-match |

### Key Cardinality Insight:
* **89.02%** of all Source 1 entities have **2 or more matches** ($k \ge 2$).
* Standard $1:1$ matching algorithms or `argmax` selection would discard over **65% of all valid links**, capping recall at under $0.35$ and destroying macro $F_{0.5}$.
* Multi-match set optimization is an essential, primary requirement.

---

## 3. Cross-Source Breakdown: S2 vs. S3 Overlap

How do the matches distribute across Source 2 and Source 3?

| Match Structure | S1 Entity Count | Percentage of S1 | Interpretation |
|---|---|---|---|
| **Both S2 and S3 Matches** | 1,776,047 | **80.48%** | Entity has counterparts in BOTH S2 and S3 |
| **Only S3 Matches** | 164,498 | **7.45%** | Entity resolved only in Source 3 |
| **Only S2 Matches** | 143,029 | **6.48%** | Entity resolved only in Source 2 |
| **Singletons (Neither)** | 123,247 | **5.58%** | Entity has no match in either S2 or S3 |
| **TOTAL** | **2,206,821** | **100.00%** | |

* Over 80% of entities appear in all three data sources simultaneously.
* Source 3 contains slightly more matching targets (3,944,746) than Source 2 (3,693,619), reflecting its larger raw record count (5.28M vs 5.03M).

---

## 4. Country Partition Invariance & Distribution

| Country | Total S1 Entities | Singletons | Singleton % | Total Matches | Mean Matches / S1 |
|---|---|---|---|---|---|
| **United States (US)** | 1,323,633 | 73,896 | **5.58%** | 4,578,522 | **3.46** |
| **India** | 883,188 | 49,351 | **5.59%** | 3,059,843 | **3.46** |

* **Zero Cross-Country Matches**: Exactly 0 out of 7,638,365 matches cross between US and India.
* **Invariant Generative Process**: Both US and India exhibit an identical 5.58% singleton rate and an identical 3.46 average match multiplicity. This proves the underlying data generation process is identical across countries, providing high confidence that the open-set test country (**France**) will follow the same mathematical properties.
