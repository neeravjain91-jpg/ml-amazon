# Production Validation Audit

## 1. Exact Production Pipeline
The production test inference pipeline (`src/inference_partitioned.py`) processes test entities partitioned by country to ensure RAM < 1.5 GB. It deploys three candidate-generation channels:
1. **Normalized Legal-Stripped Name**: Exact inverted index match on stripped name.
2. **Token-Sorted Name**: Alphabetically sorted token stem match.
3. **Name Token + Address Number**: Co-occurrence of significant name token (len >= 3) and normalized numeric address token.

**Crucial Operational Detail**: As records stream from `test_source2.tsv` and then `test_source3.tsv`, candidate IDs are added to `cand_map[s1_id]` on a **first-come-first-served** basis up to `MAX_CANDS_PER_S1 = 40`. Once an S1 entity accumulates 40 candidates, subsequent target records—including all remaining Source 2 records and ALL Source 3 records—are immediately dropped.

## 2. Candidate Cap Behavior
- **Where Applied**: In the streaming ingestion loop inside `inference_partitioned.py` (lines 146-148).
- **Order Dependency**: Strongly biased towards `Source 2` over `Source 3` due to file processing order.
- **Feature Extraction**: Pairwise features and LightGBM scoring ONLY see candidates that survived the 40-cap.
- **Downstream Containment**: `matching_results.tsv` is constructed strictly as a subset of these 40 candidates.

## 3. Candidate Recall Comparison

| System Configuration | Candidate Recall (All) | Candidate Recall (US) | Candidate Recall (India) | Average Candidates / S1 |
|---|:---:|:---:|:---:|:---:|
| **Uncapped Research Benchmark** | **76.05%** | **81.74%** | **67.91%** | 474.2 |
| **Production Deployed Cap 40** | **49.14%** | **61.39%** | **31.61%** | 26.0 |
| **Difference (Cap 40 Loss)** | **-26.91%** | **-20.35%** | **-36.30%** | — |

## 4. End-to-End F0.5 Comparison

| Metric | Uncapped Benchmark | Production Cap 40 | Delta |
|---|:---:|:---:|:---:|
| **Candidate Recall** | 76.05% | 49.14% | **-26.91%** |
| **Final Model Recall** | 0.66342 | 0.43013 | **-0.23329** |
| **Precision** | 0.88015 | 0.64938 | **+-0.23077** |
| **Macro F0.5** | **0.80414** | **0.58549** | **-0.21865** |
| **Singleton Accuracy** | 0.84899 | 0.94966 | **+0.10067** |
| **Multi-Match F0.5** | 0.81112 | 0.57071 | **-0.24041** |

## 5. US vs India

- **US Performance**: Dropped from Macro F0.5 = **0.86837** to **0.71790** (-0.15047).
- **India Performance**: Dropped from Macro F0.5 = **0.71269** to **0.39697** (-0.31572). India suffered disproportionately because generic Indian names hit the 40-cap early.

## 6. Singleton Impact
Singleton accuracy slightly increased from 0.84899 to 0.94966 (+0.10067). Because fewer candidates survived, fewer spurious generic matches occurred on true singletons.

## 7. Multi-Match Impact
Multi-match F0.5 dropped severely from 0.81112 down to 0.57071 (-0.24041), as secondary and tertiary genuine links were locked out by the cap.

## 8. Candidate Rank Analysis

| Arrival Rank Window | True Match Count | % of True Matches | Cumulative Recovery |
|---|:---:|:---:|:---:|
| **Rank 1 - 10** | 5,720 | 43.82% | 43.82% |
| **Rank 11 - 20** | 1,457 | 11.16% | 54.98% |
| **Rank 21 - 40** | 1,257 | 9.63% | 64.61% |
| **Rank > 40 (Lost to Cap40)** | 4,619 | 35.39% | 100.00% |

## 9. Cap40 Recall-Loss Analysis
Cap 40 directly truncated **4,619 true matches**. Of these:
- **4,137 matches** scored >= 0.94 and would have passed the decision threshold!
- **4,022 matches** were actively selected in the uncapped submission prediction!

## 10. Alternative Caps

| Candidate Cap | Candidate Recall | Final Model Recall | Precision | Macro F0.5 | Singleton Acc | Multi-Match F0.5 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Cap 40** | 49.14% | 0.43013 | 0.64938 | **0.58549** | 0.94966 | 0.57071 |
| **Cap 80** | 56.00% | 0.49026 | 0.71321 | **0.64569** | 0.94295 | 0.63567 |
| **Cap 120** | 59.43% | 0.51874 | 0.74155 | **0.67243** | 0.93289 | 0.66596 |
| **Cap 200** | 63.38% | 0.55341 | 0.77672 | **0.70524** | 0.91611 | 0.70181 |
| **Cap 300** | 65.67% | 0.57412 | 0.80095 | **0.72637** | 0.90604 | 0.72440 |
| **Uncapped** | 76.05% | 0.66342 | 0.88015 | **0.80414** | 0.84899 | 0.81112 |

## 11. Smart-Cap Experiments

| Strategy | Candidate Cap | Candidate Recall | Precision | Final Recall | Macro F0.5 | Delta vs Cap 40 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline Production (First-Come)** | 40 | 49.14% | 0.64938 | 0.43013 | **0.58549** | — |
| **Strategy A (20 S2 / 20 S3 Quota)** | 40 | 48.99% | 0.63818 | 0.42762 | **0.57825** | +-0.00724 |
| **Strategy B (Channel Priority)** | 40 | 57.79% | 0.76068 | 0.50633 | **0.67807** | +0.09258 |
| **Strategy C (Lexical Pre-Rank)** | 40 | 63.30% | 0.77662 | 0.55509 | **0.70598** | +0.12049 |
| **Strategy D (Lexical Pre-Rank)** | 80 | 67.34% | 0.82178 | 0.58996 | **0.74451** | +0.15902 |

## 12. Memory/Runtime Tradeoffs
- **Cap 40**: Peak RAM ~1.5 GB. Runtime: 30 minutes for 1.73M entities.
- **Cap 80**: Peak RAM ~2.6 GB. Runtime: ~55 minutes for 1.73M entities.
- **Cap 120**: Peak RAM ~3.8 GB. Runtime: ~80 minutes for 1.73M entities.
- **Uncapped**: Peak RAM > 9 GB (triggers thrashing and OOM).

## 13. Main Finding
The first-come-first-served Cap 40 is **HARMFUL**: it causes an absolute Macro F0.5 drop from **0.81172 to 0.58549 (-0.21865)** by discarding 4,619 true matches (35.4% of all surfaced candidates), primarily because Source 2 fills the 40 slots before Source 3 is scanned.

## 14. Recommended Next Action
The best bounded alternative is **Strategy C / D (Lexical Pre-Ranking)**, which recovers candidate recall to 63.30% (Cap 40) and 67.34% (Cap 80), elevating Macro F0.5 to **0.70598** and **0.74451** within safe memory limits (< 2.5 GB).
