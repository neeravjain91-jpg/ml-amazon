# Cap 40 Recall Loss & Candidate Ordering Analysis

- **Total Candidate True Matches Surfaced in Uncapped System**: 13,053 / 17,164 (76.05%)
- **True Matches Lost Solely to First-Come-First-Served Cap 40**: **4,619** (26.91% of all GT matches)
- **True Matches Lost that Scored >= 0.94**: **4,137**
- **True Matches Lost that Were Actively Predicted in Benchmark**: **4,022**

### True Match Arrival Rank Distribution

| Arrival Rank | Total Count | % of All True Matches | US Count | US % | India Count | India % | Source 2 Count | Source 3 Count |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Rank 1 - 10** | 5,720 | 43.82% | 4,304 | 52.12% | 1,416 | 29.53% | 3,480 | 2,240 |
| **Rank 11 - 20** | 1,457 | 11.16% | 1,055 | 12.78% | 402 | 8.38% | 629 | 828 |
| **Rank 21 - 40** | 1,257 | 9.63% | 843 | 10.21% | 414 | 8.63% | 527 | 730 |
| **Rank > 40 (Lost to Cap40)** | 4,619 | 35.39% | 2,056 | 24.90% | 2,563 | 53.45% | 1,522 | 3,097 |

### Root Mechanism of Loss in Cap 40
1. **Sequential Source Scanning Bias**: `train_source2.tsv` was scanned before `train_source3.tsv`. For entities where Source 2 populated 40 candidates, Source 3 candidates were 100% blocked regardless of matching quality.
2. **High-Degree Name Inundation**: Generic Indian business names collected 40 low-quality co-occurrences in the first 100,000 lines of Source 2, locking out genuine true matches appearing later.
