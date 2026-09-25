# Retrieval V3 Comprehensive Benchmark Report

## 1. Executive Summary
Retrieval V3 completely replaces the first-come-first-served candidate cap (`MAX_CANDS_PER_S1 = 40`) with an online **Multi-Signal Pre-Ranking Min-Heap**. This architecture was benchmarked across a full sweep of candidate retention limits ($K_{ret} \in [40, 60, 80, 100, 120, 160, 200]$) while holding the frozen LightGBM classifier and decision policy ($\tau=0.94, \Delta=0.05, K_{match} \le 5$) strictly constant.

## 2. K Sweep Benchmark Performance Matrix

| Architecture | Retrieval $K_{ret}$ | Candidate Recall (All) | US Cand Recall | India Cand Recall | Precision | Recall | **Macro F0.5** | US F0.5 | India F0.5 | Singleton Acc | Multi-Match F0.5 | Peak RAM (Est) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Production Cap40 (First-Come)** | 40 | 49.14% | 61.39% | 31.61% | 0.64938 | 0.43013 | **0.58549** | 0.71790 | 0.39697 | 0.94966 | 0.57071 | 1500 MB |
| **Uncapped Benchmark** | 9999 | 76.05% | 81.74% | 67.91% | 0.88015 | 0.66342 | **0.80414** | 0.86837 | 0.71269 | 0.84899 | 0.81112 | 9000 MB |
| **V3 K40** | 40 | 70.98% | 77.31% | 61.92% | 0.86994 | 0.62695 | **0.78776** | 0.85814 | 0.68757 | 0.90940 | 0.79059 | 1192 MB |
| **V3 K60** | 60 | 71.75% | 77.76% | 63.15% | 0.87379 | 0.63248 | **0.79112** | 0.85927 | 0.69410 | 0.89262 | 0.79480 | 1562 MB |
| **V3 K80** | 80 | 72.60% | 78.18% | 64.61% | 0.87865 | 0.63936 | **0.79613** | 0.86117 | 0.70354 | 0.88255 | 0.80088 | 1933 MB |
| **V3 K100** | 100 | 73.07% | 78.62% | 65.12% | 0.88085 | 0.64276 | **0.79840** | 0.86322 | 0.70612 | 0.87919 | 0.80365 | 2304 MB |
| **V3 K120** | 120 | 73.58% | 79.19% | 65.54% | 0.88197 | 0.64647 | **0.80048** | 0.86485 | 0.70883 | 0.87919 | 0.80576 | 2675 MB |
| **V3 K160** | 160 | 74.02% | 79.73% | 65.84% | 0.88291 | 0.64978 | **0.80202** | 0.86699 | 0.70952 | 0.87248 | 0.80771 | 3416 MB |
| **V3 K200** | 200 | 74.30% | 80.05% | 66.07% | 0.88256 | 0.65221 | **0.80248** | 0.86727 | 0.71023 | 0.86913 | 0.80812 | 4158 MB |

## 3. Key Quantitative Findings
1. **Massive Recovery over Production Cap40**: Even at $K_{ret}=40$, V3 reaches Macro F0.5 = **0.78776** vs **0.58549** (+0.20227).
2. **Target Attainment (> 0.78)**: At $K_{ret}=120$, V3 achieves Macro F0.5 = **0.80048** (Candidate Recall: **73.58%**), recovering over **85% of the gap to Uncapped (0.80414)** within a lean 2.7 GB RAM budget.
3. **High-Performance Ceiling**: At $K_{ret}=200$, V3 reaches Macro F0.5 = **0.80248** with **74.30% candidate recall** and **India F0.5 = 0.71023**.
4. **Source Starvation Eliminated**: Verification proves 100.00% identical candidate sets regardless of whether Source 2 or Source 3 is streamed first.
