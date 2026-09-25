# Final Clean Mode Selection: Mode F (Token Sorting Representation)
**ML Challenge 2026 — Business Entity Resolution**  
**Phase 21 Architecture Decision Record**  

---

## 1. Executive Summary & Selection Decision

Following the comprehensive data transformation ablation suite conducted across 5,000 stratified validation entities under Retrieval V3 ($K_{ret}=120$), **Mode F (Token Sorting Representation)** has been decisively selected as the primary production canonical configuration.

### Formal Performance Comparison

| Configuration | Candidate Recall | Precision | Recall | **Macro F0.5** | US F0.5 | India F0.5 | Singleton Acc | Multi-Match F0.5 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Raw Baseline V3 (K120)** | 73.58% | 0.88197 | 0.64647 | **0.80048** | 0.86485 | 0.70883 | 0.87919 | 0.80576 |
| **Mode H (Full Canonical)** | 75.25% | 0.88882 | 0.66765 | **0.81367** | 0.88162 | 0.71692 | 0.88926 | 0.82014 |
| **Mode F (Token Sorting - WINNER)** | **75.25%** | **0.89121** | **0.67336** | **0.81745** | **0.88366** | **0.72320** | **0.88926** | **0.82427** |
| **Delta (Mode F vs Baseline)** | **+1.67%** | **+0.00924** | **+0.02689** | **+0.01697** | **+0.01881** | **+0.01437** | **+0.01007** | **+0.01851** |
| **Delta (Mode F vs Mode H)** | **0.00%** | **+0.00239** | **+0.00571** | **+0.00378** | **+0.00204** | **+0.00628** | **0.00000** | **+0.00413** |

> [!IMPORTANT]
> **Documentation Correction**: Mode H ($F_{0.5} = 0.81367$) was an intermediate milestone demonstrating that canonical cleaning outperforms raw strings. However, **Mode F ($F_{0.5} = 0.81745$) is the true empirically validated winner** across every single evaluation slice (overall, US, India, Multi-Match). Mode H is officially superseded by Mode F.

---

## 2. Complete Controlled Ablation Suite (Modes A to H)

All ablations were benchmarked on the identical 5,000 Source 1 validation population (17,164 ground truth matches) using the frozen LightGBM model and decision policy ($\tau=0.94, \Delta=0.05, K_{match} \le 5$):

| Mode | Transformation Description | Precision | Recall | **Macro F0.5** | US F0.5 | India F0.5 | Singleton Acc | Multi-Match F0.5 |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A** | Raw Only (Basic Lowercase) | 0.56679 | 0.61703 | **0.53738** | 0.59033 | 0.46199 | 0.18456 | 0.57247 |
| **B** | Unicode / Whitespace Hygiene | 0.58808 | 0.62973 | **0.55608** | 0.60836 | 0.48165 | 0.19463 | 0.59142 |
| **C** | Punctuation Normalization | 0.77429 | 0.69855 | **0.72007** | 0.77744 | 0.63840 | 0.42953 | 0.75072 |
| **D** | Abbreviation Normalization (`&` / Rd / St) | 0.77475 | 0.69843 | **0.72026** | 0.77744 | 0.63886 | 0.42953 | 0.75093 |
| **E** | Legal-Suffix Representation | 0.88882 | 0.66765 | **0.81367** | 0.88162 | 0.71692 | 0.88926 | 0.82014 |
| **F** | **Token Sorting Representation (WINNER)** | **0.89121** | **0.67336** | **0.81745** | **0.88366** | **0.72320** | **0.88926** | **0.82427** |
| **G** | Numeric / Address Normalization | 0.88882 | 0.66765 | **0.81367** | 0.88162 | 0.71692 | 0.88926 | 0.82014 |
| **H** | Full Canonical Cleaned Representation | 0.88882 | 0.66765 | **0.81367** | 0.88162 | 0.71692 | 0.88926 | 0.82014 |

---

## 3. Deep Architectural Analysis: Why Mode F Outperforms Mode H

Mode F is not a blanket substitution of strings with sorted tokens; it is a **highly surgical projection assignment**:

### Exact Projection Mapping in Mode F:
```
Pairwise Record Field      Canonical Record Source Column         Design Rationale
-----------------------------------------------------------------------------------------------------------------------------
r["raw_name"]       <---   canon_rec["name_unicode"]             Preserves true raw casing/unicode for strict raw exact match
r["norm_name"]      <---   canon_rec["name_token_sorted"]        ACTIVE TOKEN SORTING: Eliminates word-order permutations
r["strip_name"]     <---   canon_rec["name_legal_stripped"]      Preserves natural token sequence for character 3-gram Dice
r["raw_addr"]       <---   canon_rec["address_unicode"]          Preserves raw address for raw exact match
r["norm_addr"]      <---   canon_rec["address_clean"]            Expanded street abbreviations without token reordering
r["nums"]           <---   canon_rec["address_numeric_tokens"]   Isolated numeric tokens with leading zeros stripped
r["country"]        <---   canon_rec["country_clean"]            Open-set standardized country string
```

### The Mechanism of Superiority:
1. **The Word-Order Invariance Trap in Mode H**:
   In business registries, corporate entities frequently appear with rearranged word order, e.g., `"Amazon Web Services LLC"` vs `"Amazon Services Web"`, or `"Hospital Memorial Valley"` vs `"Valley Memorial Hospital"`. In Mode H, `norm_name` is sequential (`name_clean`), which forces `name_norm_exact = 0.0`.
2. **Precision & Recall Unlocked**:
   In Mode F, projecting `norm_name` to `name_token_sorted` makes `name_norm_exact = 1.0` whenever the same set of words is present in any permutation. This increases true positive scores into the $\ge \tau$ range without degrading precision.
3. **Preservation of N-Gram Discriminability**:
   Importantly, `strip_name` is **NOT** token sorted. It retains sequential ordering, ensuring that `name_dice_3gram` and `name_prefix_ratio` continue to capture character-level phonetic and typographical structure without introducing artificial boundary n-grams.

---

## 4. Complete Canonical Data Coverage

Canonical cleaned representations exist consistently across all 6 raw sources:

| Source Partition | Role | Format | Row Count | Unique Entity IDs | Required Raw Columns | Canonical Columns | Missingness (Nulls) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `train_source1` | Reference (S1) | Parquet | 2,206,821 | 2,206,821 (100%) | 4 | 20 | 0 (0.00%) |
| `train_source2` | Target (S2) | Parquet | 5,034,616 | 5,034,616 (100%) | 4 | 20 | 0 (0.00%) |
| `train_source3` | Target (S3) | Parquet | 5,285,603 | 5,285,603 (100%) | 4 | 20 | 0 (0.00%) |
| `test_source1` | Reference (S1) | Parquet | 1,732,544 | 1,732,544 (100%) | 4 | 20 | 0 (0.00%) |
| `test_source2` | Target (S2) | Parquet | 4,887,273 | 4,887,273 (100%) | 4 | 20 | 0 (0.00%) |
| `test_source3` | Target (S3) | Parquet | 5,082,316 | 5,082,316 (100%) | 4 | 20 | 0 (0.00%) |

All datasets strictly follow the identical 20-column canonical schema defined in [`reports/data_cleaning_spec.md`](file:///c:/Users/ASUS/Downloads/ml%20amazon/reports/data_cleaning_spec.md).

---

## 5. Decision & Authorization

- **Primary Pipeline**: Retrieval V3 ($K_{ret}=120$) + Mode F Canonical Representation + Frozen LightGBM + ($\tau=0.94, \Delta=0.05, K_{match} \le 5$).
- **Frozen Fallback**: `deepresolve_er_submission.zip` remains 100% untouched and byte-verified.
- **Productionization**: `src/inference_v3.py` has been updated with the verified Mode F logic.
