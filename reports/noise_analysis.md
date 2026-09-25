# Noise-Pattern & Corruption Analysis Report (Phase 4)

**Dataset**: ML Challenge 2026 — Business Entity Resolution  
**Sample Size**: 50,003 Ground-Truth True Pairs across 13,682 Source 1 Entities  
**Auditor**: Senior ML Engineering & Research Team  
**Date**: September 25, 2026  

---

## 1. Quantitative Noise Metrics on True Pairs

| Metric / Feature | True Pair Value | Impact / Analysis |
|---|---|---|
| **Raw Exact Name Match Rate** | **4.58%** | Over 95.4% of real matches exhibit textual mutation in name |
| **Normalized Exact Name Match Rate** | **22.23%** | Lowercasing + punctuation stripping recovers ~17.6% |
| **Legal-Stripped Exact Name Rate** | **42.46%** | Removing corporate suffixes (Inc, Ltd, Pvt, LLC) doubles exact matches |
| **Token-Sorted Exact Name Rate** | **27.25%** | Captures word transpositions and inversion |
| **Mean Name Token Jaccard** | **0.6136** | True matches share ~61% of distinct name words |
| **Mean Name 3-Gram Character Dice** | **0.7511** | Highly discriminative even in the presence of typos/OCR errors |
| **Target Address Missing Rate** | **4.51%** | 4.5% of true matches have empty address in S2/S3 |
| **Raw Exact Address Match Rate** | **2.19%** | Less than 1 in 45 true pairs share exact address strings |
| **Normalized Exact Address Rate** | **8.26%** | Slight improvement with whitespace/punctuation cleanup |
| **Mean Address Token Jaccard** | **0.6228** | Significant token overlap across components |
| **Mean Address 3-Gram Character Dice**| **0.7740** | Strongest address similarity feature |
| **Mean Address Number Jaccard** | **0.8109** | **Strongest discriminative signal** in addresses |

---

## 2. Identified Noise Taxonomy & Empirical Evidence

### 2.1 Typographical & OCR Character Corruptions
* **Empirical Example**:
  * S1: `Gutierrez Service of Mesa Inc` $\longleftrightarrow$ Target: `Inc 6utierrez Service of Mesa` (`'G'` replaced by `'6'`).
  * S1: `10824 Edgewood Road` $\longleftrightarrow$ Target: `1082 EDGEWOOD ROAD` (truncated digit).
  * S1: `2121 Main Street` $\longleftrightarrow$ Target: `2121 MADN STREET` (`'i'` replaced by `'d'`).
* **Implication**: Exact string token matching fails on these records. Sub-string character n-grams (3-grams, 4-grams) and edit distance (Levenshtein) are mandatory.

### 2.2 Corporate Suffix & Business Extension Variations
* **Empirical Example**:
  * S1: `Modern Precision Laboratories` $\longleftrightarrow$ Target: `MODERN PRECISION LABORATORIES CORPORATION`.
  * S1: `Prem & Sons Pvt Ltd` $\longleftrightarrow$ Target: `Prem & Pvt Ltd Services` / `Prem & Pvt Ltd Partners`.
* **Implication**: Suffixes (`Inc`, `LLC`, `Corp`, `Pvt`, `Ltd`, `Services`, `Partners`) are fluid across sources. The core stem ("Prem", "Modern Precision Laboratories") carries the entity identity.

### 2.3 Token Permutation & Parenthetical Inversion
* **Empirical Example**:
  * S1: `Kerala (India) Granites Limited` $\longleftrightarrow$ Target: `KERALA GRANITES (INDIA) LIMITED`.
* **Implication**: Word ordering shifts frequently due to database entry formatting. `token_sort_ratio` and bag-of-words / TF-IDF representations are immune to these permutations.

### 2.4 Address Formatting, Landmark & Unit Variations
* **Empirical Example**:
  * S1: `2121 Main Street, Unit 2050, Mesa, AZ` $\longleftrightarrow$ Target: `2121 Main Street, # 2050, Mesa, Arizona`.
  * S1: `2121 Main Street, Unit 2050, Mesa, AZ` $\longleftrightarrow$ Target: `2121 MAIN ST, PMB 4411, MESA, AZ`.
  * S1: `Medak, Telangana, Shivajinagar, Siddipet, 8-2-67/1/A/3/1` $\longleftrightarrow$ Target: `Andhra Pradesh, Siddipet, Shivajinagar, 8-2-67/1/A/3/1, Medak`.
* **Implication**:
  1. Address component ordering is completely arbitrary in Indian and US commercial databases.
  2. Numbers (house numbers, municipal codes, PIN codes like `8-2-67/1/A/3/1`, `2121`) are highly stable identifiers with an 81.1% mean agreement rate.
  3. Numeric token matching provides an anchor for address verification.
