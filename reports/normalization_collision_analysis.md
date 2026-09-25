# Normalization Collision Analysis

## 1. Representation Collision Rates (100,000 Sample per Source)

| Source File | Raw Name Unique | Clean Name Unique | Clean Collision % | Legal Stripped Unique | Legal Collision % | Token Sorted Unique | Sort Collision % |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `train_source1.tsv` | 90,403 | 89,891 | 10.11% | 80,738 | 19.26% | 89,877 | 10.12% |
| `train_source2.tsv` | 98,731 | 97,393 | 2.61% | 91,713 | 8.29% | 97,086 | 2.91% |
| `train_source3.tsv` | 98,835 | 97,613 | 2.39% | 90,923 | 9.08% | 97,267 | 2.73% |
| `test_source1.tsv` | 90,433 | 90,110 | 9.89% | 80,905 | 19.10% | 90,092 | 9.91% |
| `test_source2.tsv` | 98,817 | 98,013 | 1.99% | 93,395 | 6.60% | 97,787 | 2.21% |
| `test_source3.tsv` | 98,901 | 98,053 | 1.95% | 91,985 | 8.02% | 97,783 | 2.22% |

## 2. Ground-Truth Match Coverage vs False-Collision Tradeoff

- **True Matched Pairs Evaluated**: 17,164
- **Sampled False Negative Pairs Evaluated**: 50,000

| Representation / Feature | True Match Coverage (Recall Ceil) | False Pair Collision (Precision Risk) | Signal-to-Noise Ratio (Coverage / Collision) |
|---|:---:|:---:|:---:|
| **`raw_name_exact`** | **4.50%** | 0.000% | **999.0x** |
| **`clean_name_exact`** | **25.60%** | 0.000% | **999.0x** |
| **`legal_name_exact`** | **49.78%** | 0.002% | **24889.3x** |
| **`sort_name_exact`** | **31.16%** | 0.000% | **999.0x** |
| **`raw_addr_exact`** | **2.15%** | 0.000% | **999.0x** |
| **`clean_addr_exact`** | **11.29%** | 0.000% | **999.0x** |
| **`num_agree`** | **78.92%** | 1.082% | **72.9x** |
| **`postal_agree`** | **4.64%** | 0.000% | **999.0x** |
