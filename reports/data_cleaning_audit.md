# Data Cleaning & Hygiene Audit Report

## 1. Raw vs Clean Audit Summary

| Source File | Rows Audited | Unique IDs | Duplicate IDs | Missing Name | Missing Addr | Missing Country | Control Char Rows | Records Changed (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `train_source1.tsv` | 500,000 | 500,000 | 0 | 0 | 0 | 0 | 149 | 353 (0.07%) |
| `train_source2.tsv` | 500,000 | 500,000 | 0 | 0 | 16614 | 0 | 2648 | 160,104 (32.02%) |
| `train_source3.tsv` | 500,000 | 500,000 | 0 | 0 | 16431 | 0 | 1430 | 135,750 (27.15%) |
| `test_source1.tsv` | 500,000 | 500,000 | 0 | 0 | 0 | 0 | 181 | 29,057 (5.81%) |
| `test_source2.tsv` | 500,000 | 500,000 | 0 | 0 | 13264 | 0 | 3324 | 185,945 (37.19%) |
| `test_source3.tsv` | 500,000 | 500,000 | 0 | 0 | 13373 | 0 | 1819 | 160,977 (32.20%) |

## 2. Transformation Counts by Category

| Source File | Whitespace Normalization | Unicode / Control Fixes | Punctuation Cleanup | Abbreviation (`&` / Rd / St) | Legal Suffix Stripped | Token Sorting Shift |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `train_source1.tsv` | 201 | 353 | 500,000 | 54,326 | 352,687 | 445,816 |
| `train_source2.tsv` | 64,382 | 160,104 | 489,300 | 162,562 | 292,601 | 435,432 |
| `train_source3.tsv` | 55,128 | 135,750 | 489,476 | 155,606 | 306,167 | 429,281 |
| `test_source1.tsv` | 224 | 29,057 | 500,000 | 50,310 | 346,586 | 439,223 |
| `test_source2.tsv` | 58,749 | 185,945 | 491,236 | 120,965 | 287,907 | 439,179 |
| `test_source3.tsv` | 51,482 | 160,977 | 491,138 | 116,617 | 306,071 | 432,428 |
