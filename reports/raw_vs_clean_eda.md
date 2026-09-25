# Cleaned Data Exploratory Data Analysis (EDA)

## 1. Text Length and Token Statistics (Raw vs Clean)

| Source File | Raw Name Len | Clean Name Len | Name Tokens | Raw Addr Len | Clean Addr Len | Addr Tokens | Numeric Tokens | Postal PIN Coverage |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `train_source1.tsv` | 24.0 | 23.9 | 3.62 | 52.1 | 48.9 | 8.51 | 1.35 | **6.74%** |
| `train_source2.tsv` | 25.1 | 24.2 | 4.25 | 46.2 | 43.1 | 7.69 | 1.29 | **7.38%** |
| `train_source3.tsv` | 25.2 | 24.5 | 4.03 | 46.8 | 43.7 | 7.55 | 1.30 | **7.24%** |
| `test_source1.tsv` | 23.8 | 23.7 | 3.58 | 57.1 | 53.7 | 9.33 | 1.35 | **4.37%** |
| `test_source2.tsv` | 25.7 | 24.7 | 4.48 | 50.3 | 46.6 | 8.34 | 1.35 | **4.98%** |
| `test_source3.tsv` | 25.7 | 25.0 | 4.15 | 48.7 | 45.1 | 8.04 | 1.33 | **5.16%** |

## 2. Country Distribution Across Sources

### `train_source1.tsv`
- **US**: 59,890 records (59.89%)
- **India**: 40,110 records (40.11%)

### `train_source2.tsv`
- **US**: 59,936 records (59.94%)
- **India**: 40,064 records (40.06%)

### `train_source3.tsv`
- **US**: 59,529 records (59.53%)
- **India**: 40,471 records (40.47%)

### `test_source1.tsv`
- **India**: 46,598 records (46.60%)
- **US**: 38,419 records (38.42%)
- **France**: 14,983 records (14.98%)

### `test_source2.tsv`
- **India**: 47,328 records (47.33%)
- **US**: 38,169 records (38.17%)
- **France**: 14,503 records (14.50%)

### `test_source3.tsv`
- **India**: 47,307 records (47.31%)
- **US**: 38,466 records (38.47%)
- **France**: 14,227 records (14.23%)

