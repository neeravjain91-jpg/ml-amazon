# Controlled Data Transformation Ablation Suite

All ablations were conducted on the exact 5,000-entity validation split under Retrieval V3 ($K_{ret}=120$) using the frozen LightGBM pairwise classifier and decision policy ($\tau=0.94, \Delta=0.05, K_{match} \le 5$):

| Mode | Transformation Description | Precision | Recall | **Macro F0.5** | US F0.5 | India F0.5 | Singleton Acc | Multi-Match F0.5 |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A** | Raw Only (Basic Lowercase) | 0.56679 | 0.61703 | **0.53738** | 0.59033 | 0.46199 | 0.18456 | 0.57247 |
| **B** | Unicode / Whitespace Hygiene | 0.58808 | 0.62973 | **0.55608** | 0.60836 | 0.48165 | 0.19463 | 0.59142 |
| **C** | Punctuation Normalization | 0.77429 | 0.69855 | **0.72007** | 0.77744 | 0.63840 | 0.42953 | 0.75072 |
| **D** | Abbreviation Normalization (& / Rd / St) | 0.77475 | 0.69843 | **0.72026** | 0.77744 | 0.63886 | 0.42953 | 0.75093 |
| **E** | Legal-Suffix Representation | 0.88882 | 0.66765 | **0.81367** | 0.88162 | 0.71692 | 0.88926 | 0.82014 |
| **F** | Token Sorting Representation | 0.89121 | 0.67336 | **0.81745** | 0.88366 | 0.72320 | 0.88926 | 0.82427 |
| **G** | Numeric / Address Normalization | 0.88882 | 0.66765 | **0.81367** | 0.88162 | 0.71692 | 0.88926 | 0.82014 |
| **H** | Full Canonical Cleaned Representation | 0.88882 | 0.66765 | **0.81367** | 0.88162 | 0.71692 | 0.88926 | 0.82014 |

