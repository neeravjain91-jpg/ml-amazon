# Critical Validation Gate: Canonical Data Layer Benchmark
**ML Challenge 2026 — Business Entity Resolution**  

## **Classification: CLEANING IMPROVES (Mode F Confirmed Winner)**

### Benchmark Comparison (Exact 5,000 Validation Entities)

| Pipeline Stage | Candidate Recall | Precision | Recall | **Macro F0.5** | US F0.5 | India F0.5 | Singleton Accuracy | Multi-Match F0.5 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Before Cleaning (Raw Baseline V3)** | 73.58% | 0.88197 | 0.64647 | **0.80048** | 0.86485 | 0.70883 | 0.87919 | 0.80576 |
| **Intermediate Cleaning (Mode H)** | 75.25% | 0.88882 | 0.66765 | **0.81367** | 0.88162 | 0.71692 | 0.88926 | 0.82014 |
| **Winning Canonical Mode (Mode F — Token Sorting)** | **75.25%** | **0.89121** | **0.67336** | **0.81745** | **0.88366** | **0.72320** | **0.88926** | **0.82427** |
| **Net Gain (Mode F vs Baseline)** | **+1.67%** | **+0.00924** | **+0.02689** | **+0.01697** | **+0.01881** | **+0.01437** | **+0.01007** | **+0.01851** |
| **Net Gain (Mode F vs Mode H)** | **0.00%** | **+0.00239** | **+0.00571** | **+0.00378** | **+0.00204** | **+0.00628** | **0.00000** | **+0.00413** |

### Architectural Conclusion
- The canonical data cleaning layer achieved Macro F0.5 = **0.81745** under Mode F (Token Sorting Representation), decisively outperforming both the uncleaned baseline (**0.80048**) and full canonical Mode H (**0.81367**).
- Mode F resolves word-order variations in entity names without sacrificing character n-gram discriminability on stripped names.
- Mode F is selected as the primary production canonical configuration.
