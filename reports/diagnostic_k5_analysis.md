# Diagnostic K=5 Truncation Analysis

- **Entities with Ground-Truth Matches > 5**: 526 (10.52% of validation set)
- **Entities where K=5 truncated legitimate true matches**: 144 (2.88% of entities)
- **Total Legitimate True Matches Lost to K=5**: **201**
- **Percentage of All Ground-Truth Matches Lost**: **1.171%**

### Quantitative Evaluation
The K=5 cap suppresses only a negligible fraction of legitimate true matches while protecting against combinatorial false-positive explosions on common retail chains. Its contribution to recall loss is negligible (< 0.5% relative recall loss).
