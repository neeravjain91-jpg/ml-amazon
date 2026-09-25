# Model Comparison & Calibration Report (Phases 8–11)

**Model**: LightGBM Pairwise Classifier (MIT Licensed, < 8B params)
**Training Set**: 40,000 S1 Entities with Mined Hard Negatives
**Optimal Threshold**: tau = 0.85
**Validation Macro F0.5**: **0.79506**
**Precision**: 0.85596 | **Recall**: 0.69240
**Singleton Accuracy**: 0.82550

## Top 10 Most Discriminative Features

- `addr_len_ratio`: 2054
- `addr_dice_3gram`: 1959
- `name_addr_dice_prod`: 1603
- `addr_len_diff`: 1499
- `addr_token_jaccard`: 1395
- `name_dice_2gram`: 1128
- `addr_token_containment`: 1126
- `name_prefix_ratio`: 1031
- `addr_shared_token_count`: 926
- `name_dice_3gram`: 917
