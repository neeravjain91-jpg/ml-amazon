# Production Inference Pipeline Audit

## 1. Candidate Generation Channels
`src/inference_partitioned.py` deploys three inverted-index candidate lookup channels:
1. **Normalized Legal-Stripped Name**: `strip_name_query[s_name]`
   - Normalized (NFKD lowercase punctuation-stripped) text stripped of corporate legal suffixes (`Inc`, `LLC`, `Pvt Ltd`, etc.).
2. **Token-Sorted Name**: `ts_name_query[ts_name]`
   - Alphabetically sorted tokens of the legal-stripped name to handle word order variations.
3. **Name Token + Address Number**: `tok_num_query[(tok, num)]`
   - Co-occurrence of significant name tokens (`len(tok) >= 3`) with normalized numeric address tokens (`extract_numbers_clean`).

## 2. Ingestion Ordering & Streaming Logic
- Ingestion processes each country partition sequentially: `France` -> `US` -> `India`.
- For each country, target sources are streamed sequentially:
  1. `test_source2.tsv` (scanned from line 1 to line EOF)
  2. `test_source3.tsv` (scanned from line 1 to line EOF)

## 3. Candidate Deduplication & Capping Mechanics
- Target records matching any channel are added to `cand_map[s1_id]`:
  ```python
  for s1_id in matched_s1:
      if len(cand_map[s1_id]) < MAX_CANDS_PER_S1: # MAX_CANDS_PER_S1 = 40
          cand_map[s1_id].add(tid)
          stored = True
  ```
- **Cap Application**: Applied **inline during line-by-line file streaming**.
- **Truncation Nature**: Strictly **first-come-first-served** based on physical file row order in `test_source2.tsv` and `test_source3.tsv`.
- **Cross-Source Starvation**: If an S1 entity accumulates 40 candidate matches during the streaming of `test_source2.tsv`, **all subsequent records from Source 2 and 100% of records from Source 3 are discarded**.
- **No Later Recovery Mechanism**: Any candidate omitted by the cap is never scored by LightGBM, never enters `candidate_pairs.tsv`, and cannot be predicted in `matching_results.tsv`.

## 4. Downstream Scoring & Decision Impact
- Feature extraction (`features.compute_pairwise_features`) and LightGBM scoring ONLY see candidates present in `cand_map[s1_id]`.
- The decision engine ($\tau = 0.94, \Delta = 0.05, K \le 5$) selects strictly among those $\le 40$ candidates.
- `candidate_pairs.tsv` and `matching_results.tsv` are written directly from `cand_map[s1_id]`, enforcing $M \subseteq C \subseteq \text{Cap40}$.
