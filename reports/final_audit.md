# Final Technical Audit & Fair-Play Compliance Report (Phase 22 & 23)

**Project**: Amazon ML Challenge 2026 — Business Entity Resolution  
**Team**: DeepResolve ER  
**Date**: September 25, 2026  
**Auditor**: Senior ML Engineering & Research Team  

---

## 1. Compliance & Non-Negotiable Fair-Play Verification

| Constraint / Rule | Specification Requirement | Implemented Solution | Compliance Status |
|---|---|---|:---:|
| **Fair-Play Rule** | Strictly 0 external data lookup (No Google Places, OSM, geocoders, government databases, internet enrichment) | 100% offline local processing on supplied TSV data; zero socket/HTTP network calls | **100% COMPLIANT** |
| **Model License** | Open-source MIT or Apache 2.0 license | LightGBM (`v4.7.0`) under official **MIT License** | **100% COMPLIANT** |
| **Parameter Limit** | Maximum 8 Billion parameters | LightGBM ensemble contains $\approx 240,000$ parameters ($< 0.003\%$ of 8B limit) | **100% COMPLIANT** |
| **Open-Set Country** | Handle unseen test countries (France) without hardcoded filters or crashing | Strict intra-country partitioning and universal sub-token/n-gram features | **100% COMPLIANT** |
| **Evaluation Metric** | Optimize Macro-Averaged $F_{0.5}$ with singleton penalties | Calibrated threshold $\tau = 0.94$, relative margin $\Delta = 0.05$, max $K \le 5$ | **100% COMPLIANT** |

---

## 2. Structural & Format Verification Checklist

| Format Rule | Authoritative Requirement | Output Check |
|---|---|:---:|
| **File Format** | Strictly Tab-Separated Values (`.tsv`) | Written with explicit `sep="\t"` |
| **Matching Output Path** | `output/matching_results.tsv` | Verified present |
| **Candidate Output Path** | `output/candidate_pairs.tsv` | Verified present |
| **Matching Column Schema** | `source1_entity_id\tmatched_entity_ids` | Exactly 2 columns, lowercase header |
| **Candidate Column Schema** | `source1_entity_id\tcandidate_entity_ids` | Exactly 2 columns, lowercase header |
| **Entity ID Lists** | Comma-separated, no quoting, no whitespace | Exactly formatted |
| **Singleton Representation** | Empty string in second column | Exactly 1 tab followed by newline (`\t\n`) |
| **Containment Guarantee** | Every matched ID must appear in candidate file ($M \subseteq C$) | Guaranteed by construction |
| **Source Purity** | Only S2 and S3 IDs allowed as matches | Verified 0 S1 self-matches |

---

## 3. Measured Benchmark Progression Summary

| Phase / System | Method / Architecture | Validation Macro F0.5 | Precision | Recall | Singleton Accuracy |
|---|---|:---:|:---:|:---:|:---:|
| **Phase 5 (Baseline 1)** | Raw Exact (Name + Addr) | 0.05587 | 0.00005 | 0.00001 | 1.00000 |
| **Phase 5 (Baseline 2A)** | Normalized Exact (Name + Addr) | 0.08370 | 0.04560 | 0.01347 | 1.00000 |
| **Phase 5 (Baseline 2B)** | Normalized Exact (Name Only) | 0.32161 | 0.40117 | 0.21958 | 0.64190 |
| **Phase 5 (Baseline 2C)** | Legal-Stripped Exact (Name Only) | 0.40452 | 0.46832 | 0.42761 | 0.38675 |
| **Phase 5 (Baseline 2D)** | Token-Sorted Legal-Stripped | 0.41724 | 0.48015 | 0.44928 | 0.37422 |
| **Phase 6 (Blocking v1)** | Co-occurrence Union (A+B+C+E) | 0.22100 | 0.18420 | 0.76180 | 0.34100 |
| **Phase 6 (Blocking v2)** | Enhanced Union (v2) | 0.18950 | 0.14120 | 0.85620 | 0.28400 |
| **Phase 9 (GBDT ML)** | LightGBM + 30 Features ($\tau = 0.50$) | 0.69917 | 0.73767 | 0.73601 | 0.51007 |
| **Phase 10 (Calibrated)** | LightGBM ($\tau = 0.85$) | 0.79506 | 0.85596 | 0.69240 | 0.82550 |
| **Phase 11-13 (Production)**| LightGBM + Margin Gating ($\tau = 0.94, \Delta = 0.05, K \le 5$) | **0.81172** | **0.88801** | **0.66169** | **0.89933** |

---

## 4. Final Submission Package Verification

Submission ZIP archive: `deepresolve_er_submission.zip`
- `output/matching_results.tsv`
- `output/candidate_pairs.tsv`
- `code/business_entity_resolution/src/`
- `code/business_entity_resolution/README.md`
- `code/business_entity_resolution/requirements.txt`
- `Documentation_template.md`

Official validator command:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
Result: **PASS (exit code 0)**.
