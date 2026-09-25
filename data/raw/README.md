# Immutable Raw Data Layer

This directory represents the immutable raw data reference for the Amazon ML Challenge 2026 Business Entity Resolution project.

All source TSV files are hosted in `dataset/train/` and `dataset/test/` and are strictly read-only and immutable.

### Source Files Summary

| File Key | Source Filepath | Records | Columns | File Size |
|---|---|:---:|---|:---:|
| `train_source1` | `dataset/train/train_source1.tsv` | 2,206,821 | `entity_id, business_name, business_address, country` | 200.3 MB |
| `train_source2` | `dataset/train/train_source2.tsv` | 5,034,616 | `entity_id, business_name, business_address, country` | 466.6 MB |
| `train_source3` | `dataset/train/train_source3.tsv` | 5,285,603 | `entity_id, business_name, business_address, country` | 480.4 MB |
| `train_ground_truth` | `dataset/train/train_ground_truth.tsv` | 2,206,821 | `source1_entity_id, matched_entity_ids` | 121.1 MB |
| `test_source1` | `dataset/test/test_source1.tsv` | 1,732,544 | `entity_id, business_name, business_address, country` | 166.9 MB |
| `test_source2` | `dataset/test/test_source2.tsv` | 4,887,273 | `entity_id, business_name, business_address, country` | 485.9 MB |
| `test_source3` | `dataset/test/test_source3.tsv` | 5,082,316 | `entity_id, business_name, business_address, country` | 482.6 MB |

**Rule**: No file in this layer may ever be modified, overwritten, or truncated.
